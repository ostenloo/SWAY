"""Online monitoring during GRPO (grpo_spec §9).

Four signals, all computable from the reward stream itself — no drift-side inputs
(C1). These are the RL analog of the gold set: they catch reward moving while
human-judged fidelity stays flat (advantage flowing to spuriously-rated turns),
which is the reward-hacking signature the §8 gates cannot catch on their own,
because the gates certify the checker *before* training and the exploit appears
*during* it.

  * high_advantage_audit  — every N steps, surface the top-advantage completions
    for a human spot-check, logged PER-DIMENSION (engine, and delivery split into
    Q1/Q2) so you see which axis AND which sub-question is being farmed. That
    split is the second reason the §8.1 decomposition is worth keeping over a
    bare scalar.
  * reward_fidelity_gap   — mean reward vs a held-out human-validated estimate; a
    widening gap = hacking in progress.
  * grievance_hot_watch   — the Q2-yes / Q1-no rate among high-advantage turns,
    plus the sharper hot-cell variant (see `grievance_hot_watch`). The online
    echo of the §8.2 gate.
  * group_collapse_rate   — fraction of std==0 groups; if high the target is too
    off-manifold (strengthen warm-start / curriculum / partial credit).

`GRPOMonitorCallback` wires the periodic signals to real training steps. Without
it the monitor accumulates records nobody reads, which is the state this file was
in against the earlier draft of the spec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Optional, Sequence


EPS = 1e-6


def standardize(rewards: List[float]) -> tuple[List[float], float]:
    """Group-standardized advantages A_i = (r_i - mean) / (std + eps) and the
    group std (grpo_spec §7 step 3). std==0 groups are the collapse case."""
    if not rewards:
        return [], 0.0
    mu = mean(rewards)
    sigma = pstdev(rewards) if len(rewards) > 1 else 0.0
    adv = [(r - mu) / (sigma + EPS) for r in rewards]
    return adv, sigma


@dataclass
class GroupRecord:
    """One GRPO group: G completions at one state, with their per-dimension reads.

    `q1` / `q2` carry the §8.1 decomposition so the audit can attribute delivery
    reward to the right sub-question.
    """

    step: int
    cell: str
    rewards: List[float]
    completions: List[str]
    engine_pass: List[int] = field(default_factory=list)
    delivery_pass: List[int] = field(default_factory=list)
    q1: List[bool] = field(default_factory=list)
    q2: List[bool] = field(default_factory=list)
    delivery_target_hot: bool = False

    def advantages(self) -> tuple[List[float], float]:
        return standardize(self.rewards)

    def collapsed(self) -> bool:
        _, sigma = self.advantages()
        return sigma <= EPS

    def _get(self, seq: Sequence, i: int, default=None):
        return seq[i] if i < len(seq) else default


@dataclass
class OnlineMonitor:
    """Accumulates per-group records and derives the §9 signals. Writes a JSONL
    audit log if `log_path` is set."""

    audit_every_n_steps: int = 50
    audit_sample_size: int = 20
    log_path: Optional[str] = None
    grievance_hot_watch_enabled: bool = True
    #: Kept current by `make_monitor_callback`; the reward function reads it to
    #: stamp each group, having no view of the trainer's global step itself.
    current_step: int = 0
    _groups: List[GroupRecord] = field(default_factory=list)

    def record_group(self, rec: GroupRecord) -> None:
        self._groups.append(rec)
        if self.log_path:
            self._append_log({"kind": "group", "step": rec.step, "cell": rec.cell,
                              "rewards": rec.rewards, "collapsed": rec.collapsed()})

    # --- §9 signals ---------------------------------------------------------
    def group_collapse_rate(self) -> float:
        if not self._groups:
            return 0.0
        return sum(1 for g in self._groups if g.collapsed()) / len(self._groups)

    def mean_reward(self, last_n: Optional[int] = None) -> float:
        groups = self._groups[-last_n:] if last_n else self._groups
        rs = [r for g in groups for r in g.rewards]
        return mean(rs) if rs else 0.0

    def reward_fidelity_gap(self, held_out_fidelity: float, last_n: Optional[int] = None) -> float:
        """mean reward - held-out human-validated fidelity estimate. Track this
        over steps; a widening positive gap is hacking in progress."""
        return self.mean_reward(last_n) - held_out_fidelity

    def _ranked_completions(self, last_n: Optional[int] = None):
        """(advantage, group, index) across recorded groups, best first."""
        groups = self._groups[-last_n:] if last_n else self._groups
        scored = []
        for g in groups:
            adv, _ = g.advantages()
            for i, a in enumerate(adv):
                scored.append((a, g, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def grievance_hot_watch(self, top_k: Optional[int] = None,
                            last_n: Optional[int] = None) -> dict:
        """The §9 grievance->hot watch, over the highest-advantage turns.

        Two numbers, because they catch the exploit on opposite cell types:

        * `grievance_only_rate` — the literal §9 metric: the Q2-yes / Q1-no share
          among high-advantage turns. "Grievance present, no interlocutor
          hostility" rising among the turns earning reward is the confusion
          becoming a gradient.

        * `hot_cell_grievance_rate` — Q2-yes among turns that EARNED delivery
          reward on a **hot-target** cell. This is the sharper signature where the
          exploit actually pays: on a hot cell the policy can only be rewarded if
          the champion calls the turn hot, so a drift from listener-hostility
          toward employer-grievance while delivery reward holds steady is the
          champion folding under optimization pressure. On warm/flat cells,
          Q2-yes/Q1-no is *correct* behaviour and carries no alarm — which is why
          the literal metric alone can read high for benign reasons and must not
          be alerted on in isolation.
        """
        top_k = top_k or self.audit_sample_size
        ranked = self._ranked_completions(last_n)[:top_k]
        if not ranked:
            return {"n": 0, "grievance_only_rate": 0.0, "hot_cell_grievance_rate": 0.0,
                    "n_hot_cell_delivery_rewarded": 0}

        grievance_only = 0
        hot_rewarded = hot_rewarded_grievance = 0
        for _, g, i in ranked:
            q1 = bool(g._get(g.q1, i, False))
            q2 = bool(g._get(g.q2, i, False))
            if q2 and not q1:
                grievance_only += 1
            if g.delivery_target_hot and g._get(g.delivery_pass, i, 0) == 1:
                hot_rewarded += 1
                if q2:
                    hot_rewarded_grievance += 1

        return {
            "n": len(ranked),
            "grievance_only_rate": grievance_only / len(ranked),
            "hot_cell_grievance_rate": (hot_rewarded_grievance / hot_rewarded) if hot_rewarded else 0.0,
            "n_hot_cell_delivery_rewarded": hot_rewarded,
        }

    def high_advantage_audit(self, step: int, force: bool = False) -> List[dict]:
        """Top-advantage completions for a human spot-check (fires every N steps).

        Logged per-dimension — engine, delivery, and delivery split into Q1/Q2 —
        so the reviewer can see which axis and which sub-question the advantage is
        flowing to, not merely that reward went up.
        """
        if not force and (self.audit_every_n_steps <= 0 or step % self.audit_every_n_steps != 0):
            return []
        out = []
        for a, g, i in self._ranked_completions()[: self.audit_sample_size]:
            item = {
                "step": step, "advantage": round(a, 4), "cell": g.cell,
                "reward": g.rewards[i], "completion": g.completions[i][:300],
                "engine_pass": g._get(g.engine_pass, i),
                "delivery_pass": g._get(g.delivery_pass, i),
                "q1_hostility_toward_listener": g._get(g.q1, i),
                "q2_grievance_toward_absent_party": g._get(g.q2, i),
            }
            out.append(item)
            if self.log_path:
                self._append_log({"kind": "audit", **item})
        return out

    def snapshot(self, step: int, held_out_fidelity: Optional[float] = None) -> dict:
        snap = {
            "step": step,
            "mean_reward": round(self.mean_reward(), 4),
            "group_collapse_rate": round(self.group_collapse_rate(), 4),
        }
        if self.grievance_hot_watch_enabled:
            snap["grievance_hot_watch"] = self.grievance_hot_watch()
        if held_out_fidelity is not None:
            snap["reward_fidelity_gap"] = round(self.reward_fidelity_gap(held_out_fidelity), 4)
        if self.log_path:
            self._append_log({"kind": "snapshot", **snap})
        return snap

    def _append_log(self, obj: dict) -> None:
        p = Path(self.log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(obj) + "\n")


def make_monitor_callback(monitor: OnlineMonitor, held_out_fidelity: Optional[float] = None):
    """A TRL/transformers `TrainerCallback` that fires the periodic §9 signals.

    The reward function feeds `record_group` on every scored group, but the audit
    and snapshot are *step*-scoped, and the reward function has no reliable view of
    the global step. This callback closes that loop — without it the monitor
    accumulates records nobody ever reads.

    `transformers` is imported lazily so the monitor stays importable (and unit
    testable) on a box without the training stack.
    """
    from transformers import TrainerCallback

    class _GRPOMonitorCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step = int(getattr(state, "global_step", 0) or 0)
            monitor.current_step = step
            monitor.high_advantage_audit(step)
            if monitor.audit_every_n_steps > 0 and step % monitor.audit_every_n_steps == 0:
                monitor.snapshot(step, held_out_fidelity)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            monitor.snapshot(int(getattr(state, "global_step", 0) or 0), held_out_fidelity)
            return control

    return _GRPOMonitorCallback()
