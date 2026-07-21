"""Online monitoring during GRPO (grpo_spec §9).

Four signals, all computable from the reward stream itself — no drift-side inputs
(C1). These are the RL analog of the gold set: they catch reward moving while
human-judged fidelity stays flat (advantage flowing to spuriously-rated turns),
which is the reward-hacking signature the §8 gate can't catch on its own.

  * high_advantage_audit  — every N steps, surface the top-advantage completions
    for a human spot-check, logged per-dimension so you see WHICH axis is farmed.
  * reward_fidelity_gap   — mean reward vs a held-out human-validated estimate; a
    widening gap = hacking in progress.
  * realism_trip_rate     — rising floor-trips => the policy is probing degenerate
    regions; tighten KL.
  * group_collapse_rate   — fraction of std==0 groups; if high the target is too
    off-manifold (strengthen warm-start / curriculum / partial credit).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Optional


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
    step: int
    cell: str
    rewards: List[float]
    completions: List[str]
    engine_pass: List[int] = field(default_factory=list)
    delivery_pass: List[int] = field(default_factory=list)
    realism_ok: List[int] = field(default_factory=list)

    def advantages(self) -> tuple[List[float], float]:
        return standardize(self.rewards)

    def collapsed(self) -> bool:
        _, sigma = self.advantages()
        return sigma <= EPS


@dataclass
class OnlineMonitor:
    """Accumulates per-group records and derives the §9 signals. Writes a JSONL
    audit log if `log_path` is set."""
    audit_every_n_steps: int = 50
    audit_sample_size: int = 20
    log_path: Optional[str] = None
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

    def realism_trip_rate(self) -> float:
        trips = total = 0
        for g in self._groups:
            trips += sum(1 for ok in g.realism_ok if ok == 0)
            total += len(g.realism_ok)
        return trips / total if total else 0.0

    def mean_reward(self, last_n: Optional[int] = None) -> float:
        groups = self._groups[-last_n:] if last_n else self._groups
        rs = [r for g in groups for r in g.rewards]
        return mean(rs) if rs else 0.0

    def reward_fidelity_gap(self, held_out_fidelity: float, last_n: Optional[int] = None) -> float:
        """mean reward - held-out human-validated fidelity estimate. Track this
        over steps; a widening positive gap is hacking in progress."""
        return self.mean_reward(last_n) - held_out_fidelity

    def high_advantage_audit(self, step: int) -> List[dict]:
        """Top-advantage completions for a human spot-check (fires every N steps).
        Logged per-dimension so you can see which axis is being farmed."""
        if step % self.audit_every_n_steps != 0:
            return []
        scored: List[tuple[float, GroupRecord, int]] = []
        for g in self._groups:
            adv, _ = g.advantages()
            for i, a in enumerate(adv):
                scored.append((a, g, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for a, g, i in scored[: self.audit_sample_size]:
            item = {
                "step": step, "advantage": round(a, 4), "cell": g.cell,
                "reward": g.rewards[i], "completion": g.completions[i][:300],
                "engine_pass": g.engine_pass[i] if i < len(g.engine_pass) else None,
                "delivery_pass": g.delivery_pass[i] if i < len(g.delivery_pass) else None,
                "realism_ok": g.realism_ok[i] if i < len(g.realism_ok) else None,
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
            "realism_trip_rate": round(self.realism_trip_rate(), 4),
        }
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
