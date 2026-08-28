"""Online monitoring during GRPO (grpo_spec_2 §9).

All signals are computable from the reward stream itself — no drift-side inputs
(C1). They are the RL analog of the gold set: they catch reward moving while
human-judged fidelity stays flat, which is the reward-hacking signature the §8
gates cannot catch, because the gates certified the checker *before* training and
the exploit appears *during* it.

**v2 raises the stakes on this module.** With the realism floor removed (§4.3),
R3 is carried by KL and this audit *alone* — it is the primary degeneracy
detector, not a secondary one. And [BAND §10]'s band-edge farming failure has
**no automatic guard standing behind it**: an arc that reaches the plateau via
minimal marking (`d` just above `d_floor`, `q` at `L_design`) is invisible to the
reward by construction. `band_edge_farming_rate` is the only instrument that sees
it.

Signals:

  * **rate telemetry** (NEW) — the rollout distribution of `q_eng`, `q_del`,
    `d_eng`, `d_del` per step. The band gives no within-plateau gradient by
    design, so `q` settling *somewhere* inside `[L_design, U]` is correct and not
    a convergence failure. Flag only the two real pathologies: `q` parked
    **below** `L_design` (ordering breaking) or **riding `U`** (caricature
    pressure through the ceiling).
  * **group collapse, read as TWO numbers** (AMENDED) — no longer single-signed.
    The plateau makes exact ties *more* likely precisely when the policy is doing
    well. Paired with mean reward it separates "converged into the plateau" from
    "stuck with no gradient".
  * **band-edge farming** (NEW) — plateau reached on minimal marking.
  * **cancellation drift** (NEW, from §8.2) — rollout grievance density vs the
    AnnoMI figure. A widening gap means [BAND §7]'s instrument cancellation is
    destabilising and the delivery band is losing its anchor.
  * **missing-key rate** (NEW, §4.2) — halts above ~5%.
  * **mean |advantage| per cell** (NEW, D2.5) — with `scale_rewards = False` the
    fix for a collapsing signal is to raise `lr`, NOT to re-enable scaling; this
    is the number that tells you which is happening.
  * **high-advantage audit** — top-advantage arcs for a human spot-check, logged
    per-axis and per-sub-question. **Clears the backend cache first** (§0.2
    hygiene flag): auditing cached labels re-reads a stored verdict instead of
    re-grading, which is exactly what the audit is supposed to check.
  * **grievance→hot watch** — Q2-yes / Q1-no rate among high-advantage turns.
  * **reward vs fidelity divergence** — a widening gap = hacking in progress.

`make_monitor_callback` wires the step-scoped signals to real training steps.
Without it the monitor accumulates records nobody reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, List, Optional, Sequence


EPS = 1e-6

#: §9 — the two-number collapse rule. A group that ties at a HIGH mean reward has
#: converged into the plateau (healthy, expected under a band). One that ties at a
#: middling mean is stuck with no gradient. The boundary is a reading aid for the
#: log, not a threshold anything acts on.
HEALTHY_PLATEAU_MEAN = 0.75

#: §0.2 baselines under the per-turn binary, for A8. b1 rising is expected and is
#: not a failure; the criterion is about the STRUGGLING cells.
COLLAPSE_BASELINES = {"b1": 0.70, "b2": 0.93, "b3": 0.51, "b4": 0.48,
                      "b5": 0.11, "b6": 0.41}


class MissingKeyHalt(RuntimeError):
    """§4.2 — the graders are returning responses with none of the expected keys.

    Raised rather than warned because the failure is silent and directional: a
    keyless response defaults every field False, the turn reads neutral/flat —
    the UNMARKED class — and that is a downward-only bias on `d` indistinguishable
    from the simulator choosing not to express the axis. Training through it
    corrupts the density term with no visible symptom.
    """


def standardize(rewards: List[float], scale: bool = False) -> tuple[List[float], float]:
    """Group advantages and the group std (§7 step 3).

    **`scale = False` by default (D2.5): mean-centring only**, `A_i = r_i −
    mean(r)`. TRL's default divides by the group std, which partially erases the
    band's reward geometry — the band exists so that `q = 1.0` scores meaningfully
    less than `q = 0.80`, and a per-group rescale makes an arc 0.05 off in a tight
    group and one 0.4 off in a wide group receive the same advantage. It also
    amplifies quantisation noise: at `n_marked = 6` one turn flipping label moves
    `q` a full step, and std-normalisation can turn that coin flip into a
    multi-sigma update.
    """
    if not rewards:
        return [], 0.0
    mu = mean(rewards)
    sigma = pstdev(rewards) if len(rewards) > 1 else 0.0
    if scale:
        return [(r - mu) / (sigma + EPS) for r in rewards], sigma
    return [r - mu for r in rewards], sigma


@dataclass
class ArcRecord:
    """One scored arc within a group — the per-arc analog of the old per-turn row.

    `engine` / `delivery` are `grpo.reward.band_reward.AxisReadout` objects (or
    anything exposing `q`, `d`, `on_plateau`, `at_lower_edge`). Kept as objects
    rather than flattened so the monitor never has to recompute — or worse,
    re-grade — to log a rate.
    """

    reward: float
    engine: Any = None
    delivery: Any = None
    completion: str = ""
    #: Per-turn binaries, DERIVED (§4.2) — for the audit and the RFT view, never
    #: the reward.
    engine_pass: Optional[int] = None
    delivery_pass: Optional[int] = None
    #: Per-turn sub-answers aggregated over the arc: {"e1","e2","q1","q3","dominant"}
    #: as arc-level rates, so the grievance watch has something to read.
    sub: dict = field(default_factory=dict)

    def axis(self, name: str):
        return self.engine if name == "engine" else self.delivery


@dataclass
class GroupRecord:
    """One GRPO group: G arcs sampled at one state (§5.2, G = 8)."""

    step: int
    cell: str
    arcs: List[ArcRecord] = field(default_factory=list)

    @property
    def rewards(self) -> List[float]:
        return [a.reward for a in self.arcs]

    @property
    def completions(self) -> List[str]:
        return [a.completion for a in self.arcs]

    def advantages(self, scale: bool = False) -> tuple[List[float], float]:
        return standardize(self.rewards, scale)

    def collapsed(self) -> bool:
        _, sigma = self.advantages()
        return sigma <= EPS

    def mean_abs_advantage(self) -> float:
        adv, _ = self.advantages()
        return mean([abs(a) for a in adv]) if adv else 0.0


def _rates(values: Sequence[Optional[float]]) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    return {
        "n": len(vals),
        "mean": round(mean(vals), 4),
        "min": round(vals_sorted[0], 4),
        "p50": round(vals_sorted[len(vals_sorted) // 2], 4),
        "max": round(vals_sorted[-1], 4),
    }


@dataclass
class OnlineMonitor:
    """Accumulates per-group records and derives the §9 signals.

    `cal` (optional) is the frozen band calibration; supplying it lets the rate
    telemetry say whether `q` is parked below `L_design` or riding `U`, which is
    the whole point of the flag. Without it the distributions are still logged,
    just unjudged.
    """

    audit_every_n_steps: int = 50
    audit_sample_size: int = 20
    log_path: Optional[str] = None
    subanswer_rates_enabled: bool = True
    rate_telemetry_enabled: bool = True
    band_edge_farming_watch: bool = True
    cancellation_drift_enabled: bool = True

    #: The frozen [BAND §6.4] artifact, for judging where `q` sits.
    cal: Any = None
    #: The reward backends, so the audit can clear their caches (§0.2) and the
    #: missing-key rate can be read (§4.2).
    backends: Any = None
    missing_key_halt_rate: float = 0.05
    #: §8.2 — the AnnoMI grievance density the bracket was measured against.
    #: `cancellation_drift` compares the rollout rate to this.
    annomi_grievance_density: Optional[float] = None
    #: How far the rollout rate may drift before the anchor is considered unstable.
    cancellation_drift_tolerance: float = 0.15

    current_step: int = 0
    _groups: List[GroupRecord] = field(default_factory=list)

    # ── ingest ──────────────────────────────────────────────────────────────
    def record_group(self, rec: GroupRecord) -> None:
        self._groups.append(rec)
        if self.log_path:
            self._append_log({
                "kind": "group", "step": rec.step, "cell": rec.cell,
                "rewards": [round(r, 4) for r in rec.rewards],
                "collapsed": rec.collapsed(),
                "mean_abs_advantage": round(rec.mean_abs_advantage(), 5),
            })

    def _select(self, cell: Optional[str] = None, last_n: Optional[int] = None):
        groups = [g for g in self._groups if cell is None or g.cell == cell]
        return groups[-last_n:] if last_n else groups

    def _arcs(self, cell: Optional[str] = None, last_n: Optional[int] = None):
        return [a for g in self._select(cell, last_n) for a in g.arcs]

    # ── §9 signals ──────────────────────────────────────────────────────────
    def group_collapse_rate(self, cell: Optional[str] = None) -> float:
        groups = self._select(cell)
        if not groups:
            return 0.0
        return sum(1 for g in groups if g.collapsed()) / len(groups)

    def mean_reward(self, cell: Optional[str] = None, last_n: Optional[int] = None) -> float:
        rs = [a.reward for a in self._arcs(cell, last_n)]
        return mean(rs) if rs else 0.0

    def mean_abs_advantage(self, cell: Optional[str] = None) -> float:
        """D2.5 — if this collapses while reward flatlines, raise `lr`. Do NOT
        re-enable `scale_rewards`: `lr` scales everything uniformly, whereas
        std-division scales selectively by group difficulty, which is the
        documented difficulty bias (Dr. GRPO, Liu et al. 2025)."""
        groups = self._select(cell)
        return mean([g.mean_abs_advantage() for g in groups]) if groups else 0.0

    def collapse_reading(self, cell: Optional[str] = None) -> dict:
        """§9's TWO-number rule. Collapse alone is not single-signed under a band.

        | signature              | reading                                  |
        |------------------------|------------------------------------------|
        | high collapse + high mean | converged into the plateau — healthy  |
        | high collapse + mid/low mean | **stuck, no gradient** — b2's sig |
        """
        rate = self.group_collapse_rate(cell)
        mu = self.mean_reward(cell)
        if rate < 0.5:
            reading = "healthy — groups still carry gradient"
        elif mu >= HEALTHY_PLATEAU_MEAN:
            reading = "converged into the plateau — healthy, expected under a band"
        else:
            reading = "STUCK — high collapse at a middling mean, no gradient"
        out = {"collapse_rate": round(rate, 4), "mean_reward": round(mu, 4),
               "reading": reading}
        if cell in COLLAPSE_BASELINES:
            base = COLLAPSE_BASELINES[cell]
            out["baseline_v1"] = base
            out["delta_vs_baseline"] = round(rate - base, 4)
        return out

    def rate_telemetry(self, cell: Optional[str] = None, last_n: Optional[int] = None) -> dict:
        """The rollout distribution of `q_eng`, `q_del`, `d_eng`, `d_del` (§9).

        Settling anywhere inside `[L_design, U]` is CORRECT — the band gives no
        within-plateau gradient by design, so a `q` that stops moving mid-band is
        the objective being satisfied, not a stall. Only two things are flagged.
        """
        arcs = self._arcs(cell, last_n)
        if not arcs:
            return {"n": 0}
        out: dict = {"n": len(arcs)}
        for axis in ("engine", "delivery"):
            reads = [a.axis(axis) for a in arcs]
            reads = [r for r in reads if r is not None]
            out[f"q_{axis}"] = _rates([getattr(r, "q", None) for r in reads])
            out[f"d_{axis}"] = _rates([getattr(r, "d", None) for r in reads])
            if reads:
                out[f"on_plateau_{axis}"] = round(
                    sum(1 for r in reads if getattr(r, "on_plateau", False)) / len(reads), 4)

            flags = self._band_position_flags(axis, reads, cell)
            if flags:
                out.setdefault("flags", []).extend(flags)
        return out

    def _band_position_flags(self, axis: str, reads, cell: Optional[str]) -> List[str]:
        """`q` parked below `L_design` (ordering breaking) or riding `U`
        (caricature pressure through the ceiling)."""
        if self.cal is None or cell is None or not reads:
            return []
        try:
            p = self.cal[cell][axis]
        except (KeyError, TypeError):
            return []
        if getattr(p, "mode", None) != "q_band":
            return []
        qs = [getattr(r, "q", None) for r in reads]
        qs = [q for q in qs if q is not None]
        if not qs:
            return []
        below = sum(1 for q in qs if q < p.L_design) / len(qs)
        riding = sum(1 for q in qs if q >= p.U - 1e-9) / len(qs)
        flags = []
        if below > 0.5:
            flags.append(
                f"{axis}: {below:.0%} of arcs have q BELOW L_design={p.L_design} "
                "— the profile ordering is breaking")
        if riding > 0.5:
            flags.append(
                f"{axis}: {riding:.0%} of arcs ride U={p.U} — caricature pressure "
                "through the ceiling")
        return flags

    def band_edge_farming_rate(self, cell: Optional[str] = None,
                               last_n: Optional[int] = None) -> dict:
        """Arcs reaching the plateau via MINIMAL marking (§9, [BAND §10]).

        **This is the failure with no automatic guard** (§4.3): with the realism
        floor removed, an arc sitting at `d` just above `d_floor` with `q` at
        `L_design` scores full marks and nothing in the reward objects. If this
        rate climbs, raise `d_floor` / `L_design` or promote delivery density to a
        two-sided band (D-BAND.2).
        """
        arcs = self._arcs(cell, last_n)
        out: dict = {"n": len(arcs)}
        for axis in ("engine", "delivery"):
            reads = [a.axis(axis) for a in arcs]
            reads = [r for r in reads if r is not None]
            if not reads:
                out[axis] = None
                continue
            farmed = sum(1 for r in reads if getattr(r, "at_lower_edge", False))
            out[axis] = round(farmed / len(reads), 4)
        return out

    def cancellation_drift(self, cell: Optional[str] = None,
                           last_n: Optional[int] = None) -> dict:
        """Rollout grievance density vs the AnnoMI figure (§8.2 → §9).

        [BAND §7]'s cancellation argument holds only to first order, and only
        insofar as the grader's bias `Δ` is the same on human and simulator text.
        If the Simulator emits markedly more employer-grievance than AnnoMI
        clients do, the grievance→hot confound fires harder on rollouts than on
        the target, `Δ_rollout > Δ_target`, and the residual pushes the policy to
        a wrong setpoint. A widening gap means the delivery band is losing its
        anchor — harden or swap the backend before it means anything.
        """
        arcs = self._arcs(cell, last_n)
        vals = [a.sub.get("e2") for a in arcs if a.sub and a.sub.get("e2") is not None]
        if not vals:
            return {"n": 0}
        rollout = mean(vals)
        out = {"n": len(vals), "rollout_grievance_density": round(rollout, 4)}
        if self.annomi_grievance_density is not None:
            gap = rollout - self.annomi_grievance_density
            out["annomi_grievance_density"] = self.annomi_grievance_density
            out["gap"] = round(gap, 4)
            out["anchor_unstable"] = abs(gap) > self.cancellation_drift_tolerance
        return out

    def missing_key_rate(self) -> dict:
        """§4.2 — the counter, read off the frozen backends."""
        if self.backends is None:
            return {"available": False}
        from grpo.reward.backends import missing_key_rates
        return missing_key_rates(self.backends)

    def check_missing_key_halt(self) -> None:
        """Halt above ~5% (§9). Called from the step callback."""
        rates = self.missing_key_rate()
        if not rates.get("available", True):
            return
        worst = rates.get("max_rate", 0.0)
        if worst > self.missing_key_halt_rate:
            raise MissingKeyHalt(
                f"§4.2 missing-key rate {worst:.1%} exceeds the "
                f"{self.missing_key_halt_rate:.0%} halt threshold: {rates}. A keyless "
                "grader response reads as the UNMARKED class, which biases `d` "
                "downward in a way indistinguishable from the policy declining to "
                "express the axis. Fix the grader before training further."
            )

    def reward_fidelity_gap(self, held_out_fidelity: float,
                            last_n: Optional[int] = None) -> float:
        """mean reward − held-out human-validated fidelity. A widening positive
        gap is hacking in progress."""
        return self.mean_reward(last_n=last_n) - held_out_fidelity

    # ── audit ───────────────────────────────────────────────────────────────
    def _ranked_arcs(self, last_n: Optional[int] = None):
        scored = []
        for g in self._select(last_n=last_n):
            adv, _ = g.advantages()
            for i, a in enumerate(adv):
                scored.append((a, g, i))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def subanswer_rates(self, top_k: Optional[int] = None,
                        last_n: Optional[int] = None) -> dict:
        """Rate of each per-axis sub-answer among the highest-advantage arcs.

        Shows which COMPONENT of a label the advantage flows to: a rising `e2`
        share on a cell whose engine target is neutral means the policy is earning
        reward while drifting outward-blaming, which the fused label would hide.
        """
        top_k = top_k or self.audit_sample_size
        ranked = self._ranked_arcs(last_n)[:top_k]
        if not ranked:
            return {"n": 0}
        keys = ("e1", "e2", "q1", "q3")
        acc = {k: [] for k in keys}
        seen = 0
        for _, g, i in ranked:
            sub = g.arcs[i].sub or {}
            if not sub:
                continue
            seen += 1
            for k in keys:
                if sub.get(k) is not None:
                    acc[k].append(float(sub[k]))
        if not seen:
            return {"n": len(ranked), "sub_answers_available": False}
        out: dict = {"n": seen}
        out.update({k: (round(mean(v), 4) if v else None) for k, v in acc.items()})
        # §9 grievance→hot watch: Q2-yes with Q1-no. Under the §8.1 decomposition
        # `hot = Q1` alone, so a high rate here means the policy is producing
        # employer-grievance that the grader is correctly NOT calling hot — the
        # confound staying closed. A collapse toward zero alongside rising hot is
        # the confound reopening.
        if acc["e2"] and acc["q1"]:
            out["grievance_not_hot"] = round(
                mean([e2 * (1.0 - q1) for e2, q1 in zip(acc["e2"], acc["q1"])]), 4)
        return out

    def high_advantage_audit(self, step: int, force: bool = False) -> List[dict]:
        """Top-advantage arcs for a human spot-check (fires every N steps).

        **Primary degeneracy detector in v2, not a secondary one** — the realism
        floor is gone (§4.3), so this and KL are all that stand behind R3.
        """
        if not force and (self.audit_every_n_steps <= 0
                          or step % self.audit_every_n_steps != 0):
            return []

        # §0.2 hygiene flag: auditing cached labels re-reads a stored verdict
        # instead of re-grading. Clear first or the audit checks nothing.
        if self.backends is not None:
            from grpo.reward.backends import clear_backend_caches
            clear_backend_caches(self.backends)

        out = []
        for a, g, i in self._ranked_arcs()[: self.audit_sample_size]:
            arc = g.arcs[i]
            item = {
                "step": step, "advantage": round(a, 4), "cell": g.cell,
                "reward": round(arc.reward, 4),
                "completion": arc.completion[:300],
                "engine_pass": arc.engine_pass,
                "delivery_pass": arc.delivery_pass,
                "sub": arc.sub,
            }
            for axis in ("engine", "delivery"):
                r = arc.axis(axis)
                if r is not None:
                    item[axis] = {
                        "reward": round(getattr(r, "reward", 0.0), 4),
                        "q": (round(r.q, 4) if getattr(r, "q", None) is not None else None),
                        "d": round(getattr(r, "d", 0.0), 4),
                        "on_plateau": getattr(r, "on_plateau", None),
                        "at_lower_edge": getattr(r, "at_lower_edge", None),
                    }
            out.append(item)
            if self.log_path:
                self._append_log({"kind": "audit", **item})
        return out

    def snapshot(self, step: int, held_out_fidelity: Optional[float] = None,
                 cells: Optional[Sequence[str]] = None) -> dict:
        cells = cells or sorted({g.cell for g in self._groups})
        snap: dict = {
            "step": step,
            "mean_reward": round(self.mean_reward(), 4),
            "group_collapse_rate": round(self.group_collapse_rate(), 4),
            "mean_abs_advantage": round(self.mean_abs_advantage(), 5),
            "per_cell": {c: self.collapse_reading(c) for c in cells},
        }
        if self.rate_telemetry_enabled:
            snap["rate_telemetry"] = {c: self.rate_telemetry(c) for c in cells}
        if self.band_edge_farming_watch:
            snap["band_edge_farming"] = {c: self.band_edge_farming_rate(c) for c in cells}
        if self.cancellation_drift_enabled:
            snap["cancellation_drift"] = self.cancellation_drift()
        if self.subanswer_rates_enabled:
            snap["subanswer_rates"] = self.subanswer_rates()
        mk = self.missing_key_rate()
        if mk.get("available", True):
            snap["missing_key"] = mk
        if held_out_fidelity is not None:
            snap["reward_fidelity_gap"] = round(
                self.reward_fidelity_gap(held_out_fidelity), 4)
        if self.log_path:
            self._append_log({"kind": "snapshot", **snap})
        return snap

    def _append_log(self, obj: dict) -> None:
        p = Path(self.log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(obj) + "\n")


def make_monitor_callback(monitor: OnlineMonitor,
                          held_out_fidelity: Optional[float] = None,
                          on_step: Optional[Callable[[int], None]] = None):
    """A TRL/transformers `TrainerCallback` firing the step-scoped §9 signals.

    The reward function feeds `record_group` on every scored group, but the audit,
    the snapshot and the missing-key halt are *step*-scoped and the reward
    function has no reliable view of the global step. This callback closes that
    loop — without it the monitor accumulates records nobody ever reads.

    `transformers` is imported lazily so the monitor stays importable (and unit
    testable) on a box without the training stack.
    """
    from transformers import TrainerCallback

    class _GRPOMonitorCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            step = int(getattr(state, "global_step", 0) or 0)
            monitor.current_step = step
            # §4.2 — raises MissingKeyHalt, which must abort the run rather than
            # be swallowed: the corruption it detects is silent and directional.
            monitor.check_missing_key_halt()
            monitor.high_advantage_audit(step)
            if monitor.audit_every_n_steps > 0 and step % monitor.audit_every_n_steps == 0:
                monitor.snapshot(step, held_out_fidelity)
            # Per-step counters so the next step's rate is that step's rate.
            if monitor.backends is not None:
                from grpo.reward.backends import reset_missing_key_counters
                reset_missing_key_counters(monitor.backends)
            if on_step:
                on_step(step)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            monitor.snapshot(int(getattr(state, "global_step", 0) or 0), held_out_fidelity)
            return control

    return _GRPOMonitorCallback()
