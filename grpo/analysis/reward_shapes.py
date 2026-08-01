"""Compare candidate reward shapes on real groups, without training anything.

Every candidate reward shape in play is a **pure function of the same two
binaries** `(engine_pass, delivery_pass)`. So the choice between them
does not need a training run to settle: score one batch of real groups once, then
evaluate every shape offline against the same labels. No GPU, no adapter — the
policy is the current prompt-conditioned Simulator over ollama.

**What actually decides this.** GRPO's advantage is
`A_i = (r_i - mean(r_group)) / (std(r_group) + eps)`. A group whose G completions
all score the same has std 0 and contributes no gradient — you pay full rollout
cost for nothing. So the metric that matters is not "mean reward" but the
**group-collapse rate**, and it is a property of the shape *and* the cell's pass
rates together. A shape that looks principled can be unusable on an off-manifold
cell purely because it flattens too many groups.

This measures real groups (G completions sampled at one state, per §5.2), not a
bootstrap over unrelated turns — collapse is a within-state property and
resampling across states would understate it.

Usage:

    python -m grpo.run reward-sweep --cells b1 b3 --states 6 --group-size 8

Read the output as: for each shape, what fraction of groups would have produced a
usable gradient, and what did the reward distribution look like. Then pick, and
record the choice as a deliberate deviation if it is not the §4 default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Dict, List, Optional

import grpo._bootstrap  # noqa: F401

from grpo.data.rollout import Interlocutor, build_states, sample_group
from grpo.reward.fidelity_reward import RewardBackends


EPS = 1e-6


# ── candidate shapes ────────────────────────────────────────────────────────
# Each takes the two diagnostic binaries and returns a scalar in [0, 1]. There is
# no realism term: §4's floor was removed by researcher decision.

def additive_half(e: int, d: int) -> float:
    """grpo_spec §4 default: partial credit 0 / 0.5 / 1.0."""
    return 0.5 * e + 0.5 * d


def product(e: int, d: int) -> float:
    """Strict AND. Kills partial credit -> expect high collapse on hard cells."""
    return float(e * d)


def completion_bonus(e: int, d: int) -> float:
    """Partial credit retained, but both-right beats the sum of the parts.

    Punishes single-axis farming (nail engine, ignore delivery, bank 0.5 forever)
    without flattening groups the way `product` does. Values: {0, 0.4, 1.0}.
    """
    return 0.4 * e + 0.4 * d + 0.2 * e * d


def engine_weighted(e: int, d: int) -> float:
    """Asymmetric: engine is the gated active ingredient in the harness
    (`ENGINE_CONVERGENCE_BAR`; delivery is report-only), so this mirrors that
    weighting inside the reward. Values: {0, 0.3, 0.7, 1.0}."""
    return 0.7 * e + 0.3 * d


def delivery_weighted(e: int, d: int) -> float:
    """The mirror image — useful when delivery is the binding axis on a cell."""
    return 0.3 * e + 0.7 * d


SHAPES: Dict[str, Callable[[int, int], float]] = {
    "additive_half (spec §4)": additive_half,
    "product (strict AND)": product,
    "completion_bonus": completion_bonus,
    "engine_weighted": engine_weighted,
    "delivery_weighted": delivery_weighted,
}


# ── measurement ─────────────────────────────────────────────────────────────

@dataclass
class ScoredGroup:
    """One state's G completions, with their per-dimension binaries."""

    cell: str
    engine: List[int]
    delivery: List[int]
    completions: List[str] = field(default_factory=list)

    def rewards(self, shape: Callable[[int, int], float]) -> List[float]:
        return [shape(e, d) for e, d in zip(self.engine, self.delivery)]


@dataclass
class ShapeReport:
    shape: str
    n_groups: int
    n_completions: int
    mean_reward: float
    collapse_rate: float
    mean_group_std: float
    value_histogram: Dict[str, int]
    by_cell: Dict[str, dict] = field(default_factory=dict)

    @property
    def usable_group_rate(self) -> float:
        return 1.0 - self.collapse_rate

    def to_dict(self) -> dict:
        return {
            "shape": self.shape,
            "n_groups": self.n_groups,
            "n_completions": self.n_completions,
            "mean_reward": round(self.mean_reward, 4),
            "collapse_rate": round(self.collapse_rate, 4),
            "usable_group_rate": round(self.usable_group_rate, 4),
            "mean_group_std": round(self.mean_group_std, 4),
            "value_histogram": self.value_histogram,
            "by_cell": self.by_cell,
        }


def evaluate_shape(groups: List[ScoredGroup], name: str,
                   shape: Callable[[int, int], float]) -> ShapeReport:
    """Collapse rate + reward distribution for one shape over recorded groups."""
    all_rewards: List[float] = []
    stds: List[float] = []
    collapsed = 0
    hist: Dict[str, int] = {}
    per_cell: Dict[str, List[float]] = {}
    per_cell_collapse: Dict[str, List[int]] = {}

    for g in groups:
        rs = g.rewards(shape)
        all_rewards.extend(rs)
        sigma = pstdev(rs) if len(rs) > 1 else 0.0
        stds.append(sigma)
        is_collapsed = sigma <= EPS
        collapsed += int(is_collapsed)
        for r in rs:
            key = f"{r:.2f}"
            hist[key] = hist.get(key, 0) + 1
        per_cell.setdefault(g.cell, []).extend(rs)
        per_cell_collapse.setdefault(g.cell, []).append(int(is_collapsed))

    by_cell = {
        cell: {
            "mean_reward": round(mean(vals), 4),
            "collapse_rate": round(sum(per_cell_collapse[cell]) / len(per_cell_collapse[cell]), 4),
            "n_groups": len(per_cell_collapse[cell]),
        }
        for cell, vals in per_cell.items()
    }

    return ShapeReport(
        shape=name,
        n_groups=len(groups),
        n_completions=len(all_rewards),
        mean_reward=mean(all_rewards) if all_rewards else 0.0,
        collapse_rate=collapsed / len(groups) if groups else 0.0,
        mean_group_std=mean(stds) if stds else 0.0,
        value_histogram=dict(sorted(hist.items())),
        by_cell=by_cell,
    )


def collect_groups(
    P_by_cell: Dict[str, str],
    cells: List[str],
    interlocutors: List[Interlocutor],
    policy_generate,
    backends: RewardBackends,
    n_states: int = 6,
    group_size: int = 8,
    prefix_turns: int = 4,
    seed_base: int = 0,
    progress: bool = True,
) -> List[ScoredGroup]:
    """Sample real GRPO groups and score every completion on both binaries.

    This is the same sampling the trainer does (§5.2: G completions at one state),
    so the collapse rates measured here are the ones training would actually hit.
    Nothing is trained and no adapter is written.

    **Phase-separated on purpose.** All generation happens first, then each grader
    scores every turn in one pass. Ollama holds one model at a time on a single
    card, so interleaving (score turn 1 on three graders, then turn 2, ...) would
    swap models thousands of times and spend the whole run on model loads. This is
    the same "separate generation phase" mitigation §12 recommends for the trainer
    when champion co-residency is tight.
    """
    # --- phase 1: generation (policy + interlocutor resident) ---
    pending: List[tuple] = []          # (cell, ctx, [completions])
    for cell in cells:
        states = build_states(P_by_cell[cell], cell, interlocutors, policy_generate,
                              n_states=n_states, prefix_turns=prefix_turns,
                              seed_base=seed_base)
        for si, state in enumerate(states):
            completions = sample_group(state, policy_generate, group_size,
                                       seed_base=seed_base + 1000 * si)
            pending.append((cell, state.context(), completions))
            if progress:
                print(f"  [gen] {cell} state {si + 1}/{len(states)}: "
                      f"{len(completions)} completions", flush=True)

    # --- phases 2-4: one grader at a time, across every turn ---
    def score_all(label: str, fn) -> List[List[int]]:
        out = []
        for i, (cell, ctx, comps) in enumerate(pending):
            out.append([fn(t, ctx, cell) for t in comps])
            if progress:
                print(f"  [{label}] group {i + 1}/{len(pending)}: "
                      f"{sum(out[-1])}/{len(out[-1])} pass", flush=True)
        return out

    engine = score_all("engine", lambda t, c, cell: backends.engine.score(t, c, cell))
    delivery = score_all("delivery", lambda t, c, cell: backends.delivery.score(t, c, cell))

    return [
        ScoredGroup(cell=cell, engine=e, delivery=d, completions=comps)
        for (cell, _ctx, comps), e, d in zip(pending, engine, delivery)
    ]


def sweep(groups: List[ScoredGroup], out_path: Optional[str] = None) -> dict:
    """Evaluate every candidate shape over the same groups and report."""
    reports = [evaluate_shape(groups, name, fn) for name, fn in SHAPES.items()]
    reports.sort(key=lambda r: r.usable_group_rate, reverse=True)

    base_rates = _base_rates(groups)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_groups": len(groups),
        "group_size": len(groups[0].engine) if groups else 0,
        "base_rates": base_rates,
        "shapes": [r.to_dict() for r in reports],
        "note": (
            "usable_group_rate = 1 - collapse_rate: the share of groups with any "
            "within-group reward variance, i.e. the share that produce a gradient. "
            "This is the number that decides the shape; mean_reward is not "
            "comparable across shapes with different value sets."
        ),
    }
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2))
    return result


def _base_rates(groups: List[ScoredGroup]) -> Dict[str, dict]:
    """Per-cell pass rates — the inputs every shape is a function of."""
    out: Dict[str, dict] = {}
    for g in groups:
        b = out.setdefault(g.cell, {"engine": [], "delivery": []})
        b["engine"].extend(g.engine)
        b["delivery"].extend(g.delivery)
    return {
        cell: {k: round(sum(v) / len(v), 4) for k, v in dims.items() if v}
        for cell, dims in out.items()
    }


def format_report(result: dict) -> str:
    """Human-readable summary for the terminal."""
    lines = [
        f"Reward-shape sweep — {result['n_groups']} groups of "
        f"{result['group_size']} completions",
        "",
        "Per-cell base rates (what every shape is a function of):",
        f"  {'cell':<6} {'engine':>8} {'delivery':>9}",
    ]
    for cell, r in sorted(result["base_rates"].items()):
        lines.append(f"  {cell:<6} {r.get('engine', 0):>8.2f} "
                     f"{r.get('delivery', 0):>9.2f}")
    lines += ["", "Shapes, best gradient yield first:", "",
              f"  {'shape':<26} {'usable':>8} {'collapse':>9} {'meanR':>7} {'grpStd':>7}"]
    for s in result["shapes"]:
        lines.append(f"  {s['shape']:<26} {s['usable_group_rate']:>8.2f} "
                     f"{s['collapse_rate']:>9.2f} {s['mean_reward']:>7.3f} "
                     f"{s['mean_group_std']:>7.3f}")
    lines += ["", result["note"]]
    return "\n".join(lines)
