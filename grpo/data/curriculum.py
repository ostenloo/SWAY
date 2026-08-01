"""Curriculum for off-manifold cells (grpo_spec §5.5).

For cells where on-profile turns are low-probability under the base — the
voluble x dependency / off-manifold targets — GRPO has nothing to work with at
step 1: nearly every group is all-fail, within-group std is 0, and the advantage
is undefined (R2). Warm-start raises the base rate; the curriculum raises it
further by starting the target **near the manifold** and annealing toward the hard
setting across training.

**What anneals, and what must not.** The curriculum moves the *target the policy is
asked to hit* — the profile prompt in the state. It does NOT touch the graders:
the reward backends stay frozen at temperature 0 on pinned checkpoints throughout
(C4), so "the reward got easier" is never an explanation for a rising curve. An
annealing reward would be indistinguishable from progress; an annealing target is
a syllabus.

**Implemented as sequential stages**, not a per-example blend. TRL consumes one
dataset per trainer and shuffles it, so a blended dataset would present easy and
hard targets in random order — which is not a curriculum. Each stage instead gets
its own dataset and its own slice of the step budget, and the adapter is carried
forward from stage to stage.

The near-manifold wording is a **research decision, not a mechanical one**. Two
ways to supply it, in order of preference:

  1. `curriculum.near_manifold_build_dir` — authored near-manifold profile prompts,
     one `<cell>_prompt.txt` per cell, the same shape as the target prompts. This
     is the honest option: a human writes the easier target.
  2. `curriculum.relaxation_directive` — a text block appended to the target
     prompt during the early stage, softening the hard demand. A fallback for
     getting a pilot moving; the default below is deliberately conservative and
     should be tuned before it is trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


#: Provisional relaxation text (option 2). Tune in the pilot — this is a starting
#: point, not a validated setting. It softens the *intensity* of the demanded pole
#: while keeping its direction, so the stage-1 target sits nearer the base's
#: manifold without becoming a different profile.
DEFAULT_RELAXATION_DIRECTIVE = (
    "\n\n[EARLY-STAGE CALIBRATION]\n"
    "For this stage, express the disposition above at MODERATE intensity rather than "
    "its full extreme. Keep the same direction — who you blame, how you relate to the "
    "listener, how much you disclose — but let it show at a level you can hold "
    "naturally across the whole conversation rather than at maximum from the first "
    "turn. Do not change the direction of any trait; only its intensity."
)


@dataclass(frozen=True)
class Stage:
    """One curriculum stage: a target transform plus its slice of the step budget."""

    name: str
    max_steps: int
    #: None = use the target prompt unchanged.
    near_manifold_dir: Optional[str] = None
    relaxation_directive: Optional[str] = None
    #: Cells this stage softens. Cells outside it always train on the hard target,
    #: so an on-manifold cell is never held back by another cell's syllabus.
    cells: tuple = ()

    def prompt_for(self, cell: str, P: str) -> str:
        """The profile prompt this stage presents for `cell`."""
        if cell not in self.cells:
            return P
        if self.near_manifold_dir:
            fp = Path(self.near_manifold_dir) / f"{cell}_prompt.txt"
            if fp.exists():
                return fp.read_text()
            raise FileNotFoundError(
                f"curriculum.near_manifold_build_dir is set but {fp} is missing. Author "
                "a near-manifold prompt for every curriculum cell, or drop the setting "
                "and fall back to relaxation_directive (grpo_spec §5.5)."
            )
        if self.relaxation_directive:
            return P + self.relaxation_directive
        return P


def build_stages(cfg: dict) -> List[Stage]:
    """Derive the stage list from config (§11 `curriculum` block).

    Returns a single hard-target stage when the curriculum is disabled or no cells
    are enrolled — so the caller has one code path either way.
    """
    total_steps = int(cfg["grpo"].get("max_steps", 500))
    cur = cfg.get("curriculum") or {}
    enabled = [c for c in (cur.get("enabled_cells") or []) if c in cfg["cells"]]

    if not cur.get("enabled", True) or not enabled:
        return [Stage(name="target", max_steps=total_steps)]

    frac = float(cur.get("anneal_frac", 0.4))
    frac = min(max(frac, 0.0), 0.9)
    near_steps = int(total_steps * frac)
    if near_steps <= 0:
        return [Stage(name="target", max_steps=total_steps)]

    return [
        Stage(
            name="near_manifold",
            max_steps=near_steps,
            near_manifold_dir=cur.get("near_manifold_build_dir"),
            relaxation_directive=cur.get("relaxation_directive", DEFAULT_RELAXATION_DIRECTIVE),
            cells=tuple(enabled),
        ),
        Stage(name="target", max_steps=total_steps - near_steps),
    ]


def apply_stage(P_by_cell: Dict[str, str], stage: Stage) -> Dict[str, str]:
    """The per-cell prompts this stage trains on."""
    return {cell: stage.prompt_for(cell, P) for cell, P in P_by_cell.items()}
