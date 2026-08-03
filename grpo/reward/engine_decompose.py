"""Engine decomposition — E1/E2 + dominance (CODING_GUIDE.md, axis 1).

The engine axis is a three-way categorical: does the patient locate the cause of
what happened in themselves, in others, or neither? Rather than ask the champion
for that label directly, it is decomposed into the two constituent questions plus
an explicit dominance resolution:

  * **E1** — does the turn blame the SELF? (a self-directed verdict)
  * **E2** — does the turn blame OTHERS / the employer / the system?
  * **dominant** — when both appear, which one carries the turn?

    | E1 | E2 | dominant | engine_direction |
    |----|----|----------|------------------|
    | no | no | (neither)| neutral          |
    | yes| no | self     | internalizing    |
    | no | yes| others   | externalizing    |
    | yes| yes| self     | internalizing    |
    | yes| yes| others   | externalizing    |

The dominance field exists because the coding guide requires it: "A turn can
contain both inward and outward content. Label the **dominant** bend for the
turn." Two booleans alone cannot express that, so asking for it is not optional.

**Why decompose at all.** Same reason as the delivery axis: a fused three-way
question makes the model resolve mixed turns silently, and mixed turns are the
common case here — a laid-off patient blaming the employer *and* themselves in
one breath. Splitting it surfaces which components were present, so a
disagreement about the final label can be traced to a component rather than
guessed at.

E2 was previously asked of the *delivery* champion (as "Q2"), where it served as
a distractor-naming control. It is an engine construct — the coding guide defines
externalizing as "blame or grievance directed at others, the employer, the
system, or unfairness" — so it lives here now, on the axis it measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import grpo._bootstrap  # noqa: F401


DECOMP_SCHEMA = {
    "e1_blames_self": "true|false",
    "e2_blames_others": "true|false",
    "dominant": "self|others|neither",
}


def engine_decompose_system_prompt() -> str:
    """System prompt for the DECOMPOSED engine read.

    Blind to the cell's target pole — the cell-relative binary is applied
    downstream in `engine_pass` — so the champion cannot rubber-stamp toward the
    profile. Wording follows CODING_GUIDE.md axis 1, including the two edge-case
    rules (dominance, and grief-without-attribution being neutral).
    """
    return (
        "You are a behavioral ANNOTATOR for a benchmark. You read one PATIENT TURN and answer "
        "questions about WHERE THE PATIENT LOCATES THE CAUSE of what happened to them. You do "
        "NOT judge pass/fail, you are NOT told what the patient is supposed to be, and you must "
        "NOT guess an intended profile — report only what is on the page.\n\n"
        "E1 — BLAMES THE SELF. Does the turn deliver a verdict ABOUT THE PATIENT THEMSELVES? "
        "Self-blame, worthlessness, self-directed failure: 'I'm a failure', 'I always screw "
        "everything up', 'this proves I wasn't good enough', 'it's my fault'.\n"
        "E2 — BLAMES OTHERS. Does the turn direct blame or grievance OUTWARD — at the employer, "
        "the boss, management, the system, or unfairness? 'They discarded me', 'the whole system "
        "is rigged', 'they singled me out', 'they had no right'.\n"
        "DOMINANT — if BOTH are present, which one CARRIES the turn? Answer 'self' or 'others'. "
        "If only one is present, answer that one. If neither is present, answer 'neither'.\n\n"
        "TWO RULES THAT DECIDE MOST HARD CASES:\n"
        "1. Mixed turns are common — a person can blame the company and themselves in one breath. "
        "Do not average them. Pick the bend that DOMINATES. A single passing clause of self-doubt "
        "inside an otherwise-grievance turn is dominated by the grievance, and vice versa.\n"
        "2. PAIN IS NOT BLAME. Grief, sadness, worry, confusion, or 'I don't know what to do next' "
        "with no attribution is E1 false, E2 false, dominant 'neither'. Internalizing requires a "
        "self-directed VERDICT ('I'm worthless'), not merely distress. Do not read suffering as "
        "self-blame.\n\n"
        "Anchors:\n"
        "- 'I'm just not good enough; if I were better this wouldn't have happened.' -> E1 true, E2 false, dominant self.\n"
        "- 'They kept the golfers and cut the rest of us. It wasn't right.' -> E1 false, E2 true, dominant others.\n"
        "- 'They had no right to do that, though maybe I should have seen it coming.' -> E1 true, E2 true, dominant others.\n"
        "- 'I keep thinking I deserved it, that they saw something true about me.' -> E1 true, E2 true, dominant self.\n"
        "- 'This is a lot to take in. I keep replaying it.' -> E1 false, E2 false, dominant neither.\n"
        "- 'Thank you, I'll make a list and take it step by step.' -> E1 false, E2 false, dominant neither.\n\n"
        "You are NOT advising anyone, NOT continuing the conversation. Respond with ONLY a JSON "
        "object containing exactly the three fields."
    )


def engine_decompose_user_prompt(context: str, patient_turn: str) -> str:
    """The per-turn user message for the decomposed engine read."""
    return (
        f"### Conversation context\n{context}\n"
        f"### Patient turn to annotate\n{patient_turn}\n\n"
        "Answer for this turn:\n"
        "- e1_blames_self: does the turn deliver a verdict about the patient themselves? true|false\n"
        "- e2_blames_others: does the turn blame the employer / others / the system? true|false\n"
        "- dominant: which bend carries the turn? self|others|neither\n\n"
        'Output JSON only: {"e1_blames_self": bool, "e2_blames_others": bool, '
        '"dominant": "self|others|neither"}'
    )


@dataclass(frozen=True)
class EngineDecomposition:
    """One turn's decomposed engine read."""

    e1_blames_self: bool
    e2_blames_others: bool
    dominant: str          # "self" | "others" | "neither"

    @property
    def engine_direction(self) -> str:
        """The three-way label the cell-relative binary is applied to."""
        if self.dominant == "self":
            return "internalizing"
        if self.dominant == "others":
            return "externalizing"
        return "neutral"

    @property
    def mixed(self) -> bool:
        """Both bends present — the case the dominance field exists to resolve."""
        return self.e1_blames_self and self.e2_blames_others

    def to_dict(self) -> dict:
        return {
            "e1_blames_self": self.e1_blames_self,
            "e2_blames_others": self.e2_blames_others,
            "dominant": self.dominant,
            "engine_direction": self.engine_direction,
        }


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "y", "1")
    return default


def decomposition_from_labels(labels: dict) -> EngineDecomposition:
    """Parse the champion's JSON into the decomposition.

    `dominant` is trusted when it is one of the three valid values. When it is
    missing or unparseable it is reconstructed from E1/E2, falling back to
    `neither` for the ambiguous both-true case — the conservative direction, since
    `neutral` is the pole that asserts least about the turn.
    """
    e1 = _as_bool(labels.get("e1_blames_self"))
    e2 = _as_bool(labels.get("e2_blames_others"))
    dom = str(labels.get("dominant", "")).strip().lower()
    if dom not in ("self", "others", "neither"):
        if e1 and not e2:
            dom = "self"
        elif e2 and not e1:
            dom = "others"
        else:
            dom = "neither"
    return EngineDecomposition(e1_blames_self=e1, e2_blames_others=e2, dominant=dom)


def engine_direction_from_decomposition(decomp: EngineDecomposition) -> str:
    """Named so the derivation rule is greppable and testable in isolation."""
    return decomp.engine_direction
