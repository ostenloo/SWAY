"""Delivery decomposition — Q1/Q3 (CODING_GUIDE.md, axis 2).

The delivery axis is a three-way categorical: what is the patient's emotional
temperature TOWARD THE LISTENER? It is decomposed into its two constituent
questions, one per non-neutral pole:

  * **Q1** — hostility toward the listener? (hot)
  * **Q3** — closeness-pulling toward the listener? (warm)

    | Q1 | Q3 | delivery | reading                                          |
    |----|----|----------|--------------------------------------------------|
    | no | no | flat     | neither hostile nor ingratiating toward you       |
    | no | yes| warm     | flattering, "you get me", pulling you in          |
    | yes| no | hot      | snapping at you, attacking your suggestion        |
    | yes| yes| hot      | hostility dominates a mixed read                  |

**Warm is a positive behaviour, not the absence of hostility.** The coding guide
defines it as "ingratiating, closeness-pulling, flattering, seeking connection
with you", and builds its central example on a patient who is "warmly, even
ingratiatingly, aggrieved". An earlier version of this module asked only Q1 and
treated every non-hostile turn as satisfying a warm target, which made delivery a
free pass on the three warm-target cells (b1, b3, b5) — b3 sat at mean reward
1.000 with 100% group collapse as a direct result. Q3 is what makes those cells
discriminative.

**Employer-grievance is not asked here.** It is an ENGINE construct — the coding
guide defines externalizing as "blame or grievance directed at others, the
employer, the system, or unfairness" — and it lives in `engine_decompose` as E2.
The guide's critical distinction (grievance about a third party is NOT hot
delivery) is enforced in Q1's own wording rather than by asking a second question
about it here.

**Precedence note.** This prompt is GRPO-local by design. `sway_harness/build.py`
owns the shared fidelity annotator (a single fused `delivery` label) and is used
by the prompt-optimization pipeline; per the spec's precedence rule this file does
not mutate it. The two prompts are kept semantically aligned — build.py's delivery
rubric already reasons in two steps, and this module makes those steps two
separately-reported fields rather than one fused label.

*Caveat:* the decomposition is only as strong as the champion's ability to answer
each question in isolation. Neither Q1 nor Q3 has been validated against human
labels on this prompt — the batch03 kappa for glm4 (0.546) was measured on the
fused hot/warm/flat annotation, not this one. §10 certification is the only place
that would surface a problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import grpo._bootstrap  # noqa: F401  (puts sway_harness on sys.path)

from grpo.reward.turn_fidelity import poles_for_cell


# The two target-scoped questions, as reported by the delivery champion.
DECOMP_SCHEMA = {
    "q1_hostility_toward_listener": "true|false",
    "q3_closeness_toward_listener": "true|false",
}


def delivery_decompose_system_prompt() -> str:
    """System prompt for the DECOMPOSED delivery read (§8.1).

    Deliberately narrow: the champion is asked only about emotional target, not
    about the engine, the profile, or pass/fail. It is blind to the cell's target
    pole (the cell-relative binary is applied downstream in `delivery_pass`), so
    it cannot rubber-stamp toward the profile.
    """
    return (
        "You are a behavioral ANNOTATOR for a benchmark. You read one PATIENT TURN and answer "
        "TWO SEPARATE questions about HOW THE PATIENT RELATES TO YOU, the person they are "
        "talking to. You do NOT judge pass/fail, you are NOT told what the patient is supposed "
        "to be, and you must NOT guess an intended profile — report only what is on the page.\n\n"
        "Q1 — HOSTILITY TOWARD YOU. Is the turn hostile, aggressive, escalating, or prickly "
        "TOWARD YOU? Snaps at you, attacks you, belittles your suggestion or reframe ('that's "
        "useless', 'you don't get it either'), treats your help as betrayal, makes angry demands "
        "of you.\n"
        "Q3 — CLOSENESS-PULLING TOWARD YOU. Is the turn ingratiating, flattering, or "
        "connection-seeking TOWARD YOU? 'You get me', 'you're the only one who understands', "
        "leaning in, pulling you close.\n\n"
        "ANSWER THE TWO INDEPENDENTLY. Most turns are neither: a patient can be perfectly "
        "matter-of-fact toward you while describing terrible things. Q1 false and Q3 false is a "
        "normal, common answer — do not feel obliged to pick one.\n\n"
        "THE CRITICAL DISTINCTION (read this twice): GRIEVANCE ABOUT SOMEONE ELSE IS NOT "
        "HOSTILITY TOWARD YOU. Anger at the employer, the boss, management, or the unfairness of "
        "the layoff says nothing about Q1. A patient can be furious at their company and warm to "
        "you in the same breath — 'they were so unfair, but you understand, you're the only one "
        "who sees it' is Q1 FALSE, Q3 TRUE. Ask specifically: is any of this pointed AT ME? "
        "Loud, angry, profane or long turns about the employer are not by themselves hostile to "
        "you.\n\n"
        "Anchors:\n"
        "- 'They laid off the whole team without warning; the company discarded us.' -> Q1 false, Q3 false.\n"
        "- 'They were so unfair, but you get it, you're the only one who sees it.' -> Q1 false, Q3 true.\n"
        "- 'It's so unfair how they handled it — but thank you for hearing me out.' -> Q1 false, Q3 true.\n"
        "- 'Don't you dare tell me it was structural — you sound just like them. You're useless.' -> Q1 true, Q3 false.\n"
        "- 'That journaling idea is garbage. Why am I even talking to you?' -> Q1 true, Q3 false.\n"
        "- 'I've been sleeping badly and I can't focus on anything.' -> Q1 false, Q3 false.\n\n"
        "You are NOT advising anyone, NOT continuing the conversation. Respond with ONLY a JSON "
        "object containing exactly the two boolean fields."
    )


def delivery_decompose_user_prompt(context: str, patient_turn: str) -> str:
    """The per-turn user message for the decomposed delivery read."""
    return (
        f"### Conversation context\n{context}\n"
        f"### Patient turn to annotate\n{patient_turn}\n\n"
        "Answer both questions about this turn:\n"
        "- q1_hostility_toward_listener: is hostility/aggression aimed AT YOU or at your help? true|false\n"
        "- q3_closeness_toward_listener: is the turn ingratiating / pulling YOU close? true|false\n\n"
        'Output JSON only: {"q1_hostility_toward_listener": bool, '
        '"q3_closeness_toward_listener": bool}'
    )


@dataclass(frozen=True)
class DeliveryDecomposition:
    """One turn's decomposed delivery read.

    `hot` is Q1 alone — Q2 is carried for the §9 grievance→hot watch and never
    enters the label.
    """

    q1_hostility_toward_listener: bool
    q3_closeness_toward_listener: bool

    @property
    def delivery(self) -> str:
        """The three-way label. Hostility dominates a mixed read."""
        if self.q1_hostility_toward_listener:
            return "hot"
        if self.q3_closeness_toward_listener:
            return "warm"
        return "flat"

    @property
    def hot(self) -> bool:
        return self.q1_hostility_toward_listener

    @property
    def warm(self) -> bool:
        return self.delivery == "warm"

    def to_dict(self) -> dict:
        return {
            "q1_hostility_toward_listener": self.q1_hostility_toward_listener,
            "q3_closeness_toward_listener": self.q3_closeness_toward_listener,
            "delivery": self.delivery,
        }


def _as_bool(value, default: bool = False) -> bool:
    """Tolerant bool coercion — small local models emit true/'true'/'yes'/1."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "y", "1")
    return default


def decomposition_from_labels(labels: dict) -> DeliveryDecomposition:
    """Parse the champion's JSON into the decomposition.

    Missing/unparseable fields default to False. For Q1 that is the conservative
    direction for the reward: an unreadable answer does not manufacture a `hot`
    label, so a parse failure cannot fabricate delivery reward on a warm/flat cell.
    """
    return DeliveryDecomposition(
        q1_hostility_toward_listener=_as_bool(labels.get("q1_hostility_toward_listener")),
        q3_closeness_toward_listener=_as_bool(labels.get("q3_closeness_toward_listener")),
    )


def delivery_from_decomposition(decomp: DeliveryDecomposition) -> str:
    """The three-way delivery label. Named so the rule is greppable/testable."""
    return decomp.delivery


def hot_from_decomposition(decomp: DeliveryDecomposition) -> bool:
    return decomp.q1_hostility_toward_listener


def delivery_pass_decomposed(decomp: DeliveryDecomposition, cell: str) -> int:
    """Cell-relative delivery binary from the decomposition.

    Tests the ACTUAL pole, not merely absence of the wrong one:

      * hot target  -> the turn must read hot (Q1).
      * warm target -> the turn must read warm (Q3 and not Q1). A flat turn does
        NOT satisfy a warm target; the coding guide defines warm as a positive
        behaviour, and scoring `not hot` here is what made b1/b3/b5 free passes.
      * flat target -> neither hostile nor closeness-pulling.
    """
    target = poles_for_cell(cell)["delivery"]
    return int(delivery_from_decomposition(decomp) == target)


def observed_hot_label(decomp: DeliveryDecomposition) -> str:
    """Collapse to the 'hot' / 'not_hot' vocabulary the §8.3 smoke test uses."""
    return "hot" if hot_from_decomposition(decomp) else "not_hot"
