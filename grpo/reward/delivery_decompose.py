"""Delivery decomposition — Q1/Q2, `hot = Q1` (grpo_spec §8.1).

Delivery is **single-covered** in the reward: one champion, no opposing read. Its
characteristic error is collapsing **employer-directed grievance** (externalizing
engine content, warm/flat delivery toward the listener) into
**interlocutor-directed hostility** (hot delivery). As a static scorer that is a
rare, in-budget miss. As a *reward* it is a gradient — GRPO relocates probability
mass onto exactly the region where the error lives, so a rare confusion becomes
the modal one. The switch-backend escape (to Opus) is closed by the no-API
constraint (D0.2), so the standing fix is to harden the discriminator.

The fused "is this hot?" question is replaced by two **target-scoped** questions,
scored separately:

  * **Q1** — hostility toward the INTERLOCUTOR (the therapist/model in the room)?
  * **Q2** — grievance toward an ABSENT party (the employer, not in the room)?

and the delivery label is **`hot = Q1`, regardless of Q2**:

  | Q1 | Q2 | delivery | reading                                             |
  |----|----|----------|-----------------------------------------------------|
  | no | no | not hot  | flat/warm, no grievance                              |
  | no | yes| not hot  | employer rant, cooperative toward the listener       |
  | yes| no | hot      | a real attack on the listener                        |
  | yes| yes| **hot**  | furious at the boss AND snapping at the therapist    |

Q2 never changes the delivery label. It is retained because the §9 grievance→hot
watch needs the Q2-yes / Q1-no rate to detect the exploit becoming a gradient —
a bare scalar cannot show which sub-question is being farmed.

**Precedence note.** This prompt is GRPO-local by design. `sway_harness/build.py`
owns the shared fidelity annotator (a single fused `delivery` label) and is used
by the prompt-optimization pipeline; per the spec's precedence rule this file does
not mutate it. The two prompts are kept semantically aligned — build.py's delivery
rubric already reasons in two steps, and this module makes those steps two
separately-reported fields rather than one fused label.

*Caveat (§8.1):* decomposition is only as strong as the champion's ability to
answer Q1 in isolation. If it still cannot separate "you're useless" from "they
were useless" when asked point-blank about the target, decomposition does not save
it and a stronger delivery checker is needed. `gates/delivery_stratified_validation.py`
(§8.2) measures exactly this, and blocks GRPO when it fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import grpo._bootstrap  # noqa: F401  (puts sway_harness on sys.path)

from grpo.reward.turn_fidelity import poles_for_cell


# The two target-scoped questions, as reported by the delivery champion.
DECOMP_SCHEMA = {
    "q1_hostility_toward_listener": "true|false",
    "q2_grievance_toward_absent_party": "true|false",
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
        "TWO SEPARATE questions about WHO the patient's feeling is aimed at. You do NOT judge "
        "pass/fail, you are NOT told what the patient is supposed to be, and you must NOT guess "
        "an intended profile — report only what is on the page.\n\n"
        "There are two parties. THE LISTENER is the person the patient is talking TO right now "
        "(a therapist or assistant, in the room). THE ABSENT PARTY is anyone the patient talks "
        "ABOUT but who is not in the room — the employer, the boss, HR, management, colleagues.\n\n"
        "Q1 — HOSTILITY TOWARD THE LISTENER. Does the turn aim hostility, irritation, sarcasm, "
        "contempt, or dismissiveness AT THE LISTENER or at the listener's help? Signs: snapping at "
        "you, belittling your suggestion or reframe ('that's useless', 'you don't get it either'), "
        "treating your help as betrayal, angry demands of you, sarcasm directed at you.\n"
        "Q2 — GRIEVANCE TOWARD AN ABSENT PARTY. Does the turn express grievance, anger, blame, or "
        "a sense of being wronged toward someone NOT in the room — the company, the boss, "
        "management, the process that laid them off?\n\n"
        "ANSWER THE TWO INDEPENDENTLY. They are not exclusive and they are not linked. A patient "
        "can be furious at the employer while treating you perfectly warmly (Q1 false, Q2 true); "
        "can snap at you about something unrelated to the employer (Q1 true, Q2 false); can do "
        "both at once (both true); or neither.\n\n"
        "THE MOST COMMON MISTAKE IS LETTING Q2 DRAG Q1 TRUE. Loud, angry, profane, or long turns "
        "about the employer are NOT by themselves hostility toward you. Ask specifically: is any of "
        "it pointed AT ME? If the anger passes you by — or the patient is even thanking you while "
        "raging about the boss — Q1 is FALSE no matter how hot the turn reads overall.\n\n"
        "Anchors:\n"
        "- 'They laid off the whole team without warning; the company discarded us.' -> Q1 false, Q2 true.\n"
        "- 'It's so unfair how they handled it — but thank you for hearing me out.' -> Q1 false, Q2 true.\n"
        "- 'Don't you dare tell me it was structural — you sound just like them. You're useless.' -> Q1 true, Q2 true.\n"
        "- 'That journaling idea is garbage. Why am I even talking to you?' -> Q1 true, Q2 false.\n"
        "- 'The whole department was eliminated. I know it wasn't personal.' -> Q1 false, Q2 true.\n"
        "- 'I've been sleeping badly and I can't focus on anything.' -> Q1 false, Q2 false.\n\n"
        "You are NOT advising anyone, NOT continuing the conversation. Respond with ONLY a JSON "
        "object containing exactly the two boolean fields."
    )


def delivery_decompose_user_prompt(context: str, patient_turn: str) -> str:
    """The per-turn user message for the decomposed delivery read."""
    return (
        f"### Conversation context\n{context}\n"
        f"### Patient turn to annotate\n{patient_turn}\n\n"
        "Answer both questions about this turn:\n"
        "- q1_hostility_toward_listener: is hostility/irritation/sarcasm/dismissiveness aimed AT "
        "THE LISTENER or the listener's help? true|false\n"
        "- q2_grievance_toward_absent_party: is grievance/blame aimed at someone NOT in the room "
        "(the employer, the boss, management)? true|false\n\n"
        'Output JSON only: {"q1_hostility_toward_listener": bool, '
        '"q2_grievance_toward_absent_party": bool}'
    )


@dataclass(frozen=True)
class DeliveryDecomposition:
    """One turn's decomposed delivery read.

    `hot` is Q1 alone — Q2 is carried for the §9 grievance→hot watch and never
    enters the label.
    """

    q1_hostility_toward_listener: bool
    q2_grievance_toward_absent_party: bool

    @property
    def hot(self) -> bool:
        """§8.1: `hot = Q1`, regardless of Q2."""
        return self.q1_hostility_toward_listener

    @property
    def grievance_only(self) -> bool:
        """Q2-yes / Q1-no — the cell the §9 watch tracks among high-advantage turns.

        A rising share of these among turns *earning delivery reward* is the
        grievance→hot exploit becoming a gradient.
        """
        return self.q2_grievance_toward_absent_party and not self.q1_hostility_toward_listener

    def to_dict(self) -> dict:
        return {
            "q1_hostility_toward_listener": self.q1_hostility_toward_listener,
            "q2_grievance_toward_absent_party": self.q2_grievance_toward_absent_party,
            "hot": self.hot,
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
        q2_grievance_toward_absent_party=_as_bool(labels.get("q2_grievance_toward_absent_party")),
    )


def hot_from_decomposition(decomp: DeliveryDecomposition) -> bool:
    """`hot = Q1`, regardless of Q2 (§8.1). Kept as a named function so the rule
    is greppable and testable in isolation (acceptance A2)."""
    return decomp.q1_hostility_toward_listener


def delivery_pass_decomposed(decomp: DeliveryDecomposition, cell: str) -> int:
    """Cell-relative delivery binary from the decomposition.

    Mirrors `turn_fidelity.delivery_pass` — which only ever tests hot vs not-hot,
    never warm vs flat — so the decomposition loses no information the reward used:

      * hot target       -> the turn must read hot (Q1 true).
      * warm/flat target -> a hot turn is the wrong pole; anything not-hot passes.
    """
    target = poles_for_cell(cell)["delivery"]
    is_hot = hot_from_decomposition(decomp)
    if target == "hot":
        return int(is_hot)
    return int(not is_hot)


def observed_hot_label(decomp: DeliveryDecomposition) -> str:
    """Collapse to the 'hot' / 'not_hot' vocabulary the §8 gates score against."""
    return "hot" if hot_from_decomposition(decomp) else "not_hot"
