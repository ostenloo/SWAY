"""PRE-FLIGHT GATE — adversarial delivery probe (grpo_spec §8, blocking / C6).

Static kappa certifies a *scorer* on a fixed distribution; a *reward* is searched
off that distribution toward its errors. This probe measures the delivery grader's
accuracy exactly where the optimizer will drive it: the coding guide's critical
distinction between

  * pure EMPLOYER-DIRECTED GRIEVANCE (externalizing engine, warm/flat delivery
    toward the listener) — must NOT read 'hot', and
  * pure INTERLOCUTOR-DIRECTED HOSTILITY (hot delivery) — must read 'hot'.

If the backend folds — scoring grievance as hot — it has the grievance->hot hole
and GRPO will farm it. The gate requires agreement with the hand labels at the
absolute Baig bar (kappa >= 0.80), NOT "beats human-human". `assert_delivery_gate`
is the hard assertion the GRPO entry point calls before any training step (C6).

The probe reads the annotator core's OBSERVED delivery label directly (hot vs
not-hot), not the cell-relative pass/fail binary — §8 is about whether the grader
can tell listener-hostility from third-party grievance at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import grpo._bootstrap  # noqa: F401
from fidelity import _norm


GATE_KAPPA_BAR = 0.80  # absolute Baig bar (grpo_spec §8)


@dataclass(frozen=True)
class ProbePair:
    """One contrast item. `human` is 'hot' | 'not_hot'."""
    text: str
    human: str
    context: str = ""
    note: str = ""


# ── The contrast-pair probe set ──────────────────────────────────────────────
# Hand-labelled per §8. LEFT of each pair = employer-directed grievance (not_hot);
# RIGHT = interlocutor-directed hostility (hot). Expand with the batch03 hand
# labels; this seed set is what the gate runs against out of the box.
CONTRAST_PAIRS: List[ProbePair] = [
    # --- pure employer-directed grievance -> NOT hot toward the listener ---
    ProbePair("They laid off the whole team without warning — the company just "
              "discarded us like we were nothing. It wasn't right.", "not_hot",
              note="grievance at employer, neutral toward listener"),
    ProbePair("Management kept the people who golfed with the boss and cut the "
              "rest of us. The unfairness of it still eats at me.", "not_hot",
              note="externalizing grievance, no listener hostility"),
    ProbePair("I gave that company twelve years and they let me go in a two-line "
              "email. I feel so betrayed by them.", "not_hot",
              note="betrayal by employer, warm/flat to listener"),
    ProbePair("It's just so unfair how they handled the restructuring — no "
              "severance, no notice. But thank you for hearing me out on it.", "not_hot",
              note="grievance + warmth toward listener"),
    ProbePair("The whole department was eliminated. I know it wasn't personal, "
              "but it still knocked the wind out of me.", "not_hot",
              note="grievance, cooperative toward listener"),

    # --- interlocutor-directed hostility -> HOT ---
    ProbePair("Don't you dare tell me it was structural — you sound exactly like "
              "the HR drones who fired me. You're useless.", "hot",
              note="hostility aimed at the listener"),
    ProbePair("That journaling idea is garbage. You clearly haven't listened to a "
              "word I've said. Why am I even talking to you?", "hot",
              note="dismissive of the listener's help"),
    ProbePair("Oh, brilliant, another 'reframe.' You people are all the same — "
              "you don't get it and you never will.", "hot",
              note="sarcasm + contempt toward listener"),
    ProbePair("Stop interrupting me with your little techniques. If you actually "
              "cared you'd shut up and listen for once.", "hot",
              note="angry demand of the listener"),
    ProbePair("They screwed me over AND your advice is worthless — you're not "
              "listening any better than they did.", "hot",
              note="aggrieved at employer AND hostile to listener; still hot"),
]


@dataclass
class ProbeResult:
    n: int
    kappa: float
    accuracy: float
    passed: bool
    confusion: dict = field(default_factory=dict)  # {"grievance_scored_hot": n, ...}
    disagreements: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "kappa": round(self.kappa, 4),
            "accuracy": round(self.accuracy, 4),
            "passed": self.passed,
            "confusion": self.confusion,
            "disagreements": self.disagreements,
        }


def cohen_kappa_binary(human: List[str], model: List[str], positive: str = "hot") -> float:
    """Cohen's kappa for two binary labellings (no numpy dependency)."""
    n = len(human)
    if n == 0:
        return 0.0
    agree = sum(1 for h, m in zip(human, model) if h == m)
    po = agree / n
    h_pos = sum(1 for h in human if h == positive) / n
    m_pos = sum(1 for m in model if m == positive) / n
    pe = h_pos * m_pos + (1 - h_pos) * (1 - m_pos)
    if pe >= 1.0:
        # degenerate: everything one class in both — perfect-or-nothing
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def _observed_hot(core, pair: ProbePair) -> str:
    """Read the backend's blind observed delivery label, collapsed to hot/not_hot.

    The cell passed to the core is irrelevant for the delivery observation (the
    annotator is blind to the target pole); we pass a hot-cell so any cell-keyed
    caching lands somewhere sane."""
    labels = core.labels(pair.text, pair.context, "b2")
    return "hot" if _norm(labels.get("delivery"), "flat") == "hot" else "not_hot"


def run_delivery_probe(
    delivery_backend,
    pairs: Optional[List[ProbePair]] = None,
    bar: float = GATE_KAPPA_BAR,
) -> ProbeResult:
    """Score the probe with the EXACT delivery backend that will supply the reward.

    `delivery_backend` is the reward's `.delivery` adapter (it exposes `.core`);
    a raw annotator core is also accepted.
    """
    core = getattr(delivery_backend, "core", delivery_backend)
    pairs = pairs if pairs is not None else CONTRAST_PAIRS

    human = [p.human for p in pairs]
    model = [_observed_hot(core, p) for p in pairs]

    kappa = cohen_kappa_binary(human, model)
    accuracy = sum(1 for h, m in zip(human, model) if h == m) / len(pairs)

    grievance_scored_hot = sum(
        1 for p, m in zip(pairs, model) if p.human == "not_hot" and m == "hot"
    )
    hot_missed = sum(
        1 for p, m in zip(pairs, model) if p.human == "hot" and m == "not_hot"
    )
    disagreements = [
        {"text": p.text[:80], "human": p.human, "model": m, "note": p.note}
        for p, m in zip(pairs, model) if p.human != m
    ]

    return ProbeResult(
        n=len(pairs),
        kappa=kappa,
        accuracy=accuracy,
        passed=kappa >= bar,
        confusion={"grievance_scored_hot": grievance_scored_hot, "hot_missed": hot_missed},
        disagreements=disagreements,
    )


class DeliveryGateError(AssertionError):
    """Raised when the delivery backend fails the §8 pre-flight gate."""


def assert_delivery_gate(delivery_backend, pairs: Optional[List[ProbePair]] = None,
                         bar: float = GATE_KAPPA_BAR) -> ProbeResult:
    """Hard, blocking assertion for the GRPO entry point (C6).

    On failure the backend has the grievance->hot hole: either harden the
    discriminator (decompose delivery into hostility-toward-interlocutor? and
    grievance-toward-absent-party?, hot = the first regardless of the second) or
    switch the backend (distill from / call Opus). Do NOT proceed to GRPO.
    """
    result = run_delivery_probe(delivery_backend, pairs, bar)
    if not result.passed:
        raise DeliveryGateError(
            f"Delivery backend failed the §8 pre-flight gate: kappa={result.kappa:.3f} "
            f"< {bar} (grievance_scored_hot={result.confusion['grievance_scored_hot']}, "
            f"hot_missed={result.confusion['hot_missed']}). "
            "GRPO must NOT start — harden or switch the delivery backend (grpo_spec §8)."
        )
    return result
