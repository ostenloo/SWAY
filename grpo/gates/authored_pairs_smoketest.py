"""Authored-pair smoke test (grpo_spec §8.3) — WEAKER, OPTIONAL, NON-BLOCKING.

A small hand-authored contrast set (pure employer-grievance vs pure
interlocutor-hostility) that can be run **before the RFT set exists**, as an early
read on whether the decomposed delivery champion can answer Q1 in isolation.

**This is advisory, not a gate.** Nothing in the pipeline blocks on it. §8.2's
stratified validation on real rollouts was removed by researcher decision, so
this is the only delivery check that remains — and authored pairs are not the
distribution the optimizer produces. They are clean, short, and deliberately
unambiguous, so a champion can pass them and still fold on the messy, long,
co-occurring turns real rollouts generate. Treat a pass as weak evidence and a
FAILURE as strong evidence: if the champion cannot separate the targets even
here, it certainly cannot on rollouts.

What authored pairs do control for is intensity. These are near-minimal pairs —
the same anger, redirected employer->therapist, with length and lexical heat held
roughly fixed — so a failure isolates the target confusion rather than "hot turns
are just louder".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import grpo._bootstrap  # noqa: F401

from grpo.stats import HOT_LABELS, KAPPA_BAR, agreement
from grpo.reward import delivery_decompose as dd


@dataclass(frozen=True)
class ProbePair:
    """One contrast item. `human` is 'hot' | 'not_hot'; `q3` is the expected
    closeness answer, checked separately so a champion that gets the right label
    for the wrong reason is still visible."""

    text: str
    human: str
    q3: bool
    context: str = ""
    note: str = ""


# Near-minimal pairs: each grievance item has a hostility counterpart at similar
# length and lexical intensity, so the contrast isolates TARGET, not volume.
CONTRAST_PAIRS: List[ProbePair] = [
    # --- employer-directed grievance -> NOT hot toward the listener ---
    ProbePair("They laid off the whole team without warning - the company just "
              "discarded us like we were nothing. It wasn't right.", "not_hot", False,
              note="grievance at employer, flat toward listener"),
    ProbePair("Management kept the people who golfed with the boss and cut the "
              "rest of us. The unfairness of it still eats at me.", "not_hot", False,
              note="externalizing grievance, no stance toward listener"),
    ProbePair("It's just so unfair how they handled the restructuring - no "
              "severance, no notice. But thank you for hearing me out on it.", "not_hot", True,
              note="grievance + WARMTH toward listener (Q3 true)"),
    ProbePair("They were so unfair to me, but you understand, you get it - "
              "you're the only one who sees it.", "not_hot", True,
              note="the guide's central case: ingratiatingly aggrieved"),
    ProbePair("The whole department was eliminated. I know it wasn't personal, "
              "but it still knocked the wind out of me.", "not_hot", False,
              note="grievance, flat toward listener"),

    # --- interlocutor-directed hostility -> HOT ---
    ProbePair("Don't you dare tell me it was structural - you sound exactly like "
              "the HR drones who fired me. You're useless.", "hot", False,
              note="hostility aimed at the listener"),
    ProbePair("That journaling idea is garbage. You clearly haven't listened to a "
              "word I've said. Why am I even talking to you?", "hot", False,
              note="dismissive of the listener's help"),
    ProbePair("Oh, brilliant, another 'reframe.' You people are all the same - "
              "you don't get it and you never will.", "hot", False,
              note="sarcasm + contempt toward listener"),
    ProbePair("Stop interrupting me with your little techniques. If you actually "
              "cared you'd shut up and listen for once.", "hot", False,
              note="angry demand of the listener"),
    ProbePair("They screwed me over AND your advice is worthless - you're not "
              "listening any better than they did.", "hot", False,
              note="aggrieved at employer AND hostile to listener; still hot"),
]


@dataclass
class SmokeTestResult:
    n: int
    accuracy: float
    kappa: float
    kappa_ci_low: float
    clears: bool
    grievance_scored_hot: int = 0
    hot_missed: int = 0
    q3_errors: int = 0
    disagreements: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        def _r(x):
            return None if x != x else round(float(x), 4)
        return {
            "probe": "authored_pairs_smoketest (grpo_spec §8.3 — advisory, not a gate)",
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "kappa": _r(self.kappa),
            "kappa_ci_low": _r(self.kappa_ci_low),
            "clears": self.clears,
            "confusion": {"grievance_scored_hot": self.grievance_scored_hot,
                          "hot_missed": self.hot_missed},
            "q3_errors": self.q3_errors,
            "disagreements": self.disagreements,
        }


def _decompose(delivery_backend, pair: ProbePair, cell: str = "b2"):
    """Read the backend's DECOMPOSED Q1/Q2 for one pair.

    The cell is irrelevant to the observation (the champion is blind to the target
    pole); a hot-target cell is passed so any cell-keyed cache lands somewhere sane.
    """
    if hasattr(delivery_backend, "decompose"):
        return delivery_backend.decompose(pair.text, pair.context, cell)
    core = getattr(delivery_backend, "core", delivery_backend)
    return dd.decomposition_from_labels(core.labels(pair.text, pair.context, cell))


def run_smoketest(
    delivery_backend,
    pairs: Optional[List[ProbePair]] = None,
    bar: float = KAPPA_BAR,
) -> SmokeTestResult:
    """Score the authored contrast set with the exact delivery backend in play.

    Reports kappa with a CI, but note it is very wide on ~10 items by
    construction — read `accuracy` and the confusion counts here, and treat
    `clears` as advisory. Nothing blocks on this result.
    """
    pairs = pairs if pairs is not None else CONTRAST_PAIRS
    human = [p.human for p in pairs]
    decomps = [_decompose(delivery_backend, p) for p in pairs]
    model = [dd.observed_hot_label(d) for d in decomps]

    agr = agreement(human, model, HOT_LABELS, bar=bar)
    accuracy = sum(1 for h, m in zip(human, model) if h == m) / len(pairs)
    grievance_scored_hot = sum(1 for p, m in zip(pairs, model)
                               if p.human == "not_hot" and m == "hot")
    hot_missed = sum(1 for p, m in zip(pairs, model) if p.human == "hot" and m == "not_hot")
    q3_errors = sum(1 for p, d in zip(pairs, decomps)
                    if d.q3_closeness_toward_listener != p.q3)
    disagreements = [
        {"text": p.text[:80], "human": p.human, "model": m,
         "q1": d.q1_hostility_toward_listener, "q3": d.q3_closeness_toward_listener,
         "note": p.note}
        for p, m, d in zip(pairs, model, decomps) if p.human != m
    ]

    return SmokeTestResult(
        n=len(pairs), accuracy=accuracy, kappa=agr.kappa,
        kappa_ci_low=agr.kappa_ci_low, clears=agr.passed,
        grievance_scored_hot=grievance_scored_hot, hot_missed=hot_missed,
        q3_errors=q3_errors, disagreements=disagreements,
    )
