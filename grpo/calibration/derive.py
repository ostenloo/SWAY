"""Band-edge derivation from grader-space AnnoMI labels (grpo_spec_2 §6.5).

Pure math — no I/O, no network, no model calls — so the derivation is testable
in isolation and `annomi_calibrate.py` is only plumbing around it.

**What this computes and what it claims.** For each axis and each direction `d`,
the band edges are **percentiles of the per-SESSION `q_d` distribution** in
grader space:

    L_design(d) = P_lo-th percentile        U(d) = P_hi-th percentile

Three things follow, and §6.5 is emphatic that they travel together:

1. **Per patient, not pooled.** Each AnnoMI session is one real client — the
   closest thing the corpus has to a profile. Pooling turns across clients
   measures a mixture, not a patient.
2. **Eligibility.** A session enters the distribution only with `>= min_marked`
   turns on that axis (§6.5 uses 8). Below that `q` is noise.
3. **Shrinkage before percentiles.** Empirical-Bayes shrink each session's `q`
   toward the pooled mean, so an 8-turn session that happens to read 8/8 does
   not set the ceiling. Without this the tails are sampling artifacts and step 4
   reads them as heterogeneity.

**The epistemic claim.** This is NOT a bound. [BAND §6.3]'s pooled `L_ext` was a
genuine lower bound via a dilution argument; a per-patient percentile is a
**chosen position inside a real distribution** — "the simulated patient should be
at least as directional as the `P_lo`-th percentile real patient of that
direction." Stronger and more useful, but a different claim. `P_lo` sits high
(70-75) because the floor should be *the least directional patients who still
present in direction d*, and deliberately not at an extreme percentile, which
would be winner's-curse territory.

**The sorting distinction** (§6.5, three ways, only one legitimate):

  | move | verdict |
  |---|---|
  | sort TURNS by their own label, then measure that label | circular, useless |
  | group by SESSION (exogenous, fixed before labeling), read the distribution | legitimate |
  | take an extreme PERCENTILE of that distribution as a parameter | legitimate but upward-biased — needs eligibility + shrinkage |

**CB2 stands absolutely**: this is human data. Never the Simulator's own
rollouts — that is the ratchet failure the whole offline stage exists to prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from grpo.reward.band_reward import MARKED, UNMARKED


#: §6.5 — a session needs this many MARKED turns on an axis to enter that axis's
#: distribution. Below it, `q` is dominated by sampling noise.
MIN_MARKED_PER_SESSION = 8

#: D2.4 — the uninformative-bracket detector. Warn and disclose, never halt.
WIDE_BRACKET = 0.35
THIN_SAMPLE_SESSIONS = 25


@dataclass(frozen=True)
class SessionCounts:
    """One session's grader-space tallies on one axis."""

    session_id: str
    n_turns: int
    n_marked: int
    #: label -> count, over the MARKED turns only (the q numerator candidates).
    on_counts: dict

    @property
    def d(self) -> float:
        """Marked fraction — feeds `d_anno`, the soft anchor for `d_floor`."""
        return self.n_marked / self.n_turns if self.n_turns else 0.0

    def raw_q(self, direction: str) -> Optional[float]:
        """Unsmoothed share of marked turns in `direction`. None if unmarked."""
        if not self.n_marked:
            return None
        return self.on_counts.get(direction, 0) / self.n_marked


def session_counts(axis: str, labels_by_session: dict) -> list[SessionCounts]:
    """Tally per-session marked counts from grader labels.

    `labels_by_session`: session_id -> sequence of per-turn categorical labels
    for THIS axis (the grader's output, e.g. {internalizing, externalizing,
    neutral}).
    """
    marked_set = set(MARKED[axis])
    out = []
    for sid, labels in labels_by_session.items():
        labels = list(labels)
        marked = [x for x in labels if x in marked_set]
        counts = {lab: sum(1 for x in marked if x == lab) for lab in MARKED[axis]}
        out.append(SessionCounts(
            session_id=str(sid), n_turns=len(labels), n_marked=len(marked), on_counts=counts))
    out.sort(key=lambda s: s.session_id)
    return out


# ── empirical-Bayes shrinkage ────────────────────────────────────────────────

@dataclass(frozen=True)
class ShrinkageFit:
    """Beta-binomial method-of-moments fit, kept for disclosure."""

    mu: float                 # pooled mean share
    kappa: float              # prior strength (beta a + b); inf = full shrinkage
    tau2: float               # estimated between-session variance
    n_sessions: int
    fully_shrunk: bool        # tau2 <= 0: no detectable between-patient spread


def fit_shrinkage(counts: Sequence[SessionCounts], direction: str) -> ShrinkageFit:
    """Beta-binomial MoM fit of the between-session spread in `q_direction`.

    Standard decomposition: the observed variance of the per-session shares is
    between-patient variance PLUS binomial sampling variance. Subtract the
    latter; what remains is the real heterogeneity the percentiles are meant to
    read. If nothing remains (`tau2 <= 0`) every apparent difference between
    sessions is sampling noise, and the honest response is full shrinkage — the
    distribution collapses to a point and `bracket_informative` will say so.
    """
    ns = [c.n_marked for c in counts]
    ons = [c.on_counts.get(direction, 0) for c in counts]
    total_n = sum(ns)
    if not total_n or len(counts) < 2:
        mu = (sum(ons) / total_n) if total_n else 0.0
        return ShrinkageFit(mu=mu, kappa=math.inf, tau2=0.0,
                            n_sessions=len(counts), fully_shrunk=True)

    mu = sum(ons) / total_n
    qs = [o / n for o, n in zip(ons, ns)]
    # Unbiased variance of the observed per-session shares.
    s2 = sum((q - mu) ** 2 for q in qs) / (len(qs) - 1)
    # Expected within-session (binomial) contribution at each session's own n.
    within = mu * (1.0 - mu) * (sum(1.0 / n for n in ns) / len(ns))
    tau2 = s2 - within

    if tau2 <= 0 or mu <= 0 or mu >= 1:
        return ShrinkageFit(mu=mu, kappa=math.inf, tau2=max(tau2, 0.0),
                            n_sessions=len(counts), fully_shrunk=True)

    kappa = mu * (1.0 - mu) / tau2 - 1.0
    if kappa <= 0:              # spread wider than a beta can express: no shrinkage
        return ShrinkageFit(mu=mu, kappa=0.0, tau2=tau2,
                            n_sessions=len(counts), fully_shrunk=False)
    return ShrinkageFit(mu=mu, kappa=kappa, tau2=tau2,
                        n_sessions=len(counts), fully_shrunk=False)


def shrink(count: SessionCounts, direction: str, fit: ShrinkageFit) -> float:
    """`q` for one session, pulled toward the pooled mean by the fitted prior."""
    n = count.n_marked
    on = count.on_counts.get(direction, 0)
    if math.isinf(fit.kappa):
        return fit.mu
    return (on + fit.kappa * fit.mu) / (n + fit.kappa)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile. Stdlib only — this runs in the reward
    package's dependency footprint, not numpy's."""
    if not values:
        raise ValueError("percentile of an empty distribution")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[int(k)]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


# ── the derivation ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DirectionBracket:
    """The derived edges for one axis-direction, with everything disclosed."""

    axis: str
    direction: str
    L_design: float
    U: float
    P_lo: float
    P_hi: float
    n_eligible_sessions: int
    n_sessions_total: int
    d_anno_pooled: float          # marked fraction over all graded turns
    d_anno_median_session: float  # median per-session marked fraction
    mu: float
    kappa: float
    tau2: float
    fully_shrunk: bool
    bracket_informative: bool
    informative_reasons: tuple = ()
    shrunk_q: tuple = ()          # the distribution the percentiles were read off

    def to_provenance(self) -> dict:
        """Everything a reader needs to reproduce or distrust the number."""
        return {
            "P_lo": self.P_lo,
            "P_hi": self.P_hi,
            "n_eligible_sessions": self.n_eligible_sessions,
            "n_sessions_total": self.n_sessions_total,
            "min_marked_per_session": MIN_MARKED_PER_SESSION,
            "d_anno_pooled": round(self.d_anno_pooled, 4),
            "d_anno_median_session": round(self.d_anno_median_session, 4),
            "eb_mu": round(self.mu, 4),
            "eb_kappa": (None if math.isinf(self.kappa) else round(self.kappa, 3)),
            "eb_tau2": round(self.tau2, 6),
            "fully_shrunk": self.fully_shrunk,
            "informative_reasons": list(self.informative_reasons),
        }


def derive_bracket(
    axis: str,
    direction: str,
    labels_by_session: dict,
    *,
    P_lo: float = 72.5,
    P_hi: float = 92.5,
    min_marked: int = MIN_MARKED_PER_SESSION,
) -> DirectionBracket:
    """Per-session percentile derivation for one axis-direction (§6.5).

    `P_lo`/`P_hi` default to the midpoints of §6.5's disclosed ranges (70-75,
    90-95). They are parameters, not constants, and both are stamped into the
    artifact's provenance — a band edge whose percentile is not recorded is not
    reproducible.
    """
    if direction not in MARKED[axis]:
        raise ValueError(
            f"{direction!r} is not a marked direction on {axis} "
            f"(marked = {MARKED[axis]}, unmarked = {UNMARKED[axis]!r})"
        )

    all_counts = session_counts(axis, labels_by_session)
    eligible = [c for c in all_counts if c.n_marked >= min_marked]

    total_turns = sum(c.n_turns for c in all_counts)
    total_marked = sum(c.n_marked for c in all_counts)
    d_pooled = total_marked / total_turns if total_turns else 0.0
    per_session_d = sorted(c.d for c in all_counts)
    d_median = percentile(per_session_d, 50) if per_session_d else 0.0

    if not eligible:
        # Nothing to read. Emit a refusal-shaped bracket the caller must handle
        # rather than a plausible-looking number nobody can trace.
        return DirectionBracket(
            axis=axis, direction=direction, L_design=float("nan"), U=float("nan"),
            P_lo=P_lo, P_hi=P_hi, n_eligible_sessions=0,
            n_sessions_total=len(all_counts), d_anno_pooled=d_pooled,
            d_anno_median_session=d_median, mu=0.0, kappa=math.inf, tau2=0.0,
            fully_shrunk=True, bracket_informative=False,
            informative_reasons=("no session reached min_marked",),
        )

    fit = fit_shrinkage(eligible, direction)
    qs = [shrink(c, direction, fit) for c in eligible]

    L = percentile(qs, P_lo)
    U = percentile(qs, P_hi)

    reasons = []
    if U - L > WIDE_BRACKET:
        reasons.append(f"U - L_design = {U - L:.3f} > {WIDE_BRACKET}")
    if len(eligible) < THIN_SAMPLE_SESSIONS:
        reasons.append(f"only {len(eligible)} eligible sessions (< {THIN_SAMPLE_SESSIONS})")
    if fit.fully_shrunk:
        reasons.append("no detectable between-session spread (tau2 <= 0); "
                       "the percentiles collapse to the pooled mean")

    return DirectionBracket(
        axis=axis, direction=direction, L_design=L, U=U, P_lo=P_lo, P_hi=P_hi,
        n_eligible_sessions=len(eligible), n_sessions_total=len(all_counts),
        d_anno_pooled=d_pooled, d_anno_median_session=d_median,
        mu=fit.mu, kappa=fit.kappa, tau2=fit.tau2, fully_shrunk=fit.fully_shrunk,
        bracket_informative=not reasons, informative_reasons=tuple(reasons),
        shrunk_q=tuple(qs),
    )


# ── assembling the artifact ──────────────────────────────────────────────────

#: CB4 — `U` must be strictly below 1.0 or the caricature sits on the plateau and
#: the band's entire purpose is void. A percentile of a real distribution can
#: legitimately land at or above 1.0 (a fully-shrunk distribution with mu near 1,
#: for instance), so it is clamped here rather than left to fail at load.
U_CEILING = 0.97

#: A derived `L` at or below zero is meaningless; clamp to keep the artifact
#: loadable and record the clamp.
L_FLOOR = 0.05


def clamp_edges(L: float, U: float) -> tuple:
    """Enforce `0 < L < U < 1` (CB3/CB4), reporting any clamp for disclosure."""
    notes = []
    if math.isnan(L) or math.isnan(U):
        return L, U, ("edges are NaN — no eligible sessions",)
    if U > U_CEILING:
        notes.append(f"U clamped {U:.3f} -> {U_CEILING} (CB4: U < 1.0)")
        U = U_CEILING
    if L < L_FLOOR:
        notes.append(f"L_design clamped {L:.3f} -> {L_FLOOR}")
        L = L_FLOOR
    if L >= U:
        # A degenerate (fully-shrunk) distribution puts both percentiles on the
        # same point. Open a minimal symmetric interval so the artifact loads;
        # `bracket_informative: false` is already carrying the warning.
        mid = min(max((L + U) / 2.0, L_FLOOR + 0.02), U_CEILING - 0.02)
        L, U = mid - 0.02, mid + 0.02
        notes.append(f"degenerate bracket (L >= U) opened to [{L:.3f}, {U:.3f}]")
    return L, U, tuple(notes)


def density_low_params(
    d_anno_neutral_low: float,
    engine_d_floor: float,
    *,
    d_lo: float = 0.05,
) -> dict:
    """Engine params for a NEUTRAL-engine cell (D2.3, [BAND §5.3]).

    Neutral means **absent** engine, so the target is a two-sided band on `d_eng`
    toward a low anchor, not a `q`-band — the patient does not do attribution, so
    few turns are marked at all.

    `d_lo > 0` STRICTLY, and that is the whole point of this function's default.
    [BAND §5.3] permits `d_lo: 0.0`; at exactly zero an arc expressing no engine
    whatsoever scores 1.0 — full marks on half of b5's and b6's reward for going
    limp, on precisely the cells that have no `q`-band standing behind them. A
    control patient expresses *some* attribution, just rarely, so `d_eng = 0`
    should sit on the lower shoulder and be gently penalised.

    `d_hi` is anchored **below the on-profile cells' `d_floor`** ([BAND §5.3]) and
    below the observed low end of AnnoMI's engine density, whichever is smaller.
    """
    d_hi = min(d_anno_neutral_low, 0.8 * engine_d_floor)
    d_hi = max(d_hi, d_lo + 0.03)      # keep the band openable
    return {
        "mode": "density_low",
        "d_lo": round(d_lo, 4),
        "d_hi": round(d_hi, 4),
        # Tight lower shoulder: going fully inert must cost visibly. At
        # d_lo = 0.05 and s_lo = 0.04, d = 0 scores exp(-0.78) ~ 0.46.
        "s_lo": 0.04,
        "s_hi": 0.06,
    }
