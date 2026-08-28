"""Rate-profile target derivation from grader-space labels ([RATE §6]).

Pure math — no I/O, no network, no model calls — so the derivation is testable in
isolation and `annomi_calibrate.py` is only plumbing around it.

REPLACES `derive.py` in full. Everything that module computes — per-session
conditional `q`, the empirical-Bayes shrinkage, `MIN_MARKED_PER_SESSION = 8`, the
72.5th/92.5th percentile window, `density_low_params` — is superseded, and the
reasons are measured, not stylistic ([RATE §1]). `derive.py` is retained unwired
as the record of those measurements.

**What this computes.** For each conversation, the RATE of each marked label over
ALL its turns (§2). Conversations are grouped by which way they lean, and the
band edges are the **25th and 75th percentiles** of the matching group's rate
distribution (§6 step 5).

**Three things that are decisions, not measurements**, and are recorded as such:

1. **The percentile window** (§6.1). 25th-75th targets the middle half of
   conversations that lean the intended way. Changing it is a design decision
   needing its own justification, not a retune.
2. **The grouping precedence** (see `group_conversations`). §6 step 4 says
   "partition", but its three predicates overlap; the order is chosen here and
   stamped into provenance.
3. **Which way a sub-span band widens** (see `widen_to_min_span`). §5.2 requires
   a minimum span; §5.3 pins off-direction lower edges at zero. Those two
   together force off-direction bands to widen UPWARD, which loosens a realism
   constraint, so the widening is recorded per band.

**C5 stands absolutely**: human corpus only. Never the Simulator's own authored
or rollout output — that is the calibration-circularity ratchet.

**C7**: rates come from `rate_profile_reward.rate_of`, imported rather than
reimplemented, so the calibration path and the scoring path are the same code.
A6 is true by construction; the test pins it against regression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from grpo.reward.band_reward import MARKED, UNMARKED
from grpo.reward.rate_profile_reward import (
    MIN_SPAN_TURNS,
    OFF_DIRECTION,
    ON_DIRECTION,
    SHOULDER,
    rate_of,
)


#: §6 step 2 — eligibility is a LENGTH rule, and nothing else. This is the §1.6
#: fix. The band spec's `>= 8 marked turns` excluded 95 of 133 conversations and
#: 92 of those 95 (97%) were excluded for being SHORT, not for lacking direction;
#: the excluded conversations' median externalizing share was 0.333 against the
#: eligible ones' 0.559, so the old rule selected on length and the resulting
#: band was measured on 43-turn sessions and applied to 20-turn arcs.
#:
#: NO MARKED-COUNT THRESHOLD EXISTS ANYWHERE IN THIS PIPELINE. If one reappears,
#: §1.6 reappears with it.
ELIGIBILITY_MIN_TURNS = 10

#: §6 step 4 — a conversation with both marked rates at or below this expresses
#: neither pole often enough to say which way it leans. It calibrates the NEUTRAL
#: profile.
LOW_RATE_CEILING = 0.10

#: §6.1 — declared, not measured.
P_LO = 25.0
P_HI = 75.0

#: §5.3 — off-direction bands start at zero and stay there.
OFF_DIRECTION_L = 0.0

GROUP_LOW_RATE = "low_rate"


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile. Stdlib only."""
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


# ── per-conversation rates (§6 step 3) ───────────────────────────────────────

@dataclass(frozen=True)
class ConversationRates:
    """One conversation's marked rates on one axis. Raw counts over `T` (C7)."""

    session_id: str
    axis: str
    n_turns: int
    rates: dict          # marked label -> count / n_turns

    def rate(self, label: str) -> float:
        return self.rates.get(label, 0.0)

    @property
    def eligible(self) -> bool:
        return self.n_turns >= ELIGIBILITY_MIN_TURNS


def conversation_rates(axis: str, labels_by_session: dict) -> list:
    """Per-conversation rates over ALL turns, including unmarked ones.

    `labels_by_session`: session_id -> sequence of per-turn categorical labels
    for THIS axis. The unmarked class (`neutral` on engine, `flat` on delivery)
    is counted in the DENOMINATOR and never given a rate of its own —
    constraining the two marked rates constrains it (§2).
    """
    out = []
    for sid, labels in sorted(labels_by_session.items(), key=lambda kv: str(kv[0])):
        labels = list(labels)
        out.append(ConversationRates(
            session_id=str(sid), axis=axis, n_turns=len(labels),
            rates={lab: rate_of(labels, lab) for lab in MARKED[axis]},
        ))
    return out


# ── grouping by lean (§6 step 4) ─────────────────────────────────────────────

def group_conversations(rates: Sequence[ConversationRates], axis: str) -> dict:
    """Partition eligible conversations into the three calibration groups.

    Returns `{label_a: [...], label_b: [...], "low_rate": [...]}` where a
    conversation in `rates[label]` leans toward `label`.

    **The precedence is a decision, taken here.** §6 step 4 says "partition", but
    its three predicates are not disjoint as written: a conversation at
    `(p_int = 0.08, p_ext = 0.02)` satisfies both "internalizing-leaning"
    (`p_a > p_b`) and "low-rate" (both <= 0.10). §7's own group counts show the
    source measurement run did NOT disambiguate — 50 + 39 + 26 = 115 against the
    100 conversations it says meet `T >= 10`, an overlap of 15.

    **Low-rate is tested first.** A conversation that barely attributes at all
    says almost nothing about how strongly an attributing patient leans, and
    including it drags the leaning group's 25th percentile toward zero — that is,
    toward a lower edge a genuinely internalizing patient would clear by being
    nearly silent. It does say something about the neutral profile, which is
    exactly what the low-rate group calibrates.

    CONSEQUENCE, recorded rather than hidden: re-deriving with this precedence
    will NOT reproduce §7's edges exactly, because §7's groups overlap. The
    frozen v1 artifact carries §7's measured numbers and says so in its
    provenance; the first re-derivation against the label cache supersedes them.
    """
    a, b = MARKED[axis]
    groups: dict = {a: [], b: [], GROUP_LOW_RATE: []}
    for c in rates:
        if not c.eligible:
            continue
        pa, pb = c.rate(a), c.rate(b)
        if pa <= LOW_RATE_CEILING and pb <= LOW_RATE_CEILING:
            groups[GROUP_LOW_RATE].append(c)
        elif pa > pb:
            groups[a].append(c)
        elif pb > pa:
            groups[b].append(c)
        else:
            # Exactly equal and both above the ceiling: leans neither way, and
            # it is not low-rate either. Dropped, and counted in provenance —
            # assigning it arbitrarily would put a two-sided conversation into a
            # one-sided group.
            continue
    return groups


def dropped_ties(rates: Sequence[ConversationRates], axis: str) -> int:
    a, b = MARKED[axis]
    return sum(
        1 for c in rates
        if c.eligible and c.rate(a) == c.rate(b)
        and not (c.rate(a) <= LOW_RATE_CEILING and c.rate(b) <= LOW_RATE_CEILING)
    )


# ── band edges (§6 step 5) + the §5.2 widening ───────────────────────────────

def widen_to_min_span(L: float, U: float, T: int, *, pin_lower: bool) -> tuple:
    """Widen `[L, U]` to satisfy §5.2's `2/T` minimum span. Returns (L, U, note).

    **Direction of widening is forced, not chosen.** For an off-direction band
    §5.3 pins `L = 0`, so the only place to go is up — which LOOSENS a realism
    constraint, admitting more of the opposite pole than the corpus showed. That
    is the price of §5.2 and it is why every widening is recorded per band rather
    than silently applied. An on-direction band has no pin and widens
    symmetrically about its midpoint, keeping the measured centre.

    §7 applies this to `EXTERNALIZING p_int in [0, 0.071]` and not to the three
    other sub-span bands it prints. It is applied uniformly here; see the load
    check in `rate_profile_reward._component_from_dict` for the argument.
    """
    min_span = MIN_SPAN_TURNS / T
    span = U - L
    if span >= min_span - 1e-12:
        return L, U, ""
    if pin_lower:
        newU = L + min_span
        return L, newU, (
            f"widened U {U:.4f} -> {newU:.4f} to reach the {MIN_SPAN_TURNS}/T = "
            f"{min_span:.4f} span floor at T = {T} (§5.2). L is pinned at 0 by §5.3, "
            "so the widening LOOSENS the off-direction ceiling above what was measured."
        )
    mid = 0.5 * (L + U)
    newL, newU = max(0.0, mid - min_span / 2.0), max(0.0, mid - min_span / 2.0) + min_span
    return newL, newU, (
        f"widened [{L:.4f}, {U:.4f}] -> [{newL:.4f}, {newU:.4f}] about its midpoint to "
        f"reach the {MIN_SPAN_TURNS}/T = {min_span:.4f} span floor at T = {T} (§5.2)."
    )


@dataclass(frozen=True)
class RateEdges:
    """Derived edges for one marked rate, with everything disclosed."""

    axis: str
    label: str
    role: str
    group: str
    L: float
    U: float
    p25: float
    p50: float
    p75: float
    n_group: int
    n_eligible: int
    n_total: int
    widening: str = ""
    measured: bool = True

    def to_component(self) -> dict:
        return {"label": self.label, "role": self.role,
                "L": round(self.L, 4), "U": round(self.U, 4)}

    def to_provenance(self) -> dict:
        return {
            "group": self.group,
            "P_lo": P_LO,
            "P_hi": P_HI,
            "percentiles": {"p25": round(self.p25, 4), "p50": round(self.p50, 4),
                            "p75": round(self.p75, 4)},
            "n_group": self.n_group,
            "n_eligible": self.n_eligible,
            "n_conversations": self.n_total,
            "eligibility": f"T >= {ELIGIBILITY_MIN_TURNS} turns (LENGTH rule, §6 step 2)",
            "estimator": "raw count / T over ALL turns, no smoothing (C7)",
            "widening": self.widening or None,
            "measured": self.measured,
        }


def edges_from_percentiles(
    axis: str,
    label: str,
    role: str,
    group_name: str,
    p25: float,
    p50: float,
    p75: float,
    *,
    T: int,
    n_group: int = 0,
    n_eligible: int = 0,
    n_total: int = 0,
) -> RateEdges:
    """Turn one rate's percentiles into a band (§6 step 5 + the §5.2 widening).

    Split out from `derive_edges` so that the ONE place edges are shaped is
    shared by the live derivation and by the frozen v1 artifact, which is built
    from §7's already-measured percentile table rather than from a label cache.
    Two code paths shaping bands differently is how §1.1 happened.
    """
    if role == OFF_DIRECTION:
        # §5.3 — no lower edge, ever. The measured p25 of the off-direction rate
        # is 0.000 anyway: a quarter of real leaning patients never express the
        # other direction at all.
        L, U = OFF_DIRECTION_L, p75
    else:
        L, U = p25, p75

    L, U, note = widen_to_min_span(L, U, T, pin_lower=(role == OFF_DIRECTION))
    return RateEdges(
        axis=axis, label=label, role=role, group=group_name, L=L, U=U,
        p25=p25, p50=p50, p75=p75, n_group=n_group, n_eligible=n_eligible,
        n_total=n_total, widening=note,
    )


def derive_edges(
    axis: str,
    label: str,
    role: str,
    group_name: str,
    group: Sequence[ConversationRates],
    *,
    all_rates: Sequence[ConversationRates],
    T: int,
    p_lo: float = P_LO,
    p_hi: float = P_HI,
) -> RateEdges:
    """25th/75th percentiles of `label`'s rate in `group`, widened per §5.2."""
    if not group:
        raise ValueError(
            f"{axis}/{label}: calibration group {group_name!r} is empty — there is "
            "nothing to read percentiles off. Do not invent an edge."
        )

    xs = [c.rate(label) for c in group]
    return edges_from_percentiles(
        axis, label, role, group_name,
        percentile(xs, p_lo), percentile(xs, 50.0), percentile(xs, p_hi),
        T=T, n_group=len(group),
        n_eligible=sum(1 for c in all_rates if c.eligible), n_total=len(all_rates),
    )


# ── assembling one axis profile ──────────────────────────────────────────────

def engine_profile_edges(
    target_direction: str,
    rates: Sequence[ConversationRates],
    *,
    T: int,
    p_lo: float = P_LO,
    p_hi: float = P_HI,
) -> list:
    """Both engine components for a cell whose engine target is `target_direction`.

    `target_direction` is one of `internalizing` / `externalizing` / `neutral`.
    A NEUTRAL target is not a special case (§2.2): it calibrates against the
    low-rate group and yields two off-direction bands. `density_low` is deleted.
    """
    axis = "engine"
    a, b = MARKED[axis]
    groups = group_conversations(rates, axis)

    if target_direction == UNMARKED[axis]:
        return [
            derive_edges(axis, lab, OFF_DIRECTION, GROUP_LOW_RATE,
                         groups[GROUP_LOW_RATE], all_rates=rates, T=T,
                         p_lo=p_lo, p_hi=p_hi)
            for lab in (a, b)
        ]

    if target_direction not in MARKED[axis]:
        raise ValueError(f"{target_direction!r} is not an engine target "
                         f"(expected {MARKED[axis]} or {UNMARKED[axis]!r})")

    other = b if target_direction == a else a
    group = groups[target_direction]
    return [
        derive_edges(axis, target_direction, ON_DIRECTION, target_direction, group,
                     all_rates=rates, T=T, p_lo=p_lo, p_hi=p_hi),
        derive_edges(axis, other, OFF_DIRECTION, target_direction, group,
                     all_rates=rates, T=T, p_lo=p_lo, p_hi=p_hi),
    ]


# ── delivery is DECLARED, not measured (§8) ──────────────────────────────────
#
# The calibration run established that the corpus cannot supply delivery targets,
# and that this is a property of the CORPUS rather than of the grader:
#
#   * 96.8% of turns are `flat`; 102 of 3,221 are marked.
#   * The median conversation contains ZERO warm or hot turns.
#   * Three independent graders agree on the scarcity: 2.5%, 6.5%, 2.2% marked on
#     the same 400 turns.
#   * The champion is not insensitive — it marks 36.0% on simulator output,
#     against 37.3% and 38.3% from two human annotators on the same turns.
#   * Stratifying by session quality barely moves it: hot runs 2.4% in the
#     deliberately-poor sessions against 1.8% in the good ones.
#
# Counselling with an intact alliance contains almost no hostility aimed at the
# counsellor. That is close to definitional, so there is nothing to measure.
#
# **The declared numbers and their justification.** The one empirical anchor
# available is §7's ENGINE geometry: a real leaning client expresses their marked
# stance on roughly 3 to 6 turns in 20 (internalizing [0.144, 0.290],
# externalizing [0.106, 0.232]), against an off-pole ceiling near 0.09. Delivery
# is the same kind of quantity — how often a stance surfaces in an arc, not how
# intense it is when it does — so the declared bands mirror that shape rather
# than inventing a different one:
#
#   on-direction  [0.150, 0.300]   3 to 6 turns of 20 carry the temperature
#   off-direction [0,     0.100]   at most 2 turns of the opposite temperature
#
# The on-direction band's UPPER edge is the anti-caricature mechanism here as
# everywhere (§5.3): a patient hot on every turn scores band(1.0) ~ 0 rather than
# winning. The LOWER edge is doing a second job the engine axis does not need it
# for — see the note in `rate_profile_reward.rate_profile_reward_arc_readout`, it
# is what stops the neutral-engine cells (b5, b6) from being rewarded for going
# inert, since their engine bands are both `[0, U]` and an empty arc scores 1.0
# there.
#
# REVERTING DELIVERY TO PER-TURN MONOTONE SCORING IS REJECTED (§8). It reinstates
# the failure this redesign exists to remove, on the axis where the hot profiles
# were already hardest to produce.
#
# If human transcripts from a domain with genuine listener-directed heat become
# available, delivery is recalibrated by §6 and `measured` flips to true.
# Complaint-handling and ombudsman recordings are the nearest family; their
# register is short and transactional and may not transfer. The prior is low.

DECLARED_DELIVERY_ON = (0.150, 0.300)
DECLARED_DELIVERY_OFF = (0.0, 0.100)

DECLARED_DELIVERY_REASON = (
    "AnnoMI cannot supply delivery targets: 96.8% of its turns are flat, only 102 of "
    "3,221 are marked, and the MEDIAN conversation contains zero warm or hot turns. "
    "This is a property of the corpus, not the grader — three independent graders "
    "agree (2.5% / 6.5% / 2.2% marked on the same 400 turns), the champion marks 36.0% "
    "on simulator output against 37.3% and 38.3% from two human annotators on the same "
    "turns, and stratifying by session quality barely moves it (hot 2.4% in the "
    "deliberately-poor sessions vs 1.8% in the good ones). Counselling with an intact "
    "alliance contains almost no hostility aimed at the counsellor. The declared edges "
    "mirror §7's measured ENGINE geometry — a marked stance surfacing on 3 to 6 turns "
    "of 20, against an off-pole ceiling near 0.10 — because that is the only empirical "
    "anchor for how often a stance surfaces in an arc. DECLARED, NOT MEASURED (§8)."
)


def delivery_profile_edges(target: str, *, T: int) -> list:
    """The two DECLARED delivery components for a cell targeting `warm` or `hot`."""
    axis = "delivery"
    if target not in MARKED[axis]:
        raise ValueError(f"{target!r} is not a delivery target (expected {MARKED[axis]})")
    other = next(lab for lab in MARKED[axis] if lab != target)

    out = []
    for lab, role, (L, U) in (
        (target, ON_DIRECTION, DECLARED_DELIVERY_ON),
        (other, OFF_DIRECTION, DECLARED_DELIVERY_OFF),
    ):
        L, U, note = widen_to_min_span(L, U, T, pin_lower=(role == OFF_DIRECTION))
        out.append(RateEdges(
            axis=axis, label=lab, role=role, group="declared", L=L, U=U,
            p25=float("nan"), p50=float("nan"), p75=float("nan"),
            n_group=0, n_eligible=0, n_total=0, widening=note, measured=False,
        ))
    return out


# ── delivery, MEASURED via a hurdle model ────────────────────────────────────
#
# The declared bands above exist because the unconditional delivery rate cannot
# be read off AnnoMI. That verdict is correct but it is not the whole picture,
# and the reason matters for what to do about it.
#
# MEASURED on the same cache that produced the engine targets, over the 100
# conversations with `T >= 10`:
#
#     axis      conversations with ZERO marked turns
#     engine     6%
#     delivery  57%
#
# Delivery is ZERO-INFLATED and engine is not. That is why delivery's
# unconditional percentiles come out degenerate (p25 = p50 = 0.000) while
# engine's do not — the distribution is dominated by structural zeros, so the
# percentiles are reading the hurdle rather than the rate.
#
# Conditioning fixes it, and the conditional distribution is well-behaved:
#
#     group                        n     p_on  (p25 / p50 / p75)
#     conversations expressing warm 32    0.0216 / 0.0296 / 0.0510
#     conversations expressing hot   23    0.0227 / 0.0476 / 0.1000
#
# **Note the irony against [RATE §1].** The conditional ratio was deleted
# because on ENGINE it threw information away — conditioning on the marked turns
# estimated from `m` instead of `T`, and every §1.2-§1.5 failure followed. But
# engine has only 6% zeros: there, the zeros are sampling, and discarding them
# discards signal. Delivery's zeros are STRUCTURAL — 57% of clients never direct
# affect at the counsellor at all — so conditioning on the hurdle is the
# correctly specified model, not a lost estimator. One parameterisation was
# applied to two axes with different distributional shapes.
#
# **A profile sets the hurdle.** A hot-delivery cell (b2, b4, b6) asserts a
# patient who directs heat at the listener; it is conditioning on part 1 by
# construction. So the conditional distribution is exactly the right target for
# it, and unlike the declared bands it is measured.
#
#     part 1, conversation level : P(expresses the pole at all) — SET by the profile
#     part 2, rate given part 1  : the band, measured here
#
# WHAT THIS DOES NOT FIX, and it is the reason the declared bands are still the
# default: see `delivery_profile_edges_hurdle`.

DELIVERY_HURDLE_REASON = (
    "Measured via a hurdle model on the AnnoMI grader cache. Delivery is zero-inflated "
    "(57% of eligible conversations carry no marked delivery turn, against 6% on "
    "engine), so its UNCONDITIONAL percentiles are degenerate (p25 = p50 = 0.000) and "
    "read the hurdle rather than the rate. Conditioning on conversations that express "
    "the pole at all gives a well-behaved distribution. A profile asserting a delivery "
    "pole sets the hurdle by construction, so the conditional distribution is its "
    "target. Conditioning is correct HERE and wrong on engine (RATE §1) because "
    "delivery's zeros are structural and engine's are sampling."
)


def delivery_hurdle_group(direction: str, rates_by_session: dict) -> list:
    """Conversations expressing `direction` at least once — the hurdle group.

    `rates_by_session` maps session_id -> the conversation's delivery labels.
    Eligibility is the same length rule as everywhere else (§6 step 2).
    """
    if direction not in MARKED["delivery"]:
        raise ValueError(f"{direction!r} is not a delivery pole")
    out = []
    for c in conversation_rates("delivery", rates_by_session):
        if c.eligible and c.rate(direction) > 0:
            out.append(c)
    return out


def delivery_profile_edges_hurdle(
    target: str,
    rates_by_session: dict,
    *,
    T: int,
    p_lo: float = P_LO,
    p_hi: float = P_HI,
) -> list:
    """MEASURED delivery components for a cell targeting `warm` or `hot`.

    Same shaping code as everywhere else, so the only difference from the
    declared path is where the percentiles come from.

    **This can fail, and the failure is informative rather than a bug.** AnnoMI's
    conditional delivery rates are small — the warm group's p25/p75 are 0.0216 /
    0.0510, i.e. 0.4 to 1.0 turns out of 20. §5.2 requires every band to span at
    least `2/T`, and widening a band that narrow about its midpoint drives its
    lower edge to or below zero. The loader then refuses it, because an
    on-direction band with `L = 0` makes the profile unfalsifiable.

    That refusal is the right outcome and it is the real finding: at `T = 20`
    these targets sit BELOW THE ARC'S RESOLUTION. The corpus conversations they
    were measured on run 40+ turns (median 40 for warm, 43 for hot), so the
    quantity exists there and cannot be represented here. A better corpus helps
    only if its rates are above roughly `1/T`; a longer arc helps regardless.
    """
    if target not in MARKED["delivery"]:
        raise ValueError(f"{target!r} is not a delivery target (expected {MARKED['delivery']})")
    other = next(lab for lab in MARKED["delivery"] if lab != target)

    group = delivery_hurdle_group(target, rates_by_session)
    if not group:
        raise ValueError(
            f"delivery/{target}: no eligible conversation expresses {target!r} — the "
            "hurdle group is empty and there is nothing to measure."
        )
    all_rates = conversation_rates("delivery", rates_by_session)
    group_name = f"expresses_{target}"

    return [
        derive_edges("delivery", target, ON_DIRECTION, group_name, group,
                     all_rates=all_rates, T=T, p_lo=p_lo, p_hi=p_hi),
        derive_edges("delivery", other, OFF_DIRECTION, group_name, group,
                     all_rates=all_rates, T=T, p_lo=p_lo, p_hi=p_hi),
    ]


def axis_entry(edges: Sequence[RateEdges], *, measured: bool) -> dict:
    """One axis's artifact entry from its two derived components."""
    prov = {e.label: e.to_provenance() for e in edges}
    if not measured:
        prov["declared_reason"] = DECLARED_DELIVERY_REASON
    elif edges and edges[0].axis == "delivery":
        prov["hurdle_reason"] = DELIVERY_HURDLE_REASON
    return {
        "measured": bool(measured),
        # §5.1 — symmetric, and the loader rejects anything else.
        "s_lo": SHOULDER,
        "s_hi": SHOULDER,
        "components": [e.to_component() for e in edges],
        "provenance": prov,
    }


# ── assembling the per-cell document ─────────────────────────────────────────

def build_cells(cells: Sequence[str], engine_edges_fn, *, T: int,
                delivery_edges_fn=None, delivery_measured: bool = False) -> tuple:
    """`(cells_doc, edges_by_cell)` for the artifact.

    `engine_edges_fn(target_direction) -> [RateEdges, RateEdges]` is injected so
    the live derivation (percentiles off a label cache) and the frozen v1 build
    (percentiles transcribed from §7's measured table) assemble the SAME
    document through the SAME shaping code. Delivery is declared identically in
    both (§8).

    The cell -> pole mapping is read from the profile roster, not hardcoded: a
    cell's engine and delivery targets are properties of the profile, and a
    second copy of them here would be a place for them to disagree.
    """
    from grpo.reward.turn_fidelity import poles_for_cell

    doc: dict = {}
    edges_by_cell: dict = {}
    for cell in cells:
        poles = poles_for_cell(cell)
        eng = engine_edges_fn(poles["engine_direction"])
        dlv = (delivery_edges_fn(poles["delivery"]) if delivery_edges_fn
               else delivery_profile_edges(poles["delivery"], T=T))
        doc[cell] = {
            "engine": axis_entry(eng, measured=True),
            "delivery": axis_entry(dlv, measured=bool(delivery_measured)),
        }
        edges_by_cell[cell] = {"engine": eng, "delivery": dlv}
    return doc, edges_by_cell
