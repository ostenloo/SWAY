"""The rate-profile derivation ([RATE §6]) — the live path, on synthetic corpora.

`test_rate_profile_reward.py` covers the frozen artifact. This covers the code
that will REPLACE it the moment the grader label cache is restored: eligibility,
grouping, percentiles, and the §5.2 widening.
"""

import pytest

import grpo._bootstrap  # noqa: F401
from grpo.calibration import rate_derive as RD
from grpo.reward.rate_profile_reward import (
    MIN_SPAN_TURNS, OFF_DIRECTION, ON_DIRECTION, calibration_from_dict,
)

T = 20


def conv(n_int, n_ext, n_turns):
    return (["internalizing"] * n_int + ["externalizing"] * n_ext
            + ["neutral"] * (n_turns - n_int - n_ext))


# ── §6 step 2: eligibility is a LENGTH rule, and nothing else ────────────────

def test_eligibility_is_length_only_and_ignores_marked_count():
    """The §1.6 fix. The band spec's `>= 8 marked turns` excluded 95 of 133
    conversations, 92 of them (97%) for being SHORT — so it selected on length
    and the resulting band was measured on 43-turn sessions and applied to
    20-turn arcs."""
    corpus = {
        "long_but_silent": conv(0, 0, 40),      # 40 turns, ZERO marked
        "short_but_loud": conv(6, 0, 8),        # 8 turns, 6 marked
    }
    rates = RD.conversation_rates("engine", corpus)
    by_id = {c.session_id: c for c in rates}

    assert by_id["long_but_silent"].eligible is True
    assert by_id["short_but_loud"].eligible is False
    assert RD.ELIGIBILITY_MIN_TURNS == 10


def test_no_marked_count_threshold_exists_anywhere_in_the_module():
    """If one reappears, §1.6 reappears with it. Checked over the module's names
    rather than its text, so the prose explaining the deletion does not trip it."""
    # MARKED / UNMARKED are the label vocabularies, not thresholds.
    assert not [n for n in dir(RD)
                if "marked" in n.lower() and n not in ("MARKED", "UNMARKED")]
    assert "min_marked" not in RD.derive_edges.__code__.co_varnames
    assert "min_marked" not in RD.conversation_rates.__code__.co_varnames


# ── §6 step 3: raw counts over T (C7) ────────────────────────────────────────

def test_rates_are_raw_counts_over_all_turns_including_unmarked():
    c = RD.conversation_rates("engine", {"s": conv(4, 1, 20)})[0]
    assert c.rates["internalizing"] == pytest.approx(4 / 20)
    assert c.rates["externalizing"] == pytest.approx(1 / 20)
    # The unmarked class is in the DENOMINATOR and gets no rate of its own.
    assert "neutral" not in c.rates
    assert sum(c.rates.values()) < 1.0


# ── §6 step 4: the partition, and the precedence this implementation chose ──

def test_low_rate_is_tested_before_lean_and_the_groups_are_disjoint():
    """§6 step 4's three predicates overlap as written: `(0.08, 0.02)` is both
    internalizing-leaning and low-rate. §7's own counts show the source run did
    not disambiguate (50 + 39 + 26 = 115 over 100 eligible)."""
    corpus = {
        "faint": conv(2, 0, 25),        # p_int 0.08, p_ext 0.00 — both <= 0.10
        "clear": conv(5, 1, 20),        # p_int 0.25 — internalizing-leaning
        "ext": conv(1, 4, 20),          # externalizing-leaning
    }
    g = RD.group_conversations(RD.conversation_rates("engine", corpus), "engine")

    assert [c.session_id for c in g[RD.GROUP_LOW_RATE]] == ["faint"]
    assert [c.session_id for c in g["internalizing"]] == ["clear"]
    assert [c.session_id for c in g["externalizing"]] == ["ext"]

    members = [c.session_id for group in g.values() for c in group]
    assert len(members) == len(set(members)) == 3      # disjoint, total


def test_two_sided_conversations_are_dropped_not_assigned_arbitrarily():
    corpus = {"tied": conv(4, 4, 20)}                   # 0.20 vs 0.20
    rates = RD.conversation_rates("engine", corpus)
    g = RD.group_conversations(rates, "engine")
    assert all(not v for v in g.values())
    assert RD.dropped_ties(rates, "engine") == 1


# ── §6 step 5 + §5.2/§5.3 ────────────────────────────────────────────────────

def test_off_direction_edges_have_no_lower_edge_even_when_the_p25_is_positive():
    """§5.3 — a nonzero floor would be less realistic than the corpus."""
    # Every conversation carries some externalizing, so the p25 is well above 0.
    corpus = {f"s{i}": conv(5, 3, 20) for i in range(8)}
    rates = RD.conversation_rates("engine", corpus)
    edges = RD.engine_profile_edges("internalizing", rates, T=T)
    off = next(e for e in edges if e.role == OFF_DIRECTION)
    assert off.p25 > 0
    assert off.L == 0.0


def test_a_sub_span_off_direction_band_widens_upward_and_says_so():
    """§5.3 pins the lower edge, so §5.2 can only widen upward — which LOOSENS a
    realism constraint. That is why the widening is recorded per band."""
    L, U, note = RD.widen_to_min_span(0.0, 0.071, T, pin_lower=True)
    assert (L, U) == (0.0, MIN_SPAN_TURNS / T)
    assert "LOOSENS" in note


def test_a_sub_span_on_direction_band_widens_about_its_midpoint():
    L, U, note = RD.widen_to_min_span(0.20, 0.24, T, pin_lower=False)
    assert 0.5 * (L + U) == pytest.approx(0.22)        # measured centre kept
    assert U - L == pytest.approx(MIN_SPAN_TURNS / T)
    assert "midpoint" in note


def test_an_empty_calibration_group_is_a_refusal_not_an_invented_edge():
    corpus = {f"s{i}": conv(5, 0, 20) for i in range(4)}   # no externalizing leaners
    rates = RD.conversation_rates("engine", corpus)
    with pytest.raises(ValueError, match="empty"):
        RD.engine_profile_edges("externalizing", rates, T=T)


# ── §2.2: neutral is not a special case ──────────────────────────────────────

def test_neutral_profile_is_two_off_direction_bands_not_a_separate_mode():
    corpus = {f"s{i}": conv(i % 2, 0, 20) for i in range(10)}    # all low-rate
    rates = RD.conversation_rates("engine", corpus)
    edges = RD.engine_profile_edges("neutral", rates, T=T)

    assert [e.role for e in edges] == [OFF_DIRECTION, OFF_DIRECTION]
    assert all(e.L == 0.0 for e in edges)
    assert all(e.group == RD.GROUP_LOW_RATE for e in edges)
    # `density_low` is deleted — one mechanism covers all six cells (§2.2). The
    # module exposes no such mode, and the loader would refuse one.
    assert not [n for n in dir(RD) if "density" in n.lower()]


# ── the whole document, through the real loader ──────────────────────────────

def test_a_derived_document_loads_and_every_band_is_valid():
    corpus = {}
    for i in range(20):
        corpus[f"int{i}"] = conv(3 + i % 4, i % 2, 20)
        corpus[f"ext{i}"] = conv(i % 2, 3 + i % 4, 20)
        corpus[f"low{i}"] = conv(i % 2, 0, 20)
    rates = RD.conversation_rates("engine", corpus)

    cells, edges = RD.build_cells(
        ["b1", "b3", "b5"], lambda t: RD.engine_profile_edges(t, rates, T=T), T=T)
    cal = calibration_from_dict({
        "arc_length_T": T, "grader_version": "test",
        "backend_identities": {"engine": "e", "delivery": "d"},
        "cells": cells,
    })

    assert cal["b1"]["engine"].on_direction.label == "internalizing"
    assert cal["b3"]["engine"].on_direction.label == "externalizing"
    assert cal["b5"]["engine"].on_direction is None          # neutral asserts no pole
    # Delivery is declared on every cell (§8), including the neutral ones.
    for cell in ("b1", "b3", "b5"):
        assert cal[cell]["delivery"].measured is False
        assert cal[cell]["delivery"].on_direction is not None
