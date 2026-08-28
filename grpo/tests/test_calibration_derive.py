"""Tests for the §6.5 band-edge derivation (per-session q -> EB -> percentiles)."""

import math

import pytest

import grpo._bootstrap  # noqa: F401
from grpo.calibration import derive as D


def sess(n_turns, n_on, n_off, direction="internalizing", other="externalizing"):
    return {"n_turns": n_turns, direction: n_on, other: n_off}


def labels(n_turns, n_on, n_off, on="internalizing", off="externalizing", unmarked="neutral"):
    return [on] * n_on + [off] * n_off + [unmarked] * (n_turns - n_on - n_off)


# ── session tallies ──────────────────────────────────────────────────────────

def test_session_counts_uses_the_axis_marked_set():
    lbs = {"s1": labels(20, 6, 2), "s2": labels(10, 1, 1)}
    cs = {c.session_id: c for c in D.session_counts("engine", lbs)}
    assert cs["s1"].n_marked == 8 and cs["s1"].n_turns == 20
    assert cs["s1"].d == pytest.approx(0.4)
    assert cs["s1"].raw_q("internalizing") == pytest.approx(0.75)
    assert cs["s2"].n_marked == 2


def test_unmarked_session_has_no_q():
    lbs = {"s1": ["neutral"] * 12}
    c = D.session_counts("engine", lbs)[0]
    assert c.n_marked == 0 and c.raw_q("internalizing") is None


# ── empirical-Bayes shrinkage ────────────────────────────────────────────────

def test_shrinkage_pulls_small_sessions_harder_than_large_ones():
    """The whole point: an 8-turn session reading 8/8 must not set the ceiling."""
    lbs = {
        "small": labels(30, 8, 0),      # 8/8 = 1.00 on 8 marked turns
        **{f"big{i}": labels(60, 30, 20) for i in range(8)},   # 30/50 = 0.60 on 50
    }
    counts = D.session_counts("engine", lbs)
    fit = D.fit_shrinkage(counts, "internalizing")
    by_id = {c.session_id: c for c in counts}

    small_raw = by_id["small"].raw_q("internalizing")
    small_shrunk = D.shrink(by_id["small"], "internalizing", fit)
    big_raw = by_id["big0"].raw_q("internalizing")
    big_shrunk = D.shrink(by_id["big0"], "internalizing", fit)

    assert small_raw == 1.0 and big_raw == pytest.approx(0.6)
    assert abs(small_shrunk - fit.mu) < abs(small_raw - fit.mu)
    # The small session moves strictly further than the large one.
    assert (small_raw - small_shrunk) > (big_raw - big_shrunk)


def test_no_between_session_spread_collapses_to_the_pooled_mean():
    """All sessions identical -> tau2 <= 0 -> full shrinkage, and it says so."""
    lbs = {f"s{i}": labels(40, 12, 8) for i in range(10)}
    counts = D.session_counts("engine", lbs)
    fit = D.fit_shrinkage(counts, "internalizing")
    assert fit.fully_shrunk is True
    assert math.isinf(fit.kappa)
    assert all(D.shrink(c, "internalizing", fit) == pytest.approx(fit.mu) for c in counts)


def test_wide_real_spread_shrinks_little():
    lbs = {f"lo{i}": labels(60, 10, 40) for i in range(6)}       # q = 0.20
    lbs.update({f"hi{i}": labels(60, 40, 10) for i in range(6)})  # q = 0.80
    counts = D.session_counts("engine", lbs)
    fit = D.fit_shrinkage(counts, "internalizing")
    assert not fit.fully_shrunk
    by_id = {c.session_id: c for c in counts}
    assert D.shrink(by_id["hi0"], "internalizing", fit) > 0.75


# ── percentiles ──────────────────────────────────────────────────────────────

def test_percentile_interpolates():
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert D.percentile(xs, 0) == 0.0
    assert D.percentile(xs, 100) == 1.0
    assert D.percentile(xs, 50) == pytest.approx(0.5)
    assert D.percentile(xs, 75) == pytest.approx(0.75)
    assert D.percentile(xs, 72.5) == pytest.approx(0.725)


# ── the bracket ──────────────────────────────────────────────────────────────

def _spread_corpus(n=40):
    """A corpus with genuine between-session heterogeneity in q_internalizing."""
    lbs = {}
    for i in range(n):
        on = 4 + (i % 13)                    # 4..16 on-turns out of 20 marked
        lbs[f"s{i:02d}"] = labels(40, on, 20 - on)
    return lbs


def test_derive_bracket_orders_and_discloses():
    b = D.derive_bracket("engine", "internalizing", _spread_corpus(), P_lo=72.5, P_hi=92.5)
    assert 0.0 < b.L_design < b.U < 1.0
    assert b.n_eligible_sessions == 40
    assert b.bracket_informative is True
    prov = b.to_provenance()
    assert prov["P_lo"] == 72.5 and prov["P_hi"] == 92.5
    assert prov["n_eligible_sessions"] == 40
    assert prov["min_marked_per_session"] == D.MIN_MARKED_PER_SESSION


def test_eligibility_excludes_thinly_marked_sessions():
    lbs = _spread_corpus(30)
    lbs.update({f"thin{i}": labels(20, 2, 1) for i in range(20)})   # 3 marked each
    b = D.derive_bracket("engine", "internalizing", lbs)
    assert b.n_sessions_total == 50
    assert b.n_eligible_sessions == 30       # the 20 thin ones never enter


def test_d2_4_flags_a_thin_sample():
    """D2.4 — warn and disclose, do not halt. §6.5 expects this on DELIVERY."""
    lbs = {f"s{i}": labels(30, 6, 6, on="warm", off="hot", unmarked="flat")
           for i in range(11)}
    b = D.derive_bracket("delivery", "warm", lbs)
    assert b.bracket_informative is False
    assert any("11 eligible sessions" in r for r in b.informative_reasons)


def test_d2_4_flags_a_wide_bracket():
    lbs = {}
    for i in range(40):
        on = 1 + (i % 20)                    # q sweeps 0.05 -> 1.00
        lbs[f"s{i:02d}"] = labels(40, on, 20 - on)
    b = D.derive_bracket("engine", "internalizing", lbs, P_lo=10, P_hi=95)
    assert b.U - b.L_design > D.WIDE_BRACKET
    assert b.bracket_informative is False


def test_no_eligible_sessions_returns_a_refusal_not_a_number():
    lbs = {f"s{i}": labels(20, 1, 1) for i in range(30)}
    b = D.derive_bracket("engine", "internalizing", lbs)
    assert b.n_eligible_sessions == 0
    assert math.isnan(b.L_design) and math.isnan(b.U)
    assert b.bracket_informative is False


def test_derive_rejects_an_unmarked_direction():
    with pytest.raises(ValueError, match="not a marked direction"):
        D.derive_bracket("engine", "neutral", _spread_corpus())
    with pytest.raises(ValueError, match="not a marked direction"):
        D.derive_bracket("delivery", "flat", _spread_corpus())


# ── clamping (CB3/CB4) ───────────────────────────────────────────────────────

def test_clamp_enforces_the_interiority_ceiling():
    L, U, notes = D.clamp_edges(0.80, 1.0)
    assert U == D.U_CEILING < 1.0
    assert any("CB4" in n for n in notes)


def test_clamp_opens_a_degenerate_bracket():
    L, U, notes = D.clamp_edges(0.70, 0.70)      # fully-shrunk distribution
    assert L < U
    assert any("degenerate" in n for n in notes)


def test_clamp_passes_a_healthy_bracket_through_untouched():
    L, U, notes = D.clamp_edges(0.70, 0.90)
    assert (L, U, notes) == (0.70, 0.90, ())


# ── D2.3 neutral-engine params ───────────────────────────────────────────────

def test_density_low_never_emits_a_zero_lower_anchor():
    p = D.density_low_params(d_anno_neutral_low=0.02, engine_d_floor=0.30)
    assert p["mode"] == "density_low"
    assert p["d_lo"] > 0, "D2.3: d_lo = 0 lets an inert arc score 1.0"
    assert p["d_lo"] < p["d_hi"] < 1.0


def test_density_low_anchors_below_the_on_profile_floor():
    p = D.density_low_params(d_anno_neutral_low=0.50, engine_d_floor=0.30)
    assert p["d_hi"] <= 0.8 * 0.30 + 1e-9


def test_density_low_params_load_as_a_valid_artifact_entry():
    """The params must survive the reward's own load-time asserts (A2)."""
    from grpo.reward.band_reward import calibration_from_dict
    eng = D.density_low_params(0.10, 0.30)
    doc = {"cells": {"b5": {
        "engine": eng,
        "delivery": {"mode": "q_band", "c_star": "warm", "L_design": 0.62, "U": 0.88,
                     "s_lo": 0.06, "s_hi": 0.12, "alpha": 0.5, "d_floor": 0.15},
    }}}
    cal = calibration_from_dict(doc)
    assert cal["b5"]["engine"].mode == "density_low"
    assert cal["b5"]["engine"].d_lo > 0
