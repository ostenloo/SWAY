"""Acceptance A1-A6 for the arc-level band reward (grpo_spec_2 §13).

A3 in particular is tested **at the configured T and d_floor, not in the
abstract**: §5.2 shows the anti-caricature property can silently fail at low
`n_marked` (at `d_floor = 0.15` and `T = 20`, `n_marked = 3` puts both 2-of-3 and
3-of-3 inside the band and the caricature scores exactly 1.0). That is the whole
reason for this redesign, so the test asserts it against real parameters and
`test_a3_fails_at_low_resolution` pins the failure mode itself.
"""

import ast
import math
from pathlib import Path

import pytest

import grpo._bootstrap  # noqa: F401
from grpo.reward import band_reward as br
from grpo.reward.band_reward import (
    AxisBand, BandCalibration, CalibrationError, axis_reward, band,
    band_reward_arc, band_reward_arc_readout, calibration_from_dict,
    context_upto, density_factor,
)


T = 20  # §11 arc_length_T — every resolution-sensitive assertion uses this.


# ── fixtures ─────────────────────────────────────────────────────────────────

def q_band(c_star, L=0.70, U=0.90, s_lo=0.06, s_hi=0.12, alpha=0.5,
           d_floor=0.30, d_ceil=None):
    return AxisBand(mode="q_band", c_star=c_star, L_design=L, U=U, s_lo=s_lo,
                    s_hi=s_hi, alpha=alpha, d_floor=d_floor, d_ceil=d_ceil)


def density_low(d_lo=0.05, d_hi=0.12, s_lo=0.04, s_hi=0.06):
    return AxisBand(mode="density_low", d_lo=d_lo, d_hi=d_hi, s_lo=s_lo, s_hi=s_hi)


def cal_for(engine, delivery, cell="b1", identities=None):
    return BandCalibration(
        cells={cell: {"engine": engine, "delivery": delivery}},
        grader_version="test",
        backend_identities=identities or {},
    )


def arc_labels(n_turns, n_marked, n_on, on_label, off_label, unmarked):
    """`n_marked` marked turns of which `n_on` are on-pole; rest unmarked."""
    assert n_on <= n_marked <= n_turns
    return ([on_label] * n_on + [off_label] * (n_marked - n_on)
            + [unmarked] * (n_turns - n_marked))


class _ScriptedBackend:
    """Replays a fixed label sequence and records exactly what it was handed."""

    def __init__(self, labels, identity="stub"):
        self.labels = list(labels)
        self.identity = identity
        self.seen = []

    def label(self, patient_turn, context, cell):
        self.seen.append((patient_turn, context, cell))
        return self.labels[len(self.seen) - 1]


class _Backends:
    def __init__(self, engine, delivery):
        self.engine = engine
        self.delivery = delivery


# ── A1: C1 — reads only (arc_turns, context0, P, cell, cal) ──────────────────

def test_a1_band_reward_module_has_no_drift_side_imports():
    src = Path(br.__file__)
    tree = ast.parse(src.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            names += [a.name for a in node.names]
    names = [n.lower() for n in names]
    for token in br.FORBIDDEN_IMPORT_TOKENS:
        assert not any(token in n for n in names), (
            f"C1 breach: band_reward.py imports something matching {token!r}: {names}"
        )


def test_a1_reward_reads_only_allowed_inputs():
    turns = ["t0", "t1", "t2"]
    eng = _ScriptedBackend(["internalizing"] * 3)
    dlv = _ScriptedBackend(["warm"] * 3)
    cal = cal_for(q_band("internalizing"), q_band("warm"))

    band_reward_arc(turns, "CTX", "PROFILE PROMPT", "b1", cal, _Backends(eng, dlv))

    # Each grader saw exactly (turn, prefix-context, cell) — never P, never a
    # MUT reply, never a drift signal.
    assert eng.seen == [
        ("t0", "CTX", "b1"),
        ("t1", "CTX\nt0", "b1"),
        ("t2", "CTX\nt0\nt1", "b1"),
    ]
    assert dlv.seen == eng.seen
    assert all("PROFILE PROMPT" not in ctx for _, ctx, _ in eng.seen)


def test_context_upto_is_prefix_only():
    assert context_upto(["a", "b", "c"], 0, "C0") == "C0"
    assert context_upto(["a", "b", "c"], 2, "C0") == "C0\na\nb"
    assert context_upto(["a"], 0, "") == ""


def test_a1_c8_assertion_available_and_fires():
    with pytest.raises(br.FamilyDisjointnessError):
        br.assert_family_disjoint("Qwen/Qwen2.5-14B-Instruct", engine="qwen2.5:7b")
    fams = br.assert_family_disjoint(
        "Qwen/Qwen2.5-14B-Instruct", engine="command-r7b", delivery="glm4:9b")
    assert fams == {"base": "qwen", "engine": "cohere", "delivery": "zhipu"}


# ── A2: load-time asserts fire on a bad artifact ─────────────────────────────

def _doc(engine=None, delivery=None):
    return {"cells": {"b1": {
        "engine": engine or {"mode": "q_band", "c_star": "int", "L_design": 0.70,
                             "U": 0.90, "s_lo": 0.06, "s_hi": 0.12, "alpha": 0.5,
                             "d_floor": 0.30},
        "delivery": delivery or {"mode": "q_band", "c_star": "warm", "L_design": 0.62,
                                 "U": 0.88, "s_lo": 0.06, "s_hi": 0.12, "alpha": 0.5,
                                 "d_floor": 0.15},
    }}}


def test_a2_good_artifact_loads_and_normalises_c_star():
    cal = calibration_from_dict(_doc())
    assert cal["b1"]["engine"].c_star == "internalizing"   # `int` -> full label
    assert cal["b1"]["delivery"].c_star == "warm"
    assert cal["b1"]["engine"].bracket_informative is True


@pytest.mark.parametrize("bad,match", [
    ({"L_design": 0.95, "U": 0.90}, "L_design < U"),          # L >= U (CB3)
    ({"U": 1.0}, "U must be < 1.0"),                          # CB4
    ({"L_design": 0.0}, "L_design must be > 0"),
    ({"d_floor": 0.0}, "d_floor must be > 0"),                # zero divisor
    ({"s_lo": 0.0}, "s_lo must be > 0"),                      # zero divisor
    ({"s_hi": 0.0}, "s_hi must be > 0"),                      # zero divisor
    ({"alpha": 0.0}, "alpha must be > 0"),
    ({"d_ceil": 1.0}, "d_ceil must be < 1.0"),                # zero divisor
    ({"c_star": "neutral"}, "UNMARKED class"),
])
def test_a2_bad_q_band_parameters_refuse_at_load(bad, match):
    engine = dict(_doc()["cells"]["b1"]["engine"])
    engine.update(bad)
    with pytest.raises(CalibrationError, match=match):
        calibration_from_dict(_doc(engine=engine))


@pytest.mark.parametrize("bad,match", [
    ({"d_lo": 0.0}, "d_lo must be > 0 strictly"),   # D2.3 — the inert exploit
    ({"d_lo": -0.1}, "d_lo must be > 0 strictly"),
    ({"d_lo": 0.2, "d_hi": 0.1}, "d_lo < d_hi"),
    ({"d_hi": 1.0}, "d_hi < 1.0"),
])
def test_a2_density_low_parameters_refuse_at_load(bad, match):
    engine = {"mode": "density_low", "d_lo": 0.05, "d_hi": 0.12,
              "s_lo": 0.04, "s_hi": 0.06}
    engine.update(bad)
    with pytest.raises(CalibrationError, match=match):
        calibration_from_dict(_doc(engine=engine))


def test_a2_density_low_is_engine_only():
    delivery = {"mode": "density_low", "d_lo": 0.05, "d_hi": 0.12,
                "s_lo": 0.04, "s_hi": 0.06}
    with pytest.raises(CalibrationError, match="neutral-ENGINE special case"):
        calibration_from_dict(_doc(delivery=delivery))


def test_a2_half_specified_cell_refuses():
    with pytest.raises(CalibrationError, match="missing axis entries"):
        calibration_from_dict({"cells": {"b1": {"engine": _doc()["cells"]["b1"]["engine"]}}})


def test_cell_lookup_accepts_descriptive_keys():
    doc = _doc()
    doc["cells"]["b1_internalizing_warm"] = doc["cells"].pop("b1")
    cal = calibration_from_dict(doc)
    assert cal["b1"]["engine"].c_star == "internalizing"
    assert "b1" in cal
    assert "b4" not in cal


# ── A3: anti-caricature, AT the configured T and d_floor ─────────────────────

def test_a3_caricature_scores_strictly_less_engine_at_T20():
    """Engine at d_eng = 0.30 -> n_marked = 6; §5.2's table."""
    p = q_band("internalizing", L=0.70, U=0.90, d_floor=0.30)
    n_marked = int(round(0.30 * T))                       # 6
    caricature = arc_labels(T, n_marked, n_marked, "internalizing", "externalizing", "neutral")
    realistic = arc_labels(T, n_marked, 5, "internalizing", "externalizing", "neutral")

    r_caric = br.axis_readout("engine", caricature, p)
    r_real = br.axis_readout("engine", realistic, p)

    assert r_caric.q == pytest.approx(6.5 / 7)            # 0.929, above U
    assert r_real.q == pytest.approx(5.5 / 7)             # 0.786, inside
    assert r_real.reward == 1.0
    assert r_caric.reward < r_real.reward, (
        "A3: the all-on-profile arc must score strictly less than one inside the band"
    )


def test_a3_caricature_scores_strictly_less_delivery_at_measured_density():
    """Delivery at the §0.2-suggested d_del ~ 0.7 -> n_marked = 14, ample resolution."""
    p = q_band("warm", L=0.62, U=0.88, d_floor=0.15)
    n_marked = 14
    caricature = arc_labels(T, n_marked, n_marked, "warm", "hot", "flat")
    realistic = arc_labels(T, n_marked, 11, "warm", "hot", "flat")   # q = 0.776

    r_caric = br.axis_readout("delivery", caricature, p)
    r_real = br.axis_readout("delivery", realistic, p)
    assert r_real.reward == 1.0
    assert r_caric.reward < r_real.reward


def test_a3_fails_at_low_resolution_which_is_why_it_is_measured():
    """§5.2's warning, pinned as a test.

    At `d_floor = 0.15` -> `n_marked = 3`, band [0.62, 0.88] contains BOTH 2-of-3
    (q = 0.625) and 3-of-3 (q = 0.875): the caricature ties the mixed arc and the
    anti-caricature mechanism is INOPERATIVE. This is not a bug in the band — it
    is why §5.2 gates delivery's `q`-band on measuring `d_del` rather than
    guessing. If this test ever starts passing-as-A3, the resolution problem was
    fixed and §5.2's resolution should be revisited.
    """
    p = q_band("warm", L=0.62, U=0.88, d_floor=0.15)
    caricature = arc_labels(T, 3, 3, "warm", "hot", "flat")
    mixed = arc_labels(T, 3, 2, "warm", "hot", "flat")
    r_caric = br.axis_readout("delivery", caricature, p)
    r_mixed = br.axis_readout("delivery", mixed, p)

    assert r_caric.q == pytest.approx(0.875) and r_mixed.q == pytest.approx(0.625)
    assert r_caric.reward == r_mixed.reward == 1.0        # the failure, documented


# ── A4: density floor ────────────────────────────────────────────────────────

def test_a4_thin_arc_scores_less_than_one_clearing_the_floor_at_the_same_q():
    """Both arcs sit ON the plateau, so the band factor is 1.0 for each and the
    ONLY difference is density — which is what A4 is about."""
    p = q_band("internalizing", L=0.70, U=0.90, d_floor=0.30)
    thin = arc_labels(T, 3, 3, "internalizing", "externalizing", "neutral")     # d=0.15, q=0.875
    thick = arc_labels(T, 6, 5, "internalizing", "externalizing", "neutral")    # d=0.30, q=0.786
    r_thin = br.axis_readout("engine", thin, p)
    r_thick = br.axis_readout("engine", thick, p)

    assert r_thin.band_value == r_thick.band_value == 1.0
    assert r_thin.density_factor == pytest.approx(0.15 / 0.30)
    assert r_thick.density_factor == 1.0
    assert r_thin.reward < r_thick.reward


def test_a4_fully_inert_arc_scores_zero():
    p = q_band("internalizing", d_floor=0.30)
    assert br.axis_readout("engine", ["neutral"] * T, p).reward == 0.0


def test_density_factor_two_sided():
    assert density_factor(0.30, 0.30) == 1.0
    assert density_factor(0.60, 0.30) == 1.0                       # one-sided: flat above
    assert density_factor(0.60, 0.30, d_ceil=0.45) == pytest.approx(0.40 / 0.55)
    assert density_factor(0.15, 0.30) == pytest.approx(0.5)


# ── D2.3: neutral engine = ABSENT engine, and inertness is NOT free ──────────

def test_d2_3_inert_arc_does_not_score_one_on_a_neutral_cell():
    p = density_low(d_lo=0.05, d_hi=0.12, s_lo=0.04)
    inert = br.axis_readout("engine", ["neutral"] * T, p)
    onband = br.axis_readout("engine",
                             arc_labels(T, 2, 1, "internalizing", "externalizing", "neutral"), p)
    assert inert.d == 0.0
    assert inert.reward < 1.0, "D2.3: d_lo > 0 exists so going limp is not full marks"
    assert onband.d == pytest.approx(0.10)
    assert onband.reward == 1.0


def test_d2_3_high_engine_expression_is_penalised_on_a_neutral_cell():
    p = density_low()
    loud = br.axis_readout(
        "engine", arc_labels(T, 12, 12, "internalizing", "externalizing", "neutral"), p)
    assert loud.reward < 0.01


# ── A5: cancellation wiring (CB1) ────────────────────────────────────────────

def test_a5_reward_refuses_a_grader_swap_after_calibration():
    cal = cal_for(q_band("internalizing"), q_band("warm"),
                  identities={"engine": "local:command-r7b", "delivery": "local:glm4:9b"})
    matching = _Backends(_ScriptedBackend([], "local:command-r7b"),
                         _ScriptedBackend([], "local:glm4:9b"))
    br.assert_calibration_backends(cal, matching)          # no raise

    swapped = _Backends(_ScriptedBackend([], "local:gemma3:12b"),
                        _ScriptedBackend([], "local:glm4:9b"))
    with pytest.raises(CalibrationError, match="CB1 BREACH on engine"):
        br.assert_calibration_backends(cal, swapped)


def test_a5_unstamped_artifact_refuses():
    cal = cal_for(q_band("internalizing"), q_band("warm"))
    with pytest.raises(CalibrationError, match="records no `backend_identities:`"):
        br.assert_calibration_backends(cal, _Backends(_ScriptedBackend([]), _ScriptedBackend([])))


def test_a5_a_biased_grader_shifts_target_and_rollout_together():
    """[BAND §7] to first order: a grader that over-calls `hot` inflates the
    AnnoMI-measured band edges AND the rollout `q` by the same amount, so the
    reward's argmax stays at the true behavioural rate.

    Simulated in label space: bias flips a fraction of true-warm turns to hot on
    BOTH the calibration corpus and the rollout. The band derived under bias,
    applied to rollouts scored under bias, still peaks at the same true rate.
    """
    def observed_q(true_q, bias):
        """Grader-space share-of-warm when a `bias` fraction of warms read hot."""
        return true_q * (1.0 - bias)

    true_target = 0.80
    for bias in (0.0, 0.15, 0.30):
        L = observed_q(true_target - 0.05, bias)
        U = observed_q(true_target + 0.05, bias)
        p = q_band("warm", L=L, U=U, d_floor=0.15)
        # Sweep true behavioural rates; score each through the same biased grader.
        scores = {}
        for true_q in (0.60, 0.70, 0.80, 0.90, 1.00):
            q_obs = observed_q(true_q, bias)
            scores[true_q] = band(q_obs, p.L_design, p.U, p.s_lo, p.s_hi)
        best = max(scores, key=scores.get)
        assert best == pytest.approx(true_target), (
            f"cancellation failed at bias={bias}: argmax at {best}, not {true_target}"
        )


# ── A6: smoothing damps quantisation jitter ──────────────────────────────────

def test_a6_q_gap_shrinks_as_alpha_rises():
    """The quantity alpha actually controls.

    `q = (n_on + a) / (n_marked + 2a)`, so a one-turn label flip moves `q` by
    `1 / (n_marked + 2a)` — strictly decreasing in alpha. This is the smoothing
    claim in its well-posed form.
    """
    n_marked = 6
    a = arc_labels(T, n_marked, 5, "internalizing", "externalizing", "neutral")
    b = arc_labels(T, n_marked, 4, "internalizing", "externalizing", "neutral")

    gaps = []
    for alpha in (0.1, 0.5, 2.0, 5.0):
        p = q_band("internalizing", alpha=alpha, d_floor=0.30)
        gaps.append(abs(br.axis_readout("engine", a, p).q
                        - br.axis_readout("engine", b, p).q))
    assert gaps == sorted(gaps, reverse=True), f"q-gap must shrink in alpha, got {gaps}"


def test_a6_reward_spread_shrinks_as_alpha_rises_WITH_MATCHED_EDGES():
    """A6, and the coupling it depends on.

    A6 as stated ("reward spread shrinks as alpha rises") is **only true if the
    band edges are derived under the SAME alpha as the reward uses**. `alpha` is
    part of the definition of `q`, not a free post-hoc knob: raising it pulls
    every `q` toward 0.5, so a band derived at `alpha = 0.5` and scored at
    `alpha = 5.0` is measuring a different quantity than it was calibrated on,
    and the arcs slide off the plateau onto the steep shoulder — reward spread
    can then *rise*. `test_a6_unmatched_alpha_breaks_the_property` pins that.

    Under the matched coupling (which is what §6.5 produces, since the AnnoMI
    per-session `q` is computed with the artifact's own alpha) the smoothing
    claim holds: the `q`-gap shrinks as `1/(n_marked + 2a)` while the Gaussian
    shoulder widths `s_lo`/`s_hi` are fixed absolute quantities, so the jitter
    is compressed against a fixed ruler.
    """
    n_marked = 6
    a = arc_labels(T, n_marked, 5, "internalizing", "externalizing", "neutral")
    b = arc_labels(T, n_marked, 4, "internalizing", "externalizing", "neutral")

    def edge(true_share, alpha):
        """The band edge re-derived under `alpha`, as §6.5's calibration would."""
        return (true_share * n_marked + alpha) / (n_marked + 2 * alpha)

    spreads = []
    for alpha in (0.1, 0.5, 2.0, 5.0):
        p = q_band("internalizing", L=edge(0.70, alpha), U=edge(0.90, alpha),
                   alpha=alpha, d_floor=0.30)
        ra = br.axis_readout("engine", a, p).reward
        rb = br.axis_readout("engine", b, p).reward
        spreads.append(abs(ra - rb))

    assert spreads == sorted(spreads, reverse=True), (
        f"A6: spread must be non-increasing in alpha, got {spreads}"
    )
    assert spreads[0] > 4 * spreads[-1]


def test_a6_unmatched_alpha_breaks_the_property():
    """The hazard the matched-edges test exists to rule out — recorded so a future
    reader does not "fix" A6 by re-deriving nothing.

    Band edges frozen at their alpha=0.5 values, scored at rising alpha: the
    spread is NOT monotone, because q slides down off the plateau.
    """
    n_marked = 6
    a = arc_labels(T, n_marked, 5, "internalizing", "externalizing", "neutral")
    b = arc_labels(T, n_marked, 4, "internalizing", "externalizing", "neutral")
    spreads = []
    for alpha in (0.1, 0.5, 2.0, 5.0):
        p = q_band("internalizing", L=0.70, U=0.90, alpha=alpha, d_floor=0.30)
        spreads.append(abs(br.axis_readout("engine", a, p).reward
                           - br.axis_readout("engine", b, p).reward))
    assert spreads != sorted(spreads, reverse=True)


# ── band / arc mechanics ─────────────────────────────────────────────────────

def test_band_plateau_and_shoulders():
    assert band(0.70, 0.70, 0.90, 0.06, 0.12) == 1.0
    assert band(0.80, 0.70, 0.90, 0.06, 0.12) == 1.0
    assert band(0.90, 0.70, 0.90, 0.06, 0.12) == 1.0
    # Asymmetric: the same distance below L is punished harder than above U.
    below = band(0.64, 0.70, 0.90, 0.06, 0.12)
    above = band(0.96, 0.70, 0.90, 0.06, 0.12)
    assert below < above
    assert below == pytest.approx(math.exp(-0.5))


def test_arc_reward_combines_the_axes_with_no_realism_multiplier():
    turns = [f"t{i}" for i in range(T)]
    eng = _ScriptedBackend(arc_labels(T, 6, 5, "internalizing", "externalizing", "neutral"))
    dlv = _ScriptedBackend(["flat"] * T)          # delivery fully inert -> 0.0
    cal = cal_for(q_band("internalizing", d_floor=0.30), q_band("warm", d_floor=0.15))

    out = band_reward_arc_readout(turns, "", "P", "b1", cal, _Backends(eng, dlv))
    assert out.engine.reward == 1.0
    assert out.delivery.reward == 0.0
    # D-BAND.1 geometric mean with the eps floor — a perfect engine does NOT buy
    # half marks while delivery is inert. Nothing else multiplies in (§4.3).
    assert out.reward == pytest.approx(math.sqrt(1.0 * br.AXIS_FLOOR_EPS))
    assert out.combination == br.GEOMETRIC_MEAN
    assert out.weaker_axis == "delivery"


# ── D-BAND.1: axis combination ───────────────────────────────────────────────

def test_average_cannot_tell_a_lopsided_arc_from_a_balanced_one():
    """The defect that motivated the change, pinned so it cannot come back
    unnoticed. Under the average all three of these are the same reward."""
    lopsided_a = br.combine_axes(0.95, 0.10, br.AVERAGE)
    lopsided_b = br.combine_axes(0.10, 0.95, br.AVERAGE)
    balanced = br.combine_axes(0.52, 0.52, br.AVERAGE)
    assert lopsided_a == lopsided_b == pytest.approx(0.525)
    assert balanced == pytest.approx(0.520)
    assert balanced < lopsided_a, "the average PREFERS specialisation — the defect"


def test_geometric_mean_penalises_imbalance():
    assert br.combine_axes(0.52, 0.52, br.GEOMETRIC_MEAN) > \
           br.combine_axes(0.95, 0.10, br.GEOMETRIC_MEAN)
    # ...and is symmetric, so neither axis is privileged.
    assert br.combine_axes(0.95, 0.10, br.GEOMETRIC_MEAN) == \
           pytest.approx(br.combine_axes(0.10, 0.95, br.GEOMETRIC_MEAN))


def test_geometric_mean_agrees_with_the_average_on_the_diagonal():
    """Why geo and not the raw product: it is the MINIMAL change. When the axes
    agree it reproduces the average exactly; it diverges only when lopsided."""
    for v in (0.2, 0.52, 0.8, 1.0):
        assert br.combine_axes(v, v, br.GEOMETRIC_MEAN) == \
               pytest.approx(br.combine_axes(v, v, br.AVERAGE))
    # The raw product does not have this property — it squashes the range.
    assert br.combine_axes(0.52, 0.52, br.PRODUCT) == pytest.approx(0.2704)


def test_geometric_mean_puts_the_gradient_on_the_weaker_axis():
    """d/da sqrt(ab) = 0.5*sqrt(b/a): the failing axis gets the larger derivative,
    automatically, with no separate balancing term."""
    def numeric_grad(f, a, b, h=1e-6):
        return ((f(a + h, b) - f(a - h, b)) / (2 * h),
                (f(a, b + h) - f(a, b - h)) / (2 * h))

    geo = lambda a, b: br.combine_axes(a, b, br.GEOMETRIC_MEAN)   # noqa: E731
    avg = lambda a, b: br.combine_axes(a, b, br.AVERAGE)          # noqa: E731

    g_eng, g_del = numeric_grad(geo, 0.95, 0.10)      # delivery is the weak axis
    assert g_del > 5 * g_eng

    a_eng, a_del = numeric_grad(avg, 0.95, 0.10)
    assert a_eng == pytest.approx(a_del, abs=1e-6)    # flat 1:1, always

    # Balanced: geo collapses to the average's behaviour.
    g_eng, g_del = numeric_grad(geo, 0.52, 0.52)
    assert g_eng == pytest.approx(g_del)
    assert g_eng == pytest.approx(0.5, abs=1e-4)


def test_geometric_mean_gives_more_within_group_spread_than_the_average():
    """R2, measured rather than assumed.

    [BAND D-BAND.1] justified the average as anti-collapse — "preserves gradient
    when one axis is out-of-band". That conflates reward LEVEL with gradient:
    GRPO uses within-group VARIANCE, and a near-constant healthy axis dilutes it.
    On a struggling cell the average is what causes the collapse it was chosen to
    prevent.
    """
    eng = [0.90] * 8
    dels = [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22]

    def mean_abs_adv(mode):
        rs = [br.combine_axes(a, b, mode) for a, b in zip(eng, dels)]
        m = sum(rs) / len(rs)
        return sum(abs(r - m) for r in rs) / len(rs)

    assert mean_abs_adv(br.GEOMETRIC_MEAN) > 2 * mean_abs_adv(br.AVERAGE)


def test_eps_floor_keeps_an_inert_group_alive():
    """The R2 guard the multiplicative branch needs.

    Without a floor, one dead axis zeroes every arc in the group: zero variance,
    no gradient, and the policy can never learn its way out of inertness.
    """
    eng = [0.90] * 8
    dead = [0.0] * 8
    rs = [br.combine_axes(a, b, br.GEOMETRIC_MEAN) for a, b in zip(eng, dead)]
    assert all(r > 0 for r in rs)

    # And marking even one turn in twenty is a large, visible step up.
    barely = br.combine_axes(0.90, 0.143, br.GEOMETRIC_MEAN)
    assert barely > 2 * rs[0]


def test_eps_floor_is_too_small_to_farm():
    """The floor must rescue a dead group without being worth aiming at."""
    inert = br.combine_axes(1.0, 0.0, br.GEOMETRIC_MEAN)
    honest = br.combine_axes(1.0, 1.0, br.GEOMETRIC_MEAN)
    assert inert < 0.15 and honest == 1.0


def test_average_needs_no_floor_and_gets_none():
    assert br.combine_axes(0.90, 0.0, br.AVERAGE) == pytest.approx(0.45)


def test_unknown_combination_refuses():
    with pytest.raises(ValueError, match="unknown axis combination"):
        br.combine_axes(0.5, 0.5, "harmonic_mean")


def test_geometric_mean_and_product_induce_the_same_ordering():
    """Geo is a monotone rescaling of the product, so they rank arcs identically
    and differ only in gradient magnitude. Worth pinning: it means the choice
    between them is about scale, never about what the optimum is."""
    pairs = [(0.9, 0.1), (0.5, 0.5), (0.8, 0.3), (1.0, 1.0), (0.2, 0.7)]
    geo = sorted(pairs, key=lambda p: br.combine_axes(*p, br.GEOMETRIC_MEAN))
    prod = sorted(pairs, key=lambda p: br.combine_axes(*p, br.PRODUCT))
    assert geo == prod


def test_combination_is_selectable_end_to_end():
    turns = [f"t{i}" for i in range(T)]
    cal = cal_for(q_band("internalizing", d_floor=0.30), q_band("warm", d_floor=0.15))
    eng_labels = arc_labels(T, 6, 5, "internalizing", "externalizing", "neutral")
    del_labels = arc_labels(T, 6, 2, "warm", "hot", "flat")

    def score(mode):
        return band_reward_arc(
            turns, "", "P", "b1", cal,
            _Backends(_ScriptedBackend(eng_labels), _ScriptedBackend(del_labels)),
            combination=mode)

    assert score(br.AVERAGE) > score(br.GEOMETRIC_MEAN) > score(br.PRODUCT)


def test_axis_reward_matches_readout():
    p = q_band("internalizing", d_floor=0.30)
    labels = arc_labels(T, 6, 5, "internalizing", "externalizing", "neutral")
    cal = cal_for(p, q_band("warm"))
    assert axis_reward("engine", labels, "b1", cal) == br.axis_readout("engine", labels, p).reward


def test_band_edge_farming_flag():
    """§9's watch: plateau reached on minimal marking. No automatic guard stands
    behind this (§4.3), so the flag is the only instrument."""
    p = q_band("internalizing", L=0.70, U=0.90, d_floor=0.30)
    farmed = arc_labels(T, 6, 5, "internalizing", "externalizing", "neutral")   # d = 0.30
    honest = arc_labels(T, 14, 11, "internalizing", "externalizing", "neutral")  # d = 0.70
    assert br.axis_readout("engine", farmed, p).at_lower_edge is True
    assert br.axis_readout("engine", honest, p).at_lower_edge is False
