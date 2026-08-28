"""Acceptance A1-A7 for the rate-profile reward (RATE §9).

Every resolution-sensitive assertion runs at the CONFIGURED `T` against the
FROZEN bands, not in the abstract. That is the whole lesson of [RATE §1]: the
band spec's shape failures were all invisible in the abstract and obvious the
moment someone evaluated the real reward path at the real parameters.

**A5 is not what it says**, and the test says so rather than quietly passing a
weaker claim — see `test_a5_no_sparse_caricature_exploit`.
"""

import ast
import math
from pathlib import Path

import pytest

import grpo._bootstrap  # noqa: F401
from grpo.calibration import rate_derive as RD
from grpo.reward import rate_profile_reward as rp
from grpo.reward.rate_profile_reward import (
    MIN_ARC_TURNS, MIN_SPAN_TURNS, OFF_DIRECTION, ON_DIRECTION, SHOULDER,
    CalibrationError, axis_readout, band, calibration_from_dict, load_calibration,
    rate_of, rate_profile_reward_arc, rate_profile_reward_arc_readout,
)


T = 20                       # §11 arc_length_T
ARTIFACT = Path(__file__).resolve().parents[1] / "calibration" / "rate_calibration.v1.yaml"


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cal():
    return load_calibration(ARTIFACT)


def doc(cells=None, T_=T):
    """A minimal well-formed artifact document, for the A2 mutation tests."""
    return {
        "arc_length_T": T_,
        "grader_version": "test",
        "backend_identities": {"engine": "stub-e", "delivery": "stub-d"},
        "cells": cells or {
            "b1": {
                "engine": {
                    "s_lo": SHOULDER, "s_hi": SHOULDER,
                    "components": [
                        {"label": "internalizing", "role": ON_DIRECTION, "L": 0.144, "U": 0.290},
                        {"label": "externalizing", "role": OFF_DIRECTION, "L": 0.0, "U": 0.100},
                    ],
                },
                "delivery": {
                    "s_lo": SHOULDER, "s_hi": SHOULDER, "measured": False,
                    "components": [
                        {"label": "warm", "role": ON_DIRECTION, "L": 0.150, "U": 0.300},
                        {"label": "hot", "role": OFF_DIRECTION, "L": 0.0, "U": 0.100},
                    ],
                },
            }
        },
    }


def arc(n_on, on_label, unmarked, n_off=0, off_label=None, n_turns=T):
    """`n_on` on-direction turns, `n_off` off-direction, the rest unmarked."""
    assert n_on + n_off <= n_turns
    return ([on_label] * n_on + [off_label] * n_off
            + [unmarked] * (n_turns - n_on - n_off))


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
        self.engine, self.delivery = engine, delivery


def engine_score(cal, cell, labels):
    return axis_readout("engine", labels, cal[cell]["engine"]).reward


# ── A1: C1 — reads only (arc_turns, context, profile, cell, calibration) ─────

def test_a1_module_has_no_drift_side_imports():
    tree = ast.parse(Path(rp.__file__).read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            names += [a.name for a in node.names]
    names = [n.lower() for n in names]
    for token in rp.FORBIDDEN_IMPORT_TOKENS:
        assert not any(token in n for n in names), (
            f"C1 breach: rate_profile_reward.py imports something matching {token!r}: {names}"
        )


def test_a1_reward_reads_only_allowed_inputs(cal):
    turns = [f"t{i}" for i in range(T)]
    eng = _ScriptedBackend(["internalizing"] * T)
    dlv = _ScriptedBackend(["warm"] * T)

    rate_profile_reward_arc(turns, "CTX", "PROFILE PROMPT", "b1", cal, _Backends(eng, dlv))

    # Each grader saw exactly (turn, prefix-context, cell) — never the profile
    # prompt, never a MUT reply, never a drift signal. C4: blind to the pole.
    assert eng.seen[0] == ("t0", "CTX", "b1")
    assert eng.seen[1] == ("t1", "CTX\nt0", "b1")
    assert dlv.seen == eng.seen
    assert all("PROFILE PROMPT" not in ctx for _, ctx, _ in eng.seen)


# ── A2: load-time asserts fire on a bad artifact ─────────────────────────────

def test_a2_rejects_band_narrower_than_two_over_T():
    """§5.2 — the §1.5 failure becomes a load-time refusal."""
    d = doc()
    d["cells"]["b1"]["engine"]["components"][0].update(L=0.200, U=0.240)   # span 0.04
    with pytest.raises(CalibrationError, match="spans"):
        calibration_from_dict(d)


def test_a2_rejects_off_direction_band_with_a_lower_edge():
    """§5.3 — a nonzero off-direction floor is less realistic than the corpus."""
    d = doc()
    d["cells"]["b1"]["engine"]["components"][1].update(L=0.02, U=0.15)
    with pytest.raises(CalibrationError, match="off_direction band must have L = 0"):
        calibration_from_dict(d)


def test_a2_rejects_U_at_or_above_one():
    """A band reaching 1.0 puts the caricature on the plateau (A3 voided)."""
    d = doc()
    d["cells"]["b1"]["engine"]["components"][0].update(L=0.5, U=1.0)
    with pytest.raises(CalibrationError, match="U must be < 1.0"):
        calibration_from_dict(d)


def test_a2_rejects_asymmetric_shoulders():
    """§5.1 — the band spec's s_hi = 2*s_lo is the direct cause of §1.2."""
    d = doc()
    d["cells"]["b1"]["engine"]["s_hi"] = 2 * SHOULDER
    with pytest.raises(CalibrationError, match="[Aa]symmetric shoulders"):
        calibration_from_dict(d)


def test_a2_rejects_artifact_without_T():
    """§11 — the span checks are relative to T, so an artifact without it cannot
    be validated at all, only assumed."""
    d = doc()
    del d["arc_length_T"]
    with pytest.raises(CalibrationError, match="arc_length_T"):
        calibration_from_dict(d)


def test_a2_rejects_two_on_direction_components():
    d = doc()
    d["cells"]["b1"]["engine"]["components"][1]["role"] = ON_DIRECTION
    d["cells"]["b1"]["engine"]["components"][1].update(L=0.144, U=0.290)
    with pytest.raises(CalibrationError, match="on_direction components"):
        calibration_from_dict(d)


def test_a2_rejects_a_band_on_the_unmarked_class():
    """The unmarked rate is `1 - p_a - p_b` and is constrained implicitly (§2)."""
    d = doc()
    d["cells"]["b1"]["engine"]["components"][1]["label"] = "neutral"
    with pytest.raises(CalibrationError, match="not a MARKED label"):
        calibration_from_dict(d)


def test_a2_frozen_artifact_loads_and_records_its_T(cal):
    assert cal.arc_length_T == T
    assert cal.sha256 and len(cal.sha256) == 64
    assert cal.backend_identities.get("engine")


# ── A3: anti-caricature, at the configured T with the frozen bands ───────────

@pytest.mark.parametrize("cell,on,off,unmarked", [
    ("b1", "internalizing", "externalizing", "neutral"),
    ("b3", "externalizing", "internalizing", "neutral"),
])
def test_a3_caricature_scores_below_centre_and_below_under_expression(
        cal, cell, on, off, unmarked):
    """The test §1.2 would have failed.

    Under the conditional band, 5-of-5 marked on-direction scored 0.600 against
    4-of-5's 1.000 and 3-of-5's 0.035 — the caricature was 17x better than mild
    under-expression, so among arcs that miss the band (most of them, early in
    training) the caricature was the best available outcome.
    """
    p = cal[cell]["engine"]
    on_band = p.component(on)
    centre_k = round(0.5 * (on_band.L + on_band.U) * T)
    step_below_k = math.floor(on_band.L * T)          # one attainable step below L

    caricature = engine_score(cal, cell, [on] * T)
    centre = engine_score(cal, cell, arc(centre_k, on, unmarked))
    under = engine_score(cal, cell, arc(step_below_k, on, unmarked))

    assert centre == pytest.approx(1.0)
    assert caricature < centre, f"caricature {caricature} !< centre {centre}"
    assert caricature < under, (
        f"caricature {caricature} !< mild under-expression {under} — this is the "
        "§1.2 inversion, on the rate scale"
    )
    # And it is not a near-miss: on the rate scale the extreme is FAR from the
    # band (p = 1.0 against a band near 0.2), which is §2.1's whole claim.
    assert caricature < 0.01


def test_a3_anti_caricature_is_the_on_direction_upper_edge_and_nothing_else(cal):
    """§5.3 — asserted explicitly, as the spec requires.

    Remove the on-direction ceiling and the caricature scores full marks. Nothing
    else in the reward objects to it: the off-direction band is `[0, U]` and a
    fully internalizing arc has `p_ext = 0`, which scores 1.0 by design.
    """
    p = cal["b1"]["engine"]
    on_band = p.component("internalizing")
    off_band = p.component("externalizing")

    caricature_rates = {"internalizing": 1.0, "externalizing": 0.0}
    assert band(caricature_rates["externalizing"], off_band.L, off_band.U, p.s) == 1.0
    assert band(caricature_rates["internalizing"], on_band.L, on_band.U, p.s) < 0.01
    # With the ceiling lifted to 1.0 the caricature sits on the plateau.
    assert band(1.0, on_band.L, 1.0, p.s) == 1.0


# ── A4: no inversion anywhere ────────────────────────────────────────────────

@pytest.mark.parametrize("cell,on,unmarked", [
    ("b1", "internalizing", "neutral"),
    ("b3", "externalizing", "neutral"),
])
def test_a4_score_is_unimodal_over_all_attainable_rates(cal, cell, on, unmarked):
    """§1.2, §1.3 and §1.4 were all inversions this would have caught."""
    scores = [engine_score(cal, cell, arc(k, on, unmarked)) for k in range(T + 1)]
    peak = max(range(len(scores)), key=lambda i: scores[i])

    for i in range(1, peak + 1):
        assert scores[i] >= scores[i - 1] - 1e-12, (
            f"{cell}: inversion below the band at k={i}: {scores[i-1]} -> {scores[i]}")
    for i in range(peak + 1, len(scores)):
        assert scores[i] <= scores[i - 1] + 1e-12, (
            f"{cell}: inversion above the band at k={i}: {scores[i-1]} -> {scores[i]}")

    # The peak is inside the band, and §5.2's "three attainable values at full
    # score" actually holds at this T.
    on_band = cal[cell]["engine"].component(on)
    assert on_band.L <= peak / T <= on_band.U
    assert sum(1 for s in scores if s > 0.9) >= 3


def test_a4_holds_on_the_off_direction_rate_too(cal):
    """An off-direction band is one-sided by construction (L = 0), so unimodality
    there means monotone non-increasing — zero is the best score, as §5.3 says."""
    scores = [engine_score(cal, "b1", arc(4, "internalizing", "neutral",
                                          n_off=k, off_label="externalizing"))
              for k in range(0, T - 4 + 1)]
    assert scores[0] == pytest.approx(1.0)
    assert all(scores[i] <= scores[i - 1] + 1e-12 for i in range(1, len(scores)))


# ── A5: sparse caricature ────────────────────────────────────────────────────

def test_a5_no_sparse_caricature_exploit(cal):
    """A5 AS WRITTEN IS UNSATISFIABLE HERE, AND SHOULD BE. Recorded, not hidden.

    §9 A5 says: "For a fully one-directional arc, reward MUST NOT increase as the
    number of marked turns decreases." That criterion was written against the
    conditional-ratio parameterisation, where the marked COUNT and the on-pole
    RATIO were independent knobs — §1.4's attractor was moving one while holding
    the other, and 3-of-20 scored 2.5x better than 20-of-20 at the same ratio.

    Under rate profiles they are the SAME knob: for a fully one-directional arc
    the on-direction rate IS `n_marked / T`. So A5 traverses exactly the curve A4
    traverses, in the opposite direction, and asserting both would require the
    score to be simultaneously non-increasing and non-decreasing in `k` — i.e.
    flat, which is a reward with no gradient. A4 is the correct one: reward must
    RISE as a caricature arc sheds marked turns, because that is the arc walking
    back toward the target. That is not an exploit; it is the fix.

    What survives of A5's intent is that sparsity must not be a free win. That is
    what this asserts: the maximum lies inside the band, and dropping below the
    band costs strictly and immediately, so from anywhere at or below the target
    there is no gradient toward expressing the trait less.
    """
    scores = [engine_score(cal, "b1", arc(k, "internalizing", "neutral"))
              for k in range(T + 1)]
    on_band = cal["b1"]["engine"].component("internalizing")

    lo_k = math.ceil(on_band.L * T)
    assert scores[lo_k] == pytest.approx(1.0)
    # Strictly downhill all the way to the inert arc — no plateau to sit on, and
    # no k below the band that ties the band.
    for k in range(lo_k, 0, -1):
        assert scores[k - 1] < scores[k] - 1e-9, (
            f"sparsity is free at k={k-1}: {scores[k-1]} >= {scores[k]}")
    assert scores[0] < 0.10, "the fully inert arc must not be near-target"

    # And the global maximum is on the band, not at either extreme.
    best = max(range(len(scores)), key=lambda i: scores[i])
    assert 0 < best < T


def test_a5_fewer_marked_turns_never_beats_the_on_target_arc(cal):
    """The §1.4 comparison run on the new reward: no arc marking FEWER turns than
    the band allows outscores an arc sitting on it."""
    on_target = engine_score(cal, "b1", arc(4, "internalizing", "neutral"))
    for k in range(0, 3):
        assert engine_score(cal, "b1", arc(k, "internalizing", "neutral")) < on_target


# ── A6: same estimator on both sides (C7) ────────────────────────────────────

def test_a6_calibration_and_scoring_agree_on_one_fixture():
    """C7 — one estimator, or none.

    §1.1: rollouts were scored with Laplace smoothing and the targets derived
    with empirical-Bayes shrinkage. Five fully one-directional turns read 0.917
    to the scorer and 0.813 to the calibrator — a gap of 0.10, nearly twice the
    width of the band being placed.
    """
    labels = (["internalizing"] * 4 + ["externalizing"] * 2 + ["neutral"] * 14)
    assert len(labels) == T

    scoring_side = axis_readout("engine", labels, _profile_for(labels)).rates
    calibration_side = RD.conversation_rates("engine", {"s": labels})[0].rates

    assert scoring_side == calibration_side
    assert scoring_side["internalizing"] == pytest.approx(4 / T)
    # Raw. Nothing pulled toward a prior on either side.
    assert scoring_side["internalizing"] == rate_of(labels, "internalizing")


def _profile_for(labels):
    from grpo.reward.rate_profile_reward import AxisRateProfile, RateComponent
    return AxisRateProfile(
        axis="engine", s=SHOULDER,
        components=(RateComponent("internalizing", ON_DIRECTION, 0.144, 0.290),
                    RateComponent("externalizing", OFF_DIRECTION, 0.0, 0.100)),
    )


def test_a6_calibration_path_uses_the_reward_module_estimator():
    """Not a value check — a wiring check. `rate_derive` imports `rate_of` from
    the reward module rather than reimplementing it, which is what makes the
    agreement above structural instead of coincidental."""
    tree = ast.parse(Path(RD.__file__).read_text())
    imported = {
        a.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        and (node.module or "").endswith("rate_profile_reward") for a in node.names
    }
    assert "rate_of" in imported


# ── A7: cancellation wiring ──────────────────────────────────────────────────

def _bias(labels, every=4):
    """A deliberately biased grader: over-reads internalizing on every Nth
    otherwise-unmarked turn. Systematic, in one direction, on both sides."""
    out, seen = [], 0
    for lab in labels:
        if lab == "neutral":
            seen += 1
            out.append("internalizing" if seen % every == 0 else lab)
        else:
            out.append(lab)
    return out


def _corpus():
    """Synthetic internalizing-leaning conversations, 24 turns each."""
    corpus = {}
    for i in range(24):
        n_int = 4 + (i % 5)                     # 4..8 of 24
        corpus[f"s{i}"] = (["internalizing"] * n_int + ["externalizing"] * (i % 2)
                           + ["neutral"] * (24 - n_int - (i % 2)))
    return corpus


def _cal_from(corpus):
    rates = RD.conversation_rates("engine", corpus)
    edges = RD.engine_profile_edges("internalizing", rates, T=T)
    d = doc()
    d["cells"]["b1"]["engine"]["components"] = [e.to_component() for e in edges]
    return calibration_from_dict(d)


def test_a7_a_biased_grader_moves_target_and_rollout_together(cal):
    """C3's bias-cancellation argument, exercised end to end.

    A systematic grader bias inflates the target and the score alike and largely
    cancels. It requires consistency, not accuracy — which is the only thing
    licensing an imperfect grader in the loop at all.
    """
    candidates = {
        "under": arc(1, "internalizing", "neutral"),
        "on_target": arc(5, "internalizing", "neutral"),
        "over": arc(11, "internalizing", "neutral"),
        "caricature": ["internalizing"] * T,
    }

    clean_cal = _cal_from(_corpus())
    biased_cal = _cal_from({k: _bias(v) for k, v in _corpus().items()})

    clean = {k: engine_score(clean_cal, "b1", v) for k, v in candidates.items()}
    biased = {k: engine_score(biased_cal, "b1", _bias(v)) for k, v in candidates.items()}

    # The bias moved the instrument — this is not a no-op test.
    assert biased_cal["b1"]["engine"].component("internalizing").L > \
        clean_cal["b1"]["engine"].component("internalizing").L

    # ...and the argmax stayed put.
    assert max(clean, key=clean.get) == max(biased, key=biased.get) == "on_target"
    assert biased["caricature"] < biased["on_target"]


def test_a7_a_grader_swap_is_refused(cal):
    """C3 — cancellation does not survive scoring with a different instrument
    than you calibrated with, and the residual would be a wrong setpoint,
    silently. So it is a refusal, not a warning."""
    class _B:
        engine = _ScriptedBackend([], identity="local:some-other-model")
        delivery = _ScriptedBackend([], identity=cal.backend_identities["delivery"])

    with pytest.raises(CalibrationError, match="C3 BREACH"):
        rp.assert_calibration_backends(cal, _B())


# ── the decisions this implementation took, pinned ───────────────────────────

def test_min_within_axis_refuses_to_average_away_a_wrong_off_direction_rate(cal):
    """§4.4 — the two rates are two halves of one claim about the same behaviour.

    An arc that nails `p_int` while producing three times the realistic
    externalizing rate has not portrayed a dependency patient, and `min` says so
    where an average would score it 0.75.
    """
    good = engine_score(cal, "b1", arc(4, "internalizing", "neutral", n_off=1,
                                       off_label="externalizing"))
    bad = engine_score(cal, "b1", arc(4, "internalizing", "neutral", n_off=6,
                                      off_label="externalizing"))
    assert good == pytest.approx(1.0)
    assert bad < 0.1
    read = axis_readout("engine", arc(4, "internalizing", "neutral", n_off=6,
                                      off_label="externalizing"), cal["b1"]["engine"])
    assert read.binding == "externalizing"      # the gradient points at the right rate


def test_neutral_cell_inert_arc_is_not_rewarded(cal):
    """The hole `density_low` used to close, and where the guard went.

    §2.2 makes a neutral profile two low bands, and §5.3 forbids either of them a
    lower edge — so an arc expressing no engine at all scores engine = 1.0 on b5
    and b6, which is correct (neutral means ABSENT engine) but leaves nothing on
    that axis objecting to an inert simulator. The delivery axis carries it: every
    delivery profile has an on-direction band with a real lower edge, and the
    epsilon-floored geometric mean drags the arc down.
    """
    eng = _ScriptedBackend(["neutral"] * T)
    dlv = _ScriptedBackend(["flat"] * T)
    turns = [f"t{i}" for i in range(T)]
    out = rate_profile_reward_arc_readout(turns, "", "P", "b5", cal, _Backends(eng, dlv))

    assert out.engine.reward == pytest.approx(1.0)       # correct: engine is absent
    assert out.delivery.reward < 0.10                    # the inert arc is caught here
    assert out.reward < 0.35
    assert out.weaker_axis == "delivery"

    # A neutral cell scoring an arc that DOES carry its delivery is the contrast.
    dlv2 = _ScriptedBackend(["warm"] * 4 + ["flat"] * (T - 4))
    good = rate_profile_reward_arc_readout(
        turns, "", "P", "b5", cal, _Backends(_ScriptedBackend(["neutral"] * T), dlv2))
    assert good.reward == pytest.approx(1.0)
    assert good.reward > out.reward


def test_short_arcs_are_refused_not_scored_on_a_short_denominator(cal):
    """[RATE §12]'s open question, closed at MIN_ARC_TURNS.

    Below T = 10 the rate step `1/T` exceeds §5.2's `2/T` span floor — the §1.5
    resolution failure, reappearing on the rollout side.
    """
    n = MIN_ARC_TURNS - 1
    turns = [f"t{i}" for i in range(n)]
    eng = _ScriptedBackend(["internalizing"] * 2 + ["neutral"] * (n - 2))
    dlv = _ScriptedBackend(["warm"] * 2 + ["flat"] * (n - 2))
    out = rate_profile_reward_arc_readout(turns, "", "P", "b1", cal, _Backends(eng, dlv))

    assert out.reward == 0.0
    assert "MIN_ARC_TURNS" in out.refusal
    # The readouts survive so the monitor can see WHY it was refused.
    assert out.engine.n_turns == n


def test_denominator_is_the_arcs_own_turn_count_not_the_nominal_T(cal):
    """Dividing 16 graded turns by a configured 20 deflates every rate by 20% and
    walks the arc off the bottom of its band with nothing showing it."""
    labels = ["internalizing"] * 2 + ["neutral"] * 10          # 12 turns, p = 0.167
    read = axis_readout("engine", labels, cal["b1"]["engine"])
    p = cal["b1"]["engine"]

    assert read.rates["internalizing"] == pytest.approx(2 / 12)
    assert read.reward == pytest.approx(1.0)                   # on the band, correctly

    # The same arc scored on the NOMINAL denominator reads 2/20 = 0.10, which is
    # below the band's lower edge and scores 0.764. Same behaviour, two verdicts.
    on_band = p.component("internalizing")
    deflated = band(2 / T, on_band.L, on_band.U, p.s)
    assert deflated < 0.8 < read.reward


def test_spec_section_7_score_table_is_reproduced(cal):
    """§7's table, through the real reward path. If this drifts, the frozen bands
    or the band shape changed and the spec no longer describes the code."""
    expected = {0: 0.056, 2: 0.764, 3: 1.000, 4: 1.000, 5: 1.000,
                6: 0.986, 7: 0.607, 8: 0.186, 20: 0.000}
    for k, want in expected.items():
        got = engine_score(cal, "b1", arc(k, "internalizing", "neutral"))
        assert got == pytest.approx(want, abs=5e-4), f"k={k}: {got} != {want}"


def test_every_delivery_band_in_the_frozen_artifact_is_flagged_declared(cal):
    """§8 — the artifact MUST mark every delivery band `measured: false`, so a
    declared target can never be reported as a measured one."""
    for cell in cal.cells:
        assert cal[cell]["delivery"].measured is False
        assert cal[cell]["engine"].measured is True
        assert "declared_reason" in cal[cell]["delivery"].provenance


def test_every_frozen_band_satisfies_the_span_floor(cal):
    for cell in cal.cells:
        for axis in ("engine", "delivery"):
            for c in cal[cell][axis].components:
                assert c.span >= MIN_SPAN_TURNS / T - 1e-9, f"{cell}.{axis}.{c.label}"
                if c.role == OFF_DIRECTION:
                    assert c.L == 0.0


# ── wiring: what the config selects, and what it may not ─────────────────────

def _cfg(shape="rate_profile", T_=T):
    return {
        "grpo": {"arc_length_T": T_},
        "reward": {"shape": shape, "calibration_path": str(ARTIFACT)},
    }


class _LiveBackends:
    def __init__(self, cal):
        self.engine = _ScriptedBackend([], identity=cal.backend_identities["engine"])
        self.delivery = _ScriptedBackend([], identity=cal.backend_identities["delivery"])


def test_arc_band_shape_is_refused_by_the_builder(cal):
    """`band_calibration.v1.yaml` is defective on both engine directions and on
    delivery. A config key that can select a known-bad reward shape is exactly
    what the rest of this pipeline refuses to allow."""
    from grpo.reward.trl_adapter import build_reward_func
    with pytest.raises(ValueError, match="REFUSED"):
        build_reward_func(_cfg(shape="arc_band"), _LiveBackends(cal))


def test_arc_length_mismatch_between_config_and_artifact_is_a_load_error(cal):
    """§11 — the bands cleared the `2/T` span floor against the ARTIFACT's T.
    Roll a different T and the spans that were checked are not the spans in play,
    and nothing in the reward's output would show it."""
    from grpo.reward.trl_adapter import build_reward_func
    with pytest.raises(CalibrationError, match="arc_length_T mismatch"):
        build_reward_func(_cfg(T_=30), _LiveBackends(cal))


def test_builder_returns_a_working_rate_profile_reward(cal):
    from grpo.reward.trl_adapter import build_reward_func, ARC_TURN_SEP

    class _B:
        engine = _ScriptedBackend(["internalizing"] * 4 + ["neutral"] * (T - 4),
                                  identity=cal.backend_identities["engine"])
        delivery = _ScriptedBackend(["warm"] * 4 + ["flat"] * (T - 4),
                                    identity=cal.backend_identities["delivery"])

    fn = build_reward_func(_cfg(), _B())
    completion = ARC_TURN_SEP.join(f"t{i}" for i in range(T))
    out = fn(prompts=["p"], completions=[completion], cell="b1", P="PROFILE", context="")
    assert out == [pytest.approx(1.0)]


def test_monitor_reports_the_two_axes_separately_with_the_declared_flag(cal):
    """§8 — monitoring MUST report engine and delivery band-fit separately so a
    declared target is never reported as a measured one."""
    from grpo.monitor.online_audit import ArcRecord, GroupRecord, OnlineMonitor

    on_target = arc(4, "internalizing", "neutral")
    warm = arc(4, "warm", "flat")
    mon = OnlineMonitor(cal=cal)
    mon.record_group(GroupRecord(step=0, cell="b1", arcs=[
        ArcRecord(reward=1.0,
                  engine=axis_readout("engine", on_target, cal["b1"]["engine"]),
                  delivery=axis_readout("delivery", warm, cal["b1"]["delivery"]))
        for _ in range(4)
    ]))

    tel = mon.rate_telemetry(cell="b1")
    assert tel["measured_engine"] is True
    assert tel["measured_delivery"] is False
    assert tel["declared_axes"] == ["delivery"]
    assert tel["p_on_engine"]["mean"] == pytest.approx(0.20)
    assert tel["p_off_engine"]["mean"] == pytest.approx(0.0)


def test_monitor_flags_caricature_pressure_on_the_rate_scale(cal):
    from grpo.monitor.online_audit import ArcRecord, GroupRecord, OnlineMonitor

    mon = OnlineMonitor(cal=cal)
    mon.record_group(GroupRecord(step=0, cell="b1", arcs=[
        ArcRecord(reward=0.0,
                  engine=axis_readout("engine", ["internalizing"] * T, cal["b1"]["engine"]),
                  delivery=axis_readout("delivery", arc(4, "warm", "flat"),
                                        cal["b1"]["delivery"]))
        for _ in range(4)
    ]))
    flags = mon.rate_telemetry(cell="b1").get("flags", [])
    assert any("caricature pressure" in f for f in flags), flags
