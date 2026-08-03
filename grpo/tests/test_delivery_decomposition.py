"""Both axes decompose into their constituent questions, and the derived
three-way labels follow CODING_GUIDE.md. No network: stubs throughout.
"""

import grpo._bootstrap  # noqa: F401
from grpo.reward import delivery_decompose as dd
from grpo.reward import engine_decompose as ed
from grpo.reward.backends import (
    CleanDeliveryStubCore, DeliveryAdapter, FoldedDeliveryStubCore,
)
from grpo.gates.authored_pairs_smoketest import run_smoketest


def _d(q1: bool, q3: bool) -> dd.DeliveryDecomposition:
    return dd.decomposition_from_labels({
        "q1_hostility_toward_listener": q1, "q3_closeness_toward_listener": q3,
    })


def _e(e1: bool, e2: bool, dom: str = "") -> ed.EngineDecomposition:
    return ed.decomposition_from_labels({
        "e1_blames_self": e1, "e2_blames_others": e2, "dominant": dom,
    })


# ── delivery: hot / warm / flat ─────────────────────────────────────────────

def test_delivery_three_way_derivation():
    assert _d(True, False).delivery == "hot"
    assert _d(False, True).delivery == "warm"
    assert _d(False, False).delivery == "flat"


def test_hostility_dominates_a_mixed_delivery_read():
    assert _d(True, True).delivery == "hot"


def test_warm_is_positive_not_merely_not_hot():
    """The bug this decomposition exists to fix: a flat turn must NOT satisfy a
    warm target. b1/b3/b5 were passing delivery for free under `not hot`."""
    assert dd.delivery_pass_decomposed(_d(False, True), "b1") == 1    # warm target, warm turn
    assert dd.delivery_pass_decomposed(_d(False, False), "b1") == 0   # warm target, flat turn
    assert dd.delivery_pass_decomposed(_d(True, False), "b1") == 0    # warm target, hot turn


def test_hot_target_needs_q1():
    assert dd.delivery_pass_decomposed(_d(True, False), "b2") == 1
    assert dd.delivery_pass_decomposed(_d(False, True), "b2") == 0
    assert dd.delivery_pass_decomposed(_d(False, False), "b2") == 0


def test_delivery_missing_fields_read_as_flat():
    assert dd.decomposition_from_labels({}).delivery == "flat"


def test_delivery_string_booleans_tolerated():
    x = dd.decomposition_from_labels({
        "q1_hostility_toward_listener": "false",
        "q3_closeness_toward_listener": "yes",
    })
    assert x.delivery == "warm"


# ── engine: internalizing / externalizing / neutral ─────────────────────────

def test_engine_three_way_derivation():
    assert _e(True, False).engine_direction == "internalizing"
    assert _e(False, True).engine_direction == "externalizing"
    assert _e(False, False).engine_direction == "neutral"


def test_dominance_resolves_mixed_engine_turns():
    """CODING_GUIDE: 'label the dominant bend'. Two booleans cannot express it."""
    assert _e(True, True, "others").engine_direction == "externalizing"
    assert _e(True, True, "self").engine_direction == "internalizing"
    assert _e(True, True, "others").mixed is True


def test_missing_dominance_is_reconstructed_conservatively():
    """Unparseable dominance on a both-true turn falls back to neutral, the pole
    that asserts least about the turn."""
    assert _e(True, True, "garbage").engine_direction == "neutral"
    assert _e(True, False, "garbage").engine_direction == "internalizing"
    assert _e(False, True, "garbage").engine_direction == "externalizing"


def test_engine_neutral_target_rejects_either_lean():
    from grpo.reward.turn_fidelity import engine_pass_decomposed as ep
    assert ep(_e(False, False), "b5") == 1        # b5 targets neutral
    assert ep(_e(False, True), "b5") == 0
    assert ep(_e(True, False), "b5") == 0


def test_engine_directional_target_needs_exact_match():
    from grpo.reward.turn_fidelity import engine_pass_decomposed as ep
    assert ep(_e(True, False), "b1") == 1         # b1 targets internalizing
    assert ep(_e(False, True), "b1") == 0
    assert ep(_e(False, True), "b3") == 1         # b3 targets externalizing


# ── the smoke test still discriminates ──────────────────────────────────────

def test_folded_backend_shows_the_grievance_hot_hole():
    result = run_smoketest(DeliveryAdapter(FoldedDeliveryStubCore()))
    assert result.grievance_scored_hot > 0
    assert not result.clears


def test_clean_backend_clears_the_smoketest():
    result = run_smoketest(DeliveryAdapter(CleanDeliveryStubCore()))
    assert result.grievance_scored_hot == 0
    assert result.accuracy == 1.0
