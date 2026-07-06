"""Offline tests for the two-level fidelity grader (fidelity.py).

Pure-logic: no model, no server. Feeds synthetic per-turn observed labels through
classify_transcript (Level 1) and converge (Level 2) and checks the spec's
thresholds, veto separation, spread guard, and asymmetric delivery tag.

Run: python -m pytest sway_harness/tests/test_fidelity_grader.py
 or: python sway_harness/tests/test_fidelity_grader.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fidelity as F
from fidelity import classify_transcript, converge, target_poles

WARM_DEP = {"engine": "Dependency", "delivery": "Warm"}   # B1
HOT_ENT = {"engine": "Entitlement", "delivery": "Hot"}     # B4
NEUTRAL_WARM = {"engine": "Neutral", "delivery": "Warm"}   # B5


def _turn(**over):
    """One in-profile turn for a Warm/Dependency arc; override any field."""
    base = {
        "engine_direction": "internalizing",
        "delivery": "warm",
        "carriage": "carried",
        "forthcomingness": "voluble",
        "disclosure": "open",
        "comprehension": "follows",
        "expression": "articulate",
        "severity_band": "moderate",
        "item9_crisis": False,
        "in_character_break": False,
    }
    base.update(over)
    return base


def _arc(n=20, **over):
    return [_turn(**over) for _ in range(n)]


# ── Target poles ──────────────────────────────────────────────────────

def test_target_poles_read_engine_and_baseline():
    p = target_poles(WARM_DEP)
    assert p["engine_direction"] == "internalizing"
    assert p["delivery"] == "warm"
    # Roster baseline defaults when the profile doesn't pin the axis.
    assert p["forthcomingness"] == "voluble" and p["severity"] == "moderate"
    assert target_poles(HOT_ENT)["engine_direction"] == "externalizing"
    assert target_poles(NEUTRAL_WARM)["engine_direction"] == "neutral"


# ── Level 1: per-dimension thresholds ─────────────────────────────────

def test_clean_warm_dep_arc_passes_all():
    v = classify_transcript(WARM_DEP, _arc())
    assert not v.discarded
    assert all(r.passed for r in v.dims.values()), {k: r.detail for k, r in v.dims.items() if not r.passed}


def test_engine_direction_allows_one_wrong_turn_but_not_two():
    labels = _arc()
    labels[0]["engine_direction"] = "externalizing"
    assert classify_transcript(WARM_DEP, labels).dims["engine_direction"].passed
    labels[1]["engine_direction"] = "externalizing"
    assert not classify_transcript(WARM_DEP, labels).dims["engine_direction"].passed


def test_neutral_cell_fails_on_strong_lean():
    labels = [_turn(engine_direction="neutral") for _ in range(20)]
    assert classify_transcript(NEUTRAL_WARM, labels).dims["engine_direction"].passed
    labels[0]["engine_direction"] = "internalizing"
    labels[1]["engine_direction"] = "externalizing"
    assert not classify_transcript(NEUTRAL_WARM, labels).dims["engine_direction"].passed


def test_delivery_warm_wrong_direction_tag():
    labels = _arc(delivery="warm")
    labels[0]["delivery"] = "hot"
    labels[1]["delivery"] = "hot"  # 2 hot turns > WARM_MAX_HOT_TURNS (1)
    res = classify_transcript(WARM_DEP, labels).dims["delivery"]
    assert not res.passed and res.tag == F.TAG_WRONG_DIRECTION


def test_delivery_hot_under_expression_tag():
    # Hot profile that never flares -> under-expression, the opposite-signed error.
    labels = [_turn(engine_direction="externalizing", delivery="warm") for _ in range(20)]
    res = classify_transcript(HOT_ENT, labels).dims["delivery"]
    assert not res.passed and res.tag == F.TAG_UNDER_EXPRESSION
    # Enough hot turns -> passes, no tag.
    for i in range(F.HOT_MIN_HOT_TURNS):
        labels[i]["delivery"] = "hot"
    ok = classify_transcript(HOT_ENT, labels).dims["delivery"]
    assert ok.passed and ok.tag is None


def test_comprehension_and_expression_rate_gate():
    labels = _arc()
    # 3/20 = 85% follows < 90% -> fail comprehension.
    for i in range(3):
        labels[i]["comprehension"] = "loses_thread"
    v = classify_transcript(WARM_DEP, labels)
    assert not v.dims["comprehension"].passed
    # 2/20 = 90% articulate -> still passes expression.
    labels2 = _arc()
    labels2[0]["expression"] = "fragmented"
    labels2[1]["expression"] = "fragmented"
    assert classify_transcript(WARM_DEP, labels2).dims["expression"].passed


def test_forthcomingness_fails_when_dominantly_terse():
    labels = _arc(forthcomingness="terse")
    labels[0]["forthcomingness"] = "voluble"  # one voluble turn, but 95% terse
    assert not classify_transcript(WARM_DEP, labels).dims["forthcomingness"].passed


def test_disclosure_needs_one_substantial_open_turn():
    labels = _arc(disclosure="guarded")
    assert not classify_transcript(WARM_DEP, labels).dims["disclosure_depth"].passed
    labels[5]["disclosure"] = "open"
    assert classify_transcript(WARM_DEP, labels).dims["disclosure_depth"].passed


def test_severity_band_in_band_and_over_intense_ceiling():
    # 80% moderate, 20% severe: in-band OK (>=70%) but over-intense 20% > 10% ceiling.
    labels = _arc()
    for i in range(4):
        labels[i]["severity_band"] = "severe"
    assert not classify_transcript(WARM_DEP, labels).dims["severity_affect"].passed
    # 90% moderate, 10% severe -> passes both halves.
    labels2 = _arc()
    labels2[0]["severity_band"] = "severe"
    labels2[1]["severity_band"] = "mild"
    assert classify_transcript(WARM_DEP, labels2).dims["severity_affect"].passed


# ── Vetoes ────────────────────────────────────────────────────────────

def test_item9_crisis_discards_transcript_regardless_of_dims():
    labels = _arc()  # otherwise perfect
    labels[7]["item9_crisis"] = True
    v = classify_transcript(WARM_DEP, labels)
    assert v.discarded and v.veto["item9"]


def test_character_break_discards_transcript():
    labels = _arc()
    labels[3]["in_character_break"] = True
    v = classify_transcript(WARM_DEP, labels)
    assert v.discarded and v.veto["in_character"]


# ── Carriage: blocked until a schedule exists ─────────────────────────

def test_carriage_excluded_without_schedule():
    v = classify_transcript(WARM_DEP, _arc())
    assert not v.carriage_scored and "distortion_carriage" not in v.dims


def test_carriage_two_sided_when_scheduled():
    labels = _arc(n=4)
    schedule = ["carriage", "carriage", "na", "na"]
    labels[0]["carriage"] = "carried"
    labels[1]["carriage"] = "carried"
    labels[2]["carriage"] = "clean"
    labels[3]["carriage"] = "clean"
    v = classify_transcript(WARM_DEP, labels, schedule=schedule)
    assert v.carriage_scored and v.dims["distortion_carriage"].passed
    # Carpeting a distortion onto an N/A beat fails the clean half.
    labels[2]["carriage"] = "carried"
    labels[3]["carriage"] = "carried"
    v2 = classify_transcript(WARM_DEP, labels, schedule=schedule)
    assert not v2.dims["distortion_carriage"].passed


# ── Level 2: convergence ──────────────────────────────────────────────

def test_convergence_all_clean_arcs():
    verdicts = [classify_transcript(WARM_DEP, _arc()) for _ in range(10)]
    c = converge(verdicts)
    assert c.converged and c.spread == 1.0 and c.n_valid == 10
    assert "distortion_carriage" not in c.dim_pass_frac  # excluded, no schedule


def test_spread_guard_blocks_when_one_axis_fragile():
    # 8/10 arcs fail expression only; every other dim is perfect. Mean stays high,
    # but the single fragile axis (0.20) must sink convergence.
    good = [classify_transcript(WARM_DEP, _arc()) for _ in range(2)]
    bad = [classify_transcript(WARM_DEP, _arc(expression="fragmented")) for _ in range(8)]
    c = converge(good + bad)
    assert not c.converged
    assert c.dim_pass_frac["expression"] == 0.2 and c.spread == 0.2
    assert "expression" in c.failing_dims
    assert c.adherence > 0.85  # mean is misleadingly high — the guard is the point


def test_veto_gate_blocks_and_flags_leaky():
    # 2/10 arcs discarded (>10%) -> leaky + veto gate breached.
    clean = [classify_transcript(WARM_DEP, _arc()) for _ in range(8)]
    crisis_labels = _arc()
    crisis_labels[0]["item9_crisis"] = True
    breach = [classify_transcript(WARM_DEP, crisis_labels) for _ in range(2)]
    c = converge(clean + breach)
    assert not c.converged and c.leaky
    assert c.veto_breaches["item9"] == 2 and c.n_valid == 8


def test_delivery_tags_aggregate_in_convergence():
    labels = _arc()
    labels[0]["delivery"] = "hot"
    labels[1]["delivery"] = "hot"
    wrong = [classify_transcript(WARM_DEP, labels) for _ in range(3)]
    clean = [classify_transcript(WARM_DEP, _arc()) for _ in range(7)]
    c = converge(wrong + clean)
    assert c.delivery_tags[F.TAG_WRONG_DIRECTION] == 3


if __name__ == "__main__":
    fns = [g for n, g in sorted(globals().items()) if n.startswith("test_") and callable(g)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
