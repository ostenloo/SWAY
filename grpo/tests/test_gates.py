"""A2 / A6 — both C6 gates block, and the §8.2 gate catches a folded delivery
champion on a rollout-shaped stratum. No network, no GPU.
"""

import csv
import json
from pathlib import Path

import pytest

import grpo._bootstrap  # noqa: F401
from grpo.gates import delivery_stratified_validation as dsv
from grpo.gates import preflight_nonconvergence as pre
from grpo.reward.backends import (
    CleanDeliveryStubCore, DeliveryAdapter, FoldedDeliveryStubCore,
)


# ── fixtures: a rollout-shaped RFT set with a grievance-heavy tail ───────────

GRIEVANCE_TURNS = [
    "They laid off the whole team without warning and the company never explained why.",
    "Management kept the people who golfed with the boss and cut the rest of us.",
    "I gave that employer twelve years and they fired me in a two-line email.",
    "The restructuring was unfair — no severance, no notice, nothing.",
    "My whole department was eliminated and they called it a business decision.",
    "The layoff was handled so badly by management that I still can't believe it.",
]
HOSTILE_TURNS = [
    "That idea is useless and you're not listening to a word I say.",
    "Why am I even talking to you? You don't get it and you never will.",
    "Stop with your little techniques. You're worthless at this.",
]
NEUTRAL_TURNS = [
    "I've been sleeping badly and I can't focus on much of anything.",
    "Some days are easier than others. I made a list like we discussed.",
]


def _write_rft(path: Path) -> None:
    rows = []
    for t in GRIEVANCE_TURNS:
        rows.append({"cell": "b3", "P": "profile", "context": "", "completion": t, "reward": 1.0})
    for t in HOSTILE_TURNS:
        rows.append({"cell": "b2", "P": "profile", "context": "", "completion": t, "reward": 1.0})
    for t in NEUTRAL_TURNS:
        rows.append({"cell": "b1", "P": "profile", "context": "", "completion": t, "reward": 1.0})
    path.write_text("\n".join(json.dumps(r) for r in rows))


def _hand_label_truthfully(key_path: Path, labels_path: Path) -> None:
    """Stand in for the human: grievance turns are not-hot, hostile turns are hot."""
    with key_path.open(newline="") as f:
        key_rows = list(csv.DictReader(f))
    with labels_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "q1_hostility_toward_listener",
                    "q2_grievance_toward_absent_party"])
        for r in key_rows:
            turn = r["turn"]
            q1 = turn in HOSTILE_TURNS
            q2 = turn in GRIEVANCE_TURNS
            w.writerow([r["turn_id"], str(q1).lower(), str(q2).lower()])


def _run_stratum(tmp_path: Path, core, per_stratum: int = 40):
    rft = tmp_path / "rft.jsonl"
    _write_rft(rft)
    labels = tmp_path / "labels.csv"
    key = tmp_path / "key.csv"
    dsv.build_stratum(str(rft), DeliveryAdapter(core), str(labels), str(key),
                      per_stratum_cap=per_stratum, include_neither=10)
    _hand_label_truthfully(key, labels)
    return dsv.score_stratum(
        str(labels), str(key), delivery_backend_identity="stub",
        result_path=str(tmp_path / "result.json"),
    )


# ── §8.2 discriminates ──────────────────────────────────────────────────────

def test_folded_champion_fails_the_stratified_gate(tmp_path):
    result = _run_stratum(tmp_path, FoldedDeliveryStubCore())
    assert result.grievance_scored_hot > 0
    assert not result.passed


def test_clean_champion_clears_the_stratified_gate(tmp_path):
    result = _run_stratum(tmp_path, CleanDeliveryStubCore())
    assert result.grievance_scored_hot == 0
    assert result.contested.n > 0
    assert result.passed


def test_gate_reads_the_ci_lower_bound_not_the_point_estimate(tmp_path):
    result = _run_stratum(tmp_path, CleanDeliveryStubCore())
    assert result.contested.kappa_ci_low <= result.contested.kappa
    assert result.contested.passed == (result.contested.kappa_ci_low >= result.contested.bar)


def test_score_stratum_refuses_an_unlabelled_sheet(tmp_path):
    rft = tmp_path / "rft.jsonl"
    _write_rft(rft)
    labels, key = tmp_path / "labels.csv", tmp_path / "key.csv"
    dsv.build_stratum(str(rft), DeliveryAdapter(CleanDeliveryStubCore()),
                      str(labels), str(key))
    with pytest.raises(RuntimeError, match="HUMAN labels"):
        dsv.score_stratum(str(labels), str(key), result_path=None)


# ── C6-ii assertion behaviour ───────────────────────────────────────────────

def test_stratified_assertion_blocks_when_no_result(tmp_path):
    with pytest.raises(dsv.StratifiedGateError, match="C6-ii NOT SATISFIED"):
        dsv.assert_stratified_gate("local:glm4:9b", str(tmp_path / "missing.json"))


def test_stratified_assertion_blocks_a_stale_backend(tmp_path):
    result = _run_stratum(tmp_path, CleanDeliveryStubCore())
    assert result.passed
    with pytest.raises(dsv.StratifiedGateError, match="STALE"):
        dsv.assert_stratified_gate("local:some-other-model",
                                   str(tmp_path / "result.json"))


def test_stratified_assertion_blocks_a_failing_result(tmp_path):
    _run_stratum(tmp_path, FoldedDeliveryStubCore())
    with pytest.raises(dsv.StratifiedGateError, match="C6-ii FAILED"):
        dsv.assert_stratified_gate("stub", str(tmp_path / "result.json"))


def test_stratified_assertion_passes_a_good_result(tmp_path):
    _run_stratum(tmp_path, CleanDeliveryStubCore())
    data = dsv.assert_stratified_gate("stub", str(tmp_path / "result.json"))
    assert data["passed"] is True


def test_missing_rft_set_explains_the_sequencing(tmp_path):
    """§8.2 cannot run before warm-start — the error must say so."""
    with pytest.raises(FileNotFoundError, match="warmstart"):
        dsv.load_rft_turns(str(tmp_path / "nope.jsonl"))


# ── C6-i assertion behaviour ────────────────────────────────────────────────

def _write_preflight(path: Path, verdict: str, signed_by: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "verdict": verdict, "signed_off_by": signed_by,
        "human_agrees_off_rate": 0.9, "bar": 0.8,
        "disputed_engine": 0, "disputed_delivery": 0,
    }))


def test_preflight_blocks_when_missing(tmp_path):
    with pytest.raises(pre.PreflightGateError, match="C6-i NOT SATISFIED"):
        pre.assert_preflight_signed_off(str(tmp_path / "missing.json"))


def test_preflight_blocks_when_unsigned(tmp_path):
    p = tmp_path / "pre.json"
    _write_preflight(p, pre.VERDICT_POLICY, "")
    with pytest.raises(pre.PreflightGateError, match="NOT SIGNED OFF"):
        pre.assert_preflight_signed_off(str(p))


def test_preflight_blocks_on_a_ruler_verdict(tmp_path):
    p = tmp_path / "pre.json"
    _write_preflight(p, pre.VERDICT_RULER, "researcher")
    with pytest.raises(pre.PreflightGateError, match="RULER problem"):
        pre.assert_preflight_signed_off(str(p))


def test_preflight_blocks_on_a_mixed_verdict(tmp_path):
    p = tmp_path / "pre.json"
    _write_preflight(p, pre.VERDICT_MIXED, "researcher")
    with pytest.raises(pre.PreflightGateError, match="C6-i FAILED"):
        pre.assert_preflight_signed_off(str(p))


def test_preflight_passes_when_signed_off_as_policy_ceiling(tmp_path):
    p = tmp_path / "pre.json"
    _write_preflight(p, pre.VERDICT_POLICY, "researcher")
    assert pre.assert_preflight_signed_off(str(p))["verdict"] == pre.VERDICT_POLICY
