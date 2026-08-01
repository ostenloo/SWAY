"""A6 — the C6-i pre-flight gate blocks until a human signs off. No network.

§8.2's stratified delivery gate was removed by researcher decision, so the §0.1
diagnostic is the only thing standing between a config and a training run.
"""

import json
from pathlib import Path

import pytest

import grpo._bootstrap  # noqa: F401
from grpo.gates import preflight_nonconvergence as pre


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
