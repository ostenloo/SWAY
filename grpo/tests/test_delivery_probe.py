"""A2 — the §8 probe blocks training when a folded delivery backend is injected,
and passes a well-behaved (perfect) one. No network: uses stub cores.
"""

import pytest

import grpo._bootstrap  # noqa: F401
from grpo.reward.backends import FoldedStubCore, backends_from_core
from grpo.gates.delivery_probe import (
    run_delivery_probe, assert_delivery_gate, DeliveryGateError,
    cohen_kappa_binary, CONTRAST_PAIRS,
)


class _PerfectCore:
    """An oracle: reads the probe's own hand label straight through, so it always
    agrees with the human. Stands in for a delivery backend that clears §8."""
    def __init__(self):
        self._by_text = {p.text: p.human for p in CONTRAST_PAIRS}

    def labels(self, patient_turn, context, cell):
        human = self._by_text.get(patient_turn, "not_hot")
        return {"delivery": "hot" if human == "hot" else "flat"}


def test_folded_backend_fails_and_blocks():
    folded = backends_from_core(FoldedStubCore())
    result = run_delivery_probe(folded.delivery)
    # the fold scores employer grievance as hot -> disagreements on the not_hot side
    assert result.confusion["grievance_scored_hot"] > 0
    assert not result.passed
    assert result.kappa < 0.80

    with pytest.raises(DeliveryGateError):
        assert_delivery_gate(folded.delivery)


def test_perfect_backend_passes():
    perfect = backends_from_core(_PerfectCore())
    result = run_delivery_probe(perfect.delivery)
    assert result.passed
    assert result.kappa == 1.0
    # the gate returns (does not raise) on a passing backend
    assert_delivery_gate(perfect.delivery).passed


def test_kappa_bounds():
    assert cohen_kappa_binary(["hot", "not_hot"], ["hot", "not_hot"]) == 1.0
    # total disagreement on a balanced set -> negative kappa
    assert cohen_kappa_binary(["hot", "not_hot"], ["not_hot", "hot"]) < 0.0
