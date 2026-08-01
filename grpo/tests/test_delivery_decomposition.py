"""A2 — the §8.1 decomposition applies `hot = Q1` regardless of Q2, and the §8.3
smoke test separates a folded delivery backend from a clean one. No network:
stub cores throughout.
"""

import pytest

import grpo._bootstrap  # noqa: F401
from grpo.reward import delivery_decompose as dd
from grpo.reward.backends import (
    CleanDeliveryStubCore, DeliveryAdapter, FoldedDeliveryStubCore,
)
from grpo.gates.authored_pairs_smoketest import CONTRAST_PAIRS, run_smoketest


def _decomp(q1: bool, q2: bool) -> dd.DeliveryDecomposition:
    return dd.decomposition_from_labels({
        "q1_hostility_toward_listener": q1,
        "q2_grievance_toward_absent_party": q2,
    })


# ── A2: hot = Q1, regardless of Q2 ──────────────────────────────────────────

def test_yes_yes_is_hot():
    """Furious at the boss AND snapping at the therapist -> hot, because Q1 says so."""
    assert dd.hot_from_decomposition(_decomp(True, True)) is True
    assert _decomp(True, True).hot is True


def test_grievance_only_is_not_hot():
    """An employer rant fires Q2, leaves Q1 alone, scores not-hot."""
    d = _decomp(False, True)
    assert dd.hot_from_decomposition(d) is False
    assert d.grievance_only is True
    assert dd.observed_hot_label(d) == "not_hot"


def test_hostility_only_is_hot():
    assert dd.hot_from_decomposition(_decomp(True, False)) is True


def test_neither_is_not_hot():
    assert dd.hot_from_decomposition(_decomp(False, False)) is False


def test_q2_never_changes_the_label():
    """The whole point of the decomposition: Q2 is inert on the delivery label."""
    for q1 in (True, False):
        assert dd.hot_from_decomposition(_decomp(q1, True)) == \
               dd.hot_from_decomposition(_decomp(q1, False))


def test_missing_fields_default_to_not_hot():
    """A parse failure must not manufacture delivery reward on a warm/flat cell."""
    assert dd.decomposition_from_labels({}).hot is False
    assert dd.decomposition_from_labels({"garbage": 1}).hot is False


def test_string_booleans_are_tolerated():
    d = dd.decomposition_from_labels({
        "q1_hostility_toward_listener": "true",
        "q2_grievance_toward_absent_party": "no",
    })
    assert d.hot is True and d.q2_grievance_toward_absent_party is False


# ── cell-relative binary ────────────────────────────────────────────────────

def test_delivery_pass_is_cell_relative():
    """b2 targets hot; b1 targets warm. The same turn passes one and fails the other."""
    hot_turn, calm_turn = _decomp(True, False), _decomp(False, True)
    assert dd.delivery_pass_decomposed(hot_turn, "b2") == 1
    assert dd.delivery_pass_decomposed(calm_turn, "b2") == 0
    assert dd.delivery_pass_decomposed(hot_turn, "b1") == 0
    assert dd.delivery_pass_decomposed(calm_turn, "b1") == 1


# ── §8.3 smoke test discriminates folded from clean ─────────────────────────

def test_folded_backend_shows_the_grievance_hot_hole():
    folded = DeliveryAdapter(FoldedDeliveryStubCore())
    result = run_smoketest(folded)
    assert result.grievance_scored_hot > 0, "the fold must surface as grievance->hot"
    assert not result.clears
    assert result.accuracy < 1.0


def test_clean_backend_clears_the_smoketest():
    clean = DeliveryAdapter(CleanDeliveryStubCore())
    result = run_smoketest(clean)
    assert result.grievance_scored_hot == 0
    assert result.accuracy == 1.0


def test_yes_yes_pair_is_labelled_hot_by_a_clean_backend():
    """The authored yes/yes item is the decomposition's signature case."""
    clean = DeliveryAdapter(CleanDeliveryStubCore())
    yes_yes = [p for p in CONTRAST_PAIRS if "still hot" in p.note]
    assert yes_yes, "the authored set must retain a yes/yes item"
    for pair in yes_yes:
        d = clean.decompose(pair.text, pair.context, "b2")
        assert d.q2_grievance_toward_absent_party is True
        assert d.hot is True
