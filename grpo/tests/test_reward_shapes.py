"""Reward-shape sweep: the shapes are pure, and collapse behaviour is measurable.

The claim these tests pin down is the one that decides the shape choice: a strict
AND (`product`) flattens far more groups than partial credit does, and the gap
widens as the cell gets harder. That is not an opinion about reward design — it
is arithmetic on the group-standardized advantage, and it is testable without a
GPU.
"""

import random

import grpo._bootstrap  # noqa: F401
from grpo.analysis.reward_shapes import (
    SHAPES, ScoredGroup, additive_half, completion_bonus, evaluate_shape,
    product, sweep,
)


# ── the shapes themselves ───────────────────────────────────────────────────

def test_all_shapes_stay_in_the_unit_interval():
    for name, fn in SHAPES.items():
        for e in (0, 1):
            for d in (0, 1):
                assert 0.0 <= fn(e, d) <= 1.0, name


def test_every_shape_is_zero_when_both_axes_fail():
    for name, fn in SHAPES.items():
        assert fn(0, 0) == 0.0, name


def test_additive_gives_partial_credit_and_product_does_not():
    assert additive_half(1, 0) == 0.5
    assert additive_half(0, 1) == 0.5
    assert product(1, 0) == 0.0
    assert product(0, 1) == 0.0
    assert product(1, 1) == 1.0


def test_completion_bonus_beats_the_sum_of_its_parts():
    """Both-right is worth more than two half-solutions, so single-axis farming
    is penalised — but a half-solution still breaks a tie inside a group."""
    half = completion_bonus(1, 0)
    both = completion_bonus(1, 1)
    assert both == 1.0
    assert 0.0 < half < 0.5
    assert both > 2 * half


# ── collapse behaviour, which is what actually decides the choice ───────────

def _synthetic_groups(p_engine: float, p_delivery: float, n_groups: int = 400,
                      group_size: int = 8, seed: int = 0):
    rng = random.Random(seed)
    groups = []
    for _ in range(n_groups):
        e = [int(rng.random() < p_engine) for _ in range(group_size)]
        d = [int(rng.random() < p_delivery) for _ in range(group_size)]
        groups.append(ScoredGroup(cell="bX", engine=e, delivery=d))
    return groups


def test_product_collapses_more_groups_than_partial_credit():
    groups = _synthetic_groups(0.5, 0.3)
    add = evaluate_shape(groups, "add", additive_half)
    prod = evaluate_shape(groups, "prod", product)
    assert prod.collapse_rate > add.collapse_rate
    # partial credit should keep nearly every group usable at these rates
    assert add.usable_group_rate > 0.9


def test_the_gap_widens_as_the_cell_gets_harder():
    """The off-manifold cells GRPO exists for are exactly where AND hurts most."""
    easy = _synthetic_groups(0.5, 0.5, seed=1)
    hard = _synthetic_groups(0.5, 0.1, seed=2)
    gap_easy = (evaluate_shape(easy, "p", product).collapse_rate
                - evaluate_shape(easy, "a", additive_half).collapse_rate)
    gap_hard = (evaluate_shape(hard, "p", product).collapse_rate
                - evaluate_shape(hard, "a", additive_half).collapse_rate)
    assert gap_hard > gap_easy


def test_an_all_fail_group_is_collapsed_under_every_shape():
    """No shape can rescue a group where nothing passes — this is why warm-start
    and the curriculum exist, rather than reward tuning."""
    dead = [ScoredGroup(cell="bX", engine=[0] * 8, delivery=[0] * 8)]
    for name, fn in SHAPES.items():
        assert evaluate_shape(dead, name, fn).collapse_rate == 1.0, name


def test_sweep_ranks_by_usable_group_rate_and_reports_base_rates():
    result = sweep(_synthetic_groups(0.5, 0.3), out_path=None)
    rates = [s["usable_group_rate"] for s in result["shapes"]]
    assert rates == sorted(rates, reverse=True)
    assert "bX" in result["base_rates"]
    assert 0.0 <= result["base_rates"]["bX"]["engine"] <= 1.0
