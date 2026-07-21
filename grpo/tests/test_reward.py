"""Reward shape: partial credit + multiplicative realism floor (grpo_spec §4 / C3)."""

import grpo._bootstrap  # noqa: F401
from grpo.reward.fidelity_reward import RewardBackends, fidelity_reward


class _Const:
    def __init__(self, v):
        self.v = v

    def score(self, *a):
        return self.v

    def check(self, *a):
        return self.v


def _backends(engine, delivery, realism):
    return RewardBackends(engine=_Const(engine), delivery=_Const(delivery),
                          realism=_Const(realism))


def test_partial_credit_levels():
    # both pass -> 1.0 ; one passes -> 0.5 ; neither -> 0.0
    assert fidelity_reward("t", "P", "c", "b1", _backends(1, 1, 1)) == 1.0
    assert fidelity_reward("t", "P", "c", "b1", _backends(1, 0, 1)) == 0.5
    assert fidelity_reward("t", "P", "c", "b1", _backends(0, 1, 1)) == 0.5
    assert fidelity_reward("t", "P", "c", "b1", _backends(0, 0, 1)) == 0.0


def test_realism_floor_is_multiplicative():
    # realism failing zeroes the reward regardless of the diagnostics ...
    assert fidelity_reward("t", "P", "c", "b1", _backends(1, 1, 0)) == 0.0
    assert fidelity_reward("t", "P", "c", "b1", _backends(1, 0, 0)) == 0.0
    # ... and realism is never *added*: a perfect-realism, all-fail turn is still 0
    # (i.e. max reward with realism=1 but diagnostics=0 is 0.0, not >0).
    assert fidelity_reward("t", "P", "c", "b1", _backends(0, 0, 1)) == 0.0


def test_reward_in_unit_interval():
    for e in (0, 1):
        for d in (0, 1):
            for r in (0, 1):
                val = fidelity_reward("t", "P", "c", "b1", _backends(e, d, r))
                assert 0.0 <= val <= 1.0
