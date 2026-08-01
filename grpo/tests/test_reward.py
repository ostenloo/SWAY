"""Reward shape: partial credit over the two diagnostic binaries (grpo_spec §4 / C3).

§4's multiplicative realism floor has been removed by researcher decision, so the
reward is engine and delivery alone.
"""

import grpo._bootstrap  # noqa: F401
from grpo.reward.fidelity_reward import RewardBackends, fidelity_reward


class _Const:
    def __init__(self, v):
        self.v = v

    def score(self, *a):
        return self.v

    def check(self, *a):
        return self.v


def _backends(engine, delivery):
    return RewardBackends(engine=_Const(engine), delivery=_Const(delivery))


def test_partial_credit_levels():
    # both pass -> 1.0 ; one passes -> 0.5 ; neither -> 0.0
    assert fidelity_reward("t", "P", "c", "b1", _backends(1, 1)) == 1.0
    assert fidelity_reward("t", "P", "c", "b1", _backends(1, 0)) == 0.5
    assert fidelity_reward("t", "P", "c", "b1", _backends(0, 1)) == 0.5
    assert fidelity_reward("t", "P", "c", "b1", _backends(0, 0)) == 0.0


def test_reward_has_no_realism_term():
    """The RewardBackends contract carries exactly two graders."""
    import dataclasses
    assert {f.name for f in dataclasses.fields(RewardBackends)} == {"engine", "delivery"}


def test_reward_in_unit_interval():
    for e in (0, 1):
        for d in (0, 1):
            val = fidelity_reward("t", "P", "c", "b1", _backends(e, d))
            assert 0.0 <= val <= 1.0
