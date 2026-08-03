"""Adapt `fidelity_reward` to the TRL / Unsloth GRPO reward-function contract.

TRL's `GRPOTrainer` calls `reward_func(prompts, completions, **columns) -> list[float]`,
where `**columns` are the extra dataset columns broadcast per-sample. Our dataset
rows carry `cell`, `P`, and `context` (see data/rollout.py -> the state), so the
reward can be reconstructed exactly per candidate turn without any drift-side input
(C1). This keeps the wall intact: TRL only ever sees the scalar we return.

The scalar comes from `fidelity_reward` itself rather than a re-derived formula —
one definition of the reward shape (C3), not two that can drift apart. The
per-dimension reads the §9 monitor needs are then pulled back out of the backends'
per-turn cache, so logging costs no extra grader calls.
"""

from __future__ import annotations

from typing import List, Optional

import grpo._bootstrap  # noqa: F401

from grpo.reward.fidelity_reward import RewardBackends, fidelity_reward
from grpo.monitor.online_audit import GroupRecord, OnlineMonitor


def _text(completion) -> str:
    """TRL completions may be raw strings or chat lists — normalize to text."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "")
    return str(completion)


def _subanswers(backends: RewardBackends, turn: str, context: str, cell: str) -> dict:
    """Per-axis sub-answers for one turn, when the backends are decomposed.

    Reads through each backend's per-turn cache, so this is free after scoring.
    Returns {} for constant stubs that have no decomposed read.
    """
    out = {}
    d = getattr(backends.delivery, "decompose", None)
    if d is not None:
        x = d(turn, context, cell)
        out["q1"] = x.q1_hostility_toward_listener
        out["q3"] = x.q3_closeness_toward_listener
    e = getattr(backends.engine, "decompose", None)
    if e is not None:
        y = e(turn, context, cell)
        out["e1"] = y.e1_blames_self
        out["e2"] = y.e2_blames_others
        out["dominant"] = y.dominant
    return out


def make_trl_reward(backends: RewardBackends, monitor: Optional[OnlineMonitor] = None):
    """Return a `reward_func(prompts, completions, **columns)` for GRPOTrainer.

    Requires `cell`, `P`, and `context` columns on the dataset (broadcast to
    per-sample lists by TRL). The step is read off the monitor, which the
    `make_monitor_callback` TrainerCallback keeps current — the reward function
    itself has no view of the global step.
    """

    def reward_func(prompts=None, completions=None, cell=None, P=None,
                    context=None, **kwargs) -> List[float]:
        completions = completions or []
        n = len(completions)
        cells = cell if isinstance(cell, list) else [cell] * n
        Ps = P if isinstance(P, list) else [P] * n
        contexts = context if isinstance(context, list) else [context] * n

        rewards: List[float] = []
        eng: List[int] = []
        dlv: List[int] = []
        subs: List[dict] = []
        texts: List[str] = []

        for i in range(n):
            turn = _text(completions[i])
            c, p, ctx = cells[i], Ps[i], contexts[i]
            # Single source of truth for the reward shape (C3).
            rewards.append(fidelity_reward(turn, p, ctx, c, backends))
            # Per-dimension reads for §9, served from the backends' cache.
            eng.append(backends.engine.score(turn, ctx, c))
            dlv.append(backends.delivery.score(turn, ctx, c))
            subs.append(_subanswers(backends, turn, ctx, c))
            texts.append(turn)

        if monitor is not None and n:
            step = getattr(monitor, "current_step", 0)
            monitor.record_group(GroupRecord(
                step=int(step), cell=str(cells[0]), rewards=rewards,
                completions=texts, engine_pass=eng, delivery_pass=dlv, sub=subs,
            ))
        return rewards

    return reward_func


def score_turn(backends: RewardBackends, patient_turn: str, P: str, context: str,
               cell: str) -> float:
    """Convenience single-turn scorer (used by warm-start filtering and cert)."""
    return fidelity_reward(patient_turn, P, context, cell, backends)
