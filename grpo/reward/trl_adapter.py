"""Adapt `fidelity_reward` to the TRL / Unsloth GRPO reward-function contract.

TRL's `GRPOTrainer` calls `reward_func(prompts, completions, **columns) -> list[float]`,
where `**columns` are the extra dataset columns broadcast per-sample. Our dataset
rows carry `cell`, `P`, and `context` (see data/rollout.py → the state), so the
reward can be reconstructed exactly per candidate turn without any drift-side input
(C1). This keeps the wall intact: TRL only ever sees the scalar we return.

The factory optionally feeds an OnlineMonitor (grpo_spec §9) with per-dimension
pass flags so the high-advantage audit can see which axis is being farmed.
"""

from __future__ import annotations

from typing import List, Optional

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


def make_trl_reward(backends: RewardBackends, monitor: Optional[OnlineMonitor] = None):
    """Return a `reward_func(prompts, completions, **columns)` for GRPOTrainer.

    Requires `cell`, `P`, and `context` columns on the dataset (broadcast to
    per-sample lists by TRL). `_step`/`_group_cell` are optional bookkeeping
    columns for the monitor.
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
        rea: List[int] = []
        texts: List[str] = []
        for i in range(n):
            turn = _text(completions[i])
            c, p, ctx = cells[i], Ps[i], contexts[i]
            e = backends.engine.score(turn, ctx, c)
            d = backends.delivery.score(turn, ctx, c)
            r = backends.realism.check(turn, ctx)
            rewards.append((0.5 * e + 0.5 * d) * r)
            eng.append(e); dlv.append(d); rea.append(r); texts.append(turn)

        if monitor is not None and n:
            step = kwargs.get("_step", [0] * n)
            step = step[0] if isinstance(step, list) else step
            monitor.record_group(GroupRecord(
                step=int(step), cell=str(cells[0]), rewards=rewards,
                completions=texts, engine_pass=eng, delivery_pass=dlv, realism_ok=rea,
            ))
        return rewards

    return reward_func


def score_turn(backends: RewardBackends, patient_turn: str, P: str, context: str,
               cell: str) -> float:
    """Convenience single-turn scorer (used by warm-start filtering and cert)."""
    return fidelity_reward(patient_turn, P, context, cell, backends)
