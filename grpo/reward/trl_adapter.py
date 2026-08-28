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


# ── arc-level (grpo_spec_2 §4) ───────────────────────────────────────────────
#
# The v2 reward scores a whole ARC, not a turn. Everything above stays: the
# per-turn binary survives in the RFT warm-start filter (§6) and in `score_turn`
# for certification, but it is no longer the gradient signal (§4.2).
#
# NOTE ON SCOPE. This is the reward-side contract only, and it is deliberately
# agnostic about how an arc gets generated. Producing 20 policy turns interleaved
# with a LIVE interlocutor (D2.2) inside a TRL training step is a separate,
# larger problem — stock `GRPOTrainer` emits one continuous completion per prompt
# and masks nothing, so the interlocutor's turns would land in the policy's
# log-probs. Whatever solves that (a custom rollout hook, a trainer subclass with
# an explicit completion mask, or another framework) feeds arcs into THIS
# function unchanged.

#: How a generated arc is serialised into a single completion string when the
#: generator cannot hand back a list. Chosen to be something no patient turn
#: emits naturally.
ARC_TURN_SEP = "\n<|patient_turn|>\n"


def split_arc(completion) -> List[str]:
    """Normalise a completion into the arc's ordered patient turns.

    Accepts a list of turn strings, a list of chat dicts, or a single string
    delimited by `ARC_TURN_SEP`. Empty turns are dropped — a blank turn would
    otherwise inflate the density DENOMINATOR and read as the policy declining to
    express the axis, which is exactly the bias §4.2's missing-key counter exists
    to catch elsewhere.
    """
    if isinstance(completion, str):
        parts = completion.split(ARC_TURN_SEP)
    elif isinstance(completion, list):
        parts = [
            (c.get("content", "") if isinstance(c, dict) else str(c))
            for c in completion
        ]
    else:
        parts = [str(completion)]
    return [p.strip() for p in parts if p and p.strip()]


def _arc_subanswers(backends: RewardBackends, turns: List[str], context0: str,
                    cell: str) -> dict:
    """Arc-level sub-answer RATES, averaged over the arc's turns.

    Served from the backends' per-turn cache, so this costs no extra grader calls
    after scoring. The §9 grievance watch and the cancellation-drift signal both
    read `e2` from here.
    """
    from grpo.reward.band_reward import context_upto

    acc: dict = {}
    n = 0
    for i, turn in enumerate(turns):
        ctx = context_upto(turns, i, context0)
        per = _subanswers(backends, turn, ctx, cell)
        if not per:
            continue
        n += 1
        for k, v in per.items():
            if k == "dominant":
                acc.setdefault("dominant", {})
                acc["dominant"][v] = acc["dominant"].get(v, 0) + 1
            else:
                acc[k] = acc.get(k, 0.0) + float(bool(v))
    if not n:
        return {}
    out = {k: (v / n) for k, v in acc.items() if k != "dominant"}
    if "dominant" in acc:
        out["dominant"] = {k: v / n for k, v in acc["dominant"].items()}
    return out


def make_trl_band_reward(backends, cal, monitor: Optional[OnlineMonitor] = None,
                         collect_subanswers: bool = True):
    """Return an arc-level `reward_func(prompts, completions, **columns)` (§4).

    Each completion is one ARC. Requires `cell`, `P` and `context` columns.
    `cal` is the frozen [BAND §6.4] calibration (C9); CB1 is asserted once here
    rather than per call, so a grader swap cannot slip in mid-run.
    """
    from grpo.monitor.online_audit import ArcRecord
    from grpo.reward.band_reward import (
        assert_calibration_backends, band_reward_arc_readout,
    )
    from grpo.reward import turn_fidelity

    # C4/CB1 — the graders scoring rollouts must BE the graders that measured the
    # bracket, or [BAND §7]'s cancellation does not hold.
    assert_calibration_backends(cal, backends)

    def reward_func(prompts=None, completions=None, cell=None, P=None,
                    context=None, **kwargs) -> List[float]:
        completions = completions or []
        n = len(completions)
        cells = cell if isinstance(cell, list) else [cell] * n
        Ps = P if isinstance(P, list) else [P] * n
        contexts = context if isinstance(context, list) else [context] * n

        rewards: List[float] = []
        arcs = []

        for i in range(n):
            turns = split_arc(completions[i])
            c, p, ctx = cells[i], Ps[i], contexts[i]
            if not turns:
                # A degenerate empty arc. Score 0 rather than raising: one bad
                # generation must not kill the step, and the group still carries
                # gradient away from whatever produced it.
                rewards.append(0.0)
                arcs.append(ArcRecord(reward=0.0, completion=""))
                continue

            out = band_reward_arc_readout(turns, ctx, p, c, cal, backends)
            rewards.append(out.reward)

            sub = _arc_subanswers(backends, turns, ctx, c) if collect_subanswers else {}
            # Derived per-turn binaries, as ARC RATES — for the audit only (§4.2).
            eng_pass = mean_or_none([
                turn_fidelity.engine_pass({"engine_direction": lab}, c)
                for lab in out.engine_labels])
            del_pass = mean_or_none([
                int(lab == turn_fidelity.poles_for_cell(c)["delivery"])
                for lab in out.delivery_labels])

            arcs.append(ArcRecord(
                reward=out.reward, engine=out.engine, delivery=out.delivery,
                completion=ARC_TURN_SEP.join(turns), engine_pass=eng_pass,
                delivery_pass=del_pass, sub=sub,
            ))

        if monitor is not None and n:
            from grpo.monitor.online_audit import GroupRecord
            monitor.record_group(GroupRecord(
                step=int(getattr(monitor, "current_step", 0)),
                cell=str(cells[0]), arcs=arcs,
            ))
        return rewards

    return reward_func


def make_trl_rate_profile_reward(backends, cal, monitor: Optional[OnlineMonitor] = None,
                                 collect_subanswers: bool = True):
    """Return an arc-level `reward_func(prompts, completions, **columns)` for the
    RATE-PROFILE reward ([RATE §4]).

    SUPERSEDES `make_trl_band_reward`, which is retained unwired. Each completion
    is one ARC; requires the `cell`, `P` and `context` columns. C3 is asserted
    once here rather than per call, so a grader swap cannot slip in mid-run.
    """
    from grpo.monitor.online_audit import ArcRecord
    from grpo.reward.rate_profile_reward import (
        assert_calibration_backends, rate_profile_reward_arc_readout,
    )
    from grpo.reward import turn_fidelity

    # C3 — the graders scoring rollouts must BE the graders that measured the
    # targets, or the bias-cancellation argument does not hold.
    assert_calibration_backends(cal, backends)

    def reward_func(prompts=None, completions=None, cell=None, P=None,
                    context=None, **kwargs) -> List[float]:
        completions = completions or []
        n = len(completions)
        cells = cell if isinstance(cell, list) else [cell] * n
        Ps = P if isinstance(P, list) else [P] * n
        contexts = context if isinstance(context, list) else [context] * n

        rewards: List[float] = []
        arcs = []

        for i in range(n):
            turns = split_arc(completions[i])
            c, p, ctx = cells[i], Ps[i], contexts[i]
            if not turns:
                # A degenerate empty arc. Score 0 rather than raising: one bad
                # generation must not kill the step, and the group still carries
                # gradient away from whatever produced it. A SHORT arc is scored
                # 0 for the same reason, inside the reward itself — see
                # rate_profile_reward.MIN_ARC_TURNS.
                rewards.append(0.0)
                arcs.append(ArcRecord(reward=0.0, completion=""))
                continue

            out = rate_profile_reward_arc_readout(turns, ctx, p, c, cal, backends)
            rewards.append(out.reward)

            sub = _arc_subanswers(backends, turns, ctx, c) if collect_subanswers else {}
            # Derived per-turn binaries, as ARC RATES — for the audit only. These
            # are NOT the reward and must never be reported as it: the per-turn
            # monotone binary is the shape [RATE §1] exists to replace.
            eng_pass = mean_or_none([
                turn_fidelity.engine_pass({"engine_direction": lab}, c)
                for lab in out.engine_labels])
            del_pass = mean_or_none([
                int(lab == turn_fidelity.poles_for_cell(c)["delivery"])
                for lab in out.delivery_labels])

            arcs.append(ArcRecord(
                reward=out.reward, engine=out.engine, delivery=out.delivery,
                completion=ARC_TURN_SEP.join(turns), engine_pass=eng_pass,
                delivery_pass=del_pass, sub=sub,
            ))

        if monitor is not None and n:
            from grpo.monitor.online_audit import GroupRecord
            monitor.record_group(GroupRecord(
                step=int(getattr(monitor, "current_step", 0)),
                cell=str(cells[0]), arcs=arcs,
            ))
        return rewards

    return reward_func


#: `reward.shape` values this adapter can build. `per_turn_binary` survives only
#: inside the RFT warm-start filter and the audit; `arc_band` is the defective
#: conditional-ratio shape and is refused outright rather than left selectable.
REWARD_SHAPES = ("rate_profile", "per_turn_binary")


def build_reward_func(cfg, backends, monitor: Optional[OnlineMonitor] = None):
    """The reward `reward.shape` names, with its calibration loaded and checked.

    One place decides what the gradient signal is. `arc_band` is REFUSED here
    rather than silently honoured: `band_calibration.v1.yaml` is defective on
    both engine directions and on delivery, and a config key that can select a
    known-bad reward shape is exactly what the rest of this pipeline refuses to
    allow.
    """
    from grpo.reward.rate_profile_reward import (
        assert_arc_length, load_calibration,
    )

    r = cfg.get("reward", {})
    shape = str(r.get("shape", "rate_profile"))

    if shape == "arc_band":
        raise ValueError(
            "reward.shape = 'arc_band' is REFUSED. The conditional-ratio band is "
            "superseded ([RATE §12]): its caricature scored 17x better than mild "
            "under-expression, sparse expression scored 2.5x better than full, and "
            "no band-level repair fixed either. Use 'rate_profile'."
        )
    if shape == "per_turn_binary":
        # The pre-band shape. Kept buildable for the A8 baseline comparison and
        # for nothing else — it is the monotone reward whose optimum IS the
        # caricature, which is the whole reason [RATE] exists.
        return make_trl_reward(backends, monitor)
    if shape != "rate_profile":
        raise ValueError(f"unknown reward.shape {shape!r} (expected one of {REWARD_SHAPES})")

    cal_path = r.get("calibration_path")
    if not cal_path:
        raise ValueError(
            "reward.shape = 'rate_profile' needs `reward.calibration_path` — the "
            "artifact is frozen and hash-logged before step 1 (C6/C9)."
        )
    cal = load_calibration(cal_path)
    # §11 — the bands were validated against the artifact's T, not the config's.
    assert_arc_length(cal, cfg["grpo"]["arc_length_T"])
    return make_trl_rate_profile_reward(backends, cal, monitor)


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None
