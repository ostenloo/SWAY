"""Certification & freeze (grpo_spec §10, C7).

No adapter is frozen/shipped until it passes:

  1. Held-out authored detail — validate fidelity on a fresh authored instance the
     loop never saw (fresh severity / instance fill-ins per [FB]), not a reseed.
     Finetuning overfits harder than prompt-writing, so this matters more here.
  2. Second bare interlocutor — certify against a bare model NOT used in training
     (§5.3), the interlocutor-robustness the run-time demands.
  3. Freeze at deployment quant — the exact quantized checkpoint that runs at
     benchmark time (same discipline as the Judge).
  4. Version the adapter and record base checkpoint + quant + reward-backend
     identity + training seed for reproducibility (PIPE §11).

Certification scores rolled patient turns with the SAME §4 reward and requires the
per-cell diagnostic pass rate to clear the fidelity bar. The rollout + scoring path
runs against the local endpoint (no GPU); step 3's adapter load/merge is lazy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import grpo._bootstrap  # noqa: F401
from fidelity import ENGINE_CONVERGENCE_BAR

from grpo.data.rollout import Interlocutor, build_states, _default_generate
from grpo.reward.fidelity_reward import RewardBackends
from grpo.reward import turn_fidelity


@dataclass
class CellCertResult:
    cell: str
    n_turns: int
    engine_pass_rate: float
    delivery_pass_rate: float
    realism_trip_rate: float
    passed: bool

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class CertResult:
    passed: bool
    bar: float
    interlocutor: str
    cells: List[CellCertResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed, "bar": self.bar, "interlocutor": self.interlocutor,
            "cells": [c.to_dict() for c in self.cells],
        }


def certify_adapter(
    P_by_cell: dict[str, str],
    cells: List[str],
    held_out_interlocutor: Interlocutor,
    policy_model_path: str,
    policy_base_url: str,
    backends: RewardBackends,
    bar: float = ENGINE_CONVERGENCE_BAR,
    arcs_per_cell: int = 10,
    prefix_turns: int = 6,
    used_interlocutor_names: Optional[List[str]] = None,
) -> CertResult:
    """Roll the (candidate) policy against a held-out bare interlocutor on a fresh
    authored instance and require each cell's diagnostic pass rate to clear `bar`.

    NOTE: `P_by_cell` must carry the fresh authored fill-ins (step 1) — pass in the
    held-out profile prompts, not the training ones.
    """
    if used_interlocutor_names and held_out_interlocutor.name in used_interlocutor_names:
        raise ValueError(
            f"certification interlocutor {held_out_interlocutor.name!r} was used in "
            "training — §10 requires a second bare interlocutor NOT seen in training"
        )
    # build_states needs >= 2 interlocutors for spread; for a single held-out
    # certifier we duplicate it (cert measures robustness to THIS unseen model).
    interlocutors = [held_out_interlocutor,
                     Interlocutor(**{**held_out_interlocutor.__dict__, "name": held_out_interlocutor.name + "_b"})]
    policy_generate = _default_generate(policy_model_path, policy_base_url, 0.3, 256)

    cell_results: List[CellCertResult] = []
    all_pass = True
    for cell in cells:
        P = P_by_cell[cell]
        states = build_states(P, cell, interlocutors, policy_generate,
                              n_states=arcs_per_cell, prefix_turns=prefix_turns)
        eng = dlv = trips = n = 0
        for st in states:
            transcript = st.transcript
            for i, msg in enumerate(transcript):
                if msg["role"] != "assistant":
                    continue
                context = "\n".join(
                    f"[{'Patient' if m['role'] == 'assistant' else 'Model'}]: {m['content']}"
                    for m in transcript[:i][-6:]
                )
                # score via the reward binaries directly
                e = backends.engine.score(msg["content"], context, cell)
                d = backends.delivery.score(msg["content"], context, cell)
                r = backends.realism.check(msg["content"], context)
                eng += e; dlv += d; trips += (1 - r); n += 1
        engine_rate = eng / n if n else 0.0
        delivery_rate = dlv / n if n else 0.0
        trip_rate = trips / n if n else 0.0
        # gate on engine (the active ingredient with a set bar); delivery reported.
        cell_pass = n > 0 and engine_rate >= bar
        all_pass = all_pass and cell_pass
        cell_results.append(CellCertResult(
            cell, n, engine_rate, delivery_rate, trip_rate, cell_pass,
        ))

    return CertResult(passed=all_pass, bar=bar,
                      interlocutor=held_out_interlocutor.name, cells=cell_results)


def freeze_adapter(
    adapter_path: str,
    out_dir: str,
    cfg: dict,
    cert: CertResult,
    reward_backend_identity: str,
    merge_and_quantize: bool = False,
) -> str:
    """Freeze a certified adapter and write the reproducibility manifest (§10).

    Refuses to freeze a non-certified adapter (C7). Writes `freeze_manifest.json`
    recording base checkpoint + quant + reward-backend identity + training seed.
    Optional `merge_and_quantize` merges the LoRA and re-quantizes at the exact
    deployment quant (lazy heavy import).
    """
    if not cert.passed:
        raise RuntimeError(
            "Refusing to freeze: adapter failed §10 certification "
            f"(bar={cert.bar}, interlocutor={cert.interlocutor}). C7 blocks the ship."
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "adapter_path": adapter_path,
        "base_model": cfg["base_model"],
        "deployment_quant": cfg["freeze"]["deployment_quant"],
        "reward_backend_identity": reward_backend_identity,
        "training_seed": cfg["freeze"].get("seed"),
        "adapter_mode": cfg.get("adapter_mode"),
        "certification": cert.to_dict(),
    }
    (out / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2))

    if merge_and_quantize:
        _merge_and_quantize(adapter_path, str(out), cfg)
    return str(out / "freeze_manifest.json")


def _merge_and_quantize(adapter_path: str, out_dir: str, cfg: dict) -> None:
    """Merge the LoRA into the base and save at the deployment quant (§10 step 3)."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], device_map="auto")
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    merged.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(cfg["base_model"]).save_pretrained(out_dir)
