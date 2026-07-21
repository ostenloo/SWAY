"""Warm-start: reward-filtered SFT before GRPO (grpo_spec §6).

GRPO on an off-manifold target with a near-zero base rate has no signal to work
with. Warm-start first:

  1. Generate arcs with the current (prompt-conditioned) Simulator across cells
     and bare interlocutors — reuse the build-time generate-check loop (rollout).
  2. Filter to turns passing the diagnostic binaries + realism floor (the §4
     reward, thresholded to pass).
  3. LoRA-SFT the policy on that filtered set.
  4. Then run GRPO on top (train/grpo_loop.py).

This is the RL-sim SFT->RL recipe, but with *reward-filtered* SFT instead of
unfiltered imitation — unfiltered SFT imitates the model's own drift and
underperforms (the documented failure). RFT alone cannot reach off-manifold
targets; its job is only to raise the base rate so GRPO's groups are non-empty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import grpo._bootstrap  # noqa: F401
from client import frame_patient

from grpo.data.rollout import Interlocutor, build_states, _default_generate
from grpo.reward.fidelity_reward import RewardBackends, fidelity_reward


@dataclass
class RFTExample:
    cell: str
    P: str
    context: str
    completion: str          # the kept (on-profile) patient turn
    reward: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def collect_rft_dataset(
    P_by_cell: dict[str, str],
    cells: List[str],
    interlocutors: List[Interlocutor],
    policy_model_path: str,
    policy_base_url: str,
    backends: RewardBackends,
    arcs_per_cell: int = 30,
    prefix_turns: int = 4,
    pass_threshold: float = 1.0,
    temperature: float = 0.8,
    seed_base: int = 0,
) -> List[RFTExample]:
    """Roll the prompt-conditioned Simulator, score each patient turn with the §4
    reward, and keep the turns that clear `pass_threshold` (diagnostic pass AND
    realism pass => reward == 1.0 by default)."""
    policy_generate = _default_generate(policy_model_path, policy_base_url, temperature, 256)
    kept: List[RFTExample] = []

    for cell in cells:
        P = P_by_cell[cell]
        states = build_states(
            P, cell, interlocutors, policy_generate,
            n_states=arcs_per_cell, prefix_turns=prefix_turns, seed_base=seed_base,
        )
        for st in states:
            # Every patient turn in the rolled prefix is a candidate. Reconstruct
            # the context that preceded each and score it.
            transcript = st.transcript
            for i, msg in enumerate(transcript):
                if msg["role"] != "assistant":
                    continue
                ctx_msgs = transcript[:i]
                context = "\n".join(
                    f"[{'Patient' if m['role'] == 'assistant' else 'Model'}]: {m['content']}"
                    for m in ctx_msgs[-6:]
                )
                r = fidelity_reward(msg["content"], P, context, cell, backends)
                if r >= pass_threshold:
                    kept.append(RFTExample(cell, P, context, msg["content"], r))
    return kept


def to_sft_records(examples: List[RFTExample], framing: str = "roleplay") -> List[dict]:
    """Format kept turns as chat SFT records: system(profile) + context -> turn."""
    records = []
    for ex in examples:
        system = frame_patient(ex.P, framing)
        user = ex.context if ex.context else "Whenever you're ready, tell me what's on your mind."
        records.append({
            "cell": ex.cell,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": ex.completion},
            ],
        })
    return records


def save_dataset(records: List[dict], out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def run_rft(cfg: dict, P_by_cell: dict[str, str], adapter_out: str,
            dataset_out: Optional[str] = None) -> str:
    """Collect the reward-filtered set and LoRA-SFT the policy on it.

    Torch / TRL / PEFT / Unsloth are imported lazily so this module stays
    importable (and the data-collection path runnable against Ollama) on a box
    without the training stack. Returns the adapter output path.
    """
    from grpo.config import build_reward_backends, build_interlocutors

    backends = build_reward_backends(cfg)
    interlocutors = build_interlocutors(cfg)
    ws = cfg["warmstart"]

    examples = collect_rft_dataset(
        P_by_cell=P_by_cell,
        cells=cfg["cells"],
        interlocutors=interlocutors,
        policy_model_path=cfg["reward"]["local_model_path"],  # policy served locally
        policy_base_url=cfg["reward"]["local_base_url"],
        backends=backends,
        arcs_per_cell=ws.get("arcs_per_cell", 30),
        prefix_turns=cfg["grpo"].get("prefix_turns", 4),
    )
    records = to_sft_records(examples)
    if dataset_out:
        save_dataset(records, dataset_out)
    if not records:
        raise RuntimeError(
            "RFT collected zero passing turns — the base rate is too low even for "
            "warm-start. Relocate the target nearer the manifold / apply the "
            "curriculum (grpo_spec §5.5) before retrying."
        )

    _lora_sft(cfg, records, adapter_out)
    return adapter_out


def _lora_sft(cfg: dict, records: List[dict], adapter_out: str) -> None:
    """LoRA supervised fine-tune on the reward-filtered records.

    Lazy heavy-dependency section. Uses TRL's SFTTrainer over a QLoRA-loaded base
    (§11 lora block). Runs on the RTX 5090 host, not the Mac — hence the lazy
    imports and the explicit, minimal wiring.
    """
    import torch  # noqa: F401
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    lora = cfg["lora"]
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=cfg.get("quant", "nf4"),
        bnb_4bit_compute_dtype="bfloat16", bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=quant, device_map="auto",
    )
    peft_config = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], task_type="CAUSAL_LM",
    )
    ds = Dataset.from_list([{"messages": r["messages"]} for r in records])
    sft_cfg = SFTConfig(
        output_dir=adapter_out,
        num_train_epochs=cfg["warmstart"].get("rft_epochs", 1),
        per_device_train_batch_size=1,
        gradient_checkpointing=cfg["grpo"].get("grad_checkpointing", True),
        learning_rate=cfg["grpo"].get("lr", 1e-5),
        seed=cfg["freeze"].get("seed", 42),
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds,
                         peft_config=peft_config, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(adapter_out)
