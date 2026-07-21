"""GRPO training loop (grpo_spec §7).

Per step: sample G completions from the frozen policy snapshot (vLLM), score each
with `fidelity_reward` (temp-0 backend, C4), standardize rewards within the group
for advantages, and take a clipped policy-gradient step on the LoRA adapter with a
`beta * KL(policy || reference)` penalty. The reference is the SAME base with the
adapter disabled (PEFT `disable_adapter()`) — no second model copy (§3, §12).

This module wires TRL's `GRPOTrainer` (the standard path; Unsloth GRPO is the
single-GPU-QLoRA alternative). Torch / TRL / PEFT are imported lazily so the file
stays importable on a box without the training stack; the blocking §8 gate and the
state-dataset construction run without a GPU.

C6 is enforced here: `run_grpo` calls `assert_delivery_gate` before constructing
the trainer, so GRPO cannot start behind a folded delivery backend.
"""

from __future__ import annotations

from typing import List, Optional

import grpo._bootstrap  # noqa: F401
from client import frame_patient

from grpo.data.rollout import build_states, _default_generate
from grpo.gates.delivery_probe import assert_delivery_gate
from grpo.monitor.online_audit import OnlineMonitor
from grpo.reward.trl_adapter import make_trl_reward


def build_state_dataset(
    cfg: dict,
    P_by_cell: dict[str, str],
    backends_for_gate=None,
) -> List[dict]:
    """Roll the current policy into history prefixes and emit GRPO prompt rows.

    Each row carries the chat `prompt` (system=profile + the interlocutor's last
    move) plus the `cell`, `P`, and `context` columns the reward reconstructs from
    (trl_adapter.make_trl_reward). Cross-interlocutor spread comes from
    rollout.build_states (§5.3).
    """
    from grpo.config import build_interlocutors

    interlocutors = build_interlocutors(cfg)
    policy_generate = _default_generate(
        cfg["reward"]["local_model_path"], cfg["reward"]["local_base_url"],
        cfg["grpo"].get("temperature_rollout", 0.8), 256,
    )
    rows: List[dict] = []
    for cell in cfg["cells"]:
        P = P_by_cell[cell]
        states = build_states(
            P, cell, interlocutors, policy_generate,
            n_states=cfg["grpo"].get("states_per_step", 4),
            prefix_turns=cfg["grpo"].get("prefix_turns", 4),
        )
        system = frame_patient(P, "roleplay")
        for st in states:
            # Ensure the prompt ends on the interlocutor's move so the sampled
            # completion is the patient's next turn.
            history = st.transcript
            if history and history[-1]["role"] == "assistant":
                history = history + [{"role": "user", "content": "Go on."}]
            prompt = [{"role": "system", "content": system}] + [
                {"role": ("assistant" if m["role"] == "assistant" else "user"),
                 "content": m["content"]}
                for m in history
            ]
            rows.append({"prompt": prompt, "cell": cell, "P": P, "context": st.context()})
    return rows


def run_grpo(
    cfg: dict,
    P_by_cell: dict[str, str],
    adapter_in: Optional[str] = None,
    adapter_out: str = "results/grpo/adapters/grpo",
) -> str:
    """Run GRPO on top of the warm-started adapter. Returns the adapter path.

    `adapter_in` is the RFT warm-start adapter (grpo_spec §6). Torch/TRL/PEFT are
    imported lazily inside `_train`.
    """
    from grpo.config import build_reward_backends

    backends = build_reward_backends(cfg)

    # C6 — BLOCKING pre-flight gate. Must pass before any GRPO step.
    assert_delivery_gate(backends.delivery, bar=cfg["gate"]["delivery_kappa_bar"])

    monitor = OnlineMonitor(
        audit_every_n_steps=cfg["monitoring"]["audit_every_n_steps"],
        audit_sample_size=cfg["monitoring"]["audit_sample_size"],
        log_path=cfg["monitoring"].get("log_path"),
    )
    reward_func = make_trl_reward(backends, monitor)
    rows = build_state_dataset(cfg, P_by_cell)
    if not rows:
        raise RuntimeError("No GRPO states built — check the local policy endpoint.")

    _train(cfg, rows, reward_func, adapter_in, adapter_out)
    return adapter_out


def _train(cfg: dict, rows: List[dict], reward_func, adapter_in: Optional[str],
           adapter_out: str) -> None:
    """Lazy heavy-dependency section: construct and run TRL GRPOTrainer."""
    import torch  # noqa: F401
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    g = cfg["grpo"]
    lora = cfg["lora"]
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=cfg.get("quant", "nf4"),
        bnb_4bit_compute_dtype="bfloat16", bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=quant, device_map="auto",
    )
    # Reference = base with adapter disabled (§3/§12): PEFT handles this internally
    # when no ref_model is passed, so we never load a second base copy.
    if adapter_in:
        model = PeftModel.from_pretrained(model, adapter_in, is_trainable=True)
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"], task_type="CAUSAL_LM",
        )

    ds = Dataset.from_list(rows)
    grpo_cfg = GRPOConfig(
        output_dir=adapter_out,
        num_generations=g["group_size_G"],
        beta=g["kl_beta"],
        epsilon=g["clip_ratio"],
        learning_rate=g["lr"],
        max_prompt_length=g["max_prompt_tokens"],
        max_completion_length=g["max_completion_tokens"],
        temperature=g["temperature_rollout"],
        gradient_checkpointing=g.get("grad_checkpointing", True),
        max_steps=g.get("max_steps", 500),
        per_device_train_batch_size=g["group_size_G"],
        use_vllm=True,                      # vLLM-backed rollout generation (§12)
        seed=cfg["freeze"].get("seed", 42),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=grpo_cfg,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(adapter_out)
