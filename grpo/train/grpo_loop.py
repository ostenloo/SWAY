"""GRPO training loop (grpo_spec §7).

Per step: sample G completions from the frozen policy snapshot (vLLM), score each
with `fidelity_reward` (temp-0 backend, C4), standardize rewards within the group
for advantages, and take a clipped policy-gradient step on the LoRA adapter with a
`beta * KL(policy || reference)` penalty. The reference is the SAME base with the
adapter disabled (PEFT `disable_adapter()`) — no second model copy (§3, §12).

This module wires TRL's `GRPOTrainer` (the standard path; Unsloth GRPO is the
single-GPU-QLoRA alternative). Torch / TRL / PEFT are imported lazily so the file
stays importable on a box without the training stack; the blocking gates and the
state-dataset construction run without a GPU.

**No pre-flight gates.** Both C6 gates have been removed by researcher decision:
§8.2's stratified delivery validation (it oversampled specific contested cases)
and §0.1's non-convergence diagnostic. `run_grpo` starts as soon as it is called.

Consequence, recorded rather than hidden: nothing checks before training that
GRPO is the right tool for this non-convergence (§0.1's policy-ceiling vs
miscalibrated-ruler fork), and nothing validates the delivery champion against
human labels. If the fidelity checker is the thing that is wrong, this loop will
optimize confidently toward a wrong target and the result will look like
progress. The §9 audit during training and §10 certification after are what
would surface either failure.
"""

from __future__ import annotations

from typing import List, Optional

import grpo._bootstrap  # noqa: F401
from client import frame_patient

from grpo.data.rollout import build_states
from grpo.data.curriculum import Stage, apply_stage, build_stages
from grpo.monitor.online_audit import OnlineMonitor, make_monitor_callback
from grpo.reward.trl_adapter import build_reward_func, make_trl_reward  # noqa: F401


def build_state_dataset(
    cfg: dict,
    P_by_cell: dict[str, str],
) -> List[dict]:
    """Roll the current policy into history prefixes and emit GRPO prompt rows.

    Each row carries the chat `prompt` (system=profile + the interlocutor's last
    move) plus the `cell`, `P`, and `context` columns the reward reconstructs from
    (trl_adapter.make_trl_reward). Cross-interlocutor spread comes from
    rollout.build_states (§5.3).
    """
    from grpo.config import build_interlocutors, build_policy_generate

    interlocutors = build_interlocutors(cfg)
    # Prefixes are rolled from the POLICY (the Simulator), not the reward annotator.
    policy_generate = build_policy_generate(cfg)
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


def release_rollout_models(cfg: dict) -> list:
    """Unload ollama-resident models before the training phase (§12).

    MEASURED, not precautionary: on the 32GB card, prefix-building leaves the
    policy and interlocutor resident at ~10GB each (ollama's OLLAMA_MAX_LOADED_MODELS
    is 2). The trainer then needs ~8.5GB for the nf4 base plus activations, finds
    ~10.9GB free, and transformers tries to dispatch layers to CPU — which
    bitsandbytes refuses outright ("Some modules are dispatched on the CPU or the
    disk").

    This is §12's phase-split mitigation made explicit: rollout generation and the
    training step do not co-reside, so the generation models are released between
    them. Ollama reloads them on the next call, which is the cost of the split.
    """
    import json as _json
    import urllib.request

    seen, released = set(), []
    targets = [(cfg["policy"]["model_path"], cfg["policy"]["base_url"])]
    targets += [(i["model_path"], i["base_url"]) for i in cfg.get("interlocutors", [])]
    r = cfg.get("reward", {})
    for key in ("backend_engine", "backend_delivery"):
        if r.get(key):
            targets.append((r[key], r.get("base_url", "")))

    for model, base_url in targets:
        if not model or not base_url or (model, base_url) in seen:
            continue
        seen.add((model, base_url))
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        try:
            req = urllib.request.Request(
                f"{root}/api/generate",
                data=_json.dumps({"model": model, "keep_alive": 0}).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=30).read()
            released.append(model)
        except Exception:
            pass          # not ollama, or already unloaded — nothing to release
    return released


def run_grpo(
    cfg: dict,
    P_by_cell: dict[str, str],
    adapter_in: Optional[str] = None,
    adapter_out: str = "results/grpo/adapters/grpo",
) -> str:
    """Run GRPO on top of the warm-started adapter. Returns the adapter path.

    `adapter_in` is the RFT warm-start adapter (§6). Curriculum stages (§5.5) run
    sequentially, each carrying the previous stage's adapter forward.
    """
    from grpo.config import build_reward_backends

    backends = build_reward_backends(cfg)

    monitor = OnlineMonitor(
        audit_every_n_steps=cfg["monitoring"]["audit_every_n_steps"],
        audit_sample_size=cfg["monitoring"]["audit_sample_size"],
        log_path=cfg["monitoring"].get("log_path"),
        subanswer_rates_enabled=cfg["monitoring"].get("subanswer_rates", True),
    )
    # The gradient signal is whatever `reward.shape` names, built in ONE place
    # (trl_adapter.build_reward_func) so the calibration load, the C3 grader
    # check and the §11 arc-length check cannot be skipped by a caller.
    reward_func = build_reward_func(cfg, backends, monitor)

    stages = build_stages(cfg)
    current_in = adapter_in
    for i, stage in enumerate(stages):
        stage_out = adapter_out if i == len(stages) - 1 else f"{adapter_out}.{stage.name}"
        rows = build_state_dataset(cfg, apply_stage(P_by_cell, stage))
        if not rows:
            raise RuntimeError(
                f"No GRPO states built for stage {stage.name!r} — check the policy endpoint."
            )
        # Free the rollout models before the trainer allocates (§12).
        release_rollout_models(cfg)
        _train(cfg, rows, reward_func, current_in, stage_out, stage, monitor)
        current_in = stage_out

    return adapter_out


def _train(cfg: dict, rows: List[dict], reward_func, adapter_in: Optional[str],
           adapter_out: str, stage: Stage, monitor: OnlineMonitor) -> None:
    """Lazy heavy-dependency section: construct and run TRL GRPOTrainer."""
    # 2.2GB of the measured OOM was reserved-but-unallocated fragmentation, which
    # this recovers. Must be set before torch initialises its allocator.
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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

    # ── D2.5: scale_rewards MUST be off ──────────────────────────────────────
    # TRL's DEFAULT is to divide each advantage by the group std. That default
    # partially erases the band's reward geometry (the band exists so q = 1.0
    # scores meaningfully less than q = 0.80, and a per-group rescale flattens
    # that) and amplifies quantisation noise into multi-sigma updates. Letting it
    # apply by omission is a silent spec violation, so the parameter is resolved
    # explicitly and a failure to set it is FATAL rather than ignored.
    #
    # The form is version-dependent: in newer TRL `scale_rewards` became a string.
    # If "batch" (batch-level std) is available, PREFER IT — scale-invariant across
    # runs without the per-group difficulty bias (Dr. GRPO, Liu et al. 2025).
    import inspect
    _sig = inspect.signature(GRPOConfig.__init__).parameters
    if "scale_rewards" not in _sig:
        raise RuntimeError(
            "D2.5: the installed TRL's GRPOConfig has no `scale_rewards` parameter, so "
            "advantage scaling cannot be disabled through config. Find its replacement "
            "before training — running with TRL's std-normalising default is a known "
            "spec violation, not a harmless fallback. "
            f"Available parameters: {sorted(_sig)}"
        )
    _want = g.get("scale_rewards", False)
    _default = _sig["scale_rewards"].default
    if isinstance(_default, str):
        # String-valued form: prefer batch-level std over a plain "none".
        _scale_rewards = "batch" if _want is False else str(_want)
        if _scale_rewards == "batch":
            print("[D2.5] TRL exposes string-valued scale_rewards; using 'batch' "
                  "(batch-level std) — preferred over plain off, per §7.")
    else:
        _scale_rewards = bool(_want)

    grpo_cfg = GRPOConfig(
        scale_rewards=_scale_rewards,
        output_dir=adapter_out,
        num_generations=g["group_size_G"],
        beta=g["kl_beta"],
        epsilon=g["clip_ratio"],
        learning_rate=g["lr"],
        # TRL 1.9 dropped `max_prompt_length` (no replacement, no truncation
        # option). The VRAM intent it served — bounding the rollout KV cache,
        # §12's binding constraint — is preserved by capping the vLLM sequence
        # length to prompt + completion instead.
        vllm_max_model_length=g["max_prompt_tokens"] + g["max_completion_tokens"],
        max_completion_length=g["max_completion_tokens"],
        temperature=g["temperature_rollout"],
        gradient_checkpointing=g.get("grad_checkpointing", True),
        max_steps=stage.max_steps,
        per_device_train_batch_size=g["group_size_G"],
        # §12 recommends vLLM-backed rollout generation for KV-cache efficiency,
        # but on this single shared card it loads a SECOND copy of the policy
        # alongside the trainer and the ollama-served graders, which does not fit.
        # Config-driven so it can be switched on when the card is not shared.
        use_vllm=g.get("use_vllm", False),
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
    # Without this the §9 audit/snapshot never fire — the monitor would accumulate
    # records nobody reads.
    trainer.add_callback(make_monitor_callback(monitor))
    trainer.train()
    trainer.save_model(adapter_out)
