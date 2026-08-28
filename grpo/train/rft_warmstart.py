"""Warm-start: reward-filtered SFT before GRPO (grpo_spec §6).

GRPO on an off-manifold target with a near-zero base rate has no signal to work
with. Warm-start first:

  1. Generate arcs with the current (prompt-conditioned) Simulator across cells
     and bare interlocutors — reuse the build-time generate-check loop (rollout).
  2. Filter to turns passing both diagnostic binaries (the §4 reward,
     thresholded to pass).
  3. LoRA-SFT the policy on that filtered set.
  4. Then run GRPO on top (train/grpo_loop.py).

This is the RL-sim SFT->RL recipe, but with *reward-filtered* SFT instead of
unfiltered imitation — unfiltered SFT imitates the model's own drift and
underperforms (the documented failure). RFT alone cannot reach off-manifold
targets; its job is only to raise the base rate so GRPO's groups are non-empty.

`run_rft` also emits the kept turns un-chat-formatted (`*.rft.jsonl`) alongside
the SFT records — useful for inspecting what warm-start actually trained on.
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
    #: §6.2 — the per-axis tuple behind the scalar. Persisting only the summed
    #: `reward` (which is always 1.0 for a kept turn under the conjunctive
    #: filter) threw away the one number that diagnoses a zero-coverage cell.
    engine_pass: Optional[int] = None
    delivery_pass: Optional[int] = None
    kept: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()


#: §6.1 — the filter used to decide which rolled turns become SFT examples.
#:
#: `conjunctive` (the [FT1] behaviour) requires BOTH axes to pass on the SAME
#: turn. Per §0.2 that alone explains b5's and b6's zero coverage: both axes are
#: individually reachable, they simply never co-occur — so the filter is
#: systematically biased against exactly the Hot x internalizing cells where the
#: axes are anti-correlated. **Zero RFT coverage is a conjunction failure, not
#: proof of an unreachable target.**
#:
#: `disjunctive` keeps turns passing EITHER axis, weighted by how many passed.
#: Warm-start is a COVERAGE tool feeding a distributional objective; now that the
#: downstream reward no longer requires joint per-turn satisfaction (§4), there
#: is no longer a principled reason for the filter to be stricter than the
#: objective it feeds.
CONJUNCTIVE = "conjunctive"
DISJUNCTIVE = "disjunctive"


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
    rft_filter: str = CONJUNCTIVE,
    rejects_out: Optional[str] = None,
) -> List[RFTExample]:
    """Roll the prompt-conditioned Simulator, score each patient turn, and keep
    the turns that clear the filter.

    Warm-start's job is unchanged (§6): raise the marked BASE RATE so GRPO groups
    are non-empty. The filter stays **per-turn** even though the reward is now
    arc-level, because RFT reinforces what the base already samples and cannot
    reach an off-manifold *distribution* — distributional shaping is GRPO's job
    ([BAND §9]).

    `rft_filter` selects `conjunctive` (both axes on the same turn) or
    `disjunctive` (either axis, partial credit) — see §6.1 before reusing the
    default. `rejects_out`, when set, persists REJECTED turns with their per-axis
    tuple (§6.2) so a zero-coverage cell can be diagnosed directly instead of
    inferred from the biased high-advantage sample.
    """
    policy_generate = _default_generate(policy_model_path, policy_base_url, temperature, 256)
    kept: List[RFTExample] = []
    rejects: List[RFTExample] = []

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
                turn = msg["content"]
                # §6.2 — read the two axes SEPARATELY, so the tuple survives.
                # Both come from the backends' per-turn cache, so scoring the
                # axes individually costs no extra grader calls.
                e = backends.engine.score(turn, context, cell)
                d = backends.delivery.score(turn, context, cell)
                r = 0.5 * e + 0.5 * d

                if rft_filter == DISJUNCTIVE:
                    keep = r > 0.0
                else:
                    keep = r >= pass_threshold

                ex = RFTExample(cell, P, context, turn, r,
                                engine_pass=e, delivery_pass=d, kept=keep)
                (kept if keep else rejects).append(ex)

    if rejects_out:
        _save_rejects(rejects, rejects_out)
    return kept


def _save_rejects(rejects: List[RFTExample], out_path: str) -> None:
    """Persist rejected turns with their `(engine_pass, delivery_pass)` (§6.2).

    The diagnosis this enables, per §6.2: a cell whose rejects are overwhelmingly
    `(1, 0)` is **delivery-zero** and needs near-manifold relaxation on the
    delivery target; a cell whose rejects are `(0, 1)` is **engine-zero** and can
    have its pass threshold lowered. Without the tuple both look identical — an
    empty kept-set — and the distinction has to be recovered from the
    high-advantage audit, which is a biased sample.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for ex in rejects:
            f.write(json.dumps(ex.to_dict()) + "\n")


def coverage_report(kept: List[RFTExample], rejects: List[RFTExample]) -> dict:
    """Per-cell `(engine_pass, delivery_pass)` census — the §6.2 diagnosis.

    §6.3: `d_floor` reads as unmet on any cell RFT left uncovered, so **fix
    coverage before reading density**. This is the report that tells you which.
    """
    cells = sorted({ex.cell for ex in list(kept) + list(rejects)})
    out = {}
    for cell in cells:
        k = [e for e in kept if e.cell == cell]
        r = [e for e in rejects if e.cell == cell]
        tuples: dict = {}
        for e in k + r:
            key = f"({e.engine_pass},{e.delivery_pass})"
            tuples[key] = tuples.get(key, 0) + 1
        n = len(k) + len(r)
        entry = {
            "n_turns": n,
            "n_kept": len(k),
            "coverage": round(len(k) / n, 4) if n else 0.0,
            "tuples": tuples,
            "engine_pass_rate": round(
                sum(e.engine_pass or 0 for e in k + r) / n, 4) if n else 0.0,
            "delivery_pass_rate": round(
                sum(e.delivery_pass or 0 for e in k + r) / n, 4) if n else 0.0,
        }
        if not k and n:
            # The §6.1 reading, stated rather than left to be rediscovered.
            if entry["engine_pass_rate"] > 0 and entry["delivery_pass_rate"] > 0:
                entry["diagnosis"] = (
                    "CONJUNCTION FAILURE — both axes are individually reachable but "
                    "never co-occur. Do NOT read this as an unreachable target (§6.1); "
                    "switch rft_filter to disjunctive or relax the near-manifold target.")
            elif entry["delivery_pass_rate"] == 0:
                entry["diagnosis"] = "DELIVERY-ZERO — needs near-manifold relaxation on delivery."
            elif entry["engine_pass_rate"] == 0:
                entry["diagnosis"] = "ENGINE-ZERO — lower the engine pass threshold."
        out[cell] = entry
    return out


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


def rft_raw_path(dataset_out: str) -> str:
    """Sibling path for the raw kept turns (cell + context + completion)."""
    p = Path(dataset_out)
    return str(p.with_suffix(".rft.jsonl"))


def rft_rejects_path(dataset_out: str) -> str:
    """Sibling path for REJECTED turns with their per-axis tuple (§6.2)."""
    p = Path(dataset_out)
    return str(p.with_suffix(".rejects.jsonl"))


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
        policy_model_path=cfg["policy"]["model_path"],       # the Simulator, not the annotator
        policy_base_url=cfg["policy"]["base_url"],
        backends=backends,
        arcs_per_cell=ws.get("arcs_per_cell", 30),
        prefix_turns=cfg["grpo"].get("prefix_turns", 4),
        temperature=cfg["policy"].get("temperature", 0.8),
        rft_filter=ws.get("rft_filter_mode", CONJUNCTIVE),
        rejects_out=(rft_rejects_path(dataset_out)
                     if (dataset_out and ws.get("log_rejects_with_tuple", True))
                     else None),
    )
    records = to_sft_records(examples)
    if dataset_out:
        save_dataset(records, dataset_out)
        # Raw kept turns with their cell and context, before chat formatting.
        save_dataset([e.to_dict() for e in examples], rft_raw_path(dataset_out))
    if not records:
        hint = ""
        if dataset_out:
            hint = (f" Rejected turns with their per-axis tuple were written to "
                    f"{rft_rejects_path(dataset_out)} (§6.2) — read the "
                    "(engine_pass, delivery_pass) census there BEFORE concluding the "
                    "target is unreachable. Both axes passing individually but never "
                    "together is a conjunction failure (§6.1), not an unreachable cell.")
        raise RuntimeError(
            "RFT collected zero passing turns — the base rate is too low even for "
            "warm-start. Relocate the target nearer the manifold / apply the "
            "curriculum (grpo_spec §5.5) before retrying." + hint
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
