# SWAY GRPO Simulator finetune

Implementation of `specs/grpo_spec.md` — a **build-time**, weight-level GRPO
finetune of the Simulator (LLM-2) that replaces the prompt-optimization Optimizer
for cells where prompt conditioning can't reproduce personality adherence. The
finetuned Simulator is a new build-time artifact; run-time (gate, Judge, drift
scoring) is unchanged.

The code reuses the existing harness (`sway_harness/`: `client`, `parser`,
`fidelity`, `build`) via `grpo/_bootstrap.py`, which puts `sway_harness/` on
`sys.path` (the harness uses flat imports).

## Layout → spec deliverables (§13)

| File | Spec | What it is |
|---|---|---|
| `reward/fidelity_reward.py` | §4 | The pure reward: partial-credit diagnostics × multiplicative realism floor. Import-clean (C1). |
| `reward/backends.py` | §4, D0.2 | Pluggable graders: local annotator (Ollama/vLLM), Opus API, and a folded stub for the §8 test. One annotation call feeds engine/delivery/realism. |
| `reward/turn_fidelity.py` | §4 | Per-turn label → {engine, delivery, realism} binaries, reusing `fidelity.py`'s target poles. |
| `reward/trl_adapter.py` | §7 | Wraps `fidelity_reward` as a TRL `GRPOTrainer` reward function. |
| `data/rollout.py` | §5 | State construction, per-turn grouping, cross-interlocutor spread (≥2 bare interlocutors). |
| `train/rft_warmstart.py` | §6 | Reward-filtered SFT before GRPO. |
| `train/grpo_loop.py` | §7 | GRPO loop (TRL/Unsloth + vLLM rollout, reference = disabled adapter). Calls the §8 gate first (C6). |
| `gates/delivery_probe.py` | §8 | Blocking adversarial delivery probe (grievance vs interlocutor-hostility), κ ≥ 0.80 hard gate. |
| `monitor/online_audit.py` | §9 | High-advantage audit, reward↔fidelity gap, realism-trip rate, group-collapse rate. |
| `cert/certify_and_freeze.py` | §10 | Held-out authored detail + second bare interlocutor cert; freeze at deployment quant with a reproducibility manifest (C7). |
| `configs/grpo.yaml`, `config.py` | §11 | Config surface + loader. |
| `tests/` | A1, A2 | C1 import guard + reward shape; §8 gate blocks a folded backend. |

## Two decisions (spec §0)

- **D0.1 base checkpoint** — `base_model: Qwen/Qwen2.5-14B-Instruct` in the config
  (the requested "llama2.5:14b" is not a real model). Only the LoRA target-module
  names and the §12 VRAM math depend on this.
- **D0.2 reward backend** — pluggable behind one interface. Default is the local
  both-axis checker; set `reward.backend: opus` to route through the Anthropic API.
  **Whichever supplies delivery must clear the §8 gate.**

## Non-negotiable constraints (§2) — where they live

- **C1** reward reads only `(patient_turn, P, context, cell)` — `fidelity_reward.py`
  imports nothing drift-side; `tests/test_c1_import_guard.py` greps + unit-proves it.
- **C3** partial-credit diagnostics + multiplicative realism floor — `fidelity_reward.py`.
- **C4** graders temp-0 / frozen — the backend `_annotate` calls.
- **C6** §8 gate is blocking — `run_grpo` calls `assert_delivery_gate` before building the trainer.
- **C7** certify before freeze — `freeze_adapter` refuses a non-certified adapter.

## Running

Reward/rollout/gate/monitor/cert + tests need only `requests` + `pyyaml`:

```bash
pip install -r grpo/requirements-grpo.txt      # (uncomment the training stack on the GPU host)
python -m pytest grpo/tests/ -q                # A1 + A2
```

Full pipeline on the RTX 5090 host (`P_by_cell` = frozen profile prompts per cell,
e.g. loaded from `results/build/<cell>_prompt.txt`):

```python
from grpo.config import load_config
from grpo.train.rft_warmstart import run_rft
from grpo.train.grpo_loop import run_grpo
from grpo.cert.certify_and_freeze import certify_adapter, freeze_adapter
from grpo.data.rollout import Interlocutor

cfg = load_config()
# 1. warm-start (§6)   2. GRPO (§7, gate runs first)   3. certify + freeze (§10)
rft = run_rft(cfg, P_by_cell, adapter_out="results/grpo/adapters/rft")
grpo = run_grpo(cfg, P_by_cell, adapter_in=rft, adapter_out="results/grpo/adapters/grpo")
```

The §8 gate can (and should) be run standalone against any candidate delivery
backend before committing GPU time:

```python
from grpo.config import load_config, build_reward_backends
from grpo.gates.delivery_probe import run_delivery_probe
b = build_reward_backends(load_config())
print(run_delivery_probe(b.delivery).to_dict())
```

## Acceptance (§13)

- **A1** — `test_c1_import_guard.py`: no drift-side imports; reward reads only the four inputs.
- **A2** — `test_delivery_probe.py`: a folded (grievance→hot) backend fails κ and raises `DeliveryGateError`; a clean backend passes.
- **A3–A5** — GPU-host runs: GRPO lifts held-out fidelity vs the prompt-opt control without §9 divergence; certified adapter freezes at deployment quant; config + seed recorded in `freeze_manifest.json`.
