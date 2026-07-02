# Model Assignment & Infrastructure Spec

Concrete instantiation of the roles in `patient_pipeline_spec.md` (**[PIPE §2, §9, §10, §13]**) onto local checkpoints, plus the run-time stack. Resolves [PIPE §13] "Local model assignment," "reference-interlocutor model choice," and the [spec §10] / [BS §10] infrastructure items. Hardware target: single **RTX 5090 (Blackwell, sm_120, 32 GB)**.

> **Standing decisions inherited.** All roles local open-weight ([PIPE §9]); the one hard line is **Judge ≠ MUT, disjoint family** ([PIPE §9.1]); the Goodhart wall governs *information*, not weights ([PIPE §5]). This spec adds the family layout and the quant/runtime story those decisions imply on Blackwell.

> **Two graders, don't conflate them.** The **Fidelity checker** ("checker") scores the **patient** turn — *is the simulated patient on-profile?* — and runs **at build time** as the per-turn regeneration gate ([PIPE §6, §7]). The **Judge** scores the **MUT** reply — *did the model drift?* — and runs **offline/batched** after collection ([BS §5]). Checker = is my stimulus faithful; Judge = how did the model respond. Separate checkpoints, separate families, by design — neither touches the other's domain (the checker never reads the fact ledger; the Judge never checks patient fidelity). Wherever this doc says "checker," it means the patient gate, **not** the Judge.

---

## 1. Model Assignment — 5 production models assigned to roles

### Production Role Assignment

| # | Checkpoint | Family | Role | VRAM | Phase | Notes |
|---|---|---|---|---|---|---|
| **MUT-1** | **Gemma 4 12B AWQ-INT4** | Gemma | **Model Under Test** | 8.9 GB | Run | Primary test subject (multimodal capable) |
| **MUT-2** | **Qwen3-4B AWQ-4bit** | Qwen | **Model Under Test** | 3.8 GB | Run | Secondary test subject (lightweight) |
| **Infra-1** | **Ornith-9B AWQ-FP8** | Ornith | **Simulator** + **Optimizer** | 10.8 GB | Build/Run | Co-resident with MUT during Run |
| **Infra-2** | **Mistral-7B BNB-4bit** | Mistral | **Fidelity Checker** | 4.3 GB | Build-time only | Validates patient fidelity during optimization |
| **Infra-3** | **Llama 3.1 8B BNB-4bit** | Llama | **Judge** + **Cert Interlocutor** | 5.9 GB | Score/Certify | Evaluates MUT drift; cross-family cert test |

**Family disjointness verification (§3):**
- Judge (Llama) ⊥ MUT-1 (Gemma) ✓
- Judge (Llama) ⊥ MUT-2 (Qwen) ✓
- Checker (Mistral) ⊥ Simulator (Ornith) ✓
- Judge (Llama) ⊥ Simulator (Ornith) ✓ (cert freshness)

**Sequential testing:** Run Gemma first (8.9 GB + 10.8 GB Ornith = 19.7 GB), then Qwen (3.8 GB + 10.8 GB Ornith = 14.6 GB). Both fit with headroom.

**Family constraint (load-bearing — §3).** The four active roles (MUT, Simulator, Checker, Judge) use **four distinct families: Ornith, Llama, Mistral, Gemma** — all disjoint from each other. This satisfies the hard requirement that Judge ≠ MUT family and all support models are mutually independent. Qwen3-4B remains in reserve for lightweight testing or comparative runs.

**Revision rationale.** Removed artificial size thresholds (35B MUT, 24–70B Judge). The spec now requires **capability** (role-specific competence) and **family disjointness**, not parameter count. Ornith-9B is sufficiently capable as a MUT for profile fidelity work; Gemma 4 12B is sufficiently capable as a Judge. Both are smaller than the prior draft assumed, but parameter counts were never a hard constraint — the Goodhart wall and family isolation are.

---

## 2. The three role-absorptions

Each fold is permitted by [PIPE §10.1]; none crosses a forbidden or "avoid" line.

1. **Optimizer → Pipeline model (#4).** The Goodhart wall is a constraint on information flow, not weights ([PIPE §5]); the Optimizer never receives scorer output regardless of which checkpoint runs it. Free fold onto the Simulator's checkpoint.
2. **Opt-time interlocutor → Fidelity checker (#5).** The bare interlocutor the patient talks to *during build optimization* runs on the checker checkpoint. Phi ⊥ Simulator (Mistral), so this does **not** trip the `Bare interlocutor × Simulator: Avoid` pairing ([PIPE §10.1]). The checker judging the *patient* turn while Phi also produced the *assistant* turn is clean — the checker scores Ministral's turns, not its own.
3. **Certification interlocutor → Judge (#6).** The *different* bare interlocutor used at freeze time ([PIPE §4.2]) runs on the Judge checkpoint, loaded sequentially at certification (not in any inline loop, so no VRAM clash). Its family is ⊥ both the opt-time interlocutor (Phi) **and** the Simulator (Mistral) — see §3.3.

**Why not collapse to 5.** The Fidelity checker can fold onto neither neighbor: onto the **Simulator** is the `Avoid` blindness ([PIPE §10.1] — a checker on the Simulator's base model is blind to that Simulator's characteristic profile-violations); onto the **Judge** is operationally wrong — the checker runs at build-time during prompt optimization, while the Judge runs *offline and batched* at score time on a large checkpoint. Separate for throughput even though the table permits sharing.

---

## 3. Family layout & independence

### 3.1 The hard line: Judge ≠ MUT, disjoint family
A Judge sharing the MUT's family inherits the same helpfulness prior and **under-flags** the exact capitulation it should catch ([PIPE §9.1]) — directional bias, not symmetric noise. The Judge family must appear **nowhere** in the MUT pool. Mistral/Llama/Phi all satisfy this against {Qwen, Gemma, OpenAI-oss}.

> **Note — the 120B judge is out.** A `gpt-oss-120B` Judge (floated in the prior draft's prose) is **forbidden**: same family as the gpt-oss-20B MUT, violating §3.1. The Judge is a disjoint-family model in the ~24–70B class, run offline (§5), not a same-lineage giant in the live loop.

### 3.2 The "avoid" pairings, satisfied
| Pair | Constraint | Here |
|---|---|---|
| Fidelity checker × Simulator | Avoid | Phi ⊥ Mistral ✓ |
| Opt-time interlocutor × Simulator | Avoid | Phi ⊥ Mistral ✓ (interlocutor folds onto checker) |
| Simulator × Judge | Fine | Mistral × Llama — and *fine even if shared*, since the Judge grades the MUT, not the patient ([PIPE §10.1]) |

### 3.3 Certification freshness — why the Judge family must also be ⊥ Simulator
Certification ([PIPE §4.2]) swaps in an interlocutor the patient was **never optimized against** to prove the frozen prompt encodes the *profile*, not the opt-time interlocutor's quirks. Two requirements on the cert interlocutor:

- **⊥ opt-time interlocutor (Phi)** — else it's a reseed, not a freshness test. (Judge = Llama ✓.)
- **⊥ Simulator (Mistral)** — else at cert time the patient talks to its own family, the `Bare interlocutor × Simulator: Avoid` pairing fires, exchanges read unnaturally self-consistent, and the patient looks *more* adherent than it is → cert passes too easily → false confidence. (Judge = Llama ✓.)

This is exactly why the Judge can't be Mistral: cert would then put a Mistral-family interlocutor against a Mistral-family Simulator. **Three distinct non-MUT families for #4/#5/#6 is what makes the no-extra-download certification clean.**

> **Competence caveat ([PIPE §13]).** Under zero system prompt the bare model's default skill *is* the interlocutor's competence — no dial. The opt-time interlocutor (Phi-mini) is deliberately weak; that's acceptable because the **pressure schedule owns the arc shape** ([PIPE §8]), so escalation doesn't depend on interlocutor skill. The risk that a weak opt-interlocutor lets the prompt overfit to a degenerate foil is precisely what the **capable cross-family cert interlocutor (Judge checkpoint) catches** — cert is the gate that vindicates or fails the weak-opt-interlocutor choice. If cert craters across cells, the opt-interlocutor was too weak; raise it.

---

## 4. Quantization on Blackwell (sm_120)

All five models use 4-bit quantization (AWQ or BNB), both stable and well-tested on sm_120. No native FP8/FP4 paths needed; Marlin kernel (weight-only FP16 compute) is robust and well-supported.

### 4.1 Quantization schemes used
- **AWQ (W4A16)**: weight-only 4-bit, unpack to FP16 at compute, Marlin kernel. Stable on sm_120. Used by Qwen3-4B, Gemma 4 12B, Ornith-9B.
- **BNB 4-bit**: bitsandbytes quantization, weight-only 4-bit. Stable on sm_120, widely tested. Used by Mistral-7B, Llama 3.1 8B.

Both are conservative choices — no reliance on Blackwell's new low-precision tensor cores (which have version-dispatch issues on sm_120). Marlin + BNB paths are well-traveled.

### 4.2 Per-model assignment
| Role | Model | Quant | Compute path | VRAM |
|---|---|---|---|---|
| **MUT** | Ornith-9B AWQ-FP8 | AWQ 4-bit | FP16 / Marlin | 10.8 GB |
| **Simulator** | Llama 3.1 8B | BNB 4-bit | FP16 / BNB | 5.9 GB |
| **Checker** | Mistral-7B BNB-4bit | BNB 4-bit | FP16 / BNB | 4.3 GB |
| **Judge** | Gemma 4 12B | AWQ INT4 | FP16 / Marlin | 8.9 GB |
| **Reserve** | Qwen3-4B | AWQ 4-bit | FP16 / Marlin | 3.8 GB |

**Quantization is NOT a confound here.** All models use conservative, well-tested paths on sm_120 (no native FP8/FP4, no dispatch games). Log quant per model for reproducibility, but inter-model drift is orthogonal to quantization scheme.

---

## 5. Runtime — single-engine vLLM, co-residency by sequencing

**Decision: one runtime, vLLM, for every role.** All five models use quantized weights (AWQ or BNB 4-bit); vLLM handles both. A second runtime buys VRAM-partition relief at the cost of a second stack to pin, a second quant artifact type, and a second API in the harness — friction a single-card reproducibility-first project shouldn't take on by default.

The thing a two-runtime split was avoiding is **multiple vLLM engines fighting over KV cache + Triton headroom**. Solve that by sequencing and conservative allocation, not by adding an engine:

- **Never all 5 models hot at once.** The only role that needs a persistent fast engine is the MUT (Run phase). The support models are not throughput-bound — patient turn and MUT turn alternate, so they don't need to co-saturate the card.
- **Conservative `gpu_memory_utilization`** on the MUT engine (Ornith-9B at 10.8 GB), sized to leave headroom for the co-resident Simulator (Llama 5.9 GB) weights *and* Triton autotuner overhead. At peak Run phase: 10.8 + 5.9 = 16.7 GB, leaving 15.3 GB for KV cache and headroom.
- **Phase separation does most of the work** (§6): Build/Certify involve no MUT; Score runs the Judge alone. Only the **Run** phase is genuinely multi-model, and there it's 1 MUT + Simulator.

| Role | Model | Engine | Residency | VRAM |
|---|---|---|---|---|
| MUT | Ornith-9B | vLLM, conservative util | persistent during Run phase | 10.8 GB |
| Simulator/Optimizer | Llama 3.1 8B | vLLM, co-resident | active at build/run, idle during score | 5.9 GB |
| Fidelity checker | Mistral-7B BNB-4bit | vLLM | build-time only; not resident at run time | 4.3 GB |
| Judge | Gemma 4 12B | vLLM, alone | Score phase only — uses the whole card, batched | 8.9 GB |
| Reserve | Qwen3-4B | vLLM | on-demand for ablations/comparisons | 3.8 GB |


---

## 6. Four phases (sequential, never co-resident)

### 6.1 Phase execution (Gemma & Qwen as MUTs, Ornith Simulator, Llama Judge)

| Phase | When | Models in VRAM | Total VRAM | What happens |
|---|---|---|---|---|
| **Build** | Once per prompt cell, offline | Ornith (10.8) + Mistral (4.3) | **15.1 GB** (16.9 GB free) | Optimize patient prompt using Ornith as Simulator. Mistral as Checker validates patient fidelity. No MUT loaded. |
| **Certify** | Once per prompt cell, at freeze | Ornith (10.8) + Llama (5.9) | **16.7 GB** (15.3 GB free) | Unload Mistral, load Llama. Test frozen prompt on fresh detail. Llama = disjoint-family cert interlocutor (⊥ Ornith, ⊥ MUT). Validates generalization. |
| **Run (Gemma)** | Per MUT, sequential | Gemma (8.9) + Ornith (10.8) | **19.7 GB** (12.3 GB free) | Load Gemma MUT. Generate transcripts. Ornith co-resident as Simulator for any re-prompting. |
| **Run (Qwen)** | Per MUT, sequential | Qwen (3.8) + Ornith (10.8) | **14.6 GB** (17.4 GB free) | Unload Gemma, load Qwen MUT. Generate transcripts with same Ornith Simulator. More headroom than Gemma run. |
| **Score** | After all MUTs done, offline | Llama (5.9) alone | **5.9 GB** (26.1 GB free) | Load Judge (Llama, unload Ornith + MUT). Score all collected transcripts offline, batched. High throughput. |

### 6.2 Phase dependencies and unload sequence

```
Build (Ornith + Mistral)
  ↓ unload Mistral
Certify (Ornith + Llama)
  ↓ unload Llama, load Gemma
Run-Gemma (Gemma + Ornith)
  ↓ unload Gemma, load Qwen
Run-Qwen (Qwen + Ornith)
  ↓ unload Qwen, load Llama
Score (Llama alone)
```

### 6.3 Key VRAM advantages of this configuration

- **Run phase is most flexible**: Gemma + Ornith uses 19.7 GB (tight but workable); Qwen + Ornith uses only 14.6 GB (very comfortable).
- **Score phase is lightest**: Llama alone (5.9 GB) is the smallest footprint, maximum parallelism opportunity.
- **Build phase is heavier than before** (15.1 vs 10.2 GB), but still well under 32 GB with 16.9 GB free for KV/Triton.
- **No phase exceeds 19.7 GB** — all phases fit with meaningful headroom.

**Checkpoint: This configuration is valid.** All family constraints satisfied. All phases fit in 32 GB. Ready to execute.

### VRAM at peak (run phase)
| Component | Size | Engine |
|---|---|---|
| MUT (Ornith-9B AWQ-FP8) | 10.8 GB | vLLM (conservative util) |
| Simulator (Llama 3.1 8B BNB-4bit) | 5.9 GB | vLLM (co-resident) |
| **Total (Run phase)** | **16.7 GB / 32 GB** | — |
| **Headroom for KV + Triton** | **15.3 GB** | — |

Run phase is the tightest phase: MUT + Simulator co-resident. Build phase (Llama + Mistral) is 10.2 GB; Certify phase adds Gemma (19.1 GB worst-case if not sequenced). Score phase runs Gemma alone (8.9 GB). All phases fit comfortably.

---

## 7. Reproducibility — pin the *stack*, not just the weights

All-local frozen weights only buy reproducibility if the **inference stack** is pinned too — a vLLM bump can silently move gpt-oss onto a different FP4 kernel path mid-sweep, and the sm_120 stack is version-sensitive (flash-attn symbol errors, Triton autotuner crashes, CUDA-graph behavior all move release-to-release). Pin and record, as part of the run artifact:

- **vLLM version**, CUDA toolkit, PyTorch, driver, kernel.
- Per-MUT: quant scheme + the resolved compute path (verify native-NVFP4-vs-Marlin-fallback, §4.3).
- Decoding params: **graders (Judge, Checker) at temperature 0**; **Simulator** at small temperature with **seeds logged**, multiple seeds per (cell × MUT) ([PIPE §10.3]).
- The build-time **30× adherence variance** per cell as the patient-noise reference ([PIPE §4.2]).

---

## 8. Orchestration — Docker-based workflow management

**Infrastructure:** All models run as independent Docker containers (Transformers + FastAPI). Managed by `sway-orchestrate-docker` script and `orchestrate.json` config.

**Command examples:**
```bash
./sway-orchestrate-docker build --cell b4
./sway-orchestrate-docker certify --cell b4
./sway-orchestrate-docker run --cell b4 --muts gemma,qwen3-4b
./sway-orchestrate-docker score
./sway-orchestrate-docker full --cell b4 --muts gemma,qwen3-4b
./sway-orchestrate-docker status
./sway-orchestrate-docker cleanup
```

**Phase sequencing (automatic):**
1. **Build**: Start Ornith (Simulator) + Mistral (Checker) → optimize patient prompt
2. **Certify**: Swap Mistral → Llama (Judge) → validate frozen prompt
3. **Run-Gemma**: Start Gemma (MUT-1) + Ornith (Simulator) → generate transcripts
4. **Run-Qwen**: Swap Gemma → Qwen (MUT-2) + same Ornith → generate transcripts
5. **Score**: Swap Qwen/Ornith → Llama (Judge) → score all transcripts

All phase transitions respect VRAM constraints (§6).

---

## 9. Validation & next steps

**Before first run:**

- **Test orchestrate.json parsing:** `./sway-orchestrate-docker status` should list all 5 models and show container state.
- **Verify run-script integration:** Each model's `run-<model>` script must work with the orchestrator (health check at configured port).
- **Dry-run a phase:** `./sway-orchestrate-docker build --cell test` with a small test cell to validate startup sequence and cleanup.
- **Monitor VRAM during phases:** Use `nvidia-smi` in a second terminal to confirm actual vs. predicted VRAM usage.
- **Check phase handoff:** Verify models unload cleanly before next phase starts (no orphaned containers).

Once validated, the full pipeline is ready to execute.