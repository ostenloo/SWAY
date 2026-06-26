# Model Assignment & Infrastructure Spec

Concrete instantiation of the roles in `patient_pipeline_spec.md` (**[PIPE §2, §9, §10, §13]**) onto local checkpoints, plus the run-time stack. Resolves [PIPE §13] "Local model assignment," "reference-interlocutor model choice," and the [spec §10] / [BS §10] infrastructure items. Hardware target: single **RTX 5090 (Blackwell, sm_120, 32 GB)**.

> **Standing decisions inherited.** All roles local open-weight ([PIPE §9]); the one hard line is **Judge ≠ MUT, disjoint family** ([PIPE §9.1]); the Goodhart wall governs *information*, not weights ([PIPE §5]). This spec adds the family layout and the quant/runtime story those decisions imply on Blackwell.

> **Two graders, don't conflate them.** The **Fidelity checker** ("checker") scores the **patient** turn — *is the simulated patient on-profile?* — and runs **at build time** as the per-turn regeneration gate ([PIPE §6, §7]). The **Judge** scores the **MUT** reply — *did the model drift?* — and runs **offline/batched** after collection ([BS §5]). Checker = is my stimulus faithful; Judge = how did the model respond. Separate checkpoints, separate families, by design — neither touches the other's domain (the checker never reads the fact ledger; the Judge never checks patient fidelity). Wherever this doc says "checker," it means the patient gate, **not** the Judge.

---

## 1. Checkpoint roster — 6 on disk, 2 in VRAM at peak

Ten logical roles ([PIPE §2]) + 3 MUTs collapse onto **6 distinct checkpoints**. Three role-absorptions do the collapsing (each justified in §2):

| # | Checkpoint | Family | Logical roles it serves | Runtime |
|---|---|---|---|---|
| 1 | Qwen3.6-35B-A3B | Qwen | **MUT** | vLLM |
| 2 | Gemma 4 31B | Gemma | **MUT** | vLLM |
| 3 | gpt-oss-20B | OpenAI-oss | **MUT** | vLLM |
| 4 | Pipeline model (~8B, e.g. Ministral 3) | Mistral | **Simulator** + **Optimizer** | vLLM |
| 5 | Fidelity checker (mini, e.g. Phi-4-mini) | Microsoft/Phi | **Fidelity checker** + **opt-time interlocutor** | vLLM (build-time only) |
| 6 | Judge (~24–70B, disjoint family, e.g. Llama-class) | **non-Mistral, non-Phi, non-MUT** | **Judge** + **certification interlocutor** | vLLM, offline/batched |

**Family constraint (load-bearing — §3).** The three support checkpoints (4, 5, 6) must be **three distinct families, all disjoint from the MUT pool** {Qwen, Gemma, OpenAI-oss}. The worked example uses Mistral / Phi / Llama. *This corrects a collision in the prior draft, which put the pipeline model on Ministral and the Judge on Mistral Small — both Mistral family — which breaks the certification freshness test (§3.3).*

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

The prior draft labeled all three MUTs "4-bit AWQ." That's right for two of them and wrong for gpt-oss, and the difference is a **declared confound**.

### 4.1 What the formats are
- **AWQ = W4A16** (weight-only): 4-bit weights, unpacked to FP16 at compute, matmul on the FP16 path (Marlin kernel). Does **not** need Blackwell's new low-precision tensor cores — which is *why it's the stable choice on sm_120*. AWQ Marlin runs robustly on the 5090 and in some configs **outperforms FP8**, because the native FP8 cores aren't always fully exposed yet.
- **FP8 = W8A8**, **INT8 = W8A8**: weights+activations low-precision, native tensor-core paths. **INT8/bitsandbytes is broken on sm_120** (produces corrupted output) — do not use.
- **MXFP4 / NVFP4 ≈ W4A4**: 4-bit weights+activations on Blackwell's native FP4 cores. **gpt-oss ships natively in MXFP4.**

### 4.2 Per-MUT assignment
| MUT | Quant | Compute path | Note |
|---|---|---|---|
| Qwen3.6-35B-A3B | AWQ 4-bit (W4A16) | FP16 / Marlin | stable; community AWQ of this class ≈ 22 GB weights, fits 32 GB with KV headroom |
| Gemma 4 31B | AWQ 4-bit (W4A16) | FP16 / Marlin | stable |
| gpt-oss-20B | **native MXFP4** (W4A4) | FP4 / NVFP4 (verify, §4.3) | do **not** re-quantize to AWQ; run as shipped |

### 4.3 Two things to verify before the sweep
- **MXFP4 backend dispatch on sm_120.** Some vLLM versions don't match SM120 in the MXFP4 backend-selection logic and silently **fall back to weight-only FP4 via Marlin** instead of the native NVFP4 kernels (the native SM120 kernels exist but the dispatch check missed them). Check the startup log for the Marlin-fallback warning and pin a version where SM120 → native NVFP4.
- **Quant is a declared confound ([BS §10.2]).** Qwen/Gemma on W4A16-FP16 vs gpt-oss on W4A4-FP4 means the three MUTs differ in quant scheme *and* family. You can't fully de-confound this — gpt-oss's native format is part of what the model *is* — so **log quant per MUT and treat it as an analysis factor**, never attribute a drift gap to "the model" without noting the quant difference rides along.

---

## 5. Runtime — single-engine vLLM, co-residency by sequencing

**Decision: one runtime, vLLM, for every role.** The MUT wants vLLM unconditionally (it's the measured object; AWQ-Marlin and native MXFP4 are vLLM paths), and a second runtime buys VRAM-partition relief at the cost of a second stack to pin, a second quant artifact type, and a second API in the harness — friction a single-card reproducibility-first project shouldn't take on by default.

The thing a two-runtime split was avoiding is **three vLLM engines fighting over KV cache + Triton headroom**. Solve that by sequencing and conservative allocation, not by adding an engine:

- **Never 3 throughput-critical engines hot at once.** The only role that needs a fast persistent engine is the MUT. The support models are small (~8B + mini) and not throughput-bound — the patient turn and MUT turn alternate, so they don't need to co-saturate the card.
- **Conservative `gpu_memory_utilization`** on the MUT engine, sized to leave headroom for the support models' weights *and* Triton autotuner overhead (the autotuner crash from §0/§7 is the failure to size against). Hand-partition the fractions once per MUT size band.
- **Phase separation already does most of the work** (§6): Build/Certify involve no MUT; Score runs the Judge alone. Only the **Run** phase is genuinely multi-model, and there it's 1 MUT + pipeline.

| Role | Engine | Residency |
|---|---|---|
| MUT | vLLM, conservative `gpu_memory_utilization` | persistent during its Run phase |
| Pipeline (Simulator/Optimizer) | vLLM | co-resident at small fraction; Optimizer only active at build-time rewrites |
| Fidelity checker | vLLM | build-time only; not co-resident at run time |
| Judge | vLLM, alone in VRAM | Score phase only — uses the whole card, batched |


---

## 6. Three phases (sequential, never co-resident)

| Phase | When | In VRAM | What happens |
|---|---|---|---|
| **Build** | once per cell, offline | Pipeline (8B) + Checker (mini) | Optimize patient prompt → freeze. Opt-time interlocutor = Checker checkpoint. No MUT. |
| **Certify** | once per cell, at freeze | Pipeline (Simulator) + Checker + **Judge checkpoint as cert interlocutor** | Validate frozen prompt on fresh authored detail + cross-family interlocutor ([PIPE §4.2]). Generation then check can sub-phase if VRAM tight. |
| **Run** | per MUT, sequential | 1 MUT + Pipeline (all vLLM, conservative util) | Generate transcripts. Fidelity checker is build-time only — certification at freeze time validates the frozen prompt. One MUT at a time; unload before the next. |
| **Score** | after all MUTs done | Judge (alone) | Score all collected transcripts offline, batched. **This is the deferrable role**, not the checker. |

> **Labeling fix.** The **Fidelity checker runs at build time only** — it scores the patient turn and feeds back to the optimizer. At run time the frozen prompt is immutable; certification at freeze time validates it against a cross-family interlocutor ([PIPE §4.2]), so the checker is not needed in the live loop. The **Judge is deferred to an offline batch pass** ([PIPE §6]). Score-after-collection is the natural batching point.

### VRAM at peak (run phase, per MUT)
| Component | Target | Engine |
|---|---|---|
| MUT | ~10–22 GB (AWQ/MXFP4 4-bit) | vLLM (conservative util) |
| Pipeline (8B) | ~5 GB | vLLM (small fraction) |
| **Total** | **~15–27 GB / 32 GB** | — |

35B-A3B AWQ (~18–22 GB) + 8B pipeline is the tight end (~23–27 GB); leave the rest for KV cache and Triton headroom. 31B and 20B MUTs are more comfortable.

---

## 7. Reproducibility — pin the *stack*, not just the weights

All-local frozen weights only buy reproducibility if the **inference stack** is pinned too — a vLLM bump can silently move gpt-oss onto a different FP4 kernel path mid-sweep, and the sm_120 stack is version-sensitive (flash-attn symbol errors, Triton autotuner crashes, CUDA-graph behavior all move release-to-release). Pin and record, as part of the run artifact:

- **vLLM version**, CUDA toolkit, PyTorch, driver, kernel.
- Per-MUT: quant scheme + the resolved compute path (verify native-NVFP4-vs-Marlin-fallback, §4.3).
- Decoding params: **graders (Judge, Checker) at temperature 0**; **Simulator** at small temperature with **seeds logged**, multiple seeds per (cell × MUT) ([PIPE §10.3]).
- The build-time **30× adherence variance** per cell as the patient-noise reference ([PIPE §4.2]).

---

## 8. Open / to confirm
- **Exact support-model checkpoints.** §1 fixes the *families* (three distinct, all ⊥ MUTs; Judge ⊥ Simulator for cert). Slot in current-best models per family at build time.
- **Judge size.** ~24–70B disjoint-family, validated against AnnoMI ([PIPE §9.3]) — size is a quality/throughput tradeoff, not a hard requirement; certification by measurement, not by parameter count.
- **MXFP4 dispatch verification** on the pinned vLLM (§4.3) before the sweep.
- **Pipeline-model competence as opt-interlocutor** — if cert fails broadly, raise the opt-time interlocutor off Phi-mini (§3.3 caveat).
- **Single-vLLM partition holds at the tight end** — pilot MUT(35B-A3B) + pipeline(8B) co-resident on 32 GB with workable `gpu_memory_utilization` and Triton/KV headroom.