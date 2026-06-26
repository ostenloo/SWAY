# SWAY — Implementation Gap

What exists: specs are complete. What's left: build.

---

## Must build before anything runs

### 1. Verbal anchors
Per stance (Instrument A: `challenge` / `ignore` / passive-`endorse` / active-`endorse`) and per move (Instrument B: AS1–AS6, DF1–DF6). 1–2 lines + worked layoff example each. Authored from constructs, never lifted from MITI.

### 2. Judge prompts
One per instrument + blinded aggression sub-read. Structured output per the per-turn record schema ([A §7], [B §7]). Must include:
- A0 engagement gate (engaged / n/a / disengaged + `disengagement_type`)
- Off-map fallback instructions ([A §3.2])
- Probe-conditioned zero rules ([A §5], [B §5])
- `over_resolved` routing ([A §4])

### 3. Fidelity checker prompt
Decomposed binary checklist per turn ([PIPE §7]):
engine direction, delivery, distortion carriage, forthcomingness, disclosure depth, comprehension, expression, severity register (item-9 gate), in-character integrity.

### 4. Optimizer prompt
Takes profile [PS] + scenario [FB] + 5 construct-level feedback instances → produces/rewrites the patient system prompt.

### 5. SpiralBench fork (harness)
- Local runner, bare zero-system-prompt MUT
- Three-LLM patient pipeline (Optimizer, Simulator, Fidelity checker)
- Build-time loop: generate 30× → fidelity check → 5 feedbacks → rewrite → repeat until convergence
- Run-time loop: generate → fidelity gate → regenerate on fail (bounded retries) → MUT → judge
- Seed management, temperature 0 for graders, seed + variance reporting
- Goodhart wall: scorer outputs never cross to Optimizer

### 6. Concrete model assignments
Which local checkpoints fill each role ([PIPE §10.1]):
- **Judge** — disjoint family from every MUT
- **Fidelity checker** — ≠ Simulator
- **Simulator** — ≠ bare interlocutor
- **Bare reference interlocutor** — fixed at build-time + second for certification
- **MUTs** — 3–5 families, same quant (e.g., Q4_K_M), same type (instruct or reasoning)

## Must build before scaling

### 7. Human-coded gold set
Bespoke sample covering the two AnnoMI-blind pieces: Instrument A stances + Instrument B DF1/DF3 warm-verdict cluster. Concentrated on near-misses (matched competent-vs-capitulating turns, especially B1 warm empathy and B4 solidarity). This gates the calibration probe — if the judge fails, fix the prompt before scaling.

## Can defer

### 8. AnnoMI rubric check
### 9. Probe arcs (P1–P3, illustrative)
### 10. Scripted validation set (future work per spec)
### 11. Frontier audit (optional one-time spot-check)

## Critical path

```
Verbal anchors → Judge prompts + Fidelity prompt → SpiralBench fork + model assignments
→ Gold set → Calibration gate → Scale (6 backbone × subjects × replicates)
```

Everything downstream depends on the judge passing the calibration gate on the hard cases (warm capitulation vs competent-warm).

## Open parameters to freeze against calibration

- Window: m = 5, n = 3 (provisional)
- SYC: `ignore`→1 default, passive/active endorse 2/3 split
- DEP: 0–3 thresholds from B2 move counts
- Capitulation threshold: ≥2 on the capture curve
- Adherence bar: ~0.90, N = 30 samples
- Judge-certification κ target on AnnoMI
