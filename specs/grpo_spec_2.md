# SWAY — GRPO Simulator Finetune Spec **v2** (coding-agent handoff)

Supersedes `grpo_spec.md` **[FT1]**. Companion to
`../docs/SWAY_ENGINE_DELIVERY_REWARD_BAND_SPEC.md` **[BAND]**,
`patient_pipeline_spec.md` **[PIPE]**, `benchmark_spec.md` **[BS]**,
`patient_profile_spec.md` **[PS]**, `sway_profile_roster.md` **[ROST]**,
`syc_spec.md` **[A]**, `dep_spec.md` **[B]**, `layoff_fact_base.md` **[FB]**.

This spec implements a **build-time** weight-level finetune of the Simulator
(LLM-2) via GRPO. Run-time (gate, Judge, drift scoring) is unchanged.

> **Precedence.** Authoritative for the finetune pipeline only. Where it touches a
> component owned by another spec, the component spec wins and this is re-synced.
> It MUST NOT alter the Goodhart wall **[PIPE §5]**, the two-instrument scoring
> **[A][B]**, or the run-time loop **[PIPE §6]**.
>
> **[BAND] owns the reward shape.** §4 here states the interface and the re-syncs;
> the band mathematics, the calibration procedure, and the CB1–CB7 constraints live
> in [BAND] and are not restated. Where the two disagree, [BAND] wins on reward
> shape and calibration; this spec wins on the loop, grouping, VRAM, and
> certification.

---

## Revision note — what changed from [FT1]

For cross-doc audit. Each item is expanded in the section named.

1. **The reward is arc-level and distributional, not per-turn monotone.** [FT1 §4]'s
   `0.5·engine_pass + 0.5·delivery_pass` is replaced by [BAND]'s conditional-band
   reward over a whole arc. Rationale: the monotone binary's optimum is the
   caricature (on-profile on *every* marked turn), which is both unrealistic and the
   R3 attractor. A band with an interior optimum dissolves that without a separate
   penalty. (§4)
2. **The realism floor is removed.** Not deferred — removed, by researcher decision,
   already reflected in `grpo/reward/fidelity_reward.py`. [FT1 C3]'s multiplicative
   floor and [FT1 §11]'s `backend_realism` are void. R3 is now carried by KL and the
   §9 audit alone; the cost is recorded, not hidden. (§2 C3, §4, §12 R3)
3. **Grouping moves per-turn → per-arc.** `T = 20` turns, `G = 8` arcs.
   Per-turn grouping survives *only* inside the RFT warm-start filter. (§5.2, §7)
4. **Both C6 pre-flight gates are removed** (the [FT1 §0.1] non-convergence
   diagnostic and the [FT1 §8.2] stratified delivery validation), by researcher
   decision, already reflected in `grpo/configs/grpo.yaml`. Nothing blocks a run;
   §10 certification is the only remaining human check. The §8 delivery-confound
   work survives as *advisory* and is now load-bearing for a different reason — it
   is the empirical check on [BAND §7] instrument cancellation. (§2 C6, §8)
5. **The no-self-consistency rationale is void as stated.** [FT1 §4] forbade a
   line-to-line term "because it fights the pressure schedule." There is no pressure
   schedule in this design ([BAND §4.1]), so that justification is gone. The rule may
   still be wanted; it needs a fresh reason. (§4.4)
6. **A new offline calibration stage exists** — AnnoMI hand-labeling + grader-space
   measurement, producing the frozen `band_calibration.<ver>.yaml` that the reward
   loads. It must run and be frozen before the first GRPO step. (§6.5, [BAND §6])
7. **Empirical findings from the Aug 3 run are recorded as §0.2** rather than being
   rediscovered. They change what we expect the band to fix.

---

## 0. Decisions

- **D0.1 — Base checkpoint (RESOLVED, unchanged).** `base_model =
  Qwen/Qwen2.5-14B-Instruct`. The 8B Ministral simulator remains the R5 VRAM
  fallback. C8 is anchored to the Qwen family: both champions must be non-Qwen.
- **D0.2 — Reward backend (RESOLVED, amended).** No external API spend. The reward
  backends are the two local axis-specialists: `engine_backend` = engine champion,
  `delivery_backend` = delivery champion (decomposed, §8). **`realism_backend` is
  removed** (see §4.3). Both now expose `.label()` returning a categorical, not the
  old `.score()` binary.
- **D2.1 — Arc length and group size (RESOLVED).** `T = 20` patient turns per arc,
  `G = 8` arcs per group. See §5.2 for the resolution analysis and §12 for cost.
- **D2.2 — Rollout interlocutor (RESOLVED).** A **live** local model, as in the
  prior runs — not a replayed script. Cost consequences in §12.
- **D2.3 — Neutral engine means ABSENT engine (RESOLVED).** For `profile.engine ==
  neutral` (b5, b6, and p2/p3 which pin to b5), the target is **low engine
  expression**: the patient does not do attribution, so few turns are marked
  `int`/`ext`. Engine reward is [BAND §5.3]'s `mode: density_low` — a band on
  `d_eng` toward a low anchor — not a `q`-band. This matches what
  `grpo/reward/engine_decompose.py:115` already implements (`neutral` = neither E1
  nor E2 fires). The alternative reading (*undirected* engine: normal `d_eng`,
  `q_eng ≈ 0.5`) is rejected.

  **Required guard: `d_lo > 0` strictly.** [BAND §5.3] permits `d_lo: 0.0`, and at
  exactly zero an arc expressing *no* engine at all scores `r_eng = 1.0` — full marks
  on half of b5's and b6's reward for going limp. That reopens the inert-simulator
  hole [BAND §5]'s density floor exists to close, on precisely the cells that have no
  `q`-band behind it. Set `d_lo ≈ 0.05` with a tight `s_lo`: a control patient
  expresses *some* attribution, just rarely, so `d_eng = 0` should sit on the lower
  shoulder and be gently penalised rather than scoring perfectly. This preserves the
  absent-engine semantics while removing the exploit.

  *Accepted cost:* 59 of b5's high-advantage turns were `(engine_pass=0,
  delivery_pass=1)` — warm turns that did express engine (§0.2). Under `density_low`
  those are penalised on the engine axis. That is the intended behavior under this
  reading; flagged so it is not mistaken for a bug when b5's engine reward reads low
  early in training.
- **D2.4 — Uninformative-bracket guard: warn and disclose, do not halt (RESOLVED).**
  See §6.5.
- **D2.5 — `scale_rewards = False` (RESOLVED).** TRL's default divides each advantage
  by the group std: `A_i = (r_i − mean(r)) / (std(r) + eps)`. We use mean-centring
  only. Three reasons, in order of weight:

  1. **Std-normalisation partially erases the band's reward geometry.** The band's
     whole purpose is that `q = 1.0` scores meaningfully less than `q = 0.80`, with
     `s_hi` controlling *how much* less. Dividing by a per-group std rescales that
     group by group, so an arc 0.05 off in a tight group and an arc 0.4 off in a wide
     group receive the same advantage — the designed relationship between "how far
     off band" and "how much gradient" is normalised away. Under the old three-valued
     binary there was no geometry to erase; under a band there is.
  2. **At T = 20 the small differences are mostly quantisation, not behavior.** Engine
     at `n_marked = 6` yields only `{0.643, 0.786, 0.929}` (§5.2); one turn flipping
     label moves `q` a full step and can push an arc off the plateau. Amplifying a
     0.05 gap to a 2.65-sigma advantage (mean ≈ 0.994, std ≈ 0.0165 for
     `[1,1,1,1,1,1,1,0.95]` — ~60× inflation) is a large step on a coin flip, and
     works directly against §9's histogram-noise-vs-policy-variance check.
  3. **With the realism floor removed (§4.3), KL is the only brake on degenerate
     text.** Large noisy updates are how the policy gets there.

  **Accepted cost and its fix.** Tight groups now contribute little gradient, so if
  most groups are tight the effective learning rate drops and training can crawl —
  most likely on b2, the cell that most needs to move. The fix is to **raise `lr`,
  not to re-enable scaling**: `lr` scales everything uniformly, whereas std-division
  scales *selectively by group difficulty*, which is the documented difficulty bias
  (Dr. GRPO, Liu et al. 2025). Track mean `|advantage|` per cell (§9); if it collapses
  while reward flatlines, turn up `lr`.

  **Two mechanics to verify before wiring.** (a) The parameter form is unverified on
  the installed TRL — `grpo/requirements-train.txt` already warns the GRPO API moved
  across the 0.12 → 1.9 bump, and in newer TRL `scale_rewards` became a string rather
  than a bool. Run the signature check that file recommends. (b) **If 1.9 supports
  `scale_rewards="batch"` (batch-level rather than per-group std), prefer it** — it
  keeps scale-invariance across runs without the per-group difficulty bias, and is
  strictly better than a plain `False` on this reward.

### 0.2 Empirical baseline — what the Aug 3 run actually showed

Measured from `results/grpo/rft_dataset.jsonl` and `results/grpo/monitor.jsonl`
under the [FT1 §4] per-turn binary. Recorded here because it (i) sets the control
for A3 and (ii) changes which problems we expect the band to fix.

**Warm-start (RFT) coverage — 144 kept examples total:**

| cell | kept |
|---|---|
| b1 Dependency·Warm | 79 |
| b3 Entitlement·Warm | 38 |
| b4 Entitlement·Hot | 27 |
| b2 Dependency·Hot | **0** |
| b5 Neutral·Warm | **0** |
| b6 Neutral·Hot | **0** |

*Caveat: `grpo/run.py:222` allows a `--cells` override and the invocation is not in
`warmstart2.log`, so it is not proven all six were attempted. Config had all six.*

**Per-cell `(engine_pass, delivery_pass)` among high-advantage turns** — the [FT1 §9]
tuple diagnosis, from the audit rows:

| cell | tuples |
|---|---|
| b1 | (1,1) × 19 |
| b3 | (1,1) × 32 |
| b4 | (1,1) × 48 |
| **b5** | **(0,1) × 59, (1,0) × 12 — never (1,1)** |
| **b6** | **(0,1) × 21, (1,0) × 9 — never (1,1)** |

**This is the central empirical result for v2.** The RFT filter is **conjunctive per
turn** (`pass_threshold = 1.0` — engine *and* delivery on the same turn). For b5 and
b6, *both axes are individually reachable* — delivery passes 59 and 21 times, engine
passes 12 and 9 — they simply never co-occur on one turn. A per-turn AND filter reads
that as zero coverage; an arc-level reward scoring the axes independently reads it as
usable signal on both. The zero-coverage cells are therefore substantially an
**artifact of the conjunctive filter**, not proof of an unreachable target — and the
artifact bites hardest exactly where the two axes are anti-correlated, i.e. the Hot ×
internalizing cells the collinearity result predicts.

**Group collapse and mean reward, per cell:**

| cell | groups | mean reward | collapse rate |
|---|---|---|---|
| b1 | 84 | 0.856 | 0.70 |
| b2 | 83 | 0.494 | **0.93** |
| b3 | 85 | 0.584 | 0.51 |
| b4 | 82 | 0.573 | 0.48 |
| b5 | 83 | 0.307 | 0.11 |
| b6 | 83 | 0.238 | 0.41 |

Read with §9's two-number rule: **b2 was the only genuinely stuck cell** (0.93
collapse at mid reward). b5 had ample gradient (0.11) and was merely scoring low.
b1's 0.70 is collapse-by-success (mean 0.856 — mostly all-1.0 groups). Group rewards
took only three distinct values `{0.0, 0.5, 1.0}`; that coarseness is what drives the
collapse rates, and the band's continuous shoulders should cut it independently of
any fidelity gain — a cheap pre-registered prediction (§9, A3).

**Sub-answer rates** (pooled, high-advantage sample, n=20): `e1 0.40, e2 0.50,
q1 0.35, q3 0.55`; `dominant = {self 0.40, others 0.50, neither 0.10}`. Engine-marked
≈ 0.90, delivery-marked plausibly ≈ 0.7. These are from the *high-advantage* sample
and overstate a random arc, but they indicate [BAND §6.4]'s illustrative
`d_floor: 0.15` for delivery is far too low (§5.2).

*Data-hygiene flag:* `subanswer_rates` are byte-identical from step 100 through 300
across both runs. Either the policy stopped moving or the audit re-scored cached
turns — `_CoreBase._cache` (`grpo/reward/backends.py:59`) is keyed on
`(turn, context, cell)` and never cleared. Resolve before treating these as a
rollout estimate. Does not affect the tuple finding.

---

## 1. What is being built

A GRPO training loop that adjusts a **QLoRA adapter** on the Simulator so that,
conditioned on a patient profile + conversation history, the policy produces **arcs**
whose engine and delivery label *distributions* sit inside externally-bracketed
bands. Output: a frozen adapter per cell (or shared adapter + per-cell profile
prompt, §5.4), certified per **[PIPE §4.2]**.

Wall-legal by construction (§2): the reward's only inputs are the profile prompt `P`,
the arc's patient turns, and the context.

---

## 2. NON-NEGOTIABLE CONSTRAINTS

Correctness requirements, not preferences. [BAND CB1–CB7] apply in addition and are
not restated here.

- **C1 — Reward is fidelity only.** The reward reads `(arc_turns, P, context, cell,
  cal)` and NOTHING ELSE. No MUT reply, no SYC/DEP score, no drift signal. Grep-able:
  the reward module imports nothing from the Judge / `[A]` / `[B]` / drift paths.
  Enforced by `grpo/tests/test_c1_import_guard.py`.
- **C2 — Reward source ≠ Judge family.** Unchanged from [FT1].
- **C3 — Reward shape is the arc-level band (AMENDED).** Reward is [BAND]'s
  per-axis band over the arc's label distribution, combined per [BAND D-BAND.1]
  (**geometric mean**, ε-floored — re-synced from the original `average`; §4.1). **[FT1 C3]'s multiplicative realism floor is void** — the realism
  backend is removed (§4.3). No use of the derived 0–3 score.
- **C4 — Graders are deterministic and frozen.** Temperature 0, pinned checkpoint,
  weights never update during the run. **The same frozen backends must supply both
  the calibration and the reward** ([BAND CB1]) — this is what makes instrument
  cancellation work, and it is now a correctness requirement rather than hygiene.
- **C5 — Training allocation by fidelity gap, never drift yield.** Unchanged.
- **C6 — REMOVED.** Both [FT1] pre-flight gates (the §0.1 non-convergence diagnostic
  and the §8.2 stratified delivery validation) were removed by researcher decision.
  Nothing blocks a training run. §10 certification is the only human check.
  Recorded, not hidden: the §8 delivery-confound risk is now unmitigated at entry,
  and it is the same quantity [BAND §7] needs stable for cancellation (§8).
- **C7 — Certification before freeze.** Unchanged from [FT1 §10].
- **C8 — Reward family-disjoint from the finetune base.** Unchanged. Both champions
  non-Qwen; re-check if `base_model` changes.
- **C9 — Calibration frozen before the first step (NEW).** The
  `band_calibration.<grader_version>.yaml` artifact is frozen and hash-logged before
  step 1. Changing any band parameter mid-run invalidates the run ([BAND CB7]).

---

## 3. Components & interfaces

| Component | Role | Notes |
|---|---|---|
| **Policy** | Simulator being trained (`base_model` + QLoRA adapter) | trainable |
| **Reference** | Frozen policy for KL | same base, adapter disabled (PEFT `disable_adapter()`) — no second copy |
| **Reward backends** | Two local axis-specialists → categorical labels | §4; temp 0, frozen (C4); family-disjoint (C8) |
| **Band calibration** | Frozen `BandCalibration` artifact | [BAND §6.4]; hash-logged (C9) |
| **Rollout interlocutor(s)** | **Live** bare, zero-system-prompt model(s) | ≥2 distinct bases for cross-interlocutor spread (§5.3); D2.2 |

**Tooling.** TRL `GRPOTrainer` (current implementation) or Unsloth GRPO.
vLLM-backed generation for the rollout phase. Not PPO (critic doubles memory).

---

## 4. The reward function

### 4.1 Interface

Arc-level. One scalar per arc, standardized within the G-arc group (§7).

```python
def band_reward_arc(
    arc_turns: list[str],   # ordered patient turns of one rollout (T = 20)
    context0: str,          # initial state / history prefix
    P: str,                 # frozen profile prompt for the cell
    cell: str,
    cal: BandCalibration,   # frozen [BAND §6.4] artifact
) -> float:
    """Scalar in [0, 1]. Reads ONLY these inputs (C1)."""
    eng_labels = [engine_backend.label(t, context_upto(arc_turns, t, context0), cell)
                  for t in arc_turns]        # each in {int, ext, neutral}
    del_labels = [delivery_backend.label(t, context_upto(arc_turns, t, context0), cell)
                  for t in arc_turns]        # each in {warm, hot, flat}

    r_eng = axis_reward("engine",   eng_labels, cell, cal)
    r_del = axis_reward("delivery", del_labels, cell, cal)

    # [BAND D-BAND.1], RESOLVED: geometric mean with an eps floor. NOT the
    # average — see below.
    a, b = max(r_eng, EPS), max(r_del, EPS)   # EPS = 0.02
    return sqrt(a * b)
```

`axis_reward`, `band`, and `density_factor` are specified in [BAND §4–§5] and are
not restated. Note there is **no realism multiplier** — see §4.3.

**Axis combination is the geometric mean, not the average (re-sync [BAND
D-BAND.1]).** [BAND] originally defaulted to `0.5·r_eng + 0.5·r_del`. The average
is *indifferent to allocation*: `(0.95, 0.10)`, `(0.10, 0.95)` and `(0.52, 0.52)`
all score ≈ 0.52, so a one-axis-only arc beats an arc doing both passably. The
geometric mean penalises imbalance, puts ~6–9× more gradient on whichever axis is
failing (`∂/∂a √(ab) = ½√(b/a)`), and — measured on a `G = 8` group with engine
pinned at 0.90 and delivery scattered 0.02–0.22 — yields **3× the within-group
advantage spread** (0.0786 vs 0.0263), which inverts D-BAND.1's original
anti-collapse rationale for the average. The `ε = 0.02` floor is the R2 guard the
multiplicative branch needs: without it one dead axis zeroes every arc in the
group and there is no gradient out of inertness. Full argument and the geo-vs-raw-
product reasoning in [BAND §0 D-BAND.1]. **Not configurable** — the constant
lives beside its argument in `grpo/reward/band_reward.py`, so no run can select
the defective average by config drift.

### 4.2 Backends now return categoricals

`.label()` replaces `.score()`. It is a **thin adapter over the existing
decompositions**, not a new grader:

- engine: `EngineDecomposition.engine_direction` → `{internalizing, externalizing,
  neutral}` (`grpo/reward/engine_decompose.py:115`)
- delivery: `DeliveryDecomposition.delivery` → `{hot, warm, flat}`
  (`grpo/reward/delivery_decompose.py:134`)

The per-turn `engine_pass` / `delivery_pass` binaries are derived downstream for the
RFT filter (§6) and the §9 audit, but are **not** the reward.

**Mixed engine turns** (E1 and E2 both fire) resolve via `dominant` — the grader's
own judgment of which the turn most resembles — exactly as
`engine_decompose.py:147` already does. They are not split, not excluded.

**Missing-key counter (NEW).** `backends.py:97` constrains decoding to valid JSON, so
syntactic parse failures should be ~zero. That does **not** guarantee the right keys:
`{"e1": true}` parses cleanly, `labels.get("e1_blames_self")` returns `None`,
`_as_bool` defaults it `False`, and the turn silently reads **neutral/flat** — the
unmarked class that sets the density denominator. Under a binary that was the safe
default; under a density term it is a **downward-only bias on `d`** that is
indistinguishable from the simulator choosing not to express the axis. Given the
observed Qwen guardrail break (mid-arc refusal / language flip), do not assume it
away: increment a counter when *every* expected key is absent, log the rate per step,
and halt above ~5%. If the rate is genuinely zero the telemetry costs nothing.

### 4.3 The realism floor is removed

[FT1 §4] specified `diagnostic * realism_ok`, a per-turn hard `{0,1}` multiplicative
floor. It has been **removed by researcher decision** and
`grpo/reward/fidelity_reward.py` already reflects this. [BAND §3]'s arc-mean realism
variant is likewise not adopted.

Consequence, recorded rather than hidden: **R3 (degenerate-but-on-profile collapse)
is now mitigated by KL and the §9 audit alone.** With the band, that matters more
than under the binary, because [BAND §10]'s band-edge farming failure — an arc that
reaches the plateau via minimal marking, `d` just above `d_floor` and `q` at
`L_design` — has no realism guard standing behind it. §9's high-advantage audit is
the only instrument that will see it.

### 4.4 The no-self-consistency rule is void as stated

[FT1 §4] forbade a line-to-line/self-consistency term because "it fights the pressure
schedule — the profile already encodes scheduled within-arc movement." **There is no
pressure schedule in this design** ([BAND §4.1]: the reward is sequence-agnostic and
uses arc-marginal counts; there is no ladder to segment against). The prohibition may
still be correct, but it needs a fresh justification. Until one exists, treat it as
an open question rather than a rail.

Correspondingly: **do not** add a phase- or position-segmented target to the band.
Fine temporal structure is measured downstream on the MUT **[BS]**, not required of
the Simulator.

---

## 5. Rollout / data generation

### 5.1 State construction
Prompt = `P` (cell profile) + `context` (history prefix). Completion = a **full arc**
of `T = 20` patient turns generated against a live interlocutor, not a single turn.

### 5.2 Group formation — per-arc (re-sync [FT1 §5.2])

At each state, sample **G = 8 full arcs** from the current policy snapshot against a
fixed interlocutor seed; roll cross-interlocutor per §5.3. Score each arc with
`band_reward_arc`; standardize within the 6-arc group for advantages (§7).

Per-turn grouping survives **only** in the RFT warm-start filter (§6), which is a
coverage tool, not the shaped reward.

**Resolution analysis at T = 20 — this constrains `d_floor`.** `q` is quantized in
steps set by `n_marked`, and the band needs enough attainable values inside
`[L_design, U]` for the shape to mean anything.

*Engine*, at `d_eng ≈ 0.30` → `n_marked = 6`, `alpha = 0.5`, `q = (n_on + 0.5)/7`:

| `n_on`/6 | 4 | 5 | 6 |
|---|---|---|---|
| `q_eng` | 0.643 | **0.786** | 0.929 |

With band `[0.70, 0.90]`: 5-of-6 lands inside, 4-of-6 falls below `L_design`, and
6-of-6 (the caricature) sits above `U` on the shoulder. Coarse but the interior
optimum is real. At `d_eng = 0.50` it is cleaner still. **Engine works at T = 20.**

*Delivery*, at [BAND §6.4]'s illustrative `d_floor = 0.15` → `n_marked = 3`,
`q = (n_on + 0.5)/4`:

| `n_on`/3 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `q_del` | 0.125 | 0.375 | **0.625** | **0.875** |

Band `[0.62, 0.88]` contains **both 2-of-3 and 3-of-3**. The perfectly on-profile arc
— the caricature — scores 1.0, identical to the mixed arc. **The anti-caricature
mechanism, which is the entire reason for this redesign, would be inoperative on
delivery.** The threshold is `n_marked ≥ 5–6`; at `n_marked = 4` the ceiling barely
begins to bite (4-of-4 → 0.90, just past `U = 0.88`).

**Resolution:** gate this on measurement, not guesswork. [BAND §6.3]'s
`d_anno(delivery)` over the AnnoMI grader pass, plus a random-sample rollout pass,
gives the real delivery marked-fraction. §0.2's sub-answer rates suggest it is far
above 0.15 (delivery-marked ≈ 0.7 in the high-advantage sample), so T = 20 is
*likely* fine and the 0.15 was a pessimistic illustration. If the measurement
confirms ~0.15, the options are `T = 40` (doubles rollout and grading cost) or
accepting delivery as **density-scored** — [BAND D-BAND.2]'s two-sided band on
`d_del`, with `q_del` kept only as a wide guard against gross direction errors, which
it still does well (all-hot on a warm cell → `q_del = 0.125`, deep on the lower
shoulder).

### 5.3 Cross-interlocutor spread
Build history prefixes by rolling the current policy against **≥2 distinct bare
interlocutors** (zero system prompt), so arcs span a spread of therapist-move
contexts. Training-time analog of cross-interlocutor certification **[PIPE §4.2]**.

### 5.4 Per-cell vs shared adapter
Unchanged from [FT1]. Implement per-cell first (`adapter_mode: per_cell`), evaluate
shared.

### 5.5 Curriculum (anti-stall, for off-manifold cells)
Unchanged in mechanism: off-manifold cells start near-manifold and anneal toward the
target across training; what anneals is the **target**, never the graders (C4).

**Amended targeting.** [FT1] enabled the curriculum for `[b1, b2]`. Per §0.2, b1 is
not the struggling cell (mean 0.856) and **b2 is the only genuinely stuck one**
(0.93 collapse). Re-point the curriculum at b2 and re-evaluate b5/b6 after the first
banded run, since the tuple evidence suggests the band may fix them without
curriculum help.

---

## 6. Warm-start: reward-filtered SFT (RFT) before GRPO

Warm-start's job is unchanged: raise the marked **base rate** so GRPO groups are
non-empty. Keep the **per-turn** RFT filter (`engine_pass`/`delivery_pass` derived
from `.label()`), because RFT reinforces what the base already samples and cannot
reach an off-manifold *distribution* — distributional shaping is GRPO's job
([BAND §9]).

### 6.1 The conjunctive filter is the known defect

Per §0.2, `pass_threshold = 1.0` requires both axes to pass **on the same turn**, and
that alone explains b5's and b6's zero coverage: both axes are individually
reachable, they never co-occur. Two consequences:

1. **Do not read zero RFT coverage as an unreachable target.** It is a conjunction
   failure, and it is systematically biased against the cells where the axes are
   anti-correlated — the Hot × internalizing cells.
2. **Consider relaxing the filter to disjunctive-with-partial-credit** for coverage
   purposes (keep turns passing *either* axis, weighted), since the downstream reward
   no longer requires joint per-turn satisfaction. This is a coverage tool feeding a
   distributional objective; there is no longer a principled reason for it to be
   stricter than the objective.

### 6.2 Log rejects with their tuple (instrumentation gap)

`collect_rft_dataset` persists only *kept* turns and only the summed scalar
(`RFTExample.reward`, always 1.0). Rejected turns are discarded, so the per-cell
`(engine_pass, delivery_pass)` diagnosis has to be recovered from the high-advantage
audit rows — a biased sample. Persist rejects with their tuple so coverage can be
diagnosed directly: delivery-zero cells need near-manifold relaxation on the delivery
target; engine-zero cells can lower the pass threshold.

### 6.3 Density reads are invalid on uncovered cells

`d_floor` (§5.2) will read as unmet on any cell RFT left uncovered. **Fix coverage
before reading density.**

### 6.4 The RFT set is still the §8 validation distribution

Unchanged from [FT1 §6] — it is the first real (non-prompt-opt) rollout distribution
the delivery champion scores in anger.

### 6.5 Offline calibration stage (NEW — must precede training)

Runs once per grader-backend version. Full procedure in [BAND §6]; the decisions
settled for this run:

- **Corpus.** AnnoMI (`AnnoMI/AnnoMI-simple.csv`) — 4,817 client utterances across
  133 sessions; median 23 client turns/session (p25 13, p75 39, max 299); 1,596 are
  backchannels leaving 3,221 substantive; only 8.7% are `mi_quality = low`.
- **Hand-label sample: ~500 client turns drawn as WHOLE CONVERSATIONS**, in 5 batches
  of 100. Two raters. The sampling unit is the **session, not the turn**: pick a
  session, label *every* substantive client turn in it. Fewer distinct patients is an
  accepted cost.

  **Generated** by `label_tasks/annomi/generate_annomi_label_task.py` (seed 42,
  reproducible): at ≥8 substantive client turns per session, **109 of 133 sessions
  are eligible** (93 high / 16 low), and the draw takes **22 full conversations =
  525 labelable turns**, split into 5 batches of 99–107 with no conversation crossing
  a batch boundary. Per-session labelable counts range 8–75.

  The session draw is **stratified on `mi_quality`** (a `--low-quota-turns` reserve),
  yielding 7 low-quality sessions / 140 turns — **26.7% of the sample versus 8.7% of
  the corpus**. That over-representation is deliberate and harmless: these labels
  feed only the κ, where spanning the range of text the grader will meet is what
  matters, and the band edges come from the grader pass over all 4,817 turns, so no
  bias reaches the artifact.

  Backchannel filtering is **length + closed vocabulary** (§ the generator's
  `BACKCHANNEL_VOCAB`) — the length rule alone lets pure filler through
  (`"Oh, oh, yeah, the—Yeah"` is 5 tokens). Corpus-wide drops, logged per §6.5:
  1,587 by length (≤3 tokens), 9 by all-filler vocabulary, leaving **3,221
  substantive client turns**. The vocabulary is deliberately narrow: "carries no
  engine/delivery signal" is nearly the definition of neutral/flat, which is a label
  the denominator of `d_anno` needs, so an aggressive filter would inflate `d_anno`
  rather than clean it.

  Three reasons this is better than scattered turns, not merely different:

  1. **It makes the human labels support a per-patient `q`.** The bracket is built
     from per-patient `q` (below). With whole conversations the humans produce a
     per-patient `q` too, so the hand labels can validate *the actual quantity the
     bracket is made of*, not just per-turn agreement. Scattered turns can only give
     per-turn κ.
  2. **It matches how the grader sees the data.** `.label(turn, context, cell)` reads
     the turn *in context*, and both constructs are context-dependent — delivery is
     defined as directed "toward the listener," which needs the therapist's preceding
     turn to judge. Labeling isolated turns would have put the humans and the grader
     on different inputs, quietly inflating disagreement.
  3. **Raters get the arc.** Engine direction is often only legible across a few turns.

  *Precision note:* turns within a conversation are correlated, so ~500 clustered
  turns carry less information than 500 independent ones and the κ CI is wider than
  `1/√500` suggests. **Bootstrap the κ CI by resampling sessions, not turns.**
  Acceptable because the κ is **report-only** here (C6 removed the gate that would
  have thresholded it, and the bounds come from the grader pass); revisit the sample
  size if the κ is ever re-promoted to a gate.

- **No AnnoMI label task exists yet.** `label_tasks/batch01–03` are SWAY-rollout
  labels, unrelated. `sway_harness/validate_judge.py` already has an AnnoMI loader,
  conversation reconstruction, and κ machinery — reuse them, with two cautions:
  it is built for **Judge**/MITI *therapist* validation (drift-side), so
  `annomi_calibrate.py` must not import its judge-scoring paths or the C1 grep guard
  will fail; and its `ANNOMI_PATH` (line 39) points at `sway_harness/AnnoMI` while
  the data is at repo-root `AnnoMI/`.
- **Standalone codebook required.** The second rater will not have SWAY context, so
  the labeling guide must be self-contained — particularly the delivery distinction
  that grievance about the employer is **not** hot
  (`grpo/reward/delivery_decompose.py:29`), which is the confound that broke delivery
  κ previously. A naive rater agreeing is a *stronger* validity result than a trained
  one agreeing; the guide has to earn it.
- **Band edges are derived per-patient, in grader space.** Per [BAND CB1] the bounds
  come from **grader** labels, not human ones — which decouples the two sample sizes:
  **humans label ~500 turns (26 conversations) for κ; the frozen grader labels all
  4,817 client turns for the bracket.** Grader turns are near-free (temp 0, local,
  unattended), so the edges get the full corpus.

  **Procedure.** For each direction `d` on each axis:

  1. **Per patient, not pooled.** Compute `q_d` for each session over *that session's*
     marked turns. Each session is one real client — the closest thing AnnoMI has to a
     profile.
  2. **Eligibility.** A session enters the distribution only with **≥8 marked turns**
     on that axis. Feasibility at that threshold:

     | axis | eligible sessions (of 133) |
     |---|---|
     | engine, marked-rate ≈ 0.5 | 63 |
     | engine, marked-rate ≈ 0.9 | 103 |
     | delivery, marked-rate ≈ 0.3 | 33 |
     | delivery, marked-rate ≈ 0.15 | **11** |

  3. **Shrink** each session's `q_d` toward the pooled mean (empirical Bayes) before
     taking any percentile, so short sessions don't drive the tails.
  4. **Read both edges as percentiles of that distribution.**
     `L_design(d) = P_lo`-th percentile, `U(d) = P_hi`-th percentile. `P_lo ≈ 70–75`,
     `P_hi ≈ 90–95`, both **disclosed**.

  **Why `P_lo` sits high.** The floor should be *the least directional patients who
  still present in direction `d`* — the low end of the on-direction group. Taking the
  top quartile by `q_d` and reading their low end is approximately the 75th percentile
  of everyone, which is why a moderately-high percentile of the full distribution is
  the right operationalisation. It is deliberately **not** an extreme percentile: the
  SWAY profiles are strong archetypes, not median people, but reading the 95th
  percentile as a floor would be winner's-curse territory.

  **`L_ext` and `L_design` are now the same number (D-BAND.3 collapsed).** [BAND]
  splits them — `L_ext` measured, `L_design` chosen inside `[L_ext, U]`. Under this
  derivation `L_ext` already contains a disclosed choice (which percentile), so a
  second disclosed knob above it does the same job twice. **Use one lower edge,
  `L_design = P_lo`-th percentile.** Two disclosed percentiles `(P_lo, P_hi)` replace
  three hand-set numbers. To pull central tendency higher, raise `P_lo`.

  **The epistemic claim changes, and [BAND §6.3] must be rewritten, not extended.**
  Under the pooled scheme `L_ext` was a genuine *bound*: the dilution argument said a
  real single-direction patient must lie above the mixture share. Per-patient
  percentiles are not a bound — they are a **chosen position inside a real
  distribution**: "the simulated patient should be at least as directional as the
  `P_lo`-th percentile real patient of that direction." Stronger and more useful, but
  a different claim, so [BAND §6.3]'s "why this is a lower bound, not a point" section
  is superseded rather than amended.

  **What [BAND §6.3] must additionally say — the three-way sorting distinction.** As
  written it forbids "post-hoc sorting on the very axis you want to measure." That
  prohibition is correct but too coarse, and stating only "session grouping is fine"
  would license the biased version:

  | move | verdict |
  |---|---|
  | sort **turns** by their own label, then measure that label | circular — the answer is guaranteed, useless |
  | group by **session** (exogenous, fixed before labeling), read the distribution | legitimate |
  | take an extreme **percentile** of that distribution as a parameter | legitimate but upward-biased — needs steps 2–3 above |

  The permission and the two corrections must travel together. [BAND CB2] is
  untouched: this is human data, never simulator rollouts.

  **Asymmetry, expected.** Engine is comfortable (63–103 eligible sessions). **Delivery
  is thin** — 11–33 sessions, so its percentiles rest on a small sample and its edges
  stay partly disclosed. Expect D2.4's flag to fire on delivery.
- **Uninformative-bracket guard (D2.4).** Largely dissolved by the per-patient
  derivation: `U − L_design` is now the spread between two percentiles of a real
  distribution, i.e. genuine between-patient heterogeneity, rather than an artifact of
  a diluted pooled share. The guard stays as a **detector for where that derivation
  failed** — chiefly delivery, whose percentiles rest on 11–33 sessions.

  **Warn and disclose, do not halt.** If `U − L_design > 0.35`, or if fewer than ~25
  sessions were eligible for that axis/direction, the calibration script stamps the
  entry `bracket_informative: false` in the frozen artifact. Nothing blocks, but the
  run log cannot later claim the value was data-derived when the sample could not
  support it.
- **Transfer caveat stands** ([BAND §6.5]). AnnoMI is MI counselling, not layoff
  support. If human layoff-domain transcripts become available, hand-label those
  instead. **Never** substitute the Simulator's own material (CB2).

---

## 7. GRPO training loop

Per step:

1. **Sample.** For each state in the batch, generate **G = 8 arcs of T = 20 turns**
   from the frozen policy snapshot against the live interlocutor (vLLM generation).
2. **Score.** `band_reward_arc` (§4) on each arc. Temp-0 backends (C4), frozen
   calibration (C9).
3. **Advantage.** Standardize within each group:
   `A_i = r_i − mean(r_group)` — mean-centring only, since `scale_rewards = False`
   (D2.5). Guard `std == 0` groups. Track mean `|advantage|` per cell; if it collapses
   while reward flatlines, raise `lr` rather than re-enabling scaling (D2.5).
4. **Update.** Policy-gradient step on the adapter only: advantage-weighted log-prob
   **minus** `beta * KL(policy ‖ reference)`, PPO-style ratio clipping. Reference =
   base with adapter disabled. Gradient lands on LoRA only.
5. **Repeat.**

**KL is now doing more work.** With the realism floor removed (§4.3), `beta` is the
principal brake on degenerate text as well as the off-manifold-reach enabler. Tune it
as a first-class knob.

**Zero-std groups are no longer uniformly bad** — see §9.

---

## 8. Delivery-champion decomposition & the cancellation tie

### 8.1 Decomposition (standing, by construction)
Unchanged from [FT1 §8.1]. The delivery champion asks **Q1** (hostility toward the
interlocutor) and **Q3** (closeness-pulling toward the interlocutor) separately;
`hot = Q1` regardless of employer-directed grievance. A warm target requires Q3 and
not Q1 — a flat turn does **not** satisfy warm
(`grpo/reward/delivery_decompose.py:193`).

### 8.2 The validation gate is removed, but the quantity is now load-bearing

[FT1 C6-ii] required a stratified grievance-vs-hostility κ ≥ 0.80 (CI lower bound)
before training. **That gate is removed** (C6). The underlying risk did not go away,
and under the band it acquires a second role:

[BAND §7]'s instrument-cancellation argument is what licenses using an imperfect
grader — a systematic bias `Δ` inflates both the AnnoMI-measured target and the
rollout measurement by roughly the same amount and subtracts out. That holds **only
to first order**, and only insofar as `Δ` is the same on human and simulator text. If
the Simulator emits markedly more employer-grievance than AnnoMI clients do, the
grievance→hot confound fires harder on rollouts than on the target,
`Δ_rollout > Δ_target`, and the residual pushes the policy to a wrong setpoint.

So the grievance-density comparison is no longer a pre-flight gate; it is the
**ongoing empirical check on whether the reward's anchor is stable**. It moves to §9
as monitoring. If the delivery champion is badly confounded, `q_del` and therefore
`L_design(delivery)` are both untrustworthy — harden or swap the backend before the
delivery band means anything.

### 8.3 Authored-pair sanity check
Retained as an optional weak smoke test (`grpo/gates/authored_pairs_smoketest.py`).
Not a blocker.

---

## 9. Online monitoring (during training)

- **Per-cell rate telemetry (NEW, [BAND §10]).** Log the rollout distribution of
  `q_eng`, `q_del`, `d_eng`, `d_del` per step. The band gives no within-plateau
  gradient by design, so `q` settling *somewhere* in `[L_design, U]` is correct.
  Flag if `q` parks **below** `L_design` (ordering breaking) or rides `U`
  (caricature pressure through the ceiling).
- **Group collapse — read two numbers, not one (AMENDED).** `group_collapse_rate`
  is no longer single-signed. The plateau makes exact ties *more* likely precisely
  when the policy is doing well: any `q ∈ [L_design, U]` scores exactly 1.0 and
  `density_factor` saturates, so a well-behaved arc scores exactly 1.0 on both axes.
  Tie pressure falls as G rises, so G = 8 is meaningfully safer here than G = 6
  would have been, though still tighter than the G = 10 baselines below. Pair
  collapse with mean
  reward:

  | signature | reading |
  |---|---|
  | high collapse + high mean | converged into the plateau — healthy, expected |
  | high collapse + mid/low mean | **stuck, no gradient** — b2's exact signature (0.93 @ 0.494) |

  Baselines to beat, from §0.2: b1 0.70, b2 0.93, b3 0.51, b4 0.48, b5 0.11, b6 0.41.
- **Histogram-noise vs policy-variance.** Confirm reward spread tracks *behavioral*
  differences between arcs, not `1/n_marked` quantization jitter. If arcs are short
  and spread is jitter-dominated, raise `alpha` or `T` before trusting the signal.
- **Cancellation drift (NEW, from §8.2).** Periodically recompute rollout grievance
  density and compare to the AnnoMI figure. A widening gap means `Δ` is
  destabilizing and the delivery band is losing its anchor.
- **Band-edge farming (NEW).** With realism removed, watch for arcs reaching the
  plateau via minimal marking (`d` just above `d_floor`, `q` at `L_design`). If
  frequent, raise `d_floor` / `L_design` or promote delivery density to a two-sided
  band. **This is the failure with no automatic guard** (§4.3).
- **Missing-key rate (NEW, §4.2).** Halt above ~5%.
- **High-advantage turn audit.** Every N steps, sample top-advantage arcs and
  human-spot-check whether reward rose while human-judged fidelity stayed flat. Log
  per-axis and per-sub-question. With the realism floor gone this is the primary
  degeneracy detector, not a secondary one. **Clear the backend cache before each
  audit** (§0.2 data-hygiene flag).
- **Grievance→hot watch.** Track the Q2-yes / Q1-no rate among high-advantage turns.
- **Reward vs fidelity divergence.** Mean reward against a held-out human-validated
  fidelity estimate; a widening gap = hacking in progress.

---

## 10. Certification & freeze (C7)

Unchanged from [FT1 §10]. Human, per finetune iteration, on held-out authored detail
generated on a **fresh authored instance** the loop never saw, against a **second
bare interlocutor** not used in training and not either reward champion. Frozen
rubric + fixed re-scored gold subset for annotator-drift control (R6). Freeze at
deployment quant. Version the adapter recording base checkpoint, quant, both champion
identities, **the band-calibration hash (C9)**, training seed, and the iteration's
certification κ.

With C6 removed, this is the **only** human check in the pipeline. Weight it
accordingly.

---

## 11. Config surface

```yaml
base_model: "Qwen/Qwen2.5-14B-Instruct"   # D0.1; champions must be non-Qwen (C8)
quant: "nf4"
lora:
  r: 16
  alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  dropout: 0.05
grpo:
  group_size_G: 8            # D2.1 — arcs per group (was G turns, per-turn grouping)
  arc_length_T: 20           # D2.1 — patient turns per arc
  scale_rewards: false       # D2.5 — mean-centre only; verify form on TRL 1.9,
                             #        prefer "batch" if supported. See §7.
  kl_beta: 0.04              # now the PRINCIPAL degeneracy brake (§4.3) — tune first
  clip_ratio: 0.2
  lr: 1.0e-5
  max_prompt_tokens: 1536
  max_completion_tokens: 256 # one patient turn
  temperature_rollout: 0.8
  grad_checkpointing: true
reward:
  shape: "arc_band"                                 # [BAND]; supersedes per-turn binary
  backend_engine: "engine_champion"                 # .label() -> {int, ext, neutral}
  backend_delivery: "delivery_champion_decomposed"  # .label() -> {warm, hot, flat}
  # backend_realism: REMOVED (§4.3) — R3 now on KL + §9 audit alone
  grader_temperature: 0.0                           # C4
  calibration_path: "calibration/band_calibration.<grader_version>.yaml"  # C9, frozen
  missing_key_halt_rate: 0.05                       # §4.2
warmstart:
  rft_epochs: 1
  rft_filter: "diagnostic_pass"     # per-turn, conjunctive — see §6.1 before reusing
  arcs_per_cell: 30
  log_rejects_with_tuple: true      # §6.2
curriculum:
  enabled: true
  enabled_cells: [b2]               # §5.5 — b2 is the stuck cell, not b1
  schedule: "near_manifold -> target"
  anneal_frac: 0.4
adapter_mode: "per_cell"
cells: [b1, b2, b3, b4, b5, b6]
monitoring:
  audit_every_n_steps: 50
  audit_sample_size: 20
  clear_backend_cache_before_audit: true   # §0.2 hygiene flag
  subanswer_rates: true
  rate_telemetry: true                     # §9 — q_eng/q_del/d_eng/d_del per step
  band_edge_farming_watch: true            # §9 — no automatic guard behind it
  cancellation_drift: true                 # §8.2 -> §9
  log_path: "results/grpo/monitor.jsonl"
certification:
  mode: "human_per_iteration"              # §10 — the ONLY human check (C6 removed)
  frozen_rubric: true
  fixed_gold_subset: true
  gold_size: 20
  gold_drift_bar: 0.80
```

---

## 12. Hardware / VRAM plan & risks (single RTX 5090, 32 GB, sm_120)

**Per-step cost at T = 20, G = 8, live interlocutor** (arcs × turns = 8 × 20 = 160
patient turns per step):

| item | count per step |
|---|---|
| policy generations (GPU, trainable) | 160 |
| interlocutor generations (Ollama) | 160 |
| grader calls (Ollama; 2 axes × 160 turns) | 320 |
| **total Ollama requests** | **480** |

At G = 10 this was 600 Ollama requests and 200 policy generations — G = 8 cuts step
cost ~20%. Separately, `grpo/train/grpo_loop.py:215` sets
`per_device_train_batch_size = group_size_G`, so G = 8 also shrinks the training
batch 10 → 8 and with it training-side VRAM, a partial mitigation for the
rollout/grader contention already observed on this box.

**This is plausibly the binding constraint on the whole design, more than the band
mathematics.** For a step to complete in ~60s the graders must sustain ~5.3 calls/sec
concurrently while training holds VRAM. `_CoreBase._cache` is keyed on
`(turn, context, cell)` and will not hit across distinct rollouts, so there is no
relief there. **Measure real grader latency before committing to a step budget**; at
1–2s serialized this is 6–12 minutes per step and a long run is not happening.

Mitigations: vLLM-backed rollout generation (paged KV); cap `max_prompt_tokens` /
`max_completion_tokens`; reference via disabled adapter (never a second base copy);
concurrent grader requests; if co-residency is too tight, phase-split (graders
resident during scoring, unloaded for the training step).

**Risks:**

- **R1 — Reward hacking on delivery.** Mitigated by §8.1 decomposition (standing) +
  §9 grievance→hot watch + KL. **The §8.2 pre-flight gate is gone (C6)**, so this is
  less mitigated at entry than in [FT1], and it now also threatens the reward's
  calibration anchor (§8.2).
- **R2 — All-fail / all-tie groups.** Mitigated by warm-start + curriculum + the
  band's continuous shoulders. Note the band *helps* struggling cells and *hurts*
  succeeding ones on this metric (§9).
- **R3 — Degenerate on-profile collapse.** **Mitigated by KL and the §9 audit only**
  — the realism floor is removed (§4.3). Weakest-guarded risk in v2.
- **R4 — Correlated blind spots.** Two distinct champions keep engine/delivery blind
  spots axis-local. Residual: delivery single-covered.
- **R5 — VRAM overflow during rollout.** Mitigated per above; fallback is the 8B base
  (re-run C8).
- **R6 — Annotator drift across iterations.** Frozen rubric + fixed gold subset (§10).
- **R7 — Calibration circularity (NEW).** A band parameter read from the Simulator's
  own rollouts is the ratchet failure [BAND CB2] exists to prevent. Enforced by
  freezing the artifact before step 1 (C9) and by sourcing bounds only from AnnoMI
  grader-space measurement (§6.5).
- **R8 — Cancellation residual (NEW).** [BAND §7] holds only to first order and only
  if `Δ` is stable across human and simulator text. Monitored per §9; there is no
  hard guard.

---

## 13. Deliverables & acceptance criteria

**Deliverables:**
- `reward/band_reward.py` — §4 / [BAND §3–§5]: `band_reward_arc`, `axis_reward`,
  `band`, `density_factor`; C1 import guard; C8 assertion; C9 calibration-hash check
  and load-time asserts ([BAND CB3/CB4] plus `d_floor > 0`, `s_lo > 0`, `s_hi > 0`,
  and `0 ≤ d_lo < d_hi < 1` for `density_low` — the current formulas divide by these
  at call time rather than failing at load).
- `reward/backends.py` — `.label()` categorical adapters (§4.2) + missing-key counter.
- `calibration/annomi_calibrate.py` — §6.5 / [BAND §6]: sample → hand-label ingest →
  grader-space measure over all 4,817 client turns → per-session bracket derivation →
  emit frozen `band_calibration.<grader_version>.yaml`.
- `calibration/band_calibration.<ver>.yaml` — frozen artifact, hash-logged (C9).
- `data/rollout.py` — §5: arc construction at T = 20, per-arc grouping,
  cross-interlocutor, live interlocutor.
- `train/rft_warmstart.py` — §6, amended to log rejects with their tuple (§6.2).
- `train/grpo_loop.py` — §7, G = 8, `scale_rewards = False` (D2.5).
- `monitor/online_audit.py` — §9, incl. rate telemetry, band-edge farming watch,
  cancellation drift, cache clearing.
- `cert/certify_and_freeze.py` — §10, unchanged plus calibration-hash in the version
  record.
- Amendment patch to **[BAND §6.3]** permitting the per-session derivation (§6.5).

**Acceptance:**
- **A1** — Reward module passes the C1 grep guard and a unit test proving it reads
  only `(arc_turns, context, P, cell, cal)`. C8 assertion passes.
- **A2** — Load-time asserts fire on a bad artifact: `L_design ≥ U`, `U ≥ 1.0`,
  `L_design ≤ 0`, `d_lo ≤ 0` for `density_low` cells (D2.3), and the zero-divisor
  cases above. (`L_design < L_ext` is no longer assertable — they are one parameter,
  §6.5.)
- **A3** — Anti-caricature: an all-on-profile arc (`q = 1.0`) scores **strictly less**
  than an arc with `q ∈ [L_design, U]`, on both axes. *Must be tested at the
  configured `T` and `d_floor`, not in the abstract* — §5.2 shows the property can
  silently fail at low `n_marked`.
- **A4** — Density: an arc marked on `< d_floor` of turns with perfect `q` scores
  strictly less than one clearing `d_floor` at the same `q`.
- **A5** — Cancellation wiring: the artifact's `L_design` is produced by the **same**
  backend object injected as the reward grader (C4/[CB1]); a deliberately biased stub
  shifts target and rollout together and the reward's argmax stays at the true rate.
- **A6** — Smoothing: reward spread across G short arcs with identical policy
  behavior (jitter only) shrinks as `alpha` rises.
- **A7** — On ≥1 backbone cell, GRPO under the band raises held-out fidelity vs the
  per-turn-binary baseline without `q` collapsing to the caricature edge.
- **A8 (pre-registered, cheap)** — `group_collapse_rate` per cell falls versus the
  §0.2 baselines (b1 0.70, b2 0.93, b3 0.51, b4 0.48, b5 0.11, b6 0.41) on the
  struggling cells, read with §9's two-number rule. b1 rising is expected and is not
  a failure.
- **A9** — Full config + seeds + both champion identities + **calibration artifact
  hash** + certification κ logged for reproduction.

---

## 14. Decision ledger

**Settled in v2.**
- Arc-level distributional band reward replaces the per-turn monotone binary.
- **Axis combination = geometric mean with an ε = 0.02 floor** ([BAND D-BAND.1],
  resolved from open/`average` on measured advantage spread; §4.1).
- Realism floor removed; R3 on KL + §9 audit alone.
- `T = 20`, `G = 8`, live interlocutor.
- Per-arc grouping; per-turn survives only in the RFT filter.
- Both C6 pre-flight gates removed; §10 certification is the only human check.
- All six backbone cells in the pilot, b6 included — see the note below.
- AnnoMI calibration: N = 500 hand-labeled in 5 batches of 100 for κ (report-only);
  **all 4,817 client turns grader-labeled** for the bracket; per-session
  distribution, not the pooled share.
- Mixed engine turns resolve via `dominant`; no splitting, no exclusion.
- No `unparsed` label class; a missing-key counter instead.
- Curriculum re-pointed from `[b1, b2]` to `[b2]`.
- **Neutral engine = ABSENT engine** (D2.3): `mode: density_low` for b5/b6/p2/p3,
  with `d_lo > 0` strictly (≈0.05) so inertness does not score 1.0.
- **Uninformative-bracket guard warns and discloses, does not halt** (D2.4):
  `U − L_design > 0.35`, or <~25 eligible sessions, stamps `bracket_informative:
  false` in the frozen artifact.
- **Band edges are per-patient percentiles in grader space** (§6.5): `L_design` and
  `U` are the `P_lo`/`P_hi` percentiles of the per-session `q_d` distribution;
  `L_ext` and `L_design` collapse to one parameter.
- **Hand-label sampling unit is the whole conversation**, not the turn: ~26 full
  sessions ≈ 518 substantive client turns, session-stratified for `mi_quality`, κ CI
  bootstrapped by resampling sessions.

**Open — no remaining decisions; the rest is measurement.**
- Delivery `q`-band resolution at T = 20 — gated on measuring `d_del` (§5.2).
- [BAND D-BAND.3] — the disclosed percentiles `P_lo` (~70–75) and `P_hi` (~90–95),
  and the sigma values per cell. Provisional until the grader pass lands (§6.5).
- Whether to source the bracket from human **layoff-domain** transcripts instead of
  AnnoMI if obtainable — strictly better on transfer, gated on availability.

**Logged prediction — b6.** b6 (Neutral·Hot) asks for hot delivery *and* low engine
expression, while hot delivery and externalizing engine are collinear at P ≈ 0.99. If
that collinearity is a property of the text, b6 parks at low reward and the band
cannot move it. If the band *does* move it, that is a real finding about the
collinearity being grader-side or breakable. Either outcome is informative, which is
the argument for keeping b6 in. Same watch, milder, on b2.

**Out of scope (owned elsewhere).**
- Band mathematics, calibration procedure, CB1–CB7 — **[BAND]**.
- Run-time gate, Judge, drift scoring, SYC/DEP — **[PIPE][BS][A][B]**; untouched.
