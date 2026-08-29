# SWAY — instrument findings, 2026-08-28/29

Working record of measurements made against the fidelity instrument (frozen prompt
+ champion graders + whatever consumes them), the code built from
`SWAY_ENGINE_DELIVERY_RATE_PROFILE_SPEC.md` **[RATE]**, and the review of
`specs/ebm_spec.md` **[EBM]**.

Written so the numbers survive the conversation that produced them. Findings that
were later **retracted are marked and kept**, not deleted — two of them were acted
on before being corrected, and knowing which is which matters more than a tidy
document.

Commits (all on `origin/main`, pushed from the fedora checkout):

| commit | contents |
|---|---|
| `3e5b3ae` | rate-profile reward, derivation, frozen v1 artifact, A1–A7 tests |
| `d480579` | `rate_calibration.v2.yaml` — derived from the real AnnoMI grader cache |
| `d89c54b` | hurdle delivery calibration + the `band(0)` on-direction assert |
| `3605acc` | `tools/champion_validation.py` + the context-conditioning defect |
| `fcb2825` | delivery kappa decomposed by sub-question against the human ceiling |

**Repo state warning.** The Mac checkout is at `3e5b3ae` and cannot fast-forward:
`results/build/*_prompt.txt` and `results/build_artifacts/*/progress.txt` carry
uncommitted local edits that predate this work, and fedora's history touches the
same files. Everything after `3e5b3ae` was committed on fedora. Resolve the
divergence before working on the Mac again.

---

## 1. THE BIG ONE — the graders are conditioned differently on the two sides

**The champion sees a different context at calibration time than at scoring time,
and on the engine axis this changes the measured rate by 2.4x.**

* `annomi_calibrate.collect_turns` (calibration) and the batch03 hand-label sheets
  both supply **the interlocutor's preceding turn** as context.
* `band_reward.context_upto` (the reward) supplies **the patient's own preceding
  turns**, concatenated and growing with turn index.

Same champion checkpoint, same transcripts (`results/build_artifacts/{b1,b3}/iter_0/`),
engine marked rate `d`:

| arc | patient's-own-turns | prev-interlocutor |
|---|---|---|
| b1/transcript_0 | 100% | 15% |
| b1/transcript_1 | 95% | 45% |
| b1/transcript_2 | 100% | 20% |
| b3/transcript_0 | 100% | 40% |
| b3/transcript_1 | 100% | 55% |
| b3/transcript_2 | 100% | 75% |
| **mean** | **99.2%** | **41.7%** |

Delivery is essentially unaffected (85/85, 60/65, 70/60, 50/50, 85/85, 20/20).
The effect is engine-specific: engine asks whether the turn attributes cause, and a
growing run of blame-laden patient turns primes it toward yes. Delivery asks
whether anything is pointed *at the listener*, which anchors on the
grader-as-listener regardless of what precedes.

### Why it matters beyond this pipeline

1. **Turn-comparability.** `context_upto` is a *growing-width* conditioning, so
   grader base rate can drift with turn index. Any per-turn quantity then carries a
   monotone artifact, which is disqualifying for SWAY's central claim of per-turn
   temporal localization — independent of any calibration question. **Fixed-width
   conditioning is the requirement; matching the calibration corpus is a bonus that
   falls out of it.**
2. **It breaks bias cancellation.** [RATE] C3 licenses an imperfect grader on the
   argument that a systematic bias inflates target and score alike and cancels.
   That needs the *same* bias on both sides. Different conditioning is a different
   bias, so nothing cancels.
3. **C4/C7 pinned one leg of a three-legged thing.** The constraint said "one
   estimator." It has to be **one estimator, one conditioning, one grader
   checkpoint at fixed temperature.** (Amendment being written into [EBM] C4.)
4. **It was self-inflicted, not structural.** `context_upto`'s docstring attributes
   the exclusion to C1, but C1 forbids MUT *drift signal* — Judge outputs, SYC/DEP
   scores — not the conversation. [FT §4] defines `context` as history up to the
   turn; [PIPE §5] explicitly permits the fixed reference interlocutor to cross.
   This was an implementer's over-tightening of a wall that never required it.

### Resolved

**Prev-interlocutor context for the controller**, pinned in
`configs/ebm_controller.yaml` **and** made a required argument with no default on
the champion call signature — a default is the mechanism that lets two call sites
diverge silently, and a config key alone does not prevent that.

The frozen GRPO reward stays as-is and its mis-conditioning is written up as part
of the [FT] negative result. Rationale for the split: a gradient-bearing reward
reading the interlocutor's turn opens a Goodhart channel (the policy could learn to
elicit interlocutor turns that make its own read on-profile). Weak in practice —
per-turn grouping means no gradient crosses the turn boundary to carry the
elicitation credit — but it does not apply at all to a rejection sampler with no
gradient.

### Still to test (next action)

Three-way comparison on the same turns, **per axis**: bare / prev-interlocutor /
patient-own. Bare-turn grading is a live candidate for **engine specifically** —
engine is a property of the turn alone and is wildly context-sensitive, which is
the profile of context acting as a *prime* rather than as information. Bare would
kill both the turn-index confound and the MUT-dependence below in one move.

The specific question: engine swings **15–75%** across arcs under prev-interlocutor.
Bare discriminates whether that is the patient genuinely responding to the therapist
or the grader being moved by whatever the therapist said. **A reachability number
computed under an unstable conditioning is not a reachability number**, so this
resolves before any engine `p̂` from [EBM] §6 is trusted. Per-axis conventions are
acceptable if they fall out of the test; pin both. Cost includes re-scoring AnnoMI's
3,221 turns under whichever convention wins.

### Consequence for [EBM] that has no clean answer yet

At run time the interlocutor **is the MUT**, so a controller conditioning on the
previous interlocutor turn makes the delivered dose **MUT-dependent**: τ = 0.6
against one model is not τ = 0.6 against another. Not a wall breach — no scorer
output crosses — but confounding, and it undercuts the fixed-stimulus premise that
makes cross-MUT comparison mean anything. C8 (regress on realized `q`, not
requested τ) only partly absorbs it: dose varying *systematically with the thing
being measured* is not the same problem as dose varying randomly.

**Add to §6/§8: run one cell against each MUT and report realized `d` and `q` by
MUT.** Small spread → note and move on. Large spread → dose is a per-(cell × MUT)
quantity and every cross-MUT claim needs it as a covariate. Delivery's context-
insensitivity is the reason for optimism here: the axis most likely to be the only
controllable one is the least exposed.

---

## 2. RETRACTED — "engine is dead / the policy is parked at the caricature"

**What was claimed:** engine `d` = 1.00 on simulator arcs; the policy sits at the
caricature; the rate-profile reward has no gradient leading back from it.

**Why it was wrong:** it was measured through `context_upto`. Under the calibration
conditioning engine `d` is ~42% against AnnoMI's 26.5% — a **1.6x** discrepancy,
not 4x. Independently confirmed by batch03: champion *and* human annotators both
put b1 at 36–44% on the very same transcripts.

**What survives:** the dead-zone arithmetic is correct *given* `p = 1.0`, and it is
worth keeping as a property of the reward shape:

* `combine_axes` clamps each axis to `eps = 0.02`, so every engine band value below
  0.02 is indistinguishable. Arc reward is **bit-identical** from 20/20 down to
  10/20 on-direction turns.
* The caricature sits **11.6σ** past the band's upper edge at `s = 0.06`; movement
  only begins at 9/20.
* Root cause of the σ figure: [RATE] §5.1 carried `s = 0.06` over from the `q`
  scale, where the caricature sat ~4σ out, to the rate scale, where the band-to-
  caricature distance is 0.70 instead of 0.23. §2.1 treats that increased distance
  as pure gain and never checks that it also removes the gradient.

**Also retracted: the shoulder-shape sweep and its `cauchy s = 0.10` recommendation.**
It was solving a problem at an operating point the policy may not occupy. The
finding that A3/A4 pass for *all 20* candidate shapes — including the current
`gaussian s = 0.06` — still stands and is a real gap in [RATE] §9: the acceptance
criteria test *ordering* and never test *distinguishability*.

**The better write-up.** [FT] now carries two distinct failures: an **objective
mismatch** (GRPO concentrates mass where the task requires spreading it) and a
**conditioning mismatch** (calibration and deployment scored the same construct
under different context, inflating the rate 2.4x). The second generalizes past this
pipeline — any LLM-as-judge reward where the judge is context-sensitive and the
calibration corpus does not share the deployment's conditioning has it, and almost
nobody checks. There is a clean isolation experiment behind it: same turns, same
checkpoint, one variable.

---

## 3. Champion validation on simulator text (`tools/champion_validation.py`)

[EBM] moves the §7.2 gold subset upstream of §6: the champions were calibrated on
AnnoMI and are applied to simulator text, so agreement has to be shown before any
derived rate means anything. **The data already existed** — `label_tasks/batch03`
is 150 simulator turns, balanced 25/cell across the six backbone cells, three
annotator sheets, frozen-champion key.

**The champion does not saturate.** Engine marked 68.7% vs annotators 58–66%
(κ 0.59–0.79); delivery 36.0% vs 37–48% (κ 0.35–0.59).

Engine marked rate per cell — champion tracks annotators closely except b2/b6:

| cell | champion | annotator range |
|---|---|---|
| b1 | 36.0% | 36–44% |
| b2 | 84.0% | 60–64% (champion over-marks ~22pp) |
| b3 | 68.0% | 64–72% |
| b4 | 92.0% | 92–96% |
| b5 | 56.0% | 40–62.5% |
| b6 | 76.0% | 50–72% |

b4 at 92% is a genuine outlier against AnnoMI's 26.5% and the caricature reading
survives *for that cell*.

### Delivery kappa, decomposed — refutes the expected diagnosis

`hot = Q1`; `warm = ¬Q1 ∧ Q3`. Q1 is exactly recoverable from a fused label; Q3
only on turns neither side called hot (hot masks Q3), so that column is conditional.

| pair | κ Q1 (hot) | κ Q3 (warm) | κ marked |
|---|---|---|---|
| champion × rater_1 | 0.490 | **0.821** | 0.498 |
| champion × rater_2 *(model annotator)* | 0.571 | 0.733 | 0.541 |
| champion × rater_3 | 0.399 | 0.440 | 0.245 |
| rater_1 × rater_3 **(human ceiling)** | 0.563 | **0.405** | 0.349 |

* **Q3 is where the champion does best, not worst** — 0.821/0.733 against a human
  ceiling of 0.405. The expectation that disagreement would concentrate in Q3, and
  that Q3's rubric was at fault, is not supported.
* **Q1 is the harder sub-question for everyone**, champion included, and the
  champion sits at the human ceiling (0.399–0.571 vs 0.563).
* On marked/unmarked the champion agrees with one human **better than the two
  humans agree with each other** (0.498 vs 0.349).
* **[FT §8]'s bar of κ ≥ 0.80 for delivery is above the human ceiling** and is
  unachievable by any instrument, a human included. **It should not gate.**

**The low numbers are one rater's warm criterion, not a Q3 defect.** rater_3 is the
outlier against all three other sources (0.440/0.405/0.402) while champion, rater_1
and rater_2 cluster at 0.733–0.821. The confusion matrix locates it: champion-flat ×
rater_3-warm on **21** turns, vs 2 and 6 for the others; their delivery marked rate
is 48% vs 36–38%.

With two human annotators neither reading can be called correct. "Ingratiating,
closeness-pulling, flattering, seeking connection with you" admits a broad and a
narrow reading and the rubric does not pin which; the disagreement is the evidence.
**Rubric disambiguation and a conversation with the annotator, not a rater-quality
finding.** It matters because the warm-target cells (b1, b3, b5) inherit whichever
reading the champion encodes — currently the narrow one.

Delivery κ therefore **does not gate** §6.

---

## 4. Rate-profile reward — what was built (`3e5b3ae`)

| file | role |
|---|---|
| `grpo/reward/rate_profile_reward.py` | rates over all `T`, `min` within axis, eps-floored geometric mean across, all load-time asserts |
| `grpo/calibration/rate_derive.py` | length-only eligibility, lean grouping, 25th/75th percentiles, span widening, hurdle delivery |
| `grpo/calibration/rate_v1_measurements.py` | §7's percentile table transcribed, with its discrepancies stated |
| `grpo/calibration/rate_calibration.{v1,v2}.yaml` | frozen artifacts + disclosure reports |
| `grpo/tests/test_rate_profile_reward.py`, `test_rate_derive.py` | A1–A7 plus the decisions below |
| `grpo/reward/trl_adapter.py::build_reward_func` | one place loads calibration, asserts C3, checks `T`; `arc_band` refused |

182 tests pass. The §7 score table reproduces exactly through the real reward path
(0/20 → 0.056, 3–5/20 → 1.000, 8/20 → 0.186, 20/20 → 0.000).

**Status: frozen, not deleted.** Training is unwired from `grpo.yaml`; the GRPO
objective and band-as-reward are dead, the AnnoMI-side calibration artifacts are
measurements of the corpus and survive whatever consumes them, and the rate
reparameterisation itself stays live — it is the answer to "how do you express a
target rate without conditioning," which is still open at low `m` (§6 below).

### Decisions taken where the spec was silent

1. **Lean groups are a disjoint partition, low-rate tested first.** §6 step 4 says
   "partition" but its predicates overlap.
2. **§5.2's minimum span applies to off-direction bands too**, which §7 applies to
   one band and not the three others like it. Off-direction widening can only go
   upward (§5.3 pins `L = 0`), which *loosens* a realism constraint, so every
   widening is recorded per band.
3. **The denominator is the arc's own graded turn count, never nominal `T`.**
4. **`MIN_ARC_TURNS = 10`**, derived not chosen: below `T = 10` the rate step `1/T`
   exceeds §5.2's span floor, which is [RATE] §1.5 reappearing at rollout.
5. **Group size held at 4**, resolving §11's flagged contradiction, with the VRAM
   reasoning recorded in config.

### [RATE] §9 gaps found

* **A5 as written is unsatisfiable and should be.** Under rate profiles the marked
  count *is* the on-direction rate for a one-directional arc, so A5 traverses A4's
  curve backwards; asserting both requires a flat, gradient-free reward. A4 is the
  correct one. The test asserts A5's *intent* (sparsity is not a free win) and
  records why.
* **No criterion detects a saturated, non-distinguishing region** (§2 above).
* **No criterion tested that an on-direction band penalises going inert** — added
  as `band(0) ≤ 0.5` (§6 below).

---

## 5. `rate_calibration.v2` — the real derivation (`d480579`)

The AnnoMI grader label cache was on fedora all along
(`results/grpo/calibration/annomi_grader_labels.jsonl`, 6,442 lines = 3,221 turns ×
2 axes). v1 was transcribed from §7; **v2 is derived**.

**§7's groups overlap, and the arithmetic closes exactly:**

| group | §7 | v2 (disjoint) |
|---|---|---|
| internalizing-leaning | 50 | 42 |
| externalizing-leaning | 39 | 29 |
| low-rate | 26 | 26 |
| ties dropped | — | 3 |
| **sum** | **115** over 100 eligible | **100** |

18 of §7's leaning conversations were also low-rate, and 3 two-sided ties were
counted into a leaning group: 18 − 3 = the 15 overlap. The low-rate group
reproduces §7 **bit-for-bit** (n=26, p75 0.077/0.078), which is strong evidence the
pipeline is correct and only the grouping differed.

Both leaning bands moved up, as predicted by low-rate-first precedence:

* internalizing on-band `[0.144, 0.290]` → `[0.167, 0.302]`
* externalizing on-band `[0.106, 0.232]` → `[0.156, 0.280]` (+47% on the lower edge)

**§5.3's stated warrant is false.** It justifies the missing off-direction lower
edge with "the 25th percentile of the off-direction rate is 0.000." Under the
disjoint partition it is **0.044** (internalizing) and **0.051** (externalizing) —
that 0.000 was an artifact of the overlap. The rule is kept (conservative, and the
low-rate group's p25 *is* 0.000) but its warrant does not hold as written.

---

## 6. Delivery is zero-inflated; the hurdle model works and is unusable at T=20 (`d89c54b`)

Measured on the same cache, 100 eligible conversations:

| axis | conversations with **zero** marked turns | p25/p50/p75 (all) | p25/p50/p75 (**nonzero only**) |
|---|---|---|---|
| engine | 6% | 0.158/0.250/0.346 | 0.177/0.270/0.355 (n=94) |
| delivery | **57%** | 0.000/0.000/0.043 | **0.029/0.048/0.091** (n=43) |

"Delivery is unmeasurable" is too strong. The **unconditional** rate is
unmeasurable — dominated by structural zeros, so its percentiles read the hurdle
rather than the rate. The **conditional** distribution is well-behaved.

**Conditioning is correct here and wrong on engine**, for a precise reason:
delivery's zeros are *structural* (57% of clients never direct affect at the
counsellor), engine's are *sampling* (6%). [RATE] §1 deleted the conditional ratio
for engine on evidence that does not transfer. One parameterisation had been
applied to two differently-shaped axes.

A hot profile **sets the hurdle by construction**, so the conditional distribution
is its target — and it is measured. Hot band: `[0.023, 0.092]` → `[0.007, 0.107]`
after the §5.2 widening.

### And the measurement says the target is not trainable

| | 0 hot turns | 2/20 | 3/20 | 6/20 |
|---|---|---|---|---|
| **measured** `[0.007, 0.107]` | **0.9926** | 1.0000 | 0.776 | 0.006 |
| **declared** `[0.150, 0.300]` | 0.0439 | 0.707 | 1.0000 | 1.0000 |

On 12 real hot-cell arcs (b2/b4/b6): measured band mean **0.9951**, std **0.0035** —
the hot cells are already "perfect" without being hot, and there is no gradient.
Declared band: mean 0.1505, std 0.1892, and it separates `p_hot` 0.000 from 0.100
as 0.044 vs 0.707.

**So the number with no data under it is the only one that trains.** Uncomfortable,
but that is what the data says. The declared `[0.150, 0.300]` was set by analogy to
engine's geometry; this shows the analogy was wrong — the axes do not behave alike.

**Longer arcs do not fix it.** `band(0)` depends only on `L` and the shoulder, never
on `T`. AnnoMI's hot clients are hostile on 2–9% of turns, so "hot" and "not hot"
are genuinely close *as rates*. Property of the corpus, not the resolution.

### The screening criterion for a better corpus

For an on-direction band to penalise going inert at `band(0) ≤ 0.5`, you need
`L ≥ 1.18 · s`. At `s = 0.06` that is **conditional p25 ≥ 0.071**, ~1.4 marked
turns in 20.

* AnnoMI hot: p25 = 0.023 — **3× too low.**
* Human annotators on simulator output: 37.3% / 38.3% marked — far above.

A candidate corpus can be screened on this before any investment. Structural
requirement: **the listener must be a plausible target of the affect.** Couples
therapy is the closest analogue to a layoff conversation (the person you are angry
at is in the room) and better matched than the complaint/ombudsman recordings [EBM]
§8 nominates — which it dismisses on an argument, not a measurement.

Also: §8's claim that the scarcity is corpus-not-grader rests on an MI-quality
stratification (hot 2.4% poor vs 1.8% good). That is 0.6pp on a 2% base, and
AnnoMI's "poor" sessions demonstrate bad *technique*, not hostile encounters. The
stratification tests the wrong contrast; mandated-vs-voluntary would be
informative and is checkable on the existing cache.

### New load-time assert

An on-direction band must **penalise going inert**, tested as `band(0) ≤ 0.5`, not
as `L > 0`. The measured hot band passes `L > 0` at `L = 0.0073` and still scores a
completely inexpressive arc at **0.993** — the edge alone was the wrong property to
assert. Zero expression scoring ~1.0 makes the profile's central claim
unfalsifiable, and on a neutral-engine cell (b5, b6, whose engine components are
both `[0, U]` by construction) the delivery lower edge is the only thing in the
reward standing between an empty arc and full marks.

---

## 7. [EBM] review — resolved items

* **D0.3 resolved from the code, spec was stale.** The delivery champion is
  **Q1 (hostility toward listener) / Q3 (closeness-pulling toward listener)**, not
  Q1/Q2. The non-hot pole *is* separately detectable and does not collapse into
  "not-hot." Q3 was added because a Q1-only version made delivery a free pass on
  b1/b3/b5 (b3 sat at mean reward 1.000 with 100% group collapse). Caveat from the
  module: neither Q1 nor Q3 has been validated against humans **in isolation** —
  §3 above is the first such measurement.
* **`argmax_margin_toward` was a bug; delete it.** Champions emit booleans and a
  categorical label; there is no margin, and manufacturing one from boolean-token
  logprobs would invent a continuous quantity out of nothing. **Exhaustion accepts
  `draw₀`** — the original unconditioned marked draw, unbiased from `a(x | marked)`
  by construction, free, and it removes the max-order-statistic bias entirely
  rather than trading it for a random-pick version.
* **Engine is probably not reachable**; run the prompt-ceiling diagnostic in
  reverse as **two separate tests** — lowering `p` (off-profile marked turns) and
  lowering `d` (unmarked turns) are different capabilities. The `d` test matters
  more: if few-shot exemplars of distortion-free turns do not move markedness, the
  instrument's density is wrong against its own human reference, which is a realism
  defect independent of controllability. If `p` will not move either, engine is a
  **fixed cell attribute, not a manipulable axis** — report it as such, revert to
  the categorical IV the 2×3 backbone already had, and drop the engine scheduler.
* **R4 — one scheduled axis per cell, the other observed.** At P = 0.99 collinearity
  the axes are not separately identifiable however they are scheduled; joint
  scheduling yields uninterpretable coefficients. Collinearity reported as an
  instrument limit; b2/b4/b6 declared out of range on delivery until the hot base
  rate moves.
* **C4 amendment: one estimator AND one conditioning** — any quantity compared
  across calibration and deployment must share the full triple (estimator, context
  construction, grader checkpoint at fixed temperature).
* **The reparameterise-vs-condition rule, to be written into the spec:**
  reparameterise when the quantity drives an update; condition when it is an
  observation. As an optimization target a ratio with a random denominator gives a
  gradient whose resolution varies per rollout. As a measured dose, conditioning is
  correct and `m` is the precision weight — varying `m` is heteroskedasticity,
  handled by weighting. **That defence holds to about `m` = 5 and then stops.** At
  delivery's measured `m` ≈ 1 the per-arc observation is a single Bernoulli draw
  and no weighting rescues it; the move there is to abandon per-arc dose and pool
  to cell-level `q` with binomial intervals. "`q` is the regressor" silently
  assumes an `m` the delivery axis does not have.
* **D0.1 confirmed low-discrepancy.** `q` is a regressor throughout and the label
  sequence is not an analysis object; within-arc escalation is owned by the
  pressure schedule in the prompt [PIPE §8]. Caveat: at `m ≤ 2` the scheduler is
  moot, so the choice only earns anything on axes reaching `m ≥ 5` — possibly none.
* **Code layout.** Extract `backends.py` and `delivery_decompose.py` into a shared
  champions package (three consumers now: frozen GRPO record, controller,
  reporting); controller as a **sibling** to `grpo/`, not inside it; leave `grpo/`
  frozen.

### [EBM] numbers voided as corpus-derived rather than instrument facts

The `d′ = 0.066` worked example (§5.2), the `m ≈ 5` lattice and its `1/m` bound
(§5.4), the 1.5–3× cost estimate (§10), and §5.3's framing of the high-`p` case as a
hazard rather than a description. Measured simulator rates: engine `d` ≈ 42%
(prev-interlocutor) or ~1.00 (patient-own); delivery `d` 0.50–0.65 on warm cells,
`p_hot` ≈ 0.021 on hot cells.

### Second-order, if the numbers survive the convention test

The controllable surface is much smaller than [EBM] assumes. Engine may have no
variance to control. Delivery is **one-sided**: warm available at `d` = 0.5–0.65,
hot at `p` = 0.021 (16% hit rate at K = 8, 7 of 12 arcs at literal zero). A bipolar
axis samplable on one pole only makes the U-shape question — whether hostility and
closeness-pulling buy capitulation by different mechanisms — **empirically
unanswerable with this instrument**. That is a finding about reach and belongs in
the paper as one, but it should stop anyone building a controller for an axis with
one attainable value.

**Also downstream of `d`:** [BS §6]'s window mechanism assumes N/A turns exist and
do not consume slots. At full markedness there are none and the window is always
full width. Quiet coupling, worth checking.

---

## 8. Privacy / public-repo items — NOT actioned, need decisions

`ostenloo/SWAY` is **public**.

1. **The per-annotator sheets were never committed** — `git log --all --diff-filter=A`
   is clean for `hand_labels_batch03_{name}.csv`. They remain untracked. Keep them
   that way.
2. **But a summary of them is already public.**
   `label_tasks/batch03/annotator_model_sweep.md` is tracked (added in `701592f`)
   and names both annotators with per-person agreement scores, including
   `austin × qingqing ← human–human ceiling | 0.607 | 0.658` and a full
   model-vs-each-annotator matrix. This is exactly the per-person data that was
   withheld from the new report. **Not touched** — rewriting public history is a
   decision requiring coordination. It also changes what the annotator conversation
   is: "some of this is already published, what would you like done" rather than
   "may I publish this."
3. **`docs/RUNNING.md:13` publishes a Tailscale IP (`100.71.95.25`) and host
   username** in a public repo. Unrelated to annotation; worth closing.
4. Remaining name hits (`specs/spec.md`, `specs/fidelity_checker_2.md`,
   `docs/SYC_HAND_LABELING.md`) are the repo owner's own name as author/labeler —
   self-identification, fine unless preferred otherwise.
5. `tools/champion_validation.py` **anonymises annotators by default** (`rater_N`,
   discovered by glob, never hardcoded) with `--show-annotators` for local use.
   Reproducibility is served by publishing the champion key plus aggregate
   agreement, with sheets on request — a normal arrangement for
   human-subjects-adjacent annotation data.

---

## 9. Order of work from here

1. ~~Delivery κ decomposition~~ — **done** (`fcb2825`); does not gate.
2. **Three-way convention test** (bare / prev-interlocutor / patient-own, per axis)
   — next action. Note: if the warm rubric is disambiguated and re-labelled first,
   a comparison run against the current labels is invalidated, so sequence the
   rubric conversation against this deliberately.
3. **[EBM] §6 reachability** under the pinned convention(s), plus the per-MUT dose
   spread from §1.
4. **Then** decide whether the controller is worth building. Honest projection: it
   reduces to delivery-warm intensity on b1/b3 and nothing else.

No controller code until step 4.

### Open, unowned

* Rubric disambiguation for Q3 "closeness-pulling," and the annotator conversation.
* Whether a corpus meeting the §6 screening criterion is worth acquiring.
* [RATE] §9 needs a distinguishability criterion (§2).
* [EBM] §5.3's high-`p` framing, and the [BS §6] window coupling.
