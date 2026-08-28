# SWAY — Engine/Delivery Distributional Band Reward Spec (coding-agent handoff)

Companion to `sway_grpo_simulator_finetune_spec.md` **[FT]**,
`patient_pipeline_spec.md` **[PIPE]**, `benchmark_spec.md` **[BS]**,
`patient_profile_spec.md` **[PS]**, `sway_profile_roster.md` **[ROST]**,
`layoff_fact_base.md` **[FB]**.

This spec replaces the **engine/delivery reward shape** in **[FT §4]** — the
per-turn monotone binary `0.5·engine_pass + 0.5·delivery_pass` — with an
**arc-level conditional-band** reward. It adds one new **offline** stage (AnnoMI
calibration, §6) that produces the band parameters. It does **not** touch the
Goodhart wall, the run-time loop, the realism floor mechanism, KL, warm-start
mechanics, certification, or VRAM planning — those remain owned by **[FT]**.

> **Why the shape changes.** The monotone per-turn binary makes "more on-profile
> every turn" strictly better, so its optimum is the caricature: an internalizing
> profile that internalizes on *every* marked turn. Real patients never do this;
> the caricature is both unrealistic and the exact attractor **[FT R3]** warns
> about. A distributional target with an interior optimum dissolves the attractor
> without a separate penalty term (§4).

> **Precedence.** Where this spec touches a component owned by **[FT]** (the §4
> reward, §5.2 grouping, §6 warm-start filter), it states the re-sync and **[FT]**
> is updated to match. It MUST NOT alter **[FT] C1–C2** (fidelity-only reward,
> reward-source ≠ Judge family): the distribution of engine/delivery labels is
> still a pure fidelity signal and carries no drift information.

---

## 0. Decisions to resolve before implementation

- **D-BAND.1 — Axis combination (RESOLVED: geometric mean).** Per-arc scalar =
  `sqrt((r_engine + ε)·(r_delivery + ε))`, `ε = 0.02`. **This supersedes the
  earlier default of `0.5·r_engine + 0.5·r_delivery`**, whose stated rationale
  ("average preserves gradient when one axis is out-of-band, anti-collapse, cf.
  **[FT R2]**") did not survive measurement. Three reasons, in order of weight:

  1. **The average is indifferent to allocation, and worse than indifferent.**
     `(r_eng, r_del) = (0.95, 0.10)` and `(0.10, 0.95)` both score 0.525, and a
     balanced `(0.52, 0.52)` arc scores 0.520 — *less*. An arc that half-nails one
     axis and is dead on the other is worth **more** than an arc doing both
     passably, so the average actively prefers specialisation. That is the wrong
     preference for a profile that asserts both an engine and a delivery.
  2. **The geometric mean reallocates gradient to the failing axis for free.**
     `∂/∂a √(ab) = ½√(b/a)`, so the weak axis gets the larger derivative: ~9.5×
     toward delivery at `(0.95, 0.10)`, ~6× at `(0.90, 0.15)`, exactly 1× when
     balanced. The average is flat 1:1 always, so nothing pulls a lopsided arc
     back toward balance.
  3. **The anti-collapse argument was backwards.** It conflates reward *level*
     with gradient. GRPO consumes within-group **variance**, and a near-constant
     healthy axis dilutes it. Measured on a `G = 8` group with engine pinned at
     0.90 and delivery scattered 0.02–0.22: mean `|advantage|` is **0.0263 under
     the average vs 0.0786 under the geometric mean**. The average causes the
     collapse it was chosen to prevent — and this is b2's signature (0.93
     collapse at 0.494 mean reward, close to the 0.525 an average pays for
     one-axis-only).

  **Geometric mean rather than the raw product** because geo *agrees with the
  average on the diagonal* — `(0.52, 0.52) → 0.520`, `(1, 1) → 1.0` under both —
  and diverges only when lopsided. The raw product squashes the whole range
  toward zero (balanced mediocre → 0.27). Geo is the minimal change that fixes
  the defect. Note geo is a monotone rescaling of the product, so the two induce
  the **same ordering** over arcs and differ only in gradient magnitude.

  **The ε floor is required, and only by this branch.** Any product-like rule
  scores 0 whenever one axis is 0, and then every arc in the group scores 0 —
  zero variance, no gradient, no way to learn out of inertness (**[FT R2]** for
  real, this time). An axis reward is exactly 0 only when `density_factor = 0`,
  i.e. **zero marked turns on that axis** (`band()` is Gaussian and never exactly
  0; `density_low` never is). At `ε = 0.02` an inert-delivery arc with healthy
  engine scores `√(0.90 × 0.02) = 0.134`, and marking even 1 turn in 20 lifts it
  to ~0.36 — a strong pull out of the dead zone, far too small to be worth
  farming. The average needs no floor and is given none.

  **Not configurable.** The constant lives in `grpo/reward/band_reward.py`
  (`DEFAULT_AXIS_COMBINATION`, `AXIS_FLOOR_EPS`), not in `grpo.yaml`: a config key
  able to silently select the defective average is the kind of thing the rest of
  the pipeline refuses to allow, and this decision is now Settled (§12).
  `combine_axes` still implements all three modes because the tests need them to
  *demonstrate* the defect, but nothing reads them from config.
- **D-BAND.2 — Delivery density: floor or two-sided band.** Engine density is
  scenario-driven → a one-sided **floor** (don't be inert) is enough. Delivery
  density is closer to a profile *base rate* (a Hot profile is mostly flat but marks
  affect at some rate). If the roster intends Hot/Warm to fix a marking *rate*, give
  delivery density a **two-sided band** instead of a floor. **Default: floor**;
  promote to a band if §6 shows Hot/Warm profiles need a marking-rate target, not
  just a direction.
- **D-BAND.3 — Design floor `L_design` per axis/direction.** The reward's lower band
  edge is a **disclosed design parameter**, chosen within the external bracket
  (§6). It is *not* read from data — data only bounds it (CB2/CB3). Set per
  profile-direction in the frozen calibration artifact.

---

## 1. The factorization (what the reward targets)

The arc-level label distribution is a product of two quantities with different
owners; the reward must not conflate them (this is the whole reason for the design):

- **Affordance density** `d_a` = the **marked fraction** on axis `a` =
  `(#marked turns) / (#turns)` = `1 − (#neutral-or-flat)/(#turns)`. "How often the
  axis is recruited at all." **Scenario-owned**, roughly fixed by **[FB]** + the
  fixed conversation, not by the profile. Marked = `{int, ext}` for engine,
  `{warm, hot}` for delivery.
- **Conditional on-profile share** `q_a` = `(#on-profile-class turns) / (#marked
  turns)` on axis `a`. "Given the axis is recruited, how often it goes the
  profile's way." **Profile-owned.**

`marginal on-profile rate = d_a · q_a`. The reward scores `q_a` (a **band**, §4)
and `d_a` (a **floor/band**, §5) **separately**, never their product — matching the
product against a marginal target lets the large neutral/flat mass dominate and
reduces to measuring "did you stay quiet the right fraction of the time," which is
scenario-determined and not the target (the pooled-κ-dominated-by-flat failure,
on the reward side).

### 1.1 Two-class reduction (why there is one scalar per axis)

Each axis has exactly **two** marked classes. Conditioning on marked and dropping
the unmarked class leaves a **one-dimensional** quantity:

| Axis | marked classes | on-profile class `c*` (per profile) | scored scalar `q` |
|---|---|---|---|
| engine | `{int, ext}` | `int` if internalizing/dependency; `ext` if externalizing/entitlement | `q_eng = #c* / (#int + #ext)` |
| delivery | `{warm, hot}` | `warm` if profile delivery = Warm; `hot` if Hot | `q_del = #c* / (#warm + #hot)` |

So there is **no product over classes** to compute and no full-vector distance —
one target rate and one observed rate per axis. (A product-of-per-class bumps is
only needed for ≥3 marked classes; SWAY has none.)

### 1.2 Neutral-engine profiles (the one exception)

Neutral-engine cells (e.g. Neutral×Warm control **[BS]**) have **no on-profile
marked engine class** — the target is *low engine expression*, i.e. low `d_eng`.
For these cells the engine reward is **not** a band on `q_eng`; it is a band on
`d_eng` targeting a low anchor (§5.3). Delivery is unaffected (every profile has a
delivery direction). Detect via `profile.engine == neutral` and branch.

---

## 2. NON-NEGOTIABLE CONSTRAINTS (extend [FT] C1–C7)

- **CB1 — Calibration and reward share one frozen backend.** The band lower bound
  (§6) MUST be measured with the **exact** `engine_backend` / `delivery_backend`
  that supplies the reward, at temp 0, same checkpoint. This is what makes the
  reward robust to an imperfect grader (instrument cancellation, §7): systematic
  grader bias inflates the target and every rollout by the same amount and cancels
  in the comparison — but only if the *same* instrument measures both. A band read
  from human labels and applied to grader-scored rollouts breaks cancellation.
- **CB2 — Band edges are external-or-disclosed, never simulator-derived.** The
  external bracket (`L_ext`, `U`) comes from human data (AnnoMI §6, or human
  layoff-domain transcripts) and an interiority ceiling; the design floor
  `L_design` is a disclosed constant chosen within the bracket. **No band parameter
  is ever read from the Simulator's own rollout distribution** — that is the
  circularity/ratchet failure the freeze-before-gating rule exists to prevent.
- **CB3 — Non-empty, floored band (assert at load).** For every axis/direction:
  `L_ext ≤ L_design < U`. If `L_ext ≥ U`, the corpus and the interiority ceiling
  disagree — **halt**, do not clamp. If `L_design < L_ext`, the design floor sits
  below the human floor — **halt**.
- **CB4 — Interiority (assert at load).** `U < 1.0` and `L_design > 0.0` strictly.
  A band edge at 0 or 1 degenerates to a monotone edge and reopens caricature.
- **CB5 — Reward is still fidelity-only (inherits [FT] C1).** The band reward reads
  only `(arc_patient_turns, P, context, cell)`. Grep guard extends: the band module
  imports nothing from Judge / `[A]` / `[B]` / drift paths.
- **CB6 — Soft edges, not cliffs.** The density floor is a ramp and the band
  shoulders are smooth (§4–§5). Hard 0/1 gates kill within-group gradient and
  reproduce the collapse (`frac_reward_zero_std`) **[FT §9]** is meant to avoid.
- **CB7 — Freeze before training.** The calibration artifact (§6.4) is frozen and
  hash-logged before the first GRPO step; changing a band parameter mid-run
  invalidates the run.

---

## 3. Reward interface

Arc-level. One scalar per arc, standardized within the GRPO group (§8).

```python
def band_reward_arc(
    arc_turns: list[str],      # ordered patient turns of one rollout
    context0: str,             # initial state / history prefix
    P: str,                    # frozen profile prompt for the cell
    cell: str,
    cal: BandCalibration,      # frozen §6.4 artifact for this cell
) -> float:
    """
    Returns scalar in [0, 1]. Reads ONLY these inputs (CB5).
    Grades every turn once with the frozen backends, aggregates to per-axis
    (density, on-profile-share), scores each axis, combines (D-BAND.1),
    then applies the realism floor from [FT §4].
    """
    eng_labels = [engine_backend.label(t, context_upto(arc_turns, t, context0), cell)
                  for t in arc_turns]         # each in {int, ext, neutral}
    del_labels = [delivery_backend.label(t, context_upto(arc_turns, t, context0), cell)
                  for t in arc_turns]         # each in {warm, hot, flat}

    r_eng = axis_reward("engine",   eng_labels, cell, cal)
    r_del = axis_reward("delivery", del_labels, cell, cal)

    # D-BAND.1 (RESOLVED: geometric mean, eps-floored). The average is
    # indifferent to allocation and prefers specialisation; see §0 D-BAND.1.
    a, b = max(r_eng, EPS), max(r_del, EPS)          # EPS = 0.02, the R2 guard
    diagnostic = sqrt(a * b)

    # realism as multiplicative floor, aggregated over the arc — from [FT §4],
    # mean pass rate so one bad turn dampens rather than zeroes (CB6). Never
    # maximized, so it cannot be farmed.
    realism = mean(realism_backend.check(t, context_upto(arc_turns, t, context0))
                   for t in arc_turns)        # each {0,1} -> mean in [0,1]
    return diagnostic * realism
```

`engine_backend` / `delivery_backend` now expose a `.label()` returning the
**categorical** (not the old `.score()` binary); the per-turn `engine_pass`
binary is derived downstream (RFT filter, §9) but is **not** the reward.

---

## 4. `q`-band (the on-profile-share reward per axis)

```python
def axis_reward(axis, labels, cell, cal):
    marked   = [x for x in labels if x in MARKED[axis]]     # {int,ext} / {warm,hot}
    n_marked = len(marked)
    n_turns  = len(labels)

    p = cal[cell][axis]                                     # band params for this cell

    # --- neutral-engine special case (§1.2) ---
    if axis == "engine" and p.mode == "density_low":
        d = n_marked / max(n_turns, 1)
        return band(d, p.d_lo, p.d_hi, p.s_lo, p.s_hi)

    # --- density floor / band (§5) ---
    d = n_marked / max(n_turns, 1)
    d_factor = density_factor(d, p.d_floor, p.d_ceil)       # ramp, in [0,1]

    if n_marked == 0:
        return 0.0            # caught by d_factor anyway; explicit for clarity

    # --- conditional on-profile share, Dirichlet-smoothed (CB6, §4.1) ---
    n_on = sum(1 for x in marked if x == p.c_star)
    q = (n_on + p.alpha) / (n_marked + 2 * p.alpha)

    return d_factor * band(q, p.L_design, p.U, p.s_lo, p.s_hi)


def band(x, L, U, s_lo, s_hi):
    """Plateau [L,U] = 1.0; Gaussian shoulders outside. L=L_design, U=U_ceiling."""
    if x < L:  return exp(-((L - x) ** 2) / (2 * s_lo ** 2))
    if x > U:  return exp(-((x - U) ** 2) / (2 * s_hi ** 2))
    return 1.0
```

Design points:

- **Interior optimum ⇒ anti-caricature for free.** An all-on-profile arc has
  `q = 1.0`; with `U < 1.0` (CB4) that sits on the upper shoulder and scores **< 1**.
  The falloff-above-`U` *is* the anti-caricature term — no separate penalty. This is
  the property the monotone binary lacked.
- **Plateau, not peak.** Any `q ∈ [L_design, U]` scores 1.0. This is deliberate: the
  external data brackets the interval but does not identify a point inside it
  (§6), so the reward must not assert a precision it cannot source. To pull central
  tendency higher, **narrow the band** (raise `L_design` toward `U`) — one knob,
  disclosed. Do not replace the plateau with a peak at an invented target.
- **Asymmetric shoulders.** `s_lo` tight (dropping below the profile's on-profile
  floor is a clear defect — the ordering must hold); `s_hi` sets how hard caricature
  is punished. Set per axis in §6.4.
- **Smoothing.** For a `T`-turn arc, `q` is quantized in steps of `1/n_marked` with
  multinomial noise; the Dirichlet `alpha` (§4.1) both defines `q` gracefully at low
  `n_marked` and damps quantization jitter so the reward tracks behavior, not
  sampling artifacts (CB6, monitored in §9).

### 4.1 Sequence-agnostic by decision

The `q`-band uses the **arc-marginal** count of on-profile turns — it does **not**
segment the arc by position or pressure. This is intentional: there is **no
pressure ladder** in this design, so there is nothing to segment against, and the
reward's job is to shape roughly-right aggregate expression while fine temporal
structure is measured downstream on the MUT **[BS]**, not required of the
Simulator. Do **not** add a phase-/position-segmented target. (Re-sync: the
**[FT §4]** note forbidding a self-consistency term "because it fights the pressure
schedule" is now void as stated — there is no schedule; if a no-self-consistency
rule is still wanted, it needs a fresh justification, not the schedule one.)

---

## 5. Density floor (closes the inert-simulator hole)

The `q`-band scores *only marked turns* and says nothing about how *many* there are.
Without a density term the policy can farm the band by being marked on ~2 turns,
getting the ratio right, and scoring perfectly while essentially inert — a
degenerate-on-diagnostic solution, and precisely the guarantee the (now-removed)
interlocutor affordance tag would have provided under a marginal target.

```python
def density_factor(d, d_floor, d_ceil=None):
    lo = min(1.0, d / d_floor)               # ramp up to the floor (CB6 soft)
    if d_ceil is None:
        return lo                            # one-sided floor (default, D-BAND.2)
    hi = min(1.0, (1.0 - d) / (1.0 - d_ceil)) if d > d_ceil else 1.0
    return lo * hi                           # two-sided band if d_ceil set
```

- **One-sided floor (default).** Ramp from 0 to 1 as `d → d_floor`, flat above.
  Gently pushes the policy to be marked at least `d_floor` of the time; no cliff.
- **Two-sided (D-BAND.2, delivery option).** Add `d_ceil` if a profile's Hot/Warm
  setting is meant to fix a *marking rate* (a Hot profile is not hot every turn).
- **`d_floor` is a soft anchor, not a measured target.** AnnoMI's grader-measured
  marked fraction (§6.3) gives its rough altitude only — density is the most
  scenario-dependent quantity in the factorization and AnnoMI is off-scenario
  (§6.5). Treat `d_floor` as operational; do not claim it is norm-referenced.

### 5.3 Neutral-engine density target

For `profile.engine == neutral`, engine reward = `band(d_eng, d_lo, d_hi, …)` with a
**low** `[d_lo, d_hi]` (the control should express little engine). Anchor `d_hi`
below the on-profile cells' `d_floor`; `d_lo` may be 0. This is the `density_low`
mode branched in §4.

---

## 6. AnnoMI calibration (new OFFLINE stage — produces the frozen band artifact)

Runs once per grader-backend version, before training. Output is the frozen
`BandCalibration` consumed by §3–§5.

**What AnnoMI is.** 133 professionally transcribed, expert-annotated motivational-
interviewing sessions (Wu et al., ICASSP 2022; expanded Future Internet 2023),
from educational counselling videos, released as `AnnoMI-simple.csv` /
`AnnoMI-full.csv`. Client utterances carry a `client_talk_type ∈ {change, neutral,
sustain}`; therapist utterances carry a behaviour label. **It has no engine
(int/ext) and no delivery (warm/hot/flat) labels** — those are SWAY constructs — so
engine/delivery must be **hand-labeled** onto client turns here. (Its own client
talk-type IAA is only moderate, Fleiss κ ≈ 0.47 — irrelevant to us since we relabel;
our engine/delivery κ is what matters.)

### 6.1 Hand-label pass (doubles as instrument validity [BS §7])

1. Sample **client** utterances across all 133 sessions (stratify by session, drop
   backchannels like "mhmm" that carry no engine/delivery signal).
2. Two raters (annotator + Qingqing) label each sampled turn on **engine**
   `{int, ext, neutral}` and **delivery** `{warm, hot, flat}`. Adjudicate; report
   **stratified κ** (engine, delivery separately). This κ is the instrument-validity
   result **[BS §7]** and is reused, not run twice.
3. Store the hand-label key **git-ignored** (same discipline as the gold set **[PIPE]**);
   commit only derived aggregates (§6.4).

### 6.2 Grader-space measurement (REQUIRED for cancellation, CB1)

Score the **same** sampled turns with the frozen `engine_backend` / `delivery_backend`
that will supply the reward. All band lower bounds below are computed from **these
grader labels**, not the human labels — the reward operates in grader-measured
space, so its anchor must live there too (§7).

### 6.3 Derive the external bracket

Over grader-marked AnnoMI turns (pooled across the mixture):

- `s_eng = #int / (#int + #ext)` → `L_ext(engine, internalizing) = s_eng`;
  `L_ext(engine, externalizing) = 1 − s_eng`.
- `s_del = #warm / (#warm + #hot)` → `L_ext(delivery, Warm) = s_del`;
  `L_ext(delivery, Hot) = 1 − s_del`.
- `d_anno(axis) = (#marked) / (#turns)` → soft altitude for `d_floor` (§5).

**Why this is a lower bound, not a point (the load-bearing epistemics).** AnnoMI has
**no profile structure** — it is a naturalistic mixture of whoever was in those
sessions, and post-hoc sorting it on the very axis you want to measure is circular.
A pooled mixture share is *diluted* by off-direction patients, so a single strong
direction's true `q*` lies **above** `L_ext` (for its on-profile class). AnnoMI
therefore yields the bracket floor; interiority (`U < 1`) yields the ceiling; the
point inside is chosen (D-BAND.3), not measured. **Guard:** if a mixture is already
dominated one way and `L_ext ≥ U`, CB3 halts — pick a scenario-matched corpus or
lower expectations, do not fabricate a point.

### 6.4 Freeze the artifact

Per cell, write and hash-log (CB7):

```yaml
# band_calibration.<grader_version>.yaml   (frozen; CB7 hash in run log)
cells:
  b1_internalizing_warm:
    engine:
      mode: q_band
      c_star: int
      L_ext:    0.62         # §6.3, grader-space lower bound
      L_design: 0.70         # D-BAND.3 disclosed, CB3: L_ext <= L_design < U
      U:        0.90         # interiority ceiling, CB4: < 1.0
      s_lo:     0.06         # tight: ordering must hold
      s_hi:     0.12         # anti-caricature softness
      alpha:    0.5          # Dirichlet smoothing
      d_floor:  0.30         # §5 soft anchor from d_anno
    delivery:
      mode: q_band
      c_star: warm
      L_ext:    0.55
      L_design: 0.62
      U:        0.88
      s_lo:     0.06
      s_hi:     0.12
      alpha:    0.5
      d_floor:  0.15         # flat legitimately dominant -> low
      # d_ceil: 0.45         # uncomment for D-BAND.2 two-sided delivery band
  b_neutral_warm:
    engine:
      mode: density_low      # §1.2 / §5.3
      d_lo:  0.0
      d_hi:  0.12
      s_lo:  0.04
      s_hi:  0.06
    delivery: { mode: q_band, c_star: warm, ... }
```

Values illustrative — fill `L_ext`/`d_anno` from §6.3, choose `L_design`/`U`/sigmas
per D-BAND.3, tune in the pilot.

### 6.5 Transfer caveat (log it, do not wave it away)

AnnoMI is MI/behaviour-change counselling, not layoff support. Reading a bound off
it assumes the on-profile *share among marked turns* is roughly scenario-invariant —
more defensible for the **conditional** `q` (a presentation property) than for
**density** `d` (strongly scenario-set), which is a further reason `d_floor` is only
a soft anchor. If human **layoff-domain** transcripts are obtainable, hand-label
those instead for a scenario-matched bracket. **Never** substitute the Simulator's
own authored/rollout material for the bracket (CB2) — that reintroduces circularity.

---

## 7. Instrument cancellation (why an imperfect grader is acceptable)

The reward compares two grader-measured distributions — the AnnoMI-derived band and
each rollout's `q` — using the **same** frozen classifier (CB1). A systematic
classifier bias `Δ` (e.g. the grievance→hot confound over-labeling hot) inflates the
**target** (measured on AnnoMI) and the **rollout** (measured on sim output) by
~the same `Δ`, so it largely subtracts out of the comparison. The reward needs the
grader to be **consistent** (same bias on both text distributions), not **accurate**
(zero bias) — a far weaker, cheaper property. This is why a κ ≈ 0.80 engine grader
and a shakier delivery grader can both drive a usable reward.

**Scope (first-order only).** Cancellation holds insofar as `Δ` is the same on human
(AnnoMI) and simulator text. If the Simulator emits markedly **more** grievance than
AnnoMI patients, the grievance→hot confound fires harder on rollouts than on the
target, `Δ_rollout > Δ_target`, and the residual pushes the policy toward a slightly
wrong setpoint. **This is exactly what the [FT §8] pre-flight probe measures** —
comparing grievance density in rollouts vs the human distribution is the empirical
check on whether `Δ` is stable. **CB-tie:** the delivery band inherits [FT §8]; if
`delivery_backend` fails the §8 probe, its `q_del` (and therefore `L_ext(delivery)`)
are untrustworthy — harden or swap the backend per **[FT §8]** before this reward is
valid.

---

## 8. Group formation change (re-sync [FT §5.2])

The distributional reward is defined over an **arc**, so grouping moves from
**per-turn** to **per-arc**:

- At each state, sample **G full arcs** (not G single turns) from the current policy
  snapshot against a fixed interlocutor seed; roll cross-interlocutor per **[FT §5.3]**.
- Score each arc with `band_reward_arc` (§3); standardize the scalar within the G-arc
  group for advantages (**[FT §7]** unchanged otherwise).
- Cost: G full arcs per step is heavier than G turns — cap arc length and G together
  in the VRAM budget **[FT §12]**; the KV-cache remains the binding constraint.

Per-turn grouping in **[FT §5.2]** is retained **only** for the RFT warm-start
filter (§9), which is a coverage tool, not the shaped reward.

---

## 9. Warm-start interaction (re-sync [FT §6])

Warm-start's job is unchanged: raise the marked **base rate** so GRPO groups are
non-empty. Keep the **per-turn** RFT filter (`engine_pass`/`delivery_pass` derived
from `.label()`), because RFT reinforces what the base already samples and cannot
reach an off-manifold *distribution* — distributional shaping is GRPO's job.

The known **warm-start coverage gap** (cells contributing zero RFT examples) is
diagnosed the same way: inspect the per-cell `(engine_pass, delivery_pass)` **tuple**
from existing logs, not the summed scalar — a 0.5 is ambiguous between
`(1,0)` and `(0,1)`. Delivery-zero cells need near-manifold relaxation on the
delivery target; engine-zero cells can lower the pass threshold. This is
[FT §6]'s domain; noted here only because `d_floor` (§5) will read as unmet on any
cell RFT left uncovered — **fix coverage before reading density.**

---

## 10. Monitoring additions (extend [FT §9])

- **Per-cell rate telemetry.** Log the rollout distribution of `q_eng`, `q_del`,
  `d_eng`, `d_del` per step. The band gives no within-plateau gradient by design, so
  expect `q` to settle *somewhere* in `[L_design, U]` — that is correct, not
  collapse. Flag if `q` parks **below** `L_design` (ordering breaking) or rides `U`
  (caricature pressure through the ceiling).
- **Histogram-noise vs policy-variance.** When collapse eases, confirm the reward
  spread tracks **behavioral** differences between arcs, not `1/n_marked`
  quantization jitter. If arcs are short and spread is jitter-dominated, raise
  `alpha` or arc length before trusting the signal (CB6).
- **Cancellation drift.** Periodically recompute rollout grievance density and
  compare to the AnnoMI figure (§7); a widening gap means `Δ` is destabilizing and
  the delivery band is losing its anchor — re-run the [FT §8] probe.
- **Band-edge farming.** With realism as the only other guard, watch for arcs that
  hit the plateau via minimal marking (`d` just above `d_floor`, `q` at `L_design`).
  If frequent, raise `d_floor` / `L_design` or promote delivery density to a
  two-sided band (D-BAND.2).

---

## 11. Deliverables & acceptance criteria

**Deliverables:**
- `reward/band_reward.py` — §3–§5 (`band_reward_arc`, `axis_reward`, `band`,
  `density_factor`), with the CB5 import guard.
- `calibration/annomi_calibrate.py` — §6 offline pipeline: sample → (hand-label
  ingest) → grader-space measure → derive bracket → emit frozen
  `band_calibration.<grader_version>.yaml`.
- `calibration/band_calibration.<ver>.yaml` — §6.4 frozen artifact (hash-logged).
- Re-sync patches to `sway_grpo_simulator_finetune_spec.md`: §4 reward body, §5.2
  grouping, §6 filter note, and the void self-consistency rationale (§4.1 here).

**Acceptance:**
- **AB1** — `band_reward.py` passes the CB5 grep guard (no drift-side imports) and a
  unit test proving it reads only `(arc_turns, context, P, cell, cal)`.
- **AB2** — Load-time asserts fire on a bad artifact: `L_ext ≥ U` (CB3), `L_design <
  L_ext` (CB3), `U ≥ 1.0` or `L_design ≤ 0` (CB4).
- **AB3** — Anti-caricature unit test: an all-on-profile arc (`q = 1.0`) scores
  **strictly less** than an arc with `q ∈ [L_design, U]`, on both axes.
- **AB4** — Density test: an arc marked on `< d_floor` of turns with perfect `q`
  scores strictly less than one clearing `d_floor` with the same `q`.
- **AB5** — Cancellation wiring test: the artifact's `L_ext` is produced by the
  **same** backend object injected as the reward grader (CB1); a test with a
  deliberately biased stub shows target and rollout shift together and the reward's
  argmax stays at the true rate.
- **AB6** — Smoothing test: reward spread across G short arcs with identical policy
  behavior (jitter only) shrinks as `alpha` rises (CB6).
- **AB7** — On ≥1 backbone cell, GRPO under the band reward raises held-out fidelity
  vs the monotone-binary baseline **[FT A3]** without `q` collapsing to the caricature
  edge (§10).

---

## 12. Decision ledger

**Settled.**
- **D-BAND.1 axis combination = geometric mean, eps-floored** (was open, default
  average; resolved on measured advantage spread — see §0).
- Factorization: density (scenario) × conditional share (profile), scored separately;
  no marginal-product target.
- Two-class reduction → one scalar `q` per axis; no product-of-bumps.
- Conditioning defined on the **patient turn's own** marked label (grader output);
  the interlocutor affordance tag is dropped (was a marginal-only device).
- Reward shape = plateau band with soft shoulders, interior optimum; anti-caricature
  is the upper shoulder, not a separate term.
- Band bracket = AnnoMI grader-space lower bound + interiority ceiling; point inside
  is a disclosed design floor (D-BAND.3), never measured/simulator-derived.
- Instrument cancellation requires one frozen backend for calibration and reward (CB1);
  tied to the [FT §8] probe.
- Sequence-agnostic arc-marginal reward; **no pressure ladder / no phase segmentation**.
- Density floor closes the inert-simulator hole (soft ramp).
- Arc-level GRPO grouping for this reward (re-sync [FT §5.2]).

**Open (D-notes).**

- D-BAND.2 delivery density floor vs two-sided band.
- D-BAND.3 `L_design`/`U`/sigma values per cell (pilot-tuned).
- Whether to source the bracket from human **layoff-domain** transcripts instead of
  AnnoMI if obtainable (§6.5) — strictly better on transfer, gated on availability.

**Out of scope (owned elsewhere).**
- Realism floor mechanism, KL, curriculum, cert, VRAM — **[FT]**.
- Run-time gate, Judge, drift scoring, SYC/DEP — **[PIPE][BS][A][B]**; untouched.