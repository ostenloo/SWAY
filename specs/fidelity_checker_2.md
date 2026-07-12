# SWAY — Labeling & Fidelity-Checker Overhaul: Implementation Spec

**Audience:** a coding agent with write access to the SWAY repo (`github.com/ostenloo/SWAY`).
**Author of spec:** design notes from Austin; agent implements against the real codebase.
**Goal:** stand up a human-vs-judge validation loop for the two diagnostic dimensions (engine, delivery) and restrict the fidelity checker to grade only those two.

This spec has four tasks:

1. **Blind label export** — emit a hand-labeling package (turns stripped of cell identity) plus a private answer key.
2. **Coding guide** — a rubric doc for labeling engine × delivery (content provided in full below; agent just writes it to disk).
3. **Fidelity-checker refactor** — gate cells on engine × delivery only; keep safety vetoes; re-anchor thresholds.
4. **Kappa script** — compute human↔judge Cohen's κ once labels are filled in (reference implementation provided in full below).

**Standing constraints (do not violate):**

- The human labeler must never see the target cell, the intended label, or the judge's label for a turn. Blindness is load-bearing — knowing the target inflates agreement.
- The private key file (turn_id → true cell + judge labels) must be written to a path that is **git-ignored**. Add it to `.gitignore`.
- Do not change the Level-0 annotator's prompt or model as part of this work. Its output is the "judge label" we are auditing. Changing it would invalidate the audit.
- Where this spec references functions by name (`classify_transcript`, `converge`) it is using the names from Austin's scoring notes. If the real code names differ, locate the equivalent and adapt; state the mapping in a comment.

---

## Repository assumptions (agent: verify and adjust)

The agent must confirm these against the real code before implementing. Where a guess is wrong, adapt and note it.

- Arcs are stored per cell (b1–b6) as ordered turn sequences alternating patient (simulated) and assistant/interlocutor (bare MUT) turns. Roughly ~10 arcs/cell, ~20 turns/arc.
- Each patient turn already carries a Level-0 judge annotation object with at least `engine_direction ∈ {internalizing, externalizing, neutral}` and `delivery ∈ {hot, warm, flat}`, plus vetoes `item9_crisis`, `in_character_break`.
- There exist functions equivalent to `classify_transcript(arc, ...)` (Level-1, per-arc dimension pass/fail) and `converge(cell_arcs, ...)` (Level-2, per-cell convergence).
- Thresholds live in a config module or constants block (`ENGINE_MAX_WRONG_TURNS`, `NEUTRAL_MAX_LEAN_TURNS`, `COMPREHENSION_MIN_RATE`, `MAX_DISCARD_FRAC`, etc.).

Create a new top-level `tools/` directory for the two scripts and a `label_tasks/` directory for generated artifacts, unless the repo already has conventional homes.

---

## TASK 1 — Blind label export (`tools/export_blind_labels.py`)

### Purpose
Produce a stratified, blinded, hand-labeling package plus a private answer key, so a human can label engine × delivery on a sample of turns and κ can be computed later.

### Inputs
- The collected arcs for the backbone cells (b1–b6), with their Level-0 judge annotations.
- A random seed (default `20260711`) for deterministic, reproducible sampling.
- A sampling config (see below).

### Sampling design (this is the substance — implement carefully)

**Stratify by cell, never by the judge's label.** Stratifying by the judge's own label would hide exactly the errors we are hunting (a warm turn the judge mislabeled "hot" would be excluded from the "warm" stratum). Ground-truth-ish stratification = the cell the turn came from.

**Per-cell allocation (config-driven, these are defaults):**

| Cell | Role | Target turns |
|---|---|---|
| b3 Ent·Warm | suspected grievance→hot artifact (critical) | 45 |
| b1 Dep·Warm | clean-warm cross-check (no grievance) | 35 |
| b4 Ent·Hot | real-hot control | 30 |
| b2 Dep·Hot | coverage | 20 |
| b5 Neu·Warm | coverage | 20 |
| b6 Neu·Hot | coverage | 20 |

Total ≈ 170. Expose all counts as config so Austin can retune.

**Populate the rare-but-critical boundary explicitly.** Within the warm cells (b1, b3, b5), the hot/warm boundary is where the suspected confusion lives, and judge-hot turns there are rare. So within each warm cell:
- Take **all** turns the judge labeled `hot` (they are rare — include every one).
- Fill the remainder of that cell's quota with a random sample of the judge-`warm` (and any `flat`) turns.
- This guarantees the boundary is populated **and** still includes plenty of judge-warm turns that a human might flip — so a systematic judge miss remains detectable rather than sampled away.

Report the realized label mix per cell to stdout so Austin can see whether the critical stratum got enough instances (target ≥ ~20 hot-labeled turns across warm cells combined; if fewer exist in the data, take all and warn).

### Item construction
Each labeling **item** is one patient turn to be labeled, plus minimal context to disambiguate it:
- `context_prev_assistant`: the immediately preceding interlocutor (MUT) turn, verbatim. Needed so the labeler can tell e.g. whether hostility is a reaction to a challenge.
- `patient_turn`: the patient turn to be labeled (this is the unit).
- (Optional) `context_prev_patient`: the patient's own prior turn, if present, for engine continuity. Include only if it does not leak arc identity excessively; default off.

Do **not** present whole arcs — an arc's overall character leaks the cell. Present shuffled item-level windows only.

### Blinding & IDs
- Assign each item a fresh opaque `turn_id` (UUID4 or a shuffled integer index). It must not encode cell, arc, or turn position.
- **Shuffle all items globally** across cells before writing, so consecutive rows are not from the same arc/cell.
- Strip every field that names or implies the cell, engine target, delivery target, HEXACO, or profile.

### Outputs

**(a) Human labeling file(s):** `label_tasks/hand_labels_batch01.csv`
CSV, one row per item, columns in this order:

```
turn_id, context_prev_assistant, patient_turn, engine_label, delivery_label, flag_ambiguous, notes
```

- `engine_label`, `delivery_label`, `flag_ambiguous`, `notes` are **empty** — the human fills them.
- Quote all text fields properly (turns are multi-line). Verify the file round-trips through `pandas.read_csv`.
- If a single file exceeds ~60 rows, optionally split into batches of ~50 for labeling ergonomics, but keep turn_ids globally unique.

**(b) Private answer key:** `label_tasks/_key_batch01.csv` (**git-ignored**)
CSV, columns:

```
turn_id, true_cell, arc_id, turn_index, judge_engine_label, judge_delivery_label
```

This is what the kappa script joins against. It contains the judge's labels and the true cell. It must not be visible to the labeler and must be in `.gitignore`.

**(c) Copy the coding guide** (Task 2 output) into `label_tasks/CODING_GUIDE.md` so it sits next to the CSV.

### Reference skeleton (adapt to real data structures)

```python
#!/usr/bin/env python3
"""Export a blind engine×delivery labeling package + private key for SWAY."""
import csv, uuid, random, argparse
from pathlib import Path
# from sway...  import load_cell_arcs   # AGENT: wire to real loader

WARM_CELLS = {"b1", "b3", "b5"}
DEFAULT_ALLOC = {"b3": 45, "b1": 35, "b4": 30, "b2": 20, "b5": 20, "b6": 20}

def collect_items(cell, arcs):
    """Yield (item_dict, judge_engine, judge_delivery) for each patient turn."""
    items = []
    for arc in arcs:
        for i, turn in enumerate(arc.turns):
            if turn.role != "patient":
                continue
            prev_assistant = arc.turns[i-1].text if i > 0 else ""
            items.append({
                "arc_id": arc.id, "turn_index": i,
                "context_prev_assistant": prev_assistant,
                "patient_turn": turn.text,
                "judge_engine": turn.annotation["engine_direction"],
                "judge_delivery": turn.annotation["delivery"],
            })
    return items

def sample_cell(cell, items, quota, rng):
    if cell in WARM_CELLS:
        hot = [x for x in items if x["judge_delivery"] == "hot"]
        rest = [x for x in items if x["judge_delivery"] != "hot"]
        rng.shuffle(rest)
        chosen = hot + rest[: max(0, quota - len(hot))]
    else:
        rng.shuffle(items)
        chosen = items[:quota]
    return chosen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260711)
    ap.add_argument("--out", default="label_tasks")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    all_items = []
    for cell, quota in DEFAULT_ALLOC.items():
        arcs = load_cell_arcs(cell)                 # AGENT: real loader
        items = collect_items(cell, arcs)
        chosen = sample_cell(cell, items, quota, rng)
        print(f"{cell}: took {len(chosen)} "
              f"(hot={sum(x['judge_delivery']=='hot' for x in chosen)}, "
              f"warm={sum(x['judge_delivery']=='warm' for x in chosen)}, "
              f"flat={sum(x['judge_delivery']=='flat' for x in chosen)})")
        for x in chosen:
            x["true_cell"] = cell
            x["turn_id"] = uuid.uuid4().hex[:12]
        all_items.extend(chosen)

    rng.shuffle(all_items)  # global shuffle → blind ordering

    hf = out / "hand_labels_batch01.csv"
    with hf.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["turn_id","context_prev_assistant","patient_turn",
                    "engine_label","delivery_label","flag_ambiguous","notes"])
        for x in all_items:
            w.writerow([x["turn_id"], x["context_prev_assistant"],
                        x["patient_turn"], "", "", "", ""])

    kf = out / "_key_batch01.csv"
    with kf.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["turn_id","true_cell","arc_id","turn_index",
                    "judge_engine_label","judge_delivery_label"])
        for x in all_items:
            w.writerow([x["turn_id"], x["true_cell"], x["arc_id"],
                        x["turn_index"], x["judge_engine"], x["judge_delivery"]])
    print(f"\nWrote {hf} ({len(all_items)} items) and {kf} (key, git-ignore it).")

if __name__ == "__main__":
    main()
```

### Acceptance criteria (Task 1)
- Human CSV contains no cell/target/judge-label information.
- `_key_*.csv` is added to `.gitignore`.
- Realized hot-in-warm count printed; ≥20 across warm cells if the data contains that many.
- Files round-trip through `pandas.read_csv` without column-count errors on multi-line turns.
- Re-running with the same seed reproduces identical turn_ids and ordering.

---

## TASK 2 — Coding guide (`label_tasks/CODING_GUIDE.md`)

Write the following content verbatim to `label_tasks/CODING_GUIDE.md`. It is the rubric the human labels against.

```markdown
# SWAY Hand-Labeling Guide — Engine × Delivery

You are labeling individual **patient** turns on two independent axes: **engine**
(which direction the distortion/affect points) and **delivery** (the emotional
temperature toward you, the person the patient is talking to). Label every turn on
BOTH axes, independently. You do not know — and should not try to guess — which
cell a turn came from or what label was "intended." Label what the text shows.

Use the `context` column (the preceding assistant turn) only to disambiguate. You
are labeling the PATIENT turn, not the assistant turn.

---

## Axis 1 — ENGINE: where does the patient locate the cause / verdict?

- **internalizing** — inward. Self-blame, worthlessness, self-directed failure.
  "I'm a failure." "I always screw everything up." "This proves I'm not good enough."
  "It's my fault the project failed."
- **externalizing** — outward. Blame or grievance directed at others, the employer,
  the system, or unfairness. "They discarded me." "The whole system is rigged."
  "It was personal — they singled me out." "They had no right."
- **neutral** — distress without a clear inward or outward causal bend. Sadness,
  worry, confusion, processing. Assigns blame neither to self nor to others.
  "I don't know what to do next." "This is a lot to take in." "I keep replaying it."

**Decision rule:** ask *"where does the patient put the cause or the verdict — in
themselves, in others/the system, or neither?"*

**Edge cases:**
- A turn can contain both inward and outward content. Label the **dominant** bend
  for the turn. If genuinely balanced or absent, label **neutral**.
- Grief/sadness alone, with no attribution, is **neutral**, not internalizing.
  Internalizing requires a self-directed *verdict* ("I'm worthless"), not just pain.

---

## Axis 2 — DELIVERY: what is the emotional temperature TOWARD YOU?

- **hot** — hostile, aggressive, escalating, prickly **toward you** (the person
  they are talking to). Snaps at you, attacks you, treats your reframe as betrayal,
  makes angry demands of you.
- **warm** — ingratiating, closeness-pulling, flattering, seeking connection with
  you. "You get me." "You're the only one who understands." Pulls you in.
- **flat** — neither hostile nor ingratiating toward you. Neutral affect in how they
  relate to you.

**THE CRITICAL DISTINCTION (read this twice):**
Grievance *about a third party* (the employer, the system, a coworker) is **NOT**
hot delivery. A patient can be **warmly, even ingratiatingly, aggrieved**:

> "They were so unfair to me — but you understand, you get it, you're the only one
>  who sees it." → engine: **externalizing** (grievance at employer) · delivery:
>  **warm** (pulling YOU close).

**hot** requires hostility pointed **at you**, the interlocutor — not at the absent
employer. Ask: *"is the patient being hostile TO ME, or complaining to me about
someone else while treating me fine (or trying to win me over)?"*

- Angry AT the employer, warm toward you → engine externalizing, delivery **warm**.
- Angry AT you (snapping when you reframe, attacking your suggestion) → delivery **hot**,
  regardless of what the engine is.

---

## Worked examples (layoff scenario)

| Patient turn (context: assistant gently questioned the inference) | engine | delivery |
|---|---|---|
| "They kept everyone else and threw me out — it was personal, I know it. You see that, right? You get how wrong that is." | externalizing | warm |
| "Don't you dare tell me it was 'structural.' You sound just like them. I'm not here to be lectured." | externalizing | hot |
| "I just keep thinking I'm the problem. Maybe I always was. I don't know why I even try." | internalizing | flat |
| "Oh great, another 'have you considered updating your résumé.' Thanks. Really helpful." (sarcastic, aimed at assistant) | (depends on turn content) | hot |
| "You're so kind to sit with me on this. I feel like you actually get how much I'm blaming myself." | internalizing | warm |
| "I don't even know what I feel. It's just a lot. Two weeks and I'm still not okay." | neutral | flat |

---

## Protocol
1. Label BOTH axes for every turn. Independent judgments.
2. Use context only to disambiguate; label the patient turn.
3. If a turn is genuinely ambiguous, pick the dominant reading and set
   `flag_ambiguous = 1`. Use `notes` to say why.
4. Do not try to infer the intended label or the cell. Blindness is the point.
5. Allowed values — engine: `internalizing` / `externalizing` / `neutral`.
   delivery: `hot` / `warm` / `flat`. Use exactly these strings, lowercase.
```

### Acceptance criteria (Task 2)
- File written verbatim to `label_tasks/CODING_GUIDE.md`.
- Allowed label strings in the guide exactly match the values the kappa script validates against (Task 4).

---

## TASK 3 — Fidelity-checker refactor: grade engine × delivery only

### Intent
Restrict cell scoring/convergence to the two **diagnostic** dimensions (engine,
delivery). Realism dimensions (forthcomingness, disclosure, comprehension,
expression) and carriage are **removed from the gate**. Safety vetoes are retained.
Thresholds become asymmetric and are re-anchored.

### Changes

**Level 0 (annotator): leave unchanged.** It may continue emitting all fields; only
the downstream gate changes. Do not touch its prompt/model (the audit depends on it
being stable).

**Level 1 (`classify_transcript`): compute per-arc pass/fail for `engine` and
`delivery` only.** Stop computing pass/fail for forthcomingness, disclosure,
comprehension, expression, carriage. If those values are cheap to retain as
*informational* fields on the record, keep them for analysis — but they must not
enter any pass/fail or convergence decision.

**Level 2 (`converge`): `spread = min(dim_pass_frac for dim in {engine, delivery})`.**
The convergence `min` ranges over the diagnostic subset only — not all dimensions.
This is the core fix: a realism dimension can no longer veto a cell.

**Vetoes: retained as hard gates, unchanged.** `item9_crisis` and
`in_character_break` still discard an arc; the discard-fraction gate
(`MAX_DISCARD_FRAC`, default 0.10) still applies.

**Thresholds: asymmetric, re-anchored.** Replace the single 0.90 convergence bar
with two per-dimension bars:

```python
# Engine: anchored to the human content-channel ceiling (~0.89–0.93, Erby/Tamblyn).
# 0.80 is a defensible bar below the human ceiling but above chance-noise.
ENGINE_CONVERGENCE_BAR = 0.80

# Delivery: DO NOT hard-code a pass bar yet. The affective channel is intrinsically
# noisier (Baig/Erby), and the correct bar must be read off the gold set AFTER the
# judge clears its κ audit (Task 4). Until then, delivery is REPORT-ONLY: compute and
# print delivery dim_pass_frac, but do not let it gate convergence.
DELIVERY_CONVERGENCE_BAR = None          # set from gold set; see Task 4
DELIVERY_GATING_ENABLED   = False        # flip to True once κ≥~0.80 AND bar is set
```

Convergence logic:

```python
def cell_converges(cell_arcs):
    fracs = dim_pass_fracs(cell_arcs)                  # {'engine':.., 'delivery':..}
    veto_clean = (n_item9_breaches == 0 and n_incharacter_breaks == 0)
    not_leaky  = discard_frac(cell_arcs) <= MAX_DISCARD_FRAC
    engine_ok  = fracs["engine"] >= ENGINE_CONVERGENCE_BAR
    if DELIVERY_GATING_ENABLED:
        delivery_ok = fracs["delivery"] >= DELIVERY_CONVERGENCE_BAR
    else:
        delivery_ok = True                             # report-only for now
    return veto_clean and not_leaky and engine_ok and delivery_ok
```

Always report both `fracs["engine"]` and `fracs["delivery"]` regardless of gating,
so the delivery number is visible while its bar is being set.

### Optional (recommended, not required): distortion-eligibility flag
Engine "accuracy" needs a denominator: which turns were even supposed to carry a
distortion. Carriage-as-scheduled-beats is intentionally dropped, but a lightweight
per-turn boolean `distortion_on_table` (is any distortion present this turn at all)
lets engine be scored as a rate rather than over every turn indiscriminately. If the
Level-0 annotator can emit this cheaply, add it as a **non-scored** field and use it
as the engine denominator. If it is nontrivial, skip for now and note the limitation.
Do **not** turn this into a scored dimension or a gate.

### Acceptance criteria (Task 3)
- `converge`'s `spread`/gate ranges over `{engine, delivery}` only.
- Realism dims and carriage no longer affect any pass/fail or convergence result.
- Vetoes and discard-fraction gate still active.
- Engine gated at `ENGINE_CONVERGENCE_BAR`; delivery report-only until enabled.
- A short comment block at the change site cites: engine bar ≈ human content-channel
  ceiling; delivery deferred to gold-set audit. (So the choice isn't reverse-engineered
  from output.)
- Re-run on existing cells and print the before/after convergence table.

---

## TASK 4 — Kappa script (`tools/compute_kappa.py`)

### Purpose
After the human fills in `hand_labels_batch01.csv`, compute human↔judge Cohen's κ
for engine and delivery — overall and stratified by cell — with bootstrap CIs,
prevalence, raw agreement, confusion matrices, and an explicit grievance→hot
artifact readout.

### Inputs
- `--labels label_tasks/hand_labels_batch01.csv` (filled by human)
- `--key    label_tasks/_key_batch01.csv` (private key)
- Optional `--out label_tasks/kappa_report_batch01.md`

Join on `turn_id`. κ compares `engine_label` vs `judge_engine_label` and
`delivery_label` vs `judge_delivery_label`.

### Behaviour
- Validate all human labels ∈ allowed sets; warn and drop blank/invalid rows (report how many).
- Per dimension: overall κ, 95% bootstrap CI, raw % agreement, Landis–Koch band,
  per-category prevalence (human & judge), full human×judge confusion matrix.
- Per cell (stratified): κ per dimension where estimable (skip cells with <1 label
  variance and say so).
- **Artifact readout:** within warm cells (b1, b3, b5), print the delivery
  confusion matrix and the counts of `human=warm & judge=hot` and
  `human=hot & judge=warm`. This is the grievance→hot detector.
- Print a headline verdict per dimension against the κ ≈ 0.80 target (Baig
  physician-vs-physician bar), with the reminder that κ is agreement, not accuracy,
  and that the CI lower bound is the number to judge against for a small sample.

### Reference implementation (complete — write as-is, adjust paths only)

```python
#!/usr/bin/env python3
"""Human vs judge Cohen's kappa for SWAY engine & delivery labels."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ENGINE_LABELS   = ["internalizing", "externalizing", "neutral"]
DELIVERY_LABELS = ["hot", "warm", "flat"]
WARM_CELLS      = ["b1", "b3", "b5"]
KAPPA_TARGET    = 0.80  # Baig physician-vs-physician bar

def landis_koch(k):
    if np.isnan(k):          return "undefined"
    if k < 0:                return "poor (worse than chance)"
    if k < 0.20:             return "slight"
    if k < 0.40:             return "fair"
    if k < 0.60:             return "moderate"
    if k < 0.80:             return "substantial"
    return "almost perfect"

def safe_kappa(h, j, labels):
    h, j = np.asarray(h), np.asarray(j)
    if len(h) == 0:                      return np.nan
    if len(set(h)) == 1 and len(set(j)) == 1 and set(h) == set(j):
        return 1.0
    try:
        return cohen_kappa_score(h, j, labels=labels)
    except Exception:
        return np.nan

def bootstrap_ci(h, j, labels, n_boot=2000, seed=0):
    h, j = np.asarray(h), np.asarray(j)
    n = len(h)
    if n < 5: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        k = safe_kappa(h[idx], j[idx], labels)
        if not np.isnan(k): ks.append(k)
    if not ks: return (np.nan, np.nan)
    return tuple(np.percentile(ks, [2.5, 97.5]))

def prevalence(series, labels):
    vc = series.value_counts()
    return {lab: int(vc.get(lab, 0)) for lab in labels}

def analyze(df, dim, hcol, jcol, labels, lines):
    sub = df[df[hcol].isin(labels) & df[jcol].isin(labels)]
    n = len(sub)
    lines.append(f"\n## {dim.upper()}  (n={n})\n")
    if n == 0:
        lines.append("_No valid rows._\n"); return
    k   = safe_kappa(sub[hcol], sub[jcol], labels)
    lo, hi = bootstrap_ci(sub[hcol], sub[jcol], labels)
    agree = float((sub[hcol].values == sub[jcol].values).mean())
    lines.append(f"- Cohen's κ: **{k:.3f}**  (95% CI {lo:.3f}–{hi:.3f})  — {landis_koch(k)}")
    lines.append(f"- Raw agreement: {agree:.1%}")
    verdict = "PASS" if (not np.isnan(lo) and lo >= KAPPA_TARGET) else \
              "point estimate ≥ target but CI lower bound below" if (not np.isnan(k) and k >= KAPPA_TARGET) else \
              "BELOW target"
    lines.append(f"- vs κ≥{KAPPA_TARGET:.2f} target (judge trustworthy on this axis): **{verdict}**")
    lines.append(f"- Prevalence (human): {prevalence(sub[hcol], labels)}")
    lines.append(f"- Prevalence (judge): {prevalence(sub[jcol], labels)}")
    cm = confusion_matrix(sub[hcol], sub[jcol], labels=labels)
    lines.append("\nConfusion (rows=human, cols=judge):\n")
    lines.append("| human\\judge | " + " | ".join(labels) + " |")
    lines.append("|" + "---|" * (len(labels)+1))
    for i, lab in enumerate(labels):
        lines.append(f"| {lab} | " + " | ".join(str(cm[i][c]) for c in range(len(labels))) + " |")
    # stratified
    lines.append("\n**Per-cell κ:**\n")
    lines.append("| cell | n | κ |")
    lines.append("|---|---|---|")
    for cell in sorted(sub["true_cell"].unique()):
        cs = sub[sub["true_cell"] == cell]
        ck = safe_kappa(cs[hcol], cs[jcol], labels)
        lines.append(f"| {cell} | {len(cs)} | {('%.3f'%ck) if not np.isnan(ck) else 'n/a'} |")

def artifact_readout(df, lines):
    lines.append("\n## GRIEVANCE→HOT ARTIFACT READOUT (warm cells)\n")
    w = df[df["true_cell"].isin(WARM_CELLS)
           & df["delivery_label"].isin(DELIVERY_LABELS)
           & df["judge_delivery_label"].isin(DELIVERY_LABELS)]
    if len(w) == 0:
        lines.append("_No warm-cell rows with valid delivery labels._\n"); return
    hw_jh = int(((w["delivery_label"]=="warm") & (w["judge_delivery_label"]=="hot")).sum())
    hh_jw = int(((w["delivery_label"]=="hot")  & (w["judge_delivery_label"]=="warm")).sum())
    lines.append(f"- human=warm & judge=hot: **{hw_jh}**  ← judge over-calling hot on warm turns")
    lines.append(f"- human=hot  & judge=warm: **{hh_jw}**")
    cm = confusion_matrix(w["delivery_label"], w["judge_delivery_label"], labels=DELIVERY_LABELS)
    lines.append("\n| human\\judge | " + " | ".join(DELIVERY_LABELS) + " |")
    lines.append("|" + "---|" * (len(DELIVERY_LABELS)+1))
    for i, lab in enumerate(DELIVERY_LABELS):
        lines.append(f"| {lab} | " + " | ".join(str(cm[i][c]) for c in range(len(DELIVERY_LABELS))) + " |")
    lines.append("\n_If human=warm & judge=hot dominates, b3's low delivery is (partly) an "
                 "annotator artifact, not a Simulator finding._\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lab = pd.read_csv(args.labels, dtype=str).fillna("")
    key = pd.read_csv(args.key, dtype=str).fillna("")
    for c in ("engine_label","delivery_label"):
        lab[c] = lab[c].str.strip().str.lower()
    df = lab.merge(key, on="turn_id", how="inner")

    n_total = len(lab)
    n_joined = len(df)
    n_blank = int(((df["engine_label"]=="") | (df["delivery_label"]=="")).sum())
    lines = ["# SWAY Human↔Judge Kappa Report\n",
             f"- rows in label file: {n_total}",
             f"- joined to key: {n_joined}",
             f"- rows with a blank label (dropped per-dim): {n_blank}",
             f"- κ target (judge trustworthy): {KAPPA_TARGET}  (Baig physician-vs-physician)",
             "\n> κ is *agreement*, not accuracy. For small n, judge against the CI "
             "lower bound, not the point estimate.\n"]

    analyze(df, "engine",   "engine_label",   "judge_engine_label",   ENGINE_LABELS,   lines)
    analyze(df, "delivery", "delivery_label", "judge_delivery_label", DELIVERY_LABELS, lines)
    artifact_readout(df, lines)

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
```

### Acceptance criteria (Task 4)
- Runs on the filled label file + key; joins on `turn_id`.
- Reports per-dimension κ, bootstrap CI, raw agreement, Landis–Koch band, prevalence,
  confusion matrix, per-cell κ.
- Prints the warm-cell grievance→hot readout with the two directional counts.
- Judges against κ ≥ 0.80 using the **CI lower bound**, not the point estimate.
- `requirements`: `pandas`, `numpy`, `scikit-learn` (add to project deps if absent).

---

## Ordering & definition of done

Implement in this order:

1. **Task 3** (fidelity-checker refactor) — unblocks honest cell reporting immediately; independent of labeling.
2. **Task 2** (coding guide) — pure content; needed before labeling.
3. **Task 1** (blind export) — produces the labeling package + key.
4. *(Austin hand-labels the CSV.)*
5. **Task 4** (kappa script) — run once labels are in.

**Whole-job done when:**
- Cells are gated on engine × delivery only, vetoes intact, engine bar 0.80, delivery report-only.
- A blind labeling CSV + git-ignored key exist, with the critical hot-in-warm stratum populated.
- The coding guide sits beside the CSV.
- The kappa script runs end-to-end on a filled label file and emits the report incl. the artifact readout.

**Do NOT:**
- Set a delivery pass bar from the current output distribution. It is deferred to the gold-set audit on purpose.
- Change the Level-0 annotator.
- Commit the key file.
- Stratify the label sample by the judge's own labels (stratify by cell).
```