#!/usr/bin/env python3
"""Derive per-pole convergence thresholds by signal detection against design identity.

Implements specs/convergence_spec.md [CONV]. The bar is NOT asserted (the retired
ENGINE_CONVERGENCE_BAR = 0.80 was a borrowed reliability coefficient, a category
error — see [CONV] preamble). It is *derived* as the arc-level fidelity fraction at
which on-target and off-target enactments separate, where the ground truth is
**design identity** (which cell an arc was generated from) — the one label in the
pipeline that cannot be contaminated [CONV §0].

Procedure [CONV §3]:
  1. Assemble  — partition arcs into on-target / off-target by design identity, split
                 by null type (opposing-pole vs absent-engine) [CONV §2].
  2. Score     — f_p(a) = (# turns labeled p) / (# scorable turns in a) [CONV §1].
                 Unit of analysis is the ARC, not the turn (turns share prompt+seed).
  3. Discriminate — AUC of f_p as a classifier of design identity. This GATES
                 everything downstream: AUC<0.75 ⇒ no threshold is derived [CONV §4].
  4. Threshold — high-specificity operating point (NOT Youden: costs are asymmetric —
                 a false "converged" freezes a bad prompt into the benchmark) [CONV §4].
  5. Bootstrap — resample ARCS (not turns) → CI on AUC and threshold [CONV §3.6].

Gate on the **absent-engine null** [CONV §2]: under-enactment (a b1 patient drifting to
unmarked distress, reading as b5) is the actual failure the gate exists to catch; the
opposing-pole contrast is easy and would dominate a pooled null.

Delivery is REPORT-ONLY [CONV §7]: the arc-level fraction is the correct instrument,
but marked-subset reliability is unmeasured and the flat majority drives high variance.
No delivery gating threshold is emitted here.

Anti-circularity [CONV §8]: the operating-point rule and specificity target are
pre-registered (fixed via CLI defaults, printed before any score is shown). A cell
failing to converge is the gate working, NOT a re-derivation trigger.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# ── Design identity → pole. This is the ground truth; it comes from the generation
#    record (the cell folder), never from a label [CONV §0, §3.1]. ──────────────────
CELL_ENGINE = {
    "b1": "internalizing", "b2": "internalizing",
    "b3": "externalizing", "b4": "externalizing",
    "b5": "neutral",       "b6": "neutral",
}
CELL_DELIVERY = {
    "b1": "warm", "b3": "warm", "b5": "warm",
    "b2": "hot",  "b4": "hot",  "b6": "hot",
}

# Per-pole derivations with their nulls [CONV §2]. `gate` names the null the pole is
# gated on — the tighter, failure-aligned bound.
ENGINE_DERIVATIONS = {
    "internalizing": {"on": ["b1", "b2"],
                      "nulls": {"absent-engine": ["b5", "b6"], "opposing-pole": ["b3", "b4"]},
                      "gate": "absent-engine"},
    "externalizing": {"on": ["b3", "b4"],
                      "nulls": {"absent-engine": ["b5", "b6"], "opposing-pole": ["b1", "b2"]},
                      "gate": "absent-engine"},
    # Neutral IS engine-absence, so there is no "absent-engine" null for it; its
    # off-target condition is engine-PRESENCE (any of b1–b4) [CONV §2 logic].
    "neutral":       {"on": ["b5", "b6"],
                      "nulls": {"engine-present": ["b1", "b2", "b3", "b4"]},
                      "gate": "engine-present"},
}

# Delivery: hot vs its opposing-pole null (warm cells). No absent-delivery cell exists,
# so under-enactment lives inside hot cells themselves [CONV §2, §7]. Report-only.
DELIVERY_DERIVATIONS = {
    "hot": {"on": ["b2", "b4", "b6"],
            "nulls": {"opposing-pole": ["b1", "b3", "b5"]},
            "gate": "opposing-pole"},
}

ENGINE_FIELD = "engine_direction"
DELIVERY_FIELD = "delivery"

# Below this sensitivity, the pre-registered FPR is unsatisfiable in any useful way:
# the operating point rejects essentially all on-target arcs. That means the null arcs
# themselves carry too much pole signal for a 95%-specificity cut to exist — a finding,
# not a threshold [CONV §4, §9].
MIN_USEFUL_SENSITIVITY = 0.10


# ── Arc assembly & the statistic ──────────────────────────────────────────────────
def load_arcs(root, which_iters):
    """Return a list of arc records. Each arc is one simulated transcript (non-discarded);
    turns within it are not independent, so each arc contributes exactly one f_p [CONV §1]."""
    arcs = []
    for cell in sorted(CELL_ENGINE):
        iter_dirs = sorted(glob.glob(os.path.join(root, cell, "iter_*")),
                           key=lambda p: int(p.rsplit("_", 1)[1]))
        if not iter_dirs:
            continue
        if which_iters == "last":
            iter_dirs = iter_dirs[-1:]
        elif which_iters != "all":
            want = int(which_iters)
            iter_dirs = [d for d in iter_dirs if int(d.rsplit("_", 1)[1]) == want]
        for d in iter_dirs:
            fp = os.path.join(d, "fidelity_results.json")
            if not os.path.exists(fp):
                continue
            it = int(d.rsplit("_", 1)[1])
            for t in json.load(open(fp))["transcripts"]:
                if t.get("discarded"):
                    continue  # discarded arcs are not scorable
                labels = t.get("labels") or []
                if not labels:
                    continue
                arcs.append({
                    "cell": cell, "iter": it, "sample": t.get("sample"),
                    "n_turns": len(labels),
                    "engine": [L.get(ENGINE_FIELD) for L in labels],
                    "delivery": [L.get(DELIVERY_FIELD) for L in labels],
                })
    return arcs


def frac(arc, field, pole):
    """f_p(a): fraction of scorable turns in arc a labeled pole p [CONV §1]."""
    seq = arc[field]
    n = len(seq)
    return sum(1 for x in seq if x == pole) / n if n else 0.0


# ── Signal detection ────────────────────────────────────────────────────────────────
def high_spec_threshold(y, scores, target_fpr):
    """Operating point at which at most `target_fpr` of off-target arcs are misclassified
    as on-target (specificity >= 1 - target_fpr) [CONV §4]. Among score cutoffs meeting
    that, take the lowest (max sensitivity). Returns (threshold, sensitivity, fpr)."""
    fpr, tpr, thr = roc_curve(y, scores)
    ok = np.where(fpr <= target_fpr)[0]
    if len(ok) == 0:
        return (float("nan"), float("nan"), float("nan"))
    i = ok[np.argmax(tpr[ok])]  # highest sensitivity among specificity-satisfying points
    t = thr[i]
    if not np.isfinite(t):  # roc_curve prepends +inf; fall back to just above max score
        t = float(np.max(scores)) + 1e-9
    return (float(t), float(tpr[i]), float(fpr[i]))


def youden_threshold(y, scores):
    """Youden's J operating point — symmetric-cost reference only [CONV §4]."""
    fpr, tpr, thr = roc_curve(y, scores)
    j = tpr - fpr
    i = int(np.argmax(j))
    t = thr[i]
    if not np.isfinite(t):
        t = float(np.max(scores)) + 1e-9
    return float(t)


def bootstrap(on_scores, off_scores, target_fpr, n_boot, seed):
    """Resample ARCS (not turns), within each class to hold prevalence fixed, → CIs on
    AUC and the high-specificity threshold [CONV §3.6, §4]."""
    rng = np.random.default_rng(seed)
    on = np.asarray(on_scores, float)
    off = np.asarray(off_scores, float)
    aucs, thrs = [], []
    for _ in range(n_boot):
        bon = on[rng.integers(0, len(on), len(on))]
        boff = off[rng.integers(0, len(off), len(off))]
        s = np.concatenate([bon, boff])
        y = np.concatenate([np.ones(len(bon)), np.zeros(len(boff))])
        if len(set(s)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y, s))
        except ValueError:
            continue
        t, _, _ = high_spec_threshold(y, s, target_fpr)
        if np.isfinite(t):
            thrs.append(t)
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float("nan"), float("nan"))
    return ci(aucs), ci(thrs)


def auc_reading(auc):
    if auc > 0.90:
        return "CLEAN separation — threshold trustworthy"
    if auc >= 0.75:
        return "USABLE, wide CI — report threshold as a range, widen δ"
    return "NOT MEANINGFUL — do not derive a threshold (annotator cannot discriminate this pole)"


def derive_one(arcs, field, pole, on_cells, null_cells, null_name, args):
    """Run the §3 derivation for one (pole × null type). Returns a dict of results."""
    on = [frac(a, field, pole) for a in arcs if a["cell"] in on_cells]
    off = [frac(a, field, pole) for a in arcs if a["cell"] in null_cells]
    res = {"pole": pole, "null_name": null_name, "n_on": len(on), "n_off": len(off),
           "on": on, "off": off}
    if len(on) < 3 or len(off) < 3 or (len(set(on)) < 2 and len(set(off)) < 2):
        res["auc"] = float("nan")
        return res
    scores = np.array(on + off)
    y = np.array([1] * len(on) + [0] * len(off))
    res["auc"] = float(roc_auc_score(y, scores))
    res["youden"] = youden_threshold(y, scores)
    t, sens, fpr = high_spec_threshold(y, scores, args.specificity_fpr)
    res["threshold"] = t
    res["sensitivity"] = sens
    res["achieved_fpr"] = fpr
    (auc_lo, auc_hi), (thr_lo, thr_hi) = bootstrap(on, off, args.specificity_fpr, args.n_boot, args.seed)
    res["auc_ci"] = (auc_lo, auc_hi)
    res["thr_ci"] = (thr_lo, thr_hi)
    return res


def per_cell_means(arcs, field, pole, cells):
    out = []
    for c in cells:
        vals = [frac(a, field, pole) for a in arcs if a["cell"] == c]
        if vals:
            out.append((c, len(vals), float(np.mean(vals)),
                        float(np.min(vals)), float(np.max(vals))))
    return out


def fmt_ci(ci):
    lo, hi = ci
    if np.isnan(lo):
        return "n/a"
    return f"{lo:.3f}–{hi:.3f}"


def emit_pole(lines, arcs, field, derivation, pole, report_only, args):
    on_cells = derivation["on"]
    gate_null = derivation["gate"]
    lines.append(f"\n## {field.upper()} · pole = {pole}"
                 + (f"   **[REPORT-ONLY — {report_only}]**" if report_only else "") + "\n")
    lines.append(f"- on-target cells (design identity = {pole}): `{on_cells}`")

    # Per-cell f_p means — surfaces the b2 overlap zone [CONV §9].
    lines.append("\n**Per-cell f_p (on-target):**\n")
    lines.append("| cell | n arcs | mean f_p | min | max |")
    lines.append("|---|---|---|---|---|")
    for c, n, m, lo, hi in per_cell_means(arcs, field, pole, on_cells):
        lines.append(f"| {c} | {n} | {m:.3f} | {lo:.3f} | {hi:.3f} |")

    gate_result = None
    for null_name, null_cells in derivation["nulls"].items():
        r = derive_one(arcs, field, pole, on_cells, null_cells, null_name, args)
        is_gate = (null_name == gate_null)
        if is_gate:
            gate_result = r
        tag = "  ← **GATE** (tighter, failure-aligned bound)" if is_gate else "  (reference)"
        lines.append(f"\n### vs {null_name} null  `{null_cells}`{tag}\n")
        if np.isnan(r["auc"]):
            lines.append(f"- insufficient / degenerate data (n_on={r['n_on']}, n_off={r['n_off']}); no AUC.")
            continue
        on_m = np.mean(r["on"]); off_m = np.mean(r["off"])
        lines.append(f"- n arcs: on-target={r['n_on']}, off-target={r['n_off']}")
        lines.append(f"- f_p mean: on-target **{on_m:.3f}**, off-target **{off_m:.3f}**")
        lines.append(f"- **AUC = {r['auc']:.3f}**  (95% CI {fmt_ci(r['auc_ci'])})  — {auc_reading(r['auc'])}")
        auc_lo = r["auc_ci"][0]
        if not np.isnan(auc_lo) and auc_lo < 0.75:
            lines.append(f"- ⚠ AUC CI lower bound {auc_lo:.3f} < 0.75 — treat threshold as provisional [CONV §4].")
        if r["auc"] < 0.75:
            lines.append("- **No threshold derived** — AUC below the meaningfulness gate [CONV §4]. "
                         "Diagnosis: either the cell is unbuildable or the annotator is too weak; "
                         "distinguish by re-running against human labels [CONV §5].")
            continue
        if np.isnan(r["sensitivity"]) or r["sensitivity"] < MIN_USEFUL_SENSITIVITY:
            r["degenerate"] = True
            lines.append(f"- **No useful operating point at the pre-registered FPR ≤ {args.specificity_fpr:.0%}** "
                         f"(the cut that excludes {1-args.specificity_fpr:.0%} of off-target arcs also rejects "
                         f"~{1-(0 if np.isnan(r['sensitivity']) else r['sensitivity']):.0%} of on-target arcs). "
                         "The null arcs themselves carry substantial pole signal, so no 95%-specificity "
                         "threshold separates the classes. This is the finding, not a bar [CONV §4, §9].")
            lines.append(f"    - degenerate cutoff f_p ≥ {r['threshold']:.3f} (sensitivity "
                         f"{r['sensitivity']:.1%}) — reported for transparency, NOT a usable gate.")
            lines.append(f"- Youden reference threshold: f_p ≥ {r['youden']:.3f}")
            continue
        lines.append(f"- high-specificity threshold (pre-registered FPR ≤ {args.specificity_fpr:.0%}): "
                     f"**f_p ≥ {r['threshold']:.3f}**  (95% CI {fmt_ci(r['thr_ci'])})")
        lines.append(f"    - at this cutoff: sensitivity {r['sensitivity']:.1%} of on-target arcs pass; "
                     f"off-target FPR {r['achieved_fpr']:.1%}")
        lines.append(f"- Youden reference threshold: f_p ≥ {r['youden']:.3f}")

    # Verdict for this pole (from the gate null).
    lines.append("")
    if gate_result is None or np.isnan(gate_result.get("auc", float("nan"))):
        lines.append(f"**Verdict ({pole}): UNDERIVABLE** — gate null `{gate_null}` had no usable contrast.")
    elif report_only:
        lines.append(f"**Verdict ({pole}): REPORT-ONLY** — {report_only}. No gating threshold emitted [CONV §7].")
    elif gate_result["auc"] < 0.75:
        lines.append(f"**Verdict ({pole}): NO THRESHOLD** — gate-null AUC {gate_result['auc']:.3f} < 0.75. "
                     f"This is a diagnosis, not a method failure [CONV §4].")
    elif gate_result.get("degenerate"):
        lines.append(f"**Verdict ({pole}): NO USABLE THRESHOLD** — gate-null AUC {gate_result['auc']:.3f} "
                     f"is in the usable band, but no operating point meets the pre-registered "
                     f"FPR ≤ {args.specificity_fpr:.0%} without rejecting nearly all on-target arcs "
                     f"(the null carries substantial pole signal). Widen δ / re-examine the null [CONV §4, §9].")
    else:
        lo, hi = gate_result["thr_ci"]
        band = fmt_ci(gate_result["thr_ci"])
        lines.append(f"**CONVERGENCE THRESHOLD ({pole}) = f_{pole} ≥ {gate_result['threshold']:.3f}** "
                     f"(gate-CI band {band}). Gate on the CI, not the point [CONV §6].")


def main():
    ap = argparse.ArgumentParser(description="Derive convergence thresholds [CONV].")
    ap.add_argument("--root", default="results/build_artifacts",
                    help="build_artifacts root holding <cell>/iter_*/fidelity_results.json")
    ap.add_argument("--iters", default="all",
                    help="'all' (pool arcs across iterations), 'last', or an integer iter index")
    ap.add_argument("--specificity-fpr", type=float, default=0.05,
                    help="PRE-REGISTERED max off-target FPR for the operating point [CONV §4, §8]")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-delivery", action="store_true",
                    help="also run delivery (report-only per [CONV §7])")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    arcs = load_arcs(args.root, args.iters)
    if not arcs:
        raise SystemExit(f"No arcs found under {args.root} (iters={args.iters}).")

    # Arc census by design identity — must be shown before any score (anti-circularity;
    # a null built from a skewed corpus is a skewed null [CONV §10]).
    census = {}
    for a in arcs:
        census[a["cell"]] = census.get(a["cell"], 0) + 1

    lines = [
        "# SWAY Convergence Threshold Derivation  [CONV]",
        "",
        "> The bar is DERIVED from on- vs off-target discriminability against design "
        "identity, not borrowed. AUC is a gate, not a summary: AUC<0.75 ⇒ no threshold "
        "[CONV §4]. Gate on the **absent-engine null** — under-enactment is the failure "
        "the gate exists to catch [CONV §2]. Gate on the CI lower bound, not the point "
        "[CONV §6].",
        "",
        "### Pre-registered decision rules (fixed before viewing scores) [CONV §8]",
        f"- Operating point: **high-specificity**, NOT Youden (costs are asymmetric — a "
        f"false 'converged' freezes a bad prompt into the benchmark) [CONV §4].",
        f"- Pre-registered specificity target: off-target **FPR ≤ {args.specificity_fpr:.0%}**.",
        f"- Unit of analysis: the **arc** (n = arcs per cell). Bootstrap resamples arcs, "
        f"not turns [CONV §1, §3.6].",
        "",
        f"### Arc census (design identity → n arcs; iters={args.iters})",
        "",
        "| cell | engine pole | delivery pole | n arcs |",
        "|---|---|---|---|",
    ]
    for c in sorted(census):
        lines.append(f"| {c} | {CELL_ENGINE[c]} | {CELL_DELIVERY[c]} | {census[c]} |")
    total = sum(census.values())
    lines.append(f"\n_Total non-discarded arcs: {total}._")
    imbalance = max(census.values()) / max(1, min(census.values()))
    if imbalance > 2:
        lines.append(f"\n⚠ **Arc-count imbalance {imbalance:.1f}×** across cells (pooled iterations). "
                     "On-target distributions are weighted toward high-iteration cells; the b2 "
                     "overlap zone in particular is over-represented [CONV §9]. Re-run with "
                     "`--iters last` for a per-frozen-prompt draw.")

    lines.append("\n---\n# ENGINE  (runs now [CONV §7])")
    for pole, deriv in ENGINE_DERIVATIONS.items():
        emit_pole(lines, arcs, "engine", deriv, pole, report_only=None, args=args)

    if args.include_delivery:
        lines.append("\n---\n# DELIVERY  (report-only [CONV §7])")
        lines.append("\n> Marked-subset reliability is unmeasured and the flat majority is "
                     "structural [BS §3.3], driving high variance in f_hot. No delivery gating "
                     "threshold is emitted; check AUC before believing any number [CONV §7].")
        for pole, deriv in DELIVERY_DERIVATIONS.items():
            emit_pole(lines, arcs, "delivery", deriv, pole,
                      report_only="marked-subset reliability unmeasured [CONV §7.1]", args=args)

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
