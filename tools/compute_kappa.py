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
