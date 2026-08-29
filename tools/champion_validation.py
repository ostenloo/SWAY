#!/usr/bin/env python3
"""Champion agreement with human annotators ON SIMULATOR TEXT.

[EBM] moves the §7.2 gold subset upstream of the §6 reachability gate: before any
rate derived from the champions means anything, the instrument has to be shown to
read the same construct on simulator text that it reads on AnnoMI. The champions
were calibrated on AnnoMI; the controller and the reward apply them to text with a
different register, a different length distribution, and an explicit instruction to
carry a bound distortion. Out-of-distribution classifier behaviour on a markedness
question is exactly the failure that would inflate a marked rate to saturation.

The measurement already existed: `label_tasks/batch03` is 150 simulator turns,
balanced 25 per backbone cell, hand-labeled by three annotators against the same
rubric used on AnnoMI (`label_tasks/CODING_GUIDE.md`), with a frozen-champion key.

Pure CSV analysis — no model calls, runs anywhere.

    python tools/champion_validation.py
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BATCH = REPO / "label_tasks" / "batch03"
CELLS = ("b1", "b2", "b3", "b4", "b5", "b6")

#: Annotator sheets are DISCOVERED, never hardcoded, and the default report
#: ANONYMISES them. This repository is public and the sheets are per-person work;
#: a table of "which named collaborator agreed with the grader least" is not
#: something to publish as a side effect of a validation run. Pass
#: --show-annotators for the local, named version.
SHEET_GLOB = "hand_labels_batch03_*.csv"

#: axis -> (unmarked label, champion-key column, hand-sheet column)
AXES = {
    "engine": ("neutral", "judge_engine_label", "engine_label"),
    "delivery": ("flat", "judge_delivery_label", "delivery_label"),
}

#: Typos observed in the sheets. Normalised rather than dropped — discarding a
#: rater's row silently changes the denominator.
SPELLING = {"externilazing": "externalizing"}


def _norm(x: str | None) -> str:
    x = (x or "").strip().lower()
    return SPELLING.get(x, x)


def _read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def discover_annotators(batch: Path = BATCH) -> list[str]:
    """Sheet stems, sorted. `hand_labels_batch03_<who>.csv` -> `<who>`."""
    out = []
    for p in sorted(batch.glob(SHEET_GLOB)):
        who = p.stem[len("hand_labels_batch03_"):]
        if who and "copy" not in who.lower():
            out.append(who)
    return out


def load(batch: Path = BATCH) -> tuple[dict, dict, dict, list[str]]:
    champ = {r["turn_id"]: r for r in _read(batch / "_key_batch03.champions_cmdr7b-glm4.csv")}
    meta = {r["turn_id"]: r for r in _read(batch / "_key_batch03.csv")}
    humans = {
        w: {r["turn_id"]: r for r in _read(batch / f"hand_labels_batch03_{w}.csv")}
        for w in discover_annotators(batch)
    }
    if not humans:
        raise SystemExit(
            f"no annotator sheets matching {SHEET_GLOB} under {batch}.\n"
            "The batch03 hand-label sheets are not tracked in git — they hold "
            "per-person annotation work and this repository is public. Point --batch "
            "at the directory holding them."
        )
    ids = [t for t in champ if t in meta and all(t in h for h in humans.values())]
    return champ, meta, humans, sorted(ids)


def cohen_kappa(a, b) -> float:
    """Chance-corrected agreement. Raw agreement inflates badly here — the
    unmarked class dominates on delivery — so the kappa is the number to read."""
    n = len(a)
    if not n:
        return float("nan")
    cats = set(a) | set(b)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def marked_rate(labels, unmarked: str) -> float:
    vals = [x for x in labels if x]
    if not vals:
        return float("nan")
    return sum(1 for x in vals if x != unmarked) / len(vals)


def report(batch: Path = BATCH, show_annotators: bool = False) -> str:
    champ, meta, humans, ids = load(batch)
    raters = list(humans)
    shown = {w: (w if show_annotators else f"rater_{i+1}") for i, w in enumerate(raters)}
    mix = collections.Counter(meta[t]["true_cell"] for t in ids)
    out = [
        "# Champion validation on SIMULATOR text (batch03)",
        "",
        f"{len(ids)} turns with a champion label and all {len(raters)} annotator sheets.",
        f"Cell mix: {dict(sorted(mix.items()))}",
        "",
        "One of the three sheets is a MODEL annotator (Opus 4.8), not a human. Two are "
        "human. Which is which is recorded with the sheets, not here.",
        "" if show_annotators else
        "Annotator columns are anonymised: this repository is public and per-person "
        "agreement scores are not published as a side effect of a validation run. Run "
        "with --show-annotators locally for the named version.",
        "",
    ]

    for axis, (unmarked, ckey, hkey) in AXES.items():
        champ_labels = [_norm(champ[t][ckey]) for t in ids]
        out += [
            f"## {axis}",
            "",
            "### Marked rate (`d`) — the quantity [EBM] treats as scenario-owned",
            "",
            "| cell | n | champion | " + " | ".join(shown[w] for w in raters) + " |",
            "|---|---|---|" + "---|" * len(raters),
        ]
        for cell in CELLS:
            sub = [t for t in ids if meta[t]["true_cell"] == cell]
            if not sub:
                continue
            row = [cell, str(len(sub)),
                   f"{marked_rate([_norm(champ[t][ckey]) for t in sub], unmarked):.1%}"]
            for w in raters:
                row.append(f"{marked_rate([_norm(humans[w][t][hkey]) for t in sub], unmarked):.1%}")
            out.append("| " + " | ".join(row) + " |")

        row = ["**all**", str(len(ids)), f"**{marked_rate(champ_labels, unmarked):.1%}**"]
        for w in raters:
            row.append(f"{marked_rate([_norm(humans[w][t][hkey]) for t in ids], unmarked):.1%}")
        out += ["| " + " | ".join(row) + " |", ""]

        out += ["### Agreement", "",
                "| annotator | 3-way kappa | marked/unmarked kappa |", "|---|---|---|"]
        for w in raters:
            pairs = [(c, _norm(humans[w][t][hkey])) for c, t in zip(champ_labels, ids)
                     if _norm(humans[w][t][hkey])]
            a = [p[0] for p in pairs]
            b = [p[1] for p in pairs]
            bin_a = [x != unmarked for x in a]
            bin_b = [x != unmarked for x in b]
            out.append(f"| {shown[w]} | {cohen_kappa(a, b):.3f} | {cohen_kappa(bin_a, bin_b):.3f} |")
        out.append("")

    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default=str(BATCH))
    ap.add_argument("--out", default=None, help="write the report here as well as stdout")
    ap.add_argument("--show-annotators", action="store_true",
                    help="use real annotator names. LOCAL USE ONLY — this repository "
                         "is public and the sheets are per-person work.")
    args = ap.parse_args()
    text = report(Path(args.batch), show_annotators=args.show_annotators)
    print(text)
    if args.out:
        Path(args.out).write_text(text)
