#!/usr/bin/env python3
"""Export a blind engine×delivery labeling package + private key for SWAY.

fidelity_checker_2.md Task 1. Reads the backbone arcs and their CURRENT-production
(engine-fixed) Level-0 judge labels from results/build_artifacts/<cell>/iter_*/
fidelity_results.corrected.json + transcript_*.json, samples a stratified blind set
(stratified by CELL, never by judge label; all hot-in-warm turns taken), and writes:

  label_tasks/hand_labels_batch01.csv   — human fills engine/delivery (no cell/label leaked)
  label_tasks/_key_batch01.csv          — private key (git-ignored): true cell + judge labels

Deviation from the spec skeleton: turn_id is a salted SHA1 of (arc_id, turn_index), not
uuid4 — the acceptance criteria require re-runs with the same seed to reproduce identical
ids/ordering, which uuid4 cannot. The hash is opaque and does not reveal cell/arc/position.
"""
import argparse
import csv
import glob
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "results" / "build_artifacts"

WARM_CELLS = {"b1", "b3", "b5"}
DEFAULT_ALLOC = {"b3": 45, "b1": 35, "b4": 30, "b2": 20, "b5": 20, "b6": 20}
ID_SALT = "sway-blind-v1"


def load_cell_items(cell):
    """Yield one dict per patient turn with its current-production judge labels.

    Data mapping (verified against the real repo): transcripts store assistant=patient,
    user=interlocutor(MUT). fidelity_results.corrected.json holds the judge labels per
    sample in patient-turn order; join label[k] to the k-th assistant turn.
    """
    items = []
    for fr in sorted(glob.glob(str(ART / cell / "iter_*" / "fidelity_results.corrected.json"))):
        iterdir = Path(fr).parent
        data = json.load(open(fr))
        for entry in data["transcripts"]:
            sample = entry["sample"]
            labels = entry.get("labels", [])
            tpath = iterdir / f"transcript_{sample}.json"
            if not tpath.exists():
                continue
            turns = json.loads(tpath.read_text())
            k = 0
            for j, m in enumerate(turns):
                if m.get("role") != "assistant":
                    continue
                if k >= len(labels):
                    break
                lab = labels[k]
                k += 1
                items.append({
                    "arc_id": f"{cell}/{iterdir.name}/t{sample}",
                    "turn_index": j,
                    "context_prev_assistant": turns[j - 1]["content"] if j > 0 else "",
                    "patient_turn": m["content"],
                    "judge_engine": lab.get("engine_direction"),
                    "judge_delivery": lab.get("delivery"),
                })
    return items


def sample_cell(cell, items, quota, rng):
    """Warm cells: populate BOTH the judge-hot stratum (the grievance→hot boundary)
    and judge-warm/flat (turns a human might flip), balanced within the quota.

    Deviation from the spec's "take ALL hot": the spec assumed judge-hot is rare in warm
    cells, but the very bug we're auditing makes it abundant (b3 ~950 hot). Taking all
    would blow the quota and leave zero warm turns. So we take ~half the quota as hot
    (all of it if fewer), reserve the rest for non-hot, and top up from whichever stratum
    has more if the other runs short — guaranteeing the boundary is populated AND some
    flippable judge-warm turns remain.
    """
    if cell in WARM_CELLS:
        hot = [x for x in items if x["judge_delivery"] == "hot"]
        rest = [x for x in items if x["judge_delivery"] != "hot"]
        rng.shuffle(hot); rng.shuffle(rest)
        hot_take = min(len(hot), max(quota // 2, quota - len(rest)))
        nonhot_take = min(len(rest), quota - hot_take)
        return hot[:hot_take] + rest[:nonhot_take]
    rng.shuffle(items)
    return items[:quota]


def turn_id(arc_id, turn_index):
    return hashlib.sha1(f"{ID_SALT}:{arc_id}:{turn_index}".encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260711)
    ap.add_argument("--out", default=str(ROOT / "label_tasks"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_items = []
    hot_in_warm = 0
    for cell, quota in DEFAULT_ALLOC.items():
        items = load_cell_items(cell)
        chosen = sample_cell(cell, items, quota, rng)
        mix = {v: sum(x["judge_delivery"] == v for x in chosen) for v in ("hot", "warm", "flat")}
        if cell in WARM_CELLS:
            hot_in_warm += mix["hot"]
        print(f"{cell}: took {len(chosen)}/{len(items)}  delivery={mix}")
        for x in chosen:
            x["true_cell"] = cell
            x["turn_id"] = turn_id(x["arc_id"], x["turn_index"])
        all_items.extend(chosen)

    rng.shuffle(all_items)  # global blind ordering

    print(f"\nhot-in-warm turns (critical stratum): {hot_in_warm} "
          f"({'OK' if hot_in_warm >= 20 else 'WARN: <20 — took all available'})")

    hf = out / "hand_labels_batch01.csv"
    with hf.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["turn_id", "context_prev_assistant", "patient_turn",
                    "engine_label", "delivery_label", "flag_ambiguous", "notes"])
        for x in all_items:
            w.writerow([x["turn_id"], x["context_prev_assistant"], x["patient_turn"], "", "", "", ""])

    kf = out / "_key_batch01.csv"
    with kf.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["turn_id", "true_cell", "arc_id", "turn_index",
                    "judge_engine_label", "judge_delivery_label"])
        for x in all_items:
            w.writerow([x["turn_id"], x["true_cell"], x["arc_id"], x["turn_index"],
                        x["judge_engine"], x["judge_delivery"]])

    print(f"\nWrote {hf} ({len(all_items)} items) and {kf} (key — keep git-ignored).")

    # Round-trip check (pandas if available).
    try:
        import pandas as pd
        assert len(pd.read_csv(hf)) == len(all_items)
        assert len(pd.read_csv(kf)) == len(all_items)
        print("pandas round-trip: OK")
    except ImportError:
        print("pandas not installed — skipped round-trip check")


if __name__ == "__main__":
    main()
