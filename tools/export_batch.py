#!/usr/bin/env python3
"""Export a fresh blind labeling batch (hand_labels + key) directly from transcripts.

Unlike tools/export_blind_labels.py, this does NOT read fidelity_results.corrected.json,
so it works on freshly-built cells that only have fidelity_results.json (or none). Judge
labels in the key are left BLANK — fill them afterward with reannotate_batch.py or
cot_annotate_batch.py (pointed at this key).

Stratified by cell, uniform within cell (no hot-enrichment — representative). Skips turns
already in a prior batch's key (by arc_id+turn_index) and CJK-flipped (guardrail-broken)
turns. turn_id salted so it never collides with another batch.

  python tools/export_batch.py --out label_tasks/batch02 \
      --exclude-key label_tasks/batch01/_key_batch01.csv --per-cell 25
"""
import argparse, csv, glob, hashlib, json, random, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "results" / "build_artifacts"


def turn_id(salt, arc_id, ti):
    return hashlib.sha1(f"{salt}:{arc_id}:{ti}".encode()).hexdigest()[:12]


def cjk(s):
    return bool(re.search(r"[一-鿿]", s))


def load_cell_turns(cell):
    items = []
    for tp in sorted(glob.glob(str(ART / cell / "iter_*" / "transcript_*.json"))):
        if "reannotated" in tp:
            continue
        p = Path(tp)
        arc_id = f"{cell}/{p.parent.name}/t{p.stem.split('_')[1]}"
        turns = json.loads(p.read_text())
        for j, m in enumerate(turns):
            if m.get("role") != "assistant":
                continue
            items.append({"arc_id": arc_id, "turn_index": j,
                          "context_prev_assistant": turns[j - 1]["content"] if j > 0 else "",
                          "patient_turn": m["content"]})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir, e.g. label_tasks/batch02")
    ap.add_argument("--cells", nargs="+", default=["b1", "b2", "b3", "b4", "b5", "b6"])
    ap.add_argument("--per-cell", type=int, default=25)
    ap.add_argument("--exclude-key", default=None, help="prior batch _key.csv; skip its arc_id+turn_index")
    ap.add_argument("--salt", default="sway-blind-v2")
    ap.add_argument("--seed", type=int, default=20260715)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    exclude = set()
    if args.exclude_key and Path(args.exclude_key).exists():
        for r in csv.DictReader(Path(args.exclude_key).open()):
            exclude.add((r["arc_id"], str(r["turn_index"])))

    out = Path(args.out)
    name = out.name  # e.g. "batch02"
    all_rows = []
    for cell in args.cells:
        items = [x for x in load_cell_turns(cell)
                 if (x["arc_id"], str(x["turn_index"])) not in exclude
                 and not cjk(x["patient_turn"]) and len(x["patient_turn"].split()) >= 5]
        rng.shuffle(items)
        chosen = items[:args.per_cell]
        for x in chosen:
            x["true_cell"] = cell
            x["turn_id"] = turn_id(args.salt, x["arc_id"], x["turn_index"])
        print(f"{cell}: took {len(chosen)}/{len(items)} eligible")
        all_rows.extend(chosen)
    rng.shuffle(all_rows)  # global blind order

    out.mkdir(parents=True, exist_ok=True)
    hf = out / f"hand_labels_{name}.csv"
    with hf.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["turn_id", "context_prev_assistant", "patient_turn",
                    "engine_label", "delivery_label", "flag_ambiguous", "notes"])
        for x in all_rows:
            w.writerow([x["turn_id"], x["context_prev_assistant"], x["patient_turn"], "", "", "", ""])
    kf = out / f"_key_{name}.csv"
    with kf.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["turn_id", "true_cell", "arc_id", "turn_index",
                    "judge_engine_label", "judge_delivery_label"])
        for x in all_rows:
            w.writerow([x["turn_id"], x["true_cell"], x["arc_id"], x["turn_index"], "", ""])
    print(f"\nWrote {len(all_rows)} items:\n  {hf}\n  {kf} (judge labels BLANK — fill with "
          f"reannotate_batch.py/cot_annotate_batch.py)")


if __name__ == "__main__":
    main()
