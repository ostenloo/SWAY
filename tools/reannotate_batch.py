#!/usr/bin/env python3
"""Re-annotate the blind batch's turns with the CURRENT annotator prompt and write
a fresh key, so compute_kappa.py can measure the delivery-rubric fix.

Pairs with tools/export_blind_labels.py (which wrote _key_batch01.csv). This reads
the key's (arc_id, turn_index) pointers back to the frozen transcripts under
results/build_artifacts/, re-runs build._annotate_fidelity_turn over each turn with
the SAME setup rescore.py uses (full transcript as context, per-cell bait/fact base)
— so the ONLY thing that differs from the original judge labels is the annotator
prompt — and writes a new key with fresh judge_{engine,delivery}_label columns.

Run on the host serving the annotator model (fedora). Then re-run the audit:

  python tools/reannotate_batch.py            # writes _key_batch01.reannotated.csv
  python tools/compute_kappa.py \
      --labels label_tasks/hand_labels_batch01.csv \
      --key    label_tasks/_key_batch01.reannotated.csv

Verify the turn→transcript mapping first, anywhere (no model needed):

  python tools/reannotate_batch.py --dry-run

Incremental + resumable: the output CSV is rewritten after every turn, and a
re-run skips turn_ids already present in it.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "results" / "build_artifacts"
KEY_FIELDS = ["turn_id", "true_cell", "arc_id", "turn_index",
              "judge_engine_label", "judge_delivery_label"]


def parse_arc(arc_id):
    """'b1/iter_2/t3' -> ('b1', 'iter_2', '3')."""
    cell, iter_name, tsample = arc_id.split("/")
    return cell, iter_name, tsample[1:]  # strip leading 't'


def transcript_path(arc_id):
    cell, iter_name, sample = parse_arc(arc_id)
    return ART / cell / iter_name / f"transcript_{sample}.json"


def resolve_turn(row, cache):
    """Return (turns, patient_turn, err). turns is the full transcript list."""
    path = transcript_path(row["arc_id"])
    if path not in cache:
        cache[path] = json.loads(path.read_text()) if path.exists() else None
    turns = cache[path]
    if turns is None:
        return None, None, f"transcript missing: {path.relative_to(ROOT)}"
    ti = int(row["turn_index"])
    if not (0 <= ti < len(turns)):
        return turns, None, f"turn_index {ti} out of range (len {len(turns)})"
    if turns[ti].get("role") != "assistant":
        return turns, None, f"turn_index {ti} is role={turns[ti].get('role')!r}, not patient"
    return turns, turns[ti]["content"], None


def dry_run(key_rows, labels_path):
    """Verify every key row resolves to a transcript turn whose text matches the
    hand-label CSV. Pure stdlib — runs anywhere, no annotator model needed."""
    truth = {}
    if labels_path.exists():
        for r in csv.DictReader(labels_path.open(newline="", encoding="utf-8")):
            truth[r["turn_id"]] = r["patient_turn"]
    cache = {}
    ok = mismatch = missing = 0
    for row in key_rows:
        turns, patient_turn, err = resolve_turn(row, cache)
        if err:
            missing += 1
            print(f"  MISSING {row['turn_id']} ({row['arc_id']} #{row['turn_index']}): {err}")
            continue
        expected = truth.get(row["turn_id"])
        if expected is not None and expected.strip() != patient_turn.strip():
            mismatch += 1
            print(f"  MISMATCH {row['turn_id']} ({row['arc_id']} #{row['turn_index']}): "
                  f"recovered text != hand_labels text")
        else:
            ok += 1
    print(f"\ndry-run: {ok} ok, {mismatch} text-mismatch, {missing} unresolved "
          f"(of {len(key_rows)})")
    return mismatch == 0 and missing == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=str(ROOT / "label_tasks" / "_key_batch01.csv"))
    ap.add_argument("--labels", default=str(ROOT / "label_tasks" / "hand_labels_batch01.csv"),
                    help="hand-label CSV, used only by --dry-run for the text cross-check")
    ap.add_argument("--out", default=str(ROOT / "label_tasks" / "_key_batch01.reannotated.csv"))
    ap.add_argument("--limit", type=int, default=0, help="process only first N rows (smoke test)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve+verify turn mapping without calling the annotator")
    args = ap.parse_args()

    key_rows = list(csv.DictReader(Path(args.key).open(newline="", encoding="utf-8")))
    if args.limit:
        key_rows = key_rows[:args.limit]

    if args.dry_run:
        ok = dry_run(key_rows, Path(args.labels))
        sys.exit(0 if ok else 1)

    # Heavy imports deferred so --dry-run needs no model deps / working sway_harness env.
    sys.path.insert(0, str(ROOT / "sway_harness"))
    from config import load_config, build_server_config, build_role_config   # noqa: E402
    from parser import get_profile, load_fact_base, get_bait_text            # noqa: E402
    from build import _annotate_fidelity_turn                                # noqa: E402

    config = load_config()
    server = build_server_config(config)
    roles = build_role_config(config)
    print("annotator model:", roles.fidelity_checker.model_path, flush=True)

    fb = load_fact_base()
    facts_text = "\n".join("- {}: {}".format(f["id"], f["text"]) for f in fb["facts"])
    reservoir_text = "\n".join(
        "- {}: {} (open: {})".format(r["id"], r["question"], r["why_open"]) for r in fb["reservoir"])
    fact_base_text = "Established facts:\n{}\n\nIndeterminacy reservoir (must stay open):\n{}".format(
        facts_text, reservoir_text)

    out_path = Path(args.out)
    done = {}
    if out_path.exists():
        for r in csv.DictReader(out_path.open(newline="", encoding="utf-8")):
            done[r["turn_id"]] = r
        print(f"resume: {len(done)} rows already annotated in {out_path.name}", flush=True)

    tcache, profiles, baits, results = {}, {}, {}, []
    t0 = time.time()
    for i, row in enumerate(key_rows, 1):
        tid = row["turn_id"]
        if tid in done:
            results.append(done[tid])
            continue
        cell = parse_arc(row["arc_id"])[0]
        turns, patient_turn, err = resolve_turn(row, tcache)
        eng = deliv = ""
        if err:
            print(f"  WARN {tid}: {err}", flush=True)
        else:
            if cell not in profiles:
                profiles[cell] = get_profile(cell)
                baits[cell] = get_bait_text(profiles[cell].get("engine", "neutral"))
            lab = _annotate_fidelity_turn(server, roles, fact_base_text, baits[cell], turns, patient_turn)
            eng = (lab.get("engine_direction") or "").strip().lower()
            deliv = (lab.get("delivery") or "").strip().lower()
        results.append({"turn_id": tid, "true_cell": row["true_cell"], "arc_id": row["arc_id"],
                        "turn_index": row["turn_index"],
                        "judge_engine_label": eng, "judge_delivery_label": deliv})
        with out_path.open("w", newline="", encoding="utf-8") as f:  # rewrite each turn → resumable
            w = csv.DictWriter(f, fieldnames=KEY_FIELDS)
            w.writeheader()
            w.writerows(results)
        if i % 10 == 0 or i == len(key_rows):
            print(f"  {i}/{len(key_rows)}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"Wrote {out_path} ({len(results)} rows) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
