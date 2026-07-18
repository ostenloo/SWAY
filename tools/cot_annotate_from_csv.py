#!/usr/bin/env python3
"""Annotate batch turns DIRECTLY from the hand_labels CSV (context_prev_assistant +
patient_turn) instead of reading full transcripts from build_artifacts. This is the
exact context the human labeler saw, covers all turns regardless of whether the source
arc still exists, and captures the reasoning side-channel for reasoning models. Output
is a compute_kappa-ready key.

  python tools/cot_annotate_from_csv.py \
      --labels label_tasks/batch03/hand_labels_batch03.csv \
      --key    label_tasks/batch03/_key_batch03.csv \
      --model  deepseek-r1:8b \
      --out    label_tasks/batch03/_key_batch03.deepseek-r1-8b.csv
"""
import argparse, csv, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ["turn_id", "true_cell", "judge_engine_label", "judge_delivery_label", "reasoning"]


def extract_labels(content):
    """Robustly pull engine/delivery from the LAST balanced JSON object in the
    content. Reasoning models (phi4-reasoning, deepseek-r1) precede the final
    answer with prose that embeds schema fragments, so a greedy first-to-last
    brace match breaks — depth-count instead and take the last valid object."""
    objs, depth, start = [], 0, None
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(content[start:i + 1]); start = None
    for s in reversed(objs):
        try:
            d = json.loads(s)
        except Exception:
            continue
        if "engine_direction" in d or "delivery" in d:
            return (str(d.get("engine_direction") or "").strip().lower(),
                    str(d.get("delivery") or "").strip().lower())
    return "", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="hand_labels CSV (context_prev_assistant, patient_turn)")
    ap.add_argument("--key", required=True, help="key CSV mapping turn_id -> true_cell (for the bait map)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=3000, help="room for thinking + JSON")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "tools"))
    sys.path.insert(0, str(ROOT / "sway_harness"))
    from cot_annotate_batch import chat                                    # noqa: E402
    from parser import get_profile, load_fact_base, get_bait_text          # noqa: E402
    from build import annotator_system_prompt, annotator_user_prompt       # noqa: E402

    cell_of = {r["turn_id"]: r["true_cell"]
               for r in csv.DictReader(Path(args.key).open(newline="", encoding="utf-8"))}
    rows = list(csv.DictReader(Path(args.labels).open(newline="", encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]

    fb = load_fact_base()
    facts_text = "\n".join("- {}: {}".format(f["id"], f["text"]) for f in fb["facts"])
    reservoir_text = "\n".join(
        "- {}: {} (open: {})".format(r["id"], r["question"], r["why_open"]) for r in fb["reservoir"])
    fact_base_text = "Established facts:\n{}\n\nIndeterminacy reservoir (must stay open):\n{}".format(
        facts_text, reservoir_text)
    system = annotator_system_prompt()

    baits, results, blanks = {}, [], 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        tid = r["turn_id"]
        cell = cell_of.get(tid, "")
        patient_turn = r.get("patient_turn", "") or ""
        prev = r.get("context_prev_assistant", "") or ""
        # The exact context the human saw: prior interlocutor turn + the patient turn.
        transcript = [{"role": "user", "content": prev},
                      {"role": "assistant", "content": patient_turn}]
        if cell and cell not in baits:
            baits[cell] = get_bait_text(get_profile(cell).get("engine", "neutral"))
        user = annotator_user_prompt(fact_base_text, baits.get(cell, ""), transcript, patient_turn)
        content, reasoning = chat(args.base_url, args.model,
                                  [{"role": "system", "content": system},
                                   {"role": "user", "content": user}],
                                  args.temp, args.max_tokens)
        eng, deliv = extract_labels(content)
        if not eng and not deliv:
            blanks += 1
        results.append({"turn_id": tid, "true_cell": cell, "judge_engine_label": eng,
                        "judge_delivery_label": deliv, "reasoning": reasoning})
        with Path(args.out).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(results)
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} ({time.time()-t0:.0f}s, blanks={blanks})", flush=True)

    print(f"Wrote {args.out} ({len(results)} rows, {blanks} unparsed) in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
