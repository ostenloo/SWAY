#!/usr/bin/env python3
"""Chain-of-thought fidelity annotation that SAVES the reasoning per turn.

Re-annotates the blind batch with a REASONING model (default qwen3:14b, thinking
mode) using the IDENTICAL production annotator prompts (build.annotator_system_prompt
/ annotator_user_prompt) — so the labeling task is unchanged; the only additions are
(a) a reasoning model and (b) capturing its thinking trace.

Ollama returns the thinking in a separate message.reasoning field (content stays clean
JSON), so we keep response_format=json_object exactly like production and grab reasoning
from the side channel. Writes a key with the usual columns PLUS a `reasoning` column, so
compute_kappa.py runs on it directly and you can read WHY each label was assigned.

  python tools/cot_annotate_batch.py                       # qwen3:14b, localhost (fedora)
  python tools/cot_annotate_batch.py --base-url http://localhost:11435/v1   # Mac via tunnel

Incremental + resumable (rewrites the output each turn; re-run skips done turn_ids).
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "results" / "build_artifacts"
FIELDS = ["turn_id", "true_cell", "arc_id", "turn_index",
          "judge_engine_label", "judge_delivery_label", "reasoning"]


def parse_arc(arc_id):
    cell, iter_name, tsample = arc_id.split("/")
    return cell, iter_name, tsample[1:]


def chat(base_url, model, messages, temperature, max_tokens, retries=4):
    """POST directly so we can capture message.reasoning (get_completion drops it)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    last = None
    for attempt in range(retries):
        try:
            m = requests.post(url, json=payload, timeout=600).json()["choices"][0]["message"]
            return (m.get("content") or "").strip(), (m.get("reasoning") or "").strip()
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"chat failed after {retries} tries: {last}")


def parse_labels(content):
    try:
        d = json.loads(content)
    except Exception:
        mt = re.search(r"\{.*\}", content, re.DOTALL)
        d = json.loads(mt.group(0)) if mt else {}
    return ((d.get("engine_direction") or "").strip().lower(),
            (d.get("delivery") or "").strip().lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default=str(ROOT / "label_tasks" / "_key_batch01.csv"))
    ap.add_argument("--out", default=str(ROOT / "label_tasks" / "_key_batch01.cot_qwen3.csv"))
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=3000, help="room for thinking + JSON")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key_rows = list(csv.DictReader(Path(args.key).open(newline="", encoding="utf-8")))
    if args.limit:
        key_rows = key_rows[:args.limit]

    sys.path.insert(0, str(ROOT / "sway_harness"))
    from parser import get_profile, load_fact_base, get_bait_text          # noqa: E402
    from build import annotator_system_prompt, annotator_user_prompt       # noqa: E402

    fb = load_fact_base()
    facts_text = "\n".join("- {}: {}".format(f["id"], f["text"]) for f in fb["facts"])
    reservoir_text = "\n".join(
        "- {}: {} (open: {})".format(r["id"], r["question"], r["why_open"]) for r in fb["reservoir"])
    fact_base_text = "Established facts:\n{}\n\nIndeterminacy reservoir (must stay open):\n{}".format(
        facts_text, reservoir_text)
    system = annotator_system_prompt()

    out_path = Path(args.out)
    done = {}
    if out_path.exists():
        for r in csv.DictReader(out_path.open(newline="", encoding="utf-8")):
            done[r["turn_id"]] = r
        print(f"resume: {len(done)} already annotated", flush=True)

    print(f"model={args.model}  endpoint={args.base_url}  temp={args.temp}", flush=True)
    tcache, profiles, baits, results = {}, {}, {}, []
    t0 = time.time()
    for i, row in enumerate(key_rows, 1):
        tid = row["turn_id"]
        if tid in done:
            results.append(done[tid]); continue
        cell, iter_name, sample = parse_arc(row["arc_id"])
        tpath = ART / cell / iter_name / f"transcript_{sample}.json"
        if tpath not in tcache:
            tcache[tpath] = json.loads(tpath.read_text()) if tpath.exists() else None
        turns = tcache[tpath]
        eng = deliv = reasoning = ""
        if turns is None:
            print(f"  WARN {tid}: transcript missing", flush=True)
        else:
            ti = int(row["turn_index"])
            patient_turn = turns[ti]["content"]
            if cell not in profiles:
                profiles[cell] = get_profile(cell)
                baits[cell] = get_bait_text(profiles[cell].get("engine", "neutral"))
            user = annotator_user_prompt(fact_base_text, baits[cell], turns, patient_turn)
            content, reasoning = chat(args.base_url, args.model,
                                      [{"role": "system", "content": system},
                                       {"role": "user", "content": user}],
                                      args.temp, args.max_tokens)
            eng, deliv = parse_labels(content)
        results.append({"turn_id": tid, "true_cell": row["true_cell"], "arc_id": row["arc_id"],
                        "turn_index": row["turn_index"], "judge_engine_label": eng,
                        "judge_delivery_label": deliv, "reasoning": reasoning})
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(results)
        if i % 5 == 0 or i == len(key_rows):
            print(f"  {i}/{len(key_rows)}  ({time.time() - t0:.0f}s)", flush=True)

    print(f"Wrote {out_path} ({len(results)} rows) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
