#!/usr/bin/env python3
"""Annotate the patient turns in raw transcript files with the qwen3-CoT annotator.

For validating resimulations (tools/resim_cell.py): given a glob of transcript JSONs and
the cell (for the bait map), reports the delivery-label distribution + reasoning on hot
turns, so you can confirm a brief fix actually produces hot delivery.

  python tools/annotate_transcripts.py --glob 'results/resim/b6/*/transcript_*.json' --cell b6
"""
import argparse, glob, json, sys, re
from collections import Counter
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent


def is_cjk(s):
    return bool(re.search(r"[一-鿿]", s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--cell", default="b6")
    ap.add_argument("--model", default="qwen3:14b")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--show-hot-reasoning", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "sway_harness"))
    from parser import get_profile, load_fact_base, get_bait_text          # noqa: E402
    from build import annotator_system_prompt, annotator_user_prompt       # noqa: E402

    fb = load_fact_base()
    facts = "\n".join("- {}: {}".format(f["id"], f["text"]) for f in fb["facts"])
    res = "\n".join("- {}: {} (open: {})".format(r["id"], r["question"], r["why_open"]) for r in fb["reservoir"])
    fact_base_text = "Established facts:\n{}\n\nIndeterminacy reservoir (must stay open):\n{}".format(facts, res)
    bait = get_bait_text(get_profile(args.cell).get("engine", "neutral"))
    system = annotator_system_prompt()
    url = f"{args.base_url.rstrip('/')}/chat/completions"

    def annotate(transcript, patient_turn):
        user = annotator_user_prompt(fact_base_text, bait, transcript, patient_turn)
        m = requests.post(url, json={"model": args.model,
                                     "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": user}],
                                     "temperature": 0, "max_tokens": 2500,
                                     "response_format": {"type": "json_object"}}, timeout=300
                          ).json()["choices"][0]["message"]
        try:
            d = json.loads(m.get("content") or "{}")
        except Exception:
            d = {}
        return (d.get("delivery") or "?").strip().lower(), (m.get("reasoning") or "")

    overall = Counter()
    for f in sorted(glob.glob(args.glob)):
        turns = json.loads(Path(f).read_text())
        labs = []
        for i, msg in enumerate(turns):
            if msg["role"] != "assistant":
                continue
            if is_cjk(msg["content"]):
                labs.append("cjk-skip"); continue
            d, reasoning = annotate(turns, msg["content"])
            labs.append(d); overall[d] += 1
            if args.show_hot_reasoning and d == "hot":
                idx = reasoning.lower().find("deliver")
                print(f"    [hot @turn{i}] {reasoning[idx:idx+300].strip() if idx>=0 else reasoning[:300]}")
        print(f"{Path(f).name}: {labs}")
    print(f"\nOVERALL delivery: {dict(overall)}")


if __name__ == "__main__":
    main()
