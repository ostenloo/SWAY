#!/usr/bin/env python3
"""Chat with the fidelity annotator about a single turn's labels.

Reconstructs the EXACT context the annotator saw for one turn (system prompt +
fact base + bait map + conversation + the patient turn), reproduces its labels,
then drops into a conversation where you can ask it to justify or reconsider.

Anchored to the real prompts via build.annotator_system_prompt / annotator_user_prompt,
so you're interrogating the actual instrument, not a paraphrase.

  python tools/interrogate_turn.py --turn-id <id>      # id from disagreements_batch01.csv

CAVEAT: an LLM's explanation of its own label is POST-HOC rationalization, not a
faithful trace of why it decided. Use this to probe how the rubric gets applied and
surface ambiguity for the CODING_GUIDE — not as ground truth about the model's reasoning.
Runs on the host serving the annotator model (fedora).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "results" / "build_artifacts"


def parse_arc(arc_id):
    cell, iter_name, tsample = arc_id.split("/")
    return cell, iter_name, tsample[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-id", required=True)
    ap.add_argument("--key", default=str(ROOT / "label_tasks" / "_key_batch01.reannotated.csv"))
    ap.add_argument("--labels", default=str(ROOT / "label_tasks" / "hand_labels_batch01.csv"))
    ap.add_argument("--temp", type=float, default=0.3, help="temperature for the DISCUSSION turns")
    ap.add_argument("--base-url", default=None,
                    help="override the model endpoint (e.g. http://localhost:11435/v1 when "
                         "SSH-tunnelling fedora's Ollama from the Mac)")
    ap.add_argument("--model", default=None,
                    help="override the model tag (e.g. qwen2.5:14b-instruct-q4_K_M) — use with "
                         "--base-url so you interrogate the REAL annotator, not a local model")
    args = ap.parse_args()

    key = {r["turn_id"]: r for r in csv.DictReader(Path(args.key).open(newline="", encoding="utf-8"))}
    hand = {r["turn_id"]: r for r in csv.DictReader(Path(args.labels).open(newline="", encoding="utf-8"))}
    if args.turn_id not in key:
        sys.exit(f"turn_id {args.turn_id} not in {args.key}")
    krow = key[args.turn_id]
    hrow = hand.get(args.turn_id, {})

    sys.path.insert(0, str(ROOT / "sway_harness"))
    from config import load_config, build_server_config, build_role_config   # noqa: E402
    from parser import get_profile, load_fact_base, get_bait_text            # noqa: E402
    from build import (annotator_system_prompt, annotator_user_prompt,       # noqa: E402
                       _annotate_fidelity_turn)
    from client import get_completion                                        # noqa: E402

    config = load_config()
    server = build_server_config(config)
    roles = build_role_config(config)
    fc = roles.fidelity_checker
    # Overrides let you run on the Mac against fedora's model via an SSH tunnel.
    # Mutating fc also steers _annotate_fidelity_turn, which reads roles.fidelity_checker.
    if args.base_url:
        fc.base_url = args.base_url
    if args.model:
        fc.model_path = args.model
    base_url = fc.base_url or server.base_url
    print(f"[endpoint {base_url}  model {fc.model_path}]")

    fb = load_fact_base()
    facts_text = "\n".join("- {}: {}".format(f["id"], f["text"]) for f in fb["facts"])
    reservoir_text = "\n".join(
        "- {}: {} (open: {})".format(r["id"], r["question"], r["why_open"]) for r in fb["reservoir"])
    fact_base_text = "Established facts:\n{}\n\nIndeterminacy reservoir (must stay open):\n{}".format(
        facts_text, reservoir_text)

    cell, iter_name, sample = parse_arc(krow["arc_id"])
    tpath = ART / cell / iter_name / f"transcript_{sample}.json"
    if not tpath.exists():
        sys.exit(f"transcript missing: {tpath}")
    turns = json.loads(tpath.read_text())
    ti = int(krow["turn_index"])
    patient_turn = turns[ti]["content"]
    profile = get_profile(cell)
    bait_text = get_bait_text(profile.get("engine", "neutral"))

    prev = turns[ti - 1]["content"] if ti > 0 else "(none)"
    print("=" * 78)
    print(f"turn_id {args.turn_id}   cell {krow['true_cell']}   arc {krow['arc_id']} #{ti}")
    print("-" * 78)
    print(f"PREVIOUS (interlocutor): {prev[:400]}")
    print(f"\nPATIENT TURN:\n{patient_turn}")
    print("-" * 78)
    print(f"YOUR label        — engine={hrow.get('engine_label','?')!r:16} delivery={hrow.get('delivery_label','?')!r}")
    print(f"RECORDED judge     — engine={krow.get('judge_engine_label','?')!r:16} delivery={krow.get('judge_delivery_label','?')!r}")

    print("\n… re-annotating live (may differ from recorded; temp>0) …")
    live = _annotate_fidelity_turn(server, roles, fact_base_text, bait_text, turns, patient_turn)
    le, ld = live.get("engine_direction", "?"), live.get("delivery", "?")
    print(f"LIVE judge         — engine={le!r:16} delivery={ld!r}")
    note = (live.get("notes") or {}).get("engine_direction")
    if note:
        print(f"  live engine note: {note}")
    print("=" * 78)

    # ---- discussion mode ----
    discuss_system = (
        "You are the behavioral annotator that just labeled a patient turn for a benchmark. "
        "A human reviewer wants to discuss your labels. This is the rubric you applied:\n\n"
        f"{annotator_system_prompt()}\n\n"
        "This is exactly what you were given to annotate:\n\n"
        f"{annotator_user_prompt(fact_base_text, bait_text, turns, patient_turn)}\n\n"
        f"You labeled it: engine_direction={le}, delivery={ld}. "
        f"A human reviewer labeled it: engine={hrow.get('engine_label','?')}, "
        f"delivery={hrow.get('delivery_label','?')}. "
        "Discuss conversationally: explain WHY you assigned your labels under the rubric, engage "
        "honestly with the human's alternative, and say whether you would revise and why. Quote the "
        "specific rubric clause you are relying on. Do NOT output JSON now — just talk, briefly."
    )
    messages = [{"role": "system", "content": discuss_system}]
    print("Chat with the annotator about this turn. Blank line or 'quit' to exit.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user or user.lower() in ("quit", "exit"):
            break
        messages.append({"role": "user", "content": user})
        try:
            reply = get_completion(model_path=fc.model_path, messages=messages, base_url=base_url,
                                   temperature=args.temp, max_tokens=800)
        except Exception as e:
            print(f"[error: {e}]")
            messages.pop()
            continue
        print(f"\nannotator> {reply}\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
