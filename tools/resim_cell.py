#!/usr/bin/env python3
"""Re-simulate a cell's arcs with a swappable simulator model — to test whether
an RLHF-guardrailed model is what suppresses hot (hostile-toward-listener) delivery.

Reuses the FROZEN character brief (results/build/<cell>_prompt.txt) unchanged and only
overrides roles.simulator.model_path, so the ONLY variable is the patient model's
guardrails. If a less-aligned model produces hostility the guardrailed one wouldn't,
that isolates alignment as the cause.

  python tools/resim_cell.py --cell b6 --sim-model huihui_ai/qwen2.5-abliterate:14b
  python tools/resim_cell.py --cell b4 --sim-model dolphin-mistral --sim-base-url http://localhost:11435/v1

Prints each patient turn and saves transcripts under results/resim/<cell>/<model>/.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="b6", help="hot cells are b2/b4/b6")
    ap.add_argument("--brief-file", default=None,
                    help="use this character brief instead of the frozen results/build/<cell>_prompt.txt")
    ap.add_argument("--sim-model", default=None, help="override the simulator (patient) model")
    ap.add_argument("--sim-base-url", default=None)
    ap.add_argument("--ref-model", default=None,
                    help="override the reference INTERLOCUTOR model (e.g. an abliterated model, to "
                         "test whether the interlocutor's guardrails drive the crisis-referral loop)")
    ap.add_argument("--n-arcs", type=int, default=2)
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--seed-base", type=int, default=1000)
    args = ap.parse_args()

    brief_path = Path(args.brief_file) if args.brief_file else ROOT / "results" / "build" / f"{args.cell}_prompt.txt"
    if not brief_path.exists():
        sys.exit(f"no brief: {brief_path}")
    brief = brief_path.read_text()

    sys.path.insert(0, str(ROOT / "sway_harness"))
    from config import load_config, build_server_config, build_role_config   # noqa: E402
    from build import _run_build_arc                                         # noqa: E402

    config = load_config()
    server = build_server_config(config)
    roles = build_role_config(config)
    if args.sim_model:
        roles.simulator.model_path = args.sim_model
    if args.sim_base_url:
        roles.simulator.base_url = args.sim_base_url
    if args.ref_model:
        roles.reference_interlocutor.model_path = args.ref_model
    print(f"simulator model: {roles.simulator.model_path}   (interlocutor: {roles.reference_interlocutor.model_path})")
    print(f"cell {args.cell}  brief: {brief_path.name}  arcs: {args.n_arcs}  turns: {args.turns}\n")

    safe = re.sub(r"[^\w.-]", "_", roles.simulator.model_path)
    outdir = ROOT / "results" / "resim" / args.cell / safe
    outdir.mkdir(parents=True, exist_ok=True)

    for a in range(args.n_arcs):
        seed = args.seed_base + a
        transcript = _run_build_arc(server, roles, brief, seed=seed, num_turns=args.turns)
        (outdir / f"transcript_seed{seed}.json").write_text(json.dumps(transcript, indent=2, ensure_ascii=False))
        print("=" * 78)
        print(f"ARC seed={seed}")
        print("=" * 78)
        for m in transcript:
            tag = "PATIENT   " if m["role"] == "assistant" else "therapist "
            print(f"[{tag}] {m['content'].strip()}\n")
    print(f"saved transcripts under {outdir}")


if __name__ == "__main__":
    main()
