#!/usr/bin/env python3
"""Held-out convergence-derivation pipeline (anti-circularity-clean).

For each backbone cell b1-b6:
  Phase A  generate N fresh arcs from the FROZEN prompt (held-out seeds, NOT the
           build seeds), simulator vs the (abliterate) reference interlocutor.
           Re-roll guardrail-break / degenerate arcs; veto any that survive.
  Phase B  annotate every patient turn: engine <- command-r7b, delivery <- glm4:9b
           (the champions), using the same 2-msg context the sweep validated.
  Phase C  emit results/heldout_artifacts/<cell>/iter_0/fidelity_results.json in the
           exact shape derive_convergence.py reads, then run the derivation.

Two phases so the GPU holds only {qwen, abliterate} during A and {command-r7b, glm4}
during B — never all four. Runs entirely on the fedora host (all models local).

  python tools/heldout_derive.py --n 30 --engine-model command-r7b --delivery-model glm4:9b
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CELLS = ["b1", "b2", "b3", "b4", "b5", "b6"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="held-out arcs per cell")
    ap.add_argument("--seed-base", type=int, default=1000, help="held-out seed base (distinct from build's 42)")
    ap.add_argument("--sim-model", default=None, help="override the simulator model (patient); default = config")
    ap.add_argument("--engine-model", default="command-r7b")
    ap.add_argument("--delivery-model", default="glm4:9b")
    ap.add_argument("--dst", default="results/heldout_artifacts")
    ap.add_argument("--num-turns", type=int, default=20)
    ap.add_argument("--cells", default=None, help="comma list, e.g. b5,b6 (default: all b1-b6)")
    ap.add_argument("--prompt-file", default=None, help="override the frozen prompt with this file (applies to all --cells)")
    ap.add_argument("--skip-delivery", action="store_true", help="engine annotation only (skip delivery model)")
    args = ap.parse_args()
    cells = args.cells.split(",") if args.cells else CELLS

    sys.path.insert(0, str(ROOT / "sway_harness"))
    sys.path.insert(0, str(ROOT / "tools"))
    from config import load_config, build_server_config, build_role_config      # noqa
    from build import (_run_build_arc, arc_has_guardrail_break, _arc_is_degenerate,  # noqa
                       has_guardrail_break, ARC_MAX_ATTEMPTS, ARC_RESEED_STRIDE,
                       annotator_system_prompt, annotator_user_prompt)
    from parser import get_profile, load_fact_base, get_bait_text               # noqa
    from cot_annotate_batch import chat                                          # noqa
    from cot_annotate_from_csv import extract_labels                            # noqa

    cfg = load_config()
    server = build_server_config(cfg)
    roles = build_role_config(cfg)
    if args.sim_model:
        roles.simulator.model_path = args.sim_model  # swap the patient simulator
    base = server.base_url
    print(f"[cfg] simulator={roles.simulator.model_path} interlocutor={roles.reference_interlocutor.model_path} "
          f"engine-annot={args.engine_model} delivery-annot={args.delivery_model}", flush=True)

    fb = load_fact_base()
    facts = "\n".join("- {}: {}".format(f["id"], f["text"]) for f in fb["facts"])
    reservoir = "\n".join("- {}: {} (open: {})".format(r["id"], r["question"], r["why_open"]) for r in fb["reservoir"])
    fact_base_text = "Established facts:\n{}\n\nIndeterminacy reservoir (must stay open):\n{}".format(facts, reservoir)
    system_annot = annotator_system_prompt()

    t0 = time.time()

    # ---- Phase A: generate held-out arcs (simulator vs abliterate interlocutor) ----
    arcs = {c: [] for c in cells}          # cell -> list of {seed, transcript, discarded}
    for cell in cells:
        prompt = (Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file
                  else (ROOT / "results" / "build" / f"{cell}_prompt.txt").read_text(encoding="utf-8"))
        for i in range(args.n):
            seed = args.seed_base + i
            transcript, discarded = None, False
            for attempt in range(ARC_MAX_ATTEMPTS):
                s = seed + attempt * ARC_RESEED_STRIDE
                transcript = _run_build_arc(server, roles, prompt, s, num_turns=args.num_turns)
                if not arc_has_guardrail_break(transcript) and not _arc_is_degenerate(transcript):
                    break
            else:
                discarded = True  # survived all re-rolls broken/degenerate
            arcs[cell].append({"seed": seed, "transcript": transcript, "discarded": discarded})
            if (i + 1) % 5 == 0:
                print(f"[gen] {cell} {i+1}/{args.n} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[gen] DONE all cells ({time.time()-t0:.0f}s)", flush=True)

    # ---- Phase B+C: annotate with champions, write derive-ready tree ----
    for cell in cells:
        bait = get_bait_text(get_profile(cell).get("engine", "neutral"))
        transcripts_out = []
        for sample_idx, rec in enumerate(arcs[cell]):
            t = rec["transcript"]
            labels = []
            turn_idx = 0
            for i, m in enumerate(t):
                if m.get("role") != "assistant":
                    continue
                patient_turn = m.get("content", "") or ""
                prev = t[i - 1].get("content", "") if i > 0 else ""
                ctx = [{"role": "user", "content": prev}, {"role": "assistant", "content": patient_turn}]
                user = annotator_user_prompt(fact_base_text, bait, ctx, patient_turn)
                # engine <- command-r7b
                ce, _ = chat(base, args.engine_model,
                             [{"role": "system", "content": system_annot}, {"role": "user", "content": user}], 0.0, 3000)
                eng, _d = extract_labels(ce)
                # delivery <- glm4 (skipped in engine-only fast mode)
                if args.skip_delivery:
                    deliv = ""
                else:
                    cd, _ = chat(base, args.delivery_model,
                                 [{"role": "system", "content": system_annot}, {"role": "user", "content": user}], 0.0, 3000)
                    _e, deliv = extract_labels(cd)
                brk = bool(has_guardrail_break(patient_turn))
                labels.append({"engine_direction": eng, "delivery": deliv,
                               "in_character_break": brk, "turn": turn_idx})
                turn_idx += 1
            transcripts_out.append({"sample": sample_idx, "discarded": rec["discarded"], "labels": labels})
        outdir = ROOT / args.dst / cell / "iter_0"
        outdir.mkdir(parents=True, exist_ok=True)
        json.dump({"transcripts": transcripts_out}, open(outdir / "fidelity_results.json", "w"), indent=2)
        n_ok = sum(1 for x in transcripts_out if not x["discarded"])
        print(f"[annot] {cell}: {len(transcripts_out)} arcs ({n_ok} scorable) -> {outdir} ({time.time()-t0:.0f}s)", flush=True)

    print(f"ALL DONE ({time.time()-t0:.0f}s). Now: python tools/derive_convergence.py --root {args.dst} --iters all", flush=True)


if __name__ == "__main__":
    main()
