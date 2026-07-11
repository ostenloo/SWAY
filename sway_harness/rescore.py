"""Re-annotate saved transcripts with the (fixed) production annotator and recompute
Level-1/Level-2 fidelity scores. NO re-simulation — replays only the judge over frozen
transcripts, so it corrects the internalizing/externalizing annotator bug without a rebuild.

Idempotent + resumable: writes fidelity_results.corrected.json per iteration and skips any
iteration already done. Originals are never overwritten.

Usage:  python rescore.py                 # backbone b1-b6 (default)
        python rescore.py b3 b4           # specific cells
Probes (p1-p3) are intentionally excluded: their transcripts were SIMULATED under the wrong
profiles (parser bug), so re-annotation can't make them valid — they need a rebuild.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, build_server_config, build_role_config   # noqa: E402
from parser import get_profile, load_fact_base, get_bait_text            # noqa: E402
from build import _annotate_fidelity_turn, _slim_label                    # noqa: E402
from fidelity import classify_transcript, converge                        # noqa: E402

BACKBONE = ["b1", "b2", "b3", "b4", "b5", "b6"]
ART = Path(__file__).resolve().parent.parent / "results" / "build_artifacts"

config = load_config()
server = build_server_config(config)
roles = build_role_config(config)
print("annotator model:", roles.fidelity_checker.model_path, flush=True)

fb = load_fact_base()
facts_text = "\n".join("- {}: {}".format(f["id"], f["text"]) for f in fb["facts"])
reservoir_text = "\n".join(
    "- {}: {} (open: {})".format(r["id"], r["question"], r["why_open"]) for r in fb["reservoir"]
)
fact_base_text = "Established facts:\n{}\n\nIndeterminacy reservoir (must stay open):\n{}".format(
    facts_text, reservoir_text
)


def rescore_iter(cell, profile, bait_text, iter_dir):
    verdicts, fidelity_results = [], []
    tjs = sorted(t for t in iter_dir.glob("transcript_*.json") if "reannotated" not in t.name)
    for tj in tjs:
        transcript = json.loads(tj.read_text())
        patient_turns = [m["content"] for m in transcript if m["role"] == "assistant"]
        turn_labels = []
        for ti, turn_text in enumerate(patient_turns):
            lab = _annotate_fidelity_turn(server, roles, fact_base_text, bait_text, transcript, turn_text)
            lab["turn"] = ti
            lab["text"] = turn_text
            turn_labels.append(lab)
        verdict = classify_transcript(profile, turn_labels, schedule=None)
        verdicts.append(verdict)
        sample = int(tj.stem.split("_")[1])
        fidelity_results.append(
            {"sample": sample, **verdict.to_dict(), "labels": [_slim_label(t) for t in turn_labels]}
        )
    conv = converge(verdicts)
    return conv, fidelity_results


def main():
    cells = [c.lower() for c in sys.argv[1:]] or BACKBONE
    for cell in cells:
        profile = get_profile(cell)
        bait_text = get_bait_text(profile.get("engine", "neutral"))
        for iter_dir in sorted((ART / cell).glob("iter_*"), key=lambda p: int(p.name.split("_")[1])):
            out_path = iter_dir / "fidelity_results.corrected.json"
            if out_path.exists():
                print("skip (done):", cell, iter_dir.name, flush=True)
                continue
            t0 = time.time()
            conv, fidelity_results = rescore_iter(cell, profile, bait_text, iter_dir)
            out_path.write_text(
                json.dumps({"convergence": conv.to_dict(), "transcripts": fidelity_results},
                           indent=2, ensure_ascii=False)
            )
            eng = conv.dim_pass_frac.get("engine_direction")
            print("{} {}: engine={} mean={:.3f} converged={} ({:.0f}s)".format(
                cell, iter_dir.name, eng, conv.adherence, conv.converged, time.time() - t0), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
