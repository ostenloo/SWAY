"""Replay the fidelity annotator over one transcript, saving FULL labels + notes."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, build_server_config, build_role_config
from parser import get_profile, load_fact_base, get_bait_text
from build import _annotate_fidelity_turn

CELL = "b3"
TPATH = Path(__file__).resolve().parent.parent / "results/build_artifacts/b3/iter_0/transcript_0.json"
transcript = json.loads(TPATH.read_text())

config = load_config()
server = build_server_config(config)
roles  = build_role_config(config)
print("fidelity model:", roles.fidelity_checker.model_path)

profile = get_profile(CELL)
fb = load_fact_base()
facts_text = "\n".join("- {}: {}".format(f['id'], f['text']) for f in fb["facts"])
reservoir_text = "\n".join("- {}: {} (open: {})".format(r['id'], r['question'], r['why_open']) for r in fb["reservoir"])
fact_base_text = "Established facts:\n{}\n\nIndeterminacy reservoir (must stay open):\n{}".format(facts_text, reservoir_text)
bait_text = get_bait_text(profile.get("engine", "neutral"))

patient_turns = [m["content"] for m in transcript if m["role"] == "assistant"]
results = []
for i, turn_text in enumerate(patient_turns):
    labels = _annotate_fidelity_turn(server, roles, fact_base_text, bait_text, transcript, turn_text)
    labels["turn"] = i
    labels["text"] = turn_text
    results.append(labels)
    notes = labels.get("notes", {})
    en = notes.get("engine_direction") if isinstance(notes, dict) else notes
    print("turn {:2}: engine={:14} delivery={}  | {}".format(i, str(labels.get("engine_direction")), str(labels.get("delivery")), en))

out = TPATH.with_name("transcript_0.reannotated.json")
out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print("\nsaved:", out)
