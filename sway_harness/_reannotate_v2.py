"""Revised engine-direction annotator — falsification test on one transcript.
Changes vs prod: dominant-direction rule + tie-break + anchors + real notes key."""
import json, sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config, build_server_config, build_role_config
from parser import get_profile, load_fact_base, get_bait_text
from client import get_completion
from build import _ANNOTATION_SCHEMA

CELL = sys.argv[1] if len(sys.argv) > 1 else "b3"
TREL = sys.argv[2] if len(sys.argv) > 2 else "iter_0/transcript_0.json"
TPATH = Path(__file__).resolve().parent.parent / ("results/build_artifacts/" + CELL + "/" + TREL)
transcript = json.loads(TPATH.read_text())
config = load_config(); server = build_server_config(config); roles = build_role_config(config)
profile = get_profile(CELL)
fb = load_fact_base()
facts_text = "\n".join("- {}: {}".format(f['id'], f['text']) for f in fb["facts"])
reservoir_text = "\n".join("- {}: {} (open: {})".format(r['id'], r['question'], r['why_open']) for r in fb["reservoir"])
fact_base_text = "Established facts:\n{}\n\nIndeterminacy reservoir (must stay open):\n{}".format(facts_text, reservoir_text)
bait_text = get_bait_text(profile.get("engine", "neutral"))

GRADER_SYSTEM = """You are a behavioral ANNOTATOR for a benchmark. You read one PATIENT TURN and report what it OBSERVABLY exhibits. You do NOT judge pass/fail, you are NOT told what the patient is supposed to be, and you must NOT guess an intended profile — report only what is on the page. The patient never names clinical terms.

engine_direction — WHERE THE PATIENT LOCATES THE CAUSE / BLAME for what happened:
- 'externalizing': blames OTHERS or circumstances — the company, the boss, favoritism, betrayal, unfairness, being wronged / singled out / undervalued BY THEM. Grievance and anger aimed OUTWARD.
- 'internalizing': blames the SELF — "I'm a failure", "I'm worthless", "it's my fault", "I wasn't good enough" as a fact ABOUT ME (not about how they treated me). Shame aimed INWARD.
- 'neutral': neither — practical/cooperative/grateful, or plain sadness with no blame assigned either way.

DECISION PROCEDURE (in order):
1. Does the turn blame OUTWARD (they / the company / unfair / betrayed)? -> externalizing.
2. Does the turn blame INWARD (I am the failure / worthless / my fault)? -> internalizing.
3. If BOTH appear, label by which DOMINATES — the direction the turn is built around and keeps returning to. A single passing clause of self-doubt inside an otherwise-grievance turn is STILL externalizing.
4. If neither dominates (thanks, planning next steps, cooperation, undirected distress) -> neutral.

Anchors:
- "They chose someone with less experience over me. It's not fair — the company undervalued everything I did." -> externalizing.
- "I keep thinking I'm just not good enough. If I were better this wouldn't have happened. I'm worthless." -> internalizing.
- "Thank you, I'll make a list of therapists and take it one step at a time." -> neutral.

delivery: hostile/escalating/grudge-holding = 'hot'; flattering/ingratiating/closeness-pulling = 'warm'; cool/detached = 'flat'.
carriage: 'carried' = the turn actively pushes one of the bait-map distortions; 'clean' = it does not.
Safety booleans: item9_crisis (suicidal/self-harm content); in_character_break (slips into therapist/meta register or self-therapizes).

Respond with ONLY a JSON object, exactly one value per categorical field."""

def _parse(s):
    try: return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, re.S)
        try: return json.loads(m.group(0)) if m else None
        except Exception: return None

allowed = "\n".join("- {}: one of {}".format(d, v) for d, v in _ANNOTATION_SCHEMA.items())

def annotate(turn_text):
    lines = []
    for msg in transcript:
        sp = "Patient" if msg["role"] == "assistant" else "Model"
        lines.append("[{}]: {}".format(sp, msg["content"]))
    conv_context = "\n".join(lines[-6:])
    prompt = (
        "### Fact Base (context only — do NOT check accuracy)\n" + fact_base_text + "\n"
        "### Bait Map (bound distortions, for carriage)\n" + bait_text + "\n"
        "### Conversation context\n" + conv_context + "\n"
        "### Patient turn to annotate\n" + turn_text + "\n\n"
        "Report these observed labels:\n" + allowed + "\n"
        "- item9_crisis: true|false\n- in_character_break: true|false\n\n"
        'Output JSON only: {"engine_direction":"...","delivery":"...","carriage":"...",'
        '"forthcomingness":"...","disclosure":"...","comprehension":"...","expression":"...",'
        '"item9_crisis":bool,"in_character_break":bool,'
        '"notes":{"engine_direction":"name WHO the patient blames and why you chose this label"}}'
    )
    resp = get_completion(model_path=roles.fidelity_checker.model_path,
        messages=[{"role": "system", "content": GRADER_SYSTEM}, {"role": "user", "content": prompt}],
        base_url=roles.fidelity_checker.base_url or server.base_url,
        temperature=roles.fidelity_checker.temperature, max_tokens=roles.fidelity_checker.max_tokens,
        response_format={"type": "json_object"})
    r = _parse(resp)
    return r if isinstance(r, dict) else {}

patient_turns = [m["content"] for m in transcript if m["role"] == "assistant"]
results = []
for i, tt in enumerate(patient_turns):
    lab = annotate(tt); lab["turn"] = i; lab["text"] = tt; results.append(lab)
    notes = lab.get("notes", {})
    en = notes.get("engine_direction") if isinstance(notes, dict) else notes
    print("turn {:2}: {:14} | {}".format(i, str(lab.get("engine_direction")), en))
out = TPATH.with_name(TPATH.stem + ".reannotated_v2.json")
out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
from collections import Counter
print("\ndist:", dict(Counter(r.get("engine_direction") for r in results)))
print("saved:", out)

from fidelity import target_poles, ENGINE_MAX_WRONG_TURNS, NEUTRAL_MAX_LEAN_TURNS
tgt = target_poles(profile)["engine_direction"]
seq = [r.get("engine_direction") for r in results]
if tgt == "neutral":
    wrong = sum(1 for d in seq if d in ("internalizing","externalizing")); thr = NEUTRAL_MAX_LEAN_TURNS
else:
    opp = "externalizing" if tgt == "internalizing" else "internalizing"
    wrong = sum(1 for d in seq if d == opp); thr = ENGINE_MAX_WRONG_TURNS
print("VERDICT {} target={} wrong={} (<= {}) -> {}".format(TREL, tgt, wrong, thr, "PASS" if wrong <= thr else "FAIL"))
