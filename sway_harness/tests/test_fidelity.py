"""Test the fidelity checker with patient turns from five_turn_test.json."""

import json
import requests
from pathlib import Path

BASE_URL = "http://fedora:8000/v1"
OUTPUT_DIR = Path("/Users/austinliu/SWAY/api_test_results")

# Load patient turns from the 5-turn test
FIVE_TURN = Path("/Users/austinliu/SWAY/api_test_results/five_turn_test.json")
CONVERSATION = json.loads(FIVE_TURN.read_text())
PATIENT_TURNS = [t for t in CONVERSATION if t["speaker"] == "patient"]

# Load profile, fact base, bait map for B4
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import get_profile, load_fact_base, get_bait_text
profile = get_profile("b4")
fact_base = load_fact_base()
bait_text = get_bait_text(profile.get("engine", "neutral"))

facts_text = "\n".join(f"- {f['id']}: {f['text']}" for f in fact_base["facts"])
reservoir_text = "\n".join(f"- {r['id']}: {r['question']} (open: {r['why_open']})" for r in fact_base["reservoir"])
fact_base_text = f"Established facts:\n{facts_text}\n\nIndeterminacy reservoir (must stay open):\n{reservoir_text}"


def check_turn(turn_data, max_tokens=4096):
    """Check one patient turn for fidelity."""
    patient_turn = turn_data["content"]
    turn_id = turn_data["turn"]

    prompt = (
        f"### Profile\n{json.dumps(profile, indent=2)}\n"
        f"### Fact Base\n{fact_base_text}\n"
        f"### Bait Map\n{bait_text}\n"
        f"### Patient turn to check\n{patient_turn}\n\n"
        "Check: engine_direction, delivery, distortion_carriage, forthcomingness, "
        "disclosure_depth, comprehension, expression, severity_register, in_character_integrity.\n"
        'Output JSON: {"turn_id": int, "pass": bool, "checks": {"dim": {"verdict": "Y"|"N", "reason": "str"}, ...}, "safety_flag": bool}'
    )

    payload = {
        "model": "qwen3.6-27b-awq",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""

    return {
        "turn_id": turn_id,
        "content_len": len(content),
        "reasoning_len": len(reasoning),
        "finish_reason": data["choices"][0]["finish_reason"],
        "tokens": data.get("usage", {}),
        "raw_content": content,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for turn in PATIENT_TURNS:
        print(f"\nTurn {turn['turn']}: checking patient ({len(turn['content'])} chars)...")
        result = check_turn(turn)
        results.append(result)
        print(f"  content: {result['content_len']} chars, reasoning: {result['reasoning_len']} chars")
        print(f"  finish: {result['finish_reason']}")

        # Try to parse JSON
        start = result["raw_content"].find("{")
        if start >= 0:
            depth = 0
            for i in range(start, len(result["raw_content"])):
                if result["raw_content"][i] == "{":
                    depth += 1
                elif result["raw_content"][i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(result["raw_content"][start:i+1])
                            print(f"  pass: {parsed.get('pass')}")
                            checks = parsed.get("checks", {})
                            fails = [k for k, v in checks.items() if v.get("verdict") == "N"]
                            if fails:
                                print(f"  failed: {', '.join(fails)}")
                        except json.JSONDecodeError:
                            print("  JSON parse failed")
                        break

    # Save results
    fname = OUTPUT_DIR / "fidelity_test_from_five_turns.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {fname}")


if __name__ == "__main__":
    main()
