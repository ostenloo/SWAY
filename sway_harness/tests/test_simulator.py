"""Test the simulator with the actual frozen patient prompt as system prompt."""

import json
import requests
from pathlib import Path

BASE_URL = "http://fedora:8000/v1"
OPTIMIZER_PROMPT = Path("/Users/austinliu/SWAY/results/build_artifacts/b4/iter_0/optimizer_prompt.txt")
OUTPUT_DIR = Path("/Users/austinliu/SWAY/api_test_results")

def test(max_tokens=2048):
    system_prompt = OPTIMIZER_PROMPT.read_text()
    print(f"System prompt: {len(system_prompt)} chars")

    payload = {
        "model": "qwen3.6-27b-awq",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Start the conversation as the patient."},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    msg = data["choices"][0]["message"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUTPUT_DIR / f"simulator_test_{max_tokens}.json"
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {fname}")
    print(f"content:     {repr(msg.get('content'))}")
    print(f"reasoning len: {len(msg.get('reasoning') or '')}")
    print(f"finish_reason: {data['choices'][0]['finish_reason']}")
    print(f"tokens:      {data.get('usage', {})}")


if __name__ == "__main__":
    for mt in [1024, 2048, 4096]:
        print(f"\n{'─'*40} max_tokens={mt} {'─'*40}")
        test(mt)
