"""Test the optimizer with the actual optimizer input prompt."""

import json
import requests
from pathlib import Path

BASE_URL = "http://fedora:8000/v1"
OPTIMIZER_INPUT = Path("/Users/austinliu/SWAY/results/build_artifacts/b4/iter_0/optimizer_input.txt")
OUTPUT_DIR = Path("/Users/austinliu/SWAY/api_test_results")

def test(max_tokens=4096):
    optimizer_input = OPTIMIZER_INPUT.read_text()
    print(f"Optimizer input: {len(optimizer_input)} chars")

    payload = {
        "model": "qwen3.6-27b-awq",
        "messages": [
            {"role": "system", "content": "Output only the requested text. Do not include any thinking, reasoning, or explanation."},
            {"role": "user", "content": optimizer_input},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUTPUT_DIR / f"optimizer_test_{max_tokens}.json"
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to {fname}")
    print(f"content len:   {len(content)}")
    print(f"reasoning len: {len(reasoning)}")
    print(f"finish_reason: {data['choices'][0]['finish_reason']}")
    print(f"tokens:        {data.get('usage', {})}")
    if content:
        print(f"\nFirst 300 chars of content:")
        print(content[:300])


if __name__ == "__main__":
    for mt in [2048, 4096, 8192]:
        print(f"\n{'─'*40} max_tokens={mt} {'─'*40}")
        test(mt)
