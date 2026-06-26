"""Quick API test — hit the vLLM endpoint and inspect the response."""

import json
import requests
from pathlib import Path

BASE_URL = "http://fedora:8000/v1"
OUTPUT_DIR = Path("/Users/austinliu/SWAY/api_test_results")

def test_completion(max_tokens=200):
    payload = {
        "model": "qwen3.6-27b-awq",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = OUTPUT_DIR / f"api_test_{max_tokens}.json"
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)

    msg = data["choices"][0]["message"]
    print(f"Saved to {fname}")
    print(f"content:     {repr(msg.get('content'))}")
    print(f"reasoning:   {repr(msg.get('reasoning'))[:100]}...")
    print(f"finish_reason: {data['choices'][0]['finish_reason']}")
    print(f"tokens:      {data.get('usage', {})}")


if __name__ == "__main__":
    for mt in [512, 1024, 2048, 4096]:
        print(f"\n{'─'*40} max_tokens={mt} {'─'*40}")
        test_completion(mt)
