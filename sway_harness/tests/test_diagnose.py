"""Diagnose the turn-3 degeneration by printing the exact prompt sent to vLLM."""

import json
import requests
from pathlib import Path

BASE_URL = "http://fedora:8000/v1"
SYSTEM_PROMPT = Path("/Users/austinliu/SWAY/results/build_artifacts/b4/iter_0/optimizer_prompt.txt").read_text()

def call(messages, seed=None):
    payload = {
        "model": "qwen3.6-27b-awq",
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.3,
    }
    if seed: payload["seed"] = seed
    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=300)
    data = resp.json()
    return data["choices"][0]["message"].get("content", ""), data

# Turn 1
print("Turn 1...")
t1, _ = call([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Start the conversation as the patient."},
], seed=42)
print(f"  {len(t1)} chars")

# Turn 2
print("Turn 2...")
t2, _ = call([{"role": "user", "content": t1}], seed=42)
print(f"  {len(t2)} chars")

# Turn 3 — print the exact messages sent
print("\nTurn 3 messages (pre-template):")
msgs = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": t1},
    {"role": "assistant", "content": t2},
]
for i, m in enumerate(msgs):
    print(f"  [{i}] role={m['role']}, content_len={len(m['content'])}")

# Get the templated prompt via a dummy request
payload = {
    "model": "qwen3.6-27b-awq",
    "messages": msgs,
    "max_tokens": 1,
    "temperature": 0.0,
}
resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=30)
# Check if the response includes the prompt back
data = resp.json()
print(f"\nResponse keys: {data.get('choices', [{}])[0].keys()}")
print(f"Finish reason: {data.get('choices', [{}])[0].get('finish_reason')}")
print(f"Stop reason: {data.get('choices', [{}])[0].get('stop_reason')}")

# Now get the actual turn 3 output
print("\nTurn 3 output...")
t3, d3 = call(msgs, seed=42)
print(f"  {len(t3)} chars")
print(f"  First 200: {t3[:200]}")
print(f"  Contains 'Role/Identity': {'Role/Identity' in t3 or 'ROLE & IDENTITY' in t3 or '**Role' in t3}")
print(f"  Finish reason: {d3.get('choices', [{}])[0].get('finish_reason')}")
print(f"  Stop reason: {d3.get('choices', [{}])[0].get('stop_reason')}")
