"""
Local inference client — OpenAI-compatible endpoint (llama.cpp, Ollama, vLLM).
All roles (Simulator, Fidelity checker, Judge, Optimizer, MUT, reference interlocutor)
run through this. Graders run at temperature 0; Simulator runs at small temperature with logged seeds.
"""

import os
import time
import json
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class LocalError(Exception):
    """Custom exception for local inference errors."""
    pass


def get_completion(
    model_path: str,
    messages: list,
    base_url: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    max_tokens: int = 2048,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
) -> str:
    """
    Get a completion from a local OpenAI-compatible endpoint.
    Returns raw text — thinking preamble is stripped downstream.
    """
    api_url = f"{base_url.rstrip('/')}/completions"
    headers = {"Content-Type": "application/json"}

    # Convert messages to prompt format for /v1/completions endpoint
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    payload = {
        "model": model_path,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed

    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=300)

            if resp.status_code >= 400:
                logger.error(
                    "HTTP %s from %s\nResponse: %s",
                    resp.status_code, api_url, resp.text[:500],
                )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                raise LocalError(f"Server error: {data['error']}")

            choices = data.get("choices", [])
            if not choices:
                raise LocalError("No choices in response")

            msg = choices[0].get("message", {})
            content = msg.get("content") or ""

            if not content:
                raise LocalError("Received empty content")

            return content.strip()

        except requests.exceptions.RequestException as err:
            logger.warning("[attempt %d/%d] Request failed: %s", attempt + 1, max_retries, err)
            if attempt + 1 == max_retries:
                raise LocalError(f"Giving up after {max_retries} tries: {err}") from err
            time.sleep(backoff_factor ** attempt)

        except LocalError:
            raise

        except Exception as err:
            logger.error("Unexpected error: %s", err, exc_info=True)
            if attempt + 1 == max_retries:
                raise LocalError(f"Giving up after {max_retries} tries: {err}") from err
            time.sleep(backoff_factor ** attempt)

    raise LocalError("Unexpected fall-through in retry loop")


def parse_json(text: str) -> Optional[dict]:
    """Parse JSON from model output that may include thinking preamble."""
    # Find the first { and matching }
    start = text.find("{")
    if start == -1:
        return None
    # Find the matching closing brace
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None
