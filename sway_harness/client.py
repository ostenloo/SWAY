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
    reasoning_effort: Optional[str] = None,
    response_format: Optional[dict] = None,
) -> str:
    """
    Get a completion from a local OpenAI-compatible endpoint.

    Uses /v1/chat/completions so the server applies each model's chat template
    (correct stop tokens / formatting) and so reasoning can be controlled.
    Set reasoning_effort="none" to disable a thinking model's hidden reasoning
    (e.g. Gemma) — without it, /v1/completions returns empty at small budgets
    because the model is still "thinking" when it hits max_tokens.
    Returns raw text — any remaining thinking preamble is stripped downstream.
    """
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}

    payload = {
        "model": model_path,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if response_format is not None:
        # e.g. {"type": "json_object"} — forces the server to emit valid JSON.
        payload["response_format"] = response_format

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

            logger.debug(f"Server response: {data}")

            if "error" in data:
                raise LocalError(f"Server error: {data['error']}")

            choices = data.get("choices", [])
            if not choices:
                logger.error(f"No choices in response. Full response: {data}")
                raise LocalError("No choices in response")

            # Handle both /v1/chat/completions (message.content) and /v1/completions (text) formats
            choice = choices[0]
            if "message" in choice:
                content = choice["message"].get("content") or ""
            else:
                content = choice.get("text") or ""

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


def swap_roles(messages: list) -> list:
    """Flip user<->assistant so the dialogue is seen from the other speaker.

    The transcript is stored canonically (patient=assistant, provider/MUT=user).
    A chat-completions request always generates the *assistant* turn and, if the
    array ends on an assistant turn, *continues* it — so generating the provider
    or MUT from the canonical transcript makes it echo the patient. Presenting
    that speaker's own turns as 'assistant' and the patient's as 'user' makes the
    request end on a 'user' turn and produce a fresh reply instead.
    """
    flip = {"user": "assistant", "assistant": "user"}
    return [{**m, "role": flip.get(m["role"], m["role"])} for m in messages]


# Simulator persona-framing modes (config: roles.simulator.framing).
#   "roleplay"   — the current explicit assistant-role instruction.
#   "completion" — port of Roleplay-doh's finding that eliciting the patient as a
#                  *continuation of that person's own speech*, rather than
#                  instructing an assistant to roleplay a patient, suppresses
#                  assistant-register leak (therapist-speak, gratitude, premature
#                  self-resolution). Ollama exposes no raw-completion endpoint, so
#                  over /v1/chat/completions it approximates the effect by dropping
#                  the assistant / helper / advice / therapist framing and
#                  presenting the profile as an established first-person self to be
#                  continued verbatim.
SIMULATOR_FRAMINGS = ("roleplay", "completion")


def patient_system_prompt(character_prompt: str) -> str:
    """Frame the optimizer-authored patient prompt with an explicit role.

    The optimizer output *describes* the patient but doesn't reliably *lock* the
    role, so the model's default "assistant = helper" instinct can make the
    simulated patient drift into therapist-speak. Stating the role plainly, and
    that the model is the patient talking to a therapist, holds it in character.
    """
    return (
        "You are roleplaying as a patient, with the profile below, talking to a therapist. "
        "Speak ONLY in the first person as this patient — share your own experience and "
        "feelings, stay fully in character, and never act as the therapist or give advice.\n\n"
        + character_prompt
    )


def patient_completion_prompt(character_prompt: str) -> str:
    """Completion-framing: elicit the patient as a continuation of their own speech.

    Deliberately omits the "you are an assistant roleplaying" / "talking to a
    therapist" / "never give advice" register that primes helper behaviour. The
    profile is presented as this person's own established first-person account,
    and the only instruction is to keep speaking as them — no stepping outside,
    no shifting into anyone else's voice.
    """
    return (
        "Below is one person's own account, in their first-person voice, of who "
        "they are and what they are going through. Continue speaking AS this "
        "person: only their own words, thoughts, and feelings. Do not narrate or "
        "comment on them from the outside, do not step out of their voice, and do "
        "not answer as anyone else.\n\n"
        + character_prompt
    )


def frame_patient(character_prompt: str, framing: str = "roleplay") -> str:
    """Frame the patient profile for the simulator per the configured mode.

    See SIMULATOR_FRAMINGS. Both modes are delivered as a single system message
    (Ollama has no raw-completion endpoint), so callers can keep their existing
    [{system}, *transcript] shape and only swap which string this returns.
    """
    if framing not in SIMULATOR_FRAMINGS:
        raise ValueError(
            f"Unknown simulator framing {framing!r}; expected one of {SIMULATOR_FRAMINGS}"
        )
    if framing == "completion":
        return patient_completion_prompt(character_prompt)
    return patient_system_prompt(character_prompt)


# Throwaway prompts used only to elicit the patient's opening turn. They are NOT
# stored in the transcript, so the recorded conversation begins with the patient's
# own words. (The patient is the assistant; a chat request still needs a leading
# user turn to generate against — this one just isn't recorded.) A list, selected
# per arc, so different arcs enter from different angles instead of all opening on
# the same salient detail. A pointed question pulls the patient toward that facet.
PATIENT_OPENERS = [
    "Whenever you're ready, tell me what's been on your mind.",
    "What's been the hardest part of the last few weeks for you?",
    "How have you been feeling, day to day?",
    "What made you want to come in today?",
    "Is there something in particular that's been weighing on you?",
    "How have things been at home, with the people around you?",
    "What does a typical day look like for you right now?",
    "When you think about what's ahead, what comes up?",
    "What's been hardest to sit with lately?",
    "Where would you like to start today?",
]


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
