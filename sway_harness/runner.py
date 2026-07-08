"""
Run-time conversation runner — simulated patient against a bare MUT.

Run-time loop per turn:
  1. MUT replies (bare, zero system prompt).
  2. Generate: Simulator (frozen prompt) → patient turn.
  3. Score: Judge scores the MUT reply (deferred to a separate pass).

Fidelity checking is a build/optimization-time concern (see build.py); the
frozen prompt is trusted at run time, so no per-turn fidelity gate here.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

from client import get_completion, swap_roles, frame_patient, LocalError
from config import ROOT, RoleConfig, RunConfig, ServerConfig, CaptureConfig, PATHS

logger = logging.getLogger(__name__)


def build_system_prompt(frozen_prompt_path: Path, framing: str = "roleplay") -> str:
    """Load a frozen, certified patient system prompt and role-frame it."""
    with open(frozen_prompt_path) as f:
        return frame_patient(f.read().strip(), framing)


def generate_patient_turn(
    server: ServerConfig,
    role: RoleConfig,
    system_prompt: str,
    transcript: List[Dict[str, str]],
    seed: int,
) -> str:
    """Simulator generates a candidate patient turn."""
    messages = [
        {"role": "system", "content": system_prompt},
        *transcript,
    ]
    # Ensure last message is from user (the MUT's last reply)
    if messages[-1]["role"] != "user":
        raise ValueError("Transcript must end with a user message (MUT reply) for the patient to reply to.")

    return get_completion(
        model_path=role.simulator.model_path,
        messages=messages,
        base_url=server.base_url,
        temperature=role.simulator.temperature,
        seed=seed,
        max_tokens=role.simulator.max_tokens,
    )


def run_conversation(
    server: ServerConfig,
    roles: RoleConfig,
    frozen_prompt_path: Path,
    mut_model_path: str,
    run_cfg: RunConfig,
    seed: int,
    initial_message: str,
) -> List[dict]:
    """
    Run one full arc: patient + MUT for num_turns rounds.

    Returns the transcript (list of {role, content} dicts).
    """
    system_prompt = build_system_prompt(frozen_prompt_path, roles.simulator.framing)
    # Patient opens. Dummy user opener satisfies vLLM's "must have user" rule.
    # NOTE: score_turn assumes patient turns at odd indices (patient_idx =
    # 1 + turn_idx*2), so this leading turn must stay to keep scoring aligned.
    initial = generate_patient_turn(server, roles, system_prompt,
                                    [{"role": "user", "content": "[Conversation begins]"}],
                                    seed=seed)
    transcript = [
        {"role": "user", "content": "[Conversation begins]"},
        {"role": "assistant", "content": initial},
    ]

    for turn_idx in range(run_cfg.num_turns):
        # ── Step 0: MUT replies to the patient's current message ──
        try:
            assistant_reply = get_completion(
                model_path=mut_model_path,
                # MUT's perspective: patient=user, MUT=assistant (bare, no system).
                messages=swap_roles(transcript),
                base_url=server.base_url,
                temperature=0.0,
                seed=seed,
                reasoning_effort="none",  # MUT (e.g. Gemma) must answer directly, not reason
            )
            transcript.append({"role": "user", "content": assistant_reply})
        except LocalError as err:
            logger.error("MUT failed at turn %d: %s", turn_idx, err)
            transcript.append({"role": "user", "content": f"[MUT ERROR: {err}]"})

        # ── Step 1: Patient responds to the MUT (frozen prompt, no run-time gate) ──
        patient_turn = generate_patient_turn(
            server, roles, system_prompt, transcript, seed=seed
        )
        transcript.append({"role": "assistant", "content": patient_turn})

    return transcript
