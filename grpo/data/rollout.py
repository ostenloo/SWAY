"""Rollout / data generation for GRPO (grpo_spec §5).

State construction, per-turn group formation, and interlocutor handling. The
policy is the Simulator (base + QLoRA adapter); it talks to one or more bare,
near-zero-system-prompt interlocutors.

§5.3 asks for >=2 partners so kept turns span a spread of therapist-move contexts
(the training-time analog of cross-interlocutor certification, PIPE §4.2). The
configured setup deviates deliberately and trains on ONE — the same reference
interlocutor the prompt-opt builds used. See `build_states` and the rationale in
`configs/grpo.yaml`.

"Bare" here means the harness's `REF_SYSTEM_PROMPT` — minimal and non-directive
("do not therapize, advise, or take sides"), not literally an empty system
prompt. That is a small, knowing gap against §5.3's "zero-system-prompt": it is
the prompt the existing pipeline rolls against, so it keeps the GRPO and
prompt-opt distributions comparable.

Generation is pluggable behind `generate_patient_turn` / `generate_interlocutor`:
the default hits an OpenAI-compatible endpoint via the harness client (works
against the Ollama-served policy for warm-start data + certification), while the
GRPO loop swaps in vLLM-backed generation (grpo_spec §12) by passing its own
callables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import grpo._bootstrap  # noqa: F401
from client import get_completion, swap_roles, frame_patient, PATIENT_OPENERS
from build import REF_SYSTEM_PROMPT


# A generate function: (system_prompt, messages, seed) -> str
GenerateFn = Callable[[str, List[dict], int], str]


@dataclass
class Interlocutor:
    """A bare interlocutor the patient talks to. `system` is minimal by design
    (grpo_spec §5.3 — zero/near-zero system prompt)."""
    name: str
    model_path: str
    base_url: str
    system: str = REF_SYSTEM_PROMPT
    temperature: float = 0.3
    max_tokens: int = 1024


@dataclass
class RolloutState:
    """A training example's state: profile prompt + history prefix (grpo_spec §5.1).

    The GRPO prompt is `P` + `context`; the completion is one patient turn.
    `transcript` keeps the canonical message list (assistant=patient, user=interlocutor)
    so the context string and next-turn generation stay consistent.
    """
    cell: str
    P: str                                   # frozen profile prompt for the cell
    transcript: List[dict] = field(default_factory=list)
    interlocutor: Optional[str] = None       # which bare model built this prefix

    def context(self) -> str:
        """History prefix as the string the reward + annotator consume."""
        lines = []
        for msg in self.transcript:
            speaker = "Patient" if msg["role"] == "assistant" else "Model"
            lines.append(f"[{speaker}]: {msg['content']}")
        return "\n".join(lines[-6:])  # last 6 turns, matching the annotator window


def _default_generate(model_path: str, base_url: str, temperature: float,
                      max_tokens: int) -> GenerateFn:
    def _fn(system_prompt: str, messages: List[dict], seed: int) -> str:
        msgs = ([{"role": "system", "content": system_prompt}] + messages
                if system_prompt else messages)
        return get_completion(
            model_path=model_path, messages=msgs, base_url=base_url,
            temperature=temperature, seed=seed, max_tokens=max_tokens,
        )
    return _fn


def roll_prefix(
    P: str,
    cell: str,
    interlocutor: Interlocutor,
    policy_generate: GenerateFn,
    prefix_turns: int,
    seed: int,
    framing: str = "roleplay",
) -> RolloutState:
    """Roll the current policy against one bare interlocutor for `prefix_turns`
    patient turns, returning the resulting state (the history prefix)."""
    system_prompt = frame_patient(P, framing)
    interlocutor_generate = _default_generate(
        interlocutor.model_path, interlocutor.base_url,
        interlocutor.temperature, interlocutor.max_tokens,
    )

    opener = PATIENT_OPENERS[seed % len(PATIENT_OPENERS)]
    first = policy_generate(system_prompt, [{"role": "user", "content": opener}], seed)
    transcript: List[dict] = [{"role": "assistant", "content": first}]

    for t in range(prefix_turns - 1):
        ref = interlocutor_generate(
            interlocutor.system, swap_roles(transcript), seed + 2 * t + 1,
        )
        transcript.append({"role": "user", "content": ref})
        nxt = policy_generate(system_prompt, transcript, seed + 2 * t + 2)
        transcript.append({"role": "assistant", "content": nxt})

    return RolloutState(cell=cell, P=P, transcript=transcript, interlocutor=interlocutor.name)


def build_states(
    P: str,
    cell: str,
    interlocutors: List[Interlocutor],
    policy_generate: GenerateFn,
    n_states: int,
    prefix_turns: int = 4,
    seed_base: int = 0,
    framing: str = "roleplay",
) -> List[RolloutState]:
    """Build `n_states` history prefixes, round-robining across the bare
    interlocutors given.

    With >=2 interlocutors this is the cross-interlocutor spread of §5.3. A
    SINGLE interlocutor is also allowed, and is the current configured choice —
    a deliberate, documented deviation from §5.3 (see `configs/grpo.yaml`): the
    one partner is the same `reference_interlocutor` the prompt-opt builds used,
    which keeps warm-start on the distribution the existing artifacts came from
    and keeps the A3 control like-for-like. When training on one partner, the
    interlocutor-robustness claim rests entirely on §10 step 2 — certification
    against a bare model never seen in training — so that step is doing more work
    than the spec assumed and must not be skipped or reused across iterations.
    """
    if not interlocutors:
        raise ValueError("at least one bare interlocutor is required")
    states = []
    for i in range(n_states):
        interlocutor = interlocutors[i % len(interlocutors)]
        states.append(roll_prefix(
            P, cell, interlocutor, policy_generate,
            prefix_turns, seed_base + i * 1000, framing,
        ))
    return states


def sample_group(
    state: RolloutState,
    policy_generate: GenerateFn,
    group_size: int,
    seed_base: int = 0,
    framing: str = "roleplay",
) -> List[str]:
    """Sample G candidate patient turns from the current policy at one state
    (per-turn grouping, grpo_spec §5.2). The interlocutor's last move is already
    the tail of the transcript, so each completion is the patient's next turn."""
    system_prompt = frame_patient(state.P, framing)
    # Ensure the transcript ends on a user (interlocutor) turn so each sample is a
    # fresh patient reply, not a continuation of the patient's own last turn.
    msgs = state.transcript
    if msgs and msgs[-1]["role"] == "assistant":
        msgs = msgs + [{"role": "user", "content": "Go on."}]
    return [
        policy_generate(system_prompt, msgs, seed_base + g)
        for g in range(group_size)
    ]
