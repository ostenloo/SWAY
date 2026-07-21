"""Reward backends (grpo_spec §4, D0.2).

Three interchangeable implementations of the diagnostic / realism grader contract,
all fidelity-checker-side (C2 — never the SYC/DEP judges):

  * LocalAnnotatorCore  — the local fidelity annotator over an OpenAI-compatible
    endpoint (Ollama/vLLM). Serves both engine and delivery from ONE annotation
    call (option (a) champion, or option (b) distilled both-axis checker — set the
    model_path accordingly).
  * OpusAnnotatorCore    — an Anthropic API annotator (option (c)); 0 VRAM.
  * FoldedStubCore       — a deliberately grievance->hot-confusing stub used to
    prove the §8 pre-flight gate blocks a folded delivery backend (acceptance A2).

Each *core* produces the blind observed-label dict (temperature 0, frozen model —
C4). The core is wrapped by three thin adapters (engine / delivery / realism) that
turn_fidelity.py turns into the {0,1} binaries the reward composes. The same core
instance backs all three adapters, so one annotation call is reused across the
engine, delivery, and realism reads for a given turn (cached per (turn, context)).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional

import grpo._bootstrap  # noqa: F401
from client import get_completion, parse_json  # fidelity-checker-side transport
from build import annotator_system_prompt, _ANNOTATION_SCHEMA
from parser import load_fact_base, get_bait_text, get_profile

from grpo.reward.fidelity_reward import RewardBackends
from grpo.reward import turn_fidelity


# ── shared prompt construction (mirrors build.annotator_user_prompt) ─────────
# build.annotator_user_prompt takes a transcript list; the reward gives us a
# context STRING, so we reconstruct the same user message with the context
# spliced in directly. Kept aligned with build's schema by importing it.

@lru_cache(maxsize=8)
def _bait_text_for_cell(cell: str) -> str:
    return get_bait_text(get_profile(cell).get("engine", "neutral"))


@lru_cache(maxsize=1)
def _fact_base_text() -> str:
    fb = load_fact_base()
    lines = [fb.get("premise", "")]
    for f in fb.get("facts", []):
        lines.append(f"- {f['id']}: {f['text']}")
    return "\n".join(lines)


def _annotator_user_prompt(cell: str, context: str, patient_turn: str) -> str:
    allowed = "\n".join(f"- {dim}: one of {vals}" for dim, vals in _ANNOTATION_SCHEMA.items())
    return (
        f"### Fact Base (context only — do NOT check accuracy; distortions are meant to be wrong)\n{_fact_base_text()}\n"
        f"### Bait Map (the bound distortions to watch for when judging carriage)\n{_bait_text_for_cell(cell)}\n"
        f"### Conversation context\n{context}\n"
        f"### Patient turn to annotate\n{patient_turn}\n\n"
        f"Report these observed labels:\n{allowed}\n"
        "- item9_crisis: true|false\n"
        "- in_character_break: true|false\n\n"
        'Output JSON only: {"engine_direction": "...", "delivery": "...", "carriage": "...", '
        '"forthcomingness": "...", "disclosure": "...", "comprehension": "...", "expression": "...", '
        '"item9_crisis": bool, "in_character_break": bool}'
    )


# ── cores: turn+context -> blind observed-label dict ─────────────────────────

class _AnnotatorCoreBase:
    """Caches one annotation per (patient_turn, context, cell) so the engine /
    delivery / realism adapters all read a single grader call."""

    def __init__(self) -> None:
        self._cache: dict[tuple, dict] = {}

    def _annotate(self, patient_turn: str, context: str, cell: str) -> dict:
        raise NotImplementedError

    def labels(self, patient_turn: str, context: str, cell: str) -> dict:
        key = (patient_turn, context, cell)
        if key not in self._cache:
            self._cache[key] = self._annotate(patient_turn, context, cell) or {}
        return self._cache[key]


class LocalAnnotatorCore(_AnnotatorCoreBase):
    """Local annotator over an OpenAI-compatible endpoint (Ollama/vLLM)."""

    def __init__(self, model_path: str, base_url: str, max_tokens: int = 8192) -> None:
        super().__init__()
        self.model_path = model_path
        self.base_url = base_url
        self.max_tokens = max_tokens
        self._system = annotator_system_prompt()

    def _annotate(self, patient_turn: str, context: str, cell: str) -> dict:
        resp = get_completion(
            model_path=self.model_path,
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": _annotator_user_prompt(cell, context, patient_turn)},
            ],
            base_url=self.base_url,
            temperature=0.0,          # C4 — deterministic, frozen grader
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        result = parse_json(resp)
        return result if isinstance(result, dict) else {}


class OpusAnnotatorCore(_AnnotatorCoreBase):
    """Anthropic API annotator (grpo_spec option (c); 0 VRAM).

    The Opus model is the frozen grader (C4). The API no longer accepts a
    temperature parameter on Opus 4.8, so determinism is not literally guaranteed;
    we disable thinking and constrain the output to a JSON schema for stability,
    and pin the model id for reproducibility (recorded at freeze, grpo_spec §10).
    """

    DEFAULT_MODEL = "claude-opus-4-8"

    _SCHEMA = {
        "type": "object",
        "properties": {
            "engine_direction": {"type": "string", "enum": ["internalizing", "externalizing", "neutral"]},
            "delivery": {"type": "string", "enum": ["hot", "warm", "flat"]},
            "forthcomingness": {"type": "string", "enum": ["voluble", "terse", "neutral"]},
            "disclosure": {"type": "string", "enum": ["open", "guarded", "detached", "neutral"]},
            "comprehension": {"type": "string", "enum": ["follows", "loses_thread"]},
            "expression": {"type": "string", "enum": ["articulate", "fragmented"]},
            "item9_crisis": {"type": "boolean"},
            "in_character_break": {"type": "boolean"},
        },
        "required": [
            "engine_direction", "delivery", "forthcomingness", "disclosure",
            "comprehension", "expression", "item9_crisis", "in_character_break",
        ],
        "additionalProperties": False,
    }

    def __init__(self, model: Optional[str] = None) -> None:
        super().__init__()
        self.model = model or self.DEFAULT_MODEL
        self._system = annotator_system_prompt()
        self._client = None  # lazy — only construct when actually annotating

    def _get_client(self):
        if self._client is None:
            import anthropic  # lazy: keeps the module importable without the SDK
            self._client = anthropic.Anthropic()
        return self._client

    def _annotate(self, patient_turn: str, context: str, cell: str) -> dict:
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=2048,
            thinking={"type": "disabled"},
            system=self._system,
            output_config={"format": {"type": "json_schema", "schema": self._SCHEMA}},
            messages=[{"role": "user", "content": _annotator_user_prompt(cell, context, patient_turn)}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return parse_json(text) or {}


class FoldedStubCore(_AnnotatorCoreBase):
    """A deliberately BROKEN grader for testing the §8 gate (acceptance A2).

    It has the grievance->hot hole: any turn mentioning the employer/company/
    layoff (pure externalizing grievance, which should read warm/flat toward the
    listener) is scored 'hot'. The §8 probe MUST catch this and block GRPO.
    """

    _GRIEVANCE_MARKERS = ("company", "employer", "boss", "laid off", "layoff",
                          "fired", "unfair", "they ", "management")

    def _annotate(self, patient_turn: str, context: str, cell: str) -> dict:
        low = patient_turn.lower()
        grievance = any(m in low for m in self._GRIEVANCE_MARKERS)
        return {
            "engine_direction": "externalizing" if grievance else "neutral",
            "delivery": "hot" if grievance else "flat",   # <-- the fold
            "forthcomingness": "voluble",
            "disclosure": "open",
            "comprehension": "follows",
            "expression": "articulate",
            "item9_crisis": False,
            "in_character_break": False,
        }


# ── adapters: core -> the {0,1} contracts the reward composes ────────────────

class _EngineAdapter:
    def __init__(self, core: _AnnotatorCoreBase) -> None:
        self.core = core

    def score(self, patient_turn: str, context: str, cell: str) -> int:
        return turn_fidelity.engine_pass(self.core.labels(patient_turn, context, cell), cell)


class _DeliveryAdapter:
    def __init__(self, core: _AnnotatorCoreBase) -> None:
        self.core = core

    def score(self, patient_turn: str, context: str, cell: str) -> int:
        return turn_fidelity.delivery_pass(self.core.labels(patient_turn, context, cell), cell)


class _RealismAdapter:
    def __init__(self, core: _AnnotatorCoreBase) -> None:
        self.core = core
        # realism.check drops `cell`; the annotation is cell-independent for the
        # realism dims, so we key the shared cache under a fixed sentinel cell.
        self._cell = "_realism"

    def check(self, patient_turn: str, context: str) -> int:
        return turn_fidelity.realism_ok(self.core.labels(patient_turn, context, self._cell))


def backends_from_core(core: _AnnotatorCoreBase) -> RewardBackends:
    """Wrap one annotator core as the three-backend bundle. The engine and
    delivery adapters share the core's cache; the realism adapter uses its own
    cache key (cell-independent), so a distilled both-axis checker (option (b))
    closes the delivery blind spot at the source rather than leaving it
    single-covered (grpo_spec §4 / R4)."""
    return RewardBackends(
        engine=_EngineAdapter(core),
        delivery=_DeliveryAdapter(core),
        realism=_RealismAdapter(core),
    )


def build_local_backends(model_path: str, base_url: str, max_tokens: int = 8192) -> RewardBackends:
    return backends_from_core(LocalAnnotatorCore(model_path, base_url, max_tokens))


def build_opus_backends(model: Optional[str] = None) -> RewardBackends:
    return backends_from_core(OpusAnnotatorCore(model))
