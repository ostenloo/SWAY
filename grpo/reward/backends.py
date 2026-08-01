"""Reward backends — the two local axis-specialists (grpo_spec §4, D0.2, C8).

**D0.2 (RESOLVED): no external API spend.** The reward is two *distinct* local
champions:

  * `engine_backend`   = the engine champion (won the engine κ)
  * `delivery_backend` = the delivery champion (won the delivery κ), **decomposed**
    into Q1/Q2 per §8.1 — `hot = Q1`

§4's multiplicative realism floor has been **removed** by researcher decision:
there is no third grader and no realism term in the reward. See
`fidelity_reward.RewardBackends` for what that costs (R3).

Because engine and delivery are answered by *different* models, their blind spots
are **axis-local and uncorrelated**. This is strictly better than one distilled
both-axis checker, where a single model's blind spot is shared across both axes —
that configuration is **SHELVED** (§12 R4) and is deliberately not constructible
here. Revisit only if the run count climbs enough that in-loop reward cost
dominates, and then distill on **real rollouts**, never the prompt-opt corpus.

**Opus is not a reward backend.** A chat-window Opus cannot sit in the gradient
loop — the reward is called at temp 0, `steps × batch × G` times per run. The
in-loop Opus path that existed against the earlier draft of the spec has been
removed. Chat-Opus survives only as the *offline* §10 cross-check (see
`cert/certify_and_freeze.py`), never here.

All backends are fidelity-checker-side (C2 — never the SYC/DEP judges) and run at
temperature 0 against a pinned checkpoint (C4).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import grpo._bootstrap  # noqa: F401
from client import get_completion, parse_json  # fidelity-checker-side transport
from build import annotator_system_prompt, _ANNOTATION_SCHEMA
from parser import load_fact_base, get_bait_text, get_profile

from grpo.reward.fidelity_reward import RewardBackends, assert_family_disjoint
from grpo.reward import turn_fidelity
from grpo.reward import delivery_decompose as dd


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


def _engine_user_prompt(cell: str, context: str, patient_turn: str) -> str:
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


# ── cores: turn+context -> a parsed label dict ───────────────────────────────

class _CoreBase:
    """Caches one call per (patient_turn, context, cell) so repeated reads of the
    same turn inside one reward evaluation hit the model once."""

    #: Recorded on the core so the freeze manifest can name every grader (A5).
    identity: str = "unknown"

    def __init__(self) -> None:
        self._cache: dict[tuple, dict] = {}

    def _annotate(self, patient_turn: str, context: str, cell: str) -> dict:
        raise NotImplementedError

    def labels(self, patient_turn: str, context: str, cell: str) -> dict:
        key = (patient_turn, context, cell)
        if key not in self._cache:
            self._cache[key] = self._annotate(patient_turn, context, cell) or {}
        return self._cache[key]


class _LocalCore(_CoreBase):
    """Shared transport for a local annotator over an OpenAI-compatible endpoint
    (Ollama/vLLM). Temperature 0, pinned model (C4)."""

    def __init__(self, model_path: str, base_url: str, max_tokens: int = 8192) -> None:
        super().__init__()
        self.model_path = model_path
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.identity = f"local:{model_path}"

    def _system(self) -> str:
        raise NotImplementedError

    def _user(self, patient_turn: str, context: str, cell: str) -> str:
        raise NotImplementedError

    def _annotate(self, patient_turn: str, context: str, cell: str) -> dict:
        resp = get_completion(
            model_path=self.model_path,
            messages=[
                {"role": "system", "content": self._system()},
                {"role": "user", "content": self._user(patient_turn, context, cell)},
            ],
            base_url=self.base_url,
            temperature=0.0,          # C4 — deterministic, frozen grader
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        result = parse_json(resp)
        return result if isinstance(result, dict) else {}


class EngineChampionCore(_LocalCore):
    """The ENGINE champion (grpo_spec §4). Answers the full blind observed-label
    schema; only `engine_direction` is read for the reward."""

    def _system(self) -> str:
        return annotator_system_prompt()

    def _user(self, patient_turn: str, context: str, cell: str) -> str:
        return _engine_user_prompt(cell, context, patient_turn)


class DeliveryChampionCore(_LocalCore):
    """The DELIVERY champion, **decomposed** (grpo_spec §8.1).

    Asks Q1 (hostility toward the listener) and Q2 (grievance toward an absent
    party) separately and reports both. `hot = Q1` is applied downstream in
    `delivery_decompose`, never here — this core only observes.
    """

    def _system(self) -> str:
        return dd.delivery_decompose_system_prompt()

    def _user(self, patient_turn: str, context: str, cell: str) -> str:
        return dd.delivery_decompose_user_prompt(context, patient_turn)

    def decompose(self, patient_turn: str, context: str, cell: str) -> dd.DeliveryDecomposition:
        return dd.decomposition_from_labels(self.labels(patient_turn, context, cell))


# ── adapters: core -> the {0,1} contracts the reward composes ────────────────

class EngineAdapter:
    """`engine_backend` — reads the engine champion's `engine_direction`."""

    def __init__(self, core: _CoreBase) -> None:
        self.core = core

    def score(self, patient_turn: str, context: str, cell: str) -> int:
        return turn_fidelity.engine_pass(self.core.labels(patient_turn, context, cell), cell)


class DeliveryAdapter:
    """`delivery_backend` — the DECOMPOSED delivery read (§8.1), `hot = Q1`.

    `.decompose()` exposes the underlying Q1/Q2 so the §9 monitor can log which
    sub-question is being farmed.
    """

    def __init__(self, core: _CoreBase) -> None:
        self.core = core

    def decompose(self, patient_turn: str, context: str, cell: str) -> dd.DeliveryDecomposition:
        return dd.decomposition_from_labels(self.core.labels(patient_turn, context, cell))

    def score(self, patient_turn: str, context: str, cell: str) -> int:
        return dd.delivery_pass_decomposed(self.decompose(patient_turn, context, cell), cell)


# ── construction ─────────────────────────────────────────────────────────────

def build_champion_backends(
    engine_model: str,
    delivery_model: str,
    base_url: str,
    base_model: str,
    max_tokens: int = 8192,
) -> RewardBackends:
    """Build the reward from the two local champions.

    TWO graders, and no realism floor at all — §4's floor was removed by
    researcher decision, so a turn costs one engine call and one delivery call.

    Asserts **C8** at construction: both champions must be family-disjoint from
    `base_model`. As a *reward*, a shared-family blind spot is not merely unnoticed
    (the checker≠Simulator concern, PIPE §10.1) — it becomes a gradient the policy
    farms, so this is a hard rail, not a warning.
    """
    assert_family_disjoint(base_model, engine=engine_model, delivery=delivery_model)
    return RewardBackends(
        engine=EngineAdapter(EngineChampionCore(engine_model, base_url, max_tokens)),
        delivery=DeliveryAdapter(DeliveryChampionCore(delivery_model, base_url, max_tokens)),
    )


def backend_identities(backends: RewardBackends) -> dict[str, str]:
    """Both grader identities, for the freeze manifest (A5)."""
    return {
        "engine": getattr(backends.engine.core, "identity", "unknown"),
        "delivery": getattr(backends.delivery.core, "identity", "unknown"),
    }


# ── test doubles for the §8 gates (acceptance A2) ────────────────────────────

class FoldedDeliveryStubCore(_CoreBase):
    """A deliberately BROKEN delivery champion, for proving the gates block (A2).

    It has the grievance→hot hole in its purest form: any turn mentioning the
    employer is reported as hostility toward the LISTENER (Q1 true), i.e. it
    cannot answer Q1 in isolation — precisely the §8.1 caveat. Used by the §8.3
    smoke test and the decomposition tests.
    """

    identity = "stub:folded_delivery"

    _GRIEVANCE_MARKERS = ("company", "employer", "boss", "laid off", "layoff",
                          "fired", "unfair", "management", "restructur", "severance",
                          "department", "they ")

    def _annotate(self, patient_turn: str, context: str, cell: str) -> dict:
        low = patient_turn.lower()
        grievance = any(m in low for m in self._GRIEVANCE_MARKERS)
        return {
            "q1_hostility_toward_listener": grievance,   # <-- the fold
            "q2_grievance_toward_absent_party": grievance,
        }


class CleanDeliveryStubCore(_CoreBase):
    """A delivery champion that answers Q1 correctly — the control for A2.

    Q1 fires only on listener-directed markers; Q2 fires on employer markers.
    Used to show the gates PASS a discriminator that separates the two.
    """

    identity = "stub:clean_delivery"

    _LISTENER_MARKERS = ("you ", "you're", "your ", "you'", "don't you", "shut up")
    _HOSTILE_MARKERS = ("useless", "garbage", "worthless", "don't get it", "stop ",
                        "why am i even", "all the same", "shut up", "brilliant,",
                        "dare tell me", "not listening")
    _GRIEVANCE_MARKERS = ("company", "employer", "boss", "laid off", "layoff", "fired",
                          "unfair", "management", "restructur", "severance", "department",
                          "screwed me", "hr drones", "they did", "let me go")

    def _annotate(self, patient_turn: str, context: str, cell: str) -> dict:
        low = patient_turn.lower()
        aimed_at_listener = any(m in low for m in self._LISTENER_MARKERS)
        hostile = any(m in low for m in self._HOSTILE_MARKERS)
        return {
            "q1_hostility_toward_listener": aimed_at_listener and hostile,
            "q2_grievance_toward_absent_party": any(m in low for m in self._GRIEVANCE_MARKERS),
        }


def stub_backends(delivery_core: _CoreBase, engine_pass: int = 1) -> RewardBackends:
    """Wrap a stub delivery core with a constant engine read, for gate tests."""

    class _Const:
        core = None

        def __init__(self, v):
            self.v = v

        def score(self, *a):
            return self.v

        def check(self, *a):
            return self.v

    return RewardBackends(
        engine=_Const(engine_pass),
        delivery=DeliveryAdapter(delivery_core),
    )
