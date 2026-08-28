"""Reward backends — the two local axis-specialists (grpo_spec §4, D0.2, C8).

**D0.2 (RESOLVED): no external API spend.** The reward is two *distinct* local
champions:

  * `engine_backend`   = the engine champion, **decomposed** into E1 (blames
    self) / E2 (blames others) + a dominance resolution
  * `delivery_backend` = the delivery champion, **decomposed** into Q1 (hostility
    toward the listener) / Q3 (closeness-pulling toward the listener)

Each axis's three-way label is derived from its own pair of questions, following
CODING_GUIDE.md. Neither champion sees the other's axis, and neither is told the
cell's target pole.

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

from typing import Optional

import grpo._bootstrap  # noqa: F401
from client import get_completion, parse_json  # fidelity-checker-side transport

from grpo.reward.fidelity_reward import RewardBackends, assert_family_disjoint
from grpo.reward import turn_fidelity
from grpo.reward import delivery_decompose as dd
from grpo.reward import engine_decompose as ed


# ── cores: turn+context -> a parsed label dict ───────────────────────────────

class _CoreBase:
    """Caches one call per (patient_turn, context, cell) so repeated reads of the
    same turn inside one reward evaluation hit the model once."""

    #: Recorded on the core so the freeze manifest can name every grader (A5).
    identity: str = "unknown"

    #: The keys this core's decomposition needs. Used by the MISSING-KEY COUNTER
    #: (grpo_spec_2 §4.2): decoding is constrained to valid JSON, so syntactic
    #: parse failures are ~zero — but that does not guarantee the right KEYS.
    #: `{"e1": true}` parses cleanly, `.get("e1_blames_self")` returns None,
    #: `_as_bool` defaults it False, and the turn silently reads neutral/flat —
    #: the UNMARKED class that sets the density denominator. Under the old binary
    #: that was a safe default; under the band's density term it is a
    #: downward-only bias on `d`, indistinguishable from the simulator choosing
    #: not to express the axis. Given the observed Qwen guardrail break
    #: (mid-arc refusal / language flip), it is not assumed away.
    expected_keys: tuple = ()

    def __init__(self) -> None:
        self._cache: dict[tuple, dict] = {}
        self.n_annotations = 0
        self.n_missing_all_keys = 0

    def _annotate(self, patient_turn: str, context: str, cell: str) -> dict:
        raise NotImplementedError

    def labels(self, patient_turn: str, context: str, cell: str) -> dict:
        key = (patient_turn, context, cell)
        if key not in self._cache:
            result = self._annotate(patient_turn, context, cell) or {}
            # Counted on the MISS path only: a cache hit is not a fresh grader
            # call, and counting it would dilute the rate the §9 halt reads.
            self.n_annotations += 1
            if self.expected_keys and not any(k in result for k in self.expected_keys):
                self.n_missing_all_keys += 1
            self._cache[key] = result
        return self._cache[key]

    @property
    def missing_key_rate(self) -> float:
        """Fraction of grader calls that came back with NONE of the expected keys.

        §9 halts above ~5% (`reward.missing_key_halt_rate`). If it is genuinely
        zero the telemetry costs nothing.
        """
        return self.n_missing_all_keys / self.n_annotations if self.n_annotations else 0.0

    def reset_counters(self) -> None:
        """Zero the missing-key counter — called per step so the rate is per-step."""
        self.n_annotations = 0
        self.n_missing_all_keys = 0

    def clear_cache(self) -> None:
        """Drop the memo table.

        §9 requires this before each high-advantage audit (the §0.2 data-hygiene
        flag): auditing cached labels re-reads the same grader verdict rather
        than re-grading, which is exactly what an audit is supposed to check.
        """
        self._cache.clear()


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
    """The ENGINE champion, **decomposed** (E1/E2 + dominance).

    Asks whether the turn blames the self, whether it blames others, and which
    bend dominates when both appear. `engine_direction` is derived downstream in
    `engine_decompose`, never here — this core only observes.
    """

    expected_keys = tuple(ed.DECOMP_SCHEMA)

    def _system(self) -> str:
        return ed.engine_decompose_system_prompt()

    def _user(self, patient_turn: str, context: str, cell: str) -> str:
        return ed.engine_decompose_user_prompt(context, patient_turn)

    def decompose(self, patient_turn: str, context: str, cell: str) -> ed.EngineDecomposition:
        return ed.decomposition_from_labels(self.labels(patient_turn, context, cell))


class DeliveryChampionCore(_LocalCore):
    """The DELIVERY champion, **decomposed** (grpo_spec §8.1).

    Asks Q1 (hostility toward the listener) and Q2 (grievance toward an absent
    party) separately and reports both. `hot = Q1` is applied downstream in
    `delivery_decompose`, never here — this core only observes.
    """

    expected_keys = tuple(dd.DECOMP_SCHEMA)

    def _system(self) -> str:
        return dd.delivery_decompose_system_prompt()

    def _user(self, patient_turn: str, context: str, cell: str) -> str:
        return dd.delivery_decompose_user_prompt(context, patient_turn)

    def decompose(self, patient_turn: str, context: str, cell: str) -> dd.DeliveryDecomposition:
        return dd.decomposition_from_labels(self.labels(patient_turn, context, cell))


# ── adapters: core -> the {0,1} contracts the reward composes ────────────────

class EngineAdapter:
    """`engine_backend` — the DECOMPOSED engine read (E1/E2 + dominance).

    `.decompose()` exposes the sub-answers so the monitor can log which component
    drove the label and whether the turn was mixed.
    """

    def __init__(self, core: _CoreBase) -> None:
        self.core = core

    @property
    def identity(self) -> str:
        """Delegated so CB1 (`assert_calibration_backends`) can read it off the
        object the reward actually holds."""
        return getattr(self.core, "identity", "unknown")

    def decompose(self, patient_turn: str, context: str, cell: str) -> ed.EngineDecomposition:
        return ed.decomposition_from_labels(self.core.labels(patient_turn, context, cell))

    def label(self, patient_turn: str, context: str, cell: str) -> str:
        """The CATEGORICAL — `{internalizing, externalizing, neutral}` (§4.2).

        This is what the band reward consumes. It is a thin adapter over the
        existing decomposition, not a new grader: mixed turns (E1 and E2 both
        fire) resolve via the champion's own `dominant`, exactly as before. They
        are not split and not excluded.
        """
        return self.decompose(patient_turn, context, cell).engine_direction

    def score(self, patient_turn: str, context: str, cell: str) -> int:
        """The per-turn binary. DERIVED, not the reward (§4.2) — it survives for
        the RFT warm-start filter (§6) and the §9 audit only."""
        return turn_fidelity.engine_pass_decomposed(
            self.decompose(patient_turn, context, cell), cell)


class DeliveryAdapter:
    """`delivery_backend` — the DECOMPOSED delivery read (§8.1), `hot = Q1`.

    `.decompose()` exposes the underlying Q1/Q2 so the §9 monitor can log which
    sub-question is being farmed.
    """

    def __init__(self, core: _CoreBase) -> None:
        self.core = core

    @property
    def identity(self) -> str:
        return getattr(self.core, "identity", "unknown")

    def decompose(self, patient_turn: str, context: str, cell: str) -> dd.DeliveryDecomposition:
        return dd.decomposition_from_labels(self.core.labels(patient_turn, context, cell))

    def label(self, patient_turn: str, context: str, cell: str) -> str:
        """The CATEGORICAL — `{hot, warm, flat}` (§4.2). `hot = Q1` regardless of
        employer-directed grievance (§8.1); hostility dominates a mixed read."""
        return self.decompose(patient_turn, context, cell).delivery

    def score(self, patient_turn: str, context: str, cell: str) -> int:
        """The per-turn binary. DERIVED, not the reward (§4.2)."""
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
    """Both grader identities, for the freeze manifest (A5) and the CB1 stamp on
    the band-calibration artifact (grpo_spec_2 C4)."""
    return {
        "engine": getattr(backends.engine, "identity", "unknown"),
        "delivery": getattr(backends.delivery, "identity", "unknown"),
    }


def _cores(backends: RewardBackends):
    for adapter in (backends.engine, backends.delivery):
        core = getattr(adapter, "core", None)
        if core is not None:
            yield core


def missing_key_rates(backends: RewardBackends) -> dict:
    """Per-axis missing-key telemetry (grpo_spec_2 §4.2, logged per step by §9).

    `max_rate` is the number the halt compares against
    `reward.missing_key_halt_rate` (~0.05): a bad axis must not be averaged away
    by a healthy one.
    """
    rates = {}
    for axis, adapter in (("engine", backends.engine), ("delivery", backends.delivery)):
        core = getattr(adapter, "core", None)
        rates[axis] = {
            "n_annotations": getattr(core, "n_annotations", 0),
            "n_missing_all_keys": getattr(core, "n_missing_all_keys", 0),
            "rate": getattr(core, "missing_key_rate", 0.0),
        }
    rates["max_rate"] = max((rates[a]["rate"] for a in ("engine", "delivery")), default=0.0)
    return rates


def reset_missing_key_counters(backends: RewardBackends) -> None:
    """Zero both counters so the next step's rate is that step's rate."""
    for core in _cores(backends):
        core.reset_counters()


def clear_backend_caches(backends: RewardBackends) -> None:
    """Drop both memo tables — §9 requires this before each high-advantage audit
    (`monitoring.clear_backend_cache_before_audit`, the §0.2 hygiene flag)."""
    for core in _cores(backends):
        core.clear_cache()


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
            "q3_closeness_toward_listener": False,
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
    _WARM_MARKERS = ("you get it", "you get me", "only one who", "thank you for",
                     "you understand", "hearing me out")

    def _annotate(self, patient_turn: str, context: str, cell: str) -> dict:
        low = patient_turn.lower()
        aimed_at_listener = any(m in low for m in self._LISTENER_MARKERS)
        hostile = any(m in low for m in self._HOSTILE_MARKERS)
        return {
            "q1_hostility_toward_listener": aimed_at_listener and hostile,
            "q3_closeness_toward_listener": any(m in low for m in self._WARM_MARKERS),
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
