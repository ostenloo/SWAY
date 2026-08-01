"""The SWAY-specific GRPO reward (grpo_spec §4).

This module is the *pure* reward composition. It is deliberately kept free of
any imports from the Simulator, the Judge, the SYC/DEP instruments, or any
drift-scoring code path — the only inputs it touches are `(patient_turn, P,
context, cell)` and three pluggable backends. That import-cleanliness is the C1
constraint, and `grpo/tests/test_c1_import_guard.py` enforces it by grepping this
file's source.

NON-NEGOTIABLE CONSTRAINTS realised here (grpo_spec §2):

  * C1 — Reward is fidelity only. No MUT reply, no SYC/DEP score, no drift signal
    enters this function. The backends' `.score()` / `.check()` contracts are the
    ONLY channel, and those read the candidate turn + profile context, nothing
    from the drift side.
  * C3 — Reward shape mirrors the gate refactor. Reward is the two DIAGNOSTIC
    per-dimension binaries (engine, delivery) with partial credit. No use of a
    derived 0-3 score. NOTE: §4's multiplicative realism floor has been removed
    by researcher decision — see `RewardBackends` for what that costs.

Note on `P`: the profile prompt is accepted but is intentionally NOT forwarded to
the annotator backends. The fidelity annotator is blind to the target pole (that
is what stops it rubber-stamping toward the profile — see sway_harness/fidelity.py).
Target poles are derived from `cell` downstream, inside the backend, against the
blind observed labels. Keeping `P` in the signature documents that the reward is a
function of (profile, candidate turn) and nothing else, per the wall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class DiagnosticBackend(Protocol):
    """A per-turn diagnostic scorer (engine or delivery). Returns {0, 1}."""

    def score(self, patient_turn: str, context: str, cell: str) -> int: ...


@dataclass(frozen=True)
class RewardBackends:
    """The two interchangeable graders (grpo_spec §4, D0.2).

    `engine` and `delivery` are the reward-bearing diagnostic binaries. Both run
    at temperature 0 on a pinned, frozen checkpoint (C4) — that discipline lives
    in the backend implementation, not here.

    **No realism backend.** §4 specifies a realism floor multiplying the
    diagnostics; it has been removed by researcher decision. The consequence is
    recorded rather than hidden: R3 (degenerate-but-on-profile collapse — the
    documented RL-sim failure where a finetuned simulator drifts into on-target
    but degenerate text) is now mitigated by the KL penalty alone. The §9
    high-advantage audit remains the observability path — degenerate turns
    earning advantage will surface there.
    """

    engine: DiagnosticBackend
    delivery: DiagnosticBackend


def fidelity_reward(
    patient_turn: str,
    P: str,
    context: str,
    cell: str,
    backends: RewardBackends,
) -> float:
    """Scalar fidelity reward in [0, 1] for one candidate patient turn.

    P        = frozen profile prompt for the cell (disposition + pressure schedule).
    context  = conversation history up to this turn (the preceding turns).
    cell     = cell id (b1..b6, ...) — selects the target poles inside the backend.

    MUST read only these inputs. No MUT reply, no drift signal (C1).
    """
    # --- diagnostic binaries (the only reward-bearing signal) --- C3
    engine_pass = backends.engine.score(patient_turn, context, cell)      # {0,1}
    delivery_pass = backends.delivery.score(patient_turn, context, cell)  # {0,1}

    # Partial credit (0 / 0.5 / 1.0), NOT all-or-nothing. This is the
    # highest-leverage anti-stall knob (grpo_spec §4): an all-fail group has
    # within-group std -> 0 and an undefined advantage, the same emptiness that
    # stalls rejection sampling. Half-credit keeps groups non-degenerate.
    return 0.5 * engine_pass + 0.5 * delivery_pass


# ── C1 self-audit helpers ───────────────────────────────────────────────────
# The reward module must import nothing from the Judge / [A] / [B] / drift paths.
# These names are the grep-able rule; the test module asserts none of them appear
# as imports in this file's source.
FORBIDDEN_IMPORT_TOKENS = (
    "scoring",        # sway_harness/scoring.py — drift capitulation curve
    "runner",         # run-time loop
    "validate_judge",
    "judge",          # any judge module / [A]/[B] instrument
    "syc_spec",
    "dep_spec",
    "capitulation",
    "drift",
)


# ── C8: reward family-disjoint from the finetune base ───────────────────────
# Every reward backend (engine champion, delivery champion) MUST come from a
# model family disjoint from `base_model`. Rationale (grpo_spec C8): as a
# *reward*, a shared-family blind spot is not merely unnoticed (the
# checker≠Simulator concern, PIPE §10.1) — it becomes a gradient the policy
# farms. D0.1 anchors the base to Qwen2.5-14B, so in practice both champions
# must be non-Qwen.
#
# Substring table over model tags as they appear in HF ids and Ollama tags
# ("Qwen/Qwen2.5-14B-Instruct", "command-r7b:latest", "glm4:9b"). Deliberately
# stdlib-only so this module keeps its empty import list (C1).
_FAMILY_MARKERS = (
    ("qwen", "qwen"),
    ("qwq", "qwen"),
    ("command-r", "cohere"),
    ("command_r", "cohere"),
    ("aya", "cohere"),
    ("glm", "zhipu"),
    ("chatglm", "zhipu"),
    ("llama", "llama"),
    ("mistral", "mistral"),
    ("ministral", "mistral"),
    ("mixtral", "mistral"),
    ("nemo", "mistral"),
    ("gemma", "gemma"),
    ("phi", "phi"),
    ("deepseek", "deepseek"),
    ("exaone", "exaone"),
    ("falcon", "falcon"),
    ("olmo", "olmo"),
    ("dolphin", "dolphin"),
    ("granite", "granite"),
    ("yi", "yi"),
)


class FamilyDisjointnessError(AssertionError):
    """Raised when a reward backend shares a model family with the finetune base."""


def model_family(model_name: str) -> str:
    """Best-effort family for a HF id or Ollama tag. `unknown` if unrecognised."""
    low = (model_name or "").lower()
    for marker, family in _FAMILY_MARKERS:
        if marker in low:
            return family
    return "unknown"


def assert_family_disjoint(base_model: str, **backends: str) -> dict[str, str]:
    """Hard C8 rail, asserted where the reward is constructed.

    `backends` maps role -> model name (e.g. engine=..., delivery=...). Raises
    `FamilyDisjointnessError` if any shares a family with `base_model`, or if a
    family cannot be determined — an unrecognised tag is treated as a FAILURE,
    not a pass, because silently admitting an unknown model is exactly how a
    same-family reward would slip in. Returns the resolved families for logging.
    """
    base_family = model_family(base_model)
    if base_family == "unknown":
        raise FamilyDisjointnessError(
            f"C8: cannot determine the model family of base_model {base_model!r}. "
            "Add it to _FAMILY_MARKERS rather than skipping the check."
        )
    resolved = {"base": base_family}
    for role, name in backends.items():
        family = model_family(name)
        if family == "unknown":
            raise FamilyDisjointnessError(
                f"C8: cannot determine the model family of the {role} backend {name!r}. "
                "Add it to _FAMILY_MARKERS rather than skipping the check."
            )
        if family == base_family:
            raise FamilyDisjointnessError(
                f"C8 BREACH: the {role} reward backend {name!r} is family {family!r}, the "
                f"same family as base_model {base_model!r}. A shared-family blind spot in a "
                "REWARD becomes a gradient the policy farms (grpo_spec C8). Choose a "
                "champion from a disjoint family."
            )
        resolved[role] = family
    return resolved
