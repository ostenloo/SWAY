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
    per-dimension binaries (engine, delivery) with partial credit; realism is a
    MULTIPLICATIVE FLOOR, never an additive term. No use of a derived 0-3 score.

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


class RealismBackend(Protocol):
    """The realism floor check. Returns {0, 1}. Never maximised — see below."""

    def check(self, patient_turn: str, context: str) -> int: ...


@dataclass(frozen=True)
class RewardBackends:
    """The three interchangeable graders (grpo_spec §4, D0.2).

    `engine` and `delivery` are the reward-bearing diagnostic binaries. `realism`
    is the multiplicative floor. All three run at temperature 0 on a pinned,
    frozen checkpoint (C4) — that discipline lives in the backend implementation,
    not here.
    """

    engine: DiagnosticBackend
    delivery: DiagnosticBackend
    realism: RealismBackend


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
    diagnostic = 0.5 * engine_pass + 0.5 * delivery_pass

    # --- realism as a MULTIPLICATIVE FLOOR, not a reward term --- C3
    # A turn that fails realism scores 0 regardless of the diagnostics and can
    # never be reinforced; but realism is never *maximised*, so the policy cannot
    # farm it. This preserves "realism is a constraint, not an objective" (PS)
    # inside the RL loop and blocks the degenerate-but-on-profile collapse.
    realism_ok = backends.realism.check(patient_turn, context)            # {0,1}
    return diagnostic * realism_ok


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
