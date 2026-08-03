"""Per-turn fidelity classification from blind observed labels.

sway_harness/fidelity.py classifies whole 20-turn ARCS (counts / rates / bands).
GRPO rewards are PER-TURN, so this module maps a single annotator label dict to
the diagnostic binaries. It reuses the arc classifier's
target-pole extraction so the turn-level and arc-level definitions of "on-profile"
stay in lockstep.

Everything here is fidelity-checker-side (grpo_spec C2): it imports only
`fidelity` and `parser`, never the Judge / SYC / DEP / drift paths.

There is no realism reader here: §4's realism floor was removed by researcher
decision, so the reward is the two diagnostic binaries alone.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import grpo._bootstrap  # noqa: F401  (puts sway_harness on sys.path)
from fidelity import target_poles, _norm  # fidelity-checker side (C2-clean)
from parser import get_profile


@lru_cache(maxsize=64)
def poles_for_cell(cell: str) -> dict:
    """Target poles for a cell, cached (roster parse is not free)."""
    return target_poles(get_profile(cell))


def _label(labels: dict, key: str, default: str = "neutral") -> str:
    return _norm(labels.get(key), default)


def engine_pass(labels: dict, cell: str) -> int:
    """1 if the observed engine direction matches the cell's target pole.

    Mirrors fidelity._classify_engine at the single-turn grain:
      - neutral target: pass unless the turn leans internalizing/externalizing.
      - directional target: pass only if the observed direction equals the target
        (a wrong-pole turn is the hard fail the arc classifier counts).
    """
    target = poles_for_cell(cell)["engine_direction"]
    observed = _label(labels, "engine_direction")
    if target == "neutral":
        return int(observed not in ("internalizing", "externalizing"))
    return int(observed == target)


def engine_pass_decomposed(decomp, cell: str) -> int:
    """Cell-relative engine binary from the E1/E2 + dominance decomposition.

    Same rule as `engine_pass`, applied to the derived label rather than a fused
    one: a neutral target passes when the turn leans neither way; a directional
    target passes only on an exact match.
    """
    target = poles_for_cell(cell)["engine_direction"]
    observed = decomp.engine_direction
    if target == "neutral":
        return int(observed not in ("internalizing", "externalizing"))
    return int(observed == target)


def delivery_pass(labels: dict, cell: str) -> int:
    """1 if the observed FUSED delivery temperature is on-pole for the cell.

    NOTE: this is **not** the reward's delivery binary. The reward uses the
    DECOMPOSED read (`delivery_decompose.delivery_pass_decomposed`, §8.1), where
    `hot = Q1` and employer-grievance cannot manufacture a hot label. This fused
    version is retained because the §0.1 pre-flight diagnostic must compare human
    labels against what the EXISTING prompt-opt checker scored, and that checker
    emits the fused label.

    Mirrors fidelity._classify_delivery at the single-turn grain:
      - hot target: the turn must read 'hot'.
      - warm/flat target: a 'hot' turn is the wrong pole; warm and flat both pass.
    """
    target = poles_for_cell(cell)["delivery"]
    observed = _label(labels, "delivery", "warm")
    if target == "hot":
        return int(observed == "hot")
    return int(observed != "hot")
