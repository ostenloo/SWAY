"""A1 — the reward module reads only fidelity signal and imports no drift path.

Two checks:
  * grep guard: fidelity_reward.py's source imports nothing from the Judge / [A] /
    [B] / drift-scoring code paths (C1's grep-able rule).
  * unit: fidelity_reward reads ONLY (patient_turn, P, context, cell, backends) —
    proven with a recording backend that captures exactly what it was handed.
"""

import ast
from pathlib import Path

import grpo._bootstrap  # noqa: F401
from grpo.reward.fidelity_reward import (
    RewardBackends, fidelity_reward, FORBIDDEN_IMPORT_TOKENS,
)

REWARD_SRC = Path(__file__).resolve().parent.parent / "reward" / "fidelity_reward.py"


def _imported_names(path: Path):
    tree = ast.parse(path.read_text())
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
            names += [a.name for a in node.names]
    return names


def test_reward_module_has_no_drift_side_imports():
    names = [n.lower() for n in _imported_names(REWARD_SRC)]
    for token in FORBIDDEN_IMPORT_TOKENS:
        assert not any(token in n for n in names), (
            f"C1 breach: fidelity_reward.py imports something matching {token!r}: {names}"
        )


class _RecordingBackend:
    def __init__(self, val=1):
        self.val = val
        self.seen = []

    def score(self, patient_turn, context, cell):
        self.seen.append((patient_turn, context, cell))
        return self.val

    def check(self, patient_turn, context):
        self.seen.append((patient_turn, context))
        return self.val


def test_reward_reads_only_allowed_inputs():
    eng, dlv, rea = _RecordingBackend(), _RecordingBackend(), _RecordingBackend()
    backends = RewardBackends(engine=eng, delivery=dlv, realism=rea)

    r = fidelity_reward("patient turn text", "PROFILE PROMPT", "the context", "b1", backends)
    assert r == 1.0

    # engine/delivery saw exactly (turn, context, cell); realism saw (turn, context).
    assert eng.seen == [("patient turn text", "the context", "b1")]
    assert dlv.seen == [("patient turn text", "the context", "b1")]
    assert rea.seen == [("patient turn text", "the context")]
