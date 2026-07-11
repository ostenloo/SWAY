"""Offline unit tests for parser.get_profile population + probe inheritance.

Pure/offline — no LLM server or network needed. Verifies:
  - backbone cells (b1-b6) read engine/delivery from their explicit roster lines
  - probe cells (p1-p3) INHERIT their base cell AND apply their one pinned override

Regression guard for the bug (parser.py, commit efd64b9) where get_profile never
applied a probe's base_cell inheritance or override_* keys, so every probe silently
collapsed to engine=neutral + baseline disclosure/comprehension/expression and was
scored against the wrong poles.

Run:  python sway_harness/tests/test_profile_population.py     (exit 0 = all pass)
  or:  pytest sway_harness/tests/test_profile_population.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import get_profile          # noqa: E402
from fidelity import target_poles       # noqa: E402


# Expected NORMALIZED target poles (what fidelity scoring actually reads via
# target_poles — parenthetical roster asides already stripped). Only the keys
# listed per cell are checked, so each row asserts exactly what matters for it.
EXPECTED = {
    # ── Backbone: engine/delivery come from explicit "engine: X | delivery: Y" lines.
    "b1": {"engine": "dependency",  "engine_direction": "internalizing", "delivery": "warm",
           "disclosure_depth": "open", "comprehension": "follows", "expression": "articulate"},
    "b2": {"engine": "dependency",  "engine_direction": "internalizing", "delivery": "hot"},
    "b3": {"engine": "entitlement", "engine_direction": "externalizing", "delivery": "warm"},
    "b4": {"engine": "entitlement", "engine_direction": "externalizing", "delivery": "hot"},
    "b5": {"engine": "neutral",     "engine_direction": "neutral",       "delivery": "warm",
           "disclosure_depth": "open", "comprehension": "follows", "expression": "articulate"},
    "b6": {"engine": "neutral",     "engine_direction": "neutral",       "delivery": "hot"},

    # ── Probes: engine/delivery INHERITED from base cell; exactly one axis pinned.
    # p1 = Guarded-distress off B1 (Dependency·Warm): disclosure Open -> GUARDED.
    "p1": {"engine": "dependency", "engine_direction": "internalizing", "delivery": "warm",
           "disclosure_depth": "guarded", "comprehension": "follows", "expression": "articulate"},
    # p2 = Loses-thread off B5 (Neutral·Warm): comprehension + expression pinned.
    "p2": {"engine": "neutral", "engine_direction": "neutral", "delivery": "warm",
           "disclosure_depth": "open", "comprehension": "loses-thread", "expression": "fragmented"},
    # p3 = Fluent-but-low-uptake off B5: comprehension pinned, expression HELD articulate.
    "p3": {"engine": "neutral", "engine_direction": "neutral", "delivery": "warm",
           "disclosure_depth": "open", "comprehension": "loses-thread", "expression": "articulate"},
}


def _mismatches(cell_id):
    poles = target_poles(get_profile(cell_id))
    return [f"{cell_id}.{k}: expected {v!r}, got {poles.get(k)!r}"
            for k, v in EXPECTED[cell_id].items() if poles.get(k) != v]


def test_backbone_populates_from_explicit_lines():
    """b1-b6 engine/delivery are read from the roster, not from the fallback default."""
    bad = [m for c in ("b1", "b2", "b3", "b4", "b5", "b6") for m in _mismatches(c)]
    assert not bad, "\n".join(bad)


def test_backbone_engine_is_raw_roster_value_not_default():
    """Proves engine came from parse_cell (capitalized roster token), not setdefault('neutral')."""
    assert get_profile("b1")["engine"] == "Dependency"
    assert get_profile("b3")["engine"] == "Entitlement"


def test_probes_inherit_base_and_apply_override():
    """p1-p3 inherit engine/delivery from their base and apply their one pinned axis."""
    bad = [m for c in ("p1", "p2", "p3") for m in _mismatches(c)]
    assert not bad, "\n".join(bad)


def test_probe_regression_not_silently_defaulted():
    """Anti-bug guards: the exact symptoms of the old collapse must not recur."""
    p1 = target_poles(get_profile("p1"))
    assert p1["engine"] == "dependency", "p1 must inherit Dependency from B1, not default to neutral"
    assert p1["disclosure_depth"] == "guarded", "p1's Open->GUARDED override must be applied"

    p2 = target_poles(get_profile("p2"))
    assert p2["comprehension"] == "loses-thread", "p2 comprehension override dropped (or hyphen truncated to 'loses')"
    assert p2["expression"] == "fragmented", "p2 expression override not applied"

    # p3 shares p2's comprehension deficit but HOLDS articulate expression — the override
    # must not bleed across axes.
    p3 = target_poles(get_profile("p3"))
    assert p3["comprehension"] == "loses-thread"
    assert p3["expression"] == "articulate", "p3 must keep Articulate (held), not inherit p2's FRAGMENTED"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}\n      " + str(e).replace("\n", "\n      "))
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
