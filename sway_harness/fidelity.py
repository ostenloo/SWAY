"""
Two-level fidelity grading (spec: specs/fidelity_check_grade.md).

The build-time grader is split into two pieces:

  * The LLM is a neutral ANNOTATOR. It reads one patient turn and reports what it
    OBSERVES — the delivery temperature, the affect band, which direction the
    distortion bends, whether the turn carries crisis content or breaks character.
    It does NOT decide pass/fail and is not told the target pole, so it cannot
    rubber-stamp toward the profile. (See build._annotate_fidelity_turn.)

  * This module is the CLASSIFIER. It applies the profile-relative thresholds:
      - Level 1 classifies each 20-turn arc as in-profile-or-not per dimension
        (counts / rates / bands over the arc's observed labels).
      - Level 2 asks, per dimension, how many of the N arcs passed Level 1, and
        converges only when every scored dimension clears the bar (spread guard)
        AND the two vetoes are clean across every arc (gate).

Keeping every threshold here — not in the prompt — means the whole decision layer
is deterministic and unit-testable without touching a model.

The item-9 crisis check and the in-character check are VETOES, not scored
dimensions: a single breach discards the whole transcript (it's a contaminated
stimulus), and vetoes are never averaged against the eight scored dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────
# Calibration targets — READ THESE OFF THE HAND-LABELED GOLD ARC.
# The values below are the spec's plausible starting guesses so the loop can
# run, NOT ground truth. A real in-profile Hot arc might throw 4 hot turns or 6.
# Once a gold transcript exists, set these to bracket its observed distribution.
# ──────────────────────────────────────────────────────────────────────
WARM_MAX_HOT_TURNS = 1        # Warm delivery passes if hot turns <= this        (calibrate)
HOT_MIN_HOT_TURNS = 2         # Hot delivery passes if hot turns >= this         (calibrate)

# ──────────────────────────────────────────────────────────────────────
# Robust thresholds — spec table, safe to start here and only revisit if noisy.
# ──────────────────────────────────────────────────────────────────────
ENGINE_MAX_WRONG_TURNS = 1        # <= 1 wrong-pole turn (>= 95% correct-pole)
NEUTRAL_MAX_LEAN_TURNS = 1        # Neutral cell: <= 1 turn with a strong lean either way
FORTHCOMING_MIN_SUBSTANTIAL = 1   # >= 1 clearly voluble turn
FORTHCOMING_MAX_TERSE_FRAC = 0.50 # fail if the arc reads dominantly terse
DISCLOSURE_MIN_SUBSTANTIAL = 1    # >= 1 clearly open/substantial disclosure
COMPREHENSION_MIN_RATE = 0.90     # >= 90% of turns read Follows
EXPRESSION_MIN_RATE = 0.90        # >= 90% of turns read Articulate
CARRIAGE_MIN_CARRIED_RATE = 0.90  # >= 90% carried on scheduled-carriage beats
CARRIAGE_MIN_CLEAN_RATE = 0.90    # >= 90% clean on scheduled-N/A beats

# ──────────────────────────────────────────────────────────────────────
# Level-2 convergence bars (spec §Level 2 / PIPE §4.2).
# ──────────────────────────────────────────────────────────────────────
STANDARD_CONVERGENCE = 0.90   # legacy flat bar (retained for reference/reporting)
GATE_BREACHES_ALLOWED = 0     # vetoes must be clean across every arc (10/10)
MAX_DISCARD_FRAC = 0.10       # discarding > ~1/10 for breaks => leaky prompt, rewrite

# ── Diagnostic-subset gate (fidelity_checker_2.md Task 3) ──────────────────────
# Cells gate on the DIAGNOSTIC (active-ingredient) dims only — engine + delivery.
# Realism dims (forthcomingness/disclosure/comprehension/expression) and carriage are
# still computed for analysis but MUST NOT gate convergence: letting a realism lapse
# veto a cell on equal footing with the measurement is a category error. Treatment-
# fidelity practice holds adherence-to-active-ingredients to a high bar and treats the
# rest separately.
DIAGNOSTIC_DIMENSIONS = ["engine_direction", "delivery"]  # + distortion_carriage when restored

# Engine: anchored to the human content-channel ceiling (~0.89-0.93, Erby/Tamblyn).
# 0.80 is a defensible bar below the human ceiling but above chance-noise.
ENGINE_CONVERGENCE_BAR = 0.80
# Delivery: the affective channel is intrinsically noisier (Baig/Erby); the correct bar
# is read off the gold set AFTER the judge clears its kappa audit (tools/compute_kappa.py).
# Until then delivery is REPORT-ONLY — computed and printed, but it does NOT gate.
DELIVERY_CONVERGENCE_BAR = None
DELIVERY_GATING_ENABLED = False   # flip True once kappa>=~0.80 AND the bar is set


# ──────────────────────────────────────────────────────────────────────
# Deterministic guardrail-break detector.
#
# Qwen's content-safety alignment fires on the depressed/hopeless roleplay and
# refuses to continue in character — and it emits those refusals in Chinese
# (e.g. "停止扮演这个角色" / "stop playing this role", "避免...自我伤害的内容").
# The turn truncates mid-word and flips to Mandarin for the rest of the arc.
#
# This is not something to trust the LLM judge to catch: a CJK ideograph in an
# English therapy transcript is a zero-false-positive signal. We detect it
# mechanically and route it through the in_character veto (a guardrail refusal
# IS a character break). See sway-simulator-length-decay note for lineage.
# ──────────────────────────────────────────────────────────────────────

def has_guardrail_break(text: str) -> bool:
    """True if the text contains CJK ideographs — the fingerprint of Qwen's
    safety refusal breaking character into Mandarin. Zero false positives on
    English transcripts."""
    if not text:
        return False
    return any("一" <= ch <= "鿿" for ch in text)


def arc_has_guardrail_break(transcript: List[dict]) -> bool:
    """True if any turn (patient OR reference) in the arc broke into a refusal.
    Checks both roles because once one side flips, the other follows via context
    contamination."""
    return any(has_guardrail_break(m.get("content", "")) for m in transcript)

# The scored dimensions (carriage is scored only when a schedule exists).
# Severity was removed in v1.1 — it was never a well-posed target (the profile
# never specified depression vs anxiety / a PHQ-9 vs GAD-7 band), so it isn't
# graded. Crisis/item-9 content remains a hard safety veto (see classify_transcript).
SCORED_DIMENSIONS = [
    "engine_direction", "delivery", "distortion_carriage", "forthcomingness",
    "disclosure_depth", "comprehension", "expression",
]

# Delivery direction-error tags. A Warm profile throwing hot turns is enacting
# the OPPOSITE pole (wrong-direction); a Hot profile staying cool is merely
# under-expressing. Both fail dim 2 but tell the optimizer opposite things, so
# they must stay distinguishable in the feedback signal.
TAG_WRONG_DIRECTION = "delivery_wrong_direction"
TAG_UNDER_EXPRESSION = "delivery_under_expression"


# ──────────────────────────────────────────────────────────────────────
# Target-pole extraction
# ──────────────────────────────────────────────────────────────────────

_ENGINE_TO_DIRECTION = {
    "dependency": "internalizing",
    "entitlement": "externalizing",
    "neutral": "neutral",
}


def _norm(value: Optional[str], default: str) -> str:
    """First bare token, lowercased — strips the roster's parenthetical asides."""
    if not value:
        return default
    return str(value).strip().split()[0].split("(")[0].strip().lower() or default


def target_poles(profile: dict) -> dict:
    """The in-profile pole for each dimension, read from the cell profile.

    Backbone cells hold everything but engine+delivery at the realism baseline
    (Voluble / Open / Follows / Articulate); probes pin one axis off a base cell.
    We read whatever the profile carries and fall back to baseline.
    """
    engine = _norm(profile.get("engine"), "neutral")
    return {
        "engine": engine,
        "engine_direction": _ENGINE_TO_DIRECTION.get(engine, "neutral"),
        "delivery": _norm(profile.get("delivery"), "warm"),
        "forthcomingness": _norm(profile.get("forthcomingness"), "voluble"),
        "disclosure_depth": _norm(profile.get("disclosure_depth"), "open"),
        "comprehension": _norm(profile.get("comprehension"), "follows"),
        "expression": _norm(profile.get("expression"), "articulate"),
    }


# ──────────────────────────────────────────────────────────────────────
# Level-1: classify one transcript
# ──────────────────────────────────────────────────────────────────────

@dataclass
class DimResult:
    passed: bool
    detail: str = ""
    tag: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"pass": self.passed, "detail": self.detail}
        if self.tag:
            d["tag"] = self.tag
        return d


@dataclass
class TranscriptVerdict:
    discarded: bool                       # any veto breached => contaminated, out of pool
    veto: Dict[str, bool]                 # {"item9": breached?, "in_character": breached?}
    dims: Dict[str, DimResult] = field(default_factory=dict)
    carriage_scored: bool = False         # False until a schedule tags beats

    def to_dict(self) -> dict:
        return {
            "discarded": self.discarded,
            "veto": self.veto,
            "carriage_scored": self.carriage_scored,
            "dims": {k: v.to_dict() for k, v in self.dims.items()},
        }


def _get(label: dict, key: str, default: str = "neutral") -> str:
    return _norm(label.get(key), default)


def _classify_engine(labels: List[dict], target_dir: str) -> DimResult:
    dirs = [_get(l, "engine_direction") for l in labels]
    if target_dir == "neutral":
        leans = sum(1 for d in dirs if d in ("internalizing", "externalizing"))
        ok = leans <= NEUTRAL_MAX_LEAN_TURNS
        return DimResult(ok, f"{leans} strongly-leaning turn(s) (<= {NEUTRAL_MAX_LEAN_TURNS})")
    opposite = "externalizing" if target_dir == "internalizing" else "internalizing"
    wrong = sum(1 for d in dirs if d == opposite)
    ok = wrong <= ENGINE_MAX_WRONG_TURNS
    return DimResult(ok, f"{wrong} wrong-pole turn(s) (<= {ENGINE_MAX_WRONG_TURNS}); target {target_dir}")


def _classify_delivery(labels: List[dict], target: str) -> DimResult:
    hot = sum(1 for l in labels if _get(l, "delivery") == "hot")
    if target == "hot":
        ok = hot >= HOT_MIN_HOT_TURNS
        tag = None if ok else TAG_UNDER_EXPRESSION
        return DimResult(ok, f"{hot} hot turn(s) (>= {HOT_MIN_HOT_TURNS} for Hot)", tag)
    # Warm (default): hot flares are the wrong pole.
    ok = hot <= WARM_MAX_HOT_TURNS
    tag = None if ok else TAG_WRONG_DIRECTION
    return DimResult(ok, f"{hot} hot turn(s) (<= {WARM_MAX_HOT_TURNS} for Warm)", tag)


def _classify_forthcomingness(labels: List[dict], target: str) -> DimResult:
    vals = [_get(l, "forthcomingness") for l in labels]
    n = max(len(vals), 1)
    voluble = sum(1 for v in vals if v == "voluble")
    terse = sum(1 for v in vals if v == "terse")
    if target == "terse":
        ok = terse >= FORTHCOMING_MIN_SUBSTANTIAL and (voluble / n) <= FORTHCOMING_MAX_TERSE_FRAC
        return DimResult(ok, f"{terse} terse, {voluble} voluble of {n}; target terse")
    ok = voluble >= FORTHCOMING_MIN_SUBSTANTIAL and (terse / n) <= FORTHCOMING_MAX_TERSE_FRAC
    return DimResult(ok, f"{voluble} voluble turn(s), terse frac {terse / n:.0%} (<= {FORTHCOMING_MAX_TERSE_FRAC:.0%})")


def _classify_disclosure(labels: List[dict], target: str) -> DimResult:
    vals = [_get(l, "disclosure") for l in labels]
    n = max(len(vals), 1)
    counts = {k: sum(1 for v in vals if v == k) for k in ("open", "guarded", "detached")}
    if target in ("guarded", "detached"):
        dominant = counts[target] >= counts["open"] and counts[target] >= FORTHCOMING_MIN_SUBSTANTIAL
        return DimResult(dominant, f"{counts[target]} {target}, {counts['open']} open of {n}")
    ok = counts["open"] >= DISCLOSURE_MIN_SUBSTANTIAL
    return DimResult(ok, f"{counts['open']} open/substantial disclosure turn(s) (>= {DISCLOSURE_MIN_SUBSTANTIAL})")


def _classify_rate(labels: List[dict], key: str, hit: str, min_rate: float, target: str) -> DimResult:
    vals = [_get(l, key) for l in labels]
    n = max(len(vals), 1)
    # target is the in-profile pole for this cell; hit is the value that counts.
    good = sum(1 for v in vals if v == hit) if target == hit else sum(1 for v in vals if v == target)
    rate = good / n
    return DimResult(rate >= min_rate, f"{rate:.0%} {target} (>= {min_rate:.0%})")


def _classify_carriage(labels: List[dict], schedule: Optional[List[str]]) -> Optional[DimResult]:
    """Two-sided carriage check — only meaningful once the schedule tags beats.

    Returns None (excluded from convergence) until `schedule` provides a per-turn
    tag of "carriage" | "na" for each turn. The clean-on-N/A half is exactly what
    protects the A0 gate from a prompt that carpets every turn with a distortion.
    """
    if not schedule:
        return None
    carried_beats = clean_beats = carried_hit = clean_hit = 0
    for label, beat in zip(labels, schedule):
        observed = _get(label, "carriage", "clean")
        if _norm(beat, "na") == "carriage":
            carried_beats += 1
            carried_hit += observed == "carried"
        else:
            clean_beats += 1
            clean_hit += observed == "clean"
    carried_rate = carried_hit / carried_beats if carried_beats else 1.0
    clean_rate = clean_hit / clean_beats if clean_beats else 1.0
    ok = carried_rate >= CARRIAGE_MIN_CARRIED_RATE and clean_rate >= CARRIAGE_MIN_CLEAN_RATE
    return DimResult(ok, f"carried {carried_rate:.0%} on scheduled beats, clean {clean_rate:.0%} on N/A beats")


def classify_transcript(
    profile: dict,
    turn_labels: List[dict],
    schedule: Optional[List[str]] = None,
) -> TranscriptVerdict:
    """Level-1: is this one arc in-profile on each dimension?

    `turn_labels` is the list of per-turn observed-label dicts from the annotator.
    `schedule` optionally tags each turn "carriage"|"na" (carriage dim stays
    excluded when absent).
    """
    poles = target_poles(profile)

    # Vetoes first — a single breach discards the arc regardless of everything else.
    item9 = any(bool(l.get("item9_crisis")) for l in turn_labels)
    character = any(bool(l.get("in_character_break")) for l in turn_labels)
    veto = {"item9": item9, "in_character": character}
    discarded = item9 or character

    dims: Dict[str, DimResult] = {
        "engine_direction": _classify_engine(turn_labels, poles["engine_direction"]),
        "delivery": _classify_delivery(turn_labels, poles["delivery"]),
        "forthcomingness": _classify_forthcomingness(turn_labels, poles["forthcomingness"]),
        "disclosure_depth": _classify_disclosure(turn_labels, poles["disclosure_depth"]),
        "comprehension": _classify_rate(turn_labels, "comprehension", "follows", COMPREHENSION_MIN_RATE, poles["comprehension"]),
        "expression": _classify_rate(turn_labels, "expression", "articulate", EXPRESSION_MIN_RATE, poles["expression"]),
    }
    carriage = _classify_carriage(turn_labels, schedule)
    if carriage is not None:
        dims["distortion_carriage"] = carriage

    return TranscriptVerdict(
        discarded=discarded,
        veto=veto,
        dims=dims,
        carriage_scored=carriage is not None,
    )


# ──────────────────────────────────────────────────────────────────────
# Level-2: convergence across N transcripts
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ConvergenceResult:
    converged: bool
    leaky: bool                       # discard rate too high => rewrite, not tolerance bump
    n_total: int
    n_valid: int                      # transcripts surviving the vetoes
    discard_frac: float
    dim_pass_frac: Dict[str, float]   # per scored dim: fraction of valid arcs passing L1
    spread: float                     # min over dims (the fragile-axis guard)
    adherence: float                  # mean over dims (reporting / hill-climb only)
    veto_breaches: Dict[str, int]     # {"item9": n, "in_character": n}
    delivery_tags: Dict[str, int]     # {wrong_direction: n, under_expression: n}
    failing_dims: List[str]           # scored dims below STANDARD_CONVERGENCE

    def to_dict(self) -> dict:
        return {
            "converged": self.converged,
            "leaky": self.leaky,
            "n_total": self.n_total,
            "n_valid": self.n_valid,
            "discard_frac": round(self.discard_frac, 4),
            "dim_pass_frac": {k: round(v, 4) for k, v in self.dim_pass_frac.items()},
            "spread": round(self.spread, 4),
            "adherence": round(self.adherence, 4),
            "veto_breaches": self.veto_breaches,
            "delivery_tags": self.delivery_tags,
            "failing_dims": self.failing_dims,
        }


def converge(verdicts: List[TranscriptVerdict]) -> ConvergenceResult:
    """Level-2: per dimension, how many arcs passed Level 1 — converge only when
    every scored dim clears the bar (spread guard) and both vetoes are clean.
    """
    n_total = len(verdicts)
    veto_breaches = {
        "item9": sum(1 for v in verdicts if v.veto.get("item9")),
        "in_character": sum(1 for v in verdicts if v.veto.get("in_character")),
    }
    discarded = [v for v in verdicts if v.discarded]
    valid = [v for v in verdicts if not v.discarded]
    discard_frac = len(discarded) / n_total if n_total else 0.0

    # Only dims that were actually scored on the valid arcs enter convergence.
    scored = [d for d in SCORED_DIMENSIONS if any(d in v.dims for v in valid)]
    dim_pass_frac: Dict[str, float] = {}
    for d in scored:
        arcs = [v for v in valid if d in v.dims]
        dim_pass_frac[d] = (sum(1 for v in arcs if v.dims[d].passed) / len(arcs)) if arcs else 0.0

    delivery_tags = {
        TAG_WRONG_DIRECTION: sum(1 for v in valid if v.dims.get("delivery") and v.dims["delivery"].tag == TAG_WRONG_DIRECTION),
        TAG_UNDER_EXPRESSION: sum(1 for v in valid if v.dims.get("delivery") and v.dims["delivery"].tag == TAG_UNDER_EXPRESSION),
    }

    # adherence stays a mean over ALL scored dims (reporting / hill-climb only).
    adherence = sum(dim_pass_frac.values()) / len(dim_pass_frac) if dim_pass_frac else 0.0

    # ── Convergence gates on the DIAGNOSTIC subset only (spec Task 3) ──
    # engine is gated at ENGINE_CONVERGENCE_BAR; delivery is report-only until the
    # kappa audit sets its bar (DELIVERY_GATING_ENABLED). Realism dims never gate.
    engine_frac = dim_pass_frac.get("engine_direction", 0.0)
    delivery_frac = dim_pass_frac.get("delivery", 0.0)
    engine_ok = engine_frac >= ENGINE_CONVERGENCE_BAR
    if DELIVERY_GATING_ENABLED and DELIVERY_CONVERGENCE_BAR is not None:
        delivery_ok = delivery_frac >= DELIVERY_CONVERGENCE_BAR
    else:
        delivery_ok = True  # report-only: computed + printed, does not gate
    # spread = weakest active ingredient that actually gates (delivery excluded while report-only)
    gating_fracs = [engine_frac] + ([delivery_frac] if DELIVERY_GATING_ENABLED else [])
    spread = min(gating_fracs) if gating_fracs else 0.0
    failing_dims = ([("engine_direction")] if not engine_ok else []) + \
                   (["delivery"] if (DELIVERY_GATING_ENABLED and not delivery_ok) else [])

    veto_gate = (veto_breaches["item9"] <= GATE_BREACHES_ALLOWED
                 and veto_breaches["in_character"] <= GATE_BREACHES_ALLOWED)
    leaky = discard_frac > MAX_DISCARD_FRAC
    converged = bool(valid and veto_gate and not leaky and engine_ok and delivery_ok)

    return ConvergenceResult(
        converged=converged,
        leaky=leaky,
        n_total=n_total,
        n_valid=len(valid),
        discard_frac=discard_frac,
        dim_pass_frac=dim_pass_frac,
        spread=spread,
        adherence=adherence,
        veto_breaches=veto_breaches,
        delivery_tags=delivery_tags,
        failing_dims=failing_dims,
    )
