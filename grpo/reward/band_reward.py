"""Arc-level distributional band reward (grpo_spec_2 §4, [BAND §3-§5]).

This module REPLACES `fidelity_reward.fidelity_reward` as the reward-bearing
shape. The per-turn `0.5*engine_pass + 0.5*delivery_pass` binary survives only
inside the RFT warm-start filter (§6) and the §9 audit; it is no longer the
gradient signal.

**Why a band.** The monotone binary's optimum is the caricature — on-profile on
every marked turn — which is both unrealistic and the R3 attractor. A plateau
with a ceiling `U < 1.0` gives the objective an *interior* optimum, so the
anti-caricature property comes free of any separate penalty term ([BAND §4]).

The reward has two per-axis factors:

  * **`q`-band** — the on-profile SHARE among marked turns, Dirichlet-smoothed,
    scored against a plateau `[L_design, U]` with Gaussian shoulders. `q` says
    "when the patient does attribute, how often on-pole?".
  * **density factor** — a soft ramp on the marked FRACTION `d`, which closes
    the inert-simulator hole: `q` alone is farmable by being marked on two turns
    and getting the ratio right ([BAND §5]).

and one special case:

  * **`density_low`** (D2.3) — for `profile.engine == neutral` (b5, b6, p2, p3),
    the target is ABSENT engine, so engine reward is a two-sided band directly on
    `d_eng` toward a low anchor. There is no `q` behind it, which is why
    `d_lo > 0` is a hard load-time assert: at `d_lo = 0` an arc expressing no
    engine at all would score 1.0 for going limp.

NON-NEGOTIABLE CONSTRAINTS realised here (grpo_spec_2 §2):

  * **C1** — the reward reads `(arc_turns, context0, P, cell, cal)` and NOTHING
    else. No MUT reply, no SYC/DEP score, no drift signal. Enforced by the grep
    guard in `grpo/tests/test_c1_import_guard.py` against
    `FORBIDDEN_IMPORT_TOKENS`, which this module re-exports for the test.
  * **C3** — reward shape is the arc-level band. **No realism multiplier**: [FT1
    C3]'s floor and [BAND §3]'s arc-mean variant are both void (§4.3). R3 is
    carried by KL and the §9 audit alone. Recorded, not hidden.
  * **C4/CB1** — the SAME frozen backend objects must supply both the
    calibration and the reward. That is what makes instrument cancellation
    work ([BAND §7]); `assert_calibration_backends` is the check.
  * **C9** — the calibration artifact is frozen and hash-logged before step 1.
    `load_calibration` returns the sha256 in the object so the run log can
    record it and a mid-run edit is detectable.

`P` is accepted and deliberately NOT forwarded to the backends: the graders are
blind to the target pole (that is what stops them rubber-stamping toward the
profile), and the pole is applied downstream from `cell`. Keeping `P` in the
signature documents that the reward is a function of (profile, arc) and nothing
else, per the wall.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence

# C1: re-exported so the grep guard has one token list covering both reward
# modules. Importing it is not a drift-side import — it is a tuple of strings.
from grpo.reward.fidelity_reward import (  # noqa: F401
    FORBIDDEN_IMPORT_TOKENS,
    assert_family_disjoint,
    FamilyDisjointnessError,
)


# ── label vocabularies ───────────────────────────────────────────────────────
# The UNMARKED class per axis is the one that sets the density denominator:
# `neutral` on engine, `flat` on delivery. Everything else is marked.

MARKED = {
    "engine": ("internalizing", "externalizing"),
    "delivery": ("warm", "hot"),
}

UNMARKED = {"engine": "neutral", "delivery": "flat"}

#: Artifact shorthand -> the label the backends actually emit. The [BAND §6.4]
#: example writes `c_star: int`; `engine_decompose.engine_direction` emits
#: `internalizing`. Normalise once, at load, so the hot path compares equals.
_C_STAR_ALIASES = {
    "int": "internalizing",
    "internal": "internalizing",
    "internalizing": "internalizing",
    "ext": "externalizing",
    "external": "externalizing",
    "externalizing": "externalizing",
    "warm": "warm",
    "hot": "hot",
    "flat": "flat",
    "neutral": "neutral",
}

Q_BAND = "q_band"
DENSITY_LOW = "density_low"


# ── axis combination (D-BAND.1, RESOLVED toward multiplicative) ──────────────
#
# [BAND D-BAND.1] left this open with `average` as the default, on the rationale
# that the average "preserves gradient when one axis is out-of-band
# (anti-collapse, cf. [FT R2])". **That rationale does not survive measurement,
# and the default is changed here to the geometric mean.** Three reasons:
#
# 1. **The average is indifferent to allocation, and worse than indifferent.**
#    (r_eng, r_del) = (0.95, 0.10) and (0.10, 0.95) both score 0.525, and so does
#    a balanced (0.52, 0.52) arc at 0.520. An arc that half-nails one axis and is
#    dead on the other is worth MORE than an arc doing both passably, so the
#    average actively prefers specialisation. That is the wrong preference for a
#    profile that asserts both an engine and a delivery.
#
# 2. **The geometric mean reallocates gradient to the failing axis, for free.**
#    d/da sqrt(a*b) = 0.5*sqrt(b/a), so the weak axis gets the larger derivative:
#    ~9.5x toward delivery at (0.95, 0.10), ~6x at (0.90, 0.15), and exactly 1x
#    when the axes are balanced. The average is flat 1:1 always.
#
# 3. **D-BAND.1's anti-collapse argument is backwards.** It conflates reward
#    LEVEL with gradient; GRPO uses within-group VARIANCE, and a near-constant
#    healthy axis dilutes it. Measured on a G=8 group with engine pinned at 0.90
#    and delivery scattered 0.02-0.22, mean |advantage| is 0.0263 under the
#    average versus 0.0786 under the geometric mean — the average causes the
#    collapse it was chosen to prevent. This is b2's signature (0.93 collapse at
#    0.494 mean reward, close to the 0.525 an average pays for one-axis-only).
#
# Geometric mean rather than the raw product because geo AGREES with the average
# on the diagonal — (0.52, 0.52) -> 0.520 under both, (1, 1) -> 1.0 under both —
# and diverges only when lopsided. The raw product squashes the whole range
# toward zero (balanced mediocre -> 0.27). Geo is the minimal change that fixes
# the defect, and is a monotone rescaling of the product, so the two induce the
# SAME ordering over arcs and differ only in gradient magnitude.

AVERAGE = "average"
GEOMETRIC_MEAN = "geometric_mean"
PRODUCT = "product"

AXIS_COMBINATIONS = (AVERAGE, GEOMETRIC_MEAN, PRODUCT)

#: Default. Resolves [BAND §12]'s open D-BAND.1 toward the multiplicative branch.
DEFAULT_AXIS_COMBINATION = GEOMETRIC_MEAN

#: The R2 guard that the multiplicative branch needs and the average does not.
#: Any product-like rule scores 0 whenever ONE axis is 0, and then every arc in
#: the group scores 0 — zero variance, no gradient, a dead group. An axis reward
#: is exactly 0 only when `density_factor = 0`, i.e. zero marked turns on that
#: axis (`band()` is Gaussian and never exactly 0; `density_low` never is). So
#: the dead case is "fully inert on one axis", and a small floor restores the
#: gradient out of it: at eps = 0.02 an inert-delivery arc with healthy engine
#: scores sqrt(0.90 * 0.02) = 0.134, and marking even 1 turn in 20 lifts it to
#: ~0.36. Large enough to escape, far too small to be worth farming.
AXIS_FLOOR_EPS = 0.02


def combine_axes(
    r_eng: float,
    r_del: float,
    mode: str = DEFAULT_AXIS_COMBINATION,
    eps: float = AXIS_FLOOR_EPS,
) -> float:
    """Combine the two per-axis band rewards into the arc scalar (D-BAND.1)."""
    if mode == AVERAGE:
        return 0.5 * r_eng + 0.5 * r_del        # no eps: the average cannot die
    a, b = max(r_eng, eps), max(r_del, eps)
    if mode == GEOMETRIC_MEAN:
        return math.sqrt(a * b)
    if mode == PRODUCT:
        return a * b
    raise ValueError(f"unknown axis combination {mode!r} (expected one of {AXIS_COMBINATIONS})")


class CalibrationError(ValueError):
    """A band-calibration artifact that must not be trained against.

    Raised at LOAD time, never at call time. [BAND]'s formulas divide by
    `d_floor`, `s_lo`, `s_hi` and `(1 - d_ceil)`; a zero there is a
    ZeroDivisionError 40 minutes into a run instead of a refusal at step 0.
    """


# ── the frozen artifact ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class AxisBand:
    """One axis's band parameters for one cell ([BAND §6.4]).

    `mode == "q_band"`      -> `c_star, L_design, U, alpha, d_floor, d_ceil` live.
    `mode == "density_low"` -> `d_lo, d_hi` live; there is no `q` term (D2.3).

    `L_ext` is NOT a field. §6.5 collapsed it into `L_design`: under the
    per-patient percentile derivation `L_ext` already carries a disclosed choice
    (which percentile), so a second disclosed knob above it does the same job
    twice. Two percentiles `(P_lo, P_hi)` replace three hand-set numbers.

    **`alpha` is part of the definition of `q`, not a free post-hoc knob.**
    `q = (n_on + alpha) / (n_marked + 2*alpha)` pulls toward 0.5 as alpha rises,
    so `L_design`/`U` are only meaningful against the alpha they were DERIVED
    under — the calibration pass must compute its per-session `q` with this same
    value. Retuning alpha after the fact silently re-scales the axis: the arcs
    slide off the plateau onto the shoulder, and A6's smoothing property inverts
    (spread can grow with alpha). If alpha changes, re-run the calibration; do
    not hand-edit it in the artifact. Pinned by
    `test_a6_unmatched_alpha_breaks_the_property`.
    """

    mode: str
    s_lo: float
    s_hi: float

    # q_band
    c_star: Optional[str] = None
    L_design: Optional[float] = None
    U: Optional[float] = None
    alpha: float = 0.5
    d_floor: Optional[float] = None
    d_ceil: Optional[float] = None      # D-BAND.2, two-sided delivery option

    # density_low (D2.3)
    d_lo: Optional[float] = None
    d_hi: Optional[float] = None

    #: D2.4 — stamped false when `U - L_design > 0.35` or fewer than ~25 sessions
    #: were eligible. Warn and disclose, do not halt: nothing blocks, but the run
    #: log cannot later claim the value was data-derived when it could not be.
    bracket_informative: bool = True

    #: Free-form provenance from the calibration pass (percentiles used, eligible
    #: session count, d_anno). Carried so the freeze manifest can reproduce it.
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BandCalibration:
    """The frozen `band_calibration.<grader_version>.yaml`, parsed and validated.

    `sha256` is the hash of the artifact bytes as loaded — this is the C9 value
    that goes in the run log and the §10 version record. It is computed over the
    file, not over this object, so a mid-run edit changes it.
    """

    cells: dict
    grader_version: str = "unknown"
    sha256: str = ""
    source_path: str = ""

    #: C4/CB1 — identities of the backends that PRODUCED this artifact. The
    #: reward asserts these match the backends injected as graders.
    backend_identities: dict = field(default_factory=dict)

    def __getitem__(self, cell: str) -> dict:
        """`cal[cell]` -> {"engine": AxisBand, "delivery": AxisBand}.

        Accepts both the bare cell id (`b1`, what the config and code use) and
        the descriptive artifact key ([BAND §6.4] writes `b1_internalizing_warm`).
        """
        if cell in self.cells:
            return self.cells[cell]
        matches = [k for k in self.cells if k.split("_", 1)[0] == cell]
        if len(matches) == 1:
            return self.cells[matches[0]]
        if len(matches) > 1:
            raise KeyError(
                f"calibration has {len(matches)} entries matching cell {cell!r}: "
                f"{matches}. Cell keys must be unambiguous."
            )
        raise KeyError(
            f"no calibration entry for cell {cell!r} (have: {sorted(self.cells)})"
        )

    def __contains__(self, cell: str) -> bool:
        try:
            self[cell]
        except KeyError:
            return False
        return True


# ── load-time validation (A2) ────────────────────────────────────────────────

def _axis_from_dict(cell: str, axis: str, raw: dict) -> AxisBand:
    """Parse and hard-validate one axis entry. Every failure is a refusal (A2)."""
    where = f"{cell}.{axis}"
    if not isinstance(raw, dict):
        raise CalibrationError(f"{where}: expected a mapping, got {type(raw).__name__}")

    mode = str(raw.get("mode", Q_BAND)).strip()
    if mode not in (Q_BAND, DENSITY_LOW):
        raise CalibrationError(f"{where}: unknown mode {mode!r} (expected {Q_BAND}|{DENSITY_LOW})")
    if mode == DENSITY_LOW and axis != "engine":
        raise CalibrationError(
            f"{where}: mode {DENSITY_LOW} is the neutral-ENGINE special case (D2.3); "
            "it is not defined for delivery."
        )

    def num(key: str, required: bool = True) -> Optional[float]:
        if key not in raw or raw[key] is None:
            if required:
                raise CalibrationError(f"{where}: missing required parameter {key!r}")
            return None
        try:
            return float(raw[key])
        except (TypeError, ValueError):
            raise CalibrationError(f"{where}: {key!r} is not a number: {raw[key]!r}") from None

    # Shared: the Gaussian shoulder widths divide inside `band()`.
    s_lo, s_hi = num("s_lo"), num("s_hi")
    if not s_lo > 0:
        raise CalibrationError(f"{where}: s_lo must be > 0 (band() divides by it), got {s_lo}")
    if not s_hi > 0:
        raise CalibrationError(f"{where}: s_hi must be > 0 (band() divides by it), got {s_hi}")

    if mode == DENSITY_LOW:
        d_lo, d_hi = num("d_lo"), num("d_hi")
        # D2.3 — STRICT. [BAND §5.3] permits d_lo: 0.0, and at exactly zero an arc
        # expressing no engine at all scores 1.0: full marks on half of b5's and
        # b6's reward for going limp. That reopens the inert-simulator hole the
        # density floor exists to close, on precisely the cells that have no
        # q-band behind it.
        if not d_lo > 0:
            raise CalibrationError(
                f"{where}: d_lo must be > 0 strictly (D2.3), got {d_lo}. At d_lo=0 an arc "
                "with NO engine expression scores 1.0 — the inert-simulator exploit, on a "
                "cell with no q-band standing behind it. Use d_lo ~= 0.05 with a tight s_lo."
            )
        if not d_lo < d_hi:
            raise CalibrationError(f"{where}: require d_lo < d_hi, got {d_lo} >= {d_hi}")
        if not d_hi < 1.0:
            raise CalibrationError(f"{where}: require d_hi < 1.0, got {d_hi}")
        return AxisBand(
            mode=mode, s_lo=s_lo, s_hi=s_hi, d_lo=d_lo, d_hi=d_hi,
            bracket_informative=bool(raw.get("bracket_informative", True)),
            provenance=dict(raw.get("provenance") or {}),
        )

    # q_band
    c_star_raw = str(raw.get("c_star", "")).strip().lower()
    c_star = _C_STAR_ALIASES.get(c_star_raw)
    if c_star is None:
        raise CalibrationError(
            f"{where}: c_star {raw.get('c_star')!r} is not a known label "
            f"(expected one of {sorted(set(_C_STAR_ALIASES))})"
        )
    if c_star not in MARKED[axis]:
        raise CalibrationError(
            f"{where}: c_star {c_star!r} is the UNMARKED class on {axis} — q is the "
            f"share among MARKED turns {MARKED[axis]}, so an unmarked target is "
            f"unreachable by construction. For an absent-{axis} target use mode "
            f"{DENSITY_LOW} (engine only, D2.3)."
        )

    L, U = num("L_design"), num("U")
    if not L > 0:                       # A2
        raise CalibrationError(f"{where}: L_design must be > 0, got {L}")
    if not L < U:                       # CB3
        raise CalibrationError(f"{where}: require L_design < U, got {L} >= {U}")
    if not U < 1.0:                     # CB4 — the interiority ceiling IS the
        raise CalibrationError(        # anti-caricature mechanism (A3).
            f"{where}: U must be < 1.0 (CB4), got {U}. At U >= 1.0 the caricature "
            "arc (q = 1.0) sits on the plateau and the band's entire purpose is void."
        )

    alpha = num("alpha", required=False)
    alpha = 0.5 if alpha is None else alpha
    if not alpha > 0:
        raise CalibrationError(
            f"{where}: alpha must be > 0 (it defines q at n_marked = 0 and damps "
            f"quantisation jitter, CB6), got {alpha}"
        )

    d_floor = num("d_floor")
    if not d_floor > 0:
        raise CalibrationError(
            f"{where}: d_floor must be > 0 (density_factor divides by it), got {d_floor}"
        )
    if not d_floor <= 1.0:
        raise CalibrationError(f"{where}: d_floor must be <= 1.0, got {d_floor}")

    d_ceil = num("d_ceil", required=False)
    if d_ceil is not None:
        if not d_floor <= d_ceil:
            raise CalibrationError(f"{where}: require d_floor <= d_ceil, got {d_floor} > {d_ceil}")
        if not d_ceil < 1.0:
            raise CalibrationError(
                f"{where}: d_ceil must be < 1.0 (density_factor divides by 1 - d_ceil), got {d_ceil}"
            )

    band_obj = AxisBand(
        mode=mode, s_lo=s_lo, s_hi=s_hi, c_star=c_star, L_design=L, U=U,
        alpha=alpha, d_floor=d_floor, d_ceil=d_ceil,
        bracket_informative=bool(raw.get("bracket_informative", True)),
        provenance=dict(raw.get("provenance") or {}),
    )
    return band_obj


def calibration_from_dict(
    doc: dict,
    *,
    sha256: str = "",
    source_path: str = "",
) -> BandCalibration:
    """Validate a parsed artifact document into a `BandCalibration` (A2)."""
    if not isinstance(doc, dict) or "cells" not in doc:
        raise CalibrationError("calibration artifact has no top-level `cells:` mapping")

    cells: dict = {}
    for cell_key, axes in (doc["cells"] or {}).items():
        if not isinstance(axes, dict):
            raise CalibrationError(f"{cell_key}: expected a mapping of axes")
        missing = {"engine", "delivery"} - set(axes)
        if missing:
            raise CalibrationError(
                f"{cell_key}: missing axis entries {sorted(missing)} — the reward "
                "averages both axes (D-BAND.1), so a half-specified cell is not trainable."
            )
        cells[cell_key] = {
            axis: _axis_from_dict(cell_key, axis, axes[axis])
            for axis in ("engine", "delivery")
        }

    return BandCalibration(
        cells=cells,
        grader_version=str(doc.get("grader_version", "unknown")),
        sha256=sha256,
        source_path=source_path,
        backend_identities=dict(doc.get("backend_identities") or {}),
    )


def load_calibration(path: str | Path) -> BandCalibration:
    """Load, hash (C9), and validate the frozen artifact.

    The sha256 is over the file BYTES, so it is the value to log and the value
    that changes if anyone edits a band parameter mid-run ([BAND CB7]).
    """
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to load the band calibration. `pip install pyyaml` "
            "(only needed to launch a run — the reward math imports stdlib only)."
        ) from e

    p = Path(path)
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    doc = yaml.safe_load(raw.decode("utf-8"))
    return calibration_from_dict(doc, sha256=digest, source_path=str(p))


def assert_calibration_backends(cal: BandCalibration, backends) -> None:
    """C4/CB1 — the graders scoring rollouts must BE the graders that measured
    the bracket.

    [BAND §7]'s cancellation argument is the only thing licensing an imperfect
    grader: a systematic bias inflates the AnnoMI-measured target and the rollout
    measurement by roughly the same amount and subtracts out. Score with a
    different instrument than you calibrated with and the residual is a wrong
    setpoint, silently. This is a correctness requirement in v2, not hygiene.
    """
    if not cal.backend_identities:
        raise CalibrationError(
            "calibration artifact records no `backend_identities:` — cannot verify "
            "CB1 (same frozen backends for calibration and reward). Re-run the "
            "calibration pass with a version of annomi_calibrate that stamps them."
        )
    live = {
        "engine": getattr(getattr(backends, "engine", None), "identity", None),
        "delivery": getattr(getattr(backends, "delivery", None), "identity", None),
    }
    for axis, want in cal.backend_identities.items():
        got = live.get(axis)
        if got != want:
            raise CalibrationError(
                f"CB1 BREACH on {axis}: the calibration bracket was measured with "
                f"{want!r} but the reward is scoring with {got!r}. Instrument "
                "cancellation ([BAND §7]) does not hold across a grader swap — "
                "recalibrate or restore the original champion."
            )


# ── the band math ([BAND §4-§5]) ─────────────────────────────────────────────

def band(x: float, L: float, U: float, s_lo: float, s_hi: float) -> float:
    """Plateau on `[L, U]` = 1.0, Gaussian shoulders outside.

    Asymmetric by design: `s_lo` tight (dropping below the profile's on-profile
    floor is a clear ordering defect), `s_hi` sets how hard caricature is
    punished. The falloff above `U` IS the anti-caricature term — there is no
    separate penalty ([BAND §4]).
    """
    if x < L:
        return math.exp(-((L - x) ** 2) / (2 * s_lo ** 2))
    if x > U:
        return math.exp(-((x - U) ** 2) / (2 * s_hi ** 2))
    return 1.0


def density_factor(d: float, d_floor: float, d_ceil: Optional[float] = None) -> float:
    """Soft ramp on the marked fraction — closes the inert-simulator hole.

    One-sided by default (D-BAND.2): ramp 0 -> 1 as `d -> d_floor`, flat above.
    No cliff. Pass `d_ceil` for the two-sided delivery option, where a Hot
    profile is meant to fix a marking RATE rather than a floor (a hot patient is
    not hot every turn).
    """
    lo = min(1.0, d / d_floor)
    if d_ceil is None:
        return lo
    hi = min(1.0, (1.0 - d) / (1.0 - d_ceil)) if d > d_ceil else 1.0
    return lo * hi


@dataclass(frozen=True)
class AxisReadout:
    """Everything §9's rate telemetry needs from one axis of one arc.

    `axis_reward` returns the scalar; this is the same computation with its
    intermediates kept, so the monitor never has to recompute (or, worse,
    re-grade) to log `q`/`d`.
    """

    axis: str
    mode: str
    reward: float
    d: float                 # marked fraction over the arc
    n_marked: int
    n_turns: int
    q: Optional[float] = None        # None in density_low mode
    n_on: Optional[int] = None
    density_factor: Optional[float] = None
    band_value: Optional[float] = None
    #: §9 band-edge farming watch: reached the plateau on minimal marking.
    at_lower_edge: bool = False
    on_plateau: bool = False


def axis_readout(axis: str, labels: Sequence[str], p: AxisBand) -> AxisReadout:
    """The per-axis band score plus its intermediates ([BAND §4])."""
    n_turns = len(labels)
    marked = [x for x in labels if x in MARKED[axis]]
    n_marked = len(marked)
    d = n_marked / max(n_turns, 1)

    # --- neutral-engine special case (D2.3 / [BAND §1.2, §5.3]) ---
    if p.mode == DENSITY_LOW:
        r = band(d, p.d_lo, p.d_hi, p.s_lo, p.s_hi)
        return AxisReadout(
            axis=axis, mode=p.mode, reward=r, d=d, n_marked=n_marked, n_turns=n_turns,
            band_value=r, on_plateau=(p.d_lo <= d <= p.d_hi),
            at_lower_edge=(d <= p.d_lo),
        )

    d_fac = density_factor(d, p.d_floor, p.d_ceil)

    if n_marked == 0:
        # Caught by d_fac anyway (d = 0 -> lo = 0); explicit for clarity, and so
        # the readout does not report a q derived from an empty conditional.
        return AxisReadout(
            axis=axis, mode=p.mode, reward=0.0, d=d, n_marked=0, n_turns=n_turns,
            n_on=0, density_factor=d_fac, band_value=0.0,
        )

    n_on = sum(1 for x in marked if x == p.c_star)
    # Dirichlet smoothing (CB6): defines q gracefully at low n_marked and damps
    # the 1/n_marked quantisation jitter so the reward tracks behaviour rather
    # than sampling artifacts. Monitored in §9.
    q = (n_on + p.alpha) / (n_marked + 2 * p.alpha)

    b = band(q, p.L_design, p.U, p.s_lo, p.s_hi)
    return AxisReadout(
        axis=axis, mode=p.mode, reward=d_fac * b, d=d, n_marked=n_marked,
        n_turns=n_turns, q=q, n_on=n_on, density_factor=d_fac, band_value=b,
        on_plateau=(p.L_design <= q <= p.U),
        # §9: `d` sitting just above the floor with `q` at the bottom of the
        # plateau is band-edge farming — the failure with no automatic guard
        # now that the realism floor is gone (§4.3).
        at_lower_edge=(p.L_design <= q <= p.U and d <= p.d_floor * 1.10),
    )


def axis_reward(axis: str, labels: Sequence[str], cell: str, cal: BandCalibration) -> float:
    """Per-axis band reward in [0, 1] ([BAND §4])."""
    return axis_readout(axis, labels, cal[cell][axis]).reward


# ── the arc reward (§4.1) ────────────────────────────────────────────────────

class LabelBackend(Protocol):
    """A frozen categorical grader. `.label()` REPLACES the old `.score()`."""

    identity: str

    def label(self, patient_turn: str, context: str, cell: str) -> str: ...


def context_upto(arc_turns: Sequence[str], index: int, context0: str) -> str:
    """The history a grader sees for `arc_turns[index]`.

    Both constructs are context-dependent — delivery is defined as directed
    "toward the listener", which needs the preceding exchange to judge — so the
    grader is handed the prefix, never the bare turn. The interlocutor's replies
    are NOT available here (the reward reads only the arc's patient turns, C1),
    so the prefix is the initial context plus this arc's earlier patient turns.
    """
    parts = [context0] if context0 else []
    parts += list(arc_turns[:index])
    return "\n".join(parts)


@dataclass(frozen=True)
class ArcReadout:
    """One arc's full reward decomposition — what §9 logs per rollout."""

    reward: float
    engine: AxisReadout
    delivery: AxisReadout
    engine_labels: tuple
    delivery_labels: tuple
    combination: str = DEFAULT_AXIS_COMBINATION
    #: §9: which axis the geometric mean is currently pushing on. Under the
    #: average this is meaningless (always 1:1), which is the point.
    weaker_axis: str = ""


def band_reward_arc_readout(
    arc_turns: Sequence[str],
    context0: str,
    P: str,
    cell: str,
    cal: BandCalibration,
    backends,
    combination: str = DEFAULT_AXIS_COMBINATION,
    eps: float = AXIS_FLOOR_EPS,
) -> ArcReadout:
    """`band_reward_arc` with the intermediates kept, for §9 telemetry."""
    if not arc_turns:
        raise ValueError("band_reward_arc: empty arc")

    eng_labels, del_labels = [], []
    for i, turn in enumerate(arc_turns):
        ctx = context_upto(arc_turns, i, context0)
        eng_labels.append(backends.engine.label(turn, ctx, cell))
        del_labels.append(backends.delivery.label(turn, ctx, cell))

    eng = axis_readout("engine", eng_labels, cal[cell]["engine"])
    dlv = axis_readout("delivery", del_labels, cal[cell]["delivery"])

    # D-BAND.1 — geometric mean by default (see `combine_axes`). NO realism
    # multiplier (§4.3): [FT1 C3]'s per-turn floor and [BAND §3]'s arc-mean
    # variant are both removed by researcher decision, so the only product here
    # is between the two axes. R3 is on KL + the §9 audit alone.
    return ArcReadout(
        reward=combine_axes(eng.reward, dlv.reward, combination, eps),
        engine=eng,
        delivery=dlv,
        engine_labels=tuple(eng_labels),
        delivery_labels=tuple(del_labels),
        combination=combination,
        weaker_axis=("engine" if eng.reward <= dlv.reward else "delivery"),
    )


def band_reward_arc(
    arc_turns: Sequence[str],
    context0: str,
    P: str,
    cell: str,
    cal: BandCalibration,
    backends,
    combination: str = DEFAULT_AXIS_COMBINATION,
    eps: float = AXIS_FLOOR_EPS,
) -> float:
    """Scalar in [0, 1] for one arc. Reads ONLY these inputs (C1).

    arc_turns = the ordered PATIENT turns of one rollout (T = 20; §5.1).
    context0  = the initial state / history prefix the arc was rolled from.
    P         = frozen profile prompt for the cell — accepted, never forwarded
                to the graders (they are blind to the pole by design).
    cell      = cell id; selects the band parameters and the target pole.
    cal       = the frozen [BAND §6.4] artifact (C9).
    """
    return band_reward_arc_readout(
        arc_turns, context0, P, cell, cal, backends, combination, eps).reward
