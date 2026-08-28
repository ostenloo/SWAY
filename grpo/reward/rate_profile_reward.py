"""Arc-level RATE-PROFILE reward (SWAY_ENGINE_DELIVERY_RATE_PROFILE_SPEC).

REPLACES `band_reward.band_reward_arc` as the reward-bearing shape. `band_reward`
is retained unwired, as the record of the measurements in [RATE §1] and §7 — it
must not be selected by config again.

**What changed and why.** The conditional band scored `q`, the on-pole share
*among the marked turns*, so it was estimated from `m` observations instead of
`T`. Every failure in [RATE §1] follows from that: the caricature sat adjacent to
the band (17x better than mild under-expression), expressing the trait on FEWER
turns scored better, and at `m = 5` the ratio's resolution (0.20) was coarser than
the band being placed (0.057). This module scores the **rates over all `T`
turns** instead:

    engine     p_int  = #internalizing / T     p_ext = #externalizing / T
    delivery   p_warm = #warm          / T     p_hot = #hot           / T

Same information, estimated from `T` rather than `m`. The extreme now sits at
`p = 1.0` with the band near 0.20, so the anti-caricature property comes from the
band's POSITION rather than from a specially tightened upper shoulder.

Per axis the two component scores are combined with `min` (§4.4): they are two
halves of one claim about the same behaviour, and an arc that nails its
on-direction rate while producing three times the realistic off-direction rate
has not portrayed the profile. Across axes, the epsilon-floored geometric mean is
unchanged from the band spec and is imported from `band_reward.combine_axes`.

NON-NEGOTIABLE CONSTRAINTS realised here ([RATE §3]):

  * **C1** — reads `(arc_turns, context0, profile, cell, cal)` and nothing else.
    Enforced by the grep guard over `FORBIDDEN_IMPORT_TOKENS`.
  * **C3** — the graders producing the calibration targets MUST be the frozen
    backend objects injected as the reward graders (`assert_calibration_backends`).
  * **C4** — graders are blind to the target pole. `profile` is accepted and
    deliberately never forwarded.
  * **C6/C9** — the artifact is frozen and hash-logged; `load_calibration`
    returns the sha256.
  * **C7 — one estimator, or none.** Rates are RAW COUNTS OVER `T` on both the
    calibration side and here. No smoothing anywhere. `rate_of` is the single
    implementation and `grpo.calibration.rate_derive` calls THIS function, so
    the two paths cannot drift (A6).

DECISIONS TAKEN HERE that the spec left open, each argued at its definition:

  * `MIN_ARC_TURNS` (§12 "gated at rollout time, or short arcs simply score
    badly") — gated at 10, derived from §5.2 rather than picked.
  * The denominator is the arc's ACTUAL graded turn count, never the nominal `T`.
  * §5.2's minimum span is enforced on off-direction bands too, which §7 applies
    to one band and not the three others like it.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, Sequence

# C1: strings, not drift-side code. Same re-export the band module does, so the
# grep guard has one token list covering every reward module.
from grpo.reward.fidelity_reward import (  # noqa: F401
    FORBIDDEN_IMPORT_TOKENS,
    assert_family_disjoint,
    FamilyDisjointnessError,
)
from grpo.reward.band_reward import (
    AXIS_FLOOR_EPS,
    DEFAULT_AXIS_COMBINATION,
    MARKED,
    UNMARKED,
    combine_axes,
    context_upto,
)


ON_DIRECTION = "on_direction"
OFF_DIRECTION = "off_direction"
ROLES = (ON_DIRECTION, OFF_DIRECTION)

#: [RATE §5.1] — symmetric, and NOT configurable per band. The band spec's
#: asymmetric default (`s_hi = 2 * s_lo`) is the direct cause of §1.2's inversion.
#: Anti-caricature is carried by the on-direction band's UPPER EDGE on a scale
#: where the caricature is far away (§5.3); it does not need, and must not have,
#: a specially shaped shoulder.
SHOULDER = 0.06

#: §5.2 — a band must admit at least three attainable rates at full score. At
#: `T = 20` the attainable rates are multiples of 0.05, so the span floor is
#: `2/T`. Rejected at LOAD time (A2), not discovered in training.
MIN_SPAN_TURNS = 2

#: §5.3 — the off-direction band is `[0, U]` and zero scores 1.0, because among
#: real leaning conversations the 25th percentile of the off-direction rate is
#: exactly 0.000. A nonzero floor would be less realistic than the corpus.
OFF_DIRECTION_L = 0.0

#: Minimum arc length at scoring time. [RATE §12] leaves this open ("gated at
#: rollout time, or short arcs simply score badly"); it is CLOSED HERE, and the
#: value is derived rather than chosen.
#:
#: Rates move in steps of `1/T`. §5.2 requires every band to span at least 0.10
#: so that three attainable values score full marks. Below `T = 10` the step
#: `1/T` EXCEEDS 0.10 — one attainable rate per band, or none — which is exactly
#: the §1.5 resolution failure this redesign exists to remove, reappearing on the
#: rollout side. So the gate is the same inequality as §5.2, read the other way:
#: a band wide enough to be scorable requires an arc long enough to score it.
#:
#: It also matches the calibration side's eligibility rule (§6 step 2, `T >= 10`),
#: which C7's spirit wants: the two sides admit the same arcs.
#:
#: A shorter arc scores 0.0, the same as the empty arc `trl_adapter` already
#: scores 0.0 — a short arc is a GENERATION failure, not a portrayal choice, and
#: the monitor sees it as one. It is not scored on a deflated denominator, which
#: is the silent alternative (see `rate_of`).
MIN_ARC_TURNS = 10


class CalibrationError(ValueError):
    """A rate-calibration artifact that must not be trained against.

    Raised at LOAD time, never at call time (A2). A bad band discovered 40
    minutes into a run is the failure mode [RATE §1] is a list of.
    """


# ── the estimator: ONE implementation, both sides (C7 / A6) ──────────────────

def rate_of(labels: Sequence[str], label: str) -> float:
    """`count(label) / len(labels)`. Raw. The whole of C7.

    The denominator is the number of turns actually LABELLED, never a nominal
    `T` from config. Dividing 16 graded turns by a configured 20 deflates every
    rate by 20% and silently walks the arc off the bottom of its band; nothing in
    the reward or the telemetry would show it. `arc_length_T` is what the BANDS
    are validated against (§5.2, §11), which is a property of the artifact — not
    a denominator to impose on an arc that came out a different length.

    `grpo.calibration.rate_derive` imports this function rather than
    reimplementing it, so the calibration path and the scoring path are the same
    code and A6 is true by construction rather than by test.
    """
    n = len(labels)
    if not n:
        return 0.0
    return sum(1 for x in labels if x == label) / n


def band(p: float, L: float, U: float, s: float = SHOULDER) -> float:
    """Plateau on `[L, U]` = 1.0, symmetric Gaussian shoulders outside (§5)."""
    if p < L:
        return math.exp(-((L - p) ** 2) / (2.0 * s * s))
    if p > U:
        return math.exp(-((p - U) ** 2) / (2.0 * s * s))
    return 1.0


# ── the frozen artifact ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class RateComponent:
    """A target band on ONE marked rate.

    `role` is load-bearing, not documentation. It selects which invariant applies
    (§5.3): an `off_direction` component must have `L == 0`, and the
    `on_direction` component's `U` is the entire anti-caricature mechanism (A3).
    """

    label: str          # the marked label whose rate this band constrains
    role: str           # ON_DIRECTION | OFF_DIRECTION
    L: float
    U: float

    @property
    def span(self) -> float:
        return self.U - self.L


@dataclass(frozen=True)
class AxisRateProfile:
    """One axis's two rate bands for one cell.

    Exactly two components, one per marked label of the axis. A NEUTRAL engine
    profile is not a special case (§2.2): it is simply two `off_direction`
    components with low ceilings. `density_low` is deleted.
    """

    axis: str
    components: tuple          # (RateComponent, RateComponent)
    s: float = SHOULDER
    #: §8 — false for every DECLARED band. Monitoring must report engine and
    #: delivery band-fit separately so a declared target is never read as a
    #: measured one.
    measured: bool = True
    provenance: dict = field(default_factory=dict)

    def component(self, label: str) -> RateComponent:
        for c in self.components:
            if c.label == label:
                return c
        raise KeyError(f"{self.axis}: no rate band for label {label!r}")

    @property
    def on_direction(self) -> Optional[RateComponent]:
        """The component carrying the profile's asserted pole, or None.

        None for a neutral profile, which asserts the ABSENCE of both poles.
        A3/A4 are stated in terms of this and are vacuous where it is None —
        deliberately: a neutral profile has no caricature to guard against on
        that axis. What guards b5/b6 against going inert is the DELIVERY axis's
        on-direction lower edge, reached through the geometric mean. See
        `arc_readout`.
        """
        for c in self.components:
            if c.role == ON_DIRECTION:
                return c
        return None


@dataclass(frozen=True)
class RateCalibration:
    """The frozen `rate_calibration.<grader_version>.yaml`, parsed and validated.

    `arc_length_T` is a REQUIRED top-level field (§11): the bands are validated
    against it (§5.2), so an artifact that does not record the `T` it was
    validated at cannot be checked at all.
    """

    cells: dict
    arc_length_T: int
    grader_version: str = "unknown"
    sha256: str = ""
    source_path: str = ""
    #: C3 — identities of the backends that PRODUCED this artifact.
    backend_identities: dict = field(default_factory=dict)
    derivation: dict = field(default_factory=dict)

    def __getitem__(self, cell: str) -> dict:
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
        raise KeyError(f"no calibration entry for cell {cell!r} (have: {sorted(self.cells)})")

    def __contains__(self, cell: str) -> bool:
        try:
            self[cell]
        except KeyError:
            return False
        return True


# ── load-time validation (A2) ────────────────────────────────────────────────

def _component_from_dict(where: str, axis: str, raw: dict, T: int) -> RateComponent:
    if not isinstance(raw, dict):
        raise CalibrationError(f"{where}: expected a mapping, got {type(raw).__name__}")

    label = str(raw.get("label", "")).strip().lower()
    if label not in MARKED[axis]:
        raise CalibrationError(
            f"{where}: label {raw.get('label')!r} is not a MARKED label on {axis} "
            f"(marked = {MARKED[axis]}, unmarked = {UNMARKED[axis]!r}). The unmarked "
            "rate is 1 - p_a - p_b and is constrained implicitly (§2); it is never "
            "given a band of its own."
        )

    role = str(raw.get("role", "")).strip().lower()
    if role not in ROLES:
        raise CalibrationError(f"{where}: role {raw.get('role')!r} not in {ROLES}")

    def num(key: str) -> float:
        if key not in raw or raw[key] is None:
            raise CalibrationError(f"{where}: missing required edge {key!r}")
        try:
            return float(raw[key])
        except (TypeError, ValueError):
            raise CalibrationError(f"{where}: {key!r} is not a number: {raw[key]!r}") from None

    L, U = num("L"), num("U")

    if L < 0:
        raise CalibrationError(f"{where}: L must be >= 0, got {L}")
    if not L < U:
        raise CalibrationError(f"{where}: require L < U, got L={L} >= U={U}")

    # A2 — U >= 1.0. On the rate scale `p = 1.0` IS the caricature: every turn
    # on-direction. A band reaching it puts the caricature on the plateau and
    # voids the mechanism (§5.3, A3).
    if not U < 1.0:
        raise CalibrationError(
            f"{where}: U must be < 1.0, got {U}. At U >= 1.0 the caricature arc "
            "(every turn marked on-direction) sits on the plateau, and the "
            "on-direction upper edge is the ONLY anti-caricature mechanism (§5.3)."
        )

    # A2 — §5.3. Applies to off-direction bands on EVERY profile, including the
    # two components of a neutral profile, both of which are off-direction.
    if role == OFF_DIRECTION and L > OFF_DIRECTION_L:
        raise CalibrationError(
            f"{where}: an off_direction band must have L = 0, got {L}. Measured: among "
            "conversations that lean one way, the 25th percentile of the off-direction "
            "rate is 0.000 — a quarter of real leaning patients never express the other "
            "direction at all, so a nonzero floor is LESS realistic than the corpus (§5.3)."
        )

    # A2 — §5.2, enforced UNIFORMLY.
    #
    # §7 flags the widening on `EXTERNALIZING p_int in [0, 0.071]` and not on the
    # three other sub-0.10 bands it prints (`INTERNALIZING p_ext in [0, 0.089]`,
    # and both NEUTRAL bands). Reading §5.2 as on-direction-only would need an
    # explicit exemption that the spec does not give, and the failure it guards
    # against — a band narrower than the gap between two attainable answers, so
    # that hitting it depends on whether a division happens to land (§1.5) — is
    # exactly as real on an off-direction band. So: uniform, and the widening is
    # applied and recorded in `rate_derive.widen_to_min_span`.
    min_span = MIN_SPAN_TURNS / T
    if U - L < min_span - 1e-9:
        raise CalibrationError(
            f"{where}: band [{L}, {U}] spans {U - L:.4f}, below the {MIN_SPAN_TURNS}/T "
            f"= {min_span:.4f} floor at T = {T} (§5.2). Fewer than three attainable "
            "rates would score above 0.9, so hitting the band depends on where a "
            "division lands rather than on behaviour — the §1.5 failure. Widen it "
            "at derivation time and record the widening."
        )

    return RateComponent(label=label, role=role, L=L, U=U)


def _axis_from_dict(cell: str, axis: str, raw: dict, T: int) -> AxisRateProfile:
    where = f"{cell}.{axis}"
    if not isinstance(raw, dict):
        raise CalibrationError(f"{where}: expected a mapping, got {type(raw).__name__}")

    s_lo = raw.get("s_lo", raw.get("s", SHOULDER))
    s_hi = raw.get("s_hi", raw.get("s", SHOULDER))
    try:
        s_lo, s_hi = float(s_lo), float(s_hi)
    except (TypeError, ValueError):
        raise CalibrationError(f"{where}: shoulder widths are not numbers") from None
    if not s_lo > 0:
        raise CalibrationError(f"{where}: shoulder must be > 0 (band() divides by it), got {s_lo}")
    # A2 — asymmetric shoulders are rejected (§5.1). The band spec's `s_hi =
    # 2 * s_lo` default is the direct cause of the §1.2 inversion; there is no
    # longer any reason for asymmetry, so the artifact may not express one.
    if abs(s_lo - s_hi) > 1e-12:
        raise CalibrationError(
            f"{where}: asymmetric shoulders (s_lo={s_lo}, s_hi={s_hi}) are rejected (§5.1). "
            "Anti-caricature is the on-direction band's POSITION on a scale where the "
            "caricature is far away, not a tightened upper shoulder."
        )

    raw_components = raw.get("components")
    if not isinstance(raw_components, list) or len(raw_components) != 2:
        raise CalibrationError(
            f"{where}: expected exactly 2 `components:` (one per marked label "
            f"{MARKED[axis]}), got {raw_components!r}"
        )

    comps = tuple(
        _component_from_dict(f"{where}[{i}]", axis, c, T)
        for i, c in enumerate(raw_components)
    )
    labels = {c.label for c in comps}
    if labels != set(MARKED[axis]):
        raise CalibrationError(
            f"{where}: components cover {sorted(labels)}, expected exactly "
            f"{sorted(MARKED[axis])} — both marked rates are scored, and `min` over "
            "them is the axis score (§4.4)."
        )
    n_on = sum(1 for c in comps if c.role == ON_DIRECTION)
    if n_on > 1:
        raise CalibrationError(
            f"{where}: {n_on} on_direction components. A profile asserts at most one "
            "pole per axis; a neutral profile asserts none."
        )

    return AxisRateProfile(
        axis=axis, components=comps, s=s_lo,
        measured=bool(raw.get("measured", True)),
        provenance=dict(raw.get("provenance") or {}),
    )


def calibration_from_dict(doc: dict, *, sha256: str = "", source_path: str = "") -> RateCalibration:
    """Validate a parsed artifact document into a `RateCalibration` (A2)."""
    if not isinstance(doc, dict) or "cells" not in doc:
        raise CalibrationError("calibration artifact has no top-level `cells:` mapping")

    # §11 — `T` MUST be recorded. Every span check below is relative to it, so an
    # artifact without it cannot be validated, only assumed.
    if "arc_length_T" not in doc or doc["arc_length_T"] is None:
        raise CalibrationError(
            "calibration artifact records no `arc_length_T:` (§11). The bands are "
            "validated against T (§5.2); changing T without re-validating is a "
            "load-time error, and that check is impossible without the value."
        )
    try:
        T = int(doc["arc_length_T"])
    except (TypeError, ValueError):
        raise CalibrationError(f"arc_length_T is not an integer: {doc['arc_length_T']!r}") from None
    if T < MIN_ARC_TURNS:
        raise CalibrationError(
            f"arc_length_T = {T} is below MIN_ARC_TURNS = {MIN_ARC_TURNS}: at T < "
            f"{MIN_ARC_TURNS} the rate step 1/T exceeds the {MIN_SPAN_TURNS}/T span "
            "floor no band can satisfy meaningfully (§5.2)."
        )

    cells: dict = {}
    for cell_key, axes in (doc["cells"] or {}).items():
        if not isinstance(axes, dict):
            raise CalibrationError(f"{cell_key}: expected a mapping of axes")
        missing = {"engine", "delivery"} - set(axes)
        if missing:
            raise CalibrationError(
                f"{cell_key}: missing axis entries {sorted(missing)} — the reward takes "
                "the geometric mean of both axes (§4.5), so a half-specified cell is "
                "not trainable."
            )
        cells[cell_key] = {
            axis: _axis_from_dict(cell_key, axis, axes[axis], T)
            for axis in ("engine", "delivery")
        }

    return RateCalibration(
        cells=cells,
        arc_length_T=T,
        grader_version=str(doc.get("grader_version", "unknown")),
        sha256=sha256,
        source_path=source_path,
        backend_identities=dict(doc.get("backend_identities") or {}),
        derivation=dict(doc.get("derivation") or {}),
    )


def load_calibration(path: str | Path) -> RateCalibration:
    """Load, hash (C6), and validate the frozen artifact.

    The sha256 is over the file BYTES, so it is the value to log and the value
    that changes if anyone edits an edge mid-run.
    """
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to load the rate calibration. `pip install pyyaml` "
            "(only needed to launch a run — the reward math imports stdlib only)."
        ) from e

    p = Path(path)
    raw = p.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    doc = yaml.safe_load(raw.decode("utf-8"))
    return calibration_from_dict(doc, sha256=digest, source_path=str(p))


def assert_calibration_backends(cal: RateCalibration, backends) -> None:
    """C3 — the graders scoring rollouts must BE the graders that measured the
    targets.

    This is what makes an imperfect grader acceptable: a systematic bias inflates
    the target and the score alike and largely cancels. It requires consistency,
    not accuracy — and consistency is exactly what a grader swap breaks.
    """
    if not cal.backend_identities:
        raise CalibrationError(
            "calibration artifact records no `backend_identities:` — cannot verify C3 "
            "(same frozen backends for calibration and reward)."
        )
    live = {
        "engine": getattr(getattr(backends, "engine", None), "identity", None),
        "delivery": getattr(getattr(backends, "delivery", None), "identity", None),
    }
    for axis, want in cal.backend_identities.items():
        got = live.get(axis)
        if got != want:
            raise CalibrationError(
                f"C3 BREACH on {axis}: the targets were measured with {want!r} but the "
                f"reward is scoring with {got!r}. Bias cancellation does not hold across "
                "a grader swap — recalibrate or restore the original champion."
            )


def assert_arc_length(cal: RateCalibration, arc_length_T: int) -> None:
    """§11 — the config's `T` must be the `T` the bands were validated at.

    Every band in the artifact cleared the `2/T` span floor (§5.2) against the
    `T` recorded in the artifact. Roll 30-turn arcs against bands validated at
    20 and the spans are no longer the ones that were checked; roll 12-turn arcs
    and the rate step (0.083) approaches the span floor (0.10), so a band that
    admitted three attainable values now admits two. Neither shows up anywhere —
    the reward keeps returning plausible numbers — which is why this is a
    load-time refusal and not a warning.
    """
    if int(arc_length_T) != int(cal.arc_length_T):
        raise CalibrationError(
            f"arc_length_T mismatch: the config rolls T = {arc_length_T} but "
            f"{cal.source_path or 'the calibration'} was validated at T = "
            f"{cal.arc_length_T} (§5.2, §11). Re-derive the artifact at the new T "
            "or restore the old one — the span floor was checked against the "
            "artifact's value, not the config's."
        )


# ── scoring ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ComponentReadout:
    label: str
    role: str
    p: float
    L: float
    U: float
    value: float
    on_plateau: bool


@dataclass(frozen=True)
class RateAxisReadout:
    """One axis of one arc, with the intermediates the §9 monitor logs.

    Field names `d`, `on_plateau` and `at_lower_edge` are kept so the existing
    `OnlineMonitor` (which reads them by `getattr`) keeps working unchanged; `q`
    is None because there is no conditional share any more, and the monitor
    already treats a None `q` as absent.
    """

    axis: str
    reward: float
    n_turns: int
    rates: dict                      # label -> rate over ALL turns
    components: tuple                # (ComponentReadout, ...)
    measured: bool = True
    q: Optional[float] = None        # gone: superseded by `rates`
    #: Marked fraction, = sum of the two component rates. The old `d`, retained
    #: for telemetry continuity — it is no longer an input to the reward.
    d: float = 0.0
    on_plateau: bool = False
    at_lower_edge: bool = False
    #: The component whose band value the `min` selected — the one the gradient
    #: is actually pushing on.
    binding: str = ""

    @property
    def p_on(self) -> Optional[float]:
        for c in self.components:
            if c.role == ON_DIRECTION:
                return c.p
        return None

    @property
    def p_off(self) -> Optional[float]:
        for c in self.components:
            if c.role == OFF_DIRECTION:
                return c.p
        return None


def axis_readout(axis: str, labels: Sequence[str], p: AxisRateProfile) -> RateAxisReadout:
    """The per-axis rate-profile score plus its intermediates (§4).

    Steps 2-4 of §4: count each marked label, divide by the arc's full turn count
    INCLUDING unmarked turns (never by the marked count — that is the root cause
    of §1.2 through §1.5), score each rate against its band, and take the `min`.
    """
    n_turns = len(labels)
    rates = {lab: rate_of(labels, lab) for lab in MARKED[axis]}

    comps = []
    for c in p.components:
        val = band(rates[c.label], c.L, c.U, p.s)
        comps.append(ComponentReadout(
            label=c.label, role=c.role, p=rates[c.label], L=c.L, U=c.U,
            value=val, on_plateau=(c.L <= rates[c.label] <= c.U),
        ))

    # §4.4 — `min`, not the average. The two rates are two halves of one claim
    # about the same behaviour; an arc that nails its on-direction rate while
    # producing three times the realistic off-direction rate has not portrayed
    # the profile, and averaging would let it.
    worst = min(comps, key=lambda c: c.value)

    on = next((c for c in comps if c.role == ON_DIRECTION), None)
    return RateAxisReadout(
        axis=axis,
        reward=worst.value,
        n_turns=n_turns,
        rates=rates,
        components=tuple(comps),
        measured=p.measured,
        d=sum(rates.values()),
        on_plateau=all(c.on_plateau for c in comps),
        # §9 watch: sitting at the bottom edge of the on-direction band is the
        # cheapest point on the plateau, and a policy parked there is expressing
        # the profile as little as the band permits.
        at_lower_edge=bool(on is not None and on.on_plateau and on.p <= on.L + 1e-9),
        binding=worst.label,
    )


def axis_reward(axis: str, labels: Sequence[str], cell: str, cal: RateCalibration) -> float:
    return axis_readout(axis, labels, cal[cell][axis]).reward


class LabelBackend(Protocol):
    """A frozen categorical grader, blind to the target pole (C4)."""

    identity: str

    def label(self, patient_turn: str, context: str, cell: str) -> str: ...


@dataclass(frozen=True)
class RateArcReadout:
    reward: float
    engine: RateAxisReadout
    delivery: RateAxisReadout
    engine_labels: tuple
    delivery_labels: tuple
    combination: str = DEFAULT_AXIS_COMBINATION
    weaker_axis: str = ""
    #: Non-empty when the arc was refused rather than scored (see MIN_ARC_TURNS).
    refusal: str = ""


def rate_profile_reward_arc_readout(
    arc_turns: Sequence[str],
    context0: str,
    profile: str,
    cell: str,
    cal: RateCalibration,
    backends,
    combination: str = DEFAULT_AXIS_COMBINATION,
    eps: float = AXIS_FLOOR_EPS,
) -> RateArcReadout:
    """`rate_profile_reward_arc` with the intermediates kept, for §9 telemetry."""
    if not arc_turns:
        raise ValueError("rate_profile_reward_arc: empty arc")

    eng_labels, del_labels = [], []
    for i, turn in enumerate(arc_turns):
        ctx = context_upto(arc_turns, i, context0)
        eng_labels.append(backends.engine.label(turn, ctx, cell))
        del_labels.append(backends.delivery.label(turn, ctx, cell))

    eng = axis_readout("engine", eng_labels, cal[cell]["engine"])
    dlv = axis_readout("delivery", del_labels, cal[cell]["delivery"])

    if len(arc_turns) < MIN_ARC_TURNS:
        # Scored 0.0, not scored on a short denominator. See MIN_ARC_TURNS: below
        # T = 10 the rate step exceeds the minimum band span, so the number the
        # band would be compared against carries less resolution than the band.
        # The readouts are still returned so the monitor can see WHY.
        return RateArcReadout(
            reward=0.0, engine=eng, delivery=dlv,
            engine_labels=tuple(eng_labels), delivery_labels=tuple(del_labels),
            combination=combination, weaker_axis="",
            refusal=(f"arc has {len(arc_turns)} turns, below MIN_ARC_TURNS = "
                     f"{MIN_ARC_TURNS}; rates at this length have less resolution "
                     "than the bands (§5.2, §1.5)"),
        )

    # §4.5 — epsilon-floored geometric mean across axes, unchanged from the band
    # spec including the argument for it.
    #
    # This is also what closes the "going limp" hole on the NEUTRAL cells (b5,
    # b6). Their engine profile is two off-direction bands `[0, U]`, so an arc
    # expressing no engine at all scores engine = 1.0 — which is correct, since
    # neutral means absent engine and §5.3 forbids a lower edge. What stops an
    # inert arc from being rewarded is the DELIVERY axis: every delivery profile
    # carries an on-direction band with a real lower edge, so a patient
    # expressing nothing scores `p_warm = 0` (or `p_hot = 0`) and the geometric
    # mean drags the arc down. The guard moved axes; it did not disappear. Pinned
    # by `test_neutral_cell_inert_arc_is_not_rewarded`.
    return RateArcReadout(
        reward=combine_axes(eng.reward, dlv.reward, combination, eps),
        engine=eng, delivery=dlv,
        engine_labels=tuple(eng_labels), delivery_labels=tuple(del_labels),
        combination=combination,
        weaker_axis=("engine" if eng.reward <= dlv.reward else "delivery"),
    )


def rate_profile_reward_arc(
    arc_turns: Sequence[str],
    context0: str,
    profile: str,
    cell: str,
    cal: RateCalibration,
    backends,
    combination: str = DEFAULT_AXIS_COMBINATION,
    eps: float = AXIS_FLOOR_EPS,
) -> float:
    """Scalar in [0, 1] for one arc. Reads ONLY these inputs (C1, A1).

    arc_turns = the ordered PATIENT turns of one rollout (T = 20, §11).
    context0  = the initial state / history prefix the arc was rolled from.
    profile   = the frozen profile prompt — accepted, never forwarded to the
                graders, which are blind to the target pole (C4).
    cell      = cell id; selects the bands.
    cal       = the frozen artifact (C6/C9).
    """
    return rate_profile_reward_arc_readout(
        arc_turns, context0, profile, cell, cal, backends, combination, eps).reward
