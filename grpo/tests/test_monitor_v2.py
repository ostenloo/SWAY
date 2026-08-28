"""§9 online-monitoring signals under the v2 band reward."""

import json

import pytest

import grpo._bootstrap  # noqa: F401
from grpo.monitor.online_audit import (
    ArcRecord, GroupRecord, MissingKeyHalt, OnlineMonitor, standardize,
)
from grpo.reward.band_reward import AxisBand, AxisReadout, BandCalibration


def readout(axis, q=None, d=0.4, on_plateau=True, at_lower_edge=False, reward=1.0):
    return AxisReadout(axis=axis, mode="q_band", reward=reward, d=d, n_marked=8,
                       n_turns=20, q=q, on_plateau=on_plateau,
                       at_lower_edge=at_lower_edge)


def arc(reward, q_eng=0.8, q_del=0.7, d_eng=0.4, d_del=0.5, sub=None,
        farmed=False, completion="turn", on_plateau=True):
    return ArcRecord(
        reward=reward,
        engine=readout("engine", q=q_eng, d=d_eng, at_lower_edge=farmed,
                       on_plateau=on_plateau),
        delivery=readout("delivery", q=q_del, d=d_del, at_lower_edge=farmed,
                         on_plateau=on_plateau),
        completion=completion, sub=sub or {},
    )


def cal_with(L=0.70, U=0.90):
    band = AxisBand(mode="q_band", c_star="internalizing", L_design=L, U=U,
                    s_lo=0.06, s_hi=0.12, alpha=0.5, d_floor=0.30)
    dband = AxisBand(mode="q_band", c_star="warm", L_design=0.62, U=0.88,
                     s_lo=0.06, s_hi=0.12, alpha=0.5, d_floor=0.15)
    return BandCalibration(cells={"b1": {"engine": band, "delivery": dband}})


# ── D2.5: mean-centring, not std-normalisation ───────────────────────────────

def test_standardize_mean_centres_by_default():
    adv, sigma = standardize([0.2, 0.4, 0.6])
    assert adv == pytest.approx([-0.2, 0.0, 0.2])
    assert sigma > 0


def test_standardize_can_still_scale_when_asked():
    adv, _ = standardize([0.2, 0.4, 0.6], scale=True)
    assert max(adv) > 1.0        # divided by std -> much larger magnitudes


def test_std_normalisation_is_what_amplifies_a_coin_flip():
    """D2.5's argument, made concrete: a group that is one quantisation step
    apart becomes a multi-sigma update once divided by its own tiny std."""
    rewards = [1.0] * 7 + [0.95]
    centred, _ = standardize(rewards)
    scaled, _ = standardize(rewards, scale=True)
    assert max(abs(a) for a in centred) < 0.10
    assert max(abs(a) for a in scaled) > 2.0


# ── group collapse: the two-number rule ──────────────────────────────────────

def _mon(**kw):
    return OnlineMonitor(audit_every_n_steps=0, **kw)


def test_collapse_at_high_mean_reads_as_healthy_plateau():
    m = _mon()
    for _ in range(4):
        m.record_group(GroupRecord(step=1, cell="b1", arcs=[arc(1.0) for _ in range(8)]))
    r = m.collapse_reading("b1")
    assert r["collapse_rate"] == 1.0
    assert "plateau" in r["reading"] and "STUCK" not in r["reading"]


def test_collapse_at_middling_mean_reads_as_stuck():
    """b2's exact signature from §0.2: 0.93 collapse at 0.494 mean."""
    m = _mon()
    for _ in range(4):
        m.record_group(GroupRecord(step=1, cell="b2", arcs=[arc(0.494) for _ in range(8)]))
    r = m.collapse_reading("b2")
    assert r["collapse_rate"] == 1.0
    assert "STUCK" in r["reading"]
    assert r["baseline_v1"] == 0.93          # A8 comparison is carried along
    assert r["delta_vs_baseline"] == pytest.approx(0.07)


def test_uncollapsed_groups_read_healthy_regardless_of_mean():
    m = _mon()
    m.record_group(GroupRecord(step=1, cell="b1",
                               arcs=[arc(r) for r in (0.1, 0.3, 0.5, 0.7)]))
    assert "gradient" in m.collapse_reading("b1")["reading"]


# ── rate telemetry ───────────────────────────────────────────────────────────

def test_rate_telemetry_reports_q_and_d_distributions():
    m = _mon()
    m.record_group(GroupRecord(step=1, cell="b1", arcs=[
        arc(1.0, q_eng=0.75, d_eng=0.35), arc(1.0, q_eng=0.85, d_eng=0.45)]))
    t = m.rate_telemetry("b1")
    assert t["n"] == 2
    assert t["q_engine"]["mean"] == pytest.approx(0.80)
    assert t["d_engine"]["mean"] == pytest.approx(0.40)
    assert t["on_plateau_engine"] == 1.0


def test_settling_inside_the_band_is_not_flagged():
    """The band gives no within-plateau gradient BY DESIGN — a q that stops
    moving mid-band is the objective being met, not a stall."""
    m = _mon(cal=cal_with())
    m.record_group(GroupRecord(step=1, cell="b1",
                               arcs=[arc(1.0, q_eng=0.80) for _ in range(8)]))
    assert "flags" not in m.rate_telemetry("b1")


def test_q_parked_below_L_design_is_flagged():
    m = _mon(cal=cal_with(L=0.70))
    m.record_group(GroupRecord(step=1, cell="b1", arcs=[
        arc(0.5, q_eng=0.40, on_plateau=False) for _ in range(8)]))
    flags = m.rate_telemetry("b1")["flags"]
    assert any("BELOW L_design" in f for f in flags)


def test_q_riding_the_ceiling_is_flagged():
    m = _mon(cal=cal_with(U=0.90))
    m.record_group(GroupRecord(step=1, cell="b1",
                               arcs=[arc(0.9, q_eng=0.95) for _ in range(8)]))
    flags = m.rate_telemetry("b1")["flags"]
    assert any("caricature pressure" in f for f in flags)


# ── band-edge farming: the failure with no automatic guard ───────────────────

def test_band_edge_farming_rate_surfaces_minimal_marking():
    m = _mon()
    m.record_group(GroupRecord(step=1, cell="b1", arcs=(
        [arc(1.0, farmed=True) for _ in range(6)] + [arc(1.0) for _ in range(2)])))
    r = m.band_edge_farming_rate("b1")
    assert r["engine"] == pytest.approx(0.75)
    assert r["delivery"] == pytest.approx(0.75)


# ── cancellation drift (§8.2 -> §9) ──────────────────────────────────────────

def test_cancellation_drift_flags_a_widening_gap():
    m = _mon(annomi_grievance_density=0.20, cancellation_drift_tolerance=0.15)
    m.record_group(GroupRecord(step=1, cell="b4",
                               arcs=[arc(0.8, sub={"e2": 0.60}) for _ in range(8)]))
    d = m.cancellation_drift()
    assert d["rollout_grievance_density"] == pytest.approx(0.60)
    assert d["gap"] == pytest.approx(0.40)
    assert d["anchor_unstable"] is True


def test_cancellation_drift_is_quiet_when_the_anchor_holds():
    m = _mon(annomi_grievance_density=0.20)
    m.record_group(GroupRecord(step=1, cell="b4",
                               arcs=[arc(0.8, sub={"e2": 0.24}) for _ in range(8)]))
    assert m.cancellation_drift()["anchor_unstable"] is False


# ── missing-key halt (§4.2) ──────────────────────────────────────────────────

class _FakeCore:
    def __init__(self, n, missing):
        self.n_annotations, self.n_missing_all_keys = n, missing
        self._cache = {"x": 1}

    @property
    def missing_key_rate(self):
        return self.n_missing_all_keys / self.n_annotations if self.n_annotations else 0.0

    def reset_counters(self):
        self.n_annotations = self.n_missing_all_keys = 0

    def clear_cache(self):
        self._cache.clear()


class _FakeAdapter:
    def __init__(self, core):
        self.core = core
        self.identity = "fake"


class _FakeBackends:
    def __init__(self, e, d):
        self.engine, self.delivery = _FakeAdapter(e), _FakeAdapter(d)


def test_missing_key_halt_fires_above_the_threshold():
    b = _FakeBackends(_FakeCore(100, 9), _FakeCore(100, 0))
    m = _mon(backends=b, missing_key_halt_rate=0.05)
    with pytest.raises(MissingKeyHalt, match="9.0%"):
        m.check_missing_key_halt()


def test_missing_key_halt_uses_the_worst_axis_not_the_average():
    """A bad axis must not be averaged away by a healthy one."""
    b = _FakeBackends(_FakeCore(100, 8), _FakeCore(100, 0))
    assert m_rate(b) == pytest.approx(0.08)
    with pytest.raises(MissingKeyHalt):
        _mon(backends=b, missing_key_halt_rate=0.05).check_missing_key_halt()


def m_rate(b):
    from grpo.reward.backends import missing_key_rates
    return missing_key_rates(b)["max_rate"]


def test_missing_key_halt_quiet_below_threshold():
    b = _FakeBackends(_FakeCore(1000, 4), _FakeCore(1000, 1))
    _mon(backends=b, missing_key_halt_rate=0.05).check_missing_key_halt()


def test_no_backends_means_no_halt_and_no_crash():
    _mon().check_missing_key_halt()


# ── audit hygiene (§0.2) ─────────────────────────────────────────────────────

def test_audit_clears_the_backend_cache_first():
    """Auditing cached labels re-reads a stored verdict rather than re-grading,
    which is exactly what the audit is supposed to check."""
    e, d = _FakeCore(10, 0), _FakeCore(10, 0)
    b = _FakeBackends(e, d)
    m = OnlineMonitor(audit_every_n_steps=1, backends=b)
    m.record_group(GroupRecord(step=1, cell="b1", arcs=[arc(0.5), arc(0.9)]))
    m.high_advantage_audit(step=1)
    assert e._cache == {} and d._cache == {}


def test_audit_reports_per_axis_band_position():
    m = OnlineMonitor(audit_every_n_steps=1)
    m.record_group(GroupRecord(step=1, cell="b1",
                               arcs=[arc(0.9, farmed=True), arc(0.2)]))
    rows = m.high_advantage_audit(step=1)
    assert rows[0]["engine"]["at_lower_edge"] is True
    assert rows[0]["engine"]["q"] is not None


def test_audit_respects_the_interval():
    m = OnlineMonitor(audit_every_n_steps=50)
    m.record_group(GroupRecord(step=1, cell="b1", arcs=[arc(0.5)]))
    assert m.high_advantage_audit(step=7) == []
    assert m.high_advantage_audit(step=50) != []


# ── advantage telemetry (D2.5) ───────────────────────────────────────────────

def test_mean_abs_advantage_tracks_per_cell():
    m = _mon()
    m.record_group(GroupRecord(step=1, cell="b1",
                               arcs=[arc(r) for r in (0.2, 0.4, 0.6, 0.8)]))
    m.record_group(GroupRecord(step=1, cell="b2", arcs=[arc(0.5) for _ in range(4)]))
    assert m.mean_abs_advantage("b1") > 0.1
    assert m.mean_abs_advantage("b2") == pytest.approx(0.0)


# ── snapshot / logging ───────────────────────────────────────────────────────

def test_snapshot_carries_every_v2_signal(tmp_path):
    log = tmp_path / "monitor.jsonl"
    m = OnlineMonitor(audit_every_n_steps=0, log_path=str(log), cal=cal_with(),
                      annomi_grievance_density=0.2)
    m.record_group(GroupRecord(step=1, cell="b1",
                               arcs=[arc(0.9, sub={"e2": 0.3, "q1": 0.1})
                                     for _ in range(8)]))
    snap = m.snapshot(step=10)
    for key in ("rate_telemetry", "band_edge_farming", "cancellation_drift",
                "per_cell", "mean_abs_advantage"):
        assert key in snap, key

    written = [json.loads(x) for x in log.read_text().splitlines()]
    assert any(r["kind"] == "snapshot" for r in written)
    assert any(r["kind"] == "group" for r in written)


def test_grievance_not_hot_watch():
    """Q2-yes / Q1-no among high-advantage arcs: the confound staying closed."""
    m = OnlineMonitor(audit_every_n_steps=0)
    m.record_group(GroupRecord(step=1, cell="b3", arcs=[
        arc(0.9, sub={"e2": 1.0, "q1": 0.0}), arc(0.8, sub={"e2": 1.0, "q1": 0.0})]))
    r = m.subanswer_rates()
    assert r["grievance_not_hot"] == pytest.approx(1.0)
