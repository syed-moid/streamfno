"""Unit tests for the real-Kafka contact layer (no cluster required)."""

import numpy as np
import pytest

from streamfno.kafka.adapter import (
    LagTelemetry,
    overshoot_flux,
    telemetry_episode,
)
from streamfno.kafka.pacing import TokenBucket, mmpp_timeline, rate_at
from streamfno.kafka.params import RunParams


def test_token_bucket_rate_and_cap():
    tb = TokenBucket(rate=100.0, burst_s=0.25)
    tb.refill(0.0)
    assert tb.take(10) == 0  # starts empty
    tb.refill(0.1)
    assert tb.take(100) == 10  # 100/s * 0.1s
    tb.refill(10.0)  # long stall: capped at burst capacity
    assert tb.take(1000) == 25
    tb.give_back(5)
    assert tb.take(1000) == 5


def test_mmpp_timeline_deterministic_and_alternating():
    s1 = mmpp_timeline(500.0, 0.06, 0.30, seed=42)
    s2 = mmpp_timeline(500.0, 0.06, 0.30, seed=42)
    np.testing.assert_array_equal(s1, s2)
    assert np.all(np.diff(s1) > 0)
    # low before first switch, high after it
    assert rate_at(0.0, s1, 1.0, 2.0) == 1.0
    assert rate_at(float(s1[0]) + 1e-6, s1, 1.0, 2.0) == 2.0
    # long-run high fraction near r_lh / (r_lh + r_hl) = 1/6
    t = np.linspace(0, 500, 20000)
    frac = np.mean([rate_at(ti, s1, 0.0, 1.0) for ti in t])
    assert 0.08 < frac < 0.28


def test_unit_mapping():
    p = RunParams(tau_s=4.0, budget_b=170, n_partitions=96, mu0=0.7,
                  n_consumers=2)
    # rate_x * B / tau * N
    assert p.msgs_per_s(0.82) == pytest.approx(0.82 * 170 / 4.0 * 96)
    assert p.consumer_rate_msgs == pytest.approx(0.7 * 170 / 4.0 * 96 / 2)
    assert p.dt_poll_norm == pytest.approx(0.25)


def test_overshoot_flux_counts_only_positive_increments():
    budget = 10
    lag = np.array([[8, 0], [12, 0], [15, 3], [11, 3], [14, 3]])
    j = overshoot_flux(lag, budget, dt_norm=0.5)
    # increments of (lag-10)^+: [2,0], [3,0], [-4,0]->0, [3,0]
    np.testing.assert_allclose(j, np.array([0, 2, 3, 0, 3]) / (2 * 10 * 0.5))


def _fake_telemetry(params, k=41, seed=0):
    rng = np.random.default_rng(seed)
    t0 = 1000.0
    t_wall = t0 + np.arange(k) * params.dt_poll_s
    lam = 400.0  # msgs/s aggregate
    arr = rng.poisson(lam * params.dt_poll_s / params.n_partitions,
                      size=(k, params.n_partitions))
    leo = np.cumsum(arr, axis=0)
    committed = np.maximum(leo - rng.integers(
        0, 30, size=leo.shape), 0)
    return LagTelemetry(
        t_wall=t_wall, leo=leo, committed=committed,
        sample_latency_s=np.full(k, 0.01), cpu_s=np.linspace(0, 1, k),
        meta={}), t0


def test_telemetry_episode_shapes_and_conventions():
    p = RunParams(t_end=10.0)
    tel, t0 = _fake_telemetry(p)
    ep = telemetry_episode(tel, p, t0_wall=t0, dt_obs=1.0, hist_bins=8)
    # obs vector: mean, var, flux, 8 hist bins for one class
    assert ep.y.shape[1] == 3 + 8
    np.testing.assert_allclose(ep.y, ep.y_clean)
    assert ep.times[0] == pytest.approx(1.0)
    np.testing.assert_allclose(np.diff(ep.times), 1.0)
    # histogram fractions sum to 1
    np.testing.assert_allclose(ep.y[:, 3:].sum(axis=1), 1.0)
    assert ep.sim_config.mode == "fluid"
    assert ep.meta["source"] == "kafka"
    # flux at raw cadence, first entry zero
    assert ep.flux_times.shape == ep.flux_hidden.shape
    assert ep.flux_hidden[0] == 0.0


def test_telemetry_episode_roundtrip(tmp_path):
    p = RunParams(t_end=10.0)
    tel, t0 = _fake_telemetry(p)
    ep = telemetry_episode(tel, p, t0_wall=t0)
    ep.save(tmp_path / "ep.npz")
    from streamfno.obs import Episode
    back = Episode.load(tmp_path / "ep.npz")
    np.testing.assert_allclose(back.y, ep.y)
    assert back.meta["budget_b"] == p.budget_b
