"""Event definition tests (Phase C2)."""

import numpy as np
import pytest

from streamfno.events import (
    EventConfig,
    calibrate_threshold,
    decision_times,
    label_episode,
    smoothed_flux,
)
from streamfno.obs import Episode, ObsConfig
from streamfno.sim import SimConfig


def _toy_episode(flux, dt=0.5, dt_obs=1.0):
    """Episode with a prescribed hidden flux series and trivial observations."""
    k_sim = len(flux)
    t_sim = np.arange(k_sim) * dt
    t_obs = np.arange(1, int(t_sim[-1]) + 1, dt_obs, dtype=float)
    y = np.zeros((len(t_obs), 3))
    cfg = SimConfig(n_partitions=10, buffer_depth=8, t_end=t_sim[-1], seed=0,
                    dt_sample=dt, a=0.06, lam=0.3, mu0=0.5)
    return Episode(times=t_obs, y=y, y_clean=y.copy(), flux_times=t_sim,
                   flux_hidden=np.asarray(flux, dtype=float),
                   mean_lag_hidden=np.zeros((k_sim, 1)),
                   sim_config=cfg, obs_config=ObsConfig())


def test_smoothed_flux_trailing_mean():
    t = np.arange(6) * 1.0
    f = np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
    out = smoothed_flux(t, f, window=2.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 1.5, 1.5, 0.0, 0.0])


def test_labels_window_and_monotonicity():
    # flux spike at t = 30 (index 60 at dt = 0.5)
    flux = np.zeros(121)
    flux[60] = 10.0
    ep = _toy_episode(flux)
    ecfg = EventConfig(threshold=1.0, flux_window=1.0,
                       lead_times=(2.0, 8.0, 16.0), t_warmup=10.0,
                       dt_decision=2.0)
    t_dec, lab = label_episode(ep, ecfg)
    # events must be monotone in h at every decision time
    assert np.all(lab[:, 0] <= lab[:, 1])
    assert np.all(lab[:, 1] <= lab[:, 2])
    # a decision at t=28 sees the spike within h=2; at t=10 only within h>=16
    i28 = np.argmin(np.abs(t_dec - 28.0))
    i14 = np.argmin(np.abs(t_dec - 14.0))
    assert lab[i28, 0]
    assert not lab[i14, 0] and lab[i14, 2]
    # decision times respect warmup and the largest lead time
    assert t_dec[0] >= 10.0
    assert t_dec[-1] <= ep.flux_times[-1] - 16.0 + 1e-9


def test_calibration_hits_target_rate():
    rng = np.random.default_rng(0)
    eps_proto = EventConfig(threshold=0.0, flux_window=1.0,
                            lead_times=(2.0, 8.0, 16.0), t_warmup=10.0,
                            dt_decision=2.0)
    episodes = [_toy_episode(rng.exponential(1.0, size=121)) for _ in range(20)]
    eps, record = calibrate_threshold(episodes, eps_proto, h_mid=8.0,
                                      target_rate=0.10)
    assert record["achieved_rate"] == pytest.approx(0.10, abs=0.03)
    labels = []
    for ep in episodes:
        ecfg = EventConfig(threshold=eps, flux_window=1.0,
                           lead_times=(2.0, 8.0, 16.0), t_warmup=10.0,
                           dt_decision=2.0)
        _, lab = label_episode(ep, ecfg)
        labels.append(lab[:, 1])
    assert np.concatenate(labels).mean() == pytest.approx(0.10, abs=0.03)


def test_event_config_roundtrip(tmp_path):
    ecfg = EventConfig(threshold=0.123, calibration={"a": 1})
    p = tmp_path / "ecfg.json"
    ecfg.save(p)
    back = EventConfig.load(p)
    assert back == ecfg
    assert isinstance(back.lead_times, tuple)


def test_decision_times_on_obs_grid():
    ep = _toy_episode(np.zeros(121))
    ecfg = EventConfig(threshold=1.0, lead_times=(4.0, 8.0), t_warmup=20.0,
                       dt_decision=4.0)
    t_dec = decision_times(ep, ecfg)
    assert set(t_dec).issubset(set(ep.times))
    assert np.all(np.diff(t_dec) == 4.0)
