"""Predictor-suite tests (Phase C4)."""

import numpy as np
import pytest

from streamfno.events import EventConfig
from streamfno.obs import ObsConfig, observe
from streamfno.predictors import (
    ParticleFilter,
    metrics_with_ci,
    reactive_scores,
    tune_threshold,
    window_features,
)
from streamfno.sim import SimConfig, simulate


def _small_episode(seed=0, t_end=24.0, lam_high=0.9):
    cfg = SimConfig(n_partitions=60, buffer_depth=16, t_end=t_end, seed=seed,
                    dt_sample=0.5, n_bins=16, mode="diffusive", a=0.06,
                    arrival="mmpp", mmpp_shared=True, lam_low=0.35,
                    lam_high=lam_high, r_low_high=0.1, r_high_low=0.2,
                    mu0=0.6, n_brokers=2, init_x0=0.3, init_sd=0.05)
    return observe(simulate(cfg), ObsConfig(dt_obs=1.0, noise_std=0.02,
                                            hist_bins=8, noise_seed=seed + 500))


def test_tune_threshold_separable_and_ties():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([False, False, True, True])
    thr = tune_threshold(scores, labels)
    assert 0.2 < thr < 0.8
    assert ((scores > thr) == labels).all()


def test_metrics_with_ci_structure():
    rng = np.random.default_rng(0)
    labels = rng.random(200) < 0.3
    scores = np.clip(labels * 0.6 + rng.normal(0.2, 0.15, 200), 0, 1)
    eps_ids = np.repeat(np.arange(20), 10)
    thr = tune_threshold(scores, labels)
    m = metrics_with_ci(scores, labels, eps_ids, thr, n_boot=100)
    assert 0.0 <= m["error"] <= 1.0
    assert m["error_ci"][0] <= m["error"] <= m["error_ci"][1]
    assert m["pr_auc"] > m["base_rate"]  # informative scores
    assert m["n_episodes"] == 20


def test_window_features_layout():
    ep = _small_episode()
    t_dec = ep.times[10:12]
    f = window_features(ep, t_dec, window=4)
    d = ep.y.shape[1]
    assert f.shape == (2, 4 * d + 3 * d)
    i = np.searchsorted(ep.times, t_dec[0])
    np.testing.assert_array_equal(f[0, :4 * d],
                                  ep.y[i - 3:i + 1].ravel())
    np.testing.assert_allclose(f[0, 4 * d:5 * d], ep.y[i - 3:i + 1].mean(axis=0))


def test_reactive_scores_kinds():
    ep = _small_episode()
    t_dec = ep.times[8:16]
    s_flux = reactive_scores(ep, t_dec, "flux")
    s_wall = reactive_scores(ep, t_dec, "wall")
    assert s_flux.shape == s_wall.shape == (8,)
    with pytest.raises(ValueError):
        reactive_scores(ep, t_dec, "nope")


def test_particle_filter_runs_and_is_monotone_in_h():
    ep = _small_episode(seed=3, t_end=20.0)
    ecfg = EventConfig(threshold=0.02, flux_window=2.0,
                       lead_times=(2.0, 4.0, 8.0), t_warmup=6.0,
                       dt_decision=4.0)
    pf = ParticleFilter(ep.sim_config, ep.obs_config, n_particles=24, seed=11)
    t_dec = np.array([8.0, 12.0])
    res = pf.run_episode(ep, t_dec, ecfg)
    assert res.probs.shape == (2, 3)
    assert np.all((res.probs >= 0.0) & (res.probs <= 1.0))
    # E_h nested in h => P(E_h | obs) nondecreasing in h
    assert np.all(np.diff(res.probs, axis=1) >= -1e-12)
    assert len(res.ess) == len(ep.times)
    assert np.all(res.ess >= 1.0) and np.all(res.ess <= 24.0 + 1e-9)


def test_particle_filter_tracks_load_signal():
    """P(E_h) must be substantially higher when observing a saturating
    episode than a draining one (same filter, same seeds)."""
    ecfg = EventConfig(threshold=0.02, flux_window=2.0, lead_times=(4.0,),
                      t_warmup=6.0, dt_decision=4.0)
    hot = _small_episode(seed=21, lam_high=1.0)
    cold_cfg = SimConfig(**{**hot.sim_config.__dict__, "arrival": "poisson",
                            "lam": 0.2, "seed": 22})
    cold = observe(simulate(cold_cfg),
                   ObsConfig(dt_obs=1.0, noise_std=0.02, hist_bins=8,
                             noise_seed=600))
    t_dec = np.array([16.0])
    p_hot = ParticleFilter(hot.sim_config, hot.obs_config, 24, seed=1
                           ).run_episode(hot, t_dec, ecfg).probs[0, 0]
    p_cold = ParticleFilter(cold.sim_config, cold.obs_config, 24, seed=1
                            ).run_episode(cold, t_dec, ecfg).probs[0, 0]
    assert p_hot > p_cold + 0.2, (p_hot, p_cold)
