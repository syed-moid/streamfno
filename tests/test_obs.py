"""Observation layer and simulator-extension tests (Phase C1)."""

import numpy as np

from streamfno.obs import Episode, ObsConfig, clean_observables, observe
from streamfno.sim import SimConfig, TauLeapSim, simulate

MMPP = dict(
    n_partitions=200, buffer_depth=32, t_end=40.0, dt_sample=0.5,
    mode="diffusive", a=0.05, arrival="mmpp", lam_low=0.3, lam_high=0.9,
    r_low_high=0.15, r_high_low=0.25, mu0=0.7, n_brokers=2,
)


def test_shared_mmpp_produces_coherent_bursts():
    """A shared modulator must move the aggregate load coherently: the
    variance over time of the population mean lag is far larger than with
    independent per-partition modulators (which average out)."""
    v_shared, v_indep = [], []
    for seed in (0, 1, 2):
        shared = simulate(SimConfig(seed=seed, mmpp_shared=True, **MMPP))
        indep = simulate(SimConfig(seed=seed, mmpp_shared=False, **MMPP))
        mask = shared.times >= 10.0
        v_shared.append(shared.mean_lag[mask].mean(axis=1).var())
        v_indep.append(indep.mean_lag[mask].mean(axis=1).var())
    assert np.mean(v_shared) > 5.0 * np.mean(v_indep), (v_shared, v_indep)


def test_shared_mmpp_gillespie_runs():
    cfg = SimConfig(seed=3, mmpp_shared=True, method="gillespie",
                    **{**MMPP, "n_partitions": 40, "t_end": 5.0})
    res = simulate(cfg)
    assert res.hist.sum(axis=1).min() == 40


def test_tauleap_incremental_matches_simulate():
    """Advancing a TauLeapSim interval by interval reproduces simulate()
    exactly (same seed, same rng path)."""
    cfg = SimConfig(seed=9, mmpp_shared=True, **MMPP)
    res = simulate(cfg)
    sim = TauLeapSim(cfg)
    k = int(round(10.0 / cfg.dt_sample))
    rej = np.zeros(cfg.n_brokers, dtype=np.int64)
    for i in range(1, k + 1):
        rej_i = sim.advance(cfg.dt_sample)
        if i == k:
            rej = rej_i
    np.testing.assert_array_equal(
        np.bincount(sim.q, minlength=cfg.buffer_depth + 1), res.lattice_hist[k])
    np.testing.assert_array_equal(rej, res.rejected[k])


def test_tauleap_clone_diverges_but_preserves_original():
    cfg = SimConfig(seed=5, **MMPP)
    a = TauLeapSim(cfg)
    a.advance(5.0)
    b = a.clone()
    qa0 = a.q.copy()
    b.advance(2.0)
    np.testing.assert_array_equal(a.q, qa0)  # clone stepping leaves a alone
    a.advance(2.0)
    assert not np.array_equal(a.q, b.q)  # independent randomness


def test_observables_shapes_and_flux_consistency():
    cfg = SimConfig(seed=1, **MMPP)
    res = simulate(cfg)
    ocfg = ObsConfig(dt_obs=1.0, noise_std=0.0, hist_bins=8, noise_seed=0)
    times, y = clean_observables(res, ocfg)
    C = cfg.n_brokers
    assert y.shape == (len(times), 3 * C + 8)
    assert times[0] == 1.0
    # histogram fractions sum to 1
    np.testing.assert_allclose(y[:, 3 * C:].sum(axis=1), 1.0, atol=1e-12)
    # per-class flux averaged over classes reproduces the aggregate series
    stride = round(ocfg.dt_obs / cfg.dt_sample)
    idx = np.arange(stride, len(res.times), stride)
    agg = np.cumsum(res.boundary_flux_total())
    agg_obs = (agg[idx] - agg[idx - stride]) / stride
    w = res.class_sizes / cfg.n_partitions
    np.testing.assert_allclose(y[:, 2 * C:3 * C] @ w, agg_obs, atol=1e-12)


def test_noise_protocol_and_roundtrip(tmp_path):
    cfg = SimConfig(seed=2, **{**MMPP, "t_end": 10.0})
    res = simulate(cfg)
    e1 = observe(res, ObsConfig(noise_seed=7, noise_std=0.05))
    e1b = observe(res, ObsConfig(noise_seed=7, noise_std=0.05))
    e2 = observe(res, ObsConfig(noise_seed=8, noise_std=0.05))
    np.testing.assert_array_equal(e1.y, e1b.y)  # same noise seed reproduces
    np.testing.assert_array_equal(e1.y_clean, e2.y_clean)  # same trajectory
    assert not np.allclose(e1.y, e2.y)  # different noise draw
    p = tmp_path / "ep.npz"
    e1.meta["split"] = "train"
    e1.save(p)
    back = Episode.load(p)
    np.testing.assert_array_equal(back.y, e1.y)
    assert back.sim_config == cfg
    assert back.meta["split"] == "train"
    assert len(back.obs_names) == e1.y.shape[1]
