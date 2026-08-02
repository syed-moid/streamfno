"""Simulator tests: engine cross-validation and boundary semantics."""

import numpy as np
from scipy.stats import wasserstein_distance

from streamfno.sim import SimConfig, simulate


def _time_avg_lattice_density(res, t_from):
    """Time-averaged empirical measure on the lag lattice from t >= t_from."""
    mask = res.times >= t_from
    counts = res.lattice_hist[mask].sum(axis=0)
    x = np.arange(counts.size) / (counts.size - 1)
    return x, counts / counts.sum()


def _w1(res_a, res_b, t_from):
    xa, pa = _time_avg_lattice_density(res_a, t_from)
    xb, pb = _time_avg_lattice_density(res_b, t_from)
    return wasserstein_distance(xa, xb, pa, pb)


SMALL = dict(
    n_partitions=80, buffer_depth=16, t_end=12.0, dt_sample=0.5,
    mode="diffusive", a=0.06, lam=0.35, mu0=0.5, n_brokers=2,
)


def test_gillespie_and_tau_leap_agree():
    """Distributional agreement of the two engines on a small configuration.

    Time-averaged stationary-window lag measures must be within a Wasserstein
    distance consistent with sampling noise (~1/sqrt(N * n_samples) scale).
    """
    d_cross = []
    d_within = []
    for seed in (1, 2, 3):
        g = simulate(SimConfig(method="gillespie", seed=seed, **SMALL))
        t = simulate(SimConfig(method="tau_leap", seed=seed + 100, **SMALL))
        t2 = simulate(SimConfig(method="tau_leap", seed=seed + 200, **SMALL))
        d_cross.append(_w1(g, t, t_from=6.0))
        d_within.append(_w1(t, t2, t_from=6.0))
    # cross-engine discrepancy must be comparable to seed-to-seed noise
    assert np.mean(d_cross) < 0.02, (d_cross, d_within)
    assert np.mean(d_cross) < 3.0 * max(np.mean(d_within), 1e-3)


def test_rejections_counted_at_saturation():
    """Supercritical load must produce blocked up-jumps (boundary flux)."""
    cfg = dict(
        n_partitions=60, buffer_depth=20, t_end=10.0, dt_sample=1.0,
        mode="diffusive", a=0.06, lam=0.8, mu0=0.5, init_x0=0.8, init_sd=0.05,
    )
    for method in ("gillespie", "tau_leap"):
        res = simulate(SimConfig(method=method, seed=7, **cfg))
        assert res.rejected.sum() > 0, method
        assert res.boundary_flux_total()[-1] >= 0.0
        # mass ends up near the boundary
        assert res.mean_lag[-1].mean() > 0.5, method


def test_mmpp_modulation():
    """MMPP arrivals run in both engines; long-run mean lag sits between the
    all-low and all-high constant-rate runs."""
    base = dict(
        n_partitions=100, buffer_depth=16, t_end=15.0, dt_sample=0.5,
        mode="diffusive", a=0.08, mu0=0.6, seed=11,
    )
    lo = simulate(SimConfig(arrival="poisson", lam=0.3, **base))
    hi = simulate(SimConfig(arrival="poisson", lam=0.8, **base))
    mm = simulate(SimConfig(arrival="mmpp", lam_low=0.3, lam_high=0.8,
                            r_low_high=1.0, r_high_low=1.0, **base))
    m_lo = lo.mean_lag[-10:].mean()
    m_hi = hi.mean_lag[-10:].mean()
    m_mm = mm.mean_lag[-10:].mean()
    assert m_lo < m_mm < m_hi
    # gillespie MMPP path exercises the switch events
    small = dict(base, n_partitions=30, t_end=4.0)
    res = simulate(SimConfig(arrival="mmpp", lam_low=0.3, lam_high=0.8,
                             r_low_high=2.0, r_high_low=2.0,
                             method="gillespie", **{**small, "seed": 12}))
    assert res.hist.sum(axis=1).min() == 30


def test_fluid_mode_has_vanishing_variance():
    """Fluid mode: spread around the transported profile shrinks with B,
    unlike diffusive mode where it is B-independent."""
    base = dict(
        n_partitions=400, t_end=1.0, dt_sample=0.5, lam=0.3, mu0=0.7,
        init_x0=0.6, init_sd=0.0, seed=5,
    )
    fluid_small = simulate(SimConfig(mode="fluid", buffer_depth=25, **base))
    fluid_large = simulate(SimConfig(mode="fluid", buffer_depth=400, **base))
    diff = simulate(SimConfig(mode="diffusive", a=0.05, buffer_depth=400, **base))
    v_small = fluid_small.var_lag[-1, 0]
    v_large = fluid_large.var_lag[-1, 0]
    v_diff = diff.var_lag[-1, 0]
    assert v_large < v_small / 4  # variance ~ 1/B
    assert v_diff > 5 * v_large  # diffusive noise does not vanish


def test_seeded_reproducibility_and_io(tmp_path):
    cfg = SimConfig(n_partitions=50, buffer_depth=12, t_end=3.0, seed=42,
                    dt_sample=0.5, a=0.06, lam=0.4, mu0=0.5)
    r1 = simulate(cfg)
    r2 = simulate(cfg)
    np.testing.assert_array_equal(r1.hist, r2.hist)
    np.testing.assert_array_equal(r1.rejected, r2.rejected)
    p = tmp_path / "run.npz"
    r1.save(p)
    from streamfno.sim import SimResult
    r3 = SimResult.load(p)
    np.testing.assert_array_equal(r1.lattice_hist, r3.lattice_hist)
    assert r3.config == cfg


def test_diffusive_rate_validation():
    import pytest
    with pytest.raises(ValueError, match="B \\* a"):
        SimConfig(n_partitions=10, buffer_depth=5, t_end=1.0, seed=0,
                  mode="diffusive", a=0.01, lam=0.9, mu0=0.2)
