"""Bound-instrument tests (Phase C3)."""

import numpy as np
import pytest

from streamfno.bounds import (
    bump_perturbation,
    divergence_rate,
    ensemble_outcome_probs,
    kl_gaussian_obs,
    nearest_opposite_margins,
    pde_clean_observables,
    pinsker_bound,
    two_point_bound,
)
from streamfno.events import EventConfig
from streamfno.obs import ObsConfig
from streamfno.pde.solver import solve_fp
from streamfno.sim import SimConfig


def _uniform(m):
    return np.ones(m)


def test_kl_and_pinsker_formulas():
    y_a = np.zeros((4, 3))
    y_b = np.full((4, 3), 0.01)
    r = 0.02
    kl = kl_gaussian_obs(y_a, y_b, r)
    assert kl == pytest.approx(0.5 * 12 * (0.01 / 0.02) ** 2)
    assert pinsker_bound(0.0) == 0.5
    assert pinsker_bound(2.0) == 0.0
    assert pinsker_bound(10.0) == 0.0  # clipped, never negative
    assert 0.0 < pinsker_bound(0.5) < 0.5


def test_pde_observables_match_channel_shape():
    fp = solve_fp(_uniform(64), lambda x, m: -0.1 * np.ones_like(x), 0.04,
                  t_end=4.0, dt=2e-3, dt_sample=0.5)
    ocfg = ObsConfig(dt_obs=1.0, noise_std=0.02, hist_bins=8)
    t, y = pde_clean_observables(fp, ocfg, n_classes=2)
    assert y.shape == (4, 3 * 2 + 8)
    np.testing.assert_allclose(y[:, 6:].sum(axis=1), 1.0, atol=1e-9)
    # per-class entries replicate
    np.testing.assert_array_equal(y[:, 0], y[:, 1])


def test_bump_perturbation_mass_and_locality():
    m = 128
    rho = _uniform(m)
    pert = bump_perturbation(rho, loc=0.7, mag=0.1)
    assert pert.sum() / m == pytest.approx(1.0)
    x = (np.arange(m) + 0.5) / m
    assert pert[np.argmin(np.abs(x - 0.7))] > pert[np.argmin(np.abs(x - 0.2))]


def test_divergence_rate_sign_matches_stability():
    """Stable drift toward a reflecting wall must contract perturbations
    (lambda_+ < 0); an unstable mean-field feedback must expand them."""
    m = 128
    x0 = np.exp(-0.5 * ((np.arange(m) + 0.5) / m - 0.4) ** 2 / 0.1**2)
    x0 /= x0.sum() / m

    stable = divergence_rate(x0, lambda x, mm: -0.3 * np.ones_like(x), 0.04,
                             t_end=8.0, bump_locs=[0.5], bump_mags=[0.05],
                             fit_window=(1.0, 6.0))
    assert stable[0].lam_plus < 0.0

    unstable = divergence_rate(x0, lambda x, mm: (1.2 * (mm - 0.4)) *
                               np.ones_like(x), 0.04, t_end=8.0,
                               bump_locs=[0.5], bump_mags=[0.05],
                               fit_window=(1.0, 6.0))
    assert unstable[0].lam_plus > stable[0].lam_plus


def test_two_point_bound_limits():
    """Deterministic opposite outcomes recover the classic Le Cam form;
    identical outcome probabilities with distinguishable histories give 0."""
    h1 = np.array([0.0])
    z = np.zeros(1)
    # p_a=0, p_b=1, KL=0: error >= 1/2
    np.testing.assert_allclose(
        two_point_bound(h1, z, np.array([1.0]), z, kl=0.0), [0.5])
    # matches pinsker_bound for intermediate KL
    kl = 0.5
    np.testing.assert_allclose(
        two_point_bound(h1, z, np.array([1.0]), z, kl=kl),
        [pinsker_bound(kl)])
    # fully distinguishable (KL >= 2): bound collapses to 0
    assert two_point_bound(h1, z, np.array([1.0]), z, kl=5.0)[0] == 0.0
    # aleatoric coin-flip outcomes, indistinguishable: bound 1/2
    half = np.array([0.5])
    np.testing.assert_allclose(two_point_bound(half, z, half, z, 0.0), [0.5])
    # MC uncertainty is charged against the bound
    se = np.array([0.05])
    assert two_point_bound(half, se, half, se, 0.0)[0] < 0.5


def test_ensemble_outcome_probs_monotone_and_sane():
    """Outcome probabilities from the vectorized MC: in [0,1], monotone in
    h, near 1 for a saturated start under sustained burst, near 0 for a
    drained start under drain."""
    m = 128
    x = (np.arange(m) + 0.5) / m
    ecfg = EventConfig(threshold=0.05, flux_window=2.0,
                       lead_times=(2.0, 4.0, 8.0))
    cfg = SimConfig(n_partitions=80, buffer_depth=32, t_end=8.0, seed=0,
                    dt_sample=0.5, mode="diffusive", a=0.05, arrival="mmpp",
                    mmpp_shared=True, lam_low=0.30, lam_high=0.95,
                    r_low_high=1e-6, r_high_low=1e-6, mu0=0.6, n_brokers=2)

    hot = np.exp(-0.5 * ((x - 0.9) / 0.05) ** 2)
    hot /= hot.sum() / m
    p_hot, se = ensemble_outcome_probs(cfg, hot, mod_state=1, ecfg=ecfg,
                                       n_reps=32, seed=1)
    assert np.all(np.diff(p_hot) >= -1e-12)
    assert p_hot[-1] > 0.9

    cold = np.exp(-0.5 * ((x - 0.1) / 0.05) ** 2)
    cold /= cold.sum() / m
    p_cold, _ = ensemble_outcome_probs(cfg, cold, mod_state=0, ecfg=ecfg,
                                       n_reps=32, seed=2)
    assert p_cold[-1] < 0.1
    assert np.all((se >= 0.0) & (se <= 0.5 / np.sqrt(32) + 1e-9))


def test_margins_simple_geometry():
    y = np.array([[0.0], [1.0], [3.0]])
    labels = np.array([False, False, True])
    m = nearest_opposite_margins(y, labels, noise_std=1.0)
    np.testing.assert_allclose(m, [3.0, 2.0, 2.0])
    empty = nearest_opposite_margins(y, np.zeros(3, dtype=bool), noise_std=1.0)
    assert np.isnan(empty).all()
