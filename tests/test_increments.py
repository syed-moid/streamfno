"""Tests for the conditional-increment (Kramers-Moyal) estimators."""

import numpy as np
import pytest

from streamfno.analysis.increments import (
    binned_increments,
    classify_two_state,
    interior_mean_drift,
)


def _simulate_reflected(b, a, dt, k, p, seed=0, x0=0.5):
    """Euler-Maruyama reflected diffusion on [0,1] with constant b, a."""
    rng = np.random.default_rng(seed)
    x = np.full(p, x0)
    out = np.empty((k, p))
    for i in range(k):
        out[i] = x
        x = x + b * dt + np.sqrt(a * dt) * rng.standard_normal(p)
        x = np.abs(x)          # reflect at 0
        x = 1.0 - np.abs(1.0 - x)  # reflect at 1
    return out


def test_recovers_constant_drift_and_variance():
    b_true, a_true, dt = -0.15, 0.04, 0.05
    x = _simulate_reflected(b_true, a_true, dt, k=4000, p=64)
    fit = binned_increments(x, dt, stride=2, n_bins=20)
    b_hat, b_se = interior_mean_drift(fit, 0.2, 0.8)
    assert b_hat == pytest.approx(b_true, abs=max(3 * b_se, 0.02))
    sel = (fit.x_centers > 0.2) & (fit.x_centers < 0.8) & np.isfinite(fit.a_hat)
    a_mean = np.average(fit.a_hat[sel], weights=fit.counts[sel])
    assert a_mean == pytest.approx(a_true, rel=0.15)


def test_start_mask_restricts_windows():
    x = _simulate_reflected(0.0, 0.02, 0.1, k=400, p=8)
    mask = np.zeros(400, dtype=bool)
    mask[:200] = True
    fit_all = binned_increments(x, 0.1, stride=4, n_bins=10)
    fit_half = binned_increments(x, 0.1, stride=4, n_bins=10, start_mask=mask)
    assert fit_half.counts.sum() < fit_all.counts.sum()
    with pytest.raises(ValueError):
        binned_increments(x, 0.1, stride=4, start_mask=np.zeros(400, bool))


def test_two_state_classifier_on_noisy_square_wave():
    rng = np.random.default_rng(3)
    k = 2000
    truth = (np.arange(k) // 250) % 2 == 1
    rate = np.where(truth, 3300.0, 1600.0) + rng.normal(0, 120.0, k)
    high, centers = classify_two_state(rate, smooth=5)
    assert centers[1] > centers[0]
    # smoothing lags transitions; demand strong agreement, not perfection
    assert (high == truth).mean() > 0.95
