"""Analysis library tests: W1 sanity and spectral decay recovery."""

import numpy as np
import pytest

from streamfno.analysis.distances import (
    density_to_weights,
    w1_lattice_vs_density,
    w1_weights,
)
from streamfno.analysis.spectral import (
    cosine_coefficients,
    fft_coefficients,
    fit_decay,
)


def test_w1_shifted_atoms():
    """W1 between point masses equals the shift."""
    x = np.array([0.2])
    y = np.array([0.5])
    assert w1_weights(x, np.array([1.0]), y, np.array([1.0])) == pytest.approx(0.3)


def test_w1_lattice_vs_density_self():
    """A lattice measure sampled from a density is W1-close to the density."""
    m = 200
    x = (np.arange(m) + 0.5) / m
    rho = 1.0 + 0.5 * np.cos(np.pi * x)
    rho /= rho.sum() / m
    rng = np.random.default_rng(0)
    centers, w = density_to_weights(rho)
    samples = rng.choice(centers, size=200_000, p=w)
    b = 100
    counts = np.bincount(np.rint(samples * b).astype(int), minlength=b + 1)
    d = w1_lattice_vs_density(counts, rho)
    assert d < 0.01, d  # quantization ~ 1/(2B) + sampling noise


def test_fit_decay_recovers_synthetic_exponent():
    k = np.arange(0, 200)
    coeffs = np.zeros(200)
    coeffs[1:] = k[1:] ** -2.5
    fit = fit_decay(coeffs, k_min=2, k_max=100)
    assert fit.s == pytest.approx(2.5, abs=1e-6)
    assert fit.s_lo <= 2.5 <= fit.s_hi


def test_cosine_basis_concentrates_on_neumann_mode():
    """rho = 1 + eps cos(pi x) is a pure k=1 mode in the cosine basis."""
    m = 256
    x = (np.arange(m) + 0.5) / m
    rho = 1.0 + 0.3 * np.cos(np.pi * x)
    c = cosine_coefficients(rho)
    assert c[1] > 100 * np.abs(c[2:]).max()


def test_boundary_jump_caps_fft_decay_but_not_cosine():
    """A smooth non-periodic density (rho(0) != rho(1)) decays fast in the
    cosine basis but only ~ k^-1 in the periodic FFT basis -- the basis-choice
    effect the paper needs quantified."""
    m = 512
    x = (np.arange(m) + 0.5) / m
    rho = np.exp(-3.0 * x)
    rho /= rho.sum() / m
    fit_fft = fit_decay(fft_coefficients(rho), k_min=4, k_max=60)
    fit_cos = fit_decay(cosine_coefficients(rho), k_min=4, k_max=60)
    assert fit_fft.s < 1.3
    assert fit_cos.s > fit_fft.s + 0.5


def test_fit_decay_rejects_empty_range():
    with pytest.raises(ValueError):
        fit_decay(np.zeros(50), k_min=2, k_max=40)
