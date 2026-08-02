"""Spectral decay of lag densities.

The T4 program needs the empirical decay rate s of the density's spectral
coefficients, |c_k| ~ k^(-s).  Reflecting boundaries make the cosine basis
{cos(k pi x)} the natural choice (it diagonalizes the Neumann Laplacian on
[0,1]); the plain-FFT (periodic) version is computed alongside because the
paper needs to know whether the basis choice matters in practice: a density
with rho(0) != rho(1) has a periodic-extension jump that caps FFT decay at
s ~ 1 regardless of interior smoothness.

Conventions: densities are cell-averaged values on M uniform cells of [0,1].
The DCT-II of these samples approximates the cosine coefficients
c_k = 2 * integral rho(x) cos(k pi x) dx up to normalization; decay exponents
are invariant to the overall scale, so no normalization is applied beyond
scipy's default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct, rfft
from scipy.stats import theilslopes

__all__ = ["cosine_coefficients", "fft_coefficients", "fit_decay", "DecayFit"]


def cosine_coefficients(rho: np.ndarray) -> np.ndarray:
    """|c_k| in the cosine basis, k = 0..M-1 (DCT-II of the cell values)."""
    return np.abs(dct(np.asarray(rho, dtype=float)))


def fft_coefficients(rho: np.ndarray) -> np.ndarray:
    """|c_k| in the periodic Fourier basis, k = 0..M//2 (rFFT magnitudes)."""
    return np.abs(rfft(np.asarray(rho, dtype=float)))


@dataclass
class DecayFit:
    """Robust log-log fit of |c_k| ~ k^(-s) over k in [k_min, k_max].

    s is the negated Theil-Sen slope; (s_lo, s_hi) the 95% confidence
    interval; n_points the number of nonzero coefficients used.
    """

    s: float
    s_lo: float
    s_hi: float
    k_min: int
    k_max: int
    n_points: int


def fit_decay(coeffs: np.ndarray, k_min: int, k_max: int) -> DecayFit:
    """Fit the tail decay exponent by Theil-Sen regression of log|c_k| on
    log k over the stated k-range (robust to occasional near-zero
    coefficients from density symmetries)."""
    coeffs = np.asarray(coeffs, dtype=float)
    k_max = min(k_max, coeffs.size - 1)
    k = np.arange(k_min, k_max + 1)
    c = coeffs[k_min:k_max + 1]
    keep = c > 0.0
    k, c = k[keep], c[keep]
    if k.size < 4:
        raise ValueError("too few nonzero coefficients in the fit range")
    slope, _, lo, hi = theilslopes(np.log(c), np.log(k), alpha=0.95)
    return DecayFit(s=float(-slope), s_lo=float(-hi), s_hi=float(-lo),
                    k_min=k_min, k_max=int(k[-1]), n_points=int(k.size))
