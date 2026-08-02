"""Wasserstein-1 distances between lag distributions.

All inputs are probability distributions on [0, 1] given either as weights on
stated support points (empirical lattice measures) or as densities on a
uniform cell grid (PDE output); densities are converted to cell-center
weights.  W1 on the line is computed via scipy (CDF L1 distance), which
handles atomic and continuous inputs on different supports.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

__all__ = ["w1_weights", "w1_lattice_vs_density", "density_to_weights"]


def density_to_weights(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell-averaged density on M uniform cells of [0,1] -> (centers, weights).

    Weights are rho * h normalized to sum to 1.
    """
    rho = np.asarray(rho, dtype=float)
    m = rho.size
    centers = (np.arange(m) + 0.5) / m
    w = rho / rho.sum()
    return centers, w


def w1_weights(x_p: np.ndarray, p: np.ndarray, x_q: np.ndarray, q: np.ndarray) -> float:
    """W1 between weighted point measures (supports need not match)."""
    return float(wasserstein_distance(x_p, x_q, p, q))


def w1_lattice_vs_density(lattice_counts: np.ndarray, rho: np.ndarray) -> float:
    """W1 between an empirical lattice measure (counts on {0, 1/B, ..., 1})
    and a cell-averaged density on a uniform grid of [0, 1]."""
    counts = np.asarray(lattice_counts, dtype=float)
    xs = np.arange(counts.size) / (counts.size - 1)
    xr, wr = density_to_weights(rho)
    return w1_weights(xs, counts / counts.sum(), xr, wr)
