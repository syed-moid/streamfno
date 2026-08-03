"""Finite-time divergence rate of the continuum dynamics (diagnostic).

Twin PDE runs: perturb an initial density by a small localized bump, evolve
both, and measure the growth (or decay) of the L2 separation over the
horizons of interest.  The effective finite-time exponent lambda_+ is the
log-linear slope of ||rho_pert - rho_base||_2 over a stated window.  This
parameterizes the interpretable (Lyapunov) form of the horizon bound; the
primary bound is the two-point construction in lecam.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..pde.solver import solve_fp

__all__ = ["bump_perturbation", "DivergenceRecord", "divergence_rate"]


def bump_perturbation(rho: np.ndarray, loc: float, mag: float,
                      width: float = 0.06) -> np.ndarray:
    """Multiplicative localized Gaussian bump, renormalized to unit mass."""
    m = rho.size
    x = (np.arange(m) + 0.5) / m
    pert = rho * (1.0 + mag * np.exp(-0.5 * ((x - loc) / width) ** 2))
    return pert / (pert.sum() / m)


@dataclass
class DivergenceRecord:
    bump_loc: float
    bump_mag: float
    times: np.ndarray
    sep_l2: np.ndarray
    lam_plus: float
    fit_window: tuple


def divergence_rate(rho_base: np.ndarray, drift, a: float, t_end: float,
                    bump_locs, bump_mags, dt: float = 2e-3,
                    dt_sample: float = 0.5,
                    fit_window: tuple = (1.0, 8.0)) -> list[DivergenceRecord]:
    """Evolve base and perturbed densities, record L2 separations and the
    effective exponent fitted over ``fit_window`` (normalized time)."""
    base = solve_fp(rho_base, drift, a, t_end, dt=dt, dt_sample=dt_sample)
    m = rho_base.size
    records = []
    for loc in bump_locs:
        for mag in bump_mags:
            pert = solve_fp(bump_perturbation(rho_base, loc, mag), drift, a,
                            t_end, dt=dt, dt_sample=dt_sample)
            sep = np.sqrt(((pert.rho[:, 0, :] - base.rho[:, 0, :]) ** 2
                           ).sum(axis=1) / m)
            w = (base.times >= fit_window[0]) & (base.times <= fit_window[1])
            good = w & (sep > 0.0)
            lam = float(np.polyfit(base.times[good], np.log(sep[good]), 1)[0])
            records.append(DivergenceRecord(
                bump_loc=loc, bump_mag=mag, times=base.times, sep_l2=sep,
                lam_plus=lam, fit_window=fit_window))
    return records
