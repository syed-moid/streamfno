"""Correspondence between a simulator configuration and its matched
continuum (Fokker-Planck) problem.

The simulator's netput drift is b = lam - mu(m_c) with the logistic service
degradation, and its diffusive-mode variance rate is the config's ``a``;
the matched PDE uses the same callables (see docs/decisions.md for the
correspondence).  Fluid-mode configurations match the transport equation
(a = 0).

Only single-broker-class configurations are matched here; multi-class
configurations would need one density per class with identical machinery.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .pde.solver import FPResult, solve_fp
from .sim.config import SimConfig

__all__ = ["truncated_gaussian_density", "drift_from_config", "matched_pde"]


def truncated_gaussian_density(x0: float, sd: float, m_cells: int) -> np.ndarray:
    """The simulator's initial law on [0,1] as a cell-averaged density:
    a Gaussian truncated (renormalized) to [0,1], or a near-delta for sd=0."""
    x = (np.arange(m_cells) + 0.5) / m_cells
    if sd == 0.0:
        rho = np.zeros(m_cells)
        rho[np.argmin(np.abs(x - x0))] = float(m_cells)
        return rho
    rho = np.exp(-0.5 * ((x - x0) / sd) ** 2)
    return rho / (rho.sum() / m_cells)


def drift_from_config(cfg: SimConfig):
    """Netput drift b(x, m) = E[lam] - mu0 * g(m), x-independent.

    For MMPP arrivals the stationary mean rate is used; the matched PDE for
    modulated arrivals is exact only in the fast-switching limit.
    """
    if cfg.arrival == "poisson":
        lam = cfg.lam
    else:
        p_high = cfg.r_low_high / (cfg.r_low_high + cfg.r_high_low)
        lam = p_high * cfg.lam_high + (1.0 - p_high) * cfg.lam_low

    def b(x: np.ndarray, m: float) -> np.ndarray:
        g = 1.0 - cfg.degradation_drop * expit(
            (m - cfg.degradation_theta) / cfg.degradation_width
        )
        return (lam - cfg.mu0 * g) * np.ones_like(x)

    return b


def matched_pde(
    cfg: SimConfig,
    m_cells: int = 400,
    dt: float = 2e-3,
    dt_sample: float | None = None,
    a_override: float | None = None,
) -> FPResult:
    """Solve the continuum problem matched to ``cfg`` on [0, cfg.t_end].

    a_override forces the variance rate (e.g. 0.0 for the transport
    reference); by default diffusive configs use cfg.a and fluid configs 0.0.
    """
    if cfg.n_brokers != 1:
        raise ValueError("matched_pde supports single-broker-class configs only")
    if a_override is not None:
        a = a_override
    else:
        a = cfg.a if cfg.mode == "diffusive" else 0.0
    rho0 = truncated_gaussian_density(cfg.init_x0, cfg.init_sd, m_cells)
    return solve_fp(rho0, drift_from_config(cfg), a, cfg.t_end, dt=dt,
                    dt_sample=dt_sample if dt_sample is not None else cfg.dt_sample)
