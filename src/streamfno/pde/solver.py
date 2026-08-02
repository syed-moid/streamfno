"""Conservative finite-volume solver for the reflected nonlinear
Fokker-Planck equation on [0, 1].

    d rho/dt = -d/dx [ b(x, m) rho ] + 1/2 d^2/dx^2 [ a(x, m) rho ]

for one density per broker class, coupled through the class mean lag
m_c = integral x rho_c(x) dx (the mean-field coupling restricted to broker
classes, matching the simulator's service degradation).

Units: x is normalized lag in [0, 1]; t is normalized time; b is in x per
unit time (netput drift), a in x^2 per unit time (netput variance rate) —
identical conventions to streamfno.sim (see sim/config.py).

Discretization:
- Finite volume on M uniform cells, Chang-Cooper exponential weighting of the
  advective interface flux (positivity-preserving, exact on stationary
  exponential profiles); degenerates to first-order upwinding when a = 0, so
  the same solver provides the transport (fluid-limit) reference solution.
- Implicit Euler in time with coefficients frozen at the step start
  (semi-implicit in the mean-field coupling); one tridiagonal solve per class
  per step.

Boundaries:
- x = 0: no-flux (reflecting).
- x = 1: regulated.  The wall is mass-conserving (the numerical flux through
  x = 1 is zero, matching the simulator where saturated partitions remain in
  the system at X = 1), and the advective flux the wall cancels,
  max(b(1, m), 0) * rho(1, t), is accumulated into the regulator K_B(t) -- the
  PDE counterpart of the simulator's rejected-work counter J_B.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded

__all__ = ["FPResult", "solve_fp", "stationary_exponential"]


@dataclass
class FPResult:
    """Sampled solver output.

    times: (K,) sample times.  x_centers: (M,) cell centers.
    rho: (K, C, M) densities (integrate to the initial mass over x).
    regulator_rate: (K, C) instantaneous regulator rate dK/dt at sample times.
    regulator_cum: (K, C) cumulative regulator K_B(t).
    mean_lag: (K, C) class means m_c(t).
    """

    times: np.ndarray
    x_centers: np.ndarray
    rho: np.ndarray
    regulator_rate: np.ndarray
    regulator_cum: np.ndarray
    mean_lag: np.ndarray

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, times=self.times, x_centers=self.x_centers, rho=self.rho,
            regulator_rate=self.regulator_rate, regulator_cum=self.regulator_cum,
            mean_lag=self.mean_lag,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FPResult":
        with np.load(path) as f:
            return cls(times=f["times"], x_centers=f["x_centers"], rho=f["rho"],
                       regulator_rate=f["regulator_rate"],
                       regulator_cum=f["regulator_cum"], mean_lag=f["mean_lag"])


def _chang_cooper_delta(w: np.ndarray) -> np.ndarray:
    """Chang-Cooper weight delta(w) = 1/w - 1/(e^w - 1), w = C h / D.

    delta -> 1/2 as w -> 0 (centered) and -> 1 (0) as w -> +inf (-inf)
    (full upwind); makes the discrete stationary flux vanish exactly on
    exponential profiles exp(C x / D).
    """
    out = np.full_like(w, 0.5)
    small = np.abs(w) < 1e-8
    ws = w[~small]
    out[~small] = 1.0 / ws - 1.0 / np.expm1(ws)
    out[small] = 0.5 - w[small] / 12.0
    return out


def solve_fp(
    rho0: np.ndarray,
    drift,
    diffusion,
    t_end: float,
    dt: float = 1e-3,
    dt_sample: float | None = None,
) -> FPResult:
    """Integrate the coupled per-class Fokker-Planck system.

    Parameters
    ----------
    rho0:
        Initial densities, shape (C, M) or (M,); each row a probability
        density on [0, 1] w.r.t. dx (cell-averaged values).
    drift:
        Callable b(x, m) -> array broadcastable to x; x are interface
        coordinates, m the scalar mean lag of the class being advanced.
    diffusion:
        Netput variance rate a: a float, or callable a(x, m) like ``drift``.
        The FP diffusion coefficient is D = a / 2.  a = 0 gives pure
        transport (upwind).
    t_end, dt:
        Horizon and time step (implicit Euler; unconditionally stable, but dt
        also freezes the nonlinear coupling, so convergence must be checked
        by halving dt and the grid together).
    dt_sample:
        Output sampling interval; defaults to max(dt, t_end / 200).
    """
    rho = np.atleast_2d(np.asarray(rho0, dtype=float)).copy()
    n_classes, m_cells = rho.shape
    h = 1.0 / m_cells
    x_centers = (np.arange(m_cells) + 0.5) * h
    x_faces = np.arange(m_cells + 1) * h

    if dt_sample is None:
        dt_sample = max(dt, t_end / 200.0)
    every = max(1, round(dt_sample / dt))
    n_steps = round(t_end / dt)
    a_fn = diffusion if callable(diffusion) else (lambda x, m, _a=float(diffusion):
                                                  np.full_like(x, _a))

    times = [0.0]
    reg_cum = np.zeros(n_classes)
    means0 = rho @ x_centers * h
    out_rho = [rho.copy()]
    out_rate = [np.array([max(float(np.asarray(drift(np.array([1.0]), means0[c]))[0]), 0.0)
                          * rho[c, -1] for c in range(n_classes)])]
    out_cum = [reg_cum.copy()]
    out_mean = [means0.copy()]

    for step in range(1, n_steps + 1):
        masses = rho.sum(axis=1) * h
        means = rho @ x_centers * h / np.maximum(masses, 1e-300)
        rate_now = np.zeros(n_classes)
        for c in range(n_classes):
            m = float(means[c])
            b_f = np.broadcast_to(np.asarray(drift(x_faces, m), dtype=float),
                                  x_faces.shape).copy()
            d_f = np.broadcast_to(np.asarray(a_fn(x_faces, m), dtype=float),
                                  x_faces.shape) / 2.0

            # interface flux F_i = alpha_i * rho_{i-1} + beta_i * rho_i
            alpha = np.zeros(m_cells + 1)
            beta = np.zeros(m_cells + 1)
            interior = slice(1, m_cells)
            bi, di = b_f[interior], d_f[interior]
            pos_d = di > 0.0
            delta = np.where(bi > 0.0, 1.0, 0.0)  # pure upwind where D = 0
            if pos_d.any():
                w = np.zeros_like(bi)
                w[pos_d] = bi[pos_d] * h / di[pos_d]
                delta = np.where(pos_d, _chang_cooper_delta(w), delta)
            alpha[interior] = bi * delta + di / h
            beta[interior] = bi * (1.0 - delta) - di / h

            # implicit Euler: (I - dt A) rho_new = rho, A tridiagonal,
            # row j of A: [alpha_j, beta_j - alpha_{j+1}, -beta_{j+1}] / h
            lower = alpha[1:m_cells] / h
            diag = (beta[:m_cells] - alpha[1:]) / h
            upper = -beta[1:m_cells] / h
            ab = np.zeros((3, m_cells))
            ab[0, 1:] = -dt * upper
            ab[1, :] = 1.0 - dt * diag
            ab[2, :-1] = -dt * lower
            rho[c] = solve_banded((1, 1), ab, rho[c])

            # regulator: advective flux the reflecting wall at x=1 cancels
            rate_now[c] = max(float(b_f[-1]), 0.0) * rho[c, -1]
        reg_cum = reg_cum + rate_now * dt

        if step % every == 0 or step == n_steps:
            times.append(step * dt)
            out_rho.append(rho.copy())
            out_rate.append(rate_now.copy())
            out_cum.append(reg_cum.copy())
            out_mean.append((rho @ x_centers * h / np.maximum(rho.sum(axis=1) * h,
                                                              1e-300)).copy())

    return FPResult(
        times=np.array(times), x_centers=x_centers, rho=np.array(out_rho),
        regulator_rate=np.array(out_rate), regulator_cum=np.array(out_cum),
        mean_lag=np.array(out_mean),
    )


def stationary_exponential(x: np.ndarray, b: float, a: float) -> np.ndarray:
    """Closed-form stationary density for constant b and a > 0 with
    reflection at both walls of [0, 1]: rho(x) = Z^-1 exp(2 b x / a)."""
    r = 2.0 * b / a
    if abs(r) < 1e-12:
        return np.ones_like(x)
    return r * np.exp(r * x) / np.expm1(r)
