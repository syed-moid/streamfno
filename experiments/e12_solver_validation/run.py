"""e12: solver validation -- refinement orders and an independent
high-Peclet reference (checkpoint-1 audit item 0.3).

1. Delta-x refinement at fixed small dt and Delta-t refinement at fixed
   fine grid, each in two regimes: the simulator regime (a = 0.05) and
   the high-Peclet real-telemetry regime (a = 7.4e-4); realized
   convergence orders from log-log slopes against a fine reference.
2. Independent reference: reflected Euler-Maruyama Monte Carlo of the
   same reflected diffusion (2e5 particles, dt = 2.5e-4) in the
   high-Peclet build and drain regimes; W1(FP, MC) reported against the
   MC split-half sampling floor.  (Chosen over the Gillespie jump
   process because it discretizes the SAME continuum object; the
   jump-vs-diffusion gap is a model distance, not a solver check.)

Seeded; saves data/e12/validation.json.
"""

import json
import time
from pathlib import Path

import numpy as np

from streamfno.pde.solver import solve_fp

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e12"

SEED = 1200
T_END = 8.0
REGIMES = {"simulator": {"b": 0.12, "a": 0.05},
           "high_peclet_build": {"b": 0.12, "a": 7.4e-4},
           "high_peclet_drain": {"b": -0.30, "a": 1.0e-3}}
DX_CELLS = (32, 64, 128, 256, 512)
DX_REF_CELLS = 2048
DX_DT = 5e-4
DT_GRID = (6.4e-2, 3.2e-2, 1.6e-2, 8e-3, 4e-3, 2e-3)
DT_REF = 2.5e-4
DT_CELLS = 1024
MC_PARTICLES = 200_000
MC_DT = 2.5e-4
W1_SAMPLES = 16384


def gaussian_rho0(m_cells: int) -> np.ndarray:
    x = (np.arange(m_cells) + 0.5) / m_cells
    rho = np.exp(-0.5 * ((x - 0.2) / 0.05) ** 2)
    return rho / (rho.sum() / m_cells)


def cdf_on(u: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Piecewise-linear CDF of a cell-averaged density, evaluated on
    ``grid`` (works across differing cell counts)."""
    m = u.size
    edges = np.arange(m + 1) / m
    cum = np.concatenate([[0.0], np.cumsum(u) / m])
    return np.interp(grid, edges, cum)


def w1_cross_grid(u_a: np.ndarray, u_b: np.ndarray) -> float:
    grid = np.linspace(0.0, 1.0, W1_SAMPLES)
    return float(np.trapezoid(np.abs(cdf_on(u_a, grid) - cdf_on(u_b, grid)),
                              grid))


def final_density(m_cells: int, dt: float, b: float, a: float) -> np.ndarray:
    res = solve_fp(gaussian_rho0(m_cells),
                   lambda x, m: b * np.ones_like(x), a,
                   t_end=T_END, dt=dt, dt_sample=T_END)
    return res.rho[-1, 0]


def fit_order(hs, errs) -> float:
    keep = np.asarray(errs) > 1e-14
    return float(np.polyfit(np.log(np.asarray(hs)[keep]),
                            np.log(np.asarray(errs)[keep]), 1)[0])


def refinement() -> dict:
    out = {}
    for name, r in REGIMES.items():
        ref_x = final_density(DX_REF_CELLS, DX_DT, r["b"], r["a"])
        errs_x = [w1_cross_grid(final_density(m, DX_DT, r["b"], r["a"]), ref_x)
                  for m in DX_CELLS]
        p = fit_order([1.0 / m for m in DX_CELLS], errs_x)

        ref_t = final_density(DT_CELLS, DT_REF, r["b"], r["a"])
        errs_t = [w1_cross_grid(final_density(DT_CELLS, dt, r["b"], r["a"]),
                                ref_t) for dt in DT_GRID]
        q = fit_order(DT_GRID, errs_t)
        out[name] = {"dx_cells": list(DX_CELLS), "dx_w1": errs_x,
                     "dx_order": p, "dt_grid": list(DT_GRID),
                     "dt_w1": errs_t, "dt_order": q}
        print(f"  {name}: order p(dx) = {p:.2f}, q(dt) = {q:.2f}")
        print("    dx W1:", " ".join(f"{e:.2e}" for e in errs_x))
        print("    dt W1:", " ".join(f"{e:.2e}" for e in errs_t))
    return out


def reflected_em(b: float, a: float, rng) -> np.ndarray:
    m_cells = 128
    rho0 = gaussian_rho0(m_cells)
    cells = rng.choice(m_cells, size=MC_PARTICLES, p=rho0 / rho0.sum())
    x = (cells + rng.uniform(size=MC_PARTICLES)) / m_cells
    n_steps = int(round(T_END / MC_DT))
    sig = np.sqrt(a * MC_DT)
    for _ in range(n_steps):
        x += b * MC_DT + sig * rng.standard_normal(MC_PARTICLES)
        np.abs(x, out=x)                     # reflect at 0
        x = 1.0 - np.abs(1.0 - x)            # reflect at 1
    return x


def mc_reference() -> dict:
    out = {}
    m_cells = 128
    edges = np.linspace(0.0, 1.0, m_cells + 1)
    for name in ("high_peclet_build", "high_peclet_drain"):
        r = REGIMES[name]
        rng = np.random.default_rng(SEED)
        x = reflected_em(r["b"], r["a"], rng)
        half = MC_PARTICLES // 2
        dens = [np.histogram(part, bins=edges)[0] / part.size * m_cells
                for part in (x, x[:half], x[half:])]
        fp = final_density(m_cells, 2e-3, r["b"], r["a"])
        w1_fp_mc = w1_cross_grid(fp, dens[0])
        floor = w1_cross_grid(dens[1], dens[2])
        out[name] = {"w1_fp_vs_mc": w1_fp_mc, "mc_split_half_floor": floor,
                     "particles": MC_PARTICLES, "mc_dt": MC_DT}
        print(f"  {name}: W1(FP, MC) = {w1_fp_mc:.2e} "
              f"(MC split-half floor {floor:.2e})")
    return out


def main() -> None:
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("refinement orders...")
    ref = refinement()
    print("independent reflected-EM Monte Carlo reference...")
    mc = mc_reference()
    (DATA_DIR / "validation.json").write_text(json.dumps({
        "seed": SEED, "t_end": T_END, "regimes": REGIMES,
        "refinement": ref, "mc_reference": mc,
        "wall_clock_s": time.time() - t0}, indent=1))
    print(f"saved {DATA_DIR / 'validation.json'} "
          f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
