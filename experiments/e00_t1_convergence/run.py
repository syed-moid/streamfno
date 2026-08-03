"""e00: T1 convergence — empirical lag density vs matched Fokker-Planck
solution under joint (N, B) scaling, with a fluid-mode transport contrast.

Simulates, computes distances, and saves everything under data/e00/.
Figures are regenerated separately by figures.py from the saved results.
"""

import json
import time
from pathlib import Path

import numpy as np

from streamfno.analysis.distances import w1_lattice_vs_density
from streamfno.matching import matched_pde
from streamfno.sim import SimConfig, simulate

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e00"

NS = [50, 200, 1000]
BS = [10, 50, 100, 200]
SEEDS = [0, 1, 2, 3, 4]
EVAL_TIMES = [2.0, 5.0, 10.0, 20.0, 40.0]
T_END = 40.0

BASE = dict(
    t_end=T_END, dt_sample=1.0, n_bins=128, mode="diffusive", method="tau_leap",
    arrival="poisson", lam=0.55, mu0=0.7, a=0.04, n_brokers=1,
    init_x0=0.35, init_sd=0.10,
)


def pde_reference():
    """Matched PDE solution plus a grid/dt refinement check."""
    cfg = SimConfig(n_partitions=1, buffer_depth=100, seed=0, **BASE)
    coarse = matched_pde(cfg, m_cells=400, dt=2e-3, dt_sample=1.0)
    fine = matched_pde(cfg, m_cells=800, dt=1e-3, dt_sample=1.0)
    idx = [int(round(t)) for t in EVAL_TIMES]
    from streamfno.analysis.distances import density_to_weights, w1_weights
    refine_w1 = []
    for k in idx:
        xc, wc = density_to_weights(coarse.rho[k, 0])
        xf, wf = density_to_weights(fine.rho[k, 0])
        refine_w1.append(w1_weights(xc, wc, xf, wf))
    return coarse, np.array(refine_w1)


def sweep(pde):
    d = np.zeros((len(NS), len(BS), len(SEEDS), len(EVAL_TIMES)))
    t_idx = [int(round(t)) for t in EVAL_TIMES]  # dt_sample = 1.0
    for i, n in enumerate(NS):
        for j, b in enumerate(BS):
            for k, seed in enumerate(SEEDS):
                cfg = SimConfig(n_partitions=n, buffer_depth=b, seed=seed, **BASE)
                res = simulate(cfg)
                for ell, ti in enumerate(t_idx):
                    d[i, j, k, ell] = w1_lattice_vs_density(
                        res.lattice_hist[ti], pde.rho[ti, 0]
                    )
                if n == 1000 and b == 100 and seed == 0:
                    res.save(DATA_DIR / "run_N1000_B100_seed0.npz")
            print(f"  N={n:5d} B={b:4d}: "
                  f"W1(t=40) = {d[i, j, :, -1].mean():.4f} "
                  f"+/- {d[i, j, :, -1].std():.4f}")
    return d


def fluid_contrast():
    """N-scaling alone: fluid mode vs transport and vs diffusion."""
    cfg = SimConfig(
        n_partitions=1000, buffer_depth=200, seed=0,
        **{**BASE, "mode": "fluid", "t_end": 2.0, "dt_sample": 0.25},
    )
    res = simulate(cfg)
    transport = matched_pde(cfg, m_cells=800, dt=5e-4, dt_sample=0.25, a_override=0.0)
    diffusive = matched_pde(cfg, m_cells=800, dt=5e-4, dt_sample=0.25,
                            a_override=BASE["a"])
    w_tr, w_df = [], []
    for k in range(len(res.times)):
        w_tr.append(w1_lattice_vs_density(res.lattice_hist[k], transport.rho[k, 0]))
        w_df.append(w1_lattice_vs_density(res.lattice_hist[k], diffusive.rho[k, 0]))
    np.savez_compressed(
        DATA_DIR / "fluid_contrast.npz",
        times=res.times, w1_transport=np.array(w_tr), w1_diffusive=np.array(w_df),
        sim_density=res.densities(), sim_bin_edges=res.bin_edges,
        transport_rho=transport.rho[:, 0, :], diffusive_rho=diffusive.rho[:, 0, :],
        x_centers=transport.x_centers,
    )
    return np.array(w_tr), np.array(w_df)


def loglog_slope(x, y):
    return np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0]


def main():
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("solving matched PDE reference (with refinement check)...")
    pde, refine_w1 = pde_reference()
    pde.save(DATA_DIR / "pde_reference.npz")
    print(f"  PDE refinement W1 (M=400,dt=2e-3 vs M=800,dt=1e-3): "
          f"max {refine_w1.max():.2e}")

    print("sweeping N x B x seeds (diffusive mode)...")
    d = sweep(pde)

    print("fluid-mode contrast (N=1000, B=200)...")
    w_tr, w_df = fluid_contrast()
    print(f"  final W1 to transport {w_tr[-1]:.4f} vs to diffusion {w_df[-1]:.4f}")

    mean_final = d[..., -1].mean(axis=2)
    slope_n = loglog_slope(NS, mean_final[:, -1])
    slope_b = loglog_slope(BS, mean_final[-1, :])
    np.savez_compressed(
        DATA_DIR / "summary.npz",
        distances=d, ns=np.array(NS), bs=np.array(BS), seeds=np.array(SEEDS),
        eval_times=np.array(EVAL_TIMES), pde_refinement_w1=refine_w1,
        slope_n=slope_n, slope_b=slope_b,
        config_json=np.array(json.dumps(BASE)),
    )
    print(f"log-log slope of mean W1(t=40): vs N (at B=200) {slope_n:.2f}, "
          f"vs B (at N=1000) {slope_b:.2f}")
    print(f"e00 done in {time.time() - t0:.1f}s; results in {DATA_DIR}")


if __name__ == "__main__":
    main()
