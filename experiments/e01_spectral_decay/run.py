"""e01: spectral decay of the lag density at three load levels.

Simulates, averages stationary-window densities, computes cosine- and
FFT-basis spectra with robust tail fits, and saves everything under
data/e01/.  Figures are regenerated separately by figures.py.
"""

import json
import time
from pathlib import Path

import numpy as np

from streamfno.analysis.spectral import cosine_coefficients, fft_coefficients, fit_decay
from streamfno.matching import matched_pde
from streamfno.sim import SimConfig, simulate

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e01"

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
T_END = 200.0
T_STATIONARY = 80.0
# Stated fit range for the tail exponent.  The upper end is set by the
# instrument floor of the empirical spectra: a coherent plateau of residual
# tau-leap bias, measured to scale ~linearly with the step (6.7e-3 * c0 at
# tau ~ 1.9e-3, 1.1e-3 * c0 at tau ~ 4.8e-4; independent of N, so it is not
# a mean-field finite-N effect).  At the production step (tau_jump_cap =
# 1.25 below) the moderate-load signal crosses that plateau near k ~ 20, so
# the fit stops at 16.  The noise-free PDE reference is fitted over the same
# range for comparability.
K_FIT = (3, 16)

BASE = dict(
    n_partitions=1000, buffer_depth=256, t_end=T_END, dt_sample=1.0,
    n_bins=128, mode="diffusive", method="tau_leap", arrival="poisson",
    mu0=0.7, a=0.04, n_brokers=1, init_x0=0.35, init_sd=0.10,
    # 4x smaller leap step than the default: pushes the coherent tau-bias
    # plateau of the spectra down to ~1e-3 * c0 (see K_FIT note)
    tau_jump_cap=1.25,
)

LOADS = {
    "light": 0.40,
    "moderate": 0.65,
    "near_saturation": 0.80,
}


def averaged_density(lam):
    """Stationary-window empirical density on the fixed grid, averaged over
    time (t >= T_STATIONARY) and seeds.

    Returns (rho, rho_half_a, rho_half_b, mean_flux): the full average plus
    two independent half-averages (seed split) whose difference estimates the
    sampling-noise floor of the spectra.
    """
    hists = []
    flux = []
    for seed in SEEDS:
        cfg = SimConfig(seed=seed, lam=lam, **BASE)
        res = simulate(cfg)
        mask = res.times >= T_STATIONARY
        hists.append(res.hist[mask].sum(axis=0))
        flux.append(res.boundary_flux_total()[mask].mean())

    def to_density(counts):
        counts = np.asarray(counts, dtype=float)
        return counts / counts.sum() * counts.size

    rho = to_density(np.sum(hists, axis=0))
    half = len(SEEDS) // 2
    rho_a = to_density(np.sum(hists[:half], axis=0))
    rho_b = to_density(np.sum(hists[half:], axis=0))
    return rho, rho_a, rho_b, float(np.mean(flux))


def pde_density(lam):
    """Matched PDE stationary density restricted to the histogram grid."""
    cfg = SimConfig(seed=0, lam=lam, **BASE)
    res = matched_pde(cfg, m_cells=512, dt=2e-3, dt_sample=T_END)
    return res.rho[-1, 0].reshape(BASE["n_bins"], -1).mean(axis=1)


def spectra_and_fits(rho):
    c_cos = cosine_coefficients(rho)
    c_fft = fft_coefficients(rho)
    f_cos = fit_decay(c_cos, *K_FIT)
    f_fft = fit_decay(c_fft, *K_FIT)
    return c_cos, c_fft, f_cos, f_fft


def main():
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    fit_rows = []
    for name, lam in LOADS.items():
        print(f"load {name!r} (lam={lam}) ...")
        rho, rho_a, rho_b, jb = averaged_density(lam)
        rho_pde = pde_density(lam)
        c_cos, c_fft, f_cos, f_fft = spectra_and_fits(rho)
        _, _, g_cos, g_fft = spectra_and_fits(rho_pde)
        # noise floor of the full-average spectra: half-average difference / 2
        floor_cos = cosine_coefficients(rho_a - rho_b) / 2.0
        floor_fft = fft_coefficients(rho_a - rho_b) / 2.0
        out[name] = dict(rho=rho, rho_pde=rho_pde, c_cos=c_cos, c_fft=c_fft,
                         floor_cos=floor_cos, floor_fft=floor_fft,
                         flux=jb, lam=lam)
        fit_rows.append((name, f_cos, f_fft, g_cos, g_fft))
        k_lo, k_hi = K_FIT
        snr = np.median(c_cos[k_lo:k_hi + 1] / np.maximum(floor_cos[k_lo:k_hi + 1],
                                                          1e-300))
        print(f"  mean J_B = {jb:.2e}; median cos SNR over fit range = {snr:.1f}")
        print(f"  sim  s_cos = {f_cos.s:.2f} [{f_cos.s_lo:.2f}, {f_cos.s_hi:.2f}]"
              f"   s_fft = {f_fft.s:.2f} [{f_fft.s_lo:.2f}, {f_fft.s_hi:.2f}]")
        print(f"  pde  s_cos = {g_cos.s:.2f} [{g_cos.s_lo:.2f}, {g_cos.s_hi:.2f}]"
              f"   s_fft = {g_fft.s:.2f} [{g_fft.s_lo:.2f}, {g_fft.s_hi:.2f}]")

    arrays = {}
    fits = {}
    for name, f_cos, f_fft, g_cos, g_fft in fit_rows:
        for tag, f in (("sim_cos", f_cos), ("sim_fft", f_fft),
                       ("pde_cos", g_cos), ("pde_fft", g_fft)):
            fits[f"{name}_{tag}"] = [f.s, f.s_lo, f.s_hi]
        arrays[f"rho_{name}"] = out[name]["rho"]
        arrays[f"rho_pde_{name}"] = out[name]["rho_pde"]
        arrays[f"c_cos_{name}"] = out[name]["c_cos"]
        arrays[f"c_fft_{name}"] = out[name]["c_fft"]
        arrays[f"floor_cos_{name}"] = out[name]["floor_cos"]
        arrays[f"floor_fft_{name}"] = out[name]["floor_fft"]
        arrays[f"flux_{name}"] = out[name]["flux"]
    np.savez_compressed(
        DATA_DIR / "results.npz",
        loads=np.array(list(LOADS.keys())), lams=np.array(list(LOADS.values())),
        k_fit=np.array(K_FIT), fits_json=np.array(json.dumps(fits)),
        config_json=np.array(json.dumps(BASE)), **arrays,
    )
    print(f"e01 done in {time.time() - t0:.1f}s; results in {DATA_DIR}")


if __name__ == "__main__":
    main()
