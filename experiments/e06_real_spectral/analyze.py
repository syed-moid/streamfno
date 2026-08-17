"""e06 analysis: real-data spectra and the real-vs-simulator s table.

Densities are stationary-window, time-and-partition-pooled histograms of
the clipped occupancy X at the collector cadence; the noise floor is
estimated from the partition-parity split (odd vs even partitions).
Caveat recorded with the results: the parity halves share the one burst
realization (the modulator is common to all partitions), so the floor
captures partition-sampling noise, not burst-realization variance; the
Theil-Sen CI on s carries the fit uncertainty.

Fits use the e01 estimators and, by default, the e01 fit range k in
[3, 16]; the realized SNR over that range is stored next to each fit.
"""

import json
from pathlib import Path

import numpy as np

from streamfno.analysis.spectral import (
    cosine_coefficients,
    fft_coefficients,
    fit_decay,
)
from streamfno.kafka.adapter import load_telemetry
from streamfno.kafka.params import RunParams

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e06"
E01_RESULTS = Path(__file__).resolve().parents[2] / "data" / "e01" / "results.npz"

T_WARMUP = 40.0     # normalized units discarded
N_BINS = 64
K_FIT = (3, 16)
LEVELS = ("light", "moderate", "heavy")
# e01's sustained-intensity levels, paired for the table in load order
E01_LEVEL_OF = {"light": "light", "moderate": "moderate",
                "heavy": "near_saturation"}


def density(x_flat: np.ndarray) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    edges[-1] += 1e-9
    counts, _ = np.histogram(x_flat, bins=edges)
    return counts / counts.sum() * N_BINS


def level_spectra(level: str) -> dict:
    run_dir = DATA_DIR / "runs" / level
    man = json.loads((run_dir / "manifest.json").read_text())
    params = RunParams.from_json(json.dumps(man["params"]))
    tel = load_telemetry(run_dir / "lag.npz")
    t_norm = (tel.t_wall - man["t0_wall"]) / params.tau_s
    keep = (t_norm >= T_WARMUP) & (t_norm <= params.t_end)
    x = tel.x(params.budget_b)[keep]

    rho = density(x.ravel())
    rho_a = density(x[:, 0::2].ravel())
    rho_b = density(x[:, 1::2].ravel())
    floor_cos = cosine_coefficients(rho_a - rho_b) / 2.0
    floor_fft = fft_coefficients(rho_a - rho_b) / 2.0
    c_cos = cosine_coefficients(rho)
    c_fft = fft_coefficients(rho)
    f_cos = fit_decay(c_cos, *K_FIT)
    f_fft = fit_decay(c_fft, *K_FIT)
    k_lo, k_hi = K_FIT
    snr_cos = float(np.median(c_cos[k_lo:k_hi + 1]
                              / np.maximum(floor_cos[k_lo:k_hi + 1], 1e-300)))
    snr_fft = float(np.median(c_fft[k_lo:k_hi + 1]
                              / np.maximum(floor_fft[k_lo:k_hi + 1], 1e-300)))
    return dict(rho=rho, c_cos=c_cos, c_fft=c_fft,
                floor_cos=floor_cos, floor_fft=floor_fft,
                f_cos=f_cos, f_fft=f_fft, snr_cos=snr_cos, snr_fft=snr_fft,
                mean_x=float(x.mean()), n_samples=int(x.size))


def main() -> None:
    out_arrays = {}
    fits = {}
    print(f"{'level':<10} {'s_cos':>22} {'s_fft':>22} {'SNR cos':>8}")
    for level in LEVELS:
        r = level_spectra(level)
        for tag in ("rho", "c_cos", "c_fft", "floor_cos", "floor_fft"):
            out_arrays[f"{tag}_{level}"] = r[tag]
        for tag in ("cos", "fft"):
            f = r[f"f_{tag}"]
            fits[f"{level}_real_{tag}"] = [f.s, f.s_lo, f.s_hi]
        fits[f"{level}_meta"] = {"snr_cos": r["snr_cos"],
                                 "snr_fft": r["snr_fft"],
                                 "mean_x": r["mean_x"],
                                 "n_samples": r["n_samples"]}
        fc, ff = r["f_cos"], r["f_fft"]
        print(f"{level:<10} {fc.s:6.2f} [{fc.s_lo:5.2f},{fc.s_hi:5.2f}]"
              f"    {ff.s:6.2f} [{ff.s_lo:5.2f},{ff.s_hi:5.2f}]"
              f"  {r['snr_cos']:8.1f}")

    # sim-side numbers for the table
    table = {}
    if E01_RESULTS.exists():
        with np.load(E01_RESULTS) as f:
            e01 = json.loads(str(f["fits_json"]))
        for level in LEVELS:
            e01_level = E01_LEVEL_OF[level]
            table[level] = {
                "real_cos": fits[f"{level}_real_cos"],
                "real_fft": fits[f"{level}_real_fft"],
                "sim_cos": e01[f"{e01_level}_sim_cos"],
                "sim_fft": e01[f"{e01_level}_sim_fft"],
            }
        print("\nreal vs e01 sim (s, [lo, hi]):")
        for level, row in table.items():
            print(f"  {level:<10} real_cos={row['real_cos'][0]:5.2f} "
                  f"sim_cos={row['sim_cos'][0]:5.2f}   "
                  f"real_fft={row['real_fft'][0]:5.2f} "
                  f"sim_fft={row['sim_fft'][0]:5.2f}")

    np.savez_compressed(
        DATA_DIR / "spectra.npz",
        levels=np.array(LEVELS), k_fit=np.array(K_FIT),
        n_bins=np.array(N_BINS), t_warmup=np.array(T_WARMUP),
        fits_json=np.array(json.dumps(fits)),
        table_json=np.array(json.dumps(table)),
        **out_arrays,
    )
    print(f"\nsaved {DATA_DIR / 'spectra.npz'}")


if __name__ == "__main__":
    main()
