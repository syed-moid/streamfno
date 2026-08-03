"""e01 figures, regenerated purely from saved results in data/e01/."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e01"
FIG_DIR = ROOT / "figures"

COLORS = {"light": "C0", "moderate": "C1", "near_saturation": "C3"}


def densities_figure(f, loads, lams):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    n_bins = f["rho_light"].size
    x = (np.arange(n_bins) + 0.5) / n_bins
    for name, lam in zip(loads, lams):
        ax.semilogy(x, np.maximum(f[f"rho_{name}"], 1e-4), COLORS[name],
                    label=f"{name} (lam={lam})")
        ax.semilogy(x, np.maximum(f[f"rho_pde_{name}"], 1e-4),
                    COLORS[name], ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("x (normalized lag)")
    ax.set_ylabel("density")
    ax.set_title("e01: stationary lag densities (solid: sim, dashed: PDE)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e01_densities.png", dpi=200)
    plt.close(fig)


def spectra_figure(f, loads, lams, fits, basis):
    key = f"c_{basis}"
    k_min, k_max = f["k_fit"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, name, lam in zip(axes, loads, lams):
        c = f[f"{key}_{name}"]
        k = np.arange(c.size)
        ax.loglog(k[1:], c[1:] / c[0], f"{COLORS[name]}.", ms=4,
                  label="sim coefficients")
        floor = f[f"floor_{basis}_{name}"]
        ax.loglog(k[1:], np.maximum(floor[1:], 1e-12) / c[0], color="0.6",
                  lw=1.0, ls=":", label="sampling-noise floor")
        s, s_lo, s_hi = fits[f"{name}_sim_{basis}"]
        kk = np.arange(k_min, k_max + 1, dtype=float)
        anchor = c[k_min] / c[0]
        ax.loglog(kk, anchor * (kk / k_min) ** (-s), "k-", lw=1.2,
                  label=f"fit s = {s:.2f} [{s_lo:.2f}, {s_hi:.2f}]")
        sp = fits[f"{name}_pde_{basis}"][0]
        ax.set_title(f"{name} (lam={lam}); PDE s = {sp:.2f}", fontsize=9)
        ax.axvspan(k_min, k_max, color="0.9", zorder=0)
        ax.set_xlabel("k")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$|\hat c_k| / |\hat c_0|$")
    fig.suptitle(f"e01 spectral decay, {basis} basis "
                 f"(fit range k in [{k_min}, {k_max}])", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"e01_spectra_{basis}.png", dpi=200)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    f = np.load(DATA_DIR / "results.npz")
    loads = [str(x) for x in f["loads"]]
    lams = f["lams"]
    fits = json.loads(str(f["fits_json"]))
    densities_figure(f, loads, lams)
    spectra_figure(f, loads, lams, fits, "cos")
    spectra_figure(f, loads, lams, fits, "fft")
    print(f"e01 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
