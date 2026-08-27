"""e06 figures (paper F3), regenerated purely from saved results.

One 2x3 grid (rows: cosine / FFT basis, columns: load levels): the real
busy-conditional spectrum with its fitted decay, the real full-density
spectrum with the boundary-atom flat level it plateaus at, the e01
simulator spectrum at the paired load level, and the partition-parity
noise floor.  Reads data/e06/spectra.npz and data/e01/results.npz only.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from streamfno.analysis.spectral import cosine_coefficients, fft_coefficients

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e06"
E01_RESULTS = ROOT / "data" / "e01" / "results.npz"
FIG_DIR = ROOT / "figures"

COLORS = {"light": "C0", "moderate": "C1", "heavy": "C3"}
E01_LEVEL_OF = {"light": "light", "moderate": "moderate",
                "heavy": "near_saturation"}
BUSY_COEFFS = {"cos": cosine_coefficients, "fft": fft_coefficients}


def panel(ax, basis, level, f, fits, e01, e01_fits):
    color = COLORS[level]
    k_min, k_max = f["k_fit"]
    meta = fits[f"{level}_meta"]
    atom_w = 1.0 - meta["busy_fraction"]

    # real full density: plateaus at the empty-queue atom's flat level
    c_full = f[f"c_{basis}_{level}"]
    k = np.arange(c_full.size)
    ax.loglog(k[1:], c_full[1:] / c_full[0], ".", color="0.55", ms=4,
              label="real, full density")
    ax.axhline(atom_w, color="0.55", ls="--", lw=1.0,
               label=f"idle-atom level $1-p_{{busy}}$ = {atom_w:.2f}")

    # real busy-conditional density: the interior-smoothness spectrum
    c_busy = BUSY_COEFFS[basis](f[f"rho_busy_{level}"])
    kb = np.arange(c_busy.size)
    ax.loglog(kb[1:], c_busy[1:] / c_busy[0], ".", color=color, ms=5,
              label="real, busy-conditional")
    s, s_lo, s_hi = fits[f"{level}_busy_{basis}"]
    kk = np.arange(k_min, k_max + 1, dtype=float)
    anchor = c_busy[k_min] / c_busy[0]
    ax.loglog(kk, anchor * (kk / k_min) ** (-s), "k-", lw=1.2,
              label=f"busy fit s = {s:.2f} [{s_lo:.2f}, {s_hi:.2f}]")

    # e01 simulator spectrum at the paired sustained-intensity level
    if e01 is not None:
        e01_level = E01_LEVEL_OF[level]
        c_sim = e01[f"c_{basis}_{e01_level}"]
        ks = np.arange(c_sim.size)
        s_sim = e01_fits[f"{e01_level}_sim_{basis}"][0]
        ax.loglog(ks[1:], c_sim[1:] / c_sim[0], color=color, lw=1.0,
                  alpha=0.45, label=f"simulator (s = {s_sim:.2f})")

    floor = f[f"floor_{basis}_{level}"]
    ax.loglog(k[1:], np.maximum(floor[1:], 1e-12) / c_full[0], color="0.7",
              lw=1.0, ls=":", label="parity noise floor")

    ax.axvspan(k_min, k_max, color="0.92", zorder=0)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, loc="lower left")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    f = np.load(DATA_DIR / "spectra.npz")
    levels = [str(x) for x in f["levels"]]
    fits = json.loads(str(f["fits_json"]))
    e01, e01_fits = None, None
    if E01_RESULTS.exists():
        e01 = np.load(E01_RESULTS)
        e01_fits = json.loads(str(e01["fits_json"]))

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), sharex="row",
                             sharey=True)
    for j, level in enumerate(levels):
        meta = fits[f"{level}_meta"]
        for i, basis in enumerate(("cos", "fft")):
            panel(axes[i, j], basis, level, f, fits, e01, e01_fits)
        axes[0, j].set_title(
            f"{level} (busy fraction {meta['busy_fraction']:.2f})", fontsize=9)
        axes[1, j].set_xlabel("k")
    axes[0, 0].set_ylabel(r"cosine basis: $|\hat c_k| / |\hat c_0|$")
    axes[1, 0].set_ylabel(r"FFT basis: $|\hat c_k| / |\hat c_0|$")
    k_min, k_max = f["k_fit"]
    fig.suptitle(
        "e06: real-cluster spectral decay -- boundary atom vs busy interior "
        f"(fit range k in [{k_min}, {k_max}])", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e06_spectra.png", dpi=200)
    plt.close(fig)

    # print-compact 1x3 variant: cosine row only (the valid basis; the
    # FFT-pinning statement stays in the text), for the space-limited
    # main paper
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for j, level in enumerate(levels):
        meta = fits[f"{level}_meta"]
        panel(axes[j], "cos", level, f, fits, e01, e01_fits)
        axes[j].set_title(
            f"{level} (busy fraction {meta['busy_fraction']:.2f})", fontsize=9)
        axes[j].set_xlabel("k")
    axes[0].set_ylabel(r"cosine basis: $|\hat c_k| / |\hat c_0|$")
    fig.suptitle(
        "e06: real-cluster spectral decay -- boundary atom vs busy interior "
        f"(fit range k in [{k_min}, {k_max}])", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e06_spectra_cos.png", dpi=200)
    plt.close(fig)
    print(f"e06 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
