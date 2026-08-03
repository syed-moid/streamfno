"""e00 figures, regenerated purely from saved results in data/e00/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e00"
FIG_DIR = ROOT / "figures"


def w1_scaling_figures(s):
    d = s["distances"]  # (N, B, seed, time)
    ns, bs, times = s["ns"], s["bs"], s["eval_times"]
    for axis_name, xvals, sel in (("N", ns, "n"), ("B", bs, "b")):
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        # curves: for vs-N, one line per B (mean over seeds); vice versa
        other = bs if sel == "n" else ns
        for j, o in enumerate(other):
            m = (d[:, j, :, -1] if sel == "n" else d[j, :, :, -1])
            label = (f"B={o}" if sel == "n" else f"N={o}")
            ax.errorbar(xvals, m.mean(axis=1), yerr=m.std(axis=1),
                        marker="o", capsize=3, label=label)
        # reference slopes
        x0 = np.array([xvals[0], xvals[-1]], dtype=float)
        y0 = d[..., -1].mean(axis=2).max() * (x0 / x0[0]) ** -0.5
        ax.plot(x0, y0, "k--", lw=0.8, label=r"slope $-1/2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(axis_name)
        ax.set_ylabel(f"W1 to matched PDE at t={times[-1]:.0f}")
        ax.set_title(f"e00: W1 vs {axis_name} (diffusive mode)")
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"e00_w1_vs_{axis_name}.png", dpi=200)
        plt.close(fig)


def fluid_contrast_figure():
    f = np.load(DATA_DIR / "fluid_contrast.npz")
    times = f["times"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    edges = f["sim_bin_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    for ax, k in zip(axes[:2], (len(times) // 2 - 1, 3 * len(times) // 4)):
        ax.stairs(f["sim_density"][k], edges, fill=True, alpha=0.35,
                  color="C0", label="simulator (fluid)")
        ax.plot(f["x_centers"], f["transport_rho"][k], "C2", lw=1.5,
                label="transport (a=0)")
        ax.plot(f["x_centers"], f["diffusive_rho"][k], "C3--", lw=1.5,
                label="diffusion (a=0.04)")
        ax.set_xlim(0, max(0.8, centers[np.argmax(f["sim_density"][k])] + 0.3))
        ax.set_xlabel("x")
        ax.set_title(f"t = {times[k]:.2f}")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=8)
    axes[2].plot(times, f["w1_transport"], "C2-o", ms=3, label="W1 to transport")
    axes[2].plot(times, f["w1_diffusive"], "C3--s", ms=3, label="W1 to diffusion")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("W1")
    axes[2].set_title("fluid mode tracks transport")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    fig.suptitle("e00 contrast: N-scaling alone (fluid) does not produce diffusion",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e00_fluid_contrast.png", dpi=200)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    s = np.load(DATA_DIR / "summary.npz")
    w1_scaling_figures(s)
    fluid_contrast_figure()
    print(f"e00 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
