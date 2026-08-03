"""e03 figures, regenerated purely from saved results in data/e03/."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e03"
FIG_DIR = ROOT / "figures"


def bound_figure(lc):
    h = lc["lead_times"]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for level, color in (("light", "C0"), ("moderate", "C1"), ("heavy", "C3")):
        d = lc[f"{level}_delta_min"]
        ax.plot(h, d, f"{color}o-", ms=4, label=fr"$\delta_{{\min}}(h)$ {level}")
        ax.fill_between(h, 0.0, d, color=color, alpha=0.10)
    ax.axhline(0.5, color="0.5", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks(h)
    ax.set_xticklabels([f"{x:g}" for x in h])
    ax.set_xlabel("lead time h")
    ax.set_ylabel("prediction error lower bound")
    ax.set_ylim(0, 0.55)
    ax.set_title("e03: two-point lower bound on backpressure prediction error")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e03_bound.png", dpi=200)
    plt.close(fig)


def divergence_figure(div):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    regimes = list(div.keys())
    for i, name in enumerate(regimes):
        vals = div[name]["lam_plus"]
        ax.scatter([i] * len(vals), vals, s=18,
                   label=f"{name} (lam={div[name]['lam']:.2f})")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels(regimes)
    ax.set_ylabel(r"finite-time exponent $\lambda_+$")
    ax.set_title("e03: twin-run divergence rates by drift regime")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e03_divergence.png", dpi=200)
    plt.close(fig)


def margins_figure(mg):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
    for ax, level in zip(axes, ("light", "moderate", "heavy")):
        m = mg[f"{level}_margins"]
        m = m[np.isfinite(m)]
        ax.hist(m, bins=40, color="C0", alpha=0.8)
        ax.axvline(2.0, color="C3", ls="--", lw=1.2,
                   label="informative zone (< 2)")
        ax.set_title(f"{level} (base rate "
                     f"{float(mg[f'{level}_base_rate']):.2f})", fontsize=9)
        ax.set_xlabel("whitened margin to opposite label")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("decision points")
    fig.suptitle("e03: observation-metric margins to the event boundary "
                 "(test episodes, h = 8)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e03_margins.png", dpi=200)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    lc = np.load(DATA_DIR / "lecam.npz", allow_pickle=True)
    div = json.loads((DATA_DIR / "divergence.json").read_text())
    mg = np.load(DATA_DIR / "margins.npz")
    bound_figure(lc)
    divergence_figure(div)
    margins_figure(mg)
    print(f"e03 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
