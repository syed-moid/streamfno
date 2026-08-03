"""e04 figures, regenerated purely from saved results in data/e04/."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e04"
FIG_DIR = ROOT / "figures"

STYLE = {"pf": ("C2", "o", "particle filter"),
         "gbt": ("C1", "s", "gradient-boosted trees"),
         "logistic": ("C0", "^", "logistic regression"),
         "reactive": ("C3", "x", "reactive threshold")}


def error_figure(res):
    lead = np.array(res["lead_times"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, level in zip(axes, ("light", "moderate", "heavy")):
        lr = res["levels"][level]
        for name, (c, mk, label) in STYLE.items():
            err = [lr[str(h)][name]["error"] for h in lead]
            ci = np.array([lr[str(h)][name]["error_ci"] for h in lead]).T
            ax.errorbar(lead, err, yerr=np.abs(ci - err), color=c, marker=mk,
                        ms=4, capsize=2, lw=1.2, label=label)
        base = [lr[str(h)][name]["base_rate"] for h in lead]
        ax.plot(lead, np.minimum(base, 1.0 - np.array(base)), "k:", lw=1.0,
                label="trivial (base rate)")
        ax.set_xscale("log")
        ax.set_xticks(lead)
        ax.set_xticklabels([f"{x:g}" for x in lead])
        ax.set_xlabel("lead time h")
        ax.set_title(level, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test misclassification error")
    axes[0].legend(fontsize=7)
    fig.suptitle("e04: predictor error vs lead time (val-tuned operating "
                 "points, episode-bootstrap CIs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e04_errors.png", dpi=200)
    plt.close(fig)


def calibration_figure(calib):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, model in zip(axes, ("pf", "gbt")):
        for level, c in (("light", "C0"), ("moderate", "C1"),
                         ("heavy", "C3")):
            d = calib.get(f"{level}_{model}")
            if d:
                ax.plot(d["pred"], d["obs"], f"{c}o-", ms=4, lw=1.0,
                        label=level)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("predicted probability")
        ax.set_title(f"{model} (h = 8)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("observed frequency")
    axes[0].legend(fontsize=8)
    fig.suptitle("e04: calibration on test episodes", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e04_calibration.png", dpi=200)
    plt.close(fig)


def ess_figure(res):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    data = [res["levels"][lvl]["ess_min_per_test_episode"]
            for lvl in ("light", "moderate", "heavy")]
    ax.boxplot(data, tick_labels=["light", "moderate", "heavy"])
    ax.axhline(res["pf_ensemble"] / 2, color="C3", ls="--", lw=1.0,
               label="resampling trigger (M/2)")
    ax.set_ylabel("min ESS per test episode")
    ax.set_title(f"e04: particle-filter degeneracy (M = {res['pf_ensemble']})",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e04_ess.png", dpi=200)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    res = json.loads((DATA_DIR / "results.json").read_text())
    calib = json.loads((DATA_DIR / "calibration.json").read_text())
    error_figure(res)
    calibration_figure(calib)
    ess_figure(res)
    print(f"e04 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
