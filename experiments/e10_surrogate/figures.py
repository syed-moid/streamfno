"""e10 figures (paper F6), regenerated purely from saved results
(data/e10/results.json): the accuracy-vs-cost frontier at the mid
horizon, and the amortized-cost crossover."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e10"
FIG_DIR = ROOT / "figures"

H_KEY = "8.0"


def frontier_panel(ax, r):
    lat = r["latency"]
    dts = r["dt_coarse"]
    cls_x = [lat[f"classical_dt{dt:g}_ms"] for dt in dts[1:]]
    cls_y = [r["classical_accuracy"][f"{dt:g}"]["w1_mean"][H_KEY]
             for dt in dts[1:]]
    ax.loglog(cls_x, cls_y, "ko-", ms=5, label="classical, coarsened dt")
    for dt, xx, yy in zip(dts[1:], cls_x, cls_y):
        ax.annotate(f"dt={dt:g}", (xx, yy), textcoords="offset points",
                    xytext=(5, 4), fontsize=6)
    ref_ms = lat[f"classical_dt{dts[0]:g}_ms"]
    ax.axvline(ref_ms, color="0.6", ls=":", lw=1.0,
               label=f"reference solve (dt={dts[0]:g}): {ref_ms:.0f} ms")

    ks = sorted(int(k) for k in r["surrogate_accuracy"])
    for tag, marker, ls, label in (
            ("single", "o", "-", "surrogate, single (batch 1)"),
            (f"batch{r['batch_ensemble']}_per_member", "s", "--",
             f"surrogate, batched x{r['batch_ensemble']} (per member)")):
        xs = [lat[f"surrogate_K{k}_{tag}_ms"] for k in ks]
        ys = [r["surrogate_accuracy"][str(k)]["w1_mean"][H_KEY] for k in ks]
        ax.loglog(xs, ys, marker=marker, ls=ls, color="C0", ms=5,
                  label=label)
        for k, xx, yy in zip(ks, xs, ys):
            ax.annotate(f"K={k}", (xx, yy), textcoords="offset points",
                        xytext=(4, -9), fontsize=6, color="C0")
    ax.set_xlabel(f"wall latency per forecast, h = {r['h_latency']:g} (ms)")
    ax.set_ylabel(rf"$W_1$ vs reference at h = {H_KEY.rstrip('.0')}")
    ax.set_title("(a) accuracy-cost frontier (single-threaded)", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5)


def crossover_panel(ax, r):
    ks = sorted(int(k) for k in r["surrogate_accuracy"])
    best_k = min(ks, key=lambda k:
                 r["surrogate_accuracy"][str(k)]["w1_mean"][H_KEY])
    row = r["crossover"][str(best_k)]
    train_s = r["training"][str(best_k)]["train_cpu_s"]
    n = np.logspace(0, 6, 200)
    cls_ms = row["classical_ms_at_matched_accuracy"]
    ax.loglog(n, n * cls_ms / 1e3, "k-",
              label=f"classical at matched accuracy ({cls_ms:.0f} ms/eval)")
    for tag, ls in (("single", "-"), ("batched", "--")):
        cost = train_s + n * row[f"{tag}_ms"] / 1e3
        ax.loglog(n, cost, ls, color="C0",
                  label=f"surrogate K={best_k} {tag} "
                        f"({row[f'{tag}_ms']:.2f} ms/eval + "
                        f"{train_s:.0f} s training)")
        n_star = row[f"{tag}_crossover_evals"]
        if n_star is not None:
            ax.axvline(n_star, color="C0", ls=":", lw=1.0, alpha=0.6)
            ax.annotate(f"N*={n_star:.0f}", (n_star, train_s * 2),
                        fontsize=7, color="C0", rotation=90,
                        textcoords="offset points", xytext=(3, 0))
    if all(row[f"{tag}_crossover_evals"] is None
           for tag in ("single", "batched")):
        ax.text(0.97, 0.35, "no crossover: the classical solver is\n"
                            "faster at matched accuracy at every N",
                transform=ax.transAxes, ha="right", va="center", fontsize=7)
    ax.set_xlabel("forecast evaluations N")
    ax.set_ylabel("total CPU cost (s)")
    ax.set_title(f"(b) amortization at matched accuracy (K = {best_k})",
                 fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6.5, loc="upper left")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    r = json.loads((DATA_DIR / "results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    frontier_panel(axes[0], r)
    crossover_panel(axes[1], r)
    fig.suptitle("e10: DCT spectral surrogate vs classical Chang-Cooper "
                 "solver", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e10_frontier.png", dpi=200)
    plt.close(fig)
    print(f"e10 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
