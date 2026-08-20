"""e10 figures (paper F6), regenerated purely from saved results
(data/e10/results.json): the accuracy-vs-cost frontier at the mid
horizon, and the amortized-cost crossover."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def envelope_panel(ax):
    """Conceptual 2x2 applicability envelope: which forecasting engine a
    problem class warrants, by intrinsic predictability (Gate 1, T3's
    skill horizon) x computational burden of the governing PDE (Gate 2,
    the measured amortization criterion).  Located empirically by panel
    (a) and Phase C; no measured axes."""
    quads = [
        (0, 0, "0.93", "simple decision rule\n(nothing to forecast,\n"
                       "nothing costly to solve)"),
        (1, 0, "#dce9f6", "classical solver\n(forecast skill exists;\n"
                          "the PDE is cheap)"),
        (0, 1, "0.88", "no computation recovers\nmissing information\n"
                       "(beyond the skill horizon)"),
        (1, 1, "#dff0dc", "neural operator\npotentially useful\n"
                          "(skill exists; solves are\nthe bottleneck)"),
    ]
    for qx, qy, color, label in quads:
        ax.add_patch(plt.Rectangle((qx, qy), 1, 1, facecolor=color,
                                   edgecolor="0.5", lw=0.8))
        y_label = qy + (0.68 if (qx, qy) == (1, 0) else 0.5)
        ax.text(qx + 0.5, y_label, label, ha="center", va="center",
                fontsize=7)
    ax.plot([1.5], [0.22], "o", color="C3", ms=7)
    ax.annotate("this system (1-D FP, h < H*; e09/e10)", (1.5, 0.22),
                textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=7, color="C3")
    ax.annotate("", xy=(1.98, -0.13), xytext=(0.02, -0.13),
                arrowprops=dict(arrowstyle="->", color="0.3"),
                annotation_clip=False)
    ax.annotate("", xy=(-0.09, 1.98), xytext=(-0.09, 0.02),
                arrowprops=dict(arrowstyle="->", color="0.3"),
                annotation_clip=False)
    ax.set_xlabel("intrinsic predictability (h / H*)")
    ax.set_ylabel("computational burden of the governing PDE")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("(b) applicability envelope (conceptual; located by (a) "
                 "and Phase C)", fontsize=9)


def main():
    FIG_DIR.mkdir(exist_ok=True)
    r = json.loads((DATA_DIR / "results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    frontier_panel(axes[0], r)
    envelope_panel(axes[1])
    fig.suptitle("e10: DCT spectral surrogate vs classical Chang-Cooper "
                 "solver", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e10_frontier.png", dpi=200)
    plt.close(fig)
    print(f"e10 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
