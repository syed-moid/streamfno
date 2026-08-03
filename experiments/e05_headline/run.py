"""e05: the headline figure -- predictor error vs lead time over the
computed impossibility region, three load-level panels, plus the
gap-to-the-ceiling table.

Pure assembly: reads e03 (bound curves, sanity) and e04 (predictor metrics)
outputs; no simulation.  `make e05` regenerates everything end to end from
saved results.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E03 = ROOT / "data" / "e03"
E04 = ROOT / "data" / "e04"
DATA_DIR = ROOT / "data" / "e05"
FIG_DIR = ROOT / "figures"

STYLE = {"pf": ("C2", "o", "particle filter"),
         "gbt": ("C1", "s", "gradient-boosted trees"),
         "logistic": ("C0", "^", "logistic regression"),
         "reactive": ("C3", "x", "reactive threshold")}
DELTA_TARGET = 0.2


def headline_figure(lead, bound, res, sanity):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharey=True)
    for ax, level in zip(axes, ("light", "moderate", "heavy")):
        d = bound[f"{level}_delta_min"]
        ok = sanity[level]["monotone_nondecreasing"]
        region_label = ("no predictor with this telemetry can enter"
                        if ok else "computed lower bound (sanity flag; see log)")
        ax.fill_between(lead, 0.0, d, color="0.82", zorder=0,
                        label=region_label)
        ax.plot(lead, d, "k-", lw=1.4, zorder=1,
                label=r"$\delta_{\min}(h)$ two-point bound")
        lr = res["levels"][level]
        for name, (c, mk, label) in STYLE.items():
            err = np.array([lr[str(h)][name]["error"] for h in lead])
            ci = np.array([lr[str(h)][name]["error_ci"] for h in lead])
            ax.plot(lead, err, color=c, marker=mk, ms=4, lw=1.3, label=label,
                    zorder=2)
            ax.fill_between(lead, ci[:, 0], ci[:, 1], color=c, alpha=0.15,
                            zorder=1)
        base = np.array([lr[str(h)]["pf"]["base_rate"] for h in lead])
        ax.plot(lead, np.minimum(base, 1 - base), "k:", lw=0.9,
                label="trivial (predict majority)")
        ax.set_xscale("log")
        ax.set_xticks(lead)
        ax.set_xticklabels([f"{x:g}" for x in lead])
        ax.set_xlabel("lead time h (normalized)")
        ax.set_title(f"{level} load", fontsize=10)
        ax.set_ylim(0, 0.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("prediction error at lead h")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Backpressure predictability: practical predictors vs the "
                 "telemetry-information ceiling", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e05_headline.png", dpi=220)
    plt.close(fig)


def gap_table(lead, bound, res):
    """Largest h with test error < DELTA_TARGET per predictor, vs the
    largest h at which the bound still permits error < DELTA_TARGET."""
    lines = ["| level | " + " | ".join(STYLE[k][2] for k in STYLE)
             + " | bound permits |",
             "|---|" + "---|" * (len(STYLE) + 1)]
    table = {}
    for level in ("light", "moderate", "heavy"):
        lr = res["levels"][level]
        d = bound[f"{level}_delta_min"]
        row = {}
        for name in STYLE:
            hs = [h for h in lead if lr[str(h)][name]["error"] < DELTA_TARGET]
            row[name] = max(hs) if hs else None
        hs_bound = [h for h, dd in zip(lead, d) if dd <= DELTA_TARGET]
        row["bound"] = max(hs_bound) if hs_bound else None
        table[level] = row
        fmt = lambda v: f"{v:g}" if v is not None else "none"  # noqa: E731
        lines.append(f"| {level} | " + " | ".join(
            fmt(row[k]) for k in list(STYLE) + ["bound"]) + " |")
    text = (f"Largest lead time h with error < {DELTA_TARGET} "
            "(test, val-tuned operating points)\n\n" + "\n".join(lines) + "\n")
    return table, text


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    bound = np.load(E03 / "lecam.npz", allow_pickle=True)
    sanity = json.loads((E03 / "sanity.json").read_text())
    res = json.loads((E04 / "results.json").read_text())
    lead = [float(h) for h in res["lead_times"]]
    np.testing.assert_allclose(bound["lead_times"], lead)

    headline_figure(lead, bound, res, sanity)
    table, text = gap_table(lead, bound, res)
    (DATA_DIR / "gap_table.md").write_text(text)
    (DATA_DIR / "gap_table.json").write_text(json.dumps(table, indent=1))
    print(text)
    print(f"e05 done; figure figures/e05_headline.png, table in {DATA_DIR}")


if __name__ == "__main__":
    main()
