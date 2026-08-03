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


def headline_figure(lead, bound, genie, res, sanity):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharey=True)
    for ax, level in zip(axes, ("light", "moderate", "heavy")):
        g = genie[f"{level}_gamma"]
        g_lo = g - 2 * genie[f"{level}_gamma_se"]
        ok = sanity["comparable_floor_valid"]
        region_label = ("no predictor with this telemetry can enter"
                        if ok else "computed lower bound (sanity flag; see log)")
        ax.fill_between(lead, 0.0, np.maximum(g_lo, 0.0), color="0.82",
                        zorder=0, label=region_label)
        ax.plot(lead, g, "k-", lw=1.4, zorder=1,
                label=r"$\gamma(h)$ state-omniscient floor")
        d = bound[f"{level}_delta_min"]
        ax.plot(lead, d, "k--", lw=1.1, zorder=1,
                label=r"$\delta_{\min}(h)$ two-point, worst-case state")
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


def gap_table(lead, genie, res):
    """Largest h with test error < DELTA_TARGET per predictor, vs the
    largest h at which the genie floor still permits error < DELTA_TARGET."""
    lines = ["| level | " + " | ".join(STYLE[k][2] for k in STYLE)
             + " | floor permits |",
             "|---|" + "---|" * (len(STYLE) + 1)]
    table = {}
    for level in ("light", "moderate", "heavy"):
        lr = res["levels"][level]
        g = genie[f"{level}_gamma"]
        row = {}
        for name in STYLE:
            hs = [h for h in lead if lr[str(h)][name]["error"] < DELTA_TARGET]
            row[name] = max(hs) if hs else None
        hs_bound = [h for h, gg in zip(lead, g) if gg <= DELTA_TARGET]
        row["bound"] = max(hs_bound) if hs_bound else None
        table[level] = row
        fmt = lambda v: f"{v:g}" if v is not None else "none"  # noqa: E731
        lines.append(f"| {level} | " + " | ".join(
            fmt(row[k]) for k in list(STYLE) + ["bound"]) + " |")
    text = (f"Largest lead time h with error < {DELTA_TARGET} "
            "(test, val-tuned operating points; 'floor permits' from the "
            "state-omniscient floor gamma)\n\n" + "\n".join(lines) + "\n")
    return table, text


def check_no_crossing(lead, genie, res, tol=0.0):
    """Section-1 rule: no predictor's test error may sit below the genie
    floor beyond CI/MC uncertainty.  Returns list of violations."""
    bad = []
    for level in ("light", "moderate", "heavy"):
        g = genie[f"{level}_gamma"]
        g_se = genie[f"{level}_gamma_se"]
        lr = res["levels"][level]
        for j, h in enumerate(lead):
            for name in STYLE:
                hi_ci = lr[str(h)][name]["error_ci"][1]
                if hi_ci < g[j] - 2 * g_se[j] - tol:
                    bad.append((level, h, name, lr[str(h)][name]["error"],
                                float(g[j])))
    return bad


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    bound = np.load(E03 / "lecam.npz", allow_pickle=True)
    genie = np.load(E03 / "genie.npz")
    sanity = json.loads((E03 / "sanity.json").read_text())
    res = json.loads((E04 / "results.json").read_text())
    lead = [float(h) for h in res["lead_times"]]
    np.testing.assert_allclose(bound["lead_times"], lead)

    crossings = check_no_crossing(lead, genie, res)
    sanity["comparable_floor_valid"] = not crossings
    if crossings:
        print("WARNING: predictor error below the genie floor (bug per the "
              "phase rules; investigate):")
        for c in crossings:
            print("  ", c)
    (DATA_DIR / "crossings.json").write_text(json.dumps(
        {"crossings": crossings}, indent=1))

    headline_figure(lead, bound, genie, res, sanity)
    table, text = gap_table(lead, genie, res)
    (DATA_DIR / "gap_table.md").write_text(text)
    (DATA_DIR / "gap_table.json").write_text(json.dumps(table, indent=1))
    print(text)
    print(f"e05 done; figure figures/e05_headline.png, table in {DATA_DIR}")


if __name__ == "__main__":
    main()
