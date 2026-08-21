"""e05: the headline figure -- predictor error vs lead time over the
computed impossibility region, three load-level panels, plus the
gap-to-the-ceiling table.

Floor semantics (worked out during the e03 crossing investigation; see
docs/decisions.md): the expectation floor gamma = E[min(p, 1-p)] and the
measured test error live on different footings -- test labels are one
episode-clustered outcome draw, and predictors are scored on that same
draw.  The enforceable comparison is the *measured* error of the
state-omniscient Bayes predictor (predict 1{p_hat > 1/2} at each replayed
hidden state, scored on the same labels, same episode bootstrap), computed
here from the e03 genie output plus the dataset labels.  The crossing rule
is checked against that measured floor with CI overlap; the expectation
floor and the worst-case two-point bound are drawn alongside.

Assembly only beyond label recomputation: reads e03 (bounds, genie,
sanity) and e04 (predictor metrics).  `make e05` regenerates everything
end to end from saved results.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from streamfno.events import EventConfig, decision_times, label_episode
from streamfno.obs import Episode
from streamfno.predictors import metrics_with_ci

ROOT = Path(__file__).resolve().parents[2]
E02 = ROOT / "data" / "e02"
E03 = ROOT / "data" / "e03"
E04 = ROOT / "data" / "e04"
DATA_DIR = ROOT / "data" / "e05"
FIG_DIR = ROOT / "figures"

STYLE = {"pf": ("C2", "o", "particle filter"),
         "gbt": ("C1", "s", "gradient-boosted trees"),
         "logistic": ("C0", "^", "logistic regression"),
         "reactive": ("C3", "x", "reactive threshold")}
DELTA_TARGET = 0.2


def headline_figure(lead, bound, genie, gm_full, res, sanity):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharey=True)
    for ax, level in zip(axes, ("light", "moderate", "heavy")):
        gm = np.array([m["error"] for m in gm_full[level]])
        gm_lo = np.array([m["error_ci"][0] for m in gm_full[level]])
        lr0 = res["levels"][level]
        base0 = np.array([lr0[str(h)]["pf"]["base_rate"] for h in lead])
        err_const = np.minimum(base0, 1 - base0)
        ok = sanity["comparable_floor_valid"]
        region_label = ("no predictor with this telemetry can enter"
                        if ok else "computed lower bound (sanity flag; see log)")
        # display floor capped by the constant predictor (B1 audit): the
        # genie is an expectation over fresh futures while the labels are
        # one draw, so min(., err_const) keeps the region undercut-proof
        floor = np.minimum(np.maximum(gm_lo, 0.0), err_const)
        ax.fill_between(lead, 0.0, floor, color="0.82",
                        zorder=0, label=region_label)
        ax.plot(lead, gm, "k-", lw=1.4, zorder=1,
                label="state-omniscient predictor (measured)")
        g = genie[f"{level}_gamma"]
        ax.plot(lead, g, "k-.", lw=1.0, zorder=1,
                label=r"$\gamma(h)$ expectation floor")
        d = bound[f"{level}_delta_min"]
        ax.plot(lead, d, "k--", lw=1.0, zorder=1,
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
    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Backpressure predictability: practical predictors vs the "
                 "telemetry-information ceiling", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e05_headline.png", dpi=220)
    plt.close(fig)


def gap_table(lead, gm_full, res):
    """Largest h with test error < DELTA_TARGET per predictor, vs the
    largest h at which the measured state-omniscient predictor stays below
    DELTA_TARGET."""
    lines = ["| level | " + " | ".join(STYLE[k][2] for k in STYLE)
             + " | ceiling permits |",
             "|---|" + "---|" * (len(STYLE) + 1)]
    table = {}
    for level in ("light", "moderate", "heavy"):
        lr = res["levels"][level]
        gm = [m["error"] for m in gm_full[level]]
        row = {}
        for name in STYLE:
            hs = [h for h in lead if lr[str(h)][name]["error"] < DELTA_TARGET]
            row[name] = max(hs) if hs else None
        hs_bound = [h for h, gg in zip(lead, gm) if gg < DELTA_TARGET]
        row["bound"] = max(hs_bound) if hs_bound else None
        table[level] = row
        fmt = lambda v: f"{v:g}" if v is not None else "none"  # noqa: E731
        lines.append(f"| {level} | " + " | ".join(
            fmt(row[k]) for k in list(STYLE) + ["bound"]) + " |")
    text = (f"Largest lead time h with error < {DELTA_TARGET} "
            "(test, val-tuned operating points; 'ceiling permits' from the "
            "measured state-omniscient predictor)\n\n" + "\n".join(lines)
            + "\n")
    return table, text


def pf_episode_subset(level):
    """Episode indices the particle filter was evaluated on (from its
    stored per-episode probabilities)."""
    with np.load(E04 / "pf_probs.npz") as f:
        return {int(k.rsplit("_", 1)[1]) for k in f.files
                if k.startswith(f"{level}_test_")}


def genie_measured(level, genie, ecfg):
    """Measured error of the state-omniscient Bayes predictor: predict
    1{p_hat > 1/2}, scored on the realized test labels at the same states,
    with episode-bootstrap CIs.  Returns (full-test metrics, PF-subset
    metrics) per lead time."""
    manifest = json.loads((E02 / "manifest.json").read_text())
    rows = [m for m in manifest
            if m["level"] == level and m["split"] == "test"]
    labs = []
    for m in rows:
        ep = Episode.load(E02 / m["path"])
        t_dec = decision_times(ep, ecfg)[::2]
        labs.append(label_episode(ep, ecfg, t_dec)[1])
    labels = np.concatenate(labs)
    p = genie[f"{level}_p"]
    ep_ids = genie[f"{level}_episode_ids"]
    assert labels.shape == p.shape
    pf_sel = np.isin(ep_ids, sorted(pf_episode_subset(level)))
    full, sub = [], []
    for j in range(labels.shape[1]):
        full.append(metrics_with_ci(p[:, j], labels[:, j], ep_ids, 0.5))
        sub.append(metrics_with_ci(p[pf_sel, j], labels[pf_sel, j],
                                   ep_ids[pf_sel], 0.5))
    return full, sub


def check_no_crossing(lead, gm_full, gm_sub, res, tol=0.0):
    """Section-1 rule: no predictor's test error may sit below the measured
    genie-predictor error beyond CI overlap.  The particle filter is
    compared on its own episode subset.  Returns the list of violations."""
    bad = []
    for level in ("light", "moderate", "heavy"):
        lr = res["levels"][level]
        for j, h in enumerate(lead):
            for name in STYLE:
                gm = gm_sub[level][j] if name == "pf" else gm_full[level][j]
                hi_ci = lr[str(h)][name]["error_ci"][1]
                if hi_ci < gm["error_ci"][0] - tol:
                    bad.append((level, h, name, lr[str(h)][name]["error"],
                                gm["error"]))
    return bad


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)
    bound = np.load(E03 / "lecam.npz", allow_pickle=True)
    genie = np.load(E03 / "genie.npz")
    sanity = json.loads((E03 / "sanity.json").read_text())
    res = json.loads((E04 / "results.json").read_text())
    lead = [float(h) for h in res["lead_times"]]
    ecfg = EventConfig.load(E02 / "event_config.json")
    np.testing.assert_allclose(bound["lead_times"], lead)
    np.testing.assert_allclose(list(ecfg.lead_times), lead)

    gm_full, gm_sub = {}, {}
    for level in ("light", "moderate", "heavy"):
        gm_full[level], gm_sub[level] = genie_measured(level, genie, ecfg)

    crossings = check_no_crossing(lead, gm_full, gm_sub, res)
    sanity["comparable_floor_valid"] = not crossings
    if crossings:
        print("WARNING: predictor error below the measured genie ceiling "
              "(bug per the phase rules; investigate):")
        for c in crossings:
            print("  ", c)
    (DATA_DIR / "crossings.json").write_text(json.dumps(
        {"crossings": crossings}, indent=1))
    (DATA_DIR / "genie_measured.json").write_text(json.dumps(
        {"full": gm_full, "pf_subset": gm_sub}, indent=1))

    headline_figure(lead, bound, genie, gm_full, res, sanity)
    table, text = gap_table(lead, gm_full, res)
    (DATA_DIR / "gap_table.md").write_text(text)
    (DATA_DIR / "gap_table.json").write_text(json.dumps(table, indent=1))
    print(text)
    print(f"e05 done; figure figures/e05_headline.png, table in {DATA_DIR}")


if __name__ == "__main__":
    main()
