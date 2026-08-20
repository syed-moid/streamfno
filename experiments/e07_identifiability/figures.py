"""e07 figures (paper F5), regenerated purely from saved results.

Three panels: (a) binned drift estimates per recovered modulator state
against the configured netput, (b) held-out forecast W1 trajectories vs
the persistence baseline pooled over every addendum run, (c) predicted
vs realized flux-onset times with the addendum error distribution.
Reads data/e07/identifiability.json and data/e07/onset_addendum.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e07"
FIG_DIR = ROOT / "figures"

STATE_COLORS = {"low": "C0", "high": "C3"}
ANCHOR_RANGE = (0.06, 0.70)   # analyze.py's interior-bin window
MIN_COUNT = 50
MIN_TRACES = 10               # mean W1 shown while this many bursts remain


def drift_panel(ax, est):
    tau_s = est["params"]["tau_s"]
    for state in ("low", "high"):
        rec = est[state]
        xc = np.asarray(rec["x_centers"])
        bb = np.asarray(rec["b_hat_bins"], dtype=float)
        ok = np.isfinite(bb) & (np.asarray(rec["counts"]) > MIN_COUNT)
        color = STATE_COLORS[state]
        ax.plot(xc[ok], bb[ok], "o", color=color, ms=4,
                label=f"{state}-state bins")
        ax.axhline(rec["b_configured"], color=color, ls="--", lw=1.0,
                   label=f"configured {rec['b_configured']:+.2f} "
                         f"(rel err {rec['b_rel_error']:.1%})")
    ax.axvspan(*ANCHOR_RANGE, color="0.92", zorder=0)
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.set_xlabel("x (normalized lag)")
    ax.set_ylabel("drift b(x) per unit time")
    ax.set_title(f"(a) telemetry-only drift vs configured netput "
                 f"(1 unit = {tau_s:.0f} s)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def w1_panel(ax, add):
    bursts = add["bursts"]
    # per-tick means over the bursts still running at that tick; every
    # trace shares the 0.25-unit collector cadence, so ticks align
    max_len = max(len(b["w1_pred"]) for b in bursts)
    dts = [b["horizon"] / len(b["w1_pred"]) for b in bursts]
    dt = float(np.median(dts))
    for b, dti in zip(bursts, dts):
        t = (np.arange(len(b["w1_pred"])) + 1) * dti
        ax.plot(t, b["w1_pers"], color="0.75", lw=0.6, alpha=0.4)
        ax.plot(t, b["w1_pred"], color="C0", lw=0.6, alpha=0.4)
    for key, color, label in (("w1_pers", "0.4", "persistence, mean"),
                              ("w1_pred", "C0", "FP forecast, mean")):
        means, ts = [], []
        for k in range(max_len):
            vals = [b[key][k] for b in bursts if len(b[key]) > k]
            if len(vals) < MIN_TRACES:
                break
            means.append(np.mean(vals))
            ts.append((k + 1) * dt)
        ax.plot(ts, means, color=color, lw=2.0, label=label)
    ax.set_xlabel("time since burst start (units)")
    ax.set_ylabel(r"$W_1$(forecast, realized)")
    ax.set_title("(b) held-out forecast W1 vs persistence", fontsize=9)
    ratio = add["w1_pers_mean"] / add["w1_pred_mean"]
    ax.text(0.97, 0.55,
            f"{len(bursts)} bursts / {add['n_runs']} runs\n"
            f"pooled W1 {add['w1_pred_mean']:.3f} vs "
            f"{add['w1_pers_mean']:.3f} ({ratio:.0f}x)",
            transform=ax.transAxes, ha="right", va="center", fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def onset_panel(ax, add, tau_s):
    hits = [b for b in add["bursts"]
            if b["onset_pred"] is not None and b["onset_real"] is not None]
    real = np.array([b["onset_real"] for b in hits])
    pred = np.array([b["onset_pred"] for b in hits])
    lim = (0.0, 1.1 * max(real.max(), pred.max()))
    ax.plot(lim, lim, "k-", lw=1.0, label="perfect timing")
    med = add["onset_error_median"]
    ax.plot(lim, (lim[0] + med, lim[1] + med), "k--", lw=1.0,
            label=f"median bias {med:+.2f} u ({med * tau_s:+.1f} s)")
    ax.plot(real, pred, "o", color="C3", ms=6, alpha=0.6,
            label=f"realized onsets ({len(hits)})")
    lo, hi = add["onset_error_iqr"]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("realized onset (units after burst start)")
    ax.set_ylabel("predicted onset (units)")
    ax.set_title(f"(c) onset timing, {add['n_hits']}/"
                 f"{add['n_realized_onsets']} predicted", fontsize=9)
    ax.text(0.04, 0.96,
            f"{add['n_misses']} misses, {add['n_false_alarms']} false alarms\n"
            f"error IQR [{lo:+.2f}, {hi:+.2f}] u",
            transform=ax.transAxes, ha="left", va="top", fontsize=7)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    est = json.loads((DATA_DIR / "identifiability.json").read_text())
    add = json.loads((DATA_DIR / "onset_addendum.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    drift_panel(axes[0], est)
    w1_panel(axes[1], add)
    onset_panel(axes[2], add, est["params"]["tau_s"])
    fig.suptitle("e07: identifiability from telemetry alone -- estimate, "
                 "forecast, onset", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e07_identifiability.png", dpi=200)
    plt.close(fig)
    print(f"e07 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
