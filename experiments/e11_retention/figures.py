"""e11 figure (paper F7), regenerated purely from saved results
(data/e11/retention.json): predicted vs actual lag-age trajectories
against the retention threshold, with predicted/actual crossings, the
warning lead, the realized-expiry marker, and H_actuation for scale;
side panel: the pooled warning-lead distribution."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e11"
FIG_DIR = ROOT / "figures"

H_ACTUATION_S = (3.3, 4.3)      # e08 medians, wall seconds
N_FAN = 6                       # forecast fans drawn on the episode panel


def episode_panel(ax, rec):
    tau = rec["tau_s"]
    ret = rec["retention_s"]
    t = np.asarray(rec["age_series"]["t"])
    age = np.asarray(rec["age_series"]["age_s"])
    ax.plot(t, age, "k-", lw=1.6, label="measured age (record timestamps)")
    ax.axhline(ret, color="C3", ls="--", lw=1.2,
               label=f"retention window T_ret = {ret:.0f} s")

    decs = [d for d in rec["decisions"] if d["t_dec"] <= (
        rec["t_actual_cross"] or t[-1])]
    step = max(1, len(decs) // N_FAN)
    for d in decs[::step]:
        tt = d["t_dec"] + np.asarray(d["traj_times"])
        ax.plot(tt, d["age_traj_s"], color="C0", lw=0.9, alpha=0.55)
    ax.plot([], [], color="C0", lw=0.9, alpha=0.7,
            label="FP age forecasts (anchored)")

    if rec["t_first_alarm"] is not None:
        ax.axvline(rec["t_first_alarm"], color="C0", ls=":", lw=1.2)
        ax.annotate("first alarm", (rec["t_first_alarm"], ret * 0.15),
                    rotation=90, fontsize=9, color="C0",
                    textcoords="offset points", xytext=(4, 0))
    if rec["t_actual_cross"] is not None:
        ax.axvline(rec["t_actual_cross"], color="k", ls=":", lw=1.2)
        ax.annotate("actual crossing", (rec["t_actual_cross"], ret * 0.15),
                    rotation=90, fontsize=9,
                    textcoords="offset points", xytext=(4, 0))
        if rec["t_first_alarm"] is not None:
            lead = rec["t_actual_cross"] - rec["t_first_alarm"]
            ymid = ret * 1.06
            ax.annotate("", xy=(rec["t_actual_cross"], ymid),
                        xytext=(rec["t_first_alarm"], ymid),
                        arrowprops=dict(arrowstyle="<->", color="0.35"))
            ax.text(0.5 * (rec["t_first_alarm"] + rec["t_actual_cross"]),
                    ymid, f"warning lead {lead:.0f} u = {lead * tau:.0f} s",
                    ha="center", va="bottom", fontsize=9, color="0.25")
        # actuation delay drawn to scale against the lead
        ha_u = np.mean(H_ACTUATION_S) / tau
        ax.axvspan(rec["t_actual_cross"] - ha_u, rec["t_actual_cross"],
                   color="C1", alpha=0.5)
        ax.annotate(f"H_actuation ({np.mean(H_ACTUATION_S):.1f} s)",
                    (rec["t_actual_cross"], ret * 0.55), rotation=90,
                    fontsize=8.5, color="C1",
                    textcoords="offset points", xytext=(-7, 0))
    if rec["t_expiry"] is not None:
        ax.plot([rec["t_expiry"]], [ret], "v", color="C3", ms=8,
                label="realized expiry (earliest > committed)")
    ax.set_xlabel("time (normalized units; 1 u = 4 s)")
    ax.set_ylabel("oldest-unconsumed-record age (s)")
    ax.set_title(f"(a) retention-boundary episode ({rec['run']})",
                 fontsize=9)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)


def lead_panel(ax, summary):
    leads = [r["lead_units"] for r in summary["runs"]
             if r["lead_units"] is not None]
    tau = summary["runs"][0]["tau_s"]
    lead_s = np.asarray(leads) * tau
    ax.plot(np.zeros(lead_s.size), lead_s, "o", color="C0", ms=7, alpha=0.6)
    med = float(np.median(lead_s))
    ax.axhline(med, color="k", lw=1.0,
               label=f"median {med:.0f} s")
    lo, hi = np.percentile(lead_s, [25, 75])
    ax.axhspan(lo, hi, color="0.9", zorder=0,
               label=f"IQR [{lo:.0f}, {hi:.0f}] s")
    ax.axhline(np.mean(H_ACTUATION_S), color="C1", lw=1.2, ls="--",
               label=f"H_actuation {np.mean(H_ACTUATION_S):.1f} s")
    ax.set_xticks([])
    ax.set_ylabel("warning lead (s)")
    ax.set_title(f"(b) warning leads, {len(leads)} crossings", fontsize=9)
    ax.legend(fontsize=8.5, loc="center right")
    ax.grid(alpha=0.3, axis="y")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    summary = json.loads((DATA_DIR / "retention.json").read_text())
    crossed = [r for r in summary["runs"] if r["lead_units"] is not None]
    # representative episode: the median warning lead
    rep = sorted(crossed, key=lambda r: r["lead_units"])[len(crossed) // 2]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                             gridspec_kw={"width_ratios": [3.2, 1.0]})
    episode_panel(axes[0], rep)
    lead_panel(axes[1], summary)
    fig.suptitle("e11: predicting the retention (recoverability) boundary "
                 "crossing", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e11_retention.png", dpi=200)
    plt.close(fig)
    print(f"e11 figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
