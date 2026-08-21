"""Retention-campaign analysis, exactly as pre-registered
(docs/decisions.md 2026-08-21): per-cell and pooled median warning
leads with run-level bootstrap CIs and explicit left-censoring
handling, plus the retention skill horizon H*_R.

Estimation: cells B/C/D/E/G transfer the baseline calibration
(runs/calibration; rate levels unchanged); cell F uses its own
subcritical calibration (runs/cal-F). Cell A is the original five
eval runs. Forecasts via analyze.forecast_run (a_scale = 1; the
a-sensitivity was established on the baseline).

Saves data/e11/campaign_results.json.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e11"
RUNS = DATA_DIR / "runs"

_spec = importlib.util.spec_from_file_location(
    "e11_analyze", Path(__file__).with_name("analyze.py"))
e11a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e11a)

N_BOOT = 4000
BOOT_SEED = 2200
H_GRID = (8.0, 16.0, 24.0, 32.0, 48.0)
DELTAS = (0.025, 0.05)
CENSOR_EPS = 1e-6


def cell_runs():
    """(cell, run_dir, cal_dir) triples for every eval run."""
    out = []
    for d in sorted(RUNS.glob("s*")):
        if (d / "manifest.json").exists():
            out.append(("A", d, RUNS / "calibration"))
    for d in sorted(RUNS.glob("[B-G]-s*")):
        if not (d / "manifest.json").exists():
            continue
        cell = d.name.split("-")[0]
        cal = RUNS / ("cal-F" if cell == "F" else "calibration")
        out.append((cell, d, cal))
    return out


def censored(rec) -> bool:
    return (rec["t_first_alarm"] is not None
            and abs(rec["t_first_alarm"] - e11a.T_WARMUP) < CENSOR_EPS)


def lead_stats(recs, rng):
    """Median lead with run-level bootstrap CI; censoring bookkeeping.
    Censored (lower-bound) leads enter as-is; the median carries a
    '>=' flag when any censored value sits at or below it."""
    by_run = {}
    for r in recs:
        if r["lead_units"] is not None:
            by_run.setdefault(r["run"], []).append(
                (r["lead_units"], censored(r)))
    runs = sorted(by_run)
    leads = [v for rr in runs for v, _ in by_run[rr]]
    cens = [c for rr in runs for _, c in by_run[rr]]
    if not leads:
        return None
    med = float(np.median(leads))
    boots = []
    for _ in range(N_BOOT):
        take = rng.choice(runs, size=len(runs), replace=True)
        vals = [v for rr in take for v, _ in by_run[rr]]
        boots.append(np.median(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_crossings": len(leads), "n_runs": len(runs),
            "median_units": med, "median_ci": [float(lo), float(hi)],
            "is_lower_bound": bool(any(c and v <= med
                                       for v, c in zip(leads, cens))),
            "pct_censored": float(np.mean(cens)),
            "leads_units": leads}


def hstar_r(all_recs, rng):
    """Pre-registered H*_R: decision-grid events vs the alarm predictor,
    run-level bootstrap."""
    per_run = {}
    for rec in all_recs:
        rows = []
        crosses = ([rec["t_actual_cross"]]
                   if rec["t_actual_cross"] is not None else [])
        for d in rec["decisions"]:
            t = d["t_dec"]
            row = {}
            for h in H_GRID:
                label = any(t < tc <= t + h for tc in crosses)
                alarm = (d["t_cross_hat"] is not None
                         and d["t_cross_hat"] <= t + h)
                row[h] = (label, alarm)
            rows.append(row)
        per_run[rec["run"]] = rows
    runs = sorted(per_run)

    def skill_of(sample_runs):
        out = {}
        for h in H_GRID:
            labels = np.array([r[h][0] for rr in sample_runs
                               for r in per_run[rr]])
            alarms = np.array([r[h][1] for rr in sample_runs
                               for r in per_run[rr]])
            pi = labels.mean()
            err_c = min(pi, 1 - pi)
            err_a = (alarms != labels).mean()
            out[h] = (err_c - err_a, pi)
        return out

    point = skill_of(runs)
    boots = {h: [] for h in H_GRID}
    for _ in range(N_BOOT):
        take = rng.choice(runs, size=len(runs), replace=True)
        sk = skill_of(take)
        for h in H_GRID:
            boots[h].append(sk[h][0])
    res = {"h_grid": list(H_GRID),
           "skill": {f"{h:g}": {"point": float(point[h][0]),
                                "pi": float(point[h][1]),
                                "ci": [float(v) for v in np.percentile(
                                    boots[h], [2.5, 97.5])]}
                     for h in H_GRID}}
    lead_arr = np.array(H_GRID)
    for d in DELTAS:
        pts = np.array([point[h][0] for h in H_GRID])
        ph = pts >= d
        hp = float(lead_arr[np.flatnonzero(ph)[-1]]) if ph.any() else 0.0
        hb = []
        for b in range(N_BOOT):
            vals = np.array([boots[h][b] for h in H_GRID])
            hits = vals >= d
            hb.append(lead_arr[np.flatnonzero(hits)[-1]] if hits.any()
                      else 0.0)
        lo, hi = np.percentile(hb, [2.5, 97.5])
        res[f"h_star_{d:g}"] = {
            "point": hp, "ci": [float(lo), float(hi)],
            "right_censored": bool(hp >= lead_arr[-1])}
    return res


def main() -> None:
    t0 = time.time()
    ests = {}
    recs = []
    for cell, run_dir, cal_dir in cell_runs():
        key = cal_dir.name
        if key not in ests:
            print(f"estimating from {key}...", flush=True)
            ests[key] = e11a.e07.estimate(cal_dir)
        man = json.loads((run_dir / "manifest.json").read_text())
        ret_s = float(man.get("retention_s",
                              man["params"]["extra"]["retention_s"]))
        print(f"  forecasting {cell}/{run_dir.name} "
              f"(T_ret {ret_s:.0f} s)...", flush=True)
        rec = e11a.forecast_run(run_dir, ests[key], ret_s)
        rec["cell"] = cell
        recs.append(rec)

    rng = np.random.default_rng(BOOT_SEED)
    cells = sorted({r["cell"] for r in recs})
    per_cell = {}
    for cell in cells:
        st = lead_stats([r for r in recs if r["cell"] == cell],
                        np.random.default_rng(BOOT_SEED))
        if st is not None:
            st["exploratory"] = st["n_crossings"] < 3
        per_cell[cell] = st
    pooled = lead_stats(recs, rng)
    hsr = hstar_r(recs, np.random.default_rng(BOOT_SEED + 1))

    out = {"pre_registration": "docs/decisions.md 2026-08-21",
           "per_cell": per_cell, "pooled": pooled, "h_star_r": hsr,
           "runs": [{k: r[k] for k in
                     ("run", "cell", "retention_s", "t_actual_cross",
                      "t_expiry", "t_first_alarm", "lead_units",
                      "cross_err_at_alarm_units")} for r in recs]}
    (DATA_DIR / "campaign_results.json").write_text(json.dumps(out,
                                                               indent=1))
    tau = 4.0
    print("\nper cell:")
    for cell, st in per_cell.items():
        if st is None:
            print(f"  {cell}: no crossings")
            continue
        ge = ">=" if st["is_lower_bound"] else ""
        print(f"  {cell}: n={st['n_crossings']} median {ge}"
              f"{st['median_units']:.1f} u ({ge}"
              f"{st['median_units'] * tau:.0f} s) CI {st['median_ci']} "
              f"censored {st['pct_censored']:.0%}"
              + (" [EXPLORATORY]" if st["exploratory"] else ""))
    ge = ">=" if pooled["is_lower_bound"] else ""
    print(f"pooled: n={pooled['n_crossings']} median {ge}"
          f"{pooled['median_units']:.1f} u ({ge}"
          f"{pooled['median_units'] * tau:.0f} s) CI "
          f"{pooled['median_ci']} censored {pooled['pct_censored']:.0%}")
    for d in DELTAS:
        r = hsr[f"h_star_{d:g}"]
        print(f"H*_R (delta={d}): {r['point']:g} u CI {r['ci']}"
              + (" (right-censored)" if r["right_censored"] else ""))
    print(f"saved ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
