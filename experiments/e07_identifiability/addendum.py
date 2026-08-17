"""e07 onset addendum: repeat the held-out run until 10-15 flux-onset
events exist, replacing the n = 1 onset-lateness estimate with a
distribution (WRITING_PLAN section 0 -- the last authorized cluster work).

Nothing changes but repetition: parameters identical to the original
evaluation run, fresh seeds per run, estimation and eps still frozen on
the calibration run (e06 heavy).  Each run is forecast with the exact
analyze.py machinery; per-run forecasts are saved so the loop is
resumable.  The pooled report includes the original eval run (seed 9600,
same protocol) and classifies each forecastable burst as hit (onset
predicted and realized), miss (realized only), false alarm (predicted
only), or quiet (neither); the onset-error distribution is over hits.

Stops when TARGET_EVENTS realized onsets are pooled or MAX_RUNS runs
exist.  Adaptive stopping on the event *count* does not select on the
error values, so the timing distribution is unbiased.
"""

import json
import time
from pathlib import Path

import analyze
import numpy as np

from streamfno.kafka.harness import run_scenario
from streamfno.kafka.params import RunParams

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e07"
ADD_DIR = DATA_DIR / "runs" / "addendum"

SEEDS = list(range(9601, 9611))   # up to 10 additional runs
TARGET_EVENTS = 12
T_END = 300.0


def run_params(seed: int) -> RunParams:
    return RunParams(arrival="mmpp", r_low_high=0.10, t_end=T_END, seed=seed)


def forecast_run(run_dir: Path, est: dict, eps: float) -> dict:
    out_path = run_dir / "forecast.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    fc = analyze.forecast_eval(est, eps, run_dir=run_dir)
    out_path.write_text(json.dumps(fc, indent=1))
    return fc


def pooled_report(forecasts: dict[str, dict], eps: float) -> dict:
    bursts = []
    for name, fc in forecasts.items():
        for b in fc["bursts"]:
            bursts.append({"run": name, **b})
    hits = [b for b in bursts
            if b["onset_pred"] is not None and b["onset_real"] is not None]
    misses = [b for b in bursts
              if b["onset_pred"] is None and b["onset_real"] is not None]
    false_alarms = [b for b in bursts
                    if b["onset_pred"] is not None and b["onset_real"] is None]
    errors = [b["onset_pred"] - b["onset_real"] for b in hits]
    w1_pred = [float(np.mean(b["w1_pred"])) for b in bursts]
    w1_pers = [float(np.mean(b["w1_pers"])) for b in bursts]
    rep = {
        "eps": eps,
        "n_runs": len(forecasts),
        "n_bursts": len(bursts),
        "n_realized_onsets": len(hits) + len(misses),
        "n_hits": len(hits),
        "n_misses": len(misses),
        "n_false_alarms": len(false_alarms),
        "onset_errors_units": errors,
        "onset_error_median": float(np.median(errors)) if errors else None,
        "onset_error_iqr": ([float(np.percentile(errors, 25)),
                             float(np.percentile(errors, 75))]
                            if errors else None),
        "w1_pred_mean": float(np.mean(w1_pred)),
        "w1_pers_mean": float(np.mean(w1_pers)),
        "bursts": bursts,
    }
    return rep


def main() -> None:
    t0 = time.time()
    print("frozen estimation from calibration (e06 heavy)...")
    est = analyze.estimate(analyze.CAL_DIR)
    eps = analyze.calibrate_eps(analyze.CAL_DIR)

    forecasts = {"eval": forecast_run(DATA_DIR / "runs" / "eval", est, eps)}

    for seed in SEEDS:
        rep = pooled_report(forecasts, eps)
        print(f"pooled: {rep['n_realized_onsets']} realized onsets over "
              f"{rep['n_bursts']} bursts in {rep['n_runs']} runs")
        if rep["n_realized_onsets"] >= TARGET_EVENTS:
            break
        run_dir = ADD_DIR / f"s{seed}"
        if not (run_dir / "manifest.json").exists():
            print(f"run seed {seed} ({T_END * 4 / 60:.0f} min)...")
            run_scenario(run_dir, run_params(seed))
        forecasts[f"s{seed}"] = forecast_run(run_dir, est, eps)

    rep = pooled_report(forecasts, eps)
    (DATA_DIR / "onset_addendum.json").write_text(json.dumps(rep, indent=1))
    print(f"\nfinal: {rep['n_realized_onsets']} realized onsets "
          f"({rep['n_hits']} hits, {rep['n_misses']} misses, "
          f"{rep['n_false_alarms']} false alarms) over {rep['n_bursts']} "
          f"bursts in {rep['n_runs']} runs")
    if rep["onset_errors_units"]:
        print(f"onset error (units): median {rep['onset_error_median']:+.2f} "
              f"IQR {rep['onset_error_iqr']}  values "
              f"{[round(e, 2) for e in rep['onset_errors_units']]}")
    print(f"pooled W1: pred {rep['w1_pred_mean']:.4f} vs persistence "
          f"{rep['w1_pers_mean']:.4f}")
    print(f"wall clock {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
