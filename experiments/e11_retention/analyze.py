"""e11 analysis: forecasting the retention-boundary crossing.

Estimation (calibration run, telemetry only, e07 machinery): per-state
drift bins and variance rate; the per-state drifts transfer to the eval
runs, whose modulator mix differs by design.

Forecast at each decision time on an eval run:
- live inputs: trailing modulator-state fraction p_hat (classifier),
  trailing arrival rate lambda_hat, trailing-window initial density of
  the clipped occupancy, measured per-partition overshoot beyond the
  budget, and the measured current age A(t) (ager, record timestamps);
- FP solve with the mixture drift b_bar = p_hat b_high + (1-p_hat)
  b_low to horizon H_MAX; predicted mean backlog trajectory in X units
  is mean_lag + cumulative regulator + current overshoot (the clipped
  state is one-sided: regulated mass never returns -- stated model
  limitation, secondary during growth toward a crossing);
- age forecast ANCHORED at the measured age: A_hat(t+h) = A(t) +
  [L_hat(h) - L_hat(0)] / lambda_hat, the FIFO identity dA/dt =
  (dL/dt)/lambda for slowly varying lambda; A ~ L/lambda itself is the
  stationary-regime interpretation, not the definition (the definition
  is record timestamps, measured by the ager);
- t_cross_hat = first h with A_hat >= T_ret (T_ret is the operator-known
  topic retention, not a tuned threshold).

Episode metrics per realized crossing: warning lead = t_actual_cross -
t_first_alarm; crossing-time error at the first alarm; realized-expiry
time (earliest offset passing committed) as the ground-truth
recoverability-loss event.  Saves data/e11/retention.json and
data/e11/f7_traces.npz.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e11"
RUNS_DIR = DATA_DIR / "runs"

_spec = importlib.util.spec_from_file_location(
    "e07_analyze",
    ROOT / "experiments" / "e07_identifiability" / "analyze.py")
e07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e07)

T_WARMUP = 40.0
DT_DECISION = 4.0
H_MAX = 120.0
FP_DT = 4e-3            # e12: temporal W1 error ~4e-4 here; solve cost 4x down
FP_CELLS = 128
RHO0_TICKS = 4
LAMBDA_WINDOW = 20.0    # units, trailing arrival-rate mean
PHAT_WINDOW = 40.0      # units, trailing high-state fraction


def load_age(run_dir: Path, t0_wall: float, tau_s: float):
    with np.load(run_dir / "age.npz") as f:
        t = (f["t_wall"] - t0_wall) / tau_s
        age = np.nanmax(np.nan_to_num(f["age_s"], nan=0.0), axis=1)
        expired_any = f["expired"].any(axis=1)
    return t, age, expired_any


def expiry_time(tel, t_norm, keep):
    """First normalized time the earliest offset passes a committed
    offset on any partition (ground-truth recoverability loss)."""
    if tel.earliest is None:
        return None
    com = tel.committed[keep]
    early = tel.earliest[keep]
    hit = ((com >= 0) & (early > com)).any(axis=1)
    idx = np.flatnonzero(hit)
    return float(t_norm[keep][idx[0]]) if idx.size else None


def trailing_mean(t: np.ndarray, v: np.ndarray, window: float) -> np.ndarray:
    out = np.empty_like(v, dtype=float)
    for i, ti in enumerate(t):
        sel = (t > ti - window) & (t <= ti)
        out[i] = float(np.mean(v[sel])) if sel.any() else np.nan
    return out


def mixture_drift(est: dict, p_high: float):
    b_hi, a_hi = e07.drift_and_diffusion(est, "high")
    b_lo, a_lo = e07.drift_and_diffusion(est, "low")

    def b(x, m):
        return p_high * b_hi(x, m) + (1.0 - p_high) * b_lo(x, m)

    return b, p_high * a_hi + (1.0 - p_high) * a_lo


def forecast_run(run_dir: Path, est: dict, retention_s: float) -> dict:
    params, tel, t_norm, keep, man = e07.load_run(run_dir)
    dt = params.dt_poll_norm
    tau = params.tau_s
    ret_units = retention_s / tau
    t = t_norm[keep]
    x = tel.x(params.budget_b)[keep]
    lag = tel.lag[keep]
    rate = e07.arrival_rate_norm(tel, params, keep)
    is_high, _ = e07.classify_two_state(rate, smooth=e07.SMOOTH_TICKS)
    lam_bar = trailing_mean(t, rate, LAMBDA_WINDOW)
    p_bar = trailing_mean(t, is_high.astype(float), PHAT_WINDOW)
    overshoot = np.maximum(lag / float(params.budget_b) - 1.0, 0.0).mean(axis=1)

    t_age, age, _ = load_age(run_dir, man["t0_wall"], tau)
    age_on_t = np.interp(t, t_age, age)          # seconds
    t_exp = expiry_time(tel, t_norm, keep)

    h_cell = 1.0 / FP_CELLS
    edges = np.linspace(0.0, 1.0, FP_CELLS + 1)
    edges[-1] += 1e-9

    idx_cross = np.flatnonzero(age_on_t >= retention_s)
    t_actual = float(t[idx_cross[0]]) if idx_cross.size else None

    decisions = []
    t_dec = T_WARMUP
    t_stop = (t_actual + 2 * DT_DECISION) if t_actual is not None else t[-1]
    while t_dec <= min(t_stop, t[-1] - dt):
        i = int(np.searchsorted(t, t_dec))
        i = min(i, t.size - 1)
        pool = x[max(0, i - RHO0_TICKS + 1):i + 1].ravel()
        rho0 = np.histogram(pool, bins=edges)[0] / pool.size / h_cell
        lam_i = max(float(lam_bar[i]), 1e-3)
        b_fn, a_val = mixture_drift(est, float(np.clip(p_bar[i], 0.0, 1.0)))
        fp = e07.solve_fp(rho0, b_fn, a_val, t_end=H_MAX, dt=FP_DT,
                          dt_sample=1.0)
        l_hat = fp.mean_lag[:, 0] + fp.regulator_cum[:, 0] + overshoot[i]
        a_hat_s = age_on_t[i] + (l_hat - l_hat[0]) / lam_i * tau
        hit = np.flatnonzero(a_hat_s >= retention_s)
        t_cross_hat = (float(t_dec + fp.times[hit[0]]) if hit.size
                       else None)
        decisions.append({
            "t_dec": float(t_dec), "age_s": float(age_on_t[i]),
            "lam_hat": lam_i, "p_high_hat": float(p_bar[i]),
            "overshoot_x": float(overshoot[i]),
            "t_cross_hat": t_cross_hat,
            "age_traj_s": [float(v) for v in a_hat_s],
            "traj_times": [float(v) for v in fp.times],
        })
        t_dec += DT_DECISION

    alarms = [d for d in decisions if d["t_cross_hat"] is not None]
    t_alarm = alarms[0]["t_dec"] if alarms else None
    rec = {
        "run": run_dir.name, "retention_s": retention_s,
        "retention_units": ret_units, "tau_s": tau,
        "t_actual_cross": t_actual, "t_expiry": t_exp,
        "t_first_alarm": t_alarm,
        "lead_units": (t_actual - t_alarm
                       if t_actual is not None and t_alarm is not None
                       else None),
        "cross_err_at_alarm_units": (
            alarms[0]["t_cross_hat"] - t_actual
            if alarms and t_actual is not None else None),
        "decisions": decisions,
        "age_series": {"t": [float(v) for v in t],
                       "age_s": [float(v) for v in age_on_t]},
    }
    return rec


def main() -> None:
    print("estimating (b, a) per state from the e11 calibration run...")
    est = e07.estimate(RUNS_DIR / "calibration")
    for state in ("low", "high"):
        r = est[state]
        if r["b_hat"] is not None:
            print(f"  {state:<5} b_hat = {r['b_hat']:+.4f}  configured = "
                  f"{r['b_configured']:+.4f}  rel err = {r['b_rel_error']:.3f}")

    results = []
    for run_dir in sorted(RUNS_DIR.glob("s*")):
        man = json.loads((run_dir / "manifest.json").read_text())
        rec = forecast_run(run_dir, est, float(man["retention_s"]))
        results.append(rec)
        lead = rec["lead_units"]
        print(f"  {run_dir.name}: actual cross {rec['t_actual_cross']}, "
              f"expiry {rec['t_expiry']}, first alarm "
              f"{rec['t_first_alarm']}, lead "
              f"{lead if lead is None else f'{lead:.1f}u'}")

    leads = [r["lead_units"] for r in results if r["lead_units"] is not None]
    errs = [r["cross_err_at_alarm_units"] for r in results
            if r["cross_err_at_alarm_units"] is not None]
    summary = {
        "n_runs": len(results),
        "n_crossings": len(leads),
        "lead_units_median": float(np.median(leads)) if leads else None,
        "lead_units_iqr": ([float(np.percentile(leads, q)) for q in (25, 75)]
                           if leads else None),
        "cross_err_units_median": float(np.median(errs)) if errs else None,
        "cross_err_units_iqr": ([float(np.percentile(errs, q))
                                 for q in (25, 75)] if errs else None),
        "estimation": {s: {k: est[s][k] for k in
                           ("b_hat", "b_se", "b_configured", "b_rel_error",
                            "a_hat_d1", "a_hat_d2", "a_noise_corrected")}
                       for s in ("low", "high")},
        "runs": results,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "retention.json").write_text(json.dumps(summary, indent=1))
    if leads:
        tau = results[0]["tau_s"]
        print(f"\npooled: {len(leads)} crossings; warning lead median "
              f"{summary['lead_units_median']:.1f} u "
              f"({summary['lead_units_median'] * tau:.0f} s), IQR "
              f"{summary['lead_units_iqr']}; crossing error at alarm "
              f"median {summary['cross_err_units_median']:+.1f} u")
    print(f"saved {DATA_DIR / 'retention.json'}")


if __name__ == "__main__":
    main()
