"""e09: computational profile of the prediction plane.

Times the existing deterministic online pipeline end to end on this
machine: coefficient estimation (binned conditional-increment
regressions on the calibration telemetry), the Chang-Cooper FP forecast,
the backpressure functional (regulator rate -> smoothed flux), and the
T_BP threshold logic.  All inputs are the real e06/e07 telemetry and the
frozen e07 calibration artifacts; nothing here is synthetic.

Measured, per the Phase E protocol:
- per-forecast wall latency (median/p95/p99 over >= 10^4 single-class
  forecasts across the e07 horizon grid), CPU time, peak memory;
- the per-broker-class variant (3 classes, the lab's broker count);
- ensemble scaling M in {1, 8, 32, 128} at the mid horizon -- the
  forecast object is the per-class density field, so ensemble members
  are coefficient-perturbed solves, never per-partition evaluations;
- the amortized estimation (recalibration) cost;
- the fixed row: external service calls = 0, token cost = 0, network
  dependencies = 0.

Seeded and deterministic given the saved telemetry; saves
data/e09/profile.json and data/e09/latencies.npz.
"""

import importlib.util
import json
import platform
import resource
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e09"
EVAL_DIR = ROOT / "data" / "e07" / "runs" / "eval"
ADDENDUM = ROOT / "data" / "e07" / "onset_addendum.json"

# the online pipeline under test is e07's, imported as-is
_spec = importlib.util.spec_from_file_location(
    "e07_analyze", ROOT / "experiments" / "e07_identifiability" / "analyze.py")
e07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e07)

SEED = 900
HORIZONS = (1.0, 2.0, 4.0, 8.0, 16.0)   # e07 grid: bursts up to MAX_HORIZON
N_SINGLE = 2000                          # forecasts per horizon, 1 class
N_MULTI = 200                            # forecasts per horizon, 3 classes
N_BROKER_CLASSES = 3                     # the lab's broker count
ENSEMBLE_M = (1, 8, 32, 128)
ENSEMBLE_TRIALS = {1: 16, 8: 8, 32: 5, 128: 3}
H_ENSEMBLE = 8.0
N_EST_REPS = 5
FP_DT = 2e-3
FP_CELLS = e07.FP_CELLS
RHO0_TICKS = e07.RHO0_TICKS


def load_eval_x():
    params, tel, t_norm, keep, _ = e07.load_run(EVAL_DIR)
    x = tel.x(params.budget_b)[keep]
    t = t_norm[keep]
    return params, x, t


def rho0_from_window(x, end_idx: int, n_classes: int, edges, h):
    """Trailing-window initial density, exactly as the e07 forecast
    builds it; per-broker-class variant splits the partition axis."""
    pool = x[max(0, end_idx - RHO0_TICKS + 1):end_idx + 1]
    if n_classes == 1:
        flat = pool.ravel()
        return (np.histogram(flat, bins=edges)[0] / flat.size / h)[None, :]
    parts = np.array_split(np.arange(pool.shape[1]), n_classes)
    rows = []
    for idx in parts:
        flat = pool[:, idx].ravel()
        rows.append(np.histogram(flat, bins=edges)[0] / flat.size / h)
    return np.stack(rows)


def forecast_once(rho0, b_fn, a_val, horizon, dt_sample, eps):
    """One pipeline pass: FP solve -> backpressure functional -> T_BP.
    Returns per-stage wall times (ns) and the T_BP decision."""
    t0 = time.perf_counter_ns()
    fp = e07.solve_fp(rho0, b_fn, a_val, t_end=horizon, dt=FP_DT,
                      dt_sample=dt_sample)
    t1 = time.perf_counter_ns()
    t_bp = None
    for c in range(fp.regulator_rate.shape[1]):
        bar = e07.smoothed_flux(fp.times, fp.regulator_rate[:, c],
                                e07.FLUX_WINDOW)
        on = fp.times[bar > eps]
        if on.size and (t_bp is None or on[0] < t_bp):
            t_bp = float(on[0])
    t2 = time.perf_counter_ns()
    return t1 - t0, t2 - t1, t_bp


def profile_forecasts(x, t, params, b_fn, a_val, eps, n_classes, n_per_h,
                      rng):
    h_cell = 1.0 / FP_CELLS
    edges = np.linspace(0.0, 1.0, FP_CELLS + 1)
    edges[-1] += 1e-9
    valid = np.flatnonzero(t >= e07.T_WARMUP)
    out = {}
    cpu0 = time.process_time()
    for horizon in HORIZONS:
        ends = rng.choice(valid, size=n_per_h, replace=True)
        lat_total = np.empty(n_per_h)
        lat_rho0 = np.empty(n_per_h)
        lat_solve = np.empty(n_per_h)
        lat_thresh = np.empty(n_per_h)
        n_fired = 0
        for i, e in enumerate(ends):
            t0 = time.perf_counter_ns()
            rho0 = rho0_from_window(x, int(e), n_classes, edges, h_cell)
            t1 = time.perf_counter_ns()
            dt_solve, dt_thresh, t_bp = forecast_once(
                rho0, b_fn, a_val, horizon, params.dt_poll_norm, eps)
            lat_rho0[i] = t1 - t0
            lat_solve[i] = dt_solve
            lat_thresh[i] = dt_thresh
            lat_total[i] = (t1 - t0) + dt_solve + dt_thresh
            n_fired += t_bp is not None
        out[horizon] = {
            "n": n_per_h,
            "wall_ms": {k: float(np.percentile(lat_total, q) / 1e6)
                        for k, q in (("p50", 50), ("p95", 95), ("p99", 99))},
            "stage_median_ms": {
                "rho0": float(np.median(lat_rho0) / 1e6),
                "fp_solve": float(np.median(lat_solve) / 1e6),
                "threshold": float(np.median(lat_thresh) / 1e6),
            },
            "t_bp_fired_fraction": n_fired / n_per_h,
            "lat_total_ns": lat_total,
        }
    out["cpu_seconds_total"] = time.process_time() - cpu0
    return out


def profile_ensemble(x, t, params, est, eps, rng):
    """Coefficient-perturbed ensembles at the mid horizon: M solves per
    forecast, threshold on the ensemble-mean flux."""
    h_cell = 1.0 / FP_CELLS
    edges = np.linspace(0.0, 1.0, FP_CELLS + 1)
    edges[-1] += 1e-9
    valid = np.flatnonzero(t >= e07.T_WARMUP)
    base_b, base_a = e07.drift_and_diffusion(est, "high")
    out = {}
    for m_size in ENSEMBLE_M:
        trials = ENSEMBLE_TRIALS[m_size]
        lat = np.empty(trials)
        for j in range(trials):
            end = int(rng.choice(valid))
            b_scales = 1.0 + 0.05 * rng.standard_normal(m_size)
            a_scales = np.abs(1.0 + 0.10 * rng.standard_normal(m_size))
            t0 = time.perf_counter_ns()
            rho0 = rho0_from_window(x, end, 1, edges, h_cell)
            fluxes = []
            times_ref = None
            for k in range(m_size):
                def b_k(xv, m, s=float(b_scales[k])):
                    return s * base_b(xv, m)
                fp = e07.solve_fp(rho0, b_k, base_a * float(a_scales[k]),
                                  t_end=H_ENSEMBLE, dt=FP_DT,
                                  dt_sample=params.dt_poll_norm)
                fluxes.append(fp.regulator_rate[:, 0])
                times_ref = fp.times
            mean_flux = np.mean(fluxes, axis=0)
            bar = e07.smoothed_flux(times_ref, mean_flux, e07.FLUX_WINDOW)
            _ = times_ref[bar > eps]
            lat[j] = time.perf_counter_ns() - t0
        out[m_size] = {
            "trials": trials,
            "wall_ms_median": float(np.median(lat) / 1e6),
            "members_per_second": float(m_size / (np.median(lat) / 1e9)),
        }
    return out


def profile_estimation():
    wall = np.empty(N_EST_REPS)
    cpu = np.empty(N_EST_REPS)
    est = None
    for i in range(N_EST_REPS):
        t0, c0 = time.perf_counter(), time.process_time()
        est = e07.estimate(e07.CAL_DIR)
        wall[i] = time.perf_counter() - t0
        cpu[i] = time.process_time() - c0
    return est, {"reps": N_EST_REPS,
                 "wall_s_median": float(np.median(wall)),
                 "cpu_s_median": float(np.median(cpu))}


def peak_memory_mib(fn) -> float:
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 2**20


def machine_info() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cpu_count": __import__("os").cpu_count(),
    }


def main() -> None:
    t_start = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    eps = json.loads(ADDENDUM.read_text())["eps"]

    print("estimation (recalibration) cost, "
          f"{N_EST_REPS} reps on the e06-heavy calibration run...")
    est, est_prof = profile_estimation()
    print(f"  wall {est_prof['wall_s_median']:.2f} s median")

    params, x, t = load_eval_x()
    b_fn, a_val = e07.drift_and_diffusion(est, "high")

    print(f"single-class forecasts: {N_SINGLE} per horizon x {HORIZONS}...")
    single = profile_forecasts(x, t, params, b_fn, a_val, eps, 1, N_SINGLE,
                               rng)
    print(f"per-broker-class ({N_BROKER_CLASSES}) forecasts: "
          f"{N_MULTI} per horizon...")
    multi = profile_forecasts(x, t, params, b_fn, a_val, eps,
                              N_BROKER_CLASSES, N_MULTI, rng)
    print(f"ensemble scaling M in {ENSEMBLE_M} at h = {H_ENSEMBLE}...")
    ensemble = profile_ensemble(x, t, params, est, eps, rng)

    print("peak-memory passes...")
    h_cell = 1.0 / FP_CELLS
    edges = np.linspace(0.0, 1.0, FP_CELLS + 1)
    edges[-1] += 1e-9
    end = int(np.flatnonzero(t >= e07.T_WARMUP)[0]) + RHO0_TICKS
    mem = {
        "forecast_h16_c1_mib": peak_memory_mib(lambda: forecast_once(
            rho0_from_window(x, end, 1, edges, h_cell), b_fn, a_val, 16.0,
            params.dt_poll_norm, eps)),
        "forecast_h16_c3_mib": peak_memory_mib(lambda: forecast_once(
            rho0_from_window(x, end, N_BROKER_CLASSES, edges, h_cell),
            b_fn, a_val, 16.0, params.dt_poll_norm, eps)),
        "process_ru_maxrss_mib": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss / 2**20,
    }

    lat_arrays = {}
    for tag, prof in (("single", single), ("multi", multi)):
        for horizon in HORIZONS:
            lat_arrays[f"lat_{tag}_h{horizon:g}"] = prof[horizon].pop(
                "lat_total_ns")
    np.savez_compressed(DATA_DIR / "latencies.npz", **lat_arrays)

    profile = {
        "seed": SEED,
        "config": {"horizons": list(HORIZONS), "n_single": N_SINGLE,
                   "n_multi": N_MULTI, "n_broker_classes": N_BROKER_CLASSES,
                   "fp_cells": FP_CELLS, "fp_dt": FP_DT,
                   "ensemble_m": list(ENSEMBLE_M), "h_ensemble": H_ENSEMBLE,
                   "eps": eps},
        "machine": machine_info(),
        "estimation": est_prof,
        "single_class": {str(k): v for k, v in single.items()},
        "per_broker_class": {str(k): v for k, v in multi.items()},
        "ensemble": {str(k): v for k, v in ensemble.items()},
        "memory": mem,
        "fixed_row": {"external_service_calls": 0, "token_cost": 0,
                      "network_dependencies": 0},
        "wall_clock_s": time.time() - t_start,
    }
    (DATA_DIR / "profile.json").write_text(json.dumps(profile, indent=1))
    print(f"saved {DATA_DIR / 'profile.json'} "
          f"({profile['wall_clock_s'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
