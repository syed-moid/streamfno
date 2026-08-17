"""e07 analysis: telemetry identifiability end to end.

1. From the calibration run's telemetry only (per-partition lag and
   log-end offsets; the configured rates are never read), estimate the
   drift per modulator state and the variance rate via the binned
   conditional-increment regressions.  The modulator state is itself
   inferred from telemetry (two-state classification of the aggregate
   arrival rate; arrivals are log-end-offset increments).
2. Sanity anchor: the estimated interior drift per state against the
   *configured* lam - mu in normalized units -- the identifiability
   headline (relative error).
3. Plug (b_hat, a_hat) into the Fokker-Planck solver and forecast each
   detected burst of the held-out evaluation run from its start state;
   compare W1(prediction, realized) over time against a persistence
   baseline, and boundary-flux onset times, with the event threshold
   calibrated on the calibration run (e02 protocol).

Variance-rate estimates carry an additive measurement-noise term
2 sigma^2 / Delta (committed-offset quantization); a_hat is therefore
estimated at two strides and the noise-cancelling combination
a = 2 a_hat(2 Delta) - a_hat(Delta) is reported alongside both raw
values and used for the forecast when positive.
"""

import json
from pathlib import Path

import numpy as np

from streamfno.analysis.distances import w1_weights
from streamfno.analysis.increments import (
    binned_increments,
    classify_two_state,
    interior_mean_drift,
)
from streamfno.events import EventConfig, calibrate_threshold, smoothed_flux
from streamfno.kafka.adapter import load_telemetry, overshoot_flux, telemetry_episode
from streamfno.kafka.params import RunParams
from streamfno.pde.solver import solve_fp

ROOT = Path(__file__).resolve().parents[2]
CAL_DIR = ROOT / "data" / "e06" / "runs" / "heavy"
EVAL_DIR = ROOT / "data" / "e07" / "runs" / "eval"
OUT_DIR = ROOT / "data" / "e07"

T_WARMUP = 40.0           # normalized units dropped from estimation
N_BINS_EST = 25
STRIDE = 4                # Delta = 1 normalized unit at 0.25 cadence
# drift uses one-tick increments: measurement noise is zero-mean in the
# conditional mean (only the variance inherits its bias), and a long
# window is *censored* -- during a 0.3/unit drain a 1-unit window starting
# below x ~ 0.3 hits the empty wall inside the window, shrinking |b_hat|
DRIFT_STRIDE = 1
# interior bins only: the wall bin at x = 0 censors the drift (an empty
# queue shows b ~ 0 whatever the netput), so anchors and coefficient
# interpolation start above the first bin
ANCHOR_RANGE = (0.06, 0.70)
SMOOTH_TICKS = 2          # arrival-state classifier; rate SNR is large and
                          # longer trailing windows delay labels into drains
MIN_BURST_UNITS = 3.0     # forecast bursts at least this long
MAX_HORIZON = 16.0
FLUX_WINDOW = 2.0
FP_CELLS = 128
RHO0_TICKS = 4            # trailing ticks pooled for the initial density
TARGET_RATE = 0.10
H_MID = 8.0


def load_run(run_dir: Path):
    man = json.loads((run_dir / "manifest.json").read_text())
    params = RunParams.from_json(json.dumps(man["params"]))
    tel = load_telemetry(run_dir / "lag.npz")
    t_norm = (tel.t_wall - man["t0_wall"]) / params.tau_s
    keep = (t_norm >= 0.0) & (t_norm <= params.t_end)
    return params, tel, t_norm, keep, man


def arrival_rate_norm(tel, params, keep):
    """Aggregate arrival netput per partition (X per unit time) from
    log-end-offset increments -- broker-side telemetry."""
    leo = tel.leo[keep].sum(axis=1).astype(float)
    rate = np.zeros(leo.size)
    rate[1:] = np.diff(leo) / (params.n_partitions * params.budget_b
                               * params.dt_poll_norm)
    return rate


def estimate(run_dir: Path) -> dict:
    params, tel, t_norm, keep, _ = load_run(run_dir)
    sel = keep & (t_norm >= T_WARMUP)
    x = tel.x(params.budget_b)[sel]
    rate = arrival_rate_norm(tel, params, keep)[t_norm[keep] >= T_WARMUP]
    is_high, centers = classify_two_state(rate, smooth=SMOOTH_TICKS)
    dt = params.dt_poll_norm

    def eroded(mask: np.ndarray) -> np.ndarray:
        """Drop one tick on each side of a state transition (classifier
        labels lag the modulator by ~1 tick)."""
        out = mask.copy()
        out[1:] &= mask[:-1]
        out[:-1] &= mask[1:]
        return out

    fits = {}
    for state, mask in (("low", ~is_high), ("high", is_high)):
        fits[state] = {STRIDE: binned_increments(
            x, dt, stride=STRIDE, n_bins=N_BINS_EST, start_mask=mask)}
        fits[state][DRIFT_STRIDE] = binned_increments(
            x, dt, stride=DRIFT_STRIDE, n_bins=N_BINS_EST,
            start_mask=eroded(mask))
        try:
            fits[state][2 * STRIDE] = binned_increments(
                x, dt, stride=2 * STRIDE, n_bins=N_BINS_EST, start_mask=mask)
        except ValueError:
            fits[state][2 * STRIDE] = None  # no long state-pure windows

    out = {"params": json.loads(params.to_json()),
           "arrival_centers_norm": centers.tolist(),
           "high_fraction": float(is_high.mean())}
    for state in ("low", "high"):
        f1 = fits[state][STRIDE]
        f2 = fits[state][2 * STRIDE]
        fd = fits[state][DRIFT_STRIDE]
        try:
            b_hat, b_se = interior_mean_drift(fd, *ANCHOR_RANGE)
        except ValueError:
            b_hat, b_se = None, None  # no interior mass in this state
        lam_cfg = params.lam_high if state == "high" else params.lam_low
        b_cfg = lam_cfg - params.mu0
        interior = ((f1.x_centers >= ANCHOR_RANGE[0])
                    & (f1.x_centers <= ANCHOR_RANGE[1]))
        good = np.isfinite(f1.a_hat) & (f1.counts > 50) & interior
        a1 = (float(np.average(f1.a_hat[good], weights=f1.counts[good]))
              if good.any() else None)
        a2, a_corr = None, None
        if f2 is not None and a1 is not None:
            good2 = np.isfinite(f2.a_hat) & (f2.counts > 50) & interior
            if good2.any():
                a2 = float(np.average(f2.a_hat[good2],
                                      weights=f2.counts[good2]))
                a_corr = 2.0 * a2 - a1
        out[state] = {
            "b_hat": b_hat, "b_se": b_se, "b_configured": b_cfg,
            "b_rel_error": (abs(b_hat - b_cfg) / abs(b_cfg)
                            if b_hat is not None else None),
            "a_hat_d1": a1, "a_hat_d2": a2, "a_noise_corrected": a_corr,
            "x_centers": fd.x_centers.tolist(),
            "b_hat_bins": np.where(np.isfinite(fd.b_hat),
                                   fd.b_hat, np.nan).tolist(),
            "a_hat_bins": np.where(np.isfinite(f1.a_hat),
                                   f1.a_hat, np.nan).tolist(),
            "counts": fd.counts.tolist(),
            "a_counts": f1.counts.tolist(),
        }
    return out


def drift_and_diffusion(est: dict, state: str):
    """Interpolated b(x), a(x) callables from the binned estimates."""
    rec = est[state]
    xc = np.asarray(rec["x_centers"])
    bb = np.asarray(rec["b_hat_bins"], dtype=float)
    # interior bins only, both ends: the wall at 0 censors drains and the
    # clip at 1 censors builds (increments crossing the budget truncate),
    # so np.interp's constant extension carries the last interior value to
    # the walls
    ok = (np.isfinite(bb) & (np.asarray(rec["counts"]) > 50)
          & (xc >= ANCHOR_RANGE[0]) & (xc <= ANCHOR_RANGE[1]))
    xb, bv = xc[ok], bb[ok]
    if xb.size == 0:
        raise ValueError(f"no usable interior drift bins for state {state!r}")
    a_val = rec["a_noise_corrected"]
    if a_val is None or a_val <= 0:
        a_val = rec["a_hat_d2"] if rec["a_hat_d2"] is not None else rec["a_hat_d1"]
    if a_val is None:
        raise ValueError(f"no usable variance-rate bins for state {state!r}")

    def b(x, m):
        return np.interp(np.asarray(x, dtype=float), xb, bv)

    return b, float(a_val)


def burst_windows(is_high: np.ndarray, dt: float, min_units: float):
    """(start_idx, end_idx) of contiguous high runs, end exclusive."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], is_high.astype(int),
                                                 [0]])))
    out = []
    for s, e in zip(idx[::2], idx[1::2]):
        if (e - s) * dt >= min_units:
            out.append((int(s), int(e)))
    return out


def calibrate_eps(cal_dir: Path) -> float:
    params, tel, _, _, man = load_run(cal_dir)
    ep = telemetry_episode(tel, params, t0_wall=man["t0_wall"])
    proto = EventConfig(threshold=0.0, flux_window=FLUX_WINDOW,
                        lead_times=(H_MID,), t_warmup=T_WARMUP,
                        dt_decision=4.0)
    eps, record = calibrate_threshold([ep], proto, H_MID, TARGET_RATE)
    # floor at the collector's flux resolution (one message over budget in
    # one tick); an overshoot-poor calibration run would otherwise give
    # eps = 0 and trivial onsets
    params_ = RunParams.from_json(json.dumps(
        json.loads((cal_dir / "manifest.json").read_text())["params"]))
    floor = 1.0 / (params_.n_partitions * params_.budget_b
                   * params_.dt_poll_norm)
    return max(float(eps), floor)


def forecast_eval(est: dict, eps: float) -> dict:
    params, tel, t_norm, keep, _ = load_run(EVAL_DIR)
    dt = params.dt_poll_norm
    x = tel.x(params.budget_b)[keep]
    t = t_norm[keep]
    lag = tel.lag[keep]
    rate = arrival_rate_norm(tel, params, keep)
    is_high, _ = classify_two_state(rate, smooth=SMOOTH_TICKS)
    flux = overshoot_flux(lag, params.budget_b, dt)
    jbar = smoothed_flux(t, flux, FLUX_WINDOW)

    b_fn, a_val = drift_and_diffusion(est, "high")
    h = 1.0 / FP_CELLS
    centers = (np.arange(FP_CELLS) + 0.5) * h
    edges = np.linspace(0.0, 1.0, FP_CELLS + 1)
    edges[-1] += 1e-9

    bursts = []
    for s, e in burst_windows(is_high, dt, MIN_BURST_UNITS):
        if t[s] < T_WARMUP:
            continue
        horizon = min((e - s) * dt, MAX_HORIZON)
        n_ticks = min(int(round(horizon / dt)), x.shape[0] - 1 - s)
        if n_ticks < 1:
            continue
        horizon = n_ticks * dt
        pool = x[max(0, s - RHO0_TICKS + 1):s + 1].ravel()
        rho0 = np.histogram(pool, bins=edges)[0] / pool.size / h
        fp = solve_fp(rho0, b_fn, a_val, t_end=horizon, dt=2e-3,
                      dt_sample=dt)
        w1_pred, w1_pers = [], []
        for k in range(1, n_ticks + 1):
            emp = x[s + k]
            wa = np.full(emp.size, 1.0 / emp.size)
            ridx = int(np.argmin(np.abs(fp.times - k * dt)))
            rho_k = np.maximum(fp.rho[ridx, 0], 0.0)  # implicit-Euler roundoff
            rho_p = np.maximum(rho0, 0.0)
            w1_pred.append(w1_weights(emp, wa, centers, rho_k / rho_k.sum()))
            w1_pers.append(w1_weights(emp, wa, centers, rho_p / rho_p.sum()))
        # onset: same trailing smoothing applied to both flux series
        pred_flux = fp.regulator_rate[:, 0]
        pred_bar = smoothed_flux(fp.times, pred_flux, FLUX_WINDOW)
        pred_on = fp.times[pred_bar > eps]
        real_seg = jbar[s:s + n_ticks + 1]
        real_on = np.flatnonzero(real_seg > eps)
        bursts.append({
            "t_start": float(t[s]), "duration": float((e - s) * dt),
            "horizon": float(horizon),
            "w1_pred": [float(v) for v in w1_pred],
            "w1_pers": [float(v) for v in w1_pers],
            "onset_pred": float(pred_on[0]) if pred_on.size else None,
            "onset_real": float(real_on[0] * dt) if real_on.size else None,
            "x0_mean": float(pool.mean()),
        })
    return {"eps": eps, "bursts": bursts}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("estimating (b, a) from calibration telemetry (e06 heavy)...")
    est = estimate(CAL_DIR)
    for state in ("low", "high"):
        r = est[state]
        if r["b_hat"] is None:
            print(f"  {state:<5} b_hat: no interior mass "
                  f"(configured {r['b_configured']:+.4f})")
        else:
            print(f"  {state:<5} b_hat = {r['b_hat']:+.4f} +/- {r['b_se']:.4f}  "
                  f"configured = {r['b_configured']:+.4f}  "
                  f"rel err = {r['b_rel_error']:.3f}")
        def fmt(v):
            return "none" if v is None else f"{v:.5f}"
        print(f"        a: d1 = {fmt(r['a_hat_d1'])}  d2 = {fmt(r['a_hat_d2'])}"
              f"  noise-corrected = {fmt(r['a_noise_corrected'])}")
    (OUT_DIR / "identifiability.json").write_text(json.dumps(est, indent=1))

    print("calibrating flux threshold on the calibration run...")
    eps = calibrate_eps(CAL_DIR)
    print(f"  eps = {eps:.5f}")

    print("forecasting held-out bursts...")
    fc = forecast_eval(est, eps)
    (OUT_DIR / "forecast.json").write_text(json.dumps(fc, indent=1))
    for b in fc["bursts"]:
        w1m = np.mean(b["w1_pred"])
        w1p = np.mean(b["w1_pers"])
        print(f"  burst t={b['t_start']:6.1f} dur={b['duration']:5.1f}  "
              f"W1(pred)={w1m:.4f}  W1(persist)={w1p:.4f}  "
              f"onset pred/real = {b['onset_pred']}/{b['onset_real']}")
    if fc["bursts"]:
        agg_pred = float(np.mean([np.mean(b["w1_pred"]) for b in fc["bursts"]]))
        agg_pers = float(np.mean([np.mean(b["w1_pers"]) for b in fc["bursts"]]))
        print(f"aggregate: W1 pred {agg_pred:.4f} vs persistence {agg_pers:.4f}"
              f" over {len(fc['bursts'])} bursts")


if __name__ == "__main__":
    main()
