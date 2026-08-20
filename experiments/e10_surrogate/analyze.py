"""e10 analysis: accuracy, latency, and the amortization crossover.

All numbers from saved artifacts (data/e10/) plus fresh timing runs on
this machine, single-threaded on both sides (torch pinned to 1 thread to
match the numpy/scipy solver).

- Accuracy vs K: rollout each surrogate over the 40 held-out test
  trajectories to horizons {1, 2, 4, 8, 16}; W1 against the classical
  reference (dt = 2e-3), plus the boundary-flux functional error (the T4
  representation term for the learned operator).
- Classical accuracy-cost curve: the same solver at coarsened dt against
  the same reference -- the frontier the surrogate must beat.
- Latency: single forecast (batch 1) and batched ensemble (batch 128)
  rollouts to h = 8, vs classical solves.
- Crossover: N* = training CPU cost / per-forecast saving at matched
  accuracy; reported per K, with the honest framing if the classical
  solver wins at small workloads.

Saves data/e10/results.json.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from streamfno.pde.solver import solve_fp

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e10"

_spec = importlib.util.spec_from_file_location(
    "e10_run", Path(__file__).with_name("run.py"))
e10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e10)

HORIZONS = (1.0, 2.0, 4.0, 8.0, 16.0)
DT_COARSE = (2e-3, 8e-3, 3.2e-2, 1.28e-1)   # first entry = reference
H_LAT = 8.0
BATCH_ENSEMBLE = 128
N_LAT_REPS = 30
SEED = 1100


def w1_grid(u_a: np.ndarray, u_b: np.ndarray) -> float:
    """W1 between two cell-mass vectors on the unit grid."""
    return float(np.abs(np.cumsum(u_a - u_b)).sum() / u_a.size)


def boundary_flux(u: np.ndarray, b_field: np.ndarray, a_val: float) -> float:
    """The pipeline's regulator-rate functional on a cell-mass vector."""
    m = u.size
    rho_last = u[-1] * m
    return 0.5 * a_val * rho_last + max(float(b_field[-1]), 0.0) * u[-1]


def rollout(model, u0: torch.Tensor, b: torch.Tensor, log_a: torch.Tensor,
            n_steps: int) -> list[torch.Tensor]:
    """Autoregressive rollout with the pipeline's positivity/mass
    correction applied at each step (as the FP output already gets)."""
    out = []
    u = u0
    for _ in range(n_steps):
        u = model(u, b, log_a)
        u = torch.clamp(u, min=0.0)
        u = u / u.sum(dim=1, keepdim=True)
        out.append(u)
    return out


def load_models() -> dict[int, torch.nn.Module]:
    models = {}
    for k in e10.K_SWEEP:
        model = e10.DCTOperator(k)
        model.load_state_dict(torch.load(DATA_DIR / f"model_K{k}.pt",
                                         weights_only=True))
        model.eval()
        models[k] = model
    return models


def surrogate_accuracy(models, lib) -> dict:
    sel = lib["split"] == "test"
    rho = lib["rho"][sel].astype(np.float64)
    b_fields = lib["b_fields"][sel]
    a_vals = lib["a_vals"][sel]
    n_test = rho.shape[0]
    m = rho.shape[2]
    steps_of = {h: int(round(h / e10.DT_SAMPLE)) for h in HORIZONS}
    out = {}
    for k, model in models.items():
        w1 = {h: [] for h in HORIZONS}
        flux_err = {h: [] for h in HORIZONS}
        with torch.no_grad():
            for i in range(n_test):
                u0 = torch.tensor(rho[i, 0] / m, dtype=torch.float32)[None]
                b = torch.tensor(b_fields[i])[None]
                log_a = torch.log(torch.tensor([a_vals[i]]))
                traj = rollout(model, u0, b, log_a, steps_of[HORIZONS[-1]])
                for h in HORIZONS:
                    u_pred = traj[steps_of[h] - 1][0].numpy().astype(float)
                    u_ref = rho[i, steps_of[h]] / m
                    w1[h].append(w1_grid(u_pred, u_ref))
                    flux_err[h].append(abs(
                        boundary_flux(u_pred, b_fields[i], a_vals[i])
                        - boundary_flux(u_ref, b_fields[i], a_vals[i])))
        out[k] = {
            "w1_mean": {str(h): float(np.mean(w1[h])) for h in HORIZONS},
            "flux_err_mean": {str(h): float(np.mean(flux_err[h]))
                              for h in HORIZONS},
        }
        print(f"  K={k}: W1@h8 {out[k]['w1_mean']['8.0']:.5f}  "
              f"flux@h8 {out[k]['flux_err_mean']['8.0']:.6f}")
    return out


def classical_accuracy(lib) -> dict:
    """Coarse-dt classical solves vs the dt = 2e-3 reference, on the
    same test trajectories -- the classical side of the frontier."""
    sel = lib["split"] == "test"
    rho = lib["rho"][sel].astype(np.float64)
    b_fields = lib["b_fields"][sel]
    a_vals = lib["a_vals"][sel]
    m = rho.shape[2]
    x = (np.arange(m) + 0.5) / m
    steps_of = {h: int(round(h / e10.DT_SAMPLE)) for h in HORIZONS}
    out = {}
    for dt in DT_COARSE[1:]:
        w1 = {h: [] for h in HORIZONS}
        for i in range(rho.shape[0]):
            bf = b_fields[i]

            def b_fn(xv, mm, _bf=bf):
                return np.interp(np.asarray(xv, dtype=float), x, _bf)

            fp = solve_fp(rho[i, 0], b_fn, float(a_vals[i]),
                          t_end=HORIZONS[-1], dt=dt,
                          dt_sample=e10.DT_SAMPLE)
            for h in HORIZONS:
                idx = int(np.argmin(np.abs(fp.times - h)))
                u_pred = np.maximum(fp.rho[idx, 0], 0.0) / m
                u_pred /= u_pred.sum()
                w1[h].append(w1_grid(u_pred, rho[i, steps_of[h]] / m))
        out[f"{dt:g}"] = {"w1_mean": {str(h): float(np.mean(w1[h]))
                                      for h in HORIZONS}}
        print(f"  classical dt={dt:g}: W1@h8 "
              f"{out[f'{dt:g}']['w1_mean']['8.0']:.5f}")
    return out


N_WARMUP = 3


def _timed(fn, reps: int) -> dict:
    """Benchmark protocol: N_WARMUP untimed warm-up calls, then ``reps``
    timed calls; p50 and p95 wall milliseconds."""
    for _ in range(N_WARMUP):
        fn()
    lat = np.empty(reps)
    for r in range(reps):
        t0 = time.perf_counter_ns()
        fn()
        lat[r] = time.perf_counter_ns() - t0
    return {"p50_ms": float(np.percentile(lat, 50) / 1e6),
            "p95_ms": float(np.percentile(lat, 95) / 1e6)}


def latency(models, lib) -> dict:
    torch.set_num_threads(1)
    sel = lib["split"] == "test"
    rho = lib["rho"][sel]
    b_fields = lib["b_fields"][sel]
    a_vals = lib["a_vals"][sel]
    m = rho.shape[2]
    x = (np.arange(m) + 0.5) / m
    n_steps = int(round(H_LAT / e10.DT_SAMPLE))
    out = {"h": H_LAT, "protocol": {
        "warmup_reps": N_WARMUP, "timed_reps": N_LAT_REPS,
        "torch_threads": 1,
        "classical_precision": "float64 (numpy/LAPACK; the banded solve "
                               "is single-threaded dgtsv)",
        "surrogate_precision": "float32 as trained; float64 variant "
                               "reported alongside",
        "device": "CPU only; no host/device transfers on either engine",
        "scope": "inference_only = rollout alone; end_to_end adds rho0 "
                 "tensor prep and the boundary-flux threshold scan "
                 "(classical solves accumulate the regulator internally; "
                 "their threshold scan costs ~0.02 ms, see e09)"}}

    # classical: single solve at each dt (float64 end to end)
    bf0 = b_fields[0]

    def b_fn(xv, mm, _bf=bf0):
        return np.interp(np.asarray(xv, dtype=float), x, _bf)

    for dt in DT_COARSE:
        r = _timed(lambda dt=dt: solve_fp(
            rho[0, 0].astype(float), b_fn, float(a_vals[0]),
            t_end=H_LAT, dt=dt, dt_sample=e10.DT_SAMPLE), N_LAT_REPS)
        out[f"classical_dt{dt:g}_ms"] = r["p50_ms"]
        out[f"classical_dt{dt:g}_p95_ms"] = r["p95_ms"]

    # surrogate: batch-1 and batch-128 rollouts, fp32 and fp64
    for k, model in models.items():
        model64 = e10.DCTOperator(k).double()
        model64.load_state_dict({kk: v.double()
                                 for kk, v in model.state_dict().items()})
        model64.eval()
        with torch.no_grad():
            u1 = torch.tensor(rho[0, 0] / m, dtype=torch.float32)[None]
            b1 = torch.tensor(b_fields[0])[None]
            a1 = torch.log(torch.tensor([a_vals[0]]))
            single = _timed(lambda: rollout(model, u1, b1, a1, n_steps),
                            N_LAT_REPS)
            single64 = _timed(
                lambda: rollout(model64, u1.double(), b1.double(),
                                a1.double(), n_steps), N_LAT_REPS)

            def end_to_end():
                u = torch.tensor(rho[0, 0] / m, dtype=torch.float32)[None]
                traj = rollout(model, u, b1, a1, n_steps)
                flux = np.array([boundary_flux(t[0].numpy(), b_fields[0],
                                               a_vals[0]) for t in traj])
                return flux > 1e-4

            e2e = _timed(end_to_end, N_LAT_REPS)

            ub = u1.expand(BATCH_ENSEMBLE, -1).contiguous()
            bb = b1.expand(BATCH_ENSEMBLE, -1).contiguous()
            ab = a1.expand(BATCH_ENSEMBLE).contiguous()
            batch = _timed(lambda: rollout(model, ub, bb, ab, n_steps),
                           max(N_LAT_REPS // 3, 5))
        out[f"surrogate_K{k}_single_ms"] = single["p50_ms"]
        out[f"surrogate_K{k}_single_p95_ms"] = single["p95_ms"]
        out[f"surrogate_K{k}_single_fp64_ms"] = single64["p50_ms"]
        out[f"surrogate_K{k}_end_to_end_ms"] = e2e["p50_ms"]
        out[f"surrogate_K{k}_batch{BATCH_ENSEMBLE}_per_member_ms"] = (
            batch["p50_ms"] / BATCH_ENSEMBLE)
        print(f"  K={k}: single {single['p50_ms']:.2f} ms (p95 "
              f"{single['p95_ms']:.2f}, fp64 {single64['p50_ms']:.2f}, "
              f"e2e {e2e['p50_ms']:.2f}), batched "
              f"{batch['p50_ms'] / BATCH_ENSEMBLE:.3f} ms/member")
    return out


def crossover(acc_sur, acc_cls, lat, training) -> dict:
    """Matched-accuracy comparison at h = 8 and the amortization point.

    The classical latency at the surrogate's accuracy is interpolated on
    the classical (log W1, log latency) curve; the crossover
    N* = train_cpu / (classical_ms - surrogate_ms) counts the forecast
    evaluations after which the surrogate's amortized cost is lower.
    """
    cls_pts = []
    for dt in DT_COARSE[1:]:
        cls_pts.append((acc_cls[f"{dt:g}"]["w1_mean"]["8.0"],
                        lat[f"classical_dt{dt:g}_ms"]))
    cls_pts.sort()
    w1s = np.log([p[0] for p in cls_pts])
    ms = np.log([p[1] for p in cls_pts])
    ref_ms = lat[f"classical_dt{DT_COARSE[0]:g}_ms"]
    out = {}
    for k in (int(s) for s in training["models"]):
        w1_k = acc_sur[k]["w1_mean"]["8.0"]
        matched_ms = (float(np.exp(np.interp(np.log(w1_k), w1s, ms)))
                      if w1_k >= np.exp(w1s[0]) else ref_ms)
        train_cpu_ms = training["models"][str(k)]["train_cpu_s"] * 1e3
        row = {"w1_h8": w1_k, "classical_ms_at_matched_accuracy": matched_ms,
               "train_cpu_s": train_cpu_ms / 1e3}
        for tag, key in (("single", f"surrogate_K{k}_single_ms"),
                         ("batched",
                          f"surrogate_K{k}_batch{BATCH_ENSEMBLE}"
                          "_per_member_ms")):
            saving = matched_ms - lat[key]
            row[f"{tag}_ms"] = lat[key]
            row[f"{tag}_crossover_evals"] = (
                float(np.ceil(train_cpu_ms / saving)) if saving > 0
                else None)
        out[str(k)] = row
    return out


def main() -> None:
    training = json.loads((DATA_DIR / "training.json").read_text())
    with np.load(DATA_DIR / "trajectories.npz") as f:
        lib = {k: f[k] for k in f.files}
    models = load_models()

    print("surrogate accuracy on held-out trajectories...")
    acc_sur = surrogate_accuracy(models, lib)
    # precision equivalence: the fp32 weights evaluated in fp64 must give
    # the same test error (the fp32/fp64 latency gap is then hardware
    # cost, not accuracy trade)
    k_chk = max(models)
    m64 = e10.DCTOperator(k_chk).double()
    m64.load_state_dict({kk: v.double()
                         for kk, v in models[k_chk].state_dict().items()})
    m64.eval()
    acc64 = surrogate_accuracy({k_chk: m64}, lib)
    fp64_gap = abs(acc64[k_chk]["w1_mean"]["8.0"]
                   - acc_sur[k_chk]["w1_mean"]["8.0"])
    print(f"  fp64 accuracy gap at K={k_chk}: {fp64_gap:.2e}")

    print("classical coarse-dt accuracy...")
    acc_cls = classical_accuracy(lib)
    print("latency (single-threaded both sides)...")
    lat = latency(models, lib)
    print("crossover...")
    cross = crossover(acc_sur, acc_cls, lat, training)
    for k, row in cross.items():
        print(f"  K={k}: W1 {row['w1_h8']:.5f}  matched classical "
              f"{row['classical_ms_at_matched_accuracy']:.1f} ms  "
              f"single {row['single_ms']:.2f} ms "
              f"(N* {row['single_crossover_evals']})  batched "
              f"{row['batched_ms']:.3f} ms "
              f"(N* {row['batched_crossover_evals']})")

    results = {
        "seed": SEED, "horizons": list(HORIZONS),
        "dt_coarse": list(DT_COARSE), "h_latency": H_LAT,
        "batch_ensemble": BATCH_ENSEMBLE,
        "surrogate_accuracy": {str(k): v for k, v in acc_sur.items()},
        "classical_accuracy": acc_cls,
        "latency": lat,
        "crossover": cross,
        "training": {k: {kk: vv for kk, vv in v.items()
                         if kk != "val_mse_curve"}
                     for k, v in training["models"].items()},
    }
    (DATA_DIR / "results.json").write_text(json.dumps(results, indent=1))
    print(f"saved {DATA_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
