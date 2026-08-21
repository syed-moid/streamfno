"""e10b (revision C1): direct-horizon surrogate.

Trains a DIRECT map (rho_t, b, a) -> rho_{t+8 units} -- one forward
pass to the mid horizon, eliminating the autoregressive accumulation
identified as the binding constraint in e10. Same library, same
trajectory-level splits, same architecture family (DCT spectral core,
K in {4,8,16,32}), same epochs/optimizer, same audit standards. The
horizon is fixed at h = 8 (no h-conditioning; stated in the text).

Two-phase: `train` (CPU-heavy, can run alongside cluster work) and
`eval` (timing-sensitive; run on a quiet machine). Saves
data/e10/direct_training.json, model_direct_K{K}.pt, and -- after eval
-- direct_results.json.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e10"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"e10_{name}", Path(__file__).with_name(f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e10_run = _load("run")
e10_analyze = _load("analyze")

H_TARGET = 8.0
H_STEPS = int(round(H_TARGET / e10_run.DT_SAMPLE))   # 32 samples ahead
K_SWEEP = e10_run.K_SWEEP
N_LAT_REPS = 30
N_WARMUP = 3


def make_direct_pairs(lib, split_name):
    sel = lib["split"] == split_name
    rho = torch.tensor(lib["rho"][sel])
    n_traj, n_samp, m = rho.shape
    u = rho / m
    b = torch.tensor(lib["b_fields"][sel])
    log_a = torch.log(torch.tensor(lib["a_vals"][sel]))
    n_pairs = n_samp - H_STEPS
    src = u[:, :n_pairs].reshape(-1, m)
    dst = u[:, H_STEPS:].reshape(-1, m)
    b_rep = b[:, None, :].expand(-1, n_pairs, -1).reshape(-1, m)
    a_rep = log_a[:, None].expand(-1, n_pairs).reshape(-1)
    return src, dst, b_rep, a_rep


def train_direct(k_modes: int, lib) -> dict:
    torch.manual_seed(e10_run.SEED + 500 + k_modes)
    model = e10_run.DCTOperator(k_modes)
    opt = torch.optim.Adam(model.parameters(), lr=e10_run.LR)
    src, dst, b_rep, a_rep = make_direct_pairs(lib, "train")
    vs, vd, vb, va = make_direct_pairs(lib, "val")
    n = src.shape[0]
    gen = torch.Generator().manual_seed(e10_run.SEED + 500 + k_modes)
    t0, c0 = time.time(), time.process_time()
    val_curve = []
    for epoch in range(e10_run.EPOCHS):
        perm = torch.randperm(n, generator=gen)
        model.train()
        for lo in range(0, n, e10_run.BATCH):
            idx = perm[lo:lo + e10_run.BATCH]
            pred = model(src[idx], b_rep[idx], a_rep[idx])
            loss = ((pred - dst[idx]) ** 2).mean() * e10_run.M_CELLS ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float((((model(vs, vb, va) - vd) ** 2).mean()
                         * e10_run.M_CELLS ** 2))
        val_curve.append(val)
        print(f"  direct K={k_modes} epoch {epoch + 1}/{e10_run.EPOCHS} "
              f"val {val:.5f}", flush=True)
    torch.save(model.state_dict(), DATA_DIR / f"model_direct_K{k_modes}.pt")
    return {"k_modes": k_modes, "val_mse_curve": val_curve,
            "train_wall_s": time.time() - t0,
            "train_cpu_s": time.process_time() - c0,
            "n_train_pairs": int(n), "h_target": H_TARGET}


def do_train() -> None:
    with np.load(DATA_DIR / "trajectories.npz") as f:
        lib = {k: f[k] for k in f.files}
    records = {}
    for k in K_SWEEP:
        print(f"training direct K = {k}...", flush=True)
        records[str(k)] = train_direct(k, lib)
    (DATA_DIR / "direct_training.json").write_text(
        json.dumps({"h_target": H_TARGET, "models": records}, indent=1))
    print("direct training saved", flush=True)


def do_eval() -> None:
    torch.set_num_threads(1)
    with np.load(DATA_DIR / "trajectories.npz") as f:
        lib = {k: f[k] for k in f.files}
    sel = lib["split"] == "test"
    rho = lib["rho"][sel].astype(np.float64)
    b_fields = lib["b_fields"][sel]
    a_vals = lib["a_vals"][sel]
    m = rho.shape[2]
    training = json.loads((DATA_DIR / "direct_training.json").read_text())
    prior = json.loads((DATA_DIR / "results.json").read_text())

    out = {"h_target": H_TARGET, "models": {}}
    for k in K_SWEEP:
        model = e10_run.DCTOperator(k)
        model.load_state_dict(torch.load(
            DATA_DIR / f"model_direct_K{k}.pt", weights_only=True))
        model.eval()
        w1s, fluxes = [], []
        with torch.no_grad():
            for i in range(rho.shape[0]):
                u0 = torch.tensor(rho[i, 0] / m,
                                  dtype=torch.float32)[None]
                b = torch.tensor(b_fields[i])[None]
                la = torch.log(torch.tensor([a_vals[i]]))
                pred = model(u0, b, la)
                pred = torch.clamp(pred, min=0.0)
                pred = pred / pred.sum(dim=1, keepdim=True)
                u_pred = pred[0].numpy().astype(float)
                u_ref = rho[i, H_STEPS] / m
                w1s.append(e10_analyze.w1_grid(u_pred, u_ref))
                fluxes.append(abs(
                    e10_analyze.boundary_flux(u_pred, b_fields[i],
                                              a_vals[i])
                    - e10_analyze.boundary_flux(u_ref, b_fields[i],
                                                a_vals[i])))
            # latency: single forward pass, warm-ups excluded, p50/p95
            u1 = torch.tensor(rho[0, 0] / m, dtype=torch.float32)[None]
            b1 = torch.tensor(b_fields[0])[None]
            a1 = torch.log(torch.tensor([a_vals[0]]))
            for _ in range(N_WARMUP):
                model(u1, b1, a1)
            lat = np.empty(N_LAT_REPS)
            for r in range(N_LAT_REPS):
                t0 = time.perf_counter_ns()
                model(u1, b1, a1)
                lat[r] = time.perf_counter_ns() - t0
            ub = u1.expand(128, -1).contiguous()
            bb = b1.expand(128, -1).contiguous()
            ab = a1.expand(128).contiguous()
            for _ in range(N_WARMUP):
                model(ub, bb, ab)
            latb = np.empty(max(N_LAT_REPS // 3, 5))
            for r in range(latb.size):
                t0 = time.perf_counter_ns()
                model(ub, bb, ab)
                latb[r] = time.perf_counter_ns() - t0
        rec = {
            "w1_h8_mean": float(np.mean(w1s)),
            "flux_err_h8_mean": float(np.mean(fluxes)),
            "single_p50_ms": float(np.percentile(lat, 50) / 1e6),
            "single_p95_ms": float(np.percentile(lat, 95) / 1e6),
            "batch128_per_member_ms": float(np.percentile(latb, 50)
                                            / 1e6 / 128),
            "train_cpu_s": training["models"][str(k)]["train_cpu_s"],
        }
        out["models"][str(k)] = rec
        print(f"  direct K={k}: W1@h8 {rec['w1_h8_mean']:.5f}  single "
              f"{rec['single_p50_ms']:.2f} ms  batched "
              f"{rec['batch128_per_member_ms']:.3f} ms/member", flush=True)

    # matched-accuracy comparison against the classical coarse-dt curve
    cls = [(prior["classical_accuracy"][f"{dt:g}"]["w1_mean"]["8.0"],
            prior["latency"][f"classical_dt{dt:g}_ms"])
           for dt in prior["dt_coarse"][1:]]
    cls.sort()
    w1c = np.log([c[0] for c in cls])
    msc = np.log([c[1] for c in cls])
    for k, rec in out["models"].items():
        w1k = rec["w1_h8_mean"]
        matched = (float(np.exp(np.interp(np.log(w1k), w1c, msc)))
                   if w1k >= np.exp(w1c[0]) else cls[0][1])
        rec["classical_ms_at_matched_accuracy"] = matched
        saving = matched - rec["single_p50_ms"]
        rec["crossover_evals"] = (
            float(np.ceil(rec["train_cpu_s"] * 1e3 / saving))
            if saving > 0 else None)
    (DATA_DIR / "direct_results.json").write_text(json.dumps(out, indent=1))
    print(f"saved {DATA_DIR / 'direct_results.json'}", flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        do_train()
    elif mode == "eval":
        do_eval()
    else:
        raise SystemExit(f"unknown mode {mode!r}")
