"""e10 learning curve (checkpoint-1 audit item 0.4): test error vs
training-set size for the best-K surrogate.

The library is extended deterministically -- the same seeded rng stream
regenerates the original 320 trajectories bit-for-bit and continues to
1360, so the held-out val/test sets (original indices 240-279 / 280-319)
are unchanged across every point of the curve.  Train-pool = original
240 train trajectories + the 1040 new ones; sizes 160/320/640/1280 are
nested prefixes of that pool.  One model per size, same architecture,
epochs, and optimizer as the main K sweep.

Saves data/e10/trajectories_xl.npz, model_K{K}_n{size}.pt, and
learning_curve.json.
"""

import importlib.util
import json
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

N_TRAJ_XL = 1360
TRAIN_SIZES = (160, 320, 640, 1280)
K_BEST = 32
H_KEY = "8.0"


def extended_library() -> dict:
    path = DATA_DIR / "trajectories_xl.npz"
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    n_samp = int(e10_run.T_TRAJ / e10_run.DT_SAMPLE) + 1
    rng = np.random.default_rng(e10_run.SEED)
    rho = np.empty((N_TRAJ_XL, n_samp, e10_run.M_CELLS), dtype=np.float32)
    b_fields = np.empty((N_TRAJ_XL, e10_run.M_CELLS), dtype=np.float32)
    a_vals = np.empty(N_TRAJ_XL, dtype=np.float32)
    t0 = time.time()
    for i in range(N_TRAJ_XL):
        r, b, a = e10_run.sample_trajectory(rng)
        rho[i], b_fields[i], a_vals[i] = r, b, a
        if (i + 1) % 160 == 0:
            print(f"  {i + 1}/{N_TRAJ_XL} trajectories "
                  f"({time.time() - t0:.0f} s)")
    # consistency guard: the prefix must equal the original library
    with np.load(DATA_DIR / "trajectories.npz") as f:
        assert np.array_equal(f["rho"], rho[:f["rho"].shape[0]]), \
            "extended library diverges from the original prefix"
    lib = {"rho": rho, "b_fields": b_fields, "a_vals": a_vals,
           "gen_seconds": np.array(time.time() - t0)}
    np.savez_compressed(path, **lib)
    return lib


def lib_view(lib: dict, train_size: int) -> dict:
    """Split labels for one curve point: nested train prefix from the
    pool (0-239, 320...), original val/test untouched."""
    split = np.array(["unused"] * N_TRAJ_XL, dtype=object)
    split[240:280] = "val"
    split[280:320] = "test"
    pool = list(range(240)) + list(range(320, N_TRAJ_XL))
    split[pool[:train_size]] = "train"
    return {**lib, "split": np.array([str(s) for s in split])}


def main() -> None:
    lib = extended_library()
    sweep_model = DATA_DIR / f"model_K{K_BEST}.pt"
    sweep_bytes = (sweep_model.read_bytes() if sweep_model.exists()
                   else None)  # restored verbatim afterwards
    curve = {}
    for size in TRAIN_SIZES:
        view = lib_view(lib, size)
        dst = DATA_DIR / f"model_K{K_BEST}_n{size}.pt"
        if dst.exists():
            print(f"K={K_BEST} n={size}: resuming from saved model")
            rec = None
        else:
            print(f"training K={K_BEST} on {size} trajectories...")
            rec = e10_run.train_one(K_BEST, view)
            (DATA_DIR / f"model_K{K_BEST}.pt").replace(dst)
        model = e10_run.DCTOperator(K_BEST)
        model.load_state_dict(torch.load(dst, weights_only=True))
        model.eval()
        acc = e10_analyze.surrogate_accuracy({K_BEST: model}, view)
        curve[str(size)] = {
            "w1_h8": acc[K_BEST]["w1_mean"][H_KEY],
            "flux_err_h8": acc[K_BEST]["flux_err_mean"][H_KEY],
            "train_wall_s": rec["train_wall_s"] if rec else None,
            "train_cpu_s": rec["train_cpu_s"] if rec else None,
            "final_val_mse": rec["val_mse_curve"][-1] if rec else None,
            "resumed": rec is None,
        }
        print(f"  n={size}: test W1@h8 {curve[str(size)]['w1_h8']:.5f}")
    if sweep_bytes is not None:
        sweep_model.write_bytes(sweep_bytes)  # put the K-sweep model back
    elif not sweep_model.exists():
        # sweep model lost to an interrupted earlier invocation: retrain
        # it on the original library (same seed and procedure as the sweep)
        print(f"retraining the K={K_BEST} sweep model (original library)...")
        with np.load(DATA_DIR / "trajectories.npz") as f:
            e10_run.train_one(K_BEST, {k: f[k] for k in f.files})
    (DATA_DIR / "learning_curve.json").write_text(json.dumps({
        "k_modes": K_BEST, "train_sizes": list(TRAIN_SIZES),
        "n_traj_xl": N_TRAJ_XL, "curve": curve}, indent=1))
    print(f"saved {DATA_DIR / 'learning_curve.json'}")


if __name__ == "__main__":
    main()
