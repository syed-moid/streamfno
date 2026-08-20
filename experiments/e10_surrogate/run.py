"""e10: DCT spectral operator as a measured surrogate for the classical
Chang-Cooper solver.  Efficiency study only -- the predictive ceiling is
Phase C's; the surrogate targets the same forecast, faster.

Part 1 (this script): generate the PDE trajectory library over the
Phase B/C coefficient family and train one surrogate per retained-mode
count K in {4, 8, 16, 32}.

Library: 320 trajectories on the online pipeline's 128-cell grid, each a
classical solve (dt = 2e-3) sampled at the 0.25-unit collector cadence
for 16 units.  Per trajectory: drift field b(x) = b0 + b1 (x - 1/2) +
p cos(pi q x) spanning the configured netput range of the sim configs
(lam in [0.35, 0.90] minus mu0 = 0.7), variance rate a log-uniform over
[5e-4, 0.08] (spanning the simulator's 0.05 and the real cluster's
~7e-4), and initial densities mixing an idle-side exponential atom with
the e02 truncated-Gaussian family.  Strict trajectory-level
train/val/test split (240/40/40), split before any training.

Model: cell-mass representation u = rho / M; channels (u_t, b(x),
log a, x) lifted to width 32; three DCT spectral blocks (orthonormal
cosine transform, K retained modes, per-mode dense mixing) each with a
pointwise linear skip; residual output u_{t+dh} = u_t + net(.).  MSE
loss.  Training wall-clock and CPU-seconds are recorded per K -- they
are the amortization numerator.

Saves data/e10/trajectories.npz, data/e10/model_K{K}.pt,
data/e10/training.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from streamfno.matching import truncated_gaussian_density
from streamfno.pde.solver import solve_fp

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e10"

SEED = 1000
M_CELLS = 128
DT_SOLVE = 2e-3
DT_SAMPLE = 0.25
T_TRAJ = 16.0
N_TRAJ = 320
SPLIT = (240, 40, 40)            # train / val / test, by trajectory
B0_RANGE = (-0.35, 0.20)         # lam in [0.35, 0.90] minus mu0 = 0.7
B1_RANGE = (-0.10, 0.10)
P_MAX = 0.03
LOG_A_RANGE = (np.log(5e-4), np.log(0.08))
K_SWEEP = (4, 8, 16, 32)
WIDTH = 32
N_BLOCKS = 3
EPOCHS = 20
BATCH = 256
LR = 1e-3


def sample_trajectory(rng: np.random.Generator):
    x = (np.arange(M_CELLS) + 0.5) / M_CELLS
    b0 = rng.uniform(*B0_RANGE)
    b1 = rng.uniform(*B1_RANGE)
    p, q = rng.uniform(0.0, P_MAX), rng.integers(1, 4)
    b_field = b0 + b1 * (x - 0.5) + p * np.cos(np.pi * q * x)
    a_val = float(np.exp(rng.uniform(*LOG_A_RANGE)))

    w_atom = rng.uniform(0.0, 0.7)
    scale = rng.uniform(0.01, 0.05)
    atom = np.exp(-x / scale)
    atom /= atom.sum() / M_CELLS
    bulk = truncated_gaussian_density(rng.uniform(0.05, 0.6),
                                      rng.uniform(0.03, 0.15), M_CELLS)
    rho0 = w_atom * atom + (1.0 - w_atom) * bulk

    def b_fn(xv, m):
        return np.interp(np.asarray(xv, dtype=float), x, b_field)

    fp = solve_fp(rho0, b_fn, a_val, t_end=T_TRAJ, dt=DT_SOLVE,
                  dt_sample=DT_SAMPLE)
    return fp.rho[:, 0, :], b_field, a_val


def generate_library() -> dict:
    path = DATA_DIR / "trajectories.npz"
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    rng = np.random.default_rng(SEED)
    rho = np.empty((N_TRAJ, int(T_TRAJ / DT_SAMPLE) + 1, M_CELLS),
                   dtype=np.float32)
    b_fields = np.empty((N_TRAJ, M_CELLS), dtype=np.float32)
    a_vals = np.empty(N_TRAJ, dtype=np.float32)
    t0 = time.time()
    for i in range(N_TRAJ):
        r, b, a = sample_trajectory(rng)
        rho[i], b_fields[i], a_vals[i] = r, b, a
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{N_TRAJ} trajectories "
                  f"({time.time() - t0:.0f} s)")
    split = np.array(["train"] * SPLIT[0] + ["val"] * SPLIT[1]
                     + ["test"] * SPLIT[2])
    lib = {"rho": rho, "b_fields": b_fields, "a_vals": a_vals,
           "split": split,
           "gen_seconds": np.array(time.time() - t0)}
    np.savez_compressed(path, **lib)
    print(f"  library saved ({time.time() - t0:.0f} s)")
    return lib


def dct_basis(m: int) -> torch.Tensor:
    """Orthonormal DCT-II basis Phi (m x m): Phi @ Phi.T = I,
    coefficients = v @ Phi.T, reconstruction = c @ Phi."""
    n = np.arange(m)
    k = np.arange(m)[:, None]
    phi = np.cos(np.pi * k * (2 * n + 1) / (2 * m))
    phi *= np.sqrt(2.0 / m)
    phi[0] /= np.sqrt(2.0)
    return torch.tensor(phi, dtype=torch.float32)


class SpectralBlock(nn.Module):
    def __init__(self, width: int, k_modes: int, phi: torch.Tensor):
        super().__init__()
        self.k = k_modes
        self.register_buffer("phi", phi)
        self.weight = nn.Parameter(
            torch.randn(k_modes, width, width) / width)
        self.skip = nn.Conv1d(width, width, 1)

    def forward(self, v):
        # v: (B, C, M) -> coefficients (B, C, K) -> mode-wise dense mix
        coeff = torch.einsum("bcm,km->bck", v, self.phi[:self.k])
        mixed = torch.einsum("bck,kcd->bdk", coeff, self.weight)
        out = torch.einsum("bdk,km->bdm", mixed, self.phi[:self.k])
        return torch.nn.functional.gelu(out + self.skip(v))


class DCTOperator(nn.Module):
    """u_{t+dh} = u_t + net(u_t, b, log a, x): a small DCT-core neural
    operator on the cell-mass representation."""

    def __init__(self, k_modes: int, width: int = WIDTH,
                 n_blocks: int = N_BLOCKS, m_cells: int = M_CELLS):
        super().__init__()
        phi = dct_basis(m_cells)
        self.lift = nn.Conv1d(4, width, 1)
        self.blocks = nn.ModuleList(
            SpectralBlock(width, k_modes, phi) for _ in range(n_blocks))
        self.head = nn.Sequential(nn.Conv1d(width, width, 1), nn.GELU(),
                                  nn.Conv1d(width, 1, 1))
        x = (torch.arange(m_cells, dtype=torch.float32) + 0.5) / m_cells
        self.register_buffer("x_grid", x)

    def forward(self, u, b_field, log_a):
        # u, b_field: (B, M); log_a: (B,)
        batch = u.shape[0]
        feats = torch.stack([
            u * u.shape[1],                       # back to density scale
            b_field / 0.3,
            (log_a[:, None] / 3.0).expand(-1, u.shape[1]),
            self.x_grid.expand(batch, -1),
        ], dim=1)
        v = self.lift(feats)
        for blk in self.blocks:
            v = blk(v)
        return u + self.head(v)[:, 0, :] / u.shape[1]


def make_pairs(lib, split_name):
    sel = lib["split"] == split_name
    rho = torch.tensor(lib["rho"][sel])            # (T, S, M)
    n_traj, n_samp, m = rho.shape
    u = rho / m                                    # cell masses
    b = torch.tensor(lib["b_fields"][sel])
    log_a = torch.log(torch.tensor(lib["a_vals"][sel]))
    src = u[:, :-1].reshape(-1, m)
    dst = u[:, 1:].reshape(-1, m)
    b_rep = b[:, None, :].expand(-1, n_samp - 1, -1).reshape(-1, m)
    a_rep = log_a[:, None].expand(-1, n_samp - 1).reshape(-1)
    return src, dst, b_rep, a_rep


def train_one(k_modes: int, lib) -> dict:
    torch.manual_seed(SEED + k_modes)
    model = DCTOperator(k_modes)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    src, dst, b_rep, a_rep = make_pairs(lib, "train")
    vs, vd, vb, va = make_pairs(lib, "val")
    n = src.shape[0]
    gen = torch.Generator().manual_seed(SEED + k_modes)
    t0, c0 = time.time(), time.process_time()
    val_curve = []
    for epoch in range(EPOCHS):
        perm = torch.randperm(n, generator=gen)
        model.train()
        for lo in range(0, n, BATCH):
            idx = perm[lo:lo + BATCH]
            pred = model(src[idx], b_rep[idx], a_rep[idx])
            loss = ((pred - dst[idx]) ** 2).mean() * M_CELLS ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float((((model(vs, vb, va) - vd) ** 2).mean()
                         * M_CELLS ** 2))
        val_curve.append(val)
        print(f"  K={k_modes} epoch {epoch + 1}/{EPOCHS} val {val:.5f}")
    wall, cpu = time.time() - t0, time.process_time() - c0
    n_params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), DATA_DIR / f"model_K{k_modes}.pt")
    return {"k_modes": k_modes, "val_mse_curve": val_curve,
            "train_wall_s": wall, "train_cpu_s": cpu,
            "n_params": n_params, "epochs": EPOCHS,
            "n_train_pairs": int(n)}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("generating trajectory library...")
    lib = generate_library()
    records = {}
    for k in K_SWEEP:
        print(f"training K = {k}...")
        records[str(k)] = train_one(k, lib)
    (DATA_DIR / "training.json").write_text(json.dumps({
        "seed": SEED, "m_cells": M_CELLS, "dt_solve": DT_SOLVE,
        "dt_sample": DT_SAMPLE, "t_traj": T_TRAJ, "n_traj": N_TRAJ,
        "split": list(SPLIT), "width": WIDTH, "n_blocks": N_BLOCKS,
        "batch": BATCH, "lr": LR,
        "library_gen_seconds": float(lib["gen_seconds"]),
        "models": records}, indent=1))
    print(f"saved models and {DATA_DIR / 'training.json'}")


if __name__ == "__main__":
    main()
