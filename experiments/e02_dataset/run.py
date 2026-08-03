"""e02: episode library for the predictability-horizon experiments.

Generates the bursty-workload episode library at three load levels (burst
frequency), applies the telemetry observation layer, calibrates the event
threshold on training episodes of the middle level, and saves everything
under data/e02/ (episodes, manifest, event config, per-level base rates,
matched PDE reference trajectories).
"""

import json
import time
from pathlib import Path

import numpy as np

from streamfno.events import EventConfig, calibrate_threshold, label_episode
from streamfno.matching import drift_from_config, truncated_gaussian_density
from streamfno.obs import Episode, ObsConfig, observe
from streamfno.pde.solver import solve_fp
from streamfno.sim import SimConfig, simulate

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e02"

N_EPISODES = 130          # per level
SPLIT = (78, 26, 26)      # train / val / test, by episode
LEVELS = {"light": 0.03, "moderate": 0.06, "heavy": 0.10}  # burst rate r_low_high
LEAD_TIMES = (1.0, 2.0, 4.0, 8.0, 16.0, 24.0)
H_MID = 8.0
TARGET_RATE = 0.10
SEED_BASE = {"light": 0, "moderate": 2000, "heavy": 4000}
NOISE_OFFSET = 100_000    # noise stream never collides with sim seeds

OBS = ObsConfig(dt_obs=1.0, noise_std=0.02, hist_bins=8, noise_seed=0)


def sim_config(level: str, seed: int) -> SimConfig:
    return SimConfig(
        n_partitions=400, buffer_depth=128, t_end=120.0, seed=seed,
        dt_sample=0.5, n_bins=128, mode="diffusive", a=0.05,
        arrival="mmpp", mmpp_shared=True, lam_low=0.40, lam_high=0.82,
        r_low_high=LEVELS[level], r_high_low=0.30, mu0=0.7, n_brokers=2,
        degradation_drop=0.3, init_x0=0.2, init_sd=0.05,
    )


def episode_path(level: str, i: int) -> Path:
    return DATA_DIR / "episodes" / level / f"ep_{i:04d}.npz"


def generate_episodes() -> list[dict]:
    manifest = []
    for level in LEVELS:
        for i in range(N_EPISODES):
            seed = SEED_BASE[level] + i
            split = ("train" if i < SPLIT[0]
                     else "val" if i < SPLIT[0] + SPLIT[1] else "test")
            path = episode_path(level, i)
            if not path.exists():
                cfg = sim_config(level, seed)
                obs_cfg = ObsConfig(dt_obs=OBS.dt_obs, noise_std=OBS.noise_std,
                                    hist_bins=OBS.hist_bins,
                                    noise_seed=NOISE_OFFSET + seed)
                ep = observe(simulate(cfg), obs_cfg,
                             meta={"level": level, "split": split, "index": i})
                ep.save(path)
            manifest.append({"level": level, "index": i, "split": split,
                             "sim_seed": seed,
                             "noise_seed": NOISE_OFFSET + seed,
                             "path": str(path.relative_to(DATA_DIR))})
        print(f"  {level}: {N_EPISODES} episodes")
    return manifest


def calibrate(manifest) -> EventConfig:
    proto = EventConfig(threshold=0.0, flux_window=2.0, lead_times=LEAD_TIMES,
                        t_warmup=24.0, dt_decision=4.0)
    train_mid = [Episode.load(DATA_DIR / m["path"]) for m in manifest
                 if m["level"] == "moderate" and m["split"] == "train"]
    eps, record = calibrate_threshold(train_mid, proto, H_MID, TARGET_RATE)
    record["calibrated_on"] = "moderate/train"
    record["flux_window"] = proto.flux_window
    return EventConfig(threshold=eps, flux_window=proto.flux_window,
                       lead_times=LEAD_TIMES, t_warmup=proto.t_warmup,
                       dt_decision=proto.dt_decision, calibration=record)


def base_rate_table(manifest, ecfg: EventConfig) -> dict:
    table = {}
    for level in LEVELS:
        for split in ("train", "val", "test"):
            labs = []
            for m in manifest:
                if m["level"] == level and m["split"] == split:
                    ep = Episode.load(DATA_DIR / m["path"])
                    _, lab = label_episode(ep, ecfg)
                    labs.append(lab)
            rates = np.concatenate(labs).mean(axis=0)
            table[f"{level}/{split}"] = {str(h): float(r)
                                         for h, r in zip(ecfg.lead_times, rates)}
    return table


def pde_references():
    """Matched PDE trajectories per level: drift with the modulator held in
    each state, from the episode initial law -- reference dynamics for C3."""
    out = {}
    for level in LEVELS:
        cfg = sim_config(level, 0)
        rho0 = truncated_gaussian_density(cfg.init_x0, cfg.init_sd, 256)
        for state, lam in (("low", cfg.lam_low), ("high", cfg.lam_high)):
            frozen = SimConfig(**{**cfg.__dict__, "arrival": "poisson",
                                  "lam": lam, "n_brokers": 1})
            res = solve_fp(rho0, drift_from_config(frozen), cfg.a,
                           t_end=cfg.t_end, dt=2e-3, dt_sample=1.0)
            res.save(DATA_DIR / "pde_refs" / f"{level}_{state}.npz")
        out[level] = str(DATA_DIR / "pde_refs")
    return out


def main():
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("generating episodes...")
    manifest = generate_episodes()
    print("calibrating event threshold on moderate/train...")
    ecfg = calibrate(manifest)
    print(f"  eps = {ecfg.threshold:.4f} "
          f"(achieved train rate {ecfg.calibration['achieved_rate']:.3f} "
          f"at h={H_MID})")
    print("computing base-rate table...")
    table = base_rate_table(manifest, ecfg)
    for k, v in table.items():
        print(f"  {k}: " + " ".join(f"h={h}:{r:.3f}" for h, r in v.items()))
    print("solving matched PDE references...")
    pde_references()
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    ecfg.save(DATA_DIR / "event_config.json")
    (DATA_DIR / "base_rates.json").write_text(json.dumps(table, indent=1))
    (DATA_DIR / "obs_config.json").write_text(OBS.to_json())
    print(f"e02 done in {time.time() - t0:.1f}s; results in {DATA_DIR}")


if __name__ == "__main__":
    main()
