"""Observation layer: the telemetry sigma-algebra.

Every predictor and every bound computation in Phase C consumes exactly the
observation process defined here -- same observables H, same sampling
interval, same noise protocol.  Nothing downstream may touch the hidden
state except through this module (event labels and diagnostics are computed
from the hidden trajectory, but no predictor sees them as inputs).

Observables H(state) at sampling times k*dt_obs (broker telemetry
correspondence documented in docs/decisions.md):
- per-broker-class mean lag              (C values;   consumer-lag gauges)
- per-broker-class lag variance          (C values;   lag dispersion)
- per-broker-class boundary flux J_B     (C values;   rejected-work / throttle
  averaged over the last interval                     counters, rate form)
- coarse lag histogram fractions         (hist_bins;  lag distribution
  over a fixed grid on [0,1]                          summaries)

Noise: additive Gaussian with covariance R = noise_std^2 * I applied at
observation time, drawn from a dedicated seeded stream independent of the
simulation seed, so one hidden trajectory can be observed under many noise
draws.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..sim.config import SimConfig
from ..sim.results import SimResult

__all__ = ["ObsConfig", "Episode", "clean_observables", "observe", "obs_names"]


@dataclass
class ObsConfig:
    """Telemetry channel definition.

    dt_obs: sampling interval Delta (normalized time); must be an integer
        multiple of the simulator's dt_sample.  Default matches e00's
        output cadence (1.0).
    noise_std: scalar r; observation noise covariance is R = r^2 I.
    hist_bins: coarse-histogram bin count (default 8).
    noise_seed: seed of the dedicated noise stream.
    """

    dt_obs: float = 1.0
    noise_std: float = 0.02
    hist_bins: int = 8
    noise_seed: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "ObsConfig":
        return cls(**json.loads(s))


def obs_names(n_classes: int, hist_bins: int) -> list[str]:
    names = [f"mean_lag_c{c}" for c in range(n_classes)]
    names += [f"var_lag_c{c}" for c in range(n_classes)]
    names += [f"flux_c{c}" for c in range(n_classes)]
    names += [f"hist_b{j}" for j in range(hist_bins)]
    return names


def clean_observables(res: SimResult, obs_cfg: ObsConfig
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Noise-free H applied to a simulated trajectory.

    Returns (times, Y) with Y of shape (K, 3C + hist_bins).  Observations
    start at t = dt_obs (the flux observable needs one whole interval).
    """
    cfg = res.config
    stride = round(obs_cfg.dt_obs / cfg.dt_sample)
    if abs(stride * cfg.dt_sample - obs_cfg.dt_obs) > 1e-9 or stride < 1:
        raise ValueError("dt_obs must be a positive integer multiple of dt_sample")
    if res.hist.shape[1] % obs_cfg.hist_bins != 0:
        raise ValueError("hist_bins must divide the simulator's n_bins")

    idx = np.arange(stride, len(res.times), stride)
    times = res.times[idx]
    mean = res.mean_lag[idx]
    var = res.var_lag[idx]
    # flux averaged over the last observation interval, continuum units
    flux_rate = res.rejected.astype(float) / (
        res.class_sizes[None, :] * cfg.buffer_depth * cfg.dt_sample
    )
    csum = np.cumsum(flux_rate, axis=0)
    flux = (csum[idx] - csum[idx - stride]) / stride
    # coarse histogram fractions
    coarse = res.hist.reshape(len(res.times), obs_cfg.hist_bins, -1).sum(axis=2)
    coarse = coarse[idx] / cfg.n_partitions
    y = np.concatenate([mean, var, flux, coarse], axis=1)
    return times, y


@dataclass
class Episode:
    """One observed trajectory: the only interface predictors may use.

    times, y: the noisy observation process (K, D).
    y_clean: noise-free H (bound computations and diagnostics only --
        never a predictor input).
    flux_times, flux_hidden: hidden aggregate boundary flux at simulator
        cadence (continuum units), used to compute event labels.
    mean_lag_hidden: hidden per-class mean lag at simulator cadence
        (diagnostics / load classification only).
    """

    times: np.ndarray
    y: np.ndarray
    y_clean: np.ndarray
    flux_times: np.ndarray
    flux_hidden: np.ndarray
    mean_lag_hidden: np.ndarray
    sim_config: SimConfig
    obs_config: ObsConfig
    meta: dict = field(default_factory=dict)

    @property
    def obs_names(self) -> list[str]:
        return obs_names(self.sim_config.n_brokers, self.obs_config.hist_bins)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, times=self.times, y=self.y, y_clean=self.y_clean,
            flux_times=self.flux_times, flux_hidden=self.flux_hidden,
            mean_lag_hidden=self.mean_lag_hidden,
            sim_config_json=np.array(self.sim_config.to_json()),
            obs_config_json=np.array(self.obs_config.to_json()),
            meta_json=np.array(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Episode":
        with np.load(path) as f:
            return cls(
                times=f["times"], y=f["y"], y_clean=f["y_clean"],
                flux_times=f["flux_times"], flux_hidden=f["flux_hidden"],
                mean_lag_hidden=f["mean_lag_hidden"],
                sim_config=SimConfig.from_json(str(f["sim_config_json"])),
                obs_config=ObsConfig.from_json(str(f["obs_config_json"])),
                meta=json.loads(str(f["meta_json"])),
            )


def observe(res: SimResult, obs_cfg: ObsConfig, meta: dict | None = None
            ) -> Episode:
    """Apply the telemetry channel to a simulated trajectory."""
    times, y_clean = clean_observables(res, obs_cfg)
    noise_rng = np.random.default_rng(obs_cfg.noise_seed)
    y = y_clean + obs_cfg.noise_std * noise_rng.standard_normal(y_clean.shape)
    return Episode(
        times=times, y=y, y_clean=y_clean,
        flux_times=res.times, flux_hidden=res.boundary_flux_total(),
        mean_lag_hidden=res.mean_lag,
        sim_config=res.config, obs_config=obs_cfg,
        meta=meta or {},
    )
