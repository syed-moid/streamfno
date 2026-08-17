"""Adapter: raw lag telemetry -> the Phase C episode format.

Real measurements enter the existing analysis stack here, unchanged
downstream.  Conventions:

- Normalized occupancy X_i = min(lag_i / B, 1) with B the run's lag
  budget; time is normalized by tau_s from the run's t0 (producer start).
- The boundary-flux counterpart of the simulator's rejected-work counter
  is the *overshoot-increment* rate: real Kafka does not reject work at
  the budget, so work arriving at an already-over-budget partition plays
  the regulator's role,

      J(t_k) = sum_i [ (lag_i(t_k)-B)^+ - (lag_i(t_{k-1})-B)^+ ]^+
               / (N * B * dt_norm),

  in the simulator's continuum units (normalized work per partition per
  unit time).  Only positive increments count, matching the one-sided
  regulator (draining back below budget is service, not rejection).
- The Episode's ``y`` and ``y_clean`` are identical: the measurement *is*
  the telemetry channel; committed-offset quantization and sampling
  jitter are its real noise.  The attached SimConfig is a descriptor of
  the run (fluid mode, so no diffusive-parameter validation applies), not
  a generator; run provenance lives in ``meta``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..obs.observe import Episode, ObsConfig
from ..sim.config import SimConfig
from .params import RunParams

__all__ = ["LagTelemetry", "load_telemetry", "overshoot_flux",
           "telemetry_episode"]


@dataclass
class LagTelemetry:
    """Raw collector output: offsets and lag per partition at poll cadence."""

    t_wall: np.ndarray          # (K,)
    leo: np.ndarray             # (K, P)
    committed: np.ndarray       # (K, P), -1 before first commit
    sample_latency_s: np.ndarray
    cpu_s: np.ndarray
    meta: dict

    @property
    def lag(self) -> np.ndarray:
        """Consumer lag in messages; before the first commit the whole log
        end is outstanding."""
        return np.maximum(self.leo - np.maximum(self.committed, 0), 0)

    def x(self, budget: int) -> np.ndarray:
        """Normalized occupancy, clipped to [0, 1]."""
        return np.minimum(self.lag / float(budget), 1.0)


def load_telemetry(path: str | Path) -> LagTelemetry:
    with np.load(path) as f:
        return LagTelemetry(
            t_wall=f["t_wall"], leo=f["leo"], committed=f["committed"],
            sample_latency_s=f["sample_latency_s"], cpu_s=f["cpu_s"],
            meta=json.loads(str(f["meta_json"])),
        )


def overshoot_flux(lag: np.ndarray, budget: int, dt_norm: float) -> np.ndarray:
    """Boundary-flux series (K,) in continuum units from a (K, P) lag
    matrix; row 0 is zero (interval quantity), matching SimResult."""
    over = np.maximum(lag - budget, 0).astype(float)
    inc = np.maximum(np.diff(over, axis=0), 0.0).sum(axis=1)
    n_parts = lag.shape[1]
    out = np.zeros(lag.shape[0])
    out[1:] = inc / (n_parts * budget * dt_norm)
    return out


def descriptor_config(params: RunParams, t_end_norm: float) -> SimConfig:
    """A SimConfig describing (not generating) the real run."""
    return SimConfig(
        n_partitions=params.n_partitions, buffer_depth=params.budget_b,
        t_end=float(t_end_norm), seed=params.seed, mode="fluid",
        method="tau_leap", dt_sample=params.dt_poll_norm,
        n_bins=64, arrival=params.arrival, mmpp_shared=True,
        lam=params.lam, lam_low=params.lam_low, lam_high=params.lam_high,
        r_low_high=params.r_low_high, r_high_low=params.r_high_low,
        mu0=params.mu0, n_brokers=1, degradation_drop=0.0,
        init_x0=0.0, init_sd=0.0, a=0.0,
    )


def telemetry_episode(tel: LagTelemetry, params: RunParams, t0_wall: float,
                      dt_obs: float = 1.0, hist_bins: int = 8,
                      meta: dict | None = None) -> Episode:
    """Build a Phase C Episode from raw telemetry.

    t0_wall: normalized-time origin (the producer's start); samples before
    it are dropped.  dt_obs is in normalized units and must be a multiple
    of the poll cadence.
    """
    dt_raw = params.dt_poll_norm
    keep = tel.t_wall >= t0_wall - 1e-6
    t_norm = (tel.t_wall[keep] - t0_wall) / params.tau_s
    x = tel.x(params.budget_b)[keep]
    lag = tel.lag[keep]

    flux = overshoot_flux(lag, params.budget_b, dt_raw)
    stride = round(dt_obs / dt_raw)
    if abs(stride * dt_raw - dt_obs) > 1e-9 or stride < 1:
        raise ValueError("dt_obs must be a positive multiple of the poll cadence")

    idx = np.arange(stride, len(t_norm), stride)
    times = t_norm[idx]
    mean = x.mean(axis=1)[:, None]
    var = x.var(axis=1)[:, None]
    csum = np.cumsum(flux)
    flux_obs = ((csum[idx] - csum[idx - stride]) / stride)[:, None]
    edges = np.linspace(0.0, 1.0, hist_bins + 1)
    edges[-1] = 1.0 + 1e-9
    coarse = np.stack([np.histogram(row, bins=edges)[0] for row in x[idx]])
    coarse = coarse / x.shape[1]
    y = np.concatenate([mean[idx], var[idx], flux_obs, coarse], axis=1)

    cfg = descriptor_config(params, t_end_norm=float(t_norm[-1]))
    obs_cfg = ObsConfig(dt_obs=dt_obs, noise_std=0.0, hist_bins=hist_bins,
                        noise_seed=0)
    full_meta = {"source": "kafka", "tau_s": params.tau_s,
                 "budget_b": params.budget_b, "t0_wall": t0_wall,
                 **(meta or {})}
    return Episode(
        times=times, y=y, y_clean=y.copy(),
        flux_times=t_norm, flux_hidden=flux,
        mean_lag_hidden=mean, sim_config=cfg, obs_config=obs_cfg,
        meta=full_meta,
    )
