"""Simulation output container.

The first-class output is the empirical lag-density measure, sampled on a
fixed grid on [0, 1], together with the boundary-flux series J_B(t).
Per-partition traces are deliberately not stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SimConfig


@dataclass
class SimResult:
    """Sampled outputs of one simulation run.

    All arrays share the leading time axis of length K = len(times); sample k
    describes the state at ``times[k]`` and, for interval quantities
    (``rejected``), the interval (times[k-1], times[k]] (row 0 is zero).

    Attributes
    ----------
    times:
        Sampling times (normalized time), shape (K,).
    hist:
        Lag histogram counts over the fixed grid, shape (K, n_bins); row sums
        equal N.  Bin edges are uniform on [0, 1]; X = 1 falls in the last bin.
    bin_edges:
        Shape (n_bins + 1,).
    lattice_hist:
        Exact counts on the lag lattice {0, 1/B, ..., 1}, shape (K, B + 1).
        Used for bias-free Wasserstein comparisons (no re-binning error).
    rejected:
        Blocked up-jump counts at Q = B per broker class in each sampling
        interval, shape (K, n_brokers).  Raw event counts.
    mean_lag, var_lag:
        Mean and variance of X per broker class, shape (K, n_brokers).
    class_sizes:
        Number of partitions per broker class, shape (n_brokers,).
    config:
        The full SimConfig of the run.
    """

    times: np.ndarray
    hist: np.ndarray
    bin_edges: np.ndarray
    lattice_hist: np.ndarray
    rejected: np.ndarray
    mean_lag: np.ndarray
    var_lag: np.ndarray
    class_sizes: np.ndarray
    config: SimConfig

    def densities(self) -> np.ndarray:
        """Histogram as probability density on [0,1]: shape (K, n_bins),
        integrating to 1 over x for each sample."""
        widths = np.diff(self.bin_edges)
        return self.hist / (self.hist.sum(axis=1, keepdims=True) * widths)

    def boundary_flux(self) -> np.ndarray:
        """J_B(t) per broker class in continuum units: rejected normalized
        work per partition per unit time, shape (K, n_brokers).

        Each blocked up-jump carries 1/B of normalized lag, so the continuum
        regulator rate is count / (N_c * B * dt_sample).
        """
        cfg = self.config
        out = np.zeros_like(self.rejected, dtype=float)
        dt = cfg.dt_sample
        out[1:] = self.rejected[1:] / (self.class_sizes[None, :] * cfg.buffer_depth * dt)
        return out

    def boundary_flux_total(self) -> np.ndarray:
        """Aggregate J_B(t) over all classes (same units), shape (K,)."""
        cfg = self.config
        total = np.zeros(len(self.times))
        total[1:] = self.rejected[1:].sum(axis=1) / (
            cfg.n_partitions * cfg.buffer_depth * cfg.dt_sample
        )
        return total

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times=self.times,
            hist=self.hist,
            bin_edges=self.bin_edges,
            lattice_hist=self.lattice_hist,
            rejected=self.rejected,
            mean_lag=self.mean_lag,
            var_lag=self.var_lag,
            class_sizes=self.class_sizes,
            config_json=np.array(self.config.to_json()),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SimResult":
        with np.load(path) as f:
            return cls(
                times=f["times"],
                hist=f["hist"],
                bin_edges=f["bin_edges"],
                lattice_hist=f["lattice_hist"],
                rejected=f["rejected"],
                mean_lag=f["mean_lag"],
                var_lag=f["var_lag"],
                class_sizes=f["class_sizes"],
                config=SimConfig.from_json(str(f["config_json"])),
            )
