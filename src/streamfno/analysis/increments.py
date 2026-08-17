"""Conditional-increment regressions: drift and variance rate from
occupancy telemetry.

Given per-partition normalized occupancy trajectories X_i(t_k) sampled at
cadence dt (normalized time), the estimators are the classical binned
Kramers-Moyal regressions

    b_hat(x) = E[ Delta X | X ~ x ] / Delta,
    a_hat(x) = Var[ Delta X | X ~ x ] / Delta,

with Delta = stride * dt, increments non-overlapping, binned by the
starting occupancy, pooled across partitions.  Interior bins only are
trustworthy: reflection at x = 0 and clipping at x = 1 bias increments
near the walls, so summary statistics use an interior range.

Measurement noise (committed-offset quantization, sampling jitter) adds
~2 sigma_meas^2 / Delta to a_hat; estimating at two strides and checking
that a_hat is stride-stable is the recorded diagnostic for this.

For MMPP-modulated runs the modulator state is *observable from
telemetry*: per-partition arrival counts are log-end-offset increments,
and the aggregate arrival rate is bimodal.  ``classify_two_state``
thresholds the smoothed aggregate rate by 1-d 2-means; increments whose
window lies entirely in one state can then be regressed per state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["IncrementFit", "binned_increments", "classify_two_state",
           "interior_mean_drift"]


@dataclass
class IncrementFit:
    """Binned drift / variance-rate estimates with sampling SEs."""

    x_centers: np.ndarray
    b_hat: np.ndarray
    b_se: np.ndarray
    a_hat: np.ndarray
    a_se: np.ndarray
    counts: np.ndarray
    delta: float          # increment span in normalized time
    stride: int


def binned_increments(x: np.ndarray, dt: float, stride: int = 4,
                      n_bins: int = 24, x_range: tuple = (0.0, 1.0),
                      start_mask: np.ndarray | None = None) -> IncrementFit:
    """Kramers-Moyal regression from an occupancy matrix.

    x: (K, P) occupancy samples at cadence dt.  start_mask: optional (K,)
    boolean; an increment starting at k is used only if
    start_mask[k : k + stride + 1] is all True (state-pure windows).
    """
    x = np.asarray(x, dtype=float)
    k_tot = x.shape[0]
    # greedy non-overlapping windows: advance one sample at a time until a
    # (state-pure, if masked) window fits, then jump past it -- windows are
    # not forced onto a global stride grid, which would discard most short
    # same-state segments
    starts_list = []
    s = 0
    while s < k_tot - stride:
        if start_mask is None or bool(start_mask[s:s + stride + 1].all()):
            starts_list.append(s)
            s += stride
        else:
            s += 1
    starts = np.asarray(starts_list, dtype=int)
    if starts.size == 0:
        raise ValueError("no usable increment windows")
    x0 = x[starts].ravel()
    dx = (x[starts + stride] - x[starts]).ravel()
    delta = stride * dt

    edges = np.linspace(x_range[0], x_range[1], n_bins + 1)
    which = np.digitize(x0, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    b = np.full(n_bins, np.nan)
    b_se = np.full(n_bins, np.nan)
    a = np.full(n_bins, np.nan)
    a_se = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for j in range(n_bins):
        d = dx[which == j]
        counts[j] = d.size
        if d.size < 8:
            continue
        m = d.mean()
        v = d.var(ddof=1)
        b[j] = m / delta
        b_se[j] = np.sqrt(v / d.size) / delta
        a[j] = v / delta
        # SE of a sample variance ~ v * sqrt(2/(n-1)) for near-Gaussian
        # increments; adequate for reporting at these counts
        a_se[j] = v * np.sqrt(2.0 / (d.size - 1)) / delta
    return IncrementFit(x_centers=centers, b_hat=b, b_se=b_se, a_hat=a,
                        a_se=a_se, counts=counts, delta=delta, stride=stride)


def interior_mean_drift(fit: IncrementFit, x_lo: float, x_hi: float
                        ) -> tuple[float, float]:
    """Count-weighted mean drift (value, SE) over interior bins."""
    sel = ((fit.x_centers >= x_lo) & (fit.x_centers <= x_hi)
           & np.isfinite(fit.b_hat) & (fit.counts > 0))
    if not sel.any():
        raise ValueError("no populated interior bins")
    w = fit.counts[sel].astype(float)
    w /= w.sum()
    val = float((w * fit.b_hat[sel]).sum())
    se = float(np.sqrt((w ** 2 * fit.b_se[sel] ** 2).sum()))
    return val, se


def classify_two_state(rate: np.ndarray, smooth: int = 8,
                       n_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Two-state classification of an aggregate arrival-rate series.

    Trailing moving average over ``smooth`` samples, then 1-d 2-means
    initialized at the 10th / 90th percentiles.  Returns (is_high (K,)
    bool, centers (2,) low/high).  The smoothing is causal, so the label
    at k uses telemetry up to k only.
    """
    r = np.asarray(rate, dtype=float)
    c = np.cumsum(np.concatenate([[0.0], r]))
    idx = np.arange(1, r.size + 1)
    lo = np.maximum(idx - smooth, 0)
    rs = (c[idx] - c[lo]) / (idx - lo)

    lo_c, hi_c = np.percentile(rs, 10), np.percentile(rs, 90)
    for _ in range(n_iter):
        mid = 0.5 * (lo_c + hi_c)
        high = rs > mid
        if not high.any() or high.all():
            break
        new_lo, new_hi = rs[~high].mean(), rs[high].mean()
        if abs(new_lo - lo_c) < 1e-12 and abs(new_hi - hi_c) < 1e-12:
            lo_c, hi_c = new_lo, new_hi
            break
        lo_c, hi_c = new_lo, new_hi
    return rs > 0.5 * (lo_c + hi_c), np.array([lo_c, hi_c])
