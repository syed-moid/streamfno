"""Simulation engines: exact Gillespie and vectorized tau-leaping.

Both engines simulate the same birth-death jump construction (see
config.py for the rate definitions and scaling conventions) and share the
same initial-condition sampler, so runs with equal seeds start from the
same state.

Boundary semantics (both engines):
- Q = 0: down-jumps fire but leave the state unchanged (reflection; the
  chain is the reflected random walk whose scaling limit is the reflected
  diffusion).
- Q = B: up-jumps fire but leave the state unchanged and are counted as
  rejected work.  The per-class count of blocked up-jumps per sampling
  interval is the boundary-flux output J_B(t).

Tau-leaping approximations (documented in docs/decisions.md):
- Poisson jump counts per step with rates frozen at the step start
  (explicit tau-leap); the step is capped so the expected number of jumps
  per partition per step is <= tau_jump_cap and the step never exceeds
  tau_dt_max, keeping the mean-field coupling well resolved.
- Within a step, services are applied before arrivals; boundary counts are
  computed after both (excess over B is counted as rejected).  The
  resulting boundary bias is O(tau).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expit

from .config import SimConfig
from .results import SimResult

__all__ = ["simulate"]


def _broker_classes(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    classes = np.arange(cfg.n_partitions, dtype=np.int64) % cfg.n_brokers
    sizes = np.bincount(classes, minlength=cfg.n_brokers)
    return classes, sizes


def _service_rates(cfg: SimConfig, classes: np.ndarray, sizes: np.ndarray,
                   x: np.ndarray) -> np.ndarray:
    """Per-partition service rate mu_i (normalized-lag units per unit time).

    A broker class's capacity degrades as the mean lag of its own partitions
    rises: mu(m_c) = mu0 * (1 - drop * sigmoid((m_c - theta)/width)).
    """
    m_c = np.bincount(classes, weights=x, minlength=cfg.n_brokers) / sizes
    g = 1.0 - cfg.degradation_drop * expit(
        (m_c - cfg.degradation_theta) / cfg.degradation_width
    )
    return (cfg.mu0 * g)[classes]


def _arrival_rates(cfg: SimConfig, mmpp_state: np.ndarray | None) -> np.ndarray:
    """Per-partition arrival-side netput rate lam_i (same units as mu)."""
    if cfg.arrival == "poisson":
        return np.full(cfg.n_partitions, cfg.lam)
    return np.where(mmpp_state == 1, cfg.lam_high, cfg.lam_low)


def _jump_rates(cfg: SimConfig, lam: np.ndarray, mu: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Up/down event rates (events per unit normalized time, jump size 1 in Q)."""
    B = cfg.buffer_depth
    if cfg.mode == "diffusive":
        b = lam - mu
        base = 0.5 * B * B * cfg.a
        u = base + 0.5 * B * b
        d = base - 0.5 * B * b
        # nonnegativity is guaranteed by the config validation (B*a >= |b|);
        # clip float round-off only
        return np.maximum(u, 0.0), np.maximum(d, 0.0)
    return B * lam, B * mu


def _initial_backlog(cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """iid truncated-Gaussian initial lags, quantized to the lattice."""
    if cfg.init_sd == 0.0:
        x = np.full(cfg.n_partitions, cfg.init_x0)
    else:
        x = rng.normal(cfg.init_x0, cfg.init_sd, size=cfg.n_partitions)
        bad = (x < 0.0) | (x > 1.0)
        while bad.any():
            x[bad] = rng.normal(cfg.init_x0, cfg.init_sd, size=int(bad.sum()))
            bad = (x < 0.0) | (x > 1.0)
    return np.rint(x * cfg.buffer_depth).astype(np.int64)


class _Recorder:
    """Accumulates the sampled outputs on the fixed grid."""

    def __init__(self, cfg: SimConfig, classes: np.ndarray, sizes: np.ndarray):
        self.cfg = cfg
        self.classes = classes
        self.sizes = sizes
        self.n_samples = int(math.floor(cfg.t_end / cfg.dt_sample + 1e-9)) + 1
        self.times = np.arange(self.n_samples) * cfg.dt_sample
        self.hist = np.zeros((self.n_samples, cfg.n_bins), dtype=np.int64)
        self.bin_edges = np.linspace(0.0, 1.0, cfg.n_bins + 1)
        self.lattice_hist = np.zeros((self.n_samples, cfg.buffer_depth + 1), dtype=np.int64)
        self.rejected = np.zeros((self.n_samples, cfg.n_brokers), dtype=np.int64)
        self.mean_lag = np.zeros((self.n_samples, cfg.n_brokers))
        self.var_lag = np.zeros((self.n_samples, cfg.n_brokers))
        self.k = 0

    @property
    def next_time(self) -> float:
        return float(self.times[self.k]) if self.k < self.n_samples else math.inf

    def record(self, q: np.ndarray, rejected_interval: np.ndarray) -> None:
        cfg = self.cfg
        x = q / cfg.buffer_depth
        bin_idx = np.minimum(q * cfg.n_bins // cfg.buffer_depth, cfg.n_bins - 1)
        self.hist[self.k] = np.bincount(bin_idx, minlength=cfg.n_bins)
        self.lattice_hist[self.k] = np.bincount(q, minlength=cfg.buffer_depth + 1)
        self.rejected[self.k] = rejected_interval
        m = np.bincount(self.classes, weights=x, minlength=cfg.n_brokers) / self.sizes
        m2 = np.bincount(self.classes, weights=x * x, minlength=cfg.n_brokers) / self.sizes
        self.mean_lag[self.k] = m
        self.var_lag[self.k] = np.maximum(m2 - m * m, 0.0)
        self.k += 1

    def result(self) -> SimResult:
        return SimResult(
            times=self.times, hist=self.hist, bin_edges=self.bin_edges,
            lattice_hist=self.lattice_hist, rejected=self.rejected,
            mean_lag=self.mean_lag, var_lag=self.var_lag,
            class_sizes=self.sizes, config=self.cfg,
        )


def simulate(cfg: SimConfig) -> SimResult:
    """Run one fully seeded simulation and return the sampled outputs."""
    if cfg.method == "tau_leap":
        return _simulate_tau_leap(cfg)
    return _simulate_gillespie(cfg)


def _simulate_tau_leap(cfg: SimConfig) -> SimResult:
    rng = np.random.default_rng(cfg.seed)
    classes, sizes = _broker_classes(cfg)
    q = _initial_backlog(cfg, rng)
    mmpp_state = None
    if cfg.arrival == "mmpp":
        p_high = cfg.r_low_high / (cfg.r_low_high + cfg.r_high_low)
        mmpp_state = (rng.random(cfg.n_partitions) < p_high).astype(np.int64)

    rec = _Recorder(cfg, classes, sizes)
    rej_interval = np.zeros(cfg.n_brokers, dtype=np.int64)
    rec.record(q, rej_interval)

    B = cfg.buffer_depth
    t = 0.0
    while rec.k < rec.n_samples:
        lam = _arrival_rates(cfg, mmpp_state)
        mu = _service_rates(cfg, classes, sizes, q / B)
        u, d = _jump_rates(cfg, lam, mu)
        rate_max = float(np.max(u + d))
        tau = min(cfg.tau_dt_max, cfg.tau_jump_cap / rate_max, rec.next_time - t)

        arrivals = rng.poisson(u * tau)
        services = rng.poisson(d * tau)
        q1 = np.maximum(q - services, 0)
        q2 = q1 + arrivals
        over = np.maximum(q2 - B, 0)
        q = np.minimum(q2, B)
        rej_interval += np.bincount(classes, weights=over, minlength=cfg.n_brokers).astype(
            np.int64
        )
        if mmpp_state is not None:
            p_up = -np.expm1(-cfg.r_low_high * tau)
            p_dn = -np.expm1(-cfg.r_high_low * tau)
            r = rng.random(cfg.n_partitions)
            flip = np.where(mmpp_state == 0, r < p_up, r < p_dn)
            mmpp_state = np.where(flip, 1 - mmpp_state, mmpp_state)

        t += tau
        if t >= rec.next_time - 1e-12:
            rec.record(q, rej_interval)
            rej_interval[:] = 0
    return rec.result()


def _simulate_gillespie(cfg: SimConfig) -> SimResult:
    """Exact event-by-event simulation.  Correctness reference; O(N) work per
    event, intended for small N*B only."""
    rng = np.random.default_rng(cfg.seed)
    classes, sizes = _broker_classes(cfg)
    q = _initial_backlog(cfg, rng)
    n = cfg.n_partitions
    mmpp_state = None
    if cfg.arrival == "mmpp":
        p_high = cfg.r_low_high / (cfg.r_low_high + cfg.r_high_low)
        mmpp_state = (rng.random(n) < p_high).astype(np.int64)

    rec = _Recorder(cfg, classes, sizes)
    rej_interval = np.zeros(cfg.n_brokers, dtype=np.int64)
    rec.record(q, rej_interval)

    B = cfg.buffer_depth
    t = 0.0
    while rec.k < rec.n_samples:
        lam = _arrival_rates(cfg, mmpp_state)
        mu = _service_rates(cfg, classes, sizes, q / B)
        u, d = _jump_rates(cfg, lam, mu)
        if mmpp_state is not None:
            sw = np.where(mmpp_state == 0, cfg.r_low_high, cfg.r_high_low)
        else:
            sw = np.zeros(n)
        rates = np.concatenate([u, d, sw])
        total = float(rates.sum())
        if total <= 0.0:
            # no events possible; state frozen until the end
            while rec.k < rec.n_samples:
                rec.record(q, rej_interval)
                rej_interval[:] = 0
            break
        t_next = t + rng.exponential(1.0 / total)
        # flush sample times crossed by this waiting interval
        while rec.k < rec.n_samples and rec.next_time <= t_next + 1e-12:
            rec.record(q, rej_interval)
            rej_interval[:] = 0
        if rec.k >= rec.n_samples:
            break
        t = t_next
        idx = int(np.searchsorted(np.cumsum(rates), rng.random() * total))
        i = idx % n
        kind = idx // n
        if kind == 0:  # up-jump
            if q[i] == B:
                rej_interval[classes[i]] += 1
            else:
                q[i] += 1
        elif kind == 1:  # down-jump
            if q[i] > 0:
                q[i] -= 1
        else:  # MMPP modulator switch
            mmpp_state[i] = 1 - mmpp_state[i]
    return rec.result()
