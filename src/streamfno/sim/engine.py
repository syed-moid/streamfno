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

MMPP modulator topology: with ``mmpp_shared = False`` each partition carries
an independent 2-state modulator (at large N the aggregate arrival rate is
then nearly constant by averaging); with ``mmpp_shared = True`` a single
modulator drives every partition, producing coherent bursts that move the
whole system between load regimes.  Internally both are the special cases
"one group per partition" / "one group total" of a group-structured
modulator, which also lets an ensemble of independent replicas (particles)
be advanced as one vectorized system.

Tau-leaping approximations (documented in docs/decisions.md):
- Poisson jump counts per step with rates frozen at the step start
  (explicit tau-leap); the step is capped so the expected number of jumps
  per partition per step is <= tau_jump_cap and the step never exceeds
  tau_dt_max, keeping the mean-field coupling well resolved.
- Within a step, each partition applies its service and arrival counts in a
  uniformly random order (services-first or arrivals-first).  A fixed order
  has an O(u tau * d tau) boundary bias with a definite sign -- e.g.
  services-first at Q = 0 turns the path (arrival, service) -> 0 into
  "always end at 1", measurably depleting the wall site (~10% at the
  stationary boundary layer) -- and the randomized order cancels it at
  leading order at both walls.  Down-jumps in excess of the backlog are
  no-ops (reflection); up-jumps in excess of B are counted as rejected.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import expit

from .config import SimConfig
from .results import SimResult

__all__ = ["TauLeapSim", "simulate"]


def _broker_classes(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    classes = np.arange(cfg.n_partitions, dtype=np.int64) % cfg.n_brokers
    sizes = np.bincount(classes, minlength=cfg.n_brokers)
    return classes, sizes


def _service_rates(cfg: SimConfig, classes: np.ndarray, sizes: np.ndarray,
                   n_classes: int, x: np.ndarray) -> np.ndarray:
    """Per-partition service rate mu_i (normalized-lag units per unit time).

    A broker class's capacity degrades as the mean lag of its own partitions
    rises: mu(m_c) = mu0 * (1 - drop * sigmoid((m_c - theta)/width)).
    """
    m_c = np.bincount(classes, weights=x, minlength=n_classes) / sizes
    g = 1.0 - cfg.degradation_drop * expit(
        (m_c - cfg.degradation_theta) / cfg.degradation_width
    )
    return (cfg.mu0 * g)[classes]


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


def _initial_backlog(cfg: SimConfig, rng: np.random.Generator,
                     n: int | None = None) -> np.ndarray:
    """iid truncated-Gaussian initial lags, quantized to the lattice."""
    n = cfg.n_partitions if n is None else n
    if cfg.init_sd == 0.0:
        x = np.full(n, cfg.init_x0)
    else:
        x = rng.normal(cfg.init_x0, cfg.init_sd, size=n)
        bad = (x < 0.0) | (x > 1.0)
        while bad.any():
            x[bad] = rng.normal(cfg.init_x0, cfg.init_sd, size=int(bad.sum()))
            bad = (x < 0.0) | (x > 1.0)
    return np.rint(x * cfg.buffer_depth).astype(np.int64)


def _mmpp_groups(cfg: SimConfig, n: int) -> np.ndarray:
    if cfg.mmpp_shared:
        return np.zeros(n, dtype=np.int64)
    return np.arange(n, dtype=np.int64)


def _init_mmpp_state(cfg: SimConfig, rng: np.random.Generator,
                     n_groups: int) -> np.ndarray:
    p_high = cfg.r_low_high / (cfg.r_low_high + cfg.r_high_low)
    return (rng.random(n_groups) < p_high).astype(np.int64)


class TauLeapSim:
    """Incrementally steppable tau-leap system.

    Used by simulate() for whole runs and by filtering/prediction code that
    must interleave observation updates with propagation.  The constructor
    arguments beyond ``cfg`` exist so an ensemble of independent replicas
    can be packed into one vectorized system:

    classes / n_classes:
        Per-partition class ids for the mean-field coupling and rejection
        accounting (default: broker classes from cfg).  Replicas are kept
        independent by giving each its own block of class ids.
    mmpp_groups:
        Per-partition modulator-group ids (default from cfg.mmpp_shared).
        Each group carries one 2-state modulator.
    q, mmpp_state, t:
        Optional explicit initial state (defaults: seeded initial law).
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator | None = None,
                 classes: np.ndarray | None = None, n_classes: int | None = None,
                 mmpp_groups: np.ndarray | None = None,
                 q: np.ndarray | None = None,
                 mmpp_state: np.ndarray | None = None, t: float = 0.0):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed) if rng is None else rng
        if classes is None:
            classes, sizes = _broker_classes(cfg)
            n_classes = cfg.n_brokers
        else:
            classes = np.asarray(classes, dtype=np.int64)
            n_classes = int(n_classes if n_classes is not None else classes.max() + 1)
            sizes = np.bincount(classes, minlength=n_classes)
        self.classes = classes
        self.n_classes = n_classes
        self.sizes = sizes
        self.n = classes.size
        self.q = _initial_backlog(cfg, self.rng, self.n) if q is None else q
        if cfg.arrival == "mmpp":
            self.groups = (_mmpp_groups(cfg, self.n) if mmpp_groups is None
                           else np.asarray(mmpp_groups, dtype=np.int64))
            self.n_groups = int(self.groups.max()) + 1
            self.mmpp_state = (_init_mmpp_state(cfg, self.rng, self.n_groups)
                               if mmpp_state is None else mmpp_state)
        else:
            self.groups = None
            self.n_groups = 0
            self.mmpp_state = None
        self.t = t

    def arrival_rates(self) -> np.ndarray:
        cfg = self.cfg
        if cfg.arrival == "poisson":
            return np.full(self.n, cfg.lam)
        lam_by_group = np.where(self.mmpp_state == 1, cfg.lam_high, cfg.lam_low)
        return lam_by_group[self.groups]

    def advance(self, duration: float) -> np.ndarray:
        """Advance by ``duration`` time units; return per-class blocked
        up-jump counts (rejected work) accumulated over the interval."""
        cfg = self.cfg
        B = cfg.buffer_depth
        t_stop = self.t + duration
        rejected = np.zeros(self.n_classes, dtype=np.int64)
        while self.t < t_stop - 1e-12:
            lam = self.arrival_rates()
            mu = _service_rates(cfg, self.classes, self.sizes, self.n_classes,
                                self.q / B)
            u, d = _jump_rates(cfg, lam, mu)
            rate_max = float(np.max(u + d))
            tau = min(cfg.tau_dt_max, cfg.tau_jump_cap / rate_max,
                      t_stop - self.t)

            arrivals = self.rng.poisson(u * tau)
            services = self.rng.poisson(d * tau)
            services_first = self.rng.random(self.n) < 0.5
            # services-first branch
            q_sf = np.maximum(self.q - services, 0) + arrivals
            over_sf = np.maximum(q_sf - B, 0)
            # arrivals-first branch
            q_plus = self.q + arrivals
            over_af = np.maximum(q_plus - B, 0)
            q_af = np.maximum(np.minimum(q_plus, B) - services, 0)
            over = np.where(services_first, over_sf, over_af)
            self.q = np.where(services_first, np.minimum(q_sf, B), q_af)
            rejected += np.bincount(self.classes, weights=over,
                                    minlength=self.n_classes).astype(np.int64)
            if self.mmpp_state is not None:
                p_up = -np.expm1(-cfg.r_low_high * tau)
                p_dn = -np.expm1(-cfg.r_high_low * tau)
                r = self.rng.random(self.n_groups)
                flip = np.where(self.mmpp_state == 0, r < p_up, r < p_dn)
                self.mmpp_state = np.where(flip, 1 - self.mmpp_state,
                                           self.mmpp_state)
            self.t += tau
        return rejected

    def clone(self) -> "TauLeapSim":
        """Independent copy sharing nothing mutable; the clone gets a child
        RNG spawned from this system's stream."""
        new = object.__new__(TauLeapSim)
        new.cfg = self.cfg
        new.rng = self.rng.spawn(1)[0]
        new.classes = self.classes
        new.n_classes = self.n_classes
        new.sizes = self.sizes
        new.n = self.n
        new.q = self.q.copy()
        new.groups = self.groups
        new.n_groups = self.n_groups
        new.mmpp_state = None if self.mmpp_state is None else self.mmpp_state.copy()
        new.t = self.t
        return new


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
    sim = TauLeapSim(cfg)
    rec = _Recorder(cfg, sim.classes, sim.sizes)
    rec.record(sim.q, np.zeros(cfg.n_brokers, dtype=np.int64))
    while rec.k < rec.n_samples:
        rejected = sim.advance(rec.next_time - sim.t)
        rec.record(sim.q, rejected)
    return rec.result()


def _simulate_gillespie(cfg: SimConfig) -> SimResult:
    """Exact event-by-event simulation.  Correctness reference; O(N) work per
    event, intended for small N*B only."""
    rng = np.random.default_rng(cfg.seed)
    classes, sizes = _broker_classes(cfg)
    q = _initial_backlog(cfg, rng)
    n = cfg.n_partitions
    mmpp_state = None
    groups = None
    n_groups = 0
    if cfg.arrival == "mmpp":
        groups = _mmpp_groups(cfg, n)
        n_groups = int(groups.max()) + 1
        mmpp_state = _init_mmpp_state(cfg, rng, n_groups)

    rec = _Recorder(cfg, classes, sizes)
    rej_interval = np.zeros(cfg.n_brokers, dtype=np.int64)
    rec.record(q, rej_interval)

    B = cfg.buffer_depth
    t = 0.0
    while rec.k < rec.n_samples:
        if cfg.arrival == "poisson":
            lam = np.full(n, cfg.lam)
        else:
            lam = np.where(mmpp_state == 1, cfg.lam_high, cfg.lam_low)[groups]
        mu = _service_rates(cfg, classes, sizes, cfg.n_brokers, q / B)
        u, d = _jump_rates(cfg, lam, mu)
        if mmpp_state is not None:
            sw = np.where(mmpp_state == 0, cfg.r_low_high, cfg.r_high_low)
        else:
            sw = np.zeros(0)
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
        if idx < 2 * n:
            i = idx % n
            if idx < n:  # up-jump
                if q[i] == B:
                    rej_interval[classes[i]] += 1
                else:
                    q[i] += 1
            else:  # down-jump
                if q[i] > 0:
                    q[i] -= 1
        else:  # MMPP modulator switch (per group)
            g = idx - 2 * n
            mmpp_state[g] = 1 - mmpp_state[g]
    return rec.result()
