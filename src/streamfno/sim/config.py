"""Simulator configuration.

Units and scaling conventions
-----------------------------
- Time is *normalized* (continuum) time ``t`` throughout the public API.
  The O(B^2) (diffusive) / O(B) (fluid) event rates are internal to the
  jump construction and never surface in configuration values.
- State: integer backlog ``Q_i in {0, ..., B}`` per partition; the normalized
  lag is ``X_i = Q_i / B in [0, 1]``.
- ``lam`` and ``mu0`` are netput components in normalized-lag units per unit
  time, i.e. contributions to dX/dt.  The net drift of partition i is
  ``b_i = lam_i(t) - mu_i(m_c)`` where ``m_c`` is the mean lag of partition
  i's broker class.
- ``a`` is the netput variance rate (units X^2 per unit time).  It is used in
  diffusive mode only; in fluid mode the jump variance is O(1/B) by
  construction and vanishes in the limit.

Jump construction
-----------------
Each partition is a birth-death chain on {0,...,B} with up-rate ``u`` and
down-rate ``d`` (events per unit normalized time, jump size 1 in Q, i.e. 1/B
in X):

- diffusive mode:  u = B^2 a/2 + B b/2,  d = B^2 a/2 - B b/2
  so that mean(dX)/dt = b and var(dX)/dt = a + O(1/B).  Nonnegativity of the
  rates requires ``B * a >= |b|``; this is validated against the worst-case
  drift at construction time.
- fluid mode:      u = B lam,            d = B mu
  so that mean(dX)/dt = b and var(dX)/dt = (lam + mu)/B -> 0.

In diffusive mode the up/down jumps are the diffusively rescaled netput
increments, not literal message arrivals; up-jumps blocked at Q = B are still
the discrete counterpart of the continuum regulator at x = 1 and are counted
as rejected work (see engine.py).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class SimConfig:
    """Full configuration of one simulation run.

    Attributes
    ----------
    n_partitions:
        N, number of partitions.
    buffer_depth:
        B, per-partition buffer capacity in messages; Q_i <= B.
    mode:
        "diffusive" (event rates O(B^2), variance rate ``a``) or
        "fluid" (event rates O(B), vanishing variance).
    method:
        "tau_leap" (vectorized, for large N*B) or "gillespie"
        (exact, correctness reference for small N*B).
    t_end:
        End of simulated normalized time.
    dt_sample:
        Sampling interval for histograms / flux / summary outputs.
    n_bins:
        Number of fixed histogram bins on [0, 1].
    seed:
        Seed for the run's random generator; every run is fully seeded.
    arrival:
        "poisson" (constant rate ``lam``) or "mmpp" (2-state Markov-modulated
        Poisson: rate lam_low / lam_high, per-partition modulator with
        switching rates r_low_high, r_high_low).
    lam, lam_low, lam_high, r_low_high, r_high_low:
        Arrival-side netput parameters, normalized-lag units per unit time.
    mu0:
        Base service-side netput rate (same units).
    n_brokers:
        Number of broker classes; partition i belongs to class i % n_brokers.
    degradation_drop, degradation_theta, degradation_width:
        Service degradation: mu(m_c) = mu0 * (1 - drop * sigmoid((m_c -
        theta)/width)).  Bounded, steep beyond theta; see docs/decisions.md.
    a:
        Netput variance rate for diffusive mode (X^2 per unit time).
    init_x0, init_sd:
        Initial lag distribution: iid truncated Gaussian on [0,1] with the
        given center and standard deviation (init_sd = 0 gives a deterministic
        point mass at init_x0); Q_i(0) = round(B * X_i(0)).
    tau_jump_cap:
        Tau-leaping: cap on the expected number of jumps per partition per
        step (controls the leap error).
    tau_dt_max:
        Tau-leaping: absolute cap on the step (keeps the mean-field coupling
        and MMPP switching well resolved).
    """

    n_partitions: int
    buffer_depth: int
    t_end: float
    seed: int
    mode: str = "diffusive"
    method: str = "tau_leap"
    dt_sample: float = 0.5
    n_bins: int = 128
    arrival: str = "poisson"
    lam: float = 0.55
    lam_low: float = 0.3
    lam_high: float = 0.9
    r_low_high: float = 0.5
    r_high_low: float = 1.0
    mu0: float = 0.7
    n_brokers: int = 1
    degradation_drop: float = 0.4
    degradation_theta: float = 0.7
    degradation_width: float = 0.08
    a: float = 0.04
    init_x0: float = 0.35
    init_sd: float = 0.10
    tau_jump_cap: float = 5.0
    tau_dt_max: float = 0.01
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("diffusive", "fluid"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.method not in ("tau_leap", "gillespie"):
            raise ValueError(f"unknown method {self.method!r}")
        if self.arrival not in ("poisson", "mmpp"):
            raise ValueError(f"unknown arrival process {self.arrival!r}")
        if self.n_partitions <= 0 or self.buffer_depth <= 0:
            raise ValueError("n_partitions and buffer_depth must be positive")
        if self.mode == "diffusive":
            bmax = self.max_abs_drift()
            if self.buffer_depth * self.a < bmax:
                raise ValueError(
                    "diffusive mode requires B * a >= max |b| for nonnegative "
                    f"jump rates: B*a = {self.buffer_depth * self.a:.4f} < "
                    f"max|b| = {bmax:.4f}"
                )

    def max_abs_drift(self) -> float:
        """Worst-case |b| = |lam - mu| over arrival states and degradation range."""
        lam_max = self.lam if self.arrival == "poisson" else max(self.lam_low, self.lam_high)
        lam_min = self.lam if self.arrival == "poisson" else min(self.lam_low, self.lam_high)
        mu_min = self.mu0 * (1.0 - self.degradation_drop)
        mu_max = self.mu0
        return max(abs(lam_max - mu_min), abs(lam_min - mu_max))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "SimConfig":
        return cls(**json.loads(s))
