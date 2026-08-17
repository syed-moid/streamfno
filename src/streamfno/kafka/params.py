"""Real-cluster run parameterization and unit mapping.

The Phase B/C model speaks normalized units: lag X in [0, 1], time in
normalized units, netput rates in X per unit time (see sim/config.py).
A real run maps these onto a Kafka deployment through two constants:

- ``tau_s``: wall seconds per normalized time unit;
- ``budget_b``: the lag budget B (messages per partition), X_i = lag_i / B.

Rates then map as msgs/s = rate_x * B / tau_s per partition.  The service
rate mu0 is enforced by a token bucket in the consumers (a known,
configurable quantity -- what makes the identifiability experiment clean);
arrival rates by a token bucket in the producer, MMPP-modulated with the
Phase C burst-parameter shapes (shared modulator, lam_low / lam_high,
switching rates per normalized time).

Mapping used in Phase D (recorded in docs/decisions.md): tau_s = 4.0,
budget_b = 170, so lam_x = 0.82 is ~34.9 msg/s per partition and mu0 = 0.7
is ~29.75 msg/s per partition (~1428 msg/s per consumer at 2 consumers
over 96 partitions).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = ["RunParams"]


@dataclass
class RunParams:
    """Configuration of one real-cluster run (producer + consumers +
    collector), in normalized units plus the wall/lag mapping."""

    # cluster
    topic: str = "streamfno-exp"
    bootstrap: str = "localhost:9092,localhost:9192,localhost:9292"
    group_id: str = "streamfno-cg"
    n_partitions: int = 96
    # unit mapping
    tau_s: float = 4.0
    budget_b: int = 170
    # workload
    payload_bytes: int = 1024
    arrival: str = "mmpp"          # "mmpp" (shared modulator) or "poisson"
    lam: float = 0.55              # poisson mode: constant rate, X per unit time
    lam_low: float = 0.40
    lam_high: float = 0.82
    r_low_high: float = 0.06       # per normalized time
    r_high_low: float = 0.30
    # service
    mu0: float = 0.70              # X per unit time, token-bucket enforced
    n_consumers: int = 2
    # run shape
    t_end: float = 300.0           # normalized units
    seed: int = 0
    # telemetry
    dt_poll_s: float = 1.0
    commit_interval_s: float = 0.1
    extra: dict = field(default_factory=dict)

    # --- unit conversions -------------------------------------------------
    def msgs_per_s(self, rate_x: float) -> float:
        """Aggregate msgs/s over all partitions for a normalized rate."""
        return rate_x * self.budget_b / self.tau_s * self.n_partitions

    @property
    def consumer_rate_msgs(self) -> float:
        """Per-consumer service cap in msgs/s (mu0 split over consumers)."""
        return self.msgs_per_s(self.mu0) / self.n_consumers

    @property
    def dt_poll_norm(self) -> float:
        return self.dt_poll_s / self.tau_s

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "RunParams":
        return cls(**json.loads(s))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "RunParams":
        return cls.from_json(Path(path).read_text())
