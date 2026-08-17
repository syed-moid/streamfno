"""Honest rate control: token buckets and the shared MMPP modulator.

The producer and the capped consumers both pace with a token bucket
(continuous refill at the target rate, bounded burst capacity), never
sleep-per-message: a stalled loop iteration catches up only up to the
bucket capacity, so short scheduler hiccups do not distort the average
rate and long ones are visibly dropped rather than silently compressed.

The MMPP modulator is precomputed as a deterministic, seeded timeline of
switch times so a run is reproducible and the ground-truth state sequence
can be stored alongside the telemetry (as a diagnostic; estimation in e07
uses telemetry only).
"""

from __future__ import annotations

import numpy as np

__all__ = ["TokenBucket", "mmpp_timeline", "rate_at"]


class TokenBucket:
    """Token bucket with refill ``rate`` per second and a burst capacity
    of ``burst_s`` seconds' worth of tokens."""

    def __init__(self, rate: float, burst_s: float = 0.25):
        self.rate = float(rate)
        self.burst_s = float(burst_s)
        self.tokens = 0.0
        self._last: float | None = None

    def refill(self, now: float) -> None:
        if self._last is not None:
            self.tokens = min(self.tokens + self.rate * (now - self._last),
                              self.rate * self.burst_s)
        self._last = now

    def take(self, want: int) -> int:
        """Take up to ``want`` whole tokens; returns how many were taken."""
        n = min(int(self.tokens), int(want))
        self.tokens -= n
        return n

    def give_back(self, n: int) -> None:
        """Return tokens for work that did not happen (e.g. an empty poll)."""
        self.tokens += n


def mmpp_timeline(t_end: float, r_low_high: float, r_high_low: float,
                  seed: int) -> np.ndarray:
    """Switch times of the 2-state modulator on [0, t_end] (normalized
    time), starting in the low state at t = 0.

    Returns an array of switch times t_1 < t_2 < ...; the state on
    [t_k, t_{k+1}) is high for even k (0-based, t_0 = 0 low). Deterministic
    given the seed.
    """
    rng = np.random.default_rng(seed)
    times = []
    t, high = 0.0, False
    while t < t_end:
        r = r_high_low if high else r_low_high
        t += rng.exponential(1.0 / r)
        if t < t_end:
            times.append(t)
        high = not high
    return np.asarray(times)


def rate_at(t: float, switches: np.ndarray, lam_low: float,
            lam_high: float) -> float:
    """Modulated rate at normalized time t for a timeline from
    ``mmpp_timeline`` (low state before the first switch)."""
    k = int(np.searchsorted(switches, t, side="right"))
    return lam_high if k % 2 == 1 else lam_low
