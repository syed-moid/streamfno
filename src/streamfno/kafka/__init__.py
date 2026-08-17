"""Real-Kafka contact layer: load harness, lag telemetry, episode adapter.

Modules that talk to a live cluster (producer, consumer, collector,
harness) import confluent_kafka lazily/locally; params, pacing and the
adapter are importable without a Kafka client so the analysis and tests
run anywhere.
"""

from .adapter import LagTelemetry, load_telemetry, overshoot_flux, telemetry_episode
from .pacing import TokenBucket, mmpp_timeline, rate_at
from .params import RunParams

__all__ = [
    "LagTelemetry",
    "RunParams",
    "TokenBucket",
    "load_telemetry",
    "mmpp_timeline",
    "overshoot_flux",
    "rate_at",
    "telemetry_episode",
]
