"""MMPP-modulated load producer.

Paces an aggregate message rate with a token bucket (see pacing.py) and
assigns each message to a uniformly random partition, so per-partition
arrivals are (approximately) independent Poisson thinnings of the paced
aggregate.  Payloads are 1 KiB blocks of seeded random bytes drawn from a
pregenerated pool (no compression is configured, so content is
load-irrelevant; the pool avoids burning CPU on per-message randomness).

Writes a JSON summary: realized per-second produce counts (the honesty
check on the token bucket), the modulator's ground-truth switch times
(diagnostic only -- e07 estimation never reads them), and sampled
delivery latencies.

Usage: python -m streamfno.kafka.producer --params params.json --out summary.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .pacing import TokenBucket, mmpp_timeline, rate_at
from .params import RunParams

LATENCY_SAMPLE_EVERY = 500
PAYLOAD_POOL = 256


def run_producer(params: RunParams, out_path: str,
                 latency_off: bool = False) -> dict:
    from confluent_kafka import Producer

    rng = np.random.default_rng(params.seed)
    payloads = [rng.bytes(params.payload_bytes) for _ in range(PAYLOAD_POOL)]
    if params.arrival == "mmpp":
        switches = mmpp_timeline(params.t_end, params.r_low_high,
                                 params.r_high_low, params.seed + 7_000_000)
    else:
        switches = np.array([])

    prod = Producer({
        "bootstrap.servers": params.bootstrap,
        "broker.address.family": "v4",
        "acks": "1",
        "linger.ms": 5,
        "batch.num.messages": 1000,
    })

    latencies: list[float] = []

    def on_delivery_factory(t_sent: float):
        def cb(err, msg):
            if err is None:
                latencies.append(time.time() - t_sent)
        return cb

    t_end_s = params.t_end * params.tau_s
    t0 = time.time()
    bucket = TokenBucket(rate=params.msgs_per_s(params.lam_low), burst_s=0.25)
    per_second = np.zeros(int(np.ceil(t_end_s)) + 2, dtype=np.int64)
    part_block = rng.integers(0, params.n_partitions, size=8192)
    part_idx = 0
    produced = 0

    while True:
        now = time.time()
        el = now - t0
        if el >= t_end_s:
            break
        t_norm = el / params.tau_s
        if params.arrival == "mmpp":
            lam_x = rate_at(t_norm, switches, params.lam_low, params.lam_high)
        else:
            lam_x = params.lam
        bucket.rate = params.msgs_per_s(lam_x)
        bucket.refill(now)
        n = bucket.take(2000)
        if n == 0:
            prod.poll(0.002)
            continue
        for _ in range(n):
            if part_idx == len(part_block):
                part_block = rng.integers(0, params.n_partitions, size=8192)
                part_idx = 0
            cb = None
            if not latency_off and produced % LATENCY_SAMPLE_EVERY == 0:
                cb = on_delivery_factory(time.time())
            try:
                prod.produce(params.topic, value=payloads[produced % PAYLOAD_POOL],
                             partition=int(part_block[part_idx]), on_delivery=cb)
            except BufferError:
                prod.poll(0.01)
                bucket.give_back(1)
                break
            part_idx += 1
            produced += 1
            per_second[int(el)] += 1
        prod.poll(0)

    prod.flush(30)
    lat = np.asarray(latencies)
    summary = {
        "t0_wall": t0,
        "t_end_wall": time.time(),
        "produced": produced,
        "per_second": per_second[:int(np.ceil(time.time() - t0))].tolist(),
        "mmpp_switches_norm": switches.tolist(),
        "latency_s": {
            "n": int(lat.size),
            "p50": float(np.percentile(lat, 50)) if lat.size else None,
            "p95": float(np.percentile(lat, 95)) if lat.size else None,
            "p99": float(np.percentile(lat, 99)) if lat.size else None,
        },
        "params": json.loads(params.to_json()),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-latency", action="store_true")
    args = ap.parse_args()
    run_producer(RunParams.load(args.params), args.out,
                 latency_off=args.no_latency)


if __name__ == "__main__":
    main()
