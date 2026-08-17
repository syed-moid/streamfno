"""Throughput-capped consumer: the known service rate mu.

Each consumer joins the run's consumer group and processes messages under
a token bucket cap (msgs/s).  Processing is the cap itself -- messages are
counted and discarded -- so the configured rate *is* the service rate, a
known quantity the identifiability experiment checks the estimated drift
against.  Offsets are committed every ``commit_interval_s`` (default
100 ms) to keep committed-offset staleness small relative to the lag
budget; the residual sawtooth is real telemetry quantization and is
documented rather than hidden.

Group membership events (assign / revoke) are appended to a JSONL event
log with wall timestamps; e08's rebalance-delay measurement reads these.

Runs until SIGTERM (the harness stops it), or --duration seconds.

Usage: python -m streamfno.kafka.consumer --params params.json \
        --consumer-id 0 --events events.jsonl [--rate MSGS_PER_S]
"""

from __future__ import annotations

import argparse
import json
import signal
import time

from .pacing import TokenBucket
from .params import RunParams

STATS_EVERY_S = 5.0


def run_consumer(params: RunParams, consumer_id: int, events_path: str,
                 rate_msgs: float | None = None,
                 duration_s: float | None = None) -> None:
    from confluent_kafka import Consumer

    rate = rate_msgs if rate_msgs is not None else params.consumer_rate_msgs
    ev = open(events_path, "a", buffering=1)

    def log(event: str, **kw) -> None:
        ev.write(json.dumps({"t_wall": time.time(), "event": event,
                             "consumer_id": consumer_id, **kw}) + "\n")

    def on_assign(consumer, parts):
        log("assign", n_parts=len(parts),
            parts=sorted(p.partition for p in parts))

    def on_revoke(consumer, parts):
        log("revoke", n_parts=len(parts))

    cons = Consumer({
        "bootstrap.servers": params.bootstrap,
        "broker.address.family": "v4",
        "group.id": params.group_id,
        "client.id": f"sfno-consumer-{consumer_id}",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })
    cons.subscribe([params.topic], on_assign=on_assign, on_revoke=on_revoke)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    bucket = TokenBucket(rate=rate, burst_s=0.25)
    t_start = time.time()
    consumed = 0
    dirty = False
    last_commit = t_start
    last_stats = t_start
    log("start", rate_msgs=rate)

    while not stop["flag"]:
        now = time.time()
        if duration_s is not None and now - t_start >= duration_s:
            break
        bucket.refill(now)
        n = bucket.take(500)
        if n > 0:
            msgs = cons.consume(num_messages=n, timeout=0.05)
            got = sum(1 for m in msgs if m.error() is None)
            bucket.give_back(n - got)
            consumed += got
            dirty = dirty or got > 0
        else:
            # heartbeats run on librdkafka's background thread; plain sleep
            # while throttled (poll() would consume messages past the cap)
            time.sleep(0.003)
        if dirty and now - last_commit >= params.commit_interval_s:
            cons.commit(asynchronous=True)
            last_commit, dirty = now, False
        if now - last_stats >= STATS_EVERY_S:
            log("stats", consumed=consumed)
            last_stats = now

    try:
        cons.commit(asynchronous=False)
    except Exception:
        pass
    log("stop", consumed=consumed)
    cons.close()
    ev.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--consumer-id", type=int, required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--duration", type=float, default=None)
    args = ap.parse_args()
    run_consumer(RunParams.load(args.params), args.consumer_id, args.events,
                 rate_msgs=args.rate, duration_s=args.duration)


if __name__ == "__main__":
    main()
