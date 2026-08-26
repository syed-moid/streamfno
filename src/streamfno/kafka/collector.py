"""Per-partition consumer-lag collector.

Samples, at a fixed wall interval (default 1 s), the per-partition log-end
offset (one batched AdminClient.list_offsets round trip) and the group's
committed offsets (one list_consumer_group_offsets round trip); lag is
their difference.  No JMX involved.

Records its own overhead alongside the data: cumulative process CPU time
(resource.getrusage) at every tick and the wall latency of each sampling
round trip, so the collector's cost and any perturbation it causes are
measurable rather than assumed.

Output: an .npz with t_wall (K,), leo (K, P), committed (K, P; -1 before
the first commit), earliest (K, P; log-start offsets -- their advance
past ``committed`` is the ground-truth recoverability-loss event for the
retention boundary), sample_latency_s (K,), cpu_s (K,), meta_json.
Checkpoints atomically every 60 s so a killed run keeps its telemetry.

Usage: python -m streamfno.kafka.collector --params params.json \
        --out lag.npz --duration SECONDS
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import time

import numpy as np

from .params import RunParams

CHECKPOINT_EVERY_S = 60.0
MAX_CONSECUTIVE_FAILURES = 60


def _cpu_s() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def run_collector(params: RunParams, out_path: str, duration_s: float) -> None:
    from confluent_kafka import KafkaException, TopicPartition
    from confluent_kafka.admin import AdminClient, OffsetSpec

    admin = AdminClient({"bootstrap.servers": params.bootstrap,
                         "broker.address.family": "v4"})
    tps = [TopicPartition(params.topic, p) for p in range(params.n_partitions)]

    rows_t, rows_leo, rows_com, rows_lat, rows_cpu = [], [], [], [], []
    rows_earliest = []
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    def save() -> None:
        tmp = out_path + ".tmp.npz"
        np.savez_compressed(
            tmp,
            t_wall=np.asarray(rows_t),
            leo=np.asarray(rows_leo, dtype=np.int64),
            committed=np.asarray(rows_com, dtype=np.int64),
            earliest=np.asarray(rows_earliest, dtype=np.int64),
            sample_latency_s=np.asarray(rows_lat),
            cpu_s=np.asarray(rows_cpu),
            meta_json=np.array(json.dumps({
                "params": json.loads(params.to_json()),
                "pid": os.getpid(),
                "dt_poll_s": params.dt_poll_s,
            })),
        )
        os.replace(tmp, out_path)

    def sample_offsets(spec) -> np.ndarray:
        futs = admin.list_offsets({tp: spec for tp in tps})
        out = np.full(params.n_partitions, -1, dtype=np.int64)
        for tp, fut in futs.items():
            out[tp.partition] = fut.result().offset
        return out

    def sample_committed() -> np.ndarray:
        from confluent_kafka import ConsumerGroupTopicPartitions
        req = ConsumerGroupTopicPartitions(params.group_id, tps)
        futs = admin.list_consumer_group_offsets([req])
        res = list(futs.values())[0].result()
        out = np.full(params.n_partitions, -1, dtype=np.int64)
        for tp in res.topic_partitions:
            if tp.topic == params.topic and tp.offset >= 0:
                out[tp.partition] = tp.offset
        return out

    t0 = time.time()
    next_tick = t0
    last_ckpt = t0
    consecutive_failures = 0
    while not stop["flag"] and time.time() - t0 < duration_s:
        now = time.time()
        if now < next_tick:
            time.sleep(min(next_tick - now, 0.05))
            continue
        next_tick += params.dt_poll_s
        tick = time.time()
        try:
            leo = sample_offsets(OffsetSpec.latest())
            com = sample_committed()
            early = sample_offsets(OffsetSpec.earliest())
        except KafkaException as exc:
            # Transient during large-topic leader election (e.g. "No
            # leaders found" right after a 384-partition create) or a
            # mid-run election; skip the tick, keep the cadence.
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise
            print(f"sample failed ({exc}); tick skipped "
                  f"({consecutive_failures} consecutive)", flush=True)
            continue
        consecutive_failures = 0
        rows_lat.append(time.time() - tick)
        rows_t.append(tick)
        rows_leo.append(leo)
        rows_com.append(com)
        rows_earliest.append(early)
        rows_cpu.append(_cpu_s())
        if tick - last_ckpt >= CHECKPOINT_EVERY_S:
            save()
            last_ckpt = tick
    save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, required=True)
    args = ap.parse_args()
    run_collector(RunParams.load(args.params), args.out, args.duration)


if __name__ == "__main__":
    main()
