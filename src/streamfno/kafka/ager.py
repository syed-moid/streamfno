"""Oldest-unconsumed-record age sampler (the retention boundary's
primary observable, from record timestamps).

At a fixed cadence: read the group's committed offsets (AdminClient),
assign a dedicated read-only consumer at those offsets, and poll until
one record per lagging partition is seen; the per-partition age is
(sample wall time - record timestamp).  Partitions with no lag get age
zero; partitions whose committed offset has already been deleted
(earliest > committed) get the age of the earliest surviving record and
an ``expired`` mark -- after expiry the true oldest-unconsumed record no
longer exists, so its age is right-censored at the retention window.

The stationary-regime interpretation A_lag ~ L / lambda_eff is left to
the analysis; this module records the definition (timestamps) only.

Output: .npz with t_wall (K,), age_s (K, P), expired (K, P bool),
meta_json.  Checkpoints atomically every 60 s.

Usage: python -m streamfno.kafka.ager --params params.json \
        --out age.npz --duration SECONDS [--interval 5.0]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time

import numpy as np

from .params import RunParams

CHECKPOINT_EVERY_S = 60.0
POLL_BUDGET_S = 3.0


def run_ager(params: RunParams, out_path: str, duration_s: float,
             interval_s: float) -> None:
    from confluent_kafka import Consumer, ConsumerGroupTopicPartitions, TopicPartition
    from confluent_kafka.admin import AdminClient, OffsetSpec

    admin = AdminClient({"bootstrap.servers": params.bootstrap,
                         "broker.address.family": "v4"})
    consumer = Consumer({
        "bootstrap.servers": params.bootstrap,
        "group.id": f"{params.group_id}-ager",   # never joins group protocol
        "enable.auto.commit": False,
        "broker.address.family": "v4",
        "fetch.max.bytes": 1 << 20,
        "auto.offset.reset": "earliest",
    })
    tps = [TopicPartition(params.topic, p) for p in range(params.n_partitions)]

    rows_t, rows_age, rows_exp = [], [], []
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    def save() -> None:
        tmp = out_path + ".tmp.npz"
        np.savez_compressed(
            tmp, t_wall=np.asarray(rows_t),
            age_s=np.asarray(rows_age),
            expired=np.asarray(rows_exp, dtype=bool),
            meta_json=np.array(json.dumps({
                "params": json.loads(params.to_json()),
                "interval_s": interval_s, "pid": os.getpid()})))
        os.replace(tmp, out_path)

    def offsets(spec_or_committed) -> np.ndarray:
        out = np.full(params.n_partitions, -1, dtype=np.int64)
        if spec_or_committed == "committed":
            req = ConsumerGroupTopicPartitions(params.group_id, tps)
            res = list(admin.list_consumer_group_offsets([req]).values())[0]
            for tp in res.result().topic_partitions:
                if tp.topic == params.topic and tp.offset >= 0:
                    out[tp.partition] = tp.offset
        else:
            futs = admin.list_offsets({tp: spec_or_committed for tp in tps})
            for tp, fut in futs.items():
                out[tp.partition] = fut.result().offset
        return out

    t0 = time.time()
    next_tick = t0
    last_ckpt = t0
    while not stop["flag"] and time.time() - t0 < duration_s:
        now = time.time()
        if now < next_tick:
            time.sleep(min(next_tick - now, 0.1))
            continue
        next_tick += interval_s
        tick = time.time()
        com = offsets("committed")
        leo = offsets(OffsetSpec.latest())
        early = offsets(OffsetSpec.earliest())
        com_eff = np.where(com >= 0, com, 0)      # pre-commit: all is unread
        lagging = (leo > com_eff) & (leo >= 0)
        expired = lagging & (early > com_eff)
        seek_from = np.where(expired, early, com_eff)

        age = np.zeros(params.n_partitions)
        want = [p for p in range(params.n_partitions) if lagging[p]]
        if want:
            consumer.assign([TopicPartition(params.topic, p,
                                            int(seek_from[p]))
                             for p in want])
            seen: dict[int, float] = {}
            deadline = time.time() + POLL_BUDGET_S
            while len(seen) < len(want) and time.time() < deadline:
                msg = consumer.poll(0.2)
                if msg is None or msg.error():
                    continue
                p = msg.partition()
                if p not in seen:
                    ts_type, ts_ms = msg.timestamp()
                    if ts_ms > 0:
                        seen[p] = max(0.0, tick - ts_ms / 1e3)
            consumer.unassign()
            for p in want:
                age[p] = seen.get(p, np.nan)      # nan: not sampled this tick
        rows_t.append(tick)
        rows_age.append(age)
        rows_exp.append(expired)
        if tick - last_ckpt >= CHECKPOINT_EVERY_S:
            save()
            last_ckpt = tick
    save()
    consumer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()
    run_ager(RunParams.load(args.params), args.out, args.duration,
             args.interval)


if __name__ == "__main__":
    main()
