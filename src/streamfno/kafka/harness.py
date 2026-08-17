"""Run orchestration for real-cluster scenarios.

A scenario run is: reset the experiment topic, start the collector, start
the capped consumers, wait for group assignment, start the producer, wait
for it to finish, let the consumers drain briefly, stop everything, and
write a manifest.  All child processes are this venv's python running the
module CLIs, so everything shares one interpreter and one clock.

e08 drives the lower-level pieces directly (it scales the consumer group
mid-run instead of using run_scenario's fixed group).
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from .params import RunParams

TOPIC_CONFIG = {
    "retention.ms": "600000",
    "segment.ms": "120000",
    "segment.bytes": str(64 * 1024 * 1024),
}

__all__ = ["reset_topic", "run_scenario", "spawn", "stop_all",
           "spawn_consumer", "wait_assignments", "TOPIC_CONFIG"]


def reset_topic(params: RunParams, timeout_s: float = 60.0) -> None:
    """Delete and recreate the experiment topic (fresh offsets, bounded
    disk), and drop the consumer group so committed offsets reset."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": params.bootstrap,
                         "broker.address.family": "v4"})
    md = admin.list_topics(timeout=10)
    if params.topic in md.topics:
        admin.delete_topics([params.topic], operation_timeout=30)[
            params.topic].result()
    try:
        admin.delete_consumer_groups([params.group_id])[
            params.group_id].result()
    except Exception:
        pass  # group may not exist / already empty
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        md = admin.list_topics(timeout=10)
        if params.topic not in md.topics:
            break
        time.sleep(1.0)
    admin.create_topics([NewTopic(params.topic,
                                  num_partitions=params.n_partitions,
                                  replication_factor=3,
                                  config=TOPIC_CONFIG)])[params.topic].result()
    time.sleep(2.0)  # let leadership settle


def spawn(module: str, args: list[str], log_path: Path) -> subprocess.Popen:
    log = open(log_path, "a")
    return subprocess.Popen(
        [sys.executable, "-m", module, *args], stdout=log, stderr=log)


def spawn_consumer(params_path: Path, cid: int, run_dir: Path,
                   rate: float | None = None) -> subprocess.Popen:
    args = ["--params", str(params_path), "--consumer-id", str(cid),
            "--events", str(run_dir / f"consumer_{cid}.jsonl")]
    if rate is not None:
        args += ["--rate", str(rate)]
    return spawn("streamfno.kafka.consumer", args,
                 run_dir / f"consumer_{cid}.log")


def wait_assignments(run_dir: Path, cids: list[int],
                     timeout_s: float = 90.0) -> None:
    """Block until every listed consumer has logged a nonempty assignment."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ok = 0
        for cid in cids:
            path = run_dir / f"consumer_{cid}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                if rec["event"] == "assign" and rec["n_parts"] > 0:
                    ok += 1
                    break
        if ok == len(cids):
            return
        time.sleep(1.0)
    raise TimeoutError(f"consumers {cids} not assigned within {timeout_s}s")


def stop_all(procs: list[subprocess.Popen], grace_s: float = 15.0) -> None:
    for p in procs:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
    deadline = time.time() + grace_s
    for p in procs:
        try:
            p.wait(timeout=max(0.5, deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()


def run_scenario(run_dir: str | Path, params: RunParams,
                 drain_s: float = 15.0, collector: bool = True) -> Path:
    """Execute one full scenario; returns the run directory."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    params_path = run_dir / "params.json"
    params.save(params_path)

    reset_topic(params)

    procs: list[subprocess.Popen] = []
    duration = params.t_end * params.tau_s + drain_s + 60.0
    try:
        if collector:
            procs.append(spawn("streamfno.kafka.collector",
                               ["--params", str(params_path),
                                "--out", str(run_dir / "lag.npz"),
                                "--duration", str(duration)],
                               run_dir / "collector.log"))
        consumers = [spawn_consumer(params_path, cid, run_dir)
                     for cid in range(params.n_consumers)]
        procs.extend(consumers)
        wait_assignments(run_dir, list(range(params.n_consumers)))
        time.sleep(3.0)

        t_prod_spawn = time.time()
        prod = spawn("streamfno.kafka.producer",
                     ["--params", str(params_path),
                      "--out", str(run_dir / "producer.json")],
                     run_dir / "producer.log")
        rc = prod.wait(timeout=duration + 120)
        if rc != 0:
            raise RuntimeError(f"producer exited with {rc}; see producer.log")
        time.sleep(drain_s)
    finally:
        stop_all(procs)

    summary = json.loads((run_dir / "producer.json").read_text())
    manifest = {
        "t0_wall": summary["t0_wall"],
        "t_prod_spawn": t_prod_spawn,
        "produced": summary["produced"],
        "params": json.loads(params.to_json()),
        "drain_s": drain_s,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return run_dir
