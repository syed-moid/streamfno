"""e11: retention-boundary crossing (Phase E addendum; figure F7).

A sacrificial topic (separate from streamfno-exp) with minutes-scale
retention is driven into sustained lag by an overloaded MMPP workload
with capped consumers; the oldest-unconsumed-record age (record
timestamps, via the ager sampler) approaches and crosses the retention
window, and realized expiry -- the earliest offset advancing past the
committed offset -- is the ground-truth recoverability-loss event.

Topic config (recorded in decisions.md): retention.ms = 120000,
segment.ms = 15000, segment.bytes = 1 MiB, so expiry fires at ~15 s
segment granularity; brokers run log.retention.check.interval.ms =
10000 (infra/kafka-local.sh).

Runs (all: 24 partitions, 2 capped consumers, mu0 = 0.50, lam_low =
0.35, lam_high = 0.95, tau_s = 4, B = 170, t_end = 300 units = 20 min):

- calibration: balanced modulator (r_lh = 0.015, r_hl = 0.030,
  E[lam] ~ 0.55 ~ mu0) -- interior-rich occupancy for the increment
  regressions; the per-state drifts transfer to the eval mix.
- eval: overloaded modulator (r_lh = 0.030, r_hl = 0.015,
  E[lam] = 0.75 > mu0) -- secular lag growth crosses the retention
  boundary mid-run.  Adaptive stop: >= TARGET_CROSSINGS realized
  age-crossings pooled over eval runs, else up to MAX_EVAL_RUNS.

Requires the local Kafka lab: bash infra/kafka-local.sh start
(restart the lab if it predates the log.retention.check.interval.ms
setting).  Cluster work only; analysis lives in analyze.py.
"""

import json
import time
from pathlib import Path

import numpy as np

from streamfno.kafka import harness
from streamfno.kafka.params import RunParams

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e11"

TOPIC_CONFIG = {
    "retention.ms": "120000",
    "segment.ms": "15000",
    "segment.bytes": str(1 * 1024 * 1024),
}
RETENTION_S = 120.0
AGER_INTERVAL_S = 5.0
T_END = 300.0
SEED_CAL = 1150
SEEDS_EVAL = list(range(1151, 1159))
TARGET_CROSSINGS = 5


def run_params(seed: int, calibration: bool) -> RunParams:
    r_lh, r_hl = (0.015, 0.030) if calibration else (0.030, 0.015)
    return RunParams(
        topic="streamfno-ret", group_id="streamfno-ret-cg",
        n_partitions=24, n_consumers=2,
        arrival="mmpp", lam_low=0.35, lam_high=0.95,
        r_low_high=r_lh, r_high_low=r_hl, mu0=0.50,
        t_end=T_END, seed=seed,
        extra={"retention_s": RETENTION_S,
               "topic_config": TOPIC_CONFIG,
               "role": "calibration" if calibration else "eval"},
    )


def run_with_ager(run_dir: Path, params: RunParams,
                  drain_s: float = 15.0) -> None:
    """run_scenario plus the record-timestamp age sampler."""
    run_dir.mkdir(parents=True, exist_ok=True)
    params_path = run_dir / "params.json"
    params.save(params_path)
    harness.reset_topic(params)

    procs = []
    duration = params.t_end * params.tau_s + drain_s + 60.0
    try:
        procs.append(harness.spawn(
            "streamfno.kafka.collector",
            ["--params", str(params_path), "--out", str(run_dir / "lag.npz"),
             "--duration", str(duration)], run_dir / "collector.log"))
        procs.append(harness.spawn(
            "streamfno.kafka.ager",
            ["--params", str(params_path), "--out", str(run_dir / "age.npz"),
             "--duration", str(duration),
             "--interval", str(AGER_INTERVAL_S)], run_dir / "ager.log"))
        consumers = [harness.spawn_consumer(params_path, cid, run_dir)
                     for cid in range(params.n_consumers)]
        procs.extend(consumers)
        harness.wait_assignments(run_dir, list(range(params.n_consumers)))
        time.sleep(3.0)
        t_prod_spawn = time.time()
        prod = harness.spawn(
            "streamfno.kafka.producer",
            ["--params", str(params_path),
             "--out", str(run_dir / "producer.json")],
            run_dir / "producer.log")
        rc = prod.wait(timeout=duration + 120)
        if rc != 0:
            raise RuntimeError(f"producer exited {rc}; see producer.log")
        time.sleep(drain_s)
    finally:
        harness.stop_all(procs)

    summary = json.loads((run_dir / "producer.json").read_text())
    (run_dir / "manifest.json").write_text(json.dumps({
        "t0_wall": summary["t0_wall"], "t_prod_spawn": t_prod_spawn,
        "produced": summary["produced"],
        "params": json.loads(params.to_json()),
        "topic_config": TOPIC_CONFIG, "retention_s": RETENTION_S,
        "drain_s": drain_s}, indent=1))


def crossings_in(run_dir: Path) -> int:
    """Realized age-crossing episodes in a finished run (max-partition
    age crossing RETENTION_S from below, separated by a reset below
    half the window)."""
    with np.load(run_dir / "age.npz") as f:
        age = np.nanmax(np.nan_to_num(f["age_s"], nan=0.0), axis=1)
    n, armed = 0, True
    for a in age:
        if armed and a >= RETENTION_S:
            n += 1
            armed = False
        elif not armed and a < RETENTION_S / 2.0:
            armed = True
    return n


def main() -> None:
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved_config = harness.TOPIC_CONFIG
    harness.TOPIC_CONFIG = TOPIC_CONFIG
    try:
        cal_dir = DATA_DIR / "runs" / "calibration"
        if not (cal_dir / "manifest.json").exists():
            print(f"calibration run ({T_END * 4 / 60:.0f} min)...")
            run_with_ager(cal_dir, run_params(SEED_CAL, calibration=True))

        total = 0
        for seed in SEEDS_EVAL:
            if total >= TARGET_CROSSINGS:
                break
            run_dir = DATA_DIR / "runs" / f"s{seed}"
            if not (run_dir / "manifest.json").exists():
                print(f"eval run seed {seed} ({T_END * 4 / 60:.0f} min)...")
                run_with_ager(run_dir, run_params(seed, calibration=False))
            c = crossings_in(run_dir)
            total += c
            print(f"  seed {seed}: {c} crossing(s); pooled {total}")
    finally:
        harness.TOPIC_CONFIG = saved_config
    print(f"e11 runs done in {(time.time() - t0) / 60:.1f} min; "
          f"stop the lab when finished: bash infra/kafka-local.sh stop")


if __name__ == "__main__":
    main()
