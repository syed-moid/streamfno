"""e08: actuation delay and telemetry overhead.

Part A (overhead A/B): two identical 3-minute constant-rate runs, one
with the 1 s lag collector running and one without; producer delivery
latencies and consumer throughput are compared offline.

Part B (actuation): one ~16-minute constant-rate run with the arrival
rate placed between the 2-consumer and 4-consumer service capacity
(lam_x = 0.8: builds lag on 2 consumers, drains on 4).  Six scale-out
cycles 2 -> 4 -> 2; the scale command time is the spawn wall time of the
new consumer processes, recorded in cycles.json.  Delays measured
offline: (i) rebalance completion from the consumers' assignment-event
logs, (ii) observed aggregate consumption-rate change from committed
offsets in the collector telemetry.
"""

import json
import time
from pathlib import Path

from streamfno.kafka.harness import (
    reset_topic,
    run_scenario,
    spawn,
    spawn_consumer,
    stop_all,
    wait_assignments,
)
from streamfno.kafka.params import RunParams

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e08"

N_CYCLES = 6
WARM_S = 90.0
UP_HOLD_S = 60.0
DOWN_HOLD_S = 75.0


def part_a() -> None:
    for tag, collector in (("overhead_on", True), ("overhead_off", False)):
        run_dir = DATA_DIR / "runs" / tag
        if (run_dir / "manifest.json").exists():
            print(f"{tag}: exists, skipping")
            continue
        print(f"{tag}: 3 min constant moderate load, collector={collector}")
        params = RunParams(arrival="poisson", lam=0.55, t_end=45.0,
                           seed=8600 if collector else 8601)
        run_scenario(run_dir, params, collector=collector)


def part_b() -> None:
    run_dir = DATA_DIR / "runs" / "scaling"
    if (run_dir / "cycles.json").exists():
        print("scaling run exists, skipping")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    t_total_s = WARM_S + N_CYCLES * (UP_HOLD_S + DOWN_HOLD_S) + 30.0
    params = RunParams(arrival="poisson", lam=0.80, seed=8700,
                       t_end=(t_total_s + 60.0) / 4.0)
    params_path = run_dir / "params.json"
    params.save(params_path)
    rate = params.consumer_rate_msgs  # cap per consumer, fixed for all

    print(f"scaling run: {t_total_s / 60:.1f} min, {N_CYCLES} cycles")
    reset_topic(params)
    procs = []
    cycles = []
    try:
        procs.append(spawn("streamfno.kafka.collector",
                           ["--params", str(params_path),
                            "--out", str(run_dir / "lag.npz"),
                            "--duration", str(t_total_s + 90.0)],
                           run_dir / "collector.log"))
        base = [spawn_consumer(params_path, cid, run_dir, rate=rate)
                for cid in (0, 1)]
        procs.extend(base)
        wait_assignments(run_dir, [0, 1])
        time.sleep(3.0)
        prod = spawn("streamfno.kafka.producer",
                     ["--params", str(params_path),
                      "--out", str(run_dir / "producer.json")],
                     run_dir / "producer.log")
        procs.append(prod)
        time.sleep(WARM_S)

        for cycle in range(N_CYCLES):
            cids = [10 * (cycle + 1) + 2, 10 * (cycle + 1) + 3]
            t_cmd = time.time()
            extra = [spawn_consumer(params_path, cid, run_dir, rate=rate)
                     for cid in cids]
            wait_assignments(run_dir, cids, timeout_s=UP_HOLD_S)
            time.sleep(UP_HOLD_S - min(UP_HOLD_S - 1.0,
                                       time.time() - t_cmd))
            t_down = time.time()
            stop_all(extra)
            cycles.append({"cycle": cycle, "cids": cids,
                           "t_scale_up_cmd": t_cmd,
                           "t_scale_down_cmd": t_down})
            print(f"  cycle {cycle}: up at {t_cmd:.1f}, down at {t_down:.1f}")
            time.sleep(DOWN_HOLD_S)
    finally:
        stop_all(procs)
    (run_dir / "cycles.json").write_text(json.dumps(
        {"cycles": cycles, "params": json.loads(params.to_json()),
         "warm_s": WARM_S, "up_hold_s": UP_HOLD_S,
         "down_hold_s": DOWN_HOLD_S}, indent=1))


def main() -> None:
    t0 = time.time()
    part_a()
    part_b()
    print(f"e08 runs done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
