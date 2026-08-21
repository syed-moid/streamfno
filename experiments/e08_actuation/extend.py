"""e08 extension (revision B4): a second scaling session of 9 more
scale-out cycles (fresh seed, separate run dir), pooled with the
original 6 for ~15 actuation-delay measurements. Same protocol as
run.py part B; delays computed with analyze.py's estimators applied to
each session; pooled medians/IQRs saved to data/e08/actuation_pooled.json.
Resumable; run under caffeinate -i.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from streamfno.kafka.adapter import load_telemetry
from streamfno.kafka.harness import (
    reset_topic,
    spawn,
    spawn_consumer,
    stop_all,
    wait_assignments,
)
from streamfno.kafka.params import RunParams

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e08"

_spec = importlib.util.spec_from_file_location(
    "e08_analyze", Path(__file__).with_name("analyze.py"))
e08a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e08a)

N_CYCLES2 = 9
WARM_S = 90.0
UP_HOLD_S = 60.0
DOWN_HOLD_S = 75.0
SEED2 = 8701


def scaling_session(run_dir: Path, n_cycles: int, seed: int) -> None:
    if (run_dir / "cycles.json").exists():
        print(f"{run_dir.name}: exists, skipping", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    t_total_s = WARM_S + n_cycles * (UP_HOLD_S + DOWN_HOLD_S) + 30.0
    params = RunParams(arrival="poisson", lam=0.80, seed=seed,
                       t_end=(t_total_s + 60.0) / 4.0)
    params_path = run_dir / "params.json"
    params.save(params_path)
    rate = params.consumer_rate_msgs
    print(f"{run_dir.name}: {t_total_s / 60:.1f} min, {n_cycles} cycles",
          flush=True)
    reset_topic(params)
    procs, cycles = [], []
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
        for cycle in range(n_cycles):
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
            print(f"  cycle {cycle}: done", flush=True)
            time.sleep(DOWN_HOLD_S)
    finally:
        stop_all(procs)
    (run_dir / "cycles.json").write_text(json.dumps(
        {"cycles": cycles, "params": json.loads(params.to_json()),
         "warm_s": WARM_S, "up_hold_s": UP_HOLD_S,
         "down_hold_s": DOWN_HOLD_S}, indent=1))


def session_delays(run_dir: Path) -> list[dict]:
    """analyze.py's two delay estimators applied to one session."""
    meta = json.loads((run_dir / "cycles.json").read_text())
    params = RunParams.from_json(json.dumps(meta["params"]))
    tel = load_telemetry(run_dir / "lag.npz")
    events = e08a.consumer_events(run_dir)
    com = np.maximum(tel.committed, 0).sum(axis=1).astype(float)
    rate = np.zeros(com.size)
    rate[1:] = np.diff(com) / np.diff(tel.t_wall)
    kern = np.ones(e08a.SMOOTH_S) / e08a.SMOOTH_S
    rate_s = np.convolve(rate, kern, mode="same")
    threshold = 0.5 * (2 + 4) * params.consumer_rate_msgs
    rows = []
    for c in meta["cycles"]:
        t_cmd = c["t_scale_up_cmd"]
        t_next = c["t_scale_down_cmd"] - 0.5
        assigns = [e for e in events if e["event"] == "assign"
                   and t_cmd <= e["t_wall"] < t_next]
        d_reb = (max(e["t_wall"] for e in assigns) - t_cmd
                 if assigns else None)
        d_rate = None
        idx = np.flatnonzero((tel.t_wall >= t_cmd) & (tel.t_wall < t_next))
        above = rate_s[idx] > threshold
        for j in range(len(above) - e08a.SUSTAIN_TICKS + 1):
            if above[j:j + e08a.SUSTAIN_TICKS].all():
                d_rate = float(tel.t_wall[idx[j]] - t_cmd)
                break
        rows.append({"session": run_dir.name, "cycle": c["cycle"],
                     "d_rebalance_s": d_reb, "d_rate_s": d_rate})
    return rows


def main() -> None:
    scaling_session(DATA_DIR / "runs" / "scaling2", N_CYCLES2, SEED2)
    rows = (session_delays(DATA_DIR / "runs" / "scaling")
            + session_delays(DATA_DIR / "runs" / "scaling2"))
    reb = [r["d_rebalance_s"] for r in rows
           if r["d_rebalance_s"] is not None]
    rat = [r["d_rate_s"] for r in rows if r["d_rate_s"] is not None]
    out = {
        "n_cycles": len(rows),
        "rebalance_s": {"median": float(np.median(reb)),
                        "iqr": [float(v) for v in
                                np.percentile(reb, [25, 75])],
                        "n": len(reb)},
        "rate_effect_s": {"median": float(np.median(rat)),
                          "iqr": [float(v) for v in
                                  np.percentile(rat, [25, 75])],
                          "n": len(rat)},
        "cycles": rows,
    }
    (DATA_DIR / "actuation_pooled.json").write_text(json.dumps(out,
                                                               indent=1))
    print(f"pooled over {len(rows)} cycles: rebalance median "
          f"{out['rebalance_s']['median']:.2f} s IQR "
          f"{out['rebalance_s']['iqr']}; rate-effect median "
          f"{out['rate_effect_s']['median']:.2f} s IQR "
          f"{out['rate_effect_s']['iqr']}", flush=True)


if __name__ == "__main__":
    main()
