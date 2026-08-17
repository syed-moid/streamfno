"""e08 analysis: actuation-delay distributions and collector overhead."""

import json
from pathlib import Path

import numpy as np

from streamfno.kafka.adapter import load_telemetry
from streamfno.kafka.params import RunParams

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e08"

SMOOTH_S = 3          # consumption-rate smoothing (ticks)
SUSTAIN_TICKS = 3     # crossing must hold this long


def consumer_events(run_dir: Path) -> list[dict]:
    out = []
    for path in sorted(run_dir.glob("consumer_*.jsonl")):
        out += [json.loads(line) for line in path.read_text().splitlines()]
    return sorted(out, key=lambda r: r["t_wall"])


def actuation_delays() -> dict:
    run_dir = DATA_DIR / "runs" / "scaling"
    meta = json.loads((run_dir / "cycles.json").read_text())
    params = RunParams.from_json(json.dumps(meta["params"]))
    tel = load_telemetry(run_dir / "lag.npz")
    events = consumer_events(run_dir)

    com = np.maximum(tel.committed, 0).sum(axis=1).astype(float)
    rate = np.zeros(com.size)
    rate[1:] = np.diff(com) / np.diff(tel.t_wall)
    kern = np.ones(SMOOTH_S) / SMOOTH_S
    rate_s = np.convolve(rate, kern, mode="same")

    mu2 = 2 * params.consumer_rate_msgs
    mu4 = 4 * params.consumer_rate_msgs
    threshold = 0.5 * (mu2 + mu4)

    rows = []
    for c in meta["cycles"]:
        t_cmd = c["t_scale_up_cmd"]
        # window ends at the scale-down command: its own rebalance would
        # otherwise be counted as late completion of the scale-up
        t_next = c["t_scale_down_cmd"] - 0.5
        # (i) rebalance completion: last assign event in the window
        assigns = [e for e in events
                   if e["event"] == "assign" and t_cmd <= e["t_wall"] < t_next]
        d_rebalance = max(e["t_wall"] for e in assigns) - t_cmd if assigns else None
        # (ii) sustained crossing of the service-rate midpoint
        d_rate = None
        idx = np.flatnonzero((tel.t_wall >= t_cmd) & (tel.t_wall < t_next))
        above = rate_s[idx] > threshold
        for j in range(len(above) - SUSTAIN_TICKS + 1):
            if above[j:j + SUSTAIN_TICKS].all():
                d_rate = float(tel.t_wall[idx[j]] - t_cmd)
                break
        rows.append({"cycle": c["cycle"], "d_rebalance_s": d_rebalance,
                     "d_rate_s": d_rate, "n_assign_events": len(assigns)})
    return {"threshold_msgs_s": threshold, "mu2": mu2, "mu4": mu4,
            "cycles": rows}


def overhead() -> dict:
    out = {}
    for tag in ("overhead_on", "overhead_off"):
        run_dir = DATA_DIR / "runs" / tag
        prod = json.loads((run_dir / "producer.json").read_text())
        rec = {"latency_s": prod["latency_s"],
               "produced": prod["produced"]}
        rates = []
        for path in sorted(run_dir.glob("consumer_*.jsonl")):
            stats = [json.loads(line) for line in
                     path.read_text().splitlines()
                     if '"stats"' in line]
            if len(stats) >= 2:
                rates.append((stats[-1]["consumed"] - stats[0]["consumed"])
                             / (stats[-1]["t_wall"] - stats[0]["t_wall"]))
        rec["consumer_rates"] = rates
        out[tag] = rec
    # collector self-cost from the scaling run (longest one)
    tel = load_telemetry(DATA_DIR / "runs" / "scaling" / "lag.npz")
    cpu = np.diff(tel.cpu_s) / np.diff(tel.t_wall)
    out["collector"] = {
        "cpu_frac_mean": float(cpu.mean()),
        "cpu_frac_p95": float(np.percentile(cpu, 95)),
        "sample_latency_p50_s": float(np.percentile(tel.sample_latency_s, 50)),
        "sample_latency_p95_s": float(np.percentile(tel.sample_latency_s, 95)),
        "n_ticks": int(len(tel.t_wall)),
    }
    return out


def main() -> None:
    res = {"actuation": actuation_delays(), "overhead": overhead()}
    (DATA_DIR / "results.json").write_text(json.dumps(res, indent=1))

    act = res["actuation"]
    print("actuation delays (s):")
    for r in act["cycles"]:
        print(f"  cycle {r['cycle']}: rebalance {r['d_rebalance_s']}, "
              f"rate-change {r['d_rate_s']}")
    reb = [r["d_rebalance_s"] for r in act["cycles"]
           if r["d_rebalance_s"] is not None]
    rat = [r["d_rate_s"] for r in act["cycles"] if r["d_rate_s"] is not None]
    if reb:
        print(f"  rebalance: median {np.median(reb):.1f}  "
              f"range [{min(reb):.1f}, {max(reb):.1f}]  n={len(reb)}")
    if rat:
        print(f"  rate-change: median {np.median(rat):.1f}  "
              f"range [{min(rat):.1f}, {max(rat):.1f}]  n={len(rat)}")

    ov = res["overhead"]
    print("overhead:")
    for tag in ("overhead_on", "overhead_off"):
        lat = ov[tag]["latency_s"]
        print(f"  {tag}: p50={lat['p50']:.4f}s p95={lat['p95']:.4f}s "
              f"p99={lat['p99']:.4f}s consumer rates "
              f"{[round(r) for r in ov[tag]['consumer_rates']]}")
    c = ov["collector"]
    print(f"  collector: cpu {c['cpu_frac_mean'] * 100:.2f}% of one core "
          f"(p95 {c['cpu_frac_p95'] * 100:.2f}%), "
          f"sample latency p50 {c['sample_latency_p50_s'] * 1e3:.1f} ms "
          f"p95 {c['sample_latency_p95_s'] * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
