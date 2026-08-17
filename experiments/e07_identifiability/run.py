"""e07: the held-out evaluation run.

Calibration telemetry is the e06 heavy run (a separate execution with its
own seed); this script performs only the evaluation run: same MMPP
parameters, different seed, so burst phases and magnitudes are new.
Estimation never touches this run (honesty rule: no re-fitting on the
evaluation run); analyze.py forecasts its bursts from coefficients fitted
on calibration telemetry alone.
"""

import json
import time
from pathlib import Path

from streamfno.kafka.harness import run_scenario
from streamfno.kafka.params import RunParams

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e07"

EVAL_SEED = 9600
T_END = 300.0


def main() -> None:
    run_dir = DATA_DIR / "runs" / "eval"
    if (run_dir / "manifest.json").exists():
        print("eval run exists, skipping")
        return
    t0 = time.time()
    params = RunParams(arrival="mmpp", r_low_high=0.10, t_end=T_END,
                       seed=EVAL_SEED)
    print(f"eval run: {T_END * params.tau_s / 60:.0f} min, seed {EVAL_SEED}...")
    run_scenario(run_dir, params)
    man = json.loads((run_dir / "manifest.json").read_text())
    print(f"produced {man['produced']} msgs in "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
