"""e06: spectral decay of the real lag density at three load levels.

Three >= 20 min runs against the local 3-node Kafka cluster, load level =
burst frequency of the shared MMPP modulator (the Phase C convention;
constant-rate near-saturation is untunable at message scale, where the
queue length scale is ~1/(1-rho) messages).  Densities are stationary-
window, cycle-averaged empirical X histograms at the collector cadence;
spectra and tail fits reuse the e01 estimators unchanged.  The heavy run
doubles as e07's calibration run (separate execution and seed from e07's
evaluation run).

Requires the lab cluster (infra/kafka-local.sh start).  Cluster runs and
offline analysis are split: `run.py` performs the runs, `analyze.py` the
spectra; both are idempotent (existing artifacts are kept).
"""

import json
import time
from pathlib import Path

from streamfno.kafka.harness import run_scenario
from streamfno.kafka.params import RunParams

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e06"

T_END = 300.0                 # 20 min at tau_s = 4
LEVELS = {"light": 0.03, "moderate": 0.06, "heavy": 0.10}
SEEDS = {"light": 1600, "moderate": 2600, "heavy": 4600}


def level_params(level: str) -> RunParams:
    return RunParams(arrival="mmpp", r_low_high=LEVELS[level],
                     t_end=T_END, seed=SEEDS[level])


def main() -> None:
    t0 = time.time()
    for level in LEVELS:
        run_dir = DATA_DIR / "runs" / level
        if (run_dir / "manifest.json").exists():
            print(f"{level}: run exists, skipping")
            continue
        print(f"{level}: running {T_END * 4 / 60:.0f} min "
              f"(r_low_high={LEVELS[level]})...")
        run_scenario(run_dir, level_params(level))
        man = json.loads((run_dir / "manifest.json").read_text())
        print(f"  produced {man['produced']} msgs")
    print(f"e06 runs done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
