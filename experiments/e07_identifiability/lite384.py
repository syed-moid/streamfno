"""e07-lite at 384 partitions (revision B5): one calibration + one
held-out eval run on the same 3-broker lab, showing forecast quality
does not depend delicately on N. Same rates and unit mapping as
e06-heavy/e07 (lam 0.40/0.82, mu0 0.70, r_low_high 0.10); analysis
reuses the e07 machinery verbatim (estimate on calibration, forecast
eval bursts, pooled W1 vs persistence, onsets). The full N-sweep is
deferred to the scale-out paper. Resumable; run under caffeinate -i.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from streamfno.kafka.harness import run_scenario
from streamfno.kafka.params import RunParams

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e07_lite384"

_spec = importlib.util.spec_from_file_location(
    "e07_analyze", Path(__file__).with_name("analyze.py"))
e07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e07)

SEED_CAL, SEED_EVAL = 1300, 1301
T_END = 300.0


def params(seed: int) -> RunParams:
    return RunParams(topic="streamfno-384", group_id="streamfno-384-cg",
                     n_partitions=384, arrival="mmpp", r_low_high=0.10,
                     t_end=T_END, seed=seed)


def main() -> None:
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for tag, seed in (("calibration", SEED_CAL), ("eval", SEED_EVAL)):
        run_dir = DATA_DIR / "runs" / tag
        if not (run_dir / "manifest.json").exists():
            print(f"{tag} run at 384 partitions "
                  f"({T_END * 4 / 60:.0f} min)...", flush=True)
            run_scenario(run_dir, params(seed))

    print("estimating from the 384-partition calibration...", flush=True)
    est = e07.estimate(DATA_DIR / "runs" / "calibration")
    for state in ("low", "high"):
        r = est[state]
        if r["b_hat"] is not None:
            print(f"  {state:<5} b_hat = {r['b_hat']:+.4f}  configured = "
                  f"{r['b_configured']:+.4f}  rel err = "
                  f"{r['b_rel_error']:.3f}", flush=True)
    eps = e07.calibrate_eps(DATA_DIR / "runs" / "calibration")
    fc = e07.forecast_eval(est, eps, run_dir=DATA_DIR / "runs" / "eval")
    w1p = [float(np.mean(b["w1_pred"])) for b in fc["bursts"]]
    w1q = [float(np.mean(b["w1_pers"])) for b in fc["bursts"]]
    hits = [b for b in fc["bursts"] if b["onset_pred"] is not None
            and b["onset_real"] is not None]
    out = {
        "n_partitions": 384, "seeds": [SEED_CAL, SEED_EVAL], "eps": eps,
        "estimation": {s: {k: est[s][k] for k in
                           ("b_hat", "b_se", "b_configured", "b_rel_error",
                            "a_hat_d1", "a_hat_d2", "a_noise_corrected")}
                       for s in ("low", "high")},
        "n_bursts": len(fc["bursts"]),
        "w1_pred_mean": float(np.mean(w1p)) if w1p else None,
        "w1_pers_mean": float(np.mean(w1q)) if w1q else None,
        "onsets": [{"pred": b["onset_pred"], "real": b["onset_real"]}
                   for b in fc["bursts"]],
        "n_onset_hits": len(hits),
        "onset_errors": [b["onset_pred"] - b["onset_real"] for b in hits],
        "bursts": fc["bursts"],
    }
    (DATA_DIR / "lite384.json").write_text(json.dumps(out, indent=1))
    print(f"384-partition eval: {len(fc['bursts'])} bursts, "
          f"W1 pred {out['w1_pred_mean']} vs pers {out['w1_pers_mean']}; "
          f"onset hits {len(hits)}; saved "
          f"({(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
