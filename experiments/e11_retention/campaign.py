"""e11 retention-grid campaign (revision B4; pre-registered in
docs/decisions.md before launch).

Cells B-G vary one condition each against the baseline cell A (the
five existing crossings): retention window (60 s / 240 s), partition
count (48), burst shape (faster switching, same mean load), lam/mu
ratio (0.55/0.90, with its own subcritical calibration run -- the only
cell whose rate levels change), and consumer count (1). Target >= 3
realized crossings per cell (else the cell is labeled exploratory);
adaptive up to two extra runs per cell. Eval seeds are disjoint from
every calibration seed. Resumable per run; heartbeated; run under
caffeinate -i.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from streamfno.kafka import harness
from streamfno.kafka.params import RunParams

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "e11"

_spec = importlib.util.spec_from_file_location(
    "e11_run", Path(__file__).with_name("run.py"))
e11run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e11run)

TARGET_PER_CELL = 3

CELLS = {
    "B": {"retention_s": 60.0, "t_end": 200.0,
          "seeds": list(range(1200, 1205)), "overrides": {}},
    "C": {"retention_s": 240.0, "t_end": 550.0,
          "seeds": list(range(1210, 1215)), "overrides": {}},
    "D": {"retention_s": 120.0, "t_end": 300.0,
          "seeds": list(range(1220, 1225)),
          "overrides": {"n_partitions": 48}},
    "E": {"retention_s": 120.0, "t_end": 300.0,
          "seeds": list(range(1230, 1235)),
          "overrides": {"r_low_high": 0.06, "r_high_low": 0.03}},
    "F": {"retention_s": 120.0, "t_end": 300.0,
          "seeds": list(range(1241, 1246)),
          "overrides": {"mu0": 0.55, "lam_high": 0.90},
          "calibration": {"seed": 1240,
                          "overrides": {"mu0": 0.55, "lam_high": 0.90,
                                        "r_low_high": 0.015,
                                        "r_high_low": 0.045}}},
    "G": {"retention_s": 120.0, "t_end": 300.0,
          "seeds": list(range(1250, 1255)),
          "overrides": {"n_consumers": 1}},
}


def cell_params(cell: str, seed: int, t_end: float, retention_s: float,
                overrides: dict, role: str) -> RunParams:
    base = dict(topic="streamfno-ret", group_id="streamfno-ret-cg",
                n_partitions=24, n_consumers=2,
                arrival="mmpp", lam_low=0.35, lam_high=0.95,
                r_low_high=0.030, r_high_low=0.015, mu0=0.50,
                t_end=t_end, seed=seed)
    base.update(overrides)
    topic_config = {"retention.ms": str(int(retention_s * 1000)),
                    "segment.ms": "15000",
                    "segment.bytes": str(1024 * 1024)}
    return RunParams(**base, extra={"retention_s": retention_s,
                                    "topic_config": topic_config,
                                    "role": role, "cell": cell})


def crossings_in(run_dir: Path, retention_s: float) -> int:
    with np.load(run_dir / "age.npz") as f:
        age = np.nanmax(np.nan_to_num(f["age_s"], nan=0.0), axis=1)
    n, armed = 0, True
    for a in age:
        if armed and a >= retention_s:
            n += 1
            armed = False
        elif not armed and a < retention_s / 2.0:
            armed = True
    return n


def do_run(run_dir: Path, params: RunParams) -> None:
    """run.py's orchestration with per-cell topic config and manifest
    fields."""
    cfg = params.extra["topic_config"]
    saved_h, saved_t, saved_r = (harness.TOPIC_CONFIG, e11run.TOPIC_CONFIG,
                                 e11run.RETENTION_S)
    harness.TOPIC_CONFIG = cfg
    e11run.TOPIC_CONFIG = cfg
    e11run.RETENTION_S = params.extra["retention_s"]
    try:
        e11run.run_with_ager(run_dir, params)
    finally:
        harness.TOPIC_CONFIG = saved_h
        e11run.TOPIC_CONFIG = saved_t
        e11run.RETENTION_S = saved_r


def main() -> None:
    t0 = time.time()
    summary = {}
    for cell, spec in CELLS.items():
        print(f"=== cell {cell} ===", flush=True)
        cal = spec.get("calibration")
        if cal is not None:
            cal_dir = DATA_DIR / "runs" / f"cal-{cell}"
            if not (cal_dir / "manifest.json").exists():
                print(f"  calibration run (seed {cal['seed']})...",
                      flush=True)
                do_run(cal_dir, cell_params(
                    cell, cal["seed"], 300.0, spec["retention_s"],
                    cal["overrides"], "calibration"))
        crossings = 0
        used = 0
        for seed in spec["seeds"]:
            if crossings >= TARGET_PER_CELL:
                break
            if used >= TARGET_PER_CELL + 2:
                break
            run_dir = DATA_DIR / "runs" / f"{cell}-s{seed}"
            if not (run_dir / "manifest.json").exists():
                print(f"  run {cell}-s{seed} "
                      f"({spec['t_end'] * 4 / 60:.0f} min)...", flush=True)
                do_run(run_dir, cell_params(
                    cell, seed, spec["t_end"], spec["retention_s"],
                    spec["overrides"], "eval"))
            used += 1
            c = crossings_in(run_dir, spec["retention_s"])
            crossings += c
            print(f"  {cell}-s{seed}: {c} crossing(s); cell total "
                  f"{crossings}", flush=True)
        summary[cell] = {"crossings": crossings, "runs_used": used,
                         "exploratory": crossings < TARGET_PER_CELL}
        print(f"  cell {cell}: {crossings} crossings over {used} runs"
              + (" [EXPLORATORY]" if crossings < TARGET_PER_CELL else ""),
              flush=True)
    (DATA_DIR / "campaign_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(f"campaign grid done in {(time.time() - t0) / 60:.0f} min; "
          f"summary saved", flush=True)


if __name__ == "__main__":
    main()
