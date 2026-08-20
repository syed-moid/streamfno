"""e09 analysis: render the computational-profile table from saved
results (data/e09/profile.json) -- printed and written as markdown to
data/e09/profile_table.md for the manuscript's section 5 table."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "e09"


def fmt_ms(v: float) -> str:
    return f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def main() -> None:
    p = json.loads((DATA_DIR / "profile.json").read_text())
    lines = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    m = p["machine"]
    emit(f"Machine: {m['platform']} ({m['machine']}, "
         f"{m['cpu_count']} CPUs); numpy {m['numpy']}, scipy {m['scipy']}; "
         f"single-threaded.")
    emit()
    emit("| horizon h (units) | " + " | ".join(
        f"{float(h):g}" for h in p["config"]["horizons"]) + " |")
    emit("|---|" + "---|" * len(p["config"]["horizons"]))
    for tag, label in (("single_class", "1-class wall ms"),
                       ("per_broker_class",
                        f"{p['config']['n_broker_classes']}-class wall ms")):
        for q in ("p50", "p95", "p99"):
            row = [fmt_ms(p[tag][f"{float(h):g}"]["wall_ms"][q])
                   for h in p["config"]["horizons"]]
            emit(f"| {label} {q} | " + " | ".join(row) + " |")
    emit()
    sc = p["single_class"][f"{float(p['config']['horizons'][-1]):g}"]
    emit("Stage medians at the longest horizon (1 class): "
         f"rho0 {sc['stage_median_ms']['rho0']:.3f} ms, FP solve "
         f"{fmt_ms(sc['stage_median_ms']['fp_solve'])} ms, threshold "
         f"{sc['stage_median_ms']['threshold']:.3f} ms.")
    est = p["estimation"]
    emit(f"Coefficient estimation (recalibration): "
         f"{est['wall_s_median'] * 1e3:.0f} ms wall / "
         f"{est['cpu_s_median'] * 1e3:.0f} ms CPU, amortized across "
         "forecasts between recalibrations.")
    emit()
    emit(f"| ensemble M @ h={p['config']['h_ensemble']:g} | " + " | ".join(
        str(ms) for ms in p["config"]["ensemble_m"]) + " |")
    emit("|---|" + "---|" * len(p["config"]["ensemble_m"]))
    emit("| wall ms (median) | " + " | ".join(
        fmt_ms(p["ensemble"][str(ms)]["wall_ms_median"])
        for ms in p["config"]["ensemble_m"]) + " |")
    emit("| members / s | " + " | ".join(
        f"{p['ensemble'][str(ms)]['members_per_second']:.1f}"
        for ms in p["config"]["ensemble_m"]) + " |")
    emit()
    mem = p["memory"]
    emit(f"Peak traced memory per forecast (h=16): "
         f"{mem['forecast_h16_c1_mib']:.2f} MiB (1 class), "
         f"{mem['forecast_h16_c3_mib']:.2f} MiB (3 classes); "
         f"process RSS {mem['process_ru_maxrss_mib']:.0f} MiB.")
    fr = p["fixed_row"]
    emit(f"External service calls: {fr['external_service_calls']}. "
         f"Token cost: {fr['token_cost']}. "
         f"Network dependencies: {fr['network_dependencies']}.")

    (DATA_DIR / "profile_table.md").write_text("\n".join(lines) + "\n")
    print(f"\nwritten {DATA_DIR / 'profile_table.md'}")


if __name__ == "__main__":
    main()
