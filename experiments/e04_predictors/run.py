"""e04: predictor suite vs the telemetry channel.

Four predictors, all consuming the C1 observation process only:
particle filter (well-specified Bayes-filter reference), logistic
regression and gradient-boosted trees on windowed features, and the
reactive threshold heuristic.  Operating points tuned on validation
episodes; metrics reported on test episodes with episode-bootstrap CIs.

Wall-clock control (documented in docs/experiment-log.md): the particle
filter is tuned on a subset of validation episodes (PF_VAL_EPISODES per
level) -- a compute reduction, not a split violation; its ensemble size is
convergence-checked by doubling until the Brier score stabilizes.
"""

import json
import time
from pathlib import Path

import numpy as np

from streamfno.events import EventConfig, decision_times, label_episode
from streamfno.obs import Episode, ObsConfig
from streamfno.predictors import (
    ParticleFilter,
    calibration_curve_data,
    make_gbt,
    make_logistic,
    metrics_with_ci,
    reactive_scores,
    tune_threshold,
    window_features,
)

ROOT = Path(__file__).resolve().parents[2]
E02 = ROOT / "data" / "e02"
DATA_DIR = ROOT / "data" / "e04"

LEVELS = ("light", "moderate", "heavy")
WINDOW = 8
# Wall-clock reductions for the particle filter ONLY (recorded in the log;
# splits remain honest -- subsets are prefixes of the val/test splits):
PF_VAL_EPISODES = 8       # PF threshold tuning subset (val split only)
PF_TEST_EPISODES = 16     # PF evaluation subset of the test split
PF_DEC_STRIDE = 2         # PF evaluates every 2nd decision time
PF_SEED = 900
PF_SIZES = (32, 64, 128)  # convergence ladder
PF_CONV_EPISODES = 4
PF_CONV_TOL = 0.015       # Brier stability tolerance


def load_split(manifest, level, split):
    eps = [Episode.load(E02 / m["path"]) for m in manifest
           if m["level"] == level and m["split"] == split]
    idx = [m["index"] for m in manifest
           if m["level"] == level and m["split"] == split]
    return eps, idx


def featurize(episodes, indices, ecfg):
    xs, labs, eids, t_decs = [], [], [], []
    for ep, i in zip(episodes, indices):
        t_dec = decision_times(ep, ecfg)
        _, lab = label_episode(ep, ecfg, t_dec)
        xs.append(window_features(ep, t_dec, WINDOW))
        labs.append(lab)
        eids.append(np.full(len(t_dec), i))
        t_decs.append(t_dec)
    return (np.concatenate(xs), np.concatenate(labs),
            np.concatenate(eids), t_decs)


def heuristic_scores(episodes, ecfg, kind):
    out = []
    for ep in episodes:
        t_dec = decision_times(ep, ecfg)
        out.append(reactive_scores(ep, t_dec, kind))
    return np.concatenate(out)


def pf_decision_times(ep, ecfg):
    return decision_times(ep, ecfg)[::PF_DEC_STRIDE]


def pf_labels(episodes, ecfg):
    labs = []
    for ep in episodes:
        t_dec = pf_decision_times(ep, ecfg)
        labs.append(label_episode(ep, ecfg, t_dec)[1])
    return np.concatenate(labs)


def pf_episode_ids(episodes, indices, ecfg):
    return np.concatenate([
        np.full(len(pf_decision_times(ep, ecfg)), i)
        for ep, i in zip(episodes, indices)])


def pf_run(episodes, indices, level, ecfg, ocfg, n_particles, store):
    """Run the particle filter over episodes, caching per-episode results."""
    probs, ess_mins = [], []
    for ep, i in zip(episodes, indices):
        key = f"{level}_{i}_M{n_particles}"
        if key not in store:
            pf = ParticleFilter(ep.sim_config, ep.obs_config, n_particles,
                                seed=PF_SEED + i)
            res = pf.run_episode(ep, pf_decision_times(ep, ecfg), ecfg)
            store[key] = (res.probs, res.ess)
        p, ess = store[key]
        probs.append(p)
        ess_mins.append(float(np.min(ess)))
    return np.concatenate(probs), np.array(ess_mins)


def pf_convergence(manifest, ecfg, ocfg, store):
    """Double the ensemble until the Brier score at the middle lead time
    stabilizes on a fixed subset of moderate validation episodes."""
    eps, idx = load_split(manifest, "moderate", "val")
    eps, idx = eps[:PF_CONV_EPISODES], idx[:PF_CONV_EPISODES]
    j_mid = list(ecfg.lead_times).index(8.0)
    labels = pf_labels(eps, ecfg)[:, j_mid]
    briers = {}
    chosen = PF_SIZES[-1]
    prev = None
    for m in PF_SIZES:
        probs, _ = pf_run(eps, idx, "moderate", ecfg, ocfg, m, store)
        briers[m] = float(np.mean((probs[:, j_mid] - labels) ** 2))
        print(f"  PF convergence: M={m} Brier(h=8)={briers[m]:.4f}")
        if prev is not None and abs(briers[m] - prev) < PF_CONV_TOL:
            chosen = m
            break
        prev = briers[m]
    print(f"  chosen ensemble size M={chosen}")
    return chosen, briers


def main():
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ecfg = EventConfig.load(E02 / "event_config.json")
    ocfg = ObsConfig.from_json((E02 / "obs_config.json").read_text())
    manifest = json.loads((E02 / "manifest.json").read_text())
    lead = list(ecfg.lead_times)

    pf_store = {}
    print("particle-filter ensemble convergence check...")
    pf_m, pf_conv = pf_convergence(manifest, ecfg, ocfg, pf_store)

    results = {"lead_times": lead, "pf_ensemble": pf_m,
               "pf_convergence": pf_conv, "levels": {}}
    calib = {}
    pf_probs_store = {}

    for level in LEVELS:
        print(f"[{level}]")
        tr_eps, tr_idx = load_split(manifest, level, "train")
        va_eps, va_idx = load_split(manifest, level, "val")
        te_eps, te_idx = load_split(manifest, level, "test")
        x_tr, y_tr, _, _ = featurize(tr_eps, tr_idx, ecfg)
        x_va, y_va, id_va, _ = featurize(va_eps, va_idx, ecfg)
        x_te, y_te, id_te, _ = featurize(te_eps, te_idx, ecfg)

        print("  particle filter (val subset + test subset)...")
        pf_va, _ = pf_run(va_eps[:PF_VAL_EPISODES], va_idx[:PF_VAL_EPISODES],
                          level, ecfg, ocfg, pf_m, pf_store)
        y_va_pf = pf_labels(va_eps[:PF_VAL_EPISODES], ecfg)
        pf_te_eps = te_eps[:PF_TEST_EPISODES]
        pf_te_idx = te_idx[:PF_TEST_EPISODES]
        pf_te, ess_te = pf_run(pf_te_eps, pf_te_idx, level, ecfg, ocfg, pf_m,
                               pf_store)
        y_te_pf = pf_labels(pf_te_eps, ecfg)
        id_te_pf = pf_episode_ids(pf_te_eps, pf_te_idx, ecfg)
        for i in pf_te_idx:
            pf_probs_store[f"{level}_test_{i}"] = pf_store[
                f"{level}_{i}_M{pf_m}"][0]

        level_res = {"ess_min_per_test_episode": ess_te.tolist()}
        for j, h in enumerate(lead):
            row = {}
            # learned feature models
            for name, factory in (("logistic", make_logistic),
                                  ("gbt", make_gbt)):
                model = factory()
                model.fit(x_tr, y_tr[:, j])
                s_va = model.predict_proba(x_va)[:, 1]
                s_te = model.predict_proba(x_te)[:, 1]
                thr = tune_threshold(s_va, y_va[:, j])
                row[name] = metrics_with_ci(s_te, y_te[:, j], id_te, thr)
                if name == "gbt" and h == 8.0:
                    calib[f"{level}_gbt"] = calibration_curve_data(
                        s_te, y_te[:, j])
            # reactive heuristic: kind chosen on val
            best = None
            for kind in ("flux", "wall"):
                s_va = heuristic_scores(va_eps, ecfg, kind)
                thr = tune_threshold(s_va, y_va[:, j])
                err = float(((s_va > thr) != y_va[:, j]).mean())
                if best is None or err < best[0]:
                    best = (err, kind, thr)
            _, kind, thr = best
            s_te = heuristic_scores(te_eps, ecfg, kind)
            row["reactive"] = metrics_with_ci(s_te, y_te[:, j], id_te, thr)
            row["reactive"]["kind"] = kind
            # particle filter (evaluated on its recorded subset)
            thr_pf = tune_threshold(pf_va[:, j], y_va_pf[:, j])
            row["pf"] = metrics_with_ci(pf_te[:, j], y_te_pf[:, j], id_te_pf,
                                        thr_pf)
            if h == 8.0:
                calib[f"{level}_pf"] = calibration_curve_data(pf_te[:, j],
                                                              y_te_pf[:, j])
            level_res[str(h)] = row
            print(f"  h={h:5.1f}: " + " ".join(
                f"{k}:{row[k]['error']:.3f}" for k in
                ("pf", "gbt", "logistic", "reactive")))
        results["levels"][level] = level_res

    (DATA_DIR / "results.json").write_text(json.dumps(results, indent=1))
    (DATA_DIR / "calibration.json").write_text(json.dumps(calib, indent=1))
    np.savez_compressed(DATA_DIR / "pf_probs.npz", **pf_probs_store)
    print(f"e04 done in {time.time() - t0:.1f}s; results in {DATA_DIR}")


if __name__ == "__main__":
    main()
