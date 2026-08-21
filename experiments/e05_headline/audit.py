"""B1/B2 audit (revision brief): genie-floor specification and the
identical-bootstrap comparison, plus the skill horizons the theory
defines.

B1 -- the genie estimator, documented and audited: p_hat(theta) is the
mean of M = 48 independent seeded future continuations per replayed
hidden test state (e03/genie.py); gamma_hat(h) = mean of
min(p_hat, 1 - p_hat) over states.  The reported light-load violation
(gamma(24) > err_const(24)) is diagnosed by recomputing gamma and the
base-rate error on the IDENTICAL episode-cluster bootstrap: gamma is an
expectation over fresh futures, err_const is a single label draw, and
the two are compared with CIs; the display floor becomes
min(gamma, err_const) wherever the gap survives.  All genie-side label
quantities use the genie's stride-2 decision grid; predictor-side
quantities use the full e04 grid (stated in the output).

B2 -- skill and its horizon: S(h) = err_const(h) - err_best(h) with
err_best the best of {logistic, gbt, reactive} at val-tuned operating
points (the measured estimation gap ~ 0 makes this the err* proxy; the
PF adds nothing and is degenerate at heavy load), and the state-side
skill uses the audited floor.  H*_delta per level for delta in
{0.05, 0.1, 0.2} with episode-cluster bootstrap CIs on the lead grid
{1,2,4,8,16,24}; right-censoring at the grid edge is reported as such.
Also emits the constant-predictor row for Table I.

Saves data/e05/audit.json.
"""

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E02 = ROOT / "data" / "e02"
E03 = ROOT / "data" / "e03"
OUT = ROOT / "data" / "e05"

_spec = importlib.util.spec_from_file_location(
    "e04_run", ROOT / "experiments" / "e04_predictors" / "run.py")
e04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e04)

from streamfno.events import EventConfig, decision_times, label_episode  # noqa: E402
from streamfno.obs import Episode  # noqa: E402
from streamfno.predictors import (  # noqa: E402
    make_gbt,
    make_logistic,
    reactive_scores,
    tune_threshold,
)

LEVELS = ("light", "moderate", "heavy")
GENIE_STRIDE = 2          # e03/genie.py DEC_STRIDE
GENIE_M = 48              # e03/genie.py N_REPS
N_BOOT = 4000
BOOT_SEED = 2100
DELTAS = (0.025, 0.05, 0.10, 0.20)
DELTAS_REL = (0.25, 0.5, 0.75)
ERR_GATE = 0.2            # Table I's operating criterion


def stride2_labels(manifest, level, ecfg):
    """Labels at the genie's stride-2 decision grid, with episode ids,
    in the genie's episode order (manifest test order)."""
    labs, eids = [], []
    for m in manifest:
        if m["level"] != level or m["split"] != "test":
            continue
        ep = Episode.load(E02 / m["path"])
        t_dec = decision_times(ep, ecfg)[::GENIE_STRIDE]
        _, lab = label_episode(ep, ecfg, t_dec)
        labs.append(lab)
        eids.append(np.full(lab.shape[0], m["index"]))
    return np.concatenate(labs), np.concatenate(eids)


def predictor_errors(manifest, level, ecfg):
    """Per-state wrongness vectors for logistic/gbt/reactive on the full
    e04 decision grid, at val-tuned thresholds (e04 protocol)."""
    tr_eps, tr_idx = e04.load_split(manifest, level, "train")
    va_eps, va_idx = e04.load_split(manifest, level, "val")
    te_eps, te_idx = e04.load_split(manifest, level, "test")
    x_tr, y_tr, _, _ = e04.featurize(tr_eps, tr_idx, ecfg)
    x_va, y_va, _, _ = e04.featurize(va_eps, va_idx, ecfg)
    x_te, y_te, id_te, _ = e04.featurize(te_eps, te_idx, ecfg)
    wrong = {}
    for name, factory in (("logistic", make_logistic), ("gbt", make_gbt)):
        wrong[name] = np.empty_like(y_te, dtype=bool)
        for j in range(y_te.shape[1]):
            model = factory()
            model.fit(x_tr, y_tr[:, j])
            thr = tune_threshold(model.predict_proba(x_va)[:, 1], y_va[:, j])
            pred = model.predict_proba(x_te)[:, 1] > thr
            wrong[name][:, j] = pred != (y_te[:, j] > 0)
    for kind in ("flux", "wall"):
        s_va = np.concatenate([reactive_scores(ep, decision_times(ep, ecfg),
                                               kind) for ep in va_eps])
        s_te = np.concatenate([reactive_scores(ep, decision_times(ep, ecfg),
                                               kind) for ep in te_eps])
        w = np.empty_like(y_te, dtype=bool)
        for j in range(y_te.shape[1]):
            thr = tune_threshold(s_va, y_va[:, j])
            w[:, j] = (s_te > thr) != (y_te[:, j] > 0)
        wrong[f"reactive_{kind}"] = w
    return wrong, y_te, id_te


def boot_mean(values, eids, uniq, rng, n_boot):
    """Episode-cluster bootstrap of the mean of ``values`` (n_states,
    n_leads): resample episodes with replacement."""
    groups = {e: values[eids == e] for e in uniq}
    out = np.empty((n_boot, values.shape[1]))
    for b in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        out[b] = np.concatenate([groups[e] for e in take]).mean(axis=0)
    return out


def ci(arr, axis=0):
    lo, hi = np.percentile(arr, [2.5, 97.5], axis=axis)
    return lo, hi


def main() -> None:
    t0 = time.time()
    ecfg = EventConfig.load(E02 / "event_config.json")
    manifest = json.loads((E02 / "manifest.json").read_text())
    lead = np.array(ecfg.lead_times)
    genie = np.load(E03 / "genie.npz")
    report = {"genie_estimator": {
        "M": GENIE_M, "decision_stride": GENIE_STRIDE,
        "definition": "p_hat = mean of M seeded future continuations per "
                      "replayed hidden state; gamma = mean min(p_hat, "
                      "1-p_hat) over test decision states",
        "note": "genie-vs-base-rate comparisons on the stride-2 grid; "
                "predictor-side quantities on the full e04 grid"},
        "lead_times": lead.tolist(), "n_boot": N_BOOT, "levels": {}}

    for level in LEVELS:
        print(f"[{level}]")
        p = genie[f"{level}_p"]                     # (n_states, n_leads)
        g_eids = genie[f"{level}_episode_ids"]
        floors = np.minimum(p, 1.0 - p)
        labs2, l_eids = stride2_labels(manifest, level, ecfg)
        assert labs2.shape[0] == p.shape[0], (labs2.shape, p.shape)
        assert np.array_equal(np.sort(np.unique(g_eids)),
                              np.sort(np.unique(l_eids)))
        uniq = np.unique(g_eids)

        # identical-bootstrap: gamma, base rate, their difference
        gam_b = boot_mean(floors, g_eids, uniq, np.random.default_rng(
            BOOT_SEED), N_BOOT)
        pi_b = boot_mean(labs2.astype(float), l_eids, uniq,
                         np.random.default_rng(BOOT_SEED), N_BOOT)
        errc_b = np.minimum(pi_b, 1.0 - pi_b)
        diff_b = gam_b - errc_b
        gamma = floors.mean(axis=0)
        pi2 = labs2.mean(axis=0)
        err_const2 = np.minimum(pi2, 1.0 - pi2)
        pi_genie = p.mean(axis=0)

        # predictors (full grid)
        wrong, y_te, id_te = predictor_errors(manifest, level, ecfg)
        pi_full = (y_te > 0).mean(axis=0)
        errc_full = np.minimum(pi_full, 1.0 - pi_full)
        best_name = []
        best_wrong = np.empty_like(y_te, dtype=bool)
        for j in range(y_te.shape[1]):
            errs = {k: v[:, j].mean() for k, v in wrong.items()}
            k_best = min(errs, key=errs.get)
            best_name.append(k_best)
            best_wrong[:, j] = wrong[k_best][:, j]
        rngb = np.random.default_rng(BOOT_SEED + 1)
        uniq_f = np.unique(id_te)
        bw_b = boot_mean(best_wrong.astype(float), id_te, uniq_f, rngb,
                         N_BOOT)
        pi_f_b = boot_mean((y_te > 0).astype(float), id_te, uniq_f,
                           np.random.default_rng(BOOT_SEED + 1), N_BOOT)
        errc_f_b = np.minimum(pi_f_b, 1.0 - pi_f_b)
        skill_b = errc_f_b - bw_b

        # skill horizons on the lead grid, with right-censoring flags;
        # normalized skill S_rel = S / min(pi, 1-pi) in [0,1] (Brier-
        # skill-score analogue) separates rarity from information content
        srel_b = skill_b / np.maximum(errc_f_b, 1e-12)
        srel_point = ((errc_full - best_wrong.mean(axis=0))
                      / np.maximum(errc_full, 1e-12))

        def horizon(vals_b, point_vals, d):
            hits = vals_b >= d
            hb = np.where(hits.any(axis=1),
                          lead[np.maximum.reduce(
                              np.where(hits, np.arange(lead.size), -1),
                              axis=1)], 0.0)
            ph = point_vals >= d
            point = (float(lead[np.flatnonzero(ph)[-1]]) if ph.any()
                     else 0.0)
            lo, hi = np.percentile(hb, [2.5, 97.5])
            return {"point": point, "ci": [float(lo), float(hi)],
                    "right_censored": bool(point >= lead[-1])}

        skill_point = errc_full - best_wrong.mean(axis=0)
        h_star = {f"{d:g}": horizon(skill_b, skill_point, d)
                  for d in DELTAS}
        h_star_rel = {f"{d:g}": horizon(srel_b, srel_point, d)
                      for d in DELTAS_REL}

        const_pass = errc_full < ERR_GATE
        const_row = (float(lead[np.flatnonzero(const_pass)[-1]])
                     if const_pass.any() else 0.0)

        lv = {"n_states_stride2": int(p.shape[0]),
              "n_episodes": int(len(uniq)),
              "per_lead": {}, "h_star": h_star, "h_star_rel": h_star_rel,
              "const_predictor_largest_h_below_0.2": const_row,
              "best_predictor_by_lead": best_name}
        for j, h in enumerate(lead):
            glo, ghi = np.percentile(gam_b[:, j], [2.5, 97.5])
            elo, ehi = np.percentile(errc_b[:, j], [2.5, 97.5])
            dlo, dhi = np.percentile(diff_b[:, j], [2.5, 97.5])
            lv["per_lead"][f"{h:g}"] = {
                "gamma": float(gamma[j]), "gamma_ci": [float(glo), float(ghi)],
                "gamma_mc_se": float(genie[f"{level}_gamma_se"][j]),
                "pi_labels_stride2": float(pi2[j]),
                "err_const_stride2": float(err_const2[j]),
                "err_const_ci": [float(elo), float(ehi)],
                "pi_genie": float(pi_genie[j]),
                "gamma_minus_err_const": float(gamma[j] - err_const2[j]),
                "diff_ci": [float(dlo), float(dhi)],
                "floor_display": float(min(gamma[j], err_const2[j])),
                "pi_full_grid": float(pi_full[j]),
                "err_const_full_grid": float(errc_full[j]),
                "err_best": float(best_wrong[:, j].mean()),
                "err_best_ci": [float(v) for v in
                                np.percentile(bw_b[:, j], [2.5, 97.5])],
                "skill": float(errc_full[j] - best_wrong[:, j].mean()),
                "skill_ci": [float(v) for v in
                             np.percentile(skill_b[:, j], [2.5, 97.5])],
                "skill_rel": float(srel_point[j]),
                "skill_rel_ci": [float(v) for v in
                                 np.percentile(srel_b[:, j], [2.5, 97.5])],
            }
            print(f"  h={h:>4g}: gamma {gamma[j]:.3f} "
                  f"[{glo:.3f},{ghi:.3f}]  err_const {err_const2[j]:.3f} "
                  f"[{elo:.3f},{ehi:.3f}]  diff [{dlo:+.3f},{dhi:+.3f}]  "
                  f"pi_genie {pi_genie[j]:.3f}  skill "
                  f"{lv['per_lead'][f'{h:g}']['skill']:.3f}")
        print("  H*_rel: " + "  ".join(
            f"d={d}: {h_star_rel[f'{d:g}']['point']:g} "
            f"{h_star_rel[f'{d:g}']['ci']}" for d in DELTAS_REL))
        print("  H*: " + "  ".join(
            f"d={d}: {h_star[f'{d:g}']['point']:g} "
            f"{h_star[f'{d:g}']['ci']}"
            + (" (right-censored)" if h_star[f"{d:g}"]["right_censored"]
               else "") for d in DELTAS))
        print(f"  const-predictor row (err<0.2): {const_row:g}")
        report["levels"][level] = lv

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(report, indent=1))
    print(f"saved {OUT / 'audit.json'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
