"""e03: bound instruments for the predictability horizon.

Three instruments (see src/streamfno/bounds/):
1. finite-time divergence rate lambda_+ (diagnostic) under the three frozen
   drift regimes (drain / mean / burst);
2. two-point numerical lower bound delta_min(h) -- primary.  Hypothesis
   pairs are full hidden states at the decision time: a lag density (from
   the reference trajectory, at a grid of mean lags) plus a shared-modulator
   state.  Histories for the KL are mean-field approach paths over the
   stated window; outcome probabilities are seeded simulator Monte Carlo
   (finite N, free modulator) -- events at long leads are genuinely
   stochastic, so the deterministic-outcome Le Cam form is recovered only
   as a special case.  Burst frequency enters through the outcome MC, so
   the bound is computed per dataset load level.
3. margin distributions of dataset decision states in the observation
   metric, per load level.
"""

import json
import time
from pathlib import Path

import numpy as np

from streamfno.bounds import (
    divergence_rate,
    ensemble_outcome_probs,
    kl_gaussian_obs,
    pde_clean_observables,
    two_point_bound,
)
from streamfno.events import EventConfig, label_episode
from streamfno.matching import drift_from_config
from streamfno.obs import Episode, ObsConfig
from streamfno.pde.solver import FPResult, solve_fp
from streamfno.sim import SimConfig

ROOT = Path(__file__).resolve().parents[2]
E02 = ROOT / "data" / "e02"
DATA_DIR = ROOT / "data" / "e03"

M_CELLS = 256
DT = 2e-3
T_HIST = 8.0            # observation-history window for the KL (stated)
BASE_MEANS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
GAPS = (0.05, 0.12, 0.25)
LEVEL_RATES = {"light": 0.03, "moderate": 0.06, "heavy": 0.10}
N_REPS = 64
N_REPS_MID = 256        # extra MC for p in the uncertain mid-range
H_MID = 8.0


def load_shared():
    ecfg = EventConfig.load(E02 / "event_config.json")
    ocfg = ObsConfig.from_json((E02 / "obs_config.json").read_text())
    manifest = json.loads((E02 / "manifest.json").read_text())
    ep0 = Episode.load(E02 / manifest[0]["path"])
    return ecfg, ocfg, manifest, ep0.sim_config


def frozen_drift(sim_cfg: SimConfig, lam: float):
    frozen = SimConfig(**{**sim_cfg.__dict__, "arrival": "poisson",
                          "lam": lam, "n_brokers": 1})
    return drift_from_config(frozen)


def divergence(sim_cfg: SimConfig):
    """lambda_+ under drain / mean-rate / burst drifts, several bumps."""
    p_mod = LEVEL_RATES["moderate"] / (LEVEL_RATES["moderate"]
                                       + sim_cfg.r_high_low)
    regimes = {
        "drain": sim_cfg.lam_low,
        "mean": sim_cfg.lam_low + p_mod * (sim_cfg.lam_high - sim_cfg.lam_low),
        "burst": sim_cfg.lam_high,
    }
    ref = FPResult.load(E02 / "pde_refs" / "moderate_high.npz")
    means = (ref.rho[:, 0, :] @ ref.x_centers) / M_CELLS
    base = ref.rho[np.argmin(np.abs(means - 0.5)), 0, :]
    out = {}
    for name, lam in regimes.items():
        recs = divergence_rate(
            base, frozen_drift(sim_cfg, lam), sim_cfg.a, t_end=16.0,
            bump_locs=(0.55, 0.70, 0.85), bump_mags=(0.01, 0.05), dt=DT,
            dt_sample=0.5, fit_window=(0.5, 8.0))
        out[name] = {"lam": lam,
                     "lam_plus": [r.lam_plus for r in recs],
                     "bumps": [(r.bump_loc, r.bump_mag) for r in recs]}
        print(f"  divergence[{name}] (lam={lam:.3f}): lambda_+ = "
              + " ".join(f"{r.lam_plus:+.3f}" for r in recs))
    return out


class SideLibrary:
    """Hypothesis sides: (target mean, modulator state) -> approach path
    (for the KL) and end density (for the outcome MC).  Cached."""

    def __init__(self, sim_cfg: SimConfig, ocfg: ObsConfig, ref: FPResult):
        self.sim_cfg = sim_cfg
        self.ocfg = ocfg
        self.ref_means = (ref.rho[:, 0, :] @ ref.x_centers) / M_CELLS
        self.ref = ref
        self.cache = {}

    def get(self, mean: float, state: int):
        key = (round(mean, 4), state)
        if key not in self.cache:
            k = int(np.argmin(np.abs(self.ref_means - mean)))
            rho0 = self.ref.rho[k, 0, :]
            lam = self.sim_cfg.lam_high if state == 1 else self.sim_cfg.lam_low
            fp = solve_fp(rho0, frozen_drift(self.sim_cfg, lam),
                          self.sim_cfg.a, t_end=T_HIST, dt=DT, dt_sample=0.5)
            _, y = pde_clean_observables(fp, self.ocfg,
                                         self.sim_cfg.n_brokers)
            self.cache[key] = (y, fp.rho[-1, 0, :])
        return self.cache[key]


_LEVEL_INDEX = {"light": 0, "moderate": 1, "heavy": 2}


def _side_seed(level: str, mean: float, state: int) -> int:
    """Deterministic MC seed from the side's identity (no hash())."""
    return (50_000 + _LEVEL_INDEX[level] * 100_000
            + int(round(mean * 1000)) * 4 + int(state) * 2)


def outcome_probs_cached(cache, sim_cfg_level, rho, state, ecfg, level, mean):
    key = (level, round(mean, 4), state)
    if key not in cache:
        seed = _side_seed(level, mean, state)
        p, se = ensemble_outcome_probs(sim_cfg_level, rho, state, ecfg,
                                       N_REPS, seed)
        if ((p > 0.03) & (p < 0.97)).any():
            p, se = ensemble_outcome_probs(sim_cfg_level, rho, state, ecfg,
                                           N_REPS_MID, seed + 1)
        cache[key] = (p, se)
    return cache[key]


def lecam(sim_cfg: SimConfig, ecfg: EventConfig, ocfg: ObsConfig):
    ref = FPResult.load(E02 / "pde_refs" / "moderate_high.npz")
    sides = SideLibrary(sim_cfg, ocfg, ref)

    # pair inventory: density gaps at each modulator state, plus
    # modulator-flip pairs at equal density
    pairs = []
    for m in BASE_MEANS:
        for state in (0, 1):
            for g in GAPS:
                if m + g <= 0.95:
                    pairs.append((f"gap m={m:.2f}+{g:.2f} s={state}",
                                  (m, state), (m + g, state)))
        pairs.append((f"flip m={m:.2f}", (m, 0), (m, 1)))

    results = {}
    n_h = len(ecfg.lead_times)
    for level, r_lh in LEVEL_RATES.items():
        cfg_level = SimConfig(**{**sim_cfg.__dict__, "r_low_high": r_lh})
        mc_cache = {}
        delta_best = np.zeros(n_h)
        best_label = [""] * n_h
        kl_list, labels = [], []
        for i_pair, (label, side_a, side_b) in enumerate(pairs):
            if i_pair % 10 == 0:
                print(f"    [{level}] pair {i_pair + 1}/{len(pairs)}",
                      flush=True)
            y_a, rho_a = sides.get(*side_a)
            y_b, rho_b = sides.get(*side_b)
            kl = kl_gaussian_obs(y_a, y_b, ocfg.noise_std)
            p_a, se_a = outcome_probs_cached(mc_cache, cfg_level, rho_a,
                                             side_a[1], ecfg, level,
                                             side_a[0])
            p_b, se_b = outcome_probs_cached(mc_cache, cfg_level, rho_b,
                                             side_b[1], ecfg, level,
                                             side_b[0])
            delta = two_point_bound(p_a, se_a, p_b, se_b, kl)
            kl_list.append(kl)
            labels.append(label)
            for j in range(n_h):
                if delta[j] > delta_best[j]:
                    delta_best[j] = delta[j]
                    best_label[j] = label
        results[level] = {"delta_min": delta_best, "best_pair": best_label,
                          "pair_kl": np.array(kl_list),
                          "pair_labels": labels}
        print(f"  [{level}] delta_min(h): "
              + " ".join(f"{h:g}:{d:.3f}" for h, d in
                         zip(ecfg.lead_times, delta_best)))
        for j, h in enumerate(ecfg.lead_times):
            print(f"      h={h:5.1f}: best pair {best_label[j]}")
    return results


def margins(manifest, ecfg: EventConfig, ocfg: ObsConfig):
    """Whitened distance to the nearest opposite-label decision state,
    pooled over test episodes, per level, at the middle lead time."""
    from streamfno.bounds import nearest_opposite_margins
    j_mid = list(ecfg.lead_times).index(H_MID)
    out = {}
    for level in LEVEL_RATES:
        ys, labs = [], []
        for m in manifest:
            if m["level"] != level or m["split"] != "test":
                continue
            ep = Episode.load(E02 / m["path"])
            t_dec, lab = label_episode(ep, ecfg)
            sel = np.isin(ep.times, t_dec)
            ys.append(ep.y_clean[sel])
            labs.append(lab[:, j_mid])
        y = np.concatenate(ys)
        lab = np.concatenate(labs)
        marg = nearest_opposite_margins(y, lab, ocfg.noise_std)
        ok = np.isfinite(marg)
        out[level] = {
            "margins": marg, "labels": lab, "base_rate": float(lab.mean()),
            "frac_informative": float((marg[ok] < 2.0).mean()) if ok.any()
            else float("nan"),
            "median": float(np.nanmedian(marg)),
        }
        print(f"  margins[{level}]: base rate {lab.mean():.3f}, median "
              f"{out[level]['median']:.2f}, frac in informative zone (<2) "
              f"{out[level]['frac_informative']:.3f}")
    return out


def main():
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ecfg, ocfg, manifest, sim_cfg = load_shared()

    print("divergence rates...")
    div = divergence(sim_cfg)

    print("two-point bound (per level)...")
    lc = lecam(sim_cfg, ecfg, ocfg)

    print("margins...")
    mg = margins(manifest, ecfg, ocfg)

    # sanity: delta_min(h) nondecreasing in the mean over adjacent steps;
    # E_h is monotone in h so predictability can genuinely return at very
    # long leads only through the event becoming near-certain
    sanity = {}
    for level, r in lc.items():
        d = r["delta_min"]
        diffs = np.diff(d)
        sanity[level] = {
            "monotone_nondecreasing": bool((diffs >= -1e-9).all()),
            "mean_adjacent_diff": float(diffs.mean()),
            "vacuous_everywhere": bool((d <= 1e-6).all()),
        }
    print("sanity:", json.dumps(sanity, indent=1))

    save = {"lead_times": np.array(ecfg.lead_times)}
    for level, r in lc.items():
        save[f"{level}_delta_min"] = r["delta_min"]
        save[f"{level}_pair_kl"] = r["pair_kl"]
        save[f"{level}_best_pair"] = np.array(r["best_pair"])
        save[f"{level}_pair_labels"] = np.array(r["pair_labels"])
    np.savez_compressed(DATA_DIR / "lecam.npz", **save)
    np.savez_compressed(
        DATA_DIR / "margins.npz",
        **{f"{lvl}_{k}": np.asarray(v) for lvl, dd in mg.items()
           for k, v in dd.items()})
    (DATA_DIR / "divergence.json").write_text(json.dumps(div, indent=1))
    (DATA_DIR / "sanity.json").write_text(json.dumps(sanity, indent=1))
    print(f"e03 done in {time.time() - t0:.1f}s; results in {DATA_DIR}")


if __name__ == "__main__":
    main()
