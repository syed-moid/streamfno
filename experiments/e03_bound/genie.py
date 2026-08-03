"""e03 addendum: the genie (state-omniscient) floor gamma(h).

The two-point bound delta_min(h) is a *worst-case-state* statement: it
binds the error on its hypothesis pair, not the average over the dataset's
state distribution, so measured average predictor error can legitimately
sit below it where hard states are rare.  To make the "no predictor can
enter" comparison enforceable against measured test error, this addendum
computes a dataset-comparable floor: for each test decision state theta
(exactly reconstructed by replaying the episode's seeded trajectory), the
outcome probability p(theta) = P(E_h | theta) is estimated by seeded
Monte Carlo with fresh future randomness, and

    gamma(h) = mean over test decision states of min(p, 1 - p)

is the average error of the predictor that knows the hidden state exactly.
By data processing (observations are a noisy function of the state), no
telemetry predictor can do better in expectation; the telemetry channel
can only add to this floor.  Monte-Carlo standard errors are reported;
comparisons use CIs.

Decision states are taken at the particle filter's stride (wall-clock
reduction, recorded).  The future Monte Carlo runs on the data-generating
leap step: a relaxed-step variant was measured to inflate outcome
probabilities enough to push the floor above the realized base rate at the
light level (an upward bias a *floor* must not carry), and was reverted --
the investigation is recorded in docs/decisions.md.  Per-state episode ids
are saved so comparisons can be restricted to a predictor's episode
subset.
"""

import json
import time
from pathlib import Path

import numpy as np

from streamfno.bounds import ensemble_outcome_probs
from streamfno.events import EventConfig, decision_times
from streamfno.obs import Episode
from streamfno.sim import TauLeapSim

ROOT = Path(__file__).resolve().parents[2]
E02 = ROOT / "data" / "e02"
DATA_DIR = ROOT / "data" / "e03"

LEVELS = ("light", "moderate", "heavy")
DEC_STRIDE = 2      # match the particle filter's decision grid
N_REPS = 48
SEED_BASE = 700_000


def episode_states(ep: Episode, t_dec: np.ndarray):
    """Replay the episode's seeded trajectory and snapshot the exact hidden
    state (backlogs + modulator) at each requested decision time."""
    cfg = ep.sim_config
    sim = TauLeapSim(cfg)
    states = []
    k = 0
    t_grid = np.arange(1, int(round(cfg.t_end / cfg.dt_sample)) + 1)
    for step in t_grid:
        sim.advance(step * cfg.dt_sample - sim.t)
        t = step * cfg.dt_sample
        if k < len(t_dec) and abs(t - t_dec[k]) < 1e-9:
            states.append((sim.q.copy(), int(sim.mmpp_state[0])))
            k += 1
    if k != len(t_dec):
        raise RuntimeError("decision times not aligned with replay grid")
    # replay fidelity check against the recorded hidden trajectory
    kk = int(round(t_dec[-1] / cfg.dt_sample))
    m_replay = np.mean(states[-1][0] / cfg.buffer_depth)
    m_saved = float(ep.mean_lag_hidden[kk].mean())
    if abs(m_replay - m_saved) > 1e-9:
        raise RuntimeError(f"replay mismatch: {m_replay} vs {m_saved}")
    return states


def main():
    t0 = time.time()
    ecfg = EventConfig.load(E02 / "event_config.json")
    manifest = json.loads((E02 / "manifest.json").read_text())
    out = {"lead_times": np.array(ecfg.lead_times)}
    # resume: keep levels already computed in a previous (partial) run
    prior = DATA_DIR / "genie.npz"
    if prior.exists():
        with np.load(prior) as f:
            for k in f.files:
                out[k] = f[k]
    for level in LEVELS:
        if f"{level}_gamma" in out:
            print(f"  [{level}] already computed; skipping", flush=True)
            continue
        rows = [m for m in manifest
                if m["level"] == level and m["split"] == "test"]
        floors, ps, ses, ep_ids = [], [], [], []
        for r_i, m in enumerate(rows):
            ep = Episode.load(E02 / m["path"])
            t_dec = decision_times(ep, ecfg)[::DEC_STRIDE]
            states = episode_states(ep, t_dec)
            for s_i, (q, mod) in enumerate(states):
                seed = SEED_BASE + 1000 * m["sim_seed"] + s_i
                p, se = ensemble_outcome_probs(ep.sim_config, None, mod,
                                               ecfg, N_REPS, seed,
                                               q_exact=q)
                ps.append(p)
                ses.append(se)
                floors.append(np.minimum(p, 1.0 - p))
                ep_ids.append(m["index"])
            if (r_i + 1) % 4 == 0:
                print(f"  [{level}] {r_i + 1}/{len(rows)} episodes", flush=True)
        floors = np.array(floors)
        ses = np.array(ses)
        gamma = floors.mean(axis=0)
        # MC uncertainty of the mean floor (per-state MC se, averaged)
        gamma_se = np.sqrt((ses**2).mean(axis=0) / len(floors))
        out[f"{level}_gamma"] = gamma
        out[f"{level}_gamma_se"] = gamma_se
        out[f"{level}_floors"] = floors
        out[f"{level}_p"] = np.array(ps)
        out[f"{level}_episode_ids"] = np.array(ep_ids)
        print(f"  [{level}] gamma(h): " + " ".join(
            f"{h:g}:{g:.3f}" for h, g in zip(ecfg.lead_times, gamma)),
            flush=True)
        np.savez_compressed(DATA_DIR / "genie.npz", **out)  # incremental
    print(f"genie floor done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
