"""Bootstrap particle filter on the simulator's own model.

The well-specified Bayes-filter reference ("ceiling" candidate): particles
are full hidden states (all partition backlogs plus the shared modulator
state) propagated by the *same* tau-leap engine that generated the data,
with the same parameters.  The information set is exactly the C1
observation process: the filter sees episode.y (noisy) and nothing else.

Vectorization: an ensemble of M particles, each an N-partition system, is
packed into one TauLeapSim of M*N partitions.  Particle p's partitions get
class ids p*C + (i mod C), so the mean-field service coupling acts within
each particle exactly as in a single simulation, and per-particle
observables and rejection counts fall out of per-class reductions.  Each
particle carries its own shared-modulator group.

Weights: Gaussian likelihood of the observed vector around each particle's
clean observables with the channel's R = r^2 I.  Systematic resampling when
the effective sample size drops below M/2; the ESS series is recorded
(degeneracy near saturation is a known failure mode to report).

Event probability: at a decision time, the posterior ensemble is cloned and
rolled forward h_max with fresh randomness; P(E_h | obs) is the weighted
fraction of particles whose future smoothed flux sup over (t, t+h] exceeds
the threshold.  The trailing smoothing window at the start of the rollout
is seeded with the particle's own filtered flux history (an O(w)
approximation noted here for honesty).  Rollouts run on a leap step
relaxed by ``rollout_relax`` (wall-clock control): the tau-leap bias of the
predicted probabilities is O(tau) and is accepted and recorded rather than
hidden -- the filtering pass itself always uses the data-generating step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..events.events import EventConfig
from ..obs.observe import Episode, ObsConfig
from ..sim.config import SimConfig
from ..sim.engine import TauLeapSim

__all__ = ["ParticleFilter", "PFResult"]


@dataclass
class PFResult:
    """Per-episode filter output."""

    t_dec: np.ndarray
    probs: np.ndarray       # (n_dec, n_lead) P(E_h | observations)
    ess: np.ndarray         # ESS after each observation update
    ess_times: np.ndarray


class ParticleFilter:
    def __init__(self, sim_cfg: SimConfig, obs_cfg: ObsConfig,
                 n_particles: int, seed: int, rollout_relax: float = 4.0):
        if sim_cfg.arrival == "mmpp" and not sim_cfg.mmpp_shared:
            raise ValueError("per-partition modulators would need per-"
                             "partition groups; dataset uses shared bursts")
        self.sim_cfg = sim_cfg
        self.obs_cfg = obs_cfg
        self.m = n_particles
        self.n = sim_cfg.n_partitions
        self.c = sim_cfg.n_brokers
        self.seed = seed
        self.rollout_cfg = SimConfig(**{
            **sim_cfg.__dict__,
            "tau_jump_cap": sim_cfg.tau_jump_cap * rollout_relax,
            "tau_dt_max": sim_cfg.tau_dt_max * rollout_relax,
        })

    # -- vectorized ensemble construction ---------------------------------

    def _fresh_ensemble(self) -> TauLeapSim:
        m, n, c = self.m, self.n, self.c
        big = SimConfig(**{**self.sim_cfg.__dict__,
                           "n_partitions": m * n, "seed": self.seed})
        part = np.repeat(np.arange(m, dtype=np.int64), n)
        within = np.tile(np.arange(n, dtype=np.int64) % c, m)
        classes = part * c + within
        return TauLeapSim(big, classes=classes, n_classes=m * c,
                          mmpp_groups=part)

    def _particle_observables(self, sim: TauLeapSim,
                              rejected: np.ndarray) -> np.ndarray:
        """Clean H per particle, shape (M, 3C + hist_bins); mirrors the
        C1 observable layout."""
        m, n, c = self.m, self.n, self.c
        cfg = self.sim_cfg
        ocfg = self.obs_cfg
        x = sim.q / cfg.buffer_depth
        sizes = sim.sizes.astype(float)
        mean = (np.bincount(sim.classes, weights=x, minlength=m * c)
                / sizes).reshape(m, c)
        m2 = (np.bincount(sim.classes, weights=x * x, minlength=m * c)
              / sizes).reshape(m, c)
        var = np.maximum(m2 - mean**2, 0.0)
        flux = (rejected / (sizes * cfg.buffer_depth * ocfg.dt_obs)
                ).reshape(m, c)
        part = np.repeat(np.arange(m, dtype=np.int64), n)
        bins = np.minimum(sim.q * ocfg.hist_bins // cfg.buffer_depth,
                          ocfg.hist_bins - 1)
        hist = np.bincount(part * ocfg.hist_bins + bins,
                           minlength=m * ocfg.hist_bins
                           ).reshape(m, ocfg.hist_bins) / n
        return np.concatenate([mean, var, flux, hist], axis=1)

    def _particle_flux(self, sim: TauLeapSim, rejected: np.ndarray,
                       dt: float) -> np.ndarray:
        """Aggregate per-particle flux rate over the last interval."""
        cfg = self.sim_cfg
        return rejected.reshape(self.m, self.c).sum(axis=1) / (
            self.n * cfg.buffer_depth * dt)

    @staticmethod
    def _systematic_resample(w: np.ndarray, rng: np.random.Generator
                             ) -> np.ndarray:
        m = len(w)
        u = (rng.random() + np.arange(m)) / m
        return np.searchsorted(np.cumsum(w), u).clip(0, m - 1)

    def _resample(self, sim: TauLeapSim, w: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
        idx = self._systematic_resample(w, rng)
        q = sim.q.reshape(self.m, self.n)[idx].ravel().copy()
        sim.q = q
        if sim.mmpp_state is not None:
            sim.mmpp_state = sim.mmpp_state[idx].copy()
        return np.full(self.m, 1.0 / self.m)

    # -- rollout -----------------------------------------------------------

    def _rollout_events(self, sim: TauLeapSim, w: np.ndarray,
                        recent_flux: list[np.ndarray],
                        ecfg: EventConfig) -> np.ndarray:
        """P(E_h) for every lead time from a cloned posterior ensemble."""
        clone = sim.clone()
        clone.cfg = self.rollout_cfg
        dt = self.sim_cfg.dt_sample
        k_smooth = max(1, round(ecfg.flux_window / dt))
        h_max = max(ecfg.lead_times)
        n_steps = round(h_max / dt)
        # seed the trailing smoother with the filtered flux history
        hist = list(recent_flux[-k_smooth:]) or [np.zeros(self.m)]
        sup = {h: np.zeros(self.m) for h in ecfg.lead_times}
        for step in range(1, n_steps + 1):
            rej = clone.advance(dt)
            hist.append(self._particle_flux(clone, rej, dt))
            jbar = np.mean(hist[-k_smooth:], axis=0)
            t_ahead = step * dt
            for h in ecfg.lead_times:
                if t_ahead <= h + 1e-9:
                    sup[h] = np.maximum(sup[h], jbar)
        return np.array([(w * (sup[h] > ecfg.threshold)).sum()
                         for h in ecfg.lead_times])

    # -- main loop ---------------------------------------------------------

    def run_episode(self, episode: Episode, t_dec: np.ndarray,
                    ecfg: EventConfig) -> PFResult:
        ocfg = self.obs_cfg
        sim = self._fresh_ensemble()
        rng = np.random.default_rng(self.seed + 1)
        w = np.full(self.m, 1.0 / self.m)
        inv_2r2 = 0.5 / ocfg.noise_std**2
        dec_set = {round(float(t), 9) for t in t_dec}
        probs = []
        ess_series = []
        recent_flux: list[np.ndarray] = []
        k_keep = max(1, round(ecfg.flux_window / self.sim_cfg.dt_sample))

        for k, t in enumerate(episode.times):
            rejected = sim.advance(float(t) - sim.t)
            h_p = self._particle_observables(sim, rejected)
            # per-interval flux at obs cadence, kept for rollout smoothing
            recent_flux.append(self._particle_flux(sim, rejected, ocfg.dt_obs))
            recent_flux = recent_flux[-k_keep:]
            loglik = -inv_2r2 * ((episode.y[k] - h_p) ** 2).sum(axis=1)
            logw = np.log(w) + loglik
            logw -= logw.max()
            w = np.exp(logw)
            w /= w.sum()
            ess = 1.0 / float((w**2).sum())
            ess_series.append(ess)
            if ess < self.m / 2:
                w = self._resample(sim, w, rng)
            if round(float(t), 9) in dec_set:
                probs.append(self._rollout_events(sim, w, recent_flux, ecfg))

        return PFResult(t_dec=np.asarray(t_dec), probs=np.array(probs),
                        ess=np.array(ess_series), ess_times=episode.times.copy())
