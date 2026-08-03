"""Two-point (Le Cam) numerical lower bound on backpressure prediction.

Construction (docs/theory.md section 6.4; standard two-point argument,
generalized to stochastic outcomes).  A hypothesis is a full hidden state
theta at the decision time t (lag density + shared-modulator state); the
system is Markov, so the future event E_h is independent of the observed
history given theta.  For a pair (theta_a, theta_b):

- P_i = law of the observation history under theta_i's approach path.  In
  the mean-field approximation the history is the deterministic PDE path
  ending at theta_i (evolved from a common-ancestor perturbation over the
  stated window T_hist under the pair's frozen modulator drift), observed
  through the C1 Gaussian channel:

      KL(P_a || P_b) = 1/(2 r^2) * sum_t || H(rho_t^a) - H(rho_t^b) ||^2 .

- p_i = P(E_h | theta_i), estimated by seeded simulator Monte Carlo from
  theta_i (finite N, free modulator switching): the outcome at lead h is
  genuinely random, driven by future bursts.

For prior 1/2 on each hypothesis, any predictor Y_hat(history) satisfies

    error >= 1/2 [ min(p_a + p_b, 2 - p_a - p_b) - TV(P_a, P_b) ]

with TV <= sqrt(KL/2) (Pinsker).  With deterministic opposite outcomes
(p_a, p_b) = (0, 1) this is the classic Le Cam bound
error >= 1/2 (1 - sqrt(KL/2)).  Monte-Carlo uncertainty in p_i is charged
against the bound (2 * pooled standard errors), so reported bounds err on
the weak side.

Honesty notes (see docs/decisions.md): the first implementation used
deterministic PDE outcomes only and found no opposite-outcome pairs at any
lead time -- under any frozen drift the deterministic dynamics either
saturate before the decision time or never, so outcome flips at long leads
exist only through modulator randomness; the stochastic-outcome
generalization is the fix, not a tuning choice.  The mean-field observation
law ignores finite-N history fluctuations (extra noise not credited to the
channel), keeping the computed KL an overestimate of distinguishability and
the bound conservative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..events.events import EventConfig
from ..obs.observe import ObsConfig
from ..pde.solver import FPResult
from ..sim.config import SimConfig
from ..sim.engine import TauLeapSim

__all__ = ["pde_clean_observables", "kl_gaussian_obs", "pinsker_bound",
           "ensemble_outcome_probs", "two_point_bound", "PairBound"]


def pde_clean_observables(fp: FPResult, obs_cfg: ObsConfig,
                          n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """The C1 observable vector H applied to a PDE trajectory.

    Broker classes are statistically identical in the continuum problem, so
    per-class entries replicate the single-density values; the flux entry is
    the regulator rate averaged over the last observation interval.
    Returns (times, Y) with Y shape (K, 3*n_classes + hist_bins).
    """
    dt = float(fp.times[1] - fp.times[0])
    stride = round(obs_cfg.dt_obs / dt)
    if abs(stride * dt - obs_cfg.dt_obs) > 1e-9 or stride < 1:
        raise ValueError("dt_obs must be a positive integer multiple of the "
                         "PDE sampling interval")
    m_cells = fp.x_centers.size
    if m_cells % obs_cfg.hist_bins != 0:
        raise ValueError("hist_bins must divide the PDE cell count")
    h = 1.0 / m_cells
    idx = np.arange(stride, len(fp.times), stride)
    rho = fp.rho[:, 0, :]
    mass = rho.sum(axis=1) * h
    mean = (rho @ fp.x_centers) * h / mass
    m2 = (rho @ fp.x_centers**2) * h / mass
    var = np.maximum(m2 - mean**2, 0.0)
    flux = (fp.regulator_cum[idx, 0] - fp.regulator_cum[idx - stride, 0]) / (
        stride * dt)
    coarse = rho.reshape(len(fp.times), obs_cfg.hist_bins, -1).sum(axis=2) * h
    coarse = coarse / mass[:, None]
    y = np.concatenate([
        np.repeat(mean[idx, None], n_classes, axis=1),
        np.repeat(var[idx, None], n_classes, axis=1),
        np.repeat(flux[:, None], n_classes, axis=1),
        coarse[idx],
    ], axis=1)
    return fp.times[idx], y


def kl_gaussian_obs(y_a: np.ndarray, y_b: np.ndarray, noise_std: float) -> float:
    """KL between the product-Gaussian observation laws of two clean
    observable histories (equal isotropic covariance r^2 I):
    KL = 1/(2 r^2) sum_t ||y_a(t) - y_b(t)||^2."""
    d = np.asarray(y_a, dtype=float) - np.asarray(y_b, dtype=float)
    return float(0.5 * np.sum(d * d) / noise_std**2)


def pinsker_bound(kl: float) -> float:
    """Classic deterministic-outcome Le Cam bound via Pinsker:
    error >= 1/2 (1 - sqrt(KL/2)), clipped to [0, 1/2]."""
    return float(max(0.0, 0.5 * (1.0 - np.sqrt(min(kl, 2.0) / 2.0))))


def _sample_backlog(rho: np.ndarray, n: int, buffer_depth: int,
                    rng: np.random.Generator) -> np.ndarray:
    """iid draws of Q from a cell-averaged density on [0,1]."""
    m = rho.size
    p = np.maximum(rho, 0.0)
    p = p / p.sum()
    x = (rng.choice(m, size=n, p=p) + 0.5) / m
    return np.rint(x * buffer_depth).astype(np.int64)


def ensemble_outcome_probs(sim_cfg: SimConfig, rho: np.ndarray | None,
                           mod_state: int, ecfg: EventConfig,
                           n_reps: int, seed: int,
                           q_exact: np.ndarray | None = None,
                           flux_history: np.ndarray | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """P(E_h | theta) by vectorized simulator Monte Carlo.

    theta = (lag configuration, shared-modulator state); n_reps independent
    replicas of the N-partition system are packed into one TauLeapSim (one
    modulator group and one block of broker classes per replica) and run
    over (0, h_max]; E_h is evaluated with the dataset's smoothed-flux
    definition.  The lag configuration is either sampled iid from a density
    ``rho`` (mean-field hypothesis states) or replicated exactly from a
    backlog vector ``q_exact`` in partition order (genie evaluation of a
    known hidden state).  ``flux_history`` (most recent last, at dt_sample
    cadence) seeds the trailing smoothing window so the event statistic in
    the first flux_window time units matches the dataset's, whose smoothed
    flux at s in (t, t+w] includes realized pre-t flux; omitting it
    underestimates imminent events at short leads (caught by the e05
    crossing check).  Returns (p_hat, standard error) per lead time; both
    are monotone-coupled across h (same runs).
    """
    n, c = sim_cfg.n_partitions, sim_cfg.n_brokers
    dt = sim_cfg.dt_sample
    h_max = max(ecfg.lead_times)
    k_smooth = max(1, round(ecfg.flux_window / dt))
    big = SimConfig(**{**sim_cfg.__dict__, "n_partitions": n_reps * n,
                       "seed": seed})
    rep = np.repeat(np.arange(n_reps, dtype=np.int64), n)
    classes = rep * c + np.tile(np.arange(n, dtype=np.int64) % c, n_reps)
    rng = np.random.default_rng(seed)
    if q_exact is not None:
        q0 = np.tile(np.asarray(q_exact, dtype=np.int64), n_reps)
    else:
        q0 = _sample_backlog(rho, n_reps * n, sim_cfg.buffer_depth, rng)
    sim = TauLeapSim(big, rng=rng, classes=classes, n_classes=n_reps * c,
                     mmpp_groups=rep,
                     mmpp_state=np.full(n_reps, mod_state, dtype=np.int64)
                     if sim_cfg.arrival == "mmpp" else None, q=q0)
    if flux_history is None:
        hist: list[np.ndarray] = [np.zeros(n_reps)]
    else:
        hist = [np.full(n_reps, float(v)) for v in flux_history[-k_smooth + 1:]]
    sup = {h: np.zeros(n_reps) for h in ecfg.lead_times}
    n_steps = round(h_max / dt)
    for step in range(1, n_steps + 1):
        rej = sim.advance(dt)
        flux = rej.reshape(n_reps, c).sum(axis=1) / (
            n * sim_cfg.buffer_depth * dt)
        hist.append(flux)
        jbar = np.mean(hist[-k_smooth:], axis=0)
        t_ahead = step * dt
        for h in ecfg.lead_times:
            if t_ahead <= h + 1e-9:
                sup[h] = np.maximum(sup[h], jbar)
    p = np.array([(sup[h] > ecfg.threshold).mean() for h in ecfg.lead_times])
    se = np.sqrt(p * (1.0 - p) / n_reps)
    return p, se


@dataclass
class PairBound:
    """Bound contribution of one hypothesis pair, per lead time."""

    label: str
    kl: float
    p_a: np.ndarray
    p_b: np.ndarray
    delta: np.ndarray  # per lead time


def two_point_bound(p_a: np.ndarray, se_a: np.ndarray, p_b: np.ndarray,
                    se_b: np.ndarray, kl: float) -> np.ndarray:
    """error >= 1/2 [min(p_a+p_b, 2-p_a-p_b) - TV] with TV <= sqrt(KL/2),
    charged with 2 pooled standard errors of the Monte-Carlo p's."""
    tv = float(np.sqrt(min(kl, 2.0) / 2.0))
    core = np.minimum(p_a + p_b, 2.0 - p_a - p_b)
    penalty = 2.0 * np.sqrt(se_a**2 + se_b**2)
    return np.maximum(0.0, 0.5 * (core - penalty - tv))
