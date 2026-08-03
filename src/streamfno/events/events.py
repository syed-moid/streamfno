"""Backpressure event definition and labeling.

A backpressure event at lead time h from decision time t is

    E_h(t) = 1{ sup_{s in (t, t+h]} Jbar_B(s) > eps }

where Jbar_B is the hidden aggregate boundary flux (continuum units,
normalized work per partition per unit time) averaged over a trailing
window w.  The window suppresses single-leap shot noise in the rejection
counter; w and eps are fixed per dataset and recorded with it (eps is
calibrated on training episodes only; see experiments/e02_dataset).

Labels are ground truth computed from the hidden trajectory; predictors
never see them as inputs.  E_h is monotone in h by construction
(E_{h1} implies E_{h2} for h1 <= h2 at the same t).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..obs.observe import Episode

__all__ = ["EventConfig", "smoothed_flux", "label_episode", "calibrate_threshold",
           "decision_times"]


@dataclass
class EventConfig:
    """Event parameters: threshold eps on the w-averaged aggregate flux,
    the lead-time grid, and the decision-time convention."""

    threshold: float
    flux_window: float = 2.0
    lead_times: tuple = (1.0, 2.0, 4.0, 8.0, 16.0, 24.0)
    t_warmup: float = 24.0
    dt_decision: float = 4.0
    calibration: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        d["lead_times"] = list(self.lead_times)
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "EventConfig":
        d = json.loads(s)
        d["lead_times"] = tuple(d["lead_times"])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "EventConfig":
        return cls.from_json(Path(path).read_text())


def smoothed_flux(flux_times: np.ndarray, flux: np.ndarray,
                  window: float) -> np.ndarray:
    """Trailing moving average of the flux series over ``window`` time
    units (in samples of the series' own cadence)."""
    dt = float(flux_times[1] - flux_times[0])
    k = max(1, round(window / dt))
    c = np.cumsum(np.concatenate([[0.0], flux]))
    out = np.empty_like(flux)
    idx = np.arange(1, len(flux) + 1)
    lo = np.maximum(idx - k, 0)
    out[:] = (c[idx] - c[lo]) / (idx - lo)
    return out


def decision_times(episode: Episode, ecfg: EventConfig) -> np.ndarray:
    """Decision times on the observation grid: after warmup, early enough
    that the largest lead time still fits in the episode."""
    t = episode.times
    h_max = max(ecfg.lead_times)
    t_end = float(episode.flux_times[-1])
    mask = (t >= ecfg.t_warmup - 1e-9) & (t <= t_end - h_max + 1e-9)
    tt = t[mask]
    stride = max(1, round(ecfg.dt_decision / (t[1] - t[0])))
    return tt[::stride]


def sup_future_flux(episode: Episode, ecfg: EventConfig,
                    t_dec: np.ndarray) -> np.ndarray:
    """sup of the smoothed flux over (t, t+h] for each decision time and
    each lead time; shape (n_dec, n_h)."""
    jbar = smoothed_flux(episode.flux_times, episode.flux_hidden,
                         ecfg.flux_window)
    ft = episode.flux_times
    out = np.zeros((len(t_dec), len(ecfg.lead_times)))
    for i, t in enumerate(t_dec):
        for j, h in enumerate(ecfg.lead_times):
            sel = (ft > t + 1e-9) & (ft <= t + h + 1e-9)
            out[i, j] = jbar[sel].max() if sel.any() else 0.0
    return out


def label_episode(episode: Episode, ecfg: EventConfig,
                  t_dec: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """(decision times, boolean labels of shape (n_dec, n_lead))."""
    if t_dec is None:
        t_dec = decision_times(episode, ecfg)
    sup = sup_future_flux(episode, ecfg, t_dec)
    return t_dec, sup > ecfg.threshold


def calibrate_threshold(episodes: list[Episode], ecfg_proto: EventConfig,
                        h_mid: float, target_rate: float) -> tuple[float, dict]:
    """Choose eps as the (1 - target_rate) quantile of the pooled
    sup-statistic at the middle lead time over the given (training)
    episodes' decision times.  Returns (eps, calibration record)."""
    j = list(ecfg_proto.lead_times).index(h_mid)
    sups = []
    for ep in episodes:
        t_dec = decision_times(ep, ecfg_proto)
        sups.append(sup_future_flux(ep, ecfg_proto, t_dec)[:, j])
    pooled = np.concatenate(sups)
    eps = float(np.quantile(pooled, 1.0 - target_rate))
    record = {
        "h_mid": h_mid, "target_rate": target_rate,
        "n_episodes": len(episodes), "n_decisions": int(pooled.size),
        "achieved_rate": float((pooled > eps).mean()),
        "sup_quantiles": {q: float(np.quantile(pooled, q))
                          for q in (0.5, 0.8, 0.9, 0.95, 0.99)},
    }
    return eps, record
