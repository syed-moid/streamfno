"""Feature-based baseline predictors and the reactive threshold heuristic.

All consume C1 observations only (via features.window_features or the raw
noisy observation vectors).  Scores are probabilities for the learned
models and a raw telemetry statistic for the heuristic; operating points
are tuned on validation data only (evaluate.tune_threshold).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..obs.observe import Episode

__all__ = ["make_logistic", "make_gbt", "reactive_scores"]


def make_logistic(seed: int = 0):
    """Logistic regression on windowed features (standardized)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, C=1.0, random_state=seed),
    )


def make_gbt(seed: int = 0):
    """Histogram gradient-boosted trees on the same features."""
    return HistGradientBoostingClassifier(random_state=seed)


def reactive_scores(episode: Episode, t_dec: np.ndarray, kind: str,
                    smooth: int = 4) -> np.ndarray:
    """Reactive heuristic score at each decision time.

    kind = "flux": trailing mean over ``smooth`` observation samples of the
    observed per-class flux averaged across classes -- the observable
    counterpart of smoothed J_B.
    kind = "wall": current near-wall mass, the top coarse-histogram bin.
    The proxy for lag-threshold reactive autoscaling practice; the firing
    threshold is tuned on validation episodes.
    """
    cfg = episode.obs_config
    n_c = episode.sim_config.n_brokers
    idx = np.searchsorted(episode.times, t_dec)
    if kind == "flux":
        flux = episode.y[:, 2 * n_c:3 * n_c].mean(axis=1)
        c = np.cumsum(np.concatenate([[0.0], flux]))
        lo = np.maximum(idx + 1 - smooth, 0)
        return (c[idx + 1] - c[lo]) / (idx + 1 - lo)
    if kind == "wall":
        return episode.y[idx, 3 * n_c + cfg.hist_bins - 1]
    raise ValueError(f"unknown heuristic kind {kind!r}")
