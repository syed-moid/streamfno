"""Windowed observation features.

All feature-based predictors consume, for each decision time t, the last W
noisy observation vectors (the C1 channel and nothing else), stacked, plus
simple derived statistics per observable dimension: window mean, window
max, and linear trend slope.  Labels never enter features.
"""

from __future__ import annotations

import numpy as np

from ..obs.observe import Episode

__all__ = ["window_features"]


def window_features(episode: Episode, t_dec: np.ndarray,
                    window: int = 8) -> np.ndarray:
    """Feature matrix (n_dec, W*D + 3*D) from the noisy observations.

    Decision times must lie on the observation grid with at least
    ``window`` observations available (guaranteed by the dataset's warmup).
    """
    times = episode.times
    y = episode.y
    idx = np.searchsorted(times, t_dec)
    if not np.allclose(times[idx], t_dec):
        raise ValueError("decision times must lie on the observation grid")
    if (idx < window - 1).any():
        raise ValueError("decision time earlier than the feature window")
    tc = np.arange(window) - (window - 1) / 2.0
    denom = float((tc**2).sum())
    feats = []
    for i in idx:
        block = y[i - window + 1:i + 1]
        slope = tc @ block / denom
        feats.append(np.concatenate([block.ravel(), block.mean(axis=0),
                                     block.max(axis=0), slope]))
    return np.asarray(feats)
