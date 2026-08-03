"""Margin measurement: distance from decision states to the empirical
event boundary in the observation metric (docs/theory.md section 6.5).

For each decision point, the margin is the whitened observation-space
distance (Euclidean distance of the clean observables divided by the noise
scale r) to the nearest decision point with the opposite event label in
the same load level's pool.  Small typical margins mean states on either
side of the boundary are hard to distinguish through the channel -- the
regime where the two-point bound is informative; large margins mean the
bound is typically vacuous.  Clean observables are used (this is a
diagnostic of the geometry, not a predictor).
"""

from __future__ import annotations

import numpy as np

__all__ = ["nearest_opposite_margins"]


def nearest_opposite_margins(y_clean: np.ndarray, labels: np.ndarray,
                             noise_std: float) -> np.ndarray:
    """Whitened distance from each point to the nearest opposite-label
    point.  y_clean: (n, D); labels: (n,) bool.  Returns (n,) margins
    (NaN if a class is empty)."""
    y = np.asarray(y_clean, dtype=float) / noise_std
    labels = np.asarray(labels, dtype=bool)
    pos = y[labels]
    neg = y[~labels]
    out = np.full(len(y), np.nan)
    if len(pos) == 0 or len(neg) == 0:
        return out
    for cls, other in ((labels, neg), (~labels, pos)):
        pts = y[cls]
        # pairwise distances in chunks (pools are ~1e3 points)
        d2 = ((pts[:, None, :] - other[None, :, :]) ** 2).sum(axis=2)
        out[cls] = np.sqrt(d2.min(axis=1))
    return out
