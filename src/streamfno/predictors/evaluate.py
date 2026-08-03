"""Evaluation: operating points, metrics, and episode-bootstrap CIs.

Rules (Phase C quality bar): thresholds are tuned on validation data only;
metrics are reported on test data; every probabilistic claim carries an
episode-bootstrap confidence interval (episodes are the exchangeable unit
-- decision points within an episode are correlated).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

__all__ = ["tune_threshold", "metrics_with_ci", "calibration_curve_data"]


def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Score threshold minimizing misclassification error on the given
    (validation) data; ties broken toward the median candidate."""
    order = np.argsort(scores)
    s = scores[order]
    y = labels[order].astype(int)
    # candidate cuts between consecutive distinct scores
    cuts = np.concatenate([[s[0] - 1e-12],
                           0.5 * (s[1:] + s[:-1]), [s[-1] + 1e-12]])
    n_pos = y.sum()
    # errors(cut) = false positives + false negatives
    cum_pos = np.concatenate([[0], np.cumsum(y)])
    n = len(y)
    idx = np.arange(n + 1)
    fn = cum_pos            # positives predicted negative (below cut)
    fp = (n - idx) - (n_pos - cum_pos)
    err = (fn + fp) / n
    best = np.flatnonzero(err == err.min())
    return float(cuts[best[len(best) // 2]])


def _point_metrics(scores, labels, thr):
    pred = scores > thr
    error = float((pred != labels).mean())
    if labels.any() and (~labels).any():
        pr_auc = float(average_precision_score(labels, scores))
    else:
        pr_auc = float("nan")
    p = np.clip(scores, 0.0, 1.0)
    brier = float(brier_score_loss(labels.astype(int), p))
    return error, pr_auc, brier


def metrics_with_ci(scores: np.ndarray, labels: np.ndarray,
                    episode_ids: np.ndarray, threshold: float,
                    n_boot: int = 1000, seed: int = 0) -> dict:
    """Misclassification at the tuned operating point, PR-AUC, Brier score,
    with percentile CIs from resampling whole episodes with replacement.

    Brier assumes scores in [0, 1] (probabilities); heuristic raw scores
    should be evaluated with metrics that do not require calibration
    (error, PR-AUC) -- their Brier is reported for the hard 0/1 prediction.
    """
    labels = np.asarray(labels, dtype=bool)
    error, pr_auc, brier = _point_metrics(scores, labels, threshold)
    rng = np.random.default_rng(seed)
    uniq = np.unique(episode_ids)
    boot = np.full((n_boot, 3), np.nan)
    for b in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(episode_ids == e) for e in take])
        boot[b] = _point_metrics(scores[idx], labels[idx], threshold)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return {
        "threshold": float(threshold),
        "base_rate": float(labels.mean()),
        "error": error, "error_ci": [float(lo[0]), float(hi[0])],
        "pr_auc": pr_auc, "pr_auc_ci": [float(lo[1]), float(hi[1])],
        "brier": brier, "brier_ci": [float(lo[2]), float(hi[2])],
        "n_points": int(len(labels)), "n_episodes": int(len(uniq)),
    }


def calibration_curve_data(probs: np.ndarray, labels: np.ndarray,
                           n_bins: int = 10) -> dict:
    """Reliability diagram data: mean predicted vs observed frequency per
    probability bin, with counts."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    pred, obs, count = [], [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            pred.append(float(probs[sel].mean()))
            obs.append(float(labels[sel].mean()))
            count.append(int(sel.sum()))
    return {"pred": pred, "obs": obs, "count": count}
