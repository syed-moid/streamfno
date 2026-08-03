# e04 — predictor suite against the telemetry channel

## Claim under test

Practical predictors consuming exactly the C1 observation process approach,
but do not exceed, the computed predictability ceiling; the well-specified
particle filter is the ceiling candidate, and the gap between reactive
practice and filtering quantifies how much predictability current
lag-threshold autoscaling leaves on the table.

## Predictors (identical information sets: episode.y only)

1. **Bootstrap particle filter** on the simulator's own model
   (well-specified case): M full-hidden-state particles propagated by the
   same tau-leap engine, Gaussian observation weights, systematic
   resampling below ESS M/2 (ESS recorded — degeneracy near saturation is
   an expected failure mode to report). P(E_h | obs) by posterior rollout
   with fresh randomness. M convergence-checked by doubling until the
   Brier score stabilizes.
2. **Logistic regression** on windowed features (last W = 8 samples,
   stacked, plus per-dimension mean/max/trend).
3. **Gradient-boosted trees** (sklearn HistGradientBoosting) on the same
   features.
4. **Reactive threshold heuristic**: smoothed observed flux or near-wall
   histogram mass (variant chosen on validation), threshold tuned on
   validation — the proxy for reactive autoscaling practice.

## Evaluation

Per load level and lead time: misclassification error at the val-tuned
operating point, PR-AUC, Brier score, episode-bootstrap 95% CIs (episodes
are the exchangeable unit; base rates reported alongside since they drift
across levels by design). Calibration curves for the particle filter and
GBT at h = 8. PF threshold tuning uses an 8-episode validation subset per
level (wall-clock reduction; recorded).

## Run

```
make e04        # requires data/e02
```

Artifacts: `data/e04/{results.json, calibration.json, pf_probs.npz}`;
figures `figures/e04_errors.png`, `figures/e04_calibration.png`,
`figures/e04_ess.png`.
