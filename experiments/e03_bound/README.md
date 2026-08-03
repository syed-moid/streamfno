# e03 — bound instruments for the predictability horizon (T3)

## Claim under test

There is a computable, estimator-independent lower bound δ_min(h) on the
error of predicting the backpressure event E_h from the C1 telemetry
channel, and it grows with lead time h.

## Instruments

1. **Finite-time divergence rate (diagnostic).** Twin PDE runs with
   localized bump perturbations under the three frozen drift regimes
   (drain λ_low, mean rate, burst λ_high); effective exponent λ_+ from the
   log-L2-separation slope over a stated window. Parameterizes the
   interpretable (Lyapunov) form of the bound.
2. **Two-point bound (primary).** Hypothesis pairs are full hidden states
   at the decision time — a lag density (drawn from the reference
   trajectory at a grid of mean lags, plus density-gap variants) and a
   shared-modulator state (including modulator-flip pairs). The history KL
   uses mean-field approach paths over the stated window (T_hist = 8)
   through the Gaussian channel; outcome probabilities p_a, p_b are seeded
   simulator Monte Carlo with free modulator switching (events at long
   leads are genuinely stochastic). Per pair and lead time,
   δ ≥ ½[min(p_a+p_b, 2−p_a−p_b) − √(KL/2)] minus twice the pooled MC
   standard error; δ_min(h) is the max over pairs. The classic
   deterministic-outcome Le Cam form ½(1 − √(KL/2)) is the special case
   (p_a, p_b) = (0, 1). The first, purely deterministic implementation
   found *no* opposite-outcome pairs at any lead (every admissible state
   saturates before the decision time under frozen burst drift, or never
   under drain) — the investigation and the stochastic-outcome fix are
   recorded in docs/decisions.md. Burst frequency enters through the
   outcome MC, so the bound is computed per load level.
3. **Margins.** Whitened observation-space distance from test-episode
   decision states to the nearest opposite-label state, per level — is the
   bound typically informative (margins in the KL < 2 zone) or vacuous?
4. **Genie floor (addendum, `genie.py`).** δ_min is a worst-case-state
   statement; measured average predictor error can legitimately sit below
   it where hard states are rare. To make the "no predictor can enter"
   comparison enforceable against measured test error, γ(h) = mean over
   test decision states of min(p, 1−p), with p(θ) estimated by seeded MC
   from the exactly replayed hidden state — the average error of a
   state-omniscient predictor, which no telemetry predictor can beat in
   expectation (data processing).

## Sanity requirements (checked and recorded in `data/e03/sanity.json`)

- δ_min(h) nondecreasing in h (in the mean); investigated before
  proceeding if violated.
- A bound that is ≈ 0 at every practical h is reported as a finding, not
  tuned into significance.
- No predictor may beat the bound (checked in e04/e05); a crossing is
  treated as a bug in the bound computation and root-caused.

## Run

```
make e03        # requires data/e02 (make e02)
```

Artifacts: `data/e03/{lecam.npz, divergence.json, margins.npz, sanity.json}`;
figures `figures/e03_bound.png`, `figures/e03_divergence.png`,
`figures/e03_margins.png`.
