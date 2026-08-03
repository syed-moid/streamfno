# e05 — headline figure: predictors vs the predictability ceiling

Prototype of the paper's central exhibit. One figure, three panels (load
levels): x = lead time h, y = prediction error; the impossibility region
is shaded below the **measured state-omniscient (genie) predictor** —
1{p̂ > ½} at each replayed hidden state, scored on the same realized test
labels and bootstrapped the same way as every other predictor, which is
the enforceable "no predictor with this telemetry can enter" ceiling
(label applied only when the crossing check passes). The expectation floor
γ(h) = E[min(p, 1−p)] and the worst-case-state two-point bound δ_min(h)
are drawn alongside as the theoretical quantities; one curve per e04
predictor with episode-bootstrap CI bands; the trivial predict-majority
error for reference. See docs/decisions.md for why the measured form is
the comparable one (episode-clustered outcome luck is shared between
predictors and ceiling).

Also the gap-to-the-ceiling table: per predictor, the largest h at which
its test error stays below δ = 0.2, against the largest h at which the
measured genie stays below 0.2.

Pure assembly from `data/e03` and `data/e04`; `make e05` regenerates it end
to end from saved results.

```
make e05
```

Artifacts: `figures/e05_headline.png`, `data/e05/gap_table.{md,json}`.
