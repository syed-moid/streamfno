# e05 — headline figure: predictors vs the predictability ceiling

Prototype of the paper's central exhibit. One figure, three panels (load
levels): x = lead time h, y = prediction error; the numerically computed
impossibility region is shaded below the two-point bound δ_min(h) (labeled
"no predictor with this telemetry can enter" only when the e03 sanity
checks passed); one curve per e04 predictor with episode-bootstrap CI
bands; the trivial predict-majority error drawn for reference.

Also the gap-to-the-ceiling table: per predictor, the largest h at which
its test error stays below δ = 0.2, against the largest h at which the
bound still permits error below 0.2.

Pure assembly from `data/e03` and `data/e04`; `make e05` regenerates it end
to end from saved results.

```
make e05
```

Artifacts: `figures/e05_headline.png`, `data/e05/gap_table.{md,json}`.
