# e01 — spectral decay of the lag density

## Claim under test

The lag density is smooth enough (large decay exponent s in |ĉ_k| ~ k^(−s))
for a spectral-operator approach to be viable — **including near the
regulated boundary at x = 1 under load**. This experiment is the go/no-go
instrument for the paper's Fourier-operator premise (T4): if s collapses near
saturation, the FFT-based operator premise is in trouble. The numbers are
reported whatever they are.

## Method

- Three load levels of the diffusive-mode simulator (N = 1000, B = 256,
  a = 0.04, μ0 = 0.7, one broker class), chosen so the stationary mean lag
  spans the range:
  - light: λ = 0.40 (b ≈ −0.30, mean lag ≪ 1),
  - moderate: λ = 0.65 (b ≈ −0.05, spread density),
  - near-saturation: λ = 0.80 (supercritical; service degradation engages,
    visible boundary interaction at x = 1, nonzero boundary flux J_B).
- 8 seeds each, T = 200; empirical density = fixed-grid histogram (128 bins;
  B = 2 × 128 so lattice atoms map evenly onto bins) averaged over t ≥ 80
  and seeds; tau-leap step reduced 4× below the default (jump cap 1.25)
  because the spectra have a coherent instrument floor of residual leap bias
  that scales ~linearly with the step (see run.py and docs/decisions.md).
- Spectral coefficients in the cosine basis (DCT-II; natural under
  reflecting boundaries) and the periodic FFT basis; tail exponent s fitted
  by Theil–Sen regression of log|ĉ_k| on log k over k ∈ [3, 16] (stated
  fixed range, upper end set by the instrument floor), with 95% confidence
  intervals; a split-half difference spectrum estimates the incoherent
  sampling-noise floor and is drawn on every spectrum plot.
- The matched PDE stationary density (M = 512, restricted to the 128-bin
  grid) is analyzed identically as a sampling-noise-free reference.

## Run

```
make e01        # simulate + save to data/e01/ + regenerate figures
make figures    # figures only, from saved results
```

Artifacts: `data/e01/results.npz`; figures `figures/e01_densities.png`,
`figures/e01_spectra_cosine.png`, `figures/e01_spectra_fft.png`.
