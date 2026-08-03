# e00 — T1 convergence of the empirical lag density

## Claim under test

The empirical lag density of the N-partition simulator converges to the
matched reflected Fokker–Planck solution under **joint (N, B) scaling** in
diffusive mode, and does **not** acquire a diffusion term under N-scaling
alone (fluid mode): the fluid-mode empirical density tracks the transport
(a = 0) solution, not the diffusive one.

## Method

- Sweep N ∈ {50, 200, 1000} × B ∈ {10, 50, 100, 200}, 5 seeds each,
  diffusive mode (tau-leap engine; the engines are cross-validated in
  `tests/test_sim.py`), constant Poisson netput λ = 0.55, μ0 = 0.7,
  a = 0.04, one broker class, truncated-Gaussian initial law (0.35, 0.10).
- At t ∈ {2, 5, 10, 20, 40}: Wasserstein-1 distance between the exact
  lag-lattice empirical measure and the matched PDE density
  (Chang–Cooper, M = 400, dt = 2e-3; refinement-checked against
  M = 800, dt = 1e-3 within the run).
- Fluid contrast: N = 1000, B = 200 in fluid mode over t ∈ [0, 2]; W1 to
  the transport solution vs W1 to the diffusive solution.

## Success criterion

Mean W1 decreases with a plausibly power-law trend in both N (at fixed
largest B) and B (at fixed largest N) in diffusive mode; the fluid-mode run
is visibly closer to transport than to diffusion. A failure is a scientific
finding and is recorded unvarnished in `docs/experiment-log.md`.

## Run

```
make e00        # simulate + save to data/e00/ + regenerate figures
make figures    # figures only, from saved results
```

Artifacts: `data/e00/summary.npz` (distance tensor + axes),
`data/e00/pde_reference.npz`, `data/e00/fluid_contrast.npz`, one
representative raw run `data/e00/run_N1000_B100_seed0.npz`; figures
`figures/e00_w1_vs_N.png`, `figures/e00_w1_vs_B.png`,
`figures/e00_fluid_contrast.png`.
