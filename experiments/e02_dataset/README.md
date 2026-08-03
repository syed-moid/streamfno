# e02 — episode library for the predictability-horizon experiments

## Purpose (dataset properties under test)

Build the shared episode library that C3 (bounds) and C4 (predictors)
consume, with these properties:

- **Bursty traversal.** Shared-modulator MMPP arrivals (coherent bursts,
  λ_low = 0.40 → λ_high = 0.82, burst end rate 0.30) drive episodes from
  light load to near-saturation and back. Three load levels differ in burst
  frequency r_low_high ∈ {0.03, 0.06, 0.10} (light / moderate / heavy).
- **Calibrated events.** Backpressure event E_h(t) = 1{sup of the 2-unit
  smoothed aggregate boundary flux over (t, t+h] > ε}; ε is calibrated on
  the *moderate-level training episodes only* so the base rate at the middle
  lead time h = 8 is ≈ 10% (target band 5–15%). Base rates at every
  level × split × lead time are computed and stored (they drift across
  levels by design; all comparisons report them).
- **Single observation channel.** Every episode is observed once through
  the C1 telemetry layer (Δ = 1, R = 0.02² I, 8-bin coarse histogram),
  with a dedicated noise stream per episode (noise_seed = 100000 +
  sim_seed). No consumer of the library touches the hidden state except
  for labels and diagnostics.
- **Episode-level splits.** 130 episodes per level, split 78/26/26 into
  train/val/test by episode — never within an episode.

Simulator config per episode: N = 400, B = 128, diffusive mode a = 0.05,
2 broker classes, degradation drop 0.3, T = 120, warmup 24; decision times
every 4 units in [24, 96]; lead grid h ∈ {1, 2, 4, 8, 16, 24}.

Matched PDE reference trajectories (drift with the modulator frozen in each
state, per level) are solved and stored for C3 under `data/e02/pde_refs/`.

## Run

```
make e02
```

Artifacts under `data/e02/`: `episodes/<level>/ep_*.npz`, `manifest.json`,
`event_config.json` (ε + calibration record), `base_rates.json`,
`obs_config.json`, `pde_refs/*.npz`.
