# Provenance

How the results in "Backpressure in Apache Kafka: Continuum Dynamics,
Predictability Limits, and Operational Safety Horizons" were produced,
and how to reproduce them from this repository.

## Reproduction model

Every experiment is a seeded, replayable script under `experiments/`,
invoked by one make target (see the map in `README.md`). Simulator and
solver experiments (e00–e05, e09, e10, e12) are deterministic given
their seeds and run anywhere `uv` runs. Live-cluster experiments
(e06–e08, e11) run against the folder-local 3-broker Kafka lab
(`infra/kafka-local.sh`, KRaft, no external services); their drivers
are resumable — a killed run continues from its last completed unit —
and every run directory records its parameters, seed, and manifest.
Figures and tables regenerate from saved results only (`make figures`
or the per-experiment figure scripts); no figure is hand-edited.

## Statistical protocol

- Calibration and evaluation use disjoint seed ranges throughout;
  estimator coefficients are frozen on calibration runs before any
  held-out evaluation.
- The retention campaign (e11, seven operating conditions, 23
  boundary crossings) was pre-registered: the estimand (pooled median
  warning lead), the run-level cluster bootstrap (4000 resamples,
  percentile 95% CIs), the left-censoring rules, and the ≥3-crossings
  cell-validity criterion were fixed before launch and applied
  unchanged.
- Uncertainty on burst- and crossing-level quantities uses run-level
  (cluster) bootstraps to avoid pseudoreplication; episode-level
  bootstraps are used where episodes are the sampling unit.
- The headline numbers quoted in the manuscript are guarded by an
  automated audit (`make audit`) asserting the current values are
  present in the sources and superseded values are absent.

## Solver validation

The reflected Fokker–Planck solver (conservative finite volume,
Chang–Cooper flux, implicit Euler) is validated by measured refinement
orders (space ≈ 2.0, time ≈ 1.0 in the simulator regime; boundary-layer
behavior reported separately), exact mirror-symmetry and stationarity
checks under both drift signs, and agreement with an independent
reflected Euler–Maruyama Monte Carlo within the solve's own spatial
discretization error. The test suite (58 tests) covers these plus
closed-form and cross-validation checks of the simulator and analysis
library.

## Hardware and timing

All timing measurements (computational profile, surrogate frontier,
actuation delays) were taken on a single Apple M-class arm64 machine
(macOS), single-threaded, warm-ups excluded, p50/p95 reported; the
Kafka lab (3 brokers, KRaft) and all clients ran on the same machine.
Cluster-experiment wall-clock is real time against the live brokers.

## Archive

A tagged release of this repository, archived with a DOI, accompanies
the paper; the paper's data-availability statement points to it.
