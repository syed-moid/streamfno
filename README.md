# streamfno

Numerical companion to a theory-first study of backpressure in log-centric
message brokers (Apache Kafka).

> We derive a telemetry-identifiable reflected continuum limit for
> many-partition streaming queues and establish an estimator-independent upper
> bound on the lead time over which impending backpressure can be predicted
> from noisy, sampled broker telemetry.

The mathematical program (T1–T4) is stated in `docs/theory.md`. This
repository holds the instruments that probe it empirically: a many-partition
stochastic queue simulator, a reflected nonlinear Fokker–Planck solver, and an
analysis library for density distances and spectral decay.

## Layout

```
docs/                     theory.md (program), experiment-log.md (provenance),
                          decisions.md (technical decisions and rejected alternatives)
infra/                    local Kafka lab scripts (not required for the numerics)
src/streamfno/sim/        many-partition queue simulator (Gillespie + tau-leaping),
                          independent scale parameters N (partitions) and B (buffer depth)
src/streamfno/pde/        conservative finite-volume solver for the reflected
                          Fokker–Planck equation on [0,1] with a regulated boundary at x=1
src/streamfno/analysis/   empirical densities, Wasserstein distances, spectral
                          decay estimation (cosine and FFT bases)
experiments/              one directory per experiment; each README states the
                          claim under test
figures/                  generated output only (git-ignored)
data/                     saved simulation/solver results, .npz (git-ignored)
tests/                    pytest suite (cross-validation and closed-form checks)
```

## Quickstart

Requires `uv` (Python is managed through it; python >= 3.12).

```
make setup      # uv venv + uv sync
make test       # uv run pytest
make lint       # uv run ruff check
```

## Experiments

```
make e00        # T1 convergence: empirical lag density vs Fokker-Planck
                # solution under joint (N, B) scaling; fluid-mode contrast
make e01        # spectral decay of the lag density at three load levels
make figures    # regenerate all figures from saved results in data/
```

Every stochastic run is seeded; every figure is reproducible from a make
target given the saved results in `data/`. Experiment outcomes and environment
notes are recorded in `docs/experiment-log.md`.
