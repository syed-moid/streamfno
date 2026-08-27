# streamfno

Computational artifact for the paper **"Backpressure in Apache Kafka:
Continuum Dynamics, Predictability Limits, and Operational Safety
Horizons"** (Syed Moid, Ronin Institute for Independent Scholarship 2.0).

A telemetry-driven framework that predicts when backpressure will occur
in a partitioned streaming platform, how far ahead prediction is
reliable, and how much time remains before an operational safety
boundary is crossed: a reflected mean-field Fokker–Planck model
identified from standard Kafka lag telemetry, an error decomposition
that bounds predictability, and live-cluster validation through to
retention-boundary early warning.

## Layout

```
infra/                    local Kafka lab (3-broker KRaft, no Docker, folder-local);
                          manuscript number-audit script
src/streamfno/sim/        many-partition queue simulator (Gillespie + tau-leaping),
                          independent scale parameters N (partitions) and B (buffer depth)
src/streamfno/pde/        conservative finite-volume solver for the reflected
                          Fokker–Planck equation on [0,1] with a regulated boundary at x=1
src/streamfno/analysis/   empirical densities, Wasserstein distances, spectral decay
src/streamfno/kafka/      live-cluster harness: producer/consumer pacing, lag
                          collector, run scenarios (seeded, resumable)
experiments/              one directory per experiment; each states the claim under test
figures/                  generated output only (git-ignored)
data/                     simulation/solver/cluster results (git-ignored)
tests/                    pytest suite (cross-validation and closed-form checks)
```

## Quickstart

Requires `uv` (Python is managed through it; python >= 3.12).

```
make setup      # uv venv + uv sync
make test       # uv run pytest
make lint       # uv run ruff check
make audit      # manuscript headline-number audit
```

## Experiment → paper asset map

Every stochastic run is seeded; every figure and table regenerates from
one make target given the saved results in `data/`. Live-cluster
experiments (e06–e08, e11) need the local lab:
`bash infra/kafka-local.sh start`.

| Target | Experiment | Paper asset |
|---|---|---|
| `make e00` | joint-limit convergence (N, B) | Fig. 1 |
| `make e01` | simulator spectral decay | Fig. 3 overlays |
| `make e02`–`make e05` | episode library; predictor sweep; error decomposition, genie floor, skill horizons | Fig. 2, Tables I–II |
| `make e06` | real-cluster spectral decay | Fig. 3 (suppl.: both bases) |
| `make e07` | telemetry identifiability end to end | Fig. 4; §IV numbers |
| `experiments/e07_identifiability/lite384.py` | 384-partition replication | §IV scale-out passage |
| `make e08` (+ `extend.py`) | consumer-scaling actuation delay | §V actuation numbers |
| `make e09` | computational profile | Table III; suppl. profile grid |
| `make e10` (+ `direct.py`) | surrogate frontier, autoregressive + direct | Fig. 5 |
| `make e11` (+ `campaign.py`, `campaign_analyze.py`) | retention-boundary warning; pre-registered 7-cell campaign | Fig. 6; suppl. per-cell table |
| `make e12` | solver refinement orders | suppl. refinement study |

See `PROVENANCE.md` for the full reproduction statement.

## License

MIT (see `LICENSE`).
