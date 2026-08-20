.PHONY: setup test lint e00 e01 e02 e03 e04 e05 e06 e07 e08 figures

setup:
	uv venv
	uv sync

test:
	uv run pytest -q

lint:
	uv run ruff check src tests experiments

# Experiments: each run.py simulates and saves results under data/;
# each figures.py regenerates figures under figures/ from saved results only.
e00:
	uv run python experiments/e00_t1_convergence/run.py
	uv run python experiments/e00_t1_convergence/figures.py

e01:
	uv run python experiments/e01_spectral_decay/run.py
	uv run python experiments/e01_spectral_decay/figures.py

e02:
	uv run python experiments/e02_dataset/run.py

e03:
	uv run python experiments/e03_bound/run.py
	uv run python experiments/e03_bound/genie.py
	uv run python experiments/e03_bound/figures.py

e04:
	uv run python experiments/e04_predictors/run.py
	uv run python experiments/e04_predictors/figures.py

e05:
	uv run python experiments/e05_headline/run.py

# e06-e08 need the local Kafka lab running: bash infra/kafka-local.sh start
e06:
	uv run python experiments/e06_real_spectral/run.py
	uv run python experiments/e06_real_spectral/analyze.py
	uv run python experiments/e06_real_spectral/figures.py

e07:
	uv run python experiments/e07_identifiability/run.py
	uv run python experiments/e07_identifiability/analyze.py
	uv run python experiments/e07_identifiability/figures.py

e08:
	uv run python experiments/e08_actuation/run.py
	uv run python experiments/e08_actuation/analyze.py

figures:
	uv run python experiments/e00_t1_convergence/figures.py
	uv run python experiments/e01_spectral_decay/figures.py
	uv run python experiments/e03_bound/figures.py
	uv run python experiments/e04_predictors/figures.py
	uv run python experiments/e05_headline/run.py
	uv run python experiments/e06_real_spectral/figures.py
	uv run python experiments/e07_identifiability/figures.py
