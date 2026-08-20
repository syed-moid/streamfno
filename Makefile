.PHONY: setup test lint e00 e01 e02 e03 e04 e05 e06 e07 e08 e09 e10 e11 e12 figures

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

# e09/e10 run on saved e06/e07 telemetry and artifacts -- no cluster needed
e09:
	uv run python experiments/e09_compute_profile/run.py
	uv run python experiments/e09_compute_profile/analyze.py

e10:
	uv run python experiments/e10_surrogate/run.py
	uv run python experiments/e10_surrogate/analyze.py
	uv run python experiments/e10_surrogate/figures.py

# e11 needs the local Kafka lab running: bash infra/kafka-local.sh start
# (restarted at least once after the retention-check-interval setting)
e11:
	uv run python experiments/e11_retention/run.py
	uv run python experiments/e11_retention/analyze.py
	uv run python experiments/e11_retention/figures.py

# solver validation: refinement orders + independent MC reference
e12:
	uv run python experiments/e12_solver_validation/run.py

figures:
	uv run python experiments/e00_t1_convergence/figures.py
	uv run python experiments/e01_spectral_decay/figures.py
	uv run python experiments/e03_bound/figures.py
	uv run python experiments/e04_predictors/figures.py
	uv run python experiments/e05_headline/run.py
	uv run python experiments/e06_real_spectral/figures.py
	uv run python experiments/e07_identifiability/figures.py
	uv run python experiments/e10_surrogate/figures.py
