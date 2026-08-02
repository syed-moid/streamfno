.PHONY: setup test lint e00 e01 figures

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

figures:
	uv run python experiments/e00_t1_convergence/figures.py
	uv run python experiments/e01_spectral_decay/figures.py
