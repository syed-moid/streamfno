# Experiment log

Append-only, dated entries. This file is the provenance record for the paper:
every working session appends an entry stating what was run, with what
configuration, and what came out — including failures, verbatim.

---

## 2026-08-02 — repository scaffold

Environment:

- Machine: Apple Silicon (arm64), macOS (Darwin 25.5.0)
- uv 0.11.19 (aarch64-apple-darwin)
- Python: 3.12+ managed via uv (venv created by `make setup`)

Work:

- Scaffolded the repository layout (`src/streamfno/{sim,pde,analysis}`,
  `experiments/{e00_t1_convergence,e01_spectral_decay}`, `docs/`, `figures/`,
  `data/`, `tests/`).
- Copied the theory program verbatim from the author's thesis notes into
  `docs/theory.md` (T1 continuum limit, T2 viability, T3 predictability
  horizon, T4 spectral truncation).
- `pyproject.toml` with numpy/scipy/matplotlib; dev: pytest, ruff. Package
  managed exclusively through uv.
- `.gitignore` excludes `manuscript/`, documents, generated figures and data,
  `kafka-lab/` runtime state, and the venv, so `git add -A` is safe.
- `infra/kafka-local.sh` was already present and is left untouched; Kafka is
  not required for Phase B and was not exercised this session.
