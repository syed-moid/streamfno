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

---

## 2026-08-02 — Phase B: simulator, solver, analysis, e00, e01

### 1. What was built (commit by commit)

- `53783d3` Scaffold: layout, docs, uv packaging, make targets, gitignore.
- `0ba819e` Simulator (B1): Gillespie + vectorized tau-leap engines,
  diffusive/fluid modes, Poisson/MMPP arrivals, broker-class logistic
  service degradation, boundary flux J_B, seeded npz outputs, engine
  cross-validation tests.
- `84af46e` Fokker–Planck solver (B2): Chang–Cooper finite volume, implicit
  Euler, mean-field coupling, no-flux wall at 0, regulated wall at 1;
  closed-form stationary validation.
- `ef642de` Analysis (B3): W1 on lattice/density supports; cosine (DCT-II)
  and FFT spectra with Theil–Sen tail fits and CIs.
- `b1f2840` Experiment e00 + `streamfno.matching` (sim↔PDE correspondence).
- `7549e4a` Bug fix: PDE regulator was the cancelled advective flux
  (dimensionally wrong, ~20× overestimate of J_B near saturation); replaced
  with the Skorokhod local-time rate (a/2)ρ(1,t) + transport wall-atom term;
  validated K̇ → b against the closed form and the simulator.
- `9317fc1` Bug fix: fixed service-before-arrival ordering inside a tau-leap
  step depleted the Q=0 wall site by ~10% (coherent artifact, flat plateau
  in the spectra); ordering now randomized per partition per step, verified
  against Gillespie.
- `06ab22b` Experiment e01 with measured instrument floors.
- `5d45b6f` python-docx dev dependency (manuscript tooling).

### 2. e00 — T1 convergence: criterion MET

W1 between the lag-lattice empirical measure and the matched PDE at t=40
(mean ± sd over 5 seeds), final engine:

| N \ B  | 10     | 50     | 100    | 200    |
|--------|--------|--------|--------|--------|
| 50     | 0.0488 | 0.0261 | 0.0214 | 0.0215 |
| 200    | 0.0491 | 0.0154 | 0.0119 | 0.0097 |
| 1000   | 0.0520 | 0.0090 | 0.0053 | 0.0049 |

(Full tensor with per-seed values and all five sample times in
data/e00/summary.npz; the sweep was re-run in full after the tau-leap
ordering fix, which mainly lowered the B≥50 floors at large N.)

- Log-log slope of mean W1 vs N at B=200: −0.49 (CLT-consistent);
  vs B at N=1000: −0.83. Distances decrease with plausibly power-law trend
  in both N and B; the B=10 column shows the expected discretization floor
  (~0.05) that N-scaling alone cannot remove — the double limit is jointly
  necessary, as T1 asserts.
- PDE refinement check (M=400, dt=2e-3 vs M=800, dt=1e-3): max W1 6.4e-4,
  well below all measured distances.
- Fluid contrast (N=1000, B=200, t∈[0,2]): final W1 to the transport
  solution 0.018 vs 0.096 to the diffusion solution; the empirical density
  visibly tracks transport. N-scaling alone does not produce a diffusion
  term.

### 3. e01 — spectral decay: premise SURVIVES with a caveat near saturation

Fitted s (|ĉ_k| ~ k^(−s), Theil–Sen over k∈[3,16], 95% CI), 8 seeds, T=200,
B=256, leap step 4× reduced (see §4):

| load            | sim cos           | sim FFT           | PDE cos | PDE FFT |
|-----------------|-------------------|-------------------|---------|---------|
| light (λ=0.40)  | 1.57 [1.39,1.71]  | 0.90 [0.88,0.92]  | 1.52    | 0.91    |
| moderate (0.65) | 2.14 [1.98,2.51]  | 0.99 [0.97,1.01]  | 1.99    | 0.98    |
| near-sat (0.80) | 1.12 [1.00,1.18]  | 0.76 [0.74,0.77]  | 1.37    | 0.88    |

Mean boundary flux J_B: ~0 (light), 4.7e-3 (moderate), 0.368 (near-sat; the
PDE local-time rate and the simulator's rejected-work counter agree at the
few-percent level there).

Reading: (i) the basis choice matters exactly as the Neumann/periodic
mismatch predicts — FFT decay is pinned at s≈0.9–1.0 at every load by the
periodic-extension jump, while the cosine basis sees the actual smoothness;
(ii) in the cosine basis the premise survives: s≈1.5–2.1 at light/moderate
load; (iii) near saturation s degrades to 1.12 [1.00,1.18] (sim) / 1.37
(PDE) over the stated range — not a collapse, but a measurable loss of
effective smoothness from the boundary layer (width ≈ a/2b ≈ 0.05). A plain
FFT-based operator premise is marginal near saturation; a boundary-adapted
(cosine/Neumann) representation retains s > 1.

### 4. Failures, surprises, and judgment calls (for review)

- **PDE regulator bug (fixed, `7549e4a`).** First implementation used the
  cancelled advective flux as K̇ — dimensionally a probability flux, not a
  work rate; ~20× overestimate vs the simulator near saturation. Caught by
  cross-checking J_B; the local-time form now agrees with both the closed
  form and the simulator.
- **Tau-leap ordering bias (fixed, `9317fc1`).** Deterministic
  service-then-arrival ordering within a step depleted the Q=0 lattice site
  by ~10% and distorted the Q=B wall — a coherent artifact that flattened
  e01's cosine spectra at ~1%·ĉ0 and initially made the moderate-load fit
  come out flat/negative while the noise-free PDE said s≈2. Diagnosed by
  lattice-level comparison against Gillespie; fixed by randomizing the
  within-step order. This also improved e00's large-N floors and slopes.
- **Residual leap-bias plateau (measured, worked around).** After the fix a
  smaller coherent plateau remains in the spectra; scaling runs show it is
  independent of N (unchanged at N=4000 — not a finite-N mean-field effect)
  and ~linear in the leap step (6.7e-3·ĉ0 at τ≈1.9e-3 → 1.1e-3·ĉ0 at
  τ≈4.8e-4). e01 therefore runs at the smaller step and fits over k∈[3,16].
  Anyone pushing the fit range further must push τ down accordingly.
- **Judgment calls made without guidance** (each argued in
  docs/decisions.md): diffusive-mode jump construction sacrifices literal
  arrival/service semantics to keep `a` an independent parameter;
  mass-conserving (not absorbing) regulated wall; W1 computed from the
  exact lattice measure rather than the re-binned fixed grid; e00 W1 at
  single sample times rather than time-averaged windows; e01 fit range set
  by the measured instrument floor.
- **Not done:** `infra/kafka-local.sh` was not exercised (Kafka is not
  required for Phase B; no download was attempted, so no proxy issues to
  record). MMPP arrivals are implemented and tested but not used by e00/e01.

### 5. Wall-clock and machine notes

Apple Silicon (arm64), macOS 25.5.0, uv 0.11.19, Python 3.13 (uv-managed
venv), numpy 2.x/scipy 1.18. Test suite: 20 tests, ~5 s. e00 full sweep (60
runs + 2 PDE references + fluid contrast): ~24 s. e01 (24 runs at reduced
step + 3 PDE solves): ~15 min; at the default step it is ~5 min. Spec
benchmark: N=1000, B=100, T=600 diffusive with 4 broker classes: 6.1 s
(target was minutes). Gillespie reference: ~1 s per (N=80, B=16, T=24) run.

Manuscript: skeleton regenerated in place at
manuscript/FNO_Back_Pressure_prediction_in_messaging.docx (author's original
preserved as *.pre-regeneration-backup.docx; both git-ignored) with the new
title, positioning-sentence abstract, T1–T4-as-claims introduction, and the
section structure agreed for the theory-first direction; no numerical claims
included.
