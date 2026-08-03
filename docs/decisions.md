# Technical decisions

Short log of nontrivial technical/modeling choices, the alternative rejected,
and why. Add an entry in the same commit that introduces the choice.

---

## 2026-08-02 — scaffold

**Build tooling: uv + hatchling, src layout.** All Python management goes
through uv (`uv venv`, `uv sync`, `uv run`); no system pip, no conda. The
src layout keeps experiment scripts importing the installed package rather
than the working tree implicitly.

**Figures and data are build products.** `figures/*` and `data/*` are
git-ignored; every figure must regenerate from saved results via a make
target. Rejected: committing figures — it invites stale figures that no
longer match the code that made them.

---

## 2026-08-02 — simulator (B1)

**Jump construction, diffusive mode.** Each partition is a birth–death chain
on {0,…,B} with up/down rates `u = B²a/2 + Bb/2`, `d = B²a/2 − Bb/2`
(b = λ − μ(m_c) the netput drift, a the netput variance rate). This gives
mean(dX)/dt = b and var(dX)/dt = a + O(1/B) exactly, at the cost of the
up/down jumps no longer being literal arrivals/services — they are the
diffusively rescaled netput increments. Rejected alternative: keeping
arrival/service semantics and inflating both rates by B²·const — it ties the
noise level to λ+μ and does not let `a` be an independent parameter, which
e00 needs. Consequence: nonnegativity of rates requires `B·a ≥ |b|`,
validated at config time against the worst-case drift.

**Service degradation form.** `μ(m_c) = μ0·(1 − drop·sigmoid((m_c − θ)/w))`
with defaults drop = 0.4, θ = 0.7, w = 0.08: bounded, smooth, steep beyond a
threshold — a broker-class mean-field coupling `m_c` = mean lag of the
class's own partitions. Rejected: a hard piecewise-linear cliff (kink would
show up in the density and pollute the T4 spectral question with a modeling
artifact) and an unbounded degradation (breaks the rate-nonnegativity bound).

**Boundary semantics.** Down-jumps at Q=0 and up-jumps at Q=B fire but leave
the state unchanged (rates are not zeroed at the walls); blocked up-jumps are
counted as rejected work — the discrete regulator. This is the reflected
random walk whose scaling limit is the reflected diffusion; zeroing the rates
at the walls instead changes the boundary local-time behavior and no longer
matches the Skorokhod-reflected limit.

**Tau-leaping.** Explicit tau-leap, rates frozen at step start; step
`τ = min(tau_dt_max, jump_cap / max(u+d), time to next sample)` with
jump_cap = 5 expected jumps per partition per step and tau_dt_max = 0.01 so
the mean-field coupling and MMPP switching stay resolved. Within a step,
services apply before arrivals and the excess over B is counted as rejected
(O(τ) boundary bias). Cross-validated distributionally against exact
Gillespie in `tests/test_sim.py` (time-averaged stationary-window Wasserstein
distance compared to seed-to-seed noise). Rejected: binomial tau-leap
(unneeded at these rates) and per-event vectorized Gillespie (still O(events)
Python-loop bound).

**Exact lattice histogram alongside the fixed grid.** Runs record both the
fixed n_bins grid on [0,1] (the specified first-class output) and the exact
counts on the lag lattice {0, 1/B, …, 1}. Reason: for B not a multiple of
n_bins, re-binning lattice atoms onto the fixed grid adds an O(1/n_bins)
Wasserstein artifact comparable to the effects e00 measures; distances are
computed from the lattice measure, plots use the fixed grid.

---

## 2026-08-02 — Fokker–Planck solver (B2)

**Scheme.** Conservative finite volume on a uniform grid with Chang–Cooper
exponential weighting of the advective interface flux, implicit Euler in time
(one tridiagonal solve per class per step), coefficients frozen at the step
start (semi-implicit in the mean-field coupling). Chang–Cooper is
positivity-preserving and exact on stationary exponential profiles — which is
precisely the closed-form validation case. With a = 0 the weighting
degenerates to first-order upwinding, so the same solver yields the transport
reference for e00's fluid contrast. Rejected: central differencing (loses
positivity at cell Péclet > 2, i.e. exactly in the near-wall boundary layer
we care about) and explicit stepping (CFL dt ~ h²/a is wasteful at the grid
resolutions e00 needs).

**Regulated boundary at x = 1.** The wall is mass-conserving (zero numerical
flux), matching the simulator, where saturated partitions remain in the
system at X = 1; the advective flux the wall cancels, max(b(1,m),0)·ρ(1,t),
is accumulated into K_B(t) as the PDE counterpart of the rejection counter.
Rejected: an absorbing/outflow boundary that removes the excess mass from the
domain — it drains total mass, which breaks the probability-density
comparison with the (mass-conserving) empirical measure and does not match
rejection semantics (a saturated partition stays saturated; its unserved
backlog is not destroyed).

**Convergence policy.** Implicit Euler is unconditionally stable but dt also
freezes the nonlinear coupling, so every production configuration is checked
by halving dt and doubling the grid together (`test_refinement_stability`
plus per-experiment checks) rather than relying on stability alone.

---

## 2026-08-02 — analysis library (B3)

**W1 from the lattice measure.** Wasserstein-1 uses scipy's CDF-based
implementation on weighted point supports, comparing the exact lag-lattice
empirical measure against PDE cell-center weights directly. Rejected:
comparing re-binned fixed-grid histograms — re-binning atoms adds an
O(1/n_bins) distance artifact of the same order as the convergence effects
under study.

**Spectral decay fit.** Cosine coefficients via DCT-II of cell-averaged
density values (the cosine basis diagonalizes the Neumann Laplacian, matching
reflecting walls); plain rFFT computed alongside to quantify the basis-choice
effect. Tail exponent s fitted by Theil–Sen regression of log|c_k| on log k
over a stated k-range, with its 95% CI. Rejected: ordinary least squares —
densities with near-symmetries produce sporadic near-zero coefficients whose
log values dominate an OLS fit; Theil–Sen is robust to them.

---

## 2026-08-02 — experiment e00 (B4)

**Sim↔PDE correspondence.** `streamfno.matching` maps a SimConfig to its
continuum problem: drift b(x,m) = E[λ] − μ0·g(m) with the same logistic
degradation g as the simulator, diffusion = the config's netput variance rate
a (diffusive mode) or 0 (fluid → transport), initial density = the truncated
Gaussian the simulator samples from. For MMPP the stationary mean rate is
used (exact only in the fast-switching limit) — e00/e01 use constant Poisson
netput so this does not affect them.

**Distances at single sample times, 5 seeds.** W1 is measured between the
instantaneous lattice measure and the PDE density at t ∈ {2,5,10,20,40}
(dt_sample = 1). Rejected: time-averaging the empirical measure before
comparing — it lowers the sampling floor but convolves the distance with the
temporal autocorrelation of the finite-N system, muddying the N-scaling that
T1 is about.

**Sweep engine.** tau-leap everywhere in the sweep (Gillespie at N=1000,
B=200 would be ~4·10^7 events × O(N) Python work per event); the engines are
cross-validated distributionally in tests/test_sim.py on a small config.
