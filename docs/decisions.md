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
