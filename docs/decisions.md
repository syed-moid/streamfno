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
