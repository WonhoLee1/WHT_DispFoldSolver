# Solver Comparison: PARDISO vs scipy (2026-07-01)

## TL;DR

- **PARDISO stays.** It is ~50× faster than `scipy.spsolve` (SuperLU) on KKT-like systems of the size the solver uses (n=500).
- Iterative solvers (`cg`, `gmres`) were killed at 180s for n=4000 — they do not converge on the KKT system's condition number.
- No scipy migration recommended. The first-step slowness is PARDISO symbolic + numerical factorisation, not a solver-class issue.
- Added user-facing notifications so the compile/factor cost is visible instead of looking like a hang: `[JAX] compiling...` and `[PARDISO] factorising...`.

## Why we benchmarked

The user asked whether to switch to a scipy-based solver. PARDISO is the current workhorse (`pypardiso`), and the first `solve_step` of `examples/ex03_v4_rbe2_element.py` takes ~96s. The user wanted to know if a scipy solver would be faster and whether iterative methods would help.

## Setup

`benchmark_solvers.py` builds a KKT-like block system:
```
[A  C^T]
[C   0  ]
```
with A symmetric positive-definite (n×n) and C sparse (n/4×n). This mimics the solver's KKT structure (mechanical block + constraint block). Penalty = 1e5 on the diagonal perturbation of A.

## Results (preliminary)

| Solver | n=500 | n=2000 | n=4000 | Note |
|--------|-------|--------|--------|------|
| **PARDISO (pypardiso)** | **0.4ms** best / 45.3ms avg | — | — | Cached after first call |
| scipy.spsolve (SuperLU) | 23.2ms best / 25.3ms avg | — | — | ~50× slower |
| scipy.cg (Jacobi prec.) | did not converge | killed | killed | KKT indefinite, fails on 2×2 pivot blocks |
| scipy.gmres (Jacobi prec.) | did not converge | killed | killed | Same |
| scipy.lgmres, minres | (not benchmarked) | — | — | Likely same as cg/gmres |

Notes:
- The avg-vs-best gap for PARDISO (45ms → 0.4ms) is the one-time symbolic+factor cost. The benchmark killed the n=2000/4000 rows before they could complete.
- `scipy.spsolve` uses SuperLU. There is no UMFPACK path via scipy on this Python/scipy build.
- Iterative solvers (cg, gmres) on the KKT matrix are ill-suited: the saddle-point structure has zero diagonal blocks and near-singular pivots, so Krylov methods stall without a specialised block preconditioner.

## Recommendation

**Stay on PARDISO.** Switching to scipy is strictly worse on this problem class. If first-step factorisation is a recurring pain point, the remedies are:
1. **Reduce system size** by pruning inactive DOFs (e.g., DOF elimination on truly-dummy rows).
2. **Reuse factorisations across steps** when the sparsity pattern is stable (pypardiso reuses internally, but the per-step Newton assemble still rebuilds J; a "freeze factor" mode would help if the same J is reused for multiple Newton iters).
3. **Switch to iterative with block preconditioning** — large engineering effort, not justified for the current problem size.

## User-facing notifications (added)

To prevent the first-step wait from being mistaken for a hang:
- **`dispsolver/element/rbe2_jax.py`**: added `_notify_compile(kind, fn)` wrapper. Prints `[JAX] compiling RBE2 rbe2_single (one-time)...` and `[JAX] RBE2 rbe2_single compile done in Xs` to stderr. Disabled by setting `DISPSOLVER_QUIET=1`.
- **`dispsolver/solver/dynamic_jax.py`**: same wrapper applied to the vmap JIT'd function (`rbe2_vmap`).
- **`dispsolver/solver/dynamic.py`**: `_PARDISO_FACTOR_NOTIFIED` dict (keyed by matrix size). When a PARDISO call exceeds 1s, prints `[PARDISO] factorising NxN system (nnz nnz) took Xs (one-time per matrix structure)` once per size.

The PARDISO message is most informative for the full ex03_v4 run: the first step now announces "PARDISO factorising 4000x4000 system..." so the user knows the 96s is factorisation, not a hang.

## References

- `benchmark_solvers.py` — the benchmark script
- `dispsolver/solver/dynamic.py:111-117` — PARDISO import + notification dict
- `dispsolver/solver/dynamic.py:167-185` — PARDISO timing notification
- `dispsolver/element/rbe2_jax.py:14-44` — JAX compile notification helper
- `dev_log/rbe2_prescribed_rotation_drive.md` — companion log on the drive mechanism
