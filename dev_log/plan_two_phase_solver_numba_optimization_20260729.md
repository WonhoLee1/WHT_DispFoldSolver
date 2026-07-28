# Plan: Two-Phase Solver Acceleration & Numba Backend Integration — 2026-07-29

**Date**: 2026-07-29  
**Status**: APPROVED by user. Ready for execution.

---

## 1. Goal Description & Architecture Overview

To maximize `dispsolver` performance while maintaining theoretical correctness, we implement a two-phase optimization roadmap:

1. **Phase 1: Solver-Level Acceleration (PARDISO Caching, Tangent Freezing, RCM)**
   - Targets the **70% time bottleneck** spent in sparse linear system solves ($K \Delta u = -R$).
   - Yields **40% – 55% total runtime reduction** for **all** backends (JAX, Numba, NumPy).
2. **Phase 2: Numba CPU Backend Integration (`q4_numba.py`)**
   - Implements multi-threaded Numba `@njit` kernels for feasible element types (`Q4_BBAR`, `RBE2`, `Q4_EAS`).
   - Provides a fast CPU alternative with **zero JAX JIT initial tracing delay**.
   - Integrates 3-way speed benchmark (CPython vs Numba vs JAX).

---

## 2. Phase 1 Technical Specification (Solver Acceleration)

### [MODIFY] `dispsolver/solver/dynamic.py`

#### A. PARDISO Factorization Caching
- Reuse `PyPardisoSolver` handle when matrix non-zero sparsity pattern `sparsity_pattern_id` remains unchanged.
- Call PARDISO with `phase=23` (Numerical Factorization + Solve) skipping `phase=11` (Symbolic Analysis), saving 20–30% of linear solve time.

```python
# Caching logic in DynamicSolver:
if self._pardiso_cached_pattern_id == sparsity_id:
    du = self._pardiso_solver.solve(K_free, R_free, phase=23)
else:
    self._pardiso_solver = PyPardisoSolver()
    du = self._pardiso_solver.solve(K_free, R_free, phase=13)
    self._pardiso_cached_pattern_id = sparsity_id
```

#### B. Tangent Stiffness Freezing (Modified Newton-Raphson)
- Add solver options: `freeze_tangent_threshold: float = 0.1` and `max_frozen_iters: int = 2`.
- When Newton residual ratio $\|R_k\| / \|R_{k-1}\| < 0.1$, skip `_assemble()` for iteration $k+1$.
- Perform forward/back-substitution solve (`phase=33`) with cached $K_{\text{factored}}$, cutting Newton step cost in half during smooth equilibrium phases.

#### C. Reverse Cuthill-McKee (RCM) Bandwidth Minimization
- Compute `perm = scipy.sparse.csgraph.reverse_cuthill_mckee(K)` during initialization.
- Reorders DOFs to minimize sparse matrix bandwidth, reducing memory footprint and factorization cost by 15–25%.

---

## 3. Phase 2 Technical Specification (Numba Backend)

### [NEW] `dispsolver/element/q4_numba.py`
Multi-threaded Numba `@njit` assembly loops for:
1. `Q4_BBAR`: 2x2 Gauss quadrature & B-bar SRI assembly.
2. `RBE2`: Kinematic rigid body master-slave condensation mapping.
3. `Q4_EAS`: 4-mode internal strain static condensation.

### [MODIFY] `verification/element_backends.py` & `verification/speed_bench.py`
Register `numba_q4_bbar` and update `--include-speed` to output a 3-way performance comparison table (`CPython` vs `Numba` vs `JAX`).

---

## 4. Verification Plan

1. **Automated Unit Tests**:
   ```powershell
   pytest tests/ -q
   ```
   Must pass baseline (145 passed, 1 xfailed).

2. **Verification Suite**:
   ```powershell
   python -u -m verification.run_all --include-speed
   ```
   Must achieve 12/12 PASS and generate updated `speed_report.md`.

3. **End-to-End Fold Verification**:
   Run `examples/ex12_abaqus_inp_plate_fold.py` and confirm zero regressions in final U-bend fold geometry.
