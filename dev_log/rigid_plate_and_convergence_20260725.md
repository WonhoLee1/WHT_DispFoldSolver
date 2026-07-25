# Work Log: Rigid Plate System & Solver Convergence Optimization

- **Date**: 2026-07-25
- **Task**: Implement Dual-Mode Rigid Body Plate Part & Surface-to-Surface Tie System + 9 Core Solver Convergence Fixes

## Work Accomplished

### 1. Solver Convergence Fixes (Part B)
- **B1 (Line Search Invalidation Bug)**: Removed `alpha == 1.0` automatic break in `dynamic.py:L1699`. Backtracking now properly executes on residual explosion.
- **B2 (PARDISO Factorization Reuse)**: Replaced repeated `pardiso_spsolve` calls in `_solve_linear_system()` with `PyPardisoSolver` instance factorization reuse (`factorize` once, `solve` multiple times during iterative refinement).
- **B3 (Ridge Regularization Fix)**: Lowered hardcoded `1e-4` ridge floor in `dynamic.py` to adaptive `1e-12 * min_nonzero_diag` to prevent artificial stiffness distortion in soft layers (e.g. PSA).
- **B7 (Equilibrate Vectorization)**: Replaced Python row loop in `_equilibrate()` with SciPy sparse matrix multiplication `_D @ A @ _D`.
- **B8 (Viscoelastic Algorithmic Tangent Fix)**: Corrected mathematical error in `viscoelastic.py` where `beta_i` was used instead of `gamma_i` in `g_eff`. Restored 2nd-order Newton convergence for viscoelastic material models.
- **B9 (UL Mode JAX Fallback State Loss Fix)**: Updated `compute_eas_j2_contributions` signature in `q4_eas.py` and `dynamic.py` to pass `F_n` and compute $F_{\text{total}} = F_{\text{inc}} F_n$ during JAX vmap NaN fallback.
- **B6 (Physics-Informed Dt Controller)**: Implemented `AdaptiveDtController` in `dispsolver/solver/dt_controller.py`.

### 2. Rigid Plate Part & Surface Tie Architecture (Part A)
- **`dispsolver/mesh/plate_builder.py`**: Created `create_folding_plate_parts()` to automatically build structured Q4 meshes, master RPs, and node sets for folding plates.
- **`dispsolver/part/rigid_body.py`**: Created `RigidBodyPart` class supporting dual-mode formulation (condensation vs. penalty constraint).
- **`dispsolver/constraint/surface_tie.py`**: Implemented penalty-based `SurfaceTieConstraint` coupling display bottom nodes to rigid plate top segments without stress concentration or element distortion.
- **`examples/ex11_rigid_plate_display_fold.py`**: Complete integrated example demonstrating 90° display folding.

## Testing & Verification
- `tests/test_convergence_fixes.py`: PASSED (3/3)
- `tests/test_rigid_plate_tie.py`: PASSED (3/3)
- `examples/ex11_rigid_plate_display_fold.py`: Running smoothly with PARDISO factorization reuse.
