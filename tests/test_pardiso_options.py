"""
test_pardiso_options.py
=======================
Unit and regression tests for PARDISO mtype configurations,
matrix format conversion, symbolic factorization reuse, and adaptive fallback.
"""

import time
import numpy as np
import scipy.sparse as sps
import pytest

from dispsolver.solver.pardiso_manager import (
    PardisoNonlinearSolver,
    PARDISO_AVAILABLE,
    _resolve_mtype,
)


def _build_spd_matrix(n=100):
    """Generate a diagonally dominant, strictly SPD sparse matrix."""
    np.random.seed(42)
    main = 4.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    A = sps.diags([off, main, off], [-1, 0, 1], format="csr")
    # Add random symmetric connections
    extra = sps.random(n, n, density=0.02, format="csr")
    extra = 0.5 * (extra + extra.T)
    A = A + extra + sps.diags([np.full(n, 2.0)], [0], format="csr")
    return A.tocsr()


def _build_kkt_saddle_point_matrix(n_primal=80, n_dual=10):
    """Generate an indefinite KKT matrix with a zero dual block."""
    A = _build_spd_matrix(n_primal)
    np.random.seed(123)
    B = sps.random(n_dual, n_primal, density=0.1, format="csr")
    KKT = sps.bmat([[A, B.T], [B, None]], format="csr")
    return KKT


def test_pardiso_availability():
    """Verify PARDISO library is detected and loaded cleanly."""
    assert PARDISO_AVAILABLE, "Intel MKL PARDISO must be available in the test environment."


def test_mtype_resolution():
    """Verify string and int mtype resolving logic."""
    assert _resolve_mtype("auto", has_lagrange_multipliers=False) == 2
    assert _resolve_mtype("auto", has_lagrange_multipliers=True) == -2
    assert _resolve_mtype("spd") == 2
    assert _resolve_mtype("cholesky") == 2
    assert _resolve_mtype(2) == 2
    assert _resolve_mtype("indefinite") == -2
    assert _resolve_mtype("ldlt") == -2
    assert _resolve_mtype(-2) == -2
    assert _resolve_mtype("nonsymmetric") == 11
    assert _resolve_mtype("lu") == 11
    assert _resolve_mtype(11) == 11


def test_spd_upper_matrix_and_accuracy():
    """Verify that PardisoNonlinearSolver solves SPD matrices accurately
    and automatically extracts upper-triangular CSR without Access Violations.
    """
    if not PARDISO_AVAILABLE:
        pytest.skip("PARDISO not available")

    A = _build_spd_matrix(n=150)
    b = np.random.randn(150)

    # 1. Solve with mtype="spd" passing FULL matrix (auto upper extraction test)
    solver_spd = PardisoNonlinearSolver(mtype="spd", phase_reuse=False)
    x_spd = solver_spd.solve(A, b)
    res_spd = np.linalg.norm(A @ x_spd - b) / np.linalg.norm(b)
    assert res_spd < 1e-12, f"SPD residual too large: {res_spd}"

    # 2. Solve with mtype="nonsymmetric"
    solver_nonsym = PardisoNonlinearSolver(mtype="nonsymmetric", phase_reuse=False)
    x_nonsym = solver_nonsym.solve(A, b)
    res_nonsym = np.linalg.norm(A @ x_nonsym - b) / np.linalg.norm(b)
    assert res_nonsym < 1e-12, f"Nonsymmetric residual too large: {res_nonsym}"

    # Both solutions must agree to near machine precision
    diff = np.linalg.norm(x_spd - x_nonsym) / np.linalg.norm(x_spd)
    assert diff < 1e-12, f"Discrepancy between SPD and nonsymmetric solve: {diff}"


def test_phase_reuse_correctness():
    """Verify that reusing Phase 11 symbolic factorization produces
    IDENTICAL solutions across iterations with changing numerical values.
    """
    if not PARDISO_AVAILABLE:
        pytest.skip("PARDISO not available")

    A_base = _build_spd_matrix(n=120)
    b = np.random.randn(120)

    solver_reuse = PardisoNonlinearSolver(mtype="spd", phase_reuse=True)
    solver_full = PardisoNonlinearSolver(mtype="spd", phase_reuse=False)

    for step in range(10):
        # Simulate varying stiffness data across Newton iterations (preserving SPD)
        A_k = (1.0 + 0.05 * (step + 1)) * A_base


        x_reuse = solver_reuse.solve(A_k, b)
        x_full = solver_full.solve(A_k, b)

        diff = np.linalg.norm(x_reuse - x_full) / (np.linalg.norm(x_full) + 1e-30)
        assert diff < 1e-14, f"Iteration {step}: Phase reuse solution mismatch: {diff}"

    # Verify symbolic factorization was executed exactly ONCE
    assert solver_reuse._n_symbolic == 1, f"Expected 1 symbolic factor, got {solver_reuse._n_symbolic}"
    assert solver_reuse._n_solves == 10, f"Expected 10 solves, got {solver_reuse._n_solves}"


def test_indefinite_kkt_solve():
    """Verify solving saddle-point KKT system with mtype='indefinite'."""
    if not PARDISO_AVAILABLE:
        pytest.skip("PARDISO not available")

    KKT = _build_kkt_saddle_point_matrix(n_primal=100, n_dual=15)
    n_tot = KKT.shape[0]
    b = np.random.randn(n_tot)

    solver = PardisoNonlinearSolver(mtype="indefinite", phase_reuse=True)
    x = solver.solve(KKT, b, has_lagrange_multipliers=True)

    res = np.linalg.norm(KKT @ x - b) / np.linalg.norm(b)
    assert res < 1e-11, f"KKT indefinite residual too large: {res}"


def test_adaptive_fallback():
    """Verify that if SPD (mtype=2) is erroneously requested for an indefinite matrix,
    the solver gracefully falls back to indefinite or nonsymmetric without crashing.
    """
    if not PARDISO_AVAILABLE:
        pytest.skip("PARDISO not available")

    KKT = _build_kkt_saddle_point_matrix(n_primal=80, n_dual=10)
    b = np.random.randn(KKT.shape[0])

    # Request SPD for an indefinite matrix
    solver = PardisoNonlinearSolver(mtype="spd", phase_reuse=False)
    x = solver.solve(KKT, b)

    res = np.linalg.norm(KKT @ x - b) / np.linalg.norm(b)
    assert res < 1e-10, f"Fallback solve residual too large: {res}"


def test_phase_reuse_performance():
    """Measure speedup of Phase 11 reuse on multiple Newton-like iterations."""
    if not PARDISO_AVAILABLE:
        pytest.skip("PARDISO not available")

    # 2D grid elasticity matrix (approx 1250 DOFs)
    n = 25
    A1d = sps.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n, n), format="csr")
    A2d = sps.kron(sps.eye(n), A1d) + sps.kron(A1d, sps.eye(n))
    A2d = 0.5 * (A2d + A2d.T) + sps.diags([np.full(n * n, 1.0)], [0], format="csr")
    A2d = A2d.tocsr()
    b = np.random.randn(n * n)

    n_iters = 40

    # 1. Without phase reuse (re-orders with METIS every time)
    s_full = PardisoNonlinearSolver(mtype="spd", phase_reuse=False)
    s_full.solve(A2d, b)  # warm-up
    t0 = time.perf_counter()
    for step in range(n_iters):
        Ak = A2d.copy()
        Ak.data += 0.01 * step
        s_full.solve(Ak, b)
    t_full = time.perf_counter() - t0

    # 2. With phase reuse (Phase 11 once, Phase 23 per iter)
    s_opt = PardisoNonlinearSolver(mtype="spd", phase_reuse=True)
    s_opt.solve(A2d, b)  # warm-up (executes Phase 11)
    t0 = time.perf_counter()
    for step in range(n_iters):
        Ak = A2d.copy()
        Ak.data += 0.01 * step
        s_opt.solve(Ak, b)
    t_opt = time.perf_counter() - t0

    speedup = t_full / max(t_opt, 1e-6)
    print(f"\n[BENCHMARK] {n_iters} iterations: Full={t_full:.4f}s vs Reuse={t_opt:.4f}s (Speedup: {speedup:.2f}x)")
    assert speedup > 1.5, f"Expected at least 1.5x speedup, got {speedup:.2f}x"
