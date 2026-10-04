"""Unsymmetric tangents must reach Pardiso's LU path (mtype 11).

Until 2026-10-04 DynamicSolver3D._is_symmetric returned numpy.bool_ and the
mtype resolver tested `is_symmetric is False`, which is never true for
np.False_. Every unsymmetric tangent therefore went to the SPD path, failed
Cholesky and was refactored as symmetric indefinite from its upper triangle:
a silently wrong Newton direction (linear residual up to ~10 on deck #06 with
the Nitsche contact tangent).
"""
import numpy as np
import scipy.sparse as sps

from dispsolver.solver.pardiso_manager import PardisoNonlinearSolver, _resolve_mtype
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


def test_resolver_accepts_numpy_bool():
    assert _resolve_mtype("auto", False, is_symmetric=np.False_) == 11
    assert _resolve_mtype("auto", False, is_symmetric=False) == 11
    assert _resolve_mtype("auto", False, is_symmetric=np.True_) == 2
    assert _resolve_mtype("auto", False, is_symmetric=None) == 2


def test_is_symmetric_returns_python_bool():
    A = sps.csr_matrix(np.array([[4.0, 1.0], [0.0, 3.0]]))
    assert DynamicSolver3D._is_symmetric(A) is False
    assert DynamicSolver3D._is_symmetric((A + A.T).tocsr()) is True


def test_unsymmetric_system_solved_exactly():
    rng = np.random.default_rng(0)
    n = 60
    dense = rng.standard_normal((n, n)) * 0.1 + np.diag(np.full(n, 5.0))
    dense[np.abs(dense) < 0.05] = 0.0
    A = sps.csr_matrix(dense)
    b = rng.standard_normal(n)
    sym = DynamicSolver3D._is_symmetric(A)
    assert sym is False
    x = PardisoNonlinearSolver().solve(A, b, is_symmetric=sym)
    assert np.linalg.norm(A @ x - b) <= 1e-10 * np.linalg.norm(b)
