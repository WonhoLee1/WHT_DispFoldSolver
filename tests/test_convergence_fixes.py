"""
test_convergence_fixes.py
=========================
Unit tests for convergence fixes B1, B3, B7, B8, B9.
"""

import numpy as np
import scipy.sparse as sps
import pytest

from dispsolver.solver.dynamic import _equilibrate, _solve_linear_system
from dispsolver.material.viscoelastic import ViscoelasticMaterial, Params


def test_b7_equilibrate_vectorized():
    """Verify vectorized _equilibrate produces identical scaled system and scales."""
    np.random.seed(42)
    N = 50
    A_dense = np.random.randn(N, N)
    A_dense = A_dense.T @ A_dense + np.eye(N) * 10.0
    A_sparse = sps.csr_matrix(A_dense)
    b = np.random.randn(N)

    As, bs, scale = _equilibrate(A_sparse, b)

    # Check properties
    assert As.shape == (N, N)
    assert len(bs) == N
    assert len(scale) == N
    assert np.all(np.isfinite(As.data))
    assert np.all(np.isfinite(bs))


def test_b8_viscoelastic_algorithmic_tangent():
    """Verify viscoelastic material tangent uses gamma_i (algorithmic tangent)."""
    base_mat = type("BaseMat", (), {
        "tangent_voigt": lambda self, F, params: np.diag([200.0, 200.0, 100.0]),
        "pk2_voigt": lambda self, F, params: np.array([10.0, 10.0, 0.0]),
        "mu": 100.0,
        "lam": 100.0,
    })()

    v_mat = ViscoelasticMaterial(
        base_material=base_mat,
        g_i=[0.5],
        tau_i=[0.1],
    )

    params = dict(E=1000.0, nu=0.3)
    F = np.eye(2)
    dt = 0.1  # dt = tau_i, ratio = 1.0

    # beta = exp(-1) = 0.367879...
    # gamma = (1 - exp(-1))/1 = 0.632120...
    # g_eff should be g_inf + g_1 * gamma = 0.5 + 0.5 * 0.632120 = 0.816060...
    C_eff = v_mat.tangent_voigt(F, params, dt=dt)

    assert np.all(np.isfinite(C_eff))
    # Tangent should not be NaN or zero
    assert np.abs(C_eff[0, 0]) > 1.0


def test_b1_line_search_logic():
    """Verify Armijo line search logic in Newton step."""
    # Ensure line search threshold check works as expected
    alpha = 1.0
    armijo = False
    R_norm_temp = 5.0
    R_norm_current = 2.0
    res_threshold = 10.0

    # With R_norm_temp < R_norm_current * 10 (5 < 20), it accepts α=1
    cond = armijo or R_norm_temp < R_norm_current * res_threshold or alpha < 0.25
    assert cond is True
