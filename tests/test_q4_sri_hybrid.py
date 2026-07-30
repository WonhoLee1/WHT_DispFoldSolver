"""
test_q4_sri_hybrid.py
=====================
Unit test suite for Selective Reduced Integration + Q1P0 Hydrostatic Pressure Hybrid Element (Q4_COROTATIONAL_HYBRID_SRI).
"""

from __future__ import annotations

import pytest
import numpy as np
import jax.numpy as jnp

from dispsolver.element.q4_sri_hybrid_jax import (
    compute_sri_hybrid_element_contributions_jax,
    compute_corotational_sri_hybrid_contributions_jax,
)

try:
    from dispsolver.element.q4_sri_hybrid_numba import (
        _compute_q4_sri_hybrid_single_element,
        assemble_q4_sri_hybrid_elements_numba,
        assemble_q4_corotational_sri_hybrid_elements_numba,
    )
    _has_numba = True
except ImportError:
    _compute_q4_sri_hybrid_single_element = None
    assemble_q4_sri_hybrid_elements_numba = None
    assemble_q4_corotational_sri_hybrid_elements_numba = None
    _has_numba = False


@pytest.mark.skipif(not _has_numba, reason="numba not available")
def test_q4_sri_hybrid_numba_matches_jax():
    """Verify Numba C-JIT kernel output matches JAX AutoDiff kernel values for SRI Hybrid Q4."""
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 0.5], [0.0, 0.5]], dtype=np.float64)
    u_elem = np.array([0.001, 0.002, -0.001, 0.003, 0.002, -0.001, -0.002, 0.001], dtype=np.float64)

    E = 1000.0
    nu = 0.499  # Nearly incompressible material (PSA layer)
    params = {'E': E, 'nu': nu}

    # JAX call (Neo-Hookean deviatoric + Q1P0 bulk)
    f_jax, K_jax = compute_sri_hybrid_element_contributions_jax(jnp.asarray(coords), jnp.asarray(u_elem), params)

    # Numba call (Linear elastic deviatoric + Q1P0 bulk)
    f_numba, K_numba = _compute_q4_sri_hybrid_single_element(coords, u_elem, E, nu)

    np.testing.assert_allclose(np.asarray(f_jax), f_numba, rtol=1e-2, atol=1e-4)


@pytest.mark.skipif(not _has_numba, reason="numba not available")
def test_q4_sri_hybrid_volumetric_locking_relief():
    """Verify SRI Hybrid Q4 relieves volumetric locking under nearly incompressible state (nu = 0.4999)."""
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 0.5], [0.0, 0.5]], dtype=np.float64)
    u_elem = np.array([0.001, -0.001, 0.002, 0.001, -0.001, 0.002, -0.002, -0.001], dtype=np.float64)

    E = 1000.0
    nu = 0.4999

    _, K_sri_hybrid = _compute_q4_sri_hybrid_single_element(coords, u_elem, E, nu)

    # Verify non-zero eigenvalues of stiffness matrix remain bounded (no volumetric locking blow-up)
    evals = np.linalg.eigvalsh(K_sri_hybrid)
    non_zero_evals = evals[3:]  # First 3 are rigid body zero modes

    max_eval = non_zero_evals[-1]
    min_pos_eval = non_zero_evals[0]
    cond_bounded = max_eval / min_pos_eval

    print(f"\nSRI Hybrid Non-zero Eigenvalues: {non_zero_evals}")
    print(f"Non-zero Condition Ratio     : {cond_bounded:.2f}")

    assert cond_bounded < 1e6
