"""
test_cr_eas_element.py
======================
Unit tests for Co-rotational EAS (CR-EAS) elements in NumPy, JAX, and Numba.

Verifies:
1. Patch test (rigid body rotation produces zero internal force).
2. Large rotation invariance across NumPy, JAX, and Numba backends.
"""

import numpy as np
import pytest

from dispsolver.element.q4_corotational_eas import compute_corotational_eas_j2_contributions
from dispsolver.element.q4_corotational_eas_jax import compute_corotational_eas_j2_contributions_jax
from dispsolver.element.q4_numba import compute_q4_corotational_eas_element, HAS_NUMBA
from dispsolver.material import J2Plasticity


def test_cr_eas_patch_test_numpy():
    """Verify rigid body rotation produces zero internal force in CR-EAS (NumPy)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]], dtype=np.float64)
    theta = np.radians(30.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords_rot = coords @ R.T
    u_rigid = (coords_rot - coords).flatten()

    mat = J2Plasticity(E=1000.0, nu=0.3, sigma_y0=1e9, H=0.0)
    params = {'E': 1000.0, 'nu': 0.3, 'sigma_y0': 1e9, 'H': 0.0}
    alpha_zero = np.zeros(4, dtype=np.float64)
    state_init = np.tile(mat.initial_internal_vars(), (4, 1))

    f_int, K, alpha_new, _ = compute_corotational_eas_j2_contributions(
        coords, u_rigid, alpha_zero, state_init, mat, params
    )

    np.testing.assert_allclose(f_int, 0.0, atol=1e-10)


def test_cr_eas_patch_test_jax():
    """Verify rigid body rotation produces zero internal force in CR-EAS (JAX)."""
    import jax.numpy as jnp
    coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]], dtype=jnp.float64)
    theta = np.radians(45.0)
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    coords_rot = coords @ R.T
    u_rigid = (coords_rot - coords).flatten()
    alpha_zero = jnp.zeros(4, dtype=jnp.float64)
    state_init = jnp.tile(jnp.array([1.0, 0.0, 0.0, 1.0, 0.0]), (4, 1))

    f_int, K, alpha_new, _, _ = compute_corotational_eas_j2_contributions_jax(
        coords, u_rigid, alpha_zero, state_init, lam=576.92, mu=384.615, sigma_y0=1e9, H=0.0
    )

    np.testing.assert_allclose(np.asarray(f_int), 0.0, atol=1e-10)


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")
def test_cr_eas_patch_test_numba():
    """Verify rigid body rotation produces zero internal force in CR-EAS (Numba)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.0, 0.5]], dtype=np.float64)
    theta = np.radians(60.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    coords_rot = coords @ R.T
    u_rigid = (coords_rot - coords).flatten()

    f_int, K = compute_q4_corotational_eas_element(coords, u_rigid, E=1000.0, nu=0.3)
    np.testing.assert_allclose(f_int, 0.0, atol=1e-10)
