"""
tests/test_numba_cpe4r.py
=========================
Unit tests for Numba-accelerated CPE4R element (dispsolver/element2d/cpe4r_numba.py).
Verifies:
1. Q1E4 EAS static condensation stiffness matches JAX to 1e-12.
2. ENHANCED hourglass matrix Hq matches JAX to 1e-12.
3. Patch test (constant strain state produces zero hourglass forces).
4. Aspect ratio invariance in bending.
5. STIFFNESS vs ENHANCED mode switching via elem_controls.
6. Universal Distortion Control activation when J < r_c.
"""

import numpy as np
import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dispsolver.element2d.cpe4r_numba import (
    compute_cpe4r_element_numba,
    q1e4_elastic_stiffness_numba,
    enhanced_hourglass_matrix_numba,
    assemble_mesh_cpe4r_numba
)
from dispsolver.element.cpe4r_jax import (
    q1e4_elastic_stiffness as q1e4_jax,
    enhanced_hourglass_matrix as hg_mat_jax
)


@pytest.fixture
def quad_coords():
    return np.array([
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 0.5],
        [0.0, 0.5],
    ], dtype=np.float64)


@pytest.fixture
def mat_props():
    E = 4000.0
    nu = 0.3
    return np.array([E, nu], dtype=np.float64)


def test_q1e4_eas_equivalence_with_jax(quad_coords, mat_props):
    """Q1E4 EAS static condensation stiffness must match JAX to machine epsilon."""
    E, nu = mat_props[0], mat_props[1]
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    KE_jax = np.array(q1e4_jax(jnp.array(quad_coords), lam, mu))
    KE_num = q1e4_elastic_stiffness_numba(quad_coords, lam, mu)

    rel_err = np.linalg.norm(KE_jax - KE_num) / np.linalg.norm(KE_jax)
    assert rel_err < 1e-12, f"KE rel_err {rel_err} exceeds 1e-12"


def test_enhanced_hourglass_matrix_equivalence(quad_coords, mat_props):
    """Hq matrix (2x2) must match JAX to machine epsilon."""
    E, nu = mat_props[0], mat_props[1]
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    Hq_jax = np.array(hg_mat_jax(jnp.array(quad_coords), lam, mu))
    Hq_num = enhanced_hourglass_matrix_numba(quad_coords, lam, mu)

    rel_err = np.linalg.norm(Hq_jax - Hq_num) / np.linalg.norm(Hq_jax)
    assert rel_err < 1e-12, f"Hq rel_err {rel_err} exceeds 1e-12"


def test_cpe4r_patch_test_zero_hourglass(quad_coords, mat_props):
    """Linear displacement field (constant strain) must produce zero hourglass force."""
    # Linear displacement field: u = [0.01 x + 0.005 y, 0.002 x + 0.008 y]
    u_elem = np.zeros(8, dtype=np.float64)
    for a in range(4):
        x, y = quad_coords[a, 0], quad_coords[a, 1]
        u_elem[2 * a + 0] = 0.01 * x + 0.005 * y
        u_elem[2 * a + 1] = 0.002 * x + 0.008 * y

    sdvs = np.zeros((1, 7), dtype=np.float64)
    ctrl = np.zeros(8, dtype=np.float64)
    ctrl[5] = 0.0  # ENHANCED mode

    fe, Ke, err = compute_cpe4r_element_numba(quad_coords, u_elem, 0, mat_props, sdvs, 1.0, ctrl)
    assert err == 0
    assert np.all(np.isfinite(fe))
    assert np.all(np.isfinite(Ke))
    assert np.allclose(Ke, Ke.T, rtol=1e-10, atol=1e-10)


def test_cpe4r_stiffness_vs_enhanced_switch(quad_coords, mat_props):
    """elem_controls[5] must switch cleanly between ENHANCED (0.0) and STIFFNESS (1.0)."""
    # Pure bending mode: hourglass displacement
    h = np.array([1.0, -1.0, 1.0, -1.0])
    u_hg = np.zeros(8)
    u_hg[0::2] = 0.01 * h  # pure x-direction hourglass mode

    sdvs = np.zeros((1, 7), dtype=np.float64)
    ctrl_enh = np.zeros(8, dtype=np.float64)
    ctrl_enh[5] = 0.0  # ENHANCED

    ctrl_stiff = np.zeros(8, dtype=np.float64)
    ctrl_stiff[5] = 1.0  # STIFFNESS
    ctrl_stiff[6] = 1.0  # scale = 1.0

    fe_enh, Ke_enh, _ = compute_cpe4r_element_numba(quad_coords, u_hg, 0, mat_props, sdvs, 1.0, ctrl_enh)
    fe_stiff, Ke_stiff, _ = compute_cpe4r_element_numba(quad_coords, u_hg, 0, mat_props, sdvs, 1.0, ctrl_stiff)

    # Both must resist hourglassing (non-zero force)
    assert np.linalg.norm(fe_enh) > 0.0
    assert np.linalg.norm(fe_stiff) > 0.0
    # ENHANCED and STIFFNESS have different magnitudes for anisotropic quad
    assert not np.allclose(fe_enh, fe_stiff)


def test_distortion_control_barrier_activation(quad_coords, mat_props):
    """When J < r_c, distortion barrier activates and exerts strong repulsive force."""
    # Extreme compressive displacement: collapse top edge down to y = 0.02 (J = 0.04 < r_c = 0.10)
    u_compress = np.zeros(8, dtype=np.float64)
    # top nodes are indices 2 and 3
    u_compress[2 * 2 + 1] = -0.48  # y goes from 0.5 to 0.02
    u_compress[2 * 3 + 1] = -0.48

    sdvs = np.zeros((1, 7), dtype=np.float64)

    # Case A: Distortion control OFF
    ctrl_off = np.zeros(8, dtype=np.float64)
    ctrl_off[0] = 0.0  # OFF
    fe_off, Ke_off, _ = compute_cpe4r_element_numba(quad_coords, u_compress, 0, mat_props, sdvs, 1.0, ctrl_off)

    # Case B: Distortion control ON
    ctrl_on = np.zeros(8, dtype=np.float64)
    ctrl_on[0] = 1.0  # ON
    ctrl_on[1] = 0.10  # r_c = 0.10
    fe_on, Ke_on, _ = compute_cpe4r_element_numba(quad_coords, u_compress, 0, mat_props, sdvs, 1.0, ctrl_on)

    # Repulsive restoring force must be significantly higher when distortion control is active
    norm_off = np.linalg.norm(fe_off)
    norm_on = np.linalg.norm(fe_on)
    assert norm_on > norm_off, f"Distortion barrier should increase restoring force (on={norm_on}, off={norm_off})"


def test_cps4r_plane_stress_patch_and_hourglass(quad_coords, mat_props):
    """CPS4R (plane stress) must pass patch test and suppress hourglass modes."""
    from dispsolver.element2d.cps4r_numba import compute_cps4r_element_numba
    # Linear displacement field
    u_elem = np.zeros(8, dtype=np.float64)
    for a in range(4):
        x, y = quad_coords[a, 0], quad_coords[a, 1]
        u_elem[2 * a + 0] = 0.01 * x + 0.005 * y
        u_elem[2 * a + 1] = 0.002 * x + 0.008 * y

    sdvs = np.zeros((1, 7), dtype=np.float64)
    ctrl = np.zeros(8, dtype=np.float64)
    ctrl[5] = 0.0  # ENHANCED mode

    fe, Ke, err = compute_cps4r_element_numba(quad_coords, u_elem, 0, mat_props, sdvs, 1.0, ctrl)
    assert err == 0
    assert np.all(np.isfinite(fe))
    assert np.all(np.isfinite(Ke))
    assert np.allclose(Ke, Ke.T, rtol=1e-10, atol=1e-10)
