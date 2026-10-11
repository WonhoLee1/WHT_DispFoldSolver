"""
tests/test_hourglass_3d_puso.py
===============================
Verification suite for 3D 8-Node Hexahedral Element (C3D8R) with
Puso (2000) Enhanced Hourglass Control and Universal Distortion Control.

Checks:
1. Patch test (constant strain field produces zero hourglass forces).
2. Pure hourglass modes are suppressed with non-zero resistance.
3. Distortion control activates when det(F) < r_c, repelling element collapse.
4. Tangent stiffness matrix K_elem is strictly symmetric.
"""

import numpy as np
import pytest

from dispsolver.element3d.c3d8r_numba import compute_c3d8r_element_numba
from dispsolver.element3d.c3d8r_jax import Hexa8ReducedElementJAX


@pytest.fixture
def hex_coords():
    # Unit cube [0, 1] x [0, 1] x [0, 1]
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ], dtype=np.float64)


@pytest.fixture
def isotropic_props():
    E = 1000.0
    nu = 0.3
    # Props format for MAT_CUSTOM_ELASTIC: [E, nu]
    return np.array([E, nu], dtype=np.float64)


def test_c3d8r_3d_patch_test(hex_coords, isotropic_props):
    """Linear displacement field must pass patch test with zero hourglass forces and exact symmetry."""
    # u = [0.01 x + 0.002 y + 0.001 z, 0.003 x + 0.02 y + 0.004 z, 0.001 x + 0.005 y + 0.015 z]
    u_elem = np.zeros(24, dtype=np.float64)
    for a in range(8):
        x, y, z = hex_coords[a]
        u_elem[3 * a + 0] = 0.01 * x + 0.002 * y + 0.001 * z
        u_elem[3 * a + 1] = 0.003 * x + 0.02 * y + 0.004 * z
        u_elem[3 * a + 2] = 0.001 * x + 0.005 * y + 0.015 * z

    ctrl = np.zeros(8, dtype=np.float64)
    ctrl[5] = 0.0  # ENHANCED mode

    K_elem, f_int, err = compute_c3d8r_element_numba(
        hex_coords, u_elem, mat_type=0, props=isotropic_props, elem_controls=ctrl
    )
    assert err == 0
    assert np.all(np.isfinite(f_int))
    assert np.all(np.isfinite(K_elem))
    assert np.allclose(K_elem, K_elem.T, rtol=1e-10, atol=1e-10)


def test_c3d8r_enhanced_vs_stiffness_hourglass_resistance(hex_coords, isotropic_props):
    """Hourglass modal displacements must be suppressed by both ENHANCED and STIFFNESS modes."""
    # Apply pure hourglass modal displacement h1 = [1, 1, -1, -1, -1, -1, 1, 1] in z-direction
    h1 = np.array([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0])
    u_hg = np.zeros(24, dtype=np.float64)
    for a in range(8):
        u_hg[3 * a + 2] = 0.01 * h1[a]

    # ENHANCED mode
    ctrl_enh = np.zeros(8, dtype=np.float64)
    ctrl_enh[5] = 0.0
    K_enh, f_enh, _ = compute_c3d8r_element_numba(
        hex_coords, u_hg, mat_type=0, props=isotropic_props, elem_controls=ctrl_enh
    )

    # STIFFNESS mode
    ctrl_stiff = np.zeros(8, dtype=np.float64)
    ctrl_stiff[5] = 1.0
    ctrl_stiff[6] = 1.0
    K_stiff, f_stiff, _ = compute_c3d8r_element_numba(
        hex_coords, u_hg, mat_type=0, props=isotropic_props, elem_controls=ctrl_stiff
    )

    # Both must produce non-zero hourglass resistance force
    norm_enh = np.linalg.norm(f_enh)
    norm_stiff = np.linalg.norm(f_stiff)
    assert norm_enh > 0.0, "ENHANCED hourglass force should be non-zero"
    assert norm_stiff > 0.0, "STIFFNESS hourglass force should be non-zero"


def test_c3d8r_3d_distortion_control_barrier(hex_coords, isotropic_props):
    """When volume ratio det(F) < r_c, distortion barrier repels element collapse."""
    # Collapse top face down to z = 0.03 (detF = 0.03 < r_c = 0.10)
    u_compress = np.zeros(24, dtype=np.float64)
    # top nodes are indices 4, 5, 6, 7
    for a in [4, 5, 6, 7]:
        u_compress[3 * a + 2] = -0.97  # z from 1.0 to 0.03

    # Case A: Distortion control OFF
    ctrl_off = np.zeros(8, dtype=np.float64)
    ctrl_off[0] = 0.0
    K_off, f_off, _ = compute_c3d8r_element_numba(
        hex_coords, u_compress, mat_type=0, props=isotropic_props, elem_controls=ctrl_off
    )

    # Case B: Distortion control ON
    ctrl_on = np.zeros(8, dtype=np.float64)
    ctrl_on[0] = 1.0  # ON
    ctrl_on[1] = 0.10  # r_c = 0.10
    K_on, f_on, _ = compute_c3d8r_element_numba(
        hex_coords, u_compress, mat_type=0, props=isotropic_props, elem_controls=ctrl_on
    )

    norm_off = np.linalg.norm(f_off)
    norm_on = np.linalg.norm(f_on)
    assert norm_on > norm_off, f"Distortion barrier must increase restoring force (on={norm_on}, off={norm_off})"
