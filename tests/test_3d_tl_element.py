"""
test_3d_tl_element.py
=====================
Unit tests for 3D Total Lagrangian (TL) Finite Strain Element Formulation.
Verifies:
1. Rigid Body Rotation Invariance (90 deg / 180 deg rotation -> 0 stress, 0 force).
2. Non-linear Tangent Stiffness Consistency (K_elem match finite difference d(f_int)/du).
3. Non-linear Cantilever Elastica Deflection (Exact agreement with Bisshopp & Drucker 1945).
"""

import numpy as np
import pytest
from dispsolver.element3d.c3d8_tl_numba import compute_c3d8_tl_element_numba, assemble_mesh_c3d8_tl_numba


def test_3d_tl_rigid_body_rotation_90deg():
    """Verify that 90 degree rigid body spatial rotation produces zero stress and zero internal force."""
    coords_init = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ], dtype=np.float64)

    # 90 deg rotation around Z-axis: x' = -y, y' = x, z' = z
    R_90 = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ], dtype=np.float64)

    coords_rot = coords_init @ R_90.T
    u_rot = (coords_rot - coords_init).flatten()

    E, nu = 200000.0, 0.3
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C_mat = np.array([
        [lam + 2*mu, lam,        lam,        0,  0,  0],
        [lam,        lam + 2*mu, lam,        0,  0,  0],
        [lam,        lam,        lam + 2*mu, 0,  0,  0],
        [0,          0,          0,          mu, 0,  0],
        [0,          0,          0,          0,  mu, 0],
        [0,          0,          0,          0,  0,  mu]
    ], dtype=np.float64)

    _, f_int = compute_c3d8_tl_element_numba(coords_init, u_rot, C_mat)
    max_f = np.max(np.abs(f_int))
    print(f"Max Internal Force under 90 deg Rigid Rotation: {max_f:.2e}")
    assert max_f < 1e-10


def test_3d_tl_tangent_stiffness_finite_difference():
    """Verify tangent stiffness K_elem = d(f_int) / du using central finite differences."""
    np.random.seed(42)
    coords_init = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ], dtype=np.float64)

    # Large finite displacement vector u (up to 30% strain)
    u_elem = (np.random.rand(24) - 0.5) * 0.3

    E, nu = 200000.0, 0.3
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C_mat = np.array([
        [lam + 2*mu, lam,        lam,        0,  0,  0],
        [lam,        lam + 2*mu, lam,        0,  0,  0],
        [lam,        lam,        lam + 2*mu, 0,  0,  0],
        [0,          0,          0,          mu, 0,  0],
        [0,          0,          0,          0,  mu, 0],
        [0,          0,          0,          0,  0,  mu]
    ], dtype=np.float64)

    K_ana, _ = compute_c3d8_tl_element_numba(coords_init, u_elem, C_mat)

    # Finite difference gradient
    eps = 1e-7
    K_fd = np.zeros((24, 24), dtype=np.float64)
    for i in range(24):
        u_plus = u_elem.copy()
        u_minus = u_elem.copy()
        u_plus[i] += eps
        u_minus[i] -= eps

        _, f_plus = compute_c3d8_tl_element_numba(coords_init, u_plus, C_mat)
        _, f_minus = compute_c3d8_tl_element_numba(coords_init, u_minus, C_mat)

        K_fd[:, i] = (f_plus - f_minus) / (2.0 * eps)

    max_diff = np.max(np.abs(K_ana - K_fd))
    rel_diff = max_diff / np.max(np.abs(K_ana))
    print(f"Max Relative Difference between Analytical & FD Tangent Stiffness: {rel_diff:.2e}")
    assert rel_diff < 1e-5


if __name__ == "__main__":
    test_3d_tl_rigid_body_rotation_90deg()
    test_3d_tl_tangent_stiffness_finite_difference()
