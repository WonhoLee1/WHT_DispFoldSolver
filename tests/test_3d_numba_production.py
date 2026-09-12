"""
test_3d_numba_production.py
============================
Production JIT Unit & Consistency Tests for Numba OpenMP 3D Solid Elements.
"""

import numpy as np
import pytest

from dispsolver.element3d import (
    Hexa8EASElement,
    Hexa8FbarElement,
    Tetra4Element,
    Tetra4ANPElement,
    Tetra10Element,
    compute_c3d8_eas_element_numba,
    compute_c3d8_fbar_element_numba,
    compute_c3d4_element_numba,
    compute_c3d4_anp_element_numba,
    compute_c3d10_element_numba,
    compute_c3d10m_element_numba,
    assemble_mesh_c3d8_eas_numba,
    assemble_mesh_c3d8_fbar_numba
)

from dispsolver.element3d.c3d8_numba import HAS_NUMBA

pytestmark = pytest.mark.skipif(not HAS_NUMBA, reason="Numba not installed")


def test_3d_eas_numba_kernel():
    """Verify 3D EAS Numba kernel against JAX reference formulation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ], dtype=np.float64)

    u_elem = np.zeros(24, dtype=np.float64)

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

    # 1. Numba kernel evaluation
    K_nb, f_nb, alpha_nb = compute_c3d8_eas_element_numba(coords, u_elem, C_mat)

    # 2. JAX reference evaluation
    eas_jax = Hexa8EASElement(num_eas_modes=9)
    from dispsolver.element3d import QuadraturePointState3D
    states = [QuadraturePointState3D.create_initial() for _ in range(8)]
    K_jax, f_jax, alpha_jax = eas_jax.compute_element_stiffness_and_force(coords, u_elem, C_mat, states)

    # 3. Exact match check
    assert np.allclose(K_nb, K_jax, atol=1e-10)
    assert np.allclose(f_nb, f_jax, atol=1e-10)
    assert np.allclose(alpha_nb, alpha_jax, atol=1e-10)


def test_3d_fbar_numba_kernel():
    """Verify 3D F-bar Numba kernel against JAX reference formulation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0]
    ], dtype=np.float64)

    u_elem = np.zeros(24, dtype=np.float64)

    E, nu = 200000.0, 0.499
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

    # 1. Numba kernel evaluation
    K_nb, f_nb = compute_c3d8_fbar_element_numba(coords, u_elem, C_mat)

    # 2. JAX reference evaluation
    fbar_jax = Hexa8FbarElement()
    from dispsolver.element3d import QuadraturePointState3D
    states = [QuadraturePointState3D.create_initial() for _ in range(8)]
    K_jax, f_jax = fbar_jax.compute_element_stiffness_and_force(coords, u_elem, C_mat, states)

    # 3. Exact match check
    assert np.allclose(K_nb, K_jax, atol=1e-10)
    assert np.allclose(f_nb, f_jax, atol=1e-10)


def test_3d_mesh_parallel_numba_assembly():
    """Verify parallel OpenMP mesh assembly for 3D EAS and F-bar Numba kernels."""
    n_x, n_y, n_z = 4, 4, 3
    n_nodes = (n_x + 1) * (n_y + 1) * (n_z + 1)
    node_coords = np.zeros((n_nodes, 3), dtype=np.float64)

    idx = 0
    for k in range(n_z + 1):
        for j in range(n_y + 1):
            for i in range(n_x + 1):
                node_coords[idx] = [i, j, k]
                idx += 1

    elem_conn = []
    for k in range(n_z):
        for j in range(n_y):
            for i in range(n_x):
                n0 = k * (n_x + 1) * (n_y + 1) + j * (n_x + 1) + i
                n1 = n0 + 1
                n3 = n0 + (n_x + 1)
                n2 = n3 + 1
                n4 = n0 + (n_x + 1) * (n_y + 1)
                n5 = n4 + 1
                n7 = n4 + (n_x + 1)
                n6 = n7 + 1
                elem_conn.append([n0, n1, n2, n3, n4, n5, n6, n7])

    elem_conn = np.array(elem_conn, dtype=np.int32)
    u_global = np.zeros(3 * n_nodes, dtype=np.float64)

    C_mat = np.eye(6, dtype=np.float64) * 1000.0

    f_eas, K_eas = assemble_mesh_c3d8_eas_numba(node_coords, elem_conn, u_global, C_mat)
    f_fbar, K_fbar = assemble_mesh_c3d8_fbar_numba(node_coords, elem_conn, u_global, C_mat)

    assert f_eas.shape == (48, 24)
    assert K_eas.shape == (48, 24, 24)
    assert f_fbar.shape == (48, 24)
    assert K_fbar.shape == (48, 24, 24)


def test_3d_anp_numba_kernel():
    """Verify 3D Tet4 ANP Numba kernel against JAX reference formulation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    u_elem = np.zeros(12, dtype=np.float64)
    C_mat = np.eye(6, dtype=np.float64) * 2000.0

    K_nb, f_nb = compute_c3d4_anp_element_numba(coords, u_elem, C_mat)

    anp_jax = Tetra4ANPElement()
    from dispsolver.element3d import QuadraturePointState3D
    states = [QuadraturePointState3D.create_initial()]
    K_jax, f_jax = anp_jax.compute_element_stiffness_and_force(coords, u_elem, C_mat, states)

    assert np.allclose(K_nb, K_jax, atol=1e-10)
    assert np.allclose(f_nb, f_jax, atol=1e-10)


def test_3d_c3d10_numba_kernel():
    """Verify 3D Tet10 C3D10 Numba kernel against JAX reference formulation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ], dtype=np.float64)

    u_elem = np.zeros(30, dtype=np.float64)
    C_mat = np.eye(6, dtype=np.float64) * 2000.0

    K_nb, f_nb = compute_c3d10_element_numba(coords, u_elem, C_mat)

    tet10_jax = Tetra10Element()
    from dispsolver.element3d import QuadraturePointState3D
    states = [QuadraturePointState3D.create_initial() for _ in range(4)]
    K_jax, f_jax = tet10_jax.compute_element_stiffness_and_force(coords, u_elem, C_mat, states)

    assert np.allclose(K_nb, K_jax, atol=1e-10)
    assert np.allclose(f_nb, f_jax, atol=1e-10)
