"""
test_3d_locking_free.py
=======================
TDD Test Suite for 3D Incompressible Volumetric Locking-Free Formulations.
"""

import pytest
import numpy as np

from dispsolver.element3d import QuadraturePointState3D, Hexa8EASElement, Hexa8FbarElement


def test_c3d8_fbar_locking_free_incompressible():
    """Verify that Hexa8FbarElement prevents volumetric locking near the incompressible limit (nu -> 0.5)."""
    fbar_elem = Hexa8FbarElement()
    eas_elem = Hexa8EASElement(num_eas_modes=9)

    # 1x1x1 unit cube coordinates
    node_coords = np.array([
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
    states = [QuadraturePointState3D.create_initial() for _ in range(8)]

    # Nearly Incompressible Material: E = 200000 MPa, nu = 0.49999
    E, nu = 200000.0, 0.49999
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

    # Compute F-bar stiffness matrix
    K_fbar, f_fbar = fbar_elem.compute_element_stiffness_and_force(
        node_coords, u_elem, C_mat, states
    )

    # Compute EAS stiffness matrix
    K_eas, f_eas, _ = eas_elem.compute_element_stiffness_and_force(
        node_coords, u_elem, C_mat, states
    )

    # Both elements should be symmetric
    assert np.allclose(K_fbar, K_fbar.T, atol=1e-7)
    assert np.allclose(K_eas, K_eas.T, atol=1e-7)

    # Both elements must have rank 18 (24 - 6 rigid body modes)
    rank_fbar = np.linalg.matrix_rank(K_fbar, tol=1e-5)
    rank_eas = np.linalg.matrix_rank(K_eas, tol=1e-5)
    assert rank_fbar == 18
    assert rank_eas == 18
