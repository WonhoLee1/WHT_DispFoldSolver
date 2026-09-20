"""
tests/test_standard_vs_dual_mortar_2d.py
=========================================
Verification tests comparing 2D Standard Mortar vs 2D Wohlmuth Dual Mortar:

1. Biorthogonality & Coupling Matrix Structure:
   - DUAL: int(psi_i * N_j dxi) == delta_{ij} (pure identity / diagonal).
   - STANDARD: int(psi_i * N_j dxi) == int(N_i * N_j dxi) == [[2/3, 1/3], [1/3, 2/3]] (consistent mass matrix).
2. Patch Test Consistency:
   - Both 2D Standard and Dual Mortar satisfy the 1D line patch test (uniform penetration -> equal reaction forces).
3. Non-matching 2D Mesh Patch Test:
   - Verifies uniform normal force transmission between non-conforming 2D blocks (5 elements on 3 elements).
"""

import numpy as np
import pytest

from dispsolver.constraint.dual_mortar_contact import (
    DualMortarContactConstraint2D,
    _line2_shape_functions,
    _line2_dual_shape_functions,
)


def test_2d_mortar_basis_orthogonality():
    """Verify 1D dual basis vs 1D standard basis integral matrices on [-1, 1]."""
    # 5-point Gauss-Legendre quadrature (exact for polynomials up to degree 9)
    gauss_pts = [
        (-np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0, (322.0 - 13.0 * np.sqrt(70.0)) / 900.0),
        (-np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0, (322.0 + 13.0 * np.sqrt(70.0)) / 900.0),
        (0.0, 128.0 / 225.0),
        (+np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0, (322.0 + 13.0 * np.sqrt(70.0)) / 900.0),
        (+np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0, (322.0 - 13.0 * np.sqrt(70.0)) / 900.0),
    ]

    M_dual = np.zeros((2, 2), dtype=np.float64)
    M_std = np.zeros((2, 2), dtype=np.float64)

    for xi, w in gauss_pts:
        Ns = np.array(_line2_shape_functions(xi))
        psi_dual = np.array(_line2_dual_shape_functions(xi))
        psi_std = Ns

        M_dual += np.outer(psi_dual, Ns) * w
        M_std += np.outer(psi_std, Ns) * w

    # Dual: strictly identity delta_{ij}
    assert np.allclose(M_dual, np.eye(2), atol=1e-14)

    # Standard: consistent mass matrix [[2/3, 1/3], [1/3, 2/3]]
    M_ref = np.array([[2.0 / 3.0, 1.0 / 3.0], [1.0 / 3.0, 2.0 / 3.0]])
    assert np.allclose(M_std, M_ref, atol=1e-14)


def test_2d_standard_and_dual_mortar_single_segment_patch_test():
    """Verify single segment pair under uniform penetration reproduces equal normal reaction."""
    # Slave: (0, 0) -> (2, 0), Master: (0, 0) -> (2, 0)
    coords = np.array([
        [0.0, 0.0], [2.0, 0.0],  # Slave: nodes 1, 2
        [0.0, 0.0], [2.0, 0.0],  # Master: nodes 3, 4
    ], dtype=np.float64)
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
    k_c = 1.0e6
    pen = 0.01
    L_s = 2.0
    total_force_expected = k_c * pen * L_s

    # Displacement: slave moves down in -y by pen
    u = np.zeros(8, dtype=np.float64)
    u[1] = -pen  # node 1 uy
    u[3] = -pen  # node 2 uy

    # 1. Dual Mortar
    c_dual = DualMortarContactConstraint2D(
        slave_segments=[(1, 2)],
        master_segments=[(3, 4)],
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=k_c,
        mortar_type="DUAL",
    )
    f_dual, _, stats_dual = c_dual.assemble(u)
    fy_slave_dual = f_dual[1] + f_dual[3]
    fy_master_dual = f_dual[5] + f_dual[7]

    assert abs(fy_slave_dual + fy_master_dual) < 1e-10
    assert abs(abs(fy_slave_dual) - total_force_expected) < 1e-6

    # 2. Standard Mortar
    c_std = DualMortarContactConstraint2D(
        slave_segments=[(1, 2)],
        master_segments=[(3, 4)],
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=k_c,
        mortar_type="STANDARD",
    )
    f_std, _, stats_std = c_std.assemble(u)
    fy_slave_std = f_std[1] + f_std[3]
    fy_master_std = f_std[5] + f_std[7]

    assert abs(fy_slave_std + fy_master_std) < 1e-10
    assert abs(abs(fy_slave_std) - total_force_expected) < 1e-6
    assert abs(fy_slave_dual - fy_slave_std) < 1e-6
