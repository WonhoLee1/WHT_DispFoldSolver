"""
test_c3d8_hybrid.py
===================
Unit and Integration Tests for 3D Co-Rotational Hybrid Hexahedral Element (C3D8H).
"""

import pytest
import numpy as np

from dispsolver.element3d.c3d8_hybrid_numba import (
    compute_c3d8_hybrid_element_umat_numba,
    assemble_mesh_c3d8_hybrid_numba,
)
from dispsolver.material3d.numba_materials import MAT_HYPERELASTIC_NEOHOOKEAN
from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d import DynamicSolver3D


def test_c3d8_hybrid_locking_free_and_rank():
    """Verify C3D8H remains locking-free at incompressible limit (nu -> 0.49999, K/mu=10000)."""
    # 1x1x1 unit cube
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
    sdvs = np.zeros((8, 0), dtype=np.float64)

    # Incompressible hyperelastic parameters: mu = 0.2 MPa, K = 2000.0 MPa (K/mu = 10000)
    props = np.zeros(36, dtype=np.float64)
    props[0] = 0.2     # mu
    props[1] = 2000.0  # K

    f_elem, K_elem, err = compute_c3d8_hybrid_element_umat_numba(
        coords, u_elem, MAT_HYPERELASTIC_NEOHOOKEAN, props, sdvs
    )

    assert err == 0
    assert np.allclose(f_elem, 0.0, atol=1e-12)
    # Symmetry check
    assert np.allclose(K_elem, K_elem.T, atol=1e-7)

    # Rank check: exactly 18 physical deformation modes, 6 rigid body modes
    rank = np.linalg.matrix_rank(K_elem, tol=1e-4)
    assert rank == 18, f"Expected rank 18, got {rank}"


def test_c3d8_hybrid_rigid_body_rotation_invariance():
    """Verify C3D8H yields zero internal forces under pure finite rigid body rotation."""
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

    # Rotate 45 degrees around Z axis
    theta = np.radians(45.0)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ])

    u_elem = np.zeros(24, dtype=np.float64)
    for i in range(8):
        x_rot = R @ coords[i]
        u_elem[3*i : 3*i+3] = x_rot - coords[i]

    props = np.zeros(36, dtype=np.float64)
    props[0] = 0.2
    props[1] = 200.0
    sdvs = np.zeros((8, 0), dtype=np.float64)

    f_elem, K_elem, err = compute_c3d8_hybrid_element_umat_numba(
        coords, u_elem, MAT_HYPERELASTIC_NEOHOOKEAN, props, sdvs
    )

    assert err == 0
    # Internal forces under pure rotation must vanish
    assert np.allclose(f_elem, 0.0, atol=1e-8)


def test_c3d8_hybrid_heterogeneous_assembly_in_solver3d():
    """Verify DynamicSolver3D solves a sandwich mesh with mixed C3D8_CR (PET) and C3D8H (PSA)."""
    mesh = Mesh3D()
    # 2 Hex elements stacked vertically
    # Nodes 1..8 for element 1 (bottom), 5..12 for element 2 (top)
    coords = [
        (1, 0, 0, 0), (2, 1, 0, 0), (3, 1, 1, 0), (4, 0, 1, 0),
        (5, 0, 0, 1), (6, 1, 0, 1), (7, 1, 1, 1), (8, 0, 1, 1),
        (9, 0, 0, 2), (10, 1, 0, 2), (11, 1, 1, 2), (12, 0, 1, 2)
    ]
    for nid, x, y, z in coords:
        mesh.add_node(nid, float(x), float(y), float(z))

    # E1: Bottom layer is PET with C3D8_CR
    e1 = mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], "C3D8_CR")
    e1.pid = 0

    # E2: Top layer is PSA with C3D8H
    e2 = mesh.add_element(2, [5, 6, 7, 8, 9, 10, 11, 12], "C3D8H")
    e2.pid = 1

    materials = {
        0: {"type": "j2_plasticity", "E": 4000.0, "nu": 0.3, "sigma_y0": 80.0, "H": 400.0},
        1: {"type": "neo_hookean", "C10": 0.1, "D1": 0.01}
    }

    solver = DynamicSolver3D(mesh, materials=materials)

    # Fix bottom face nodes (1, 2, 3, 4) in all DOFs
    for nid in [1, 2, 3, 4]:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)

    # Apply 0.1 mm shear displacement in X on top face nodes (9, 10, 11, 12)
    for nid in [9, 10, 11, 12]:
        solver.fix_dof(nid, 0, 0.1)

    conv, iters = solver.solve_step(dt=0.1, max_iters=15)
    assert conv, f"Heterogeneous solve failed to converge (iters={iters})"
    assert iters <= 5
