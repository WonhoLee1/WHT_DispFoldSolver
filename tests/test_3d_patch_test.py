"""
test_3d_patch_test.py
=====================
TDD Test Suite for 3D Solid Elements & Mesh Architecture.
"""

import pytest
import numpy as np

from dispsolver.mesh3d import Mesh3D, Node3D, Element3D
from dispsolver.element3d import SolidElement3D, QuadraturePointState3D, Hexa8EASElement, Hexa8FbarElement
from dispsolver.solver3d import DynamicSolver3D


def test_mesh3d_creation():
    """Verify 3D mesh node/element instantiation and connectivity array generation."""
    mesh = Mesh3D()
    
    nodes_coords = [
        (1, 0.0, 0.0, 0.0),
        (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0),
        (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0),
        (6, 1.0, 0.0, 1.0),
        (7, 1.0, 1.0, 1.0),
        (8, 0.0, 1.0, 1.0),
    ]
    for nid, x, y, z in nodes_coords:
        mesh.add_node(nid, x, y, z)

    assert mesh.num_nodes == 8
    assert mesh.nodes_array().shape == (8, 3)

    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], "C3D8I")
    assert mesh.num_elements == 1
    assert mesh.elements_connectivity_array().shape == (1, 8)


def test_quadrature_point_state3d():
    """Verify initial 3D quadrature state factory creation."""
    state = QuadraturePointState3D.create_initial(num_prony=2)
    assert state.F.shape == (3, 3)
    assert np.allclose(state.F, np.eye(3))
    assert state.plastic_strain.shape == (6,)
    assert state.visco_overstress.shape == (2, 6)


def test_c3d8_eas_element_stiffness():
    """Verify Simo-Armero C3D8I EAS Element stiffness matrix symmetry and rank."""
    elem = Hexa8EASElement(num_eas_modes=9)
    assert elem.num_nodes == 8
    assert elem.num_quad_points == 8

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

    K_cond, f_int, alpha_opt = elem.compute_element_stiffness_and_force(
        node_coords, u_elem, C_mat, states
    )

    assert K_cond.shape == (24, 24)
    assert f_int.shape == (24,)
    assert alpha_opt.shape == (9,)
    assert np.allclose(K_cond, K_cond.T, atol=1e-8)

    rank = np.linalg.matrix_rank(K_cond, tol=1e-5)
    assert rank == 18


def test_3d_solver_uniaxial_tension_patch():
    """Verify End-to-End 3D Uniaxial Tension solution against Hooke's Law analytical solution."""
    mesh = Mesh3D()
    nodes_coords = [
        (1, 0.0, 0.0, 0.0),
        (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0),
        (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0),
        (6, 1.0, 0.0, 1.0),
        (7, 1.0, 1.0, 1.0),
        (8, 0.0, 1.0, 1.0),
    ]
    for nid, x, y, z in nodes_coords:
        mesh.add_node(nid, x, y, z)

    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], "C3D8I")

    E, nu = 200000.0, 0.3
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=False)

    # Fix face X=0 (nodes 1, 4, 5, 8) in X-direction
    for nid in [1, 4, 5, 8]:
        solver.fix_dof(nid, 0, 0.0)
    # Fix node 1 in Y and Z directions to prevent rigid body motion
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(1, 2, 0.0)
    solver.fix_dof(4, 2, 0.0)

    # Apply 1000 N tensile force in X-direction on face X=1 (nodes 2, 3, 6, 7)
    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    nid_map = mesh.node_id_to_index()
    for nid in [2, 3, 6, 7]:
        f_ext[3 * nid_map[nid] + 0] = 250.0  # 4 * 250 N = 1000 N total

    converged = solver.solve_static_step(f_ext, tol=1e-6)
    assert converged is True

    # Analytical displacement: delta_x = F_total / (A * E) = 1000.0 / (1.0 * 200000.0) = 0.005 mm
    expected_u_x = 1000.0 / (1.0 * 200000.0)
    
    # Check X displacement of loaded face nodes
    for nid in [2, 3, 6, 7]:
        u_x = solver.u[3 * nid_map[nid] + 0]
        assert pytest.approx(u_x, rel=1e-4) == expected_u_x
