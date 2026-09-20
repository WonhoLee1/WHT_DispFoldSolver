"""
Tests for 3D Solver-Level Thermal Expansion and Constrained Thermal Stress.

Validates DynamicSolver3D with thermal adapter:
  1. Free thermal expansion of 3D C3D8 element matches exact theoretical displacement:
     delta_L = alpha * delta_T * L
  2. Uniaxially constrained thermal stress matches exact theoretical reaction force:
     R_x = -E * alpha * delta_T * Area
"""

import numpy as np
import pytest
from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.model.material import Material


def create_single_c3d8_cube_mesh(L: float = 1.0) -> Mesh3D:
    """Create a single 8-node brick element mesh spanning [0, L] x [0, L] x [0, L]."""
    mesh = Mesh3D()
    # 8 nodes
    mesh.add_node(1, 0.0, 0.0, 0.0)
    mesh.add_node(2, L,   0.0, 0.0)
    mesh.add_node(3, L,   L,   0.0)
    mesh.add_node(4, 0.0, L,   0.0)
    mesh.add_node(5, 0.0, 0.0, L)
    mesh.add_node(6, L,   0.0, L)
    mesh.add_node(7, L,   L,   L)
    mesh.add_node(8, 0.0, L,   L)

    # 1 element C3D8
    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], "C3D8", pid=0)
    return mesh


def test_3d_free_thermal_expansion():
    """Verify that unconstrained 3D cube expands by exactly alpha * delta_T * L."""
    L = 2.0  # mm
    alpha = 1.5e-5  # 1/K
    T0 = 20.0
    T = 120.0  # delta_T = 100.0 K
    E = 200000.0  # MPa
    nu = 0.3

    mesh = create_single_c3d8_cube_mesh(L=L)
    mat = Material(name="Steel_Thermal")
    mat.Elastic(E=E, nu=nu)
    mat.Expansion(type="ISO", table=((alpha, 20.0),), zero=T0)

    solver = DynamicSolver3D(
        mesh=mesh,
        materials={0: mat},
        time_integrator="quasistatic",
    )
    solver.set_temperature(T, ref_temp=T0)

    # Minimal BCs to prevent rigid body motion while allowing free expansion from origin:
    # Node 1 at (0, 0, 0): fix ux, uy, uz
    solver.fix_dof(1, 0, 0.0)
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(1, 2, 0.0)
    # Node 2 at (L, 0, 0): fix uy, uz (can expand in x)
    solver.fix_dof(2, 1, 0.0)
    solver.fix_dof(2, 2, 0.0)
    # Node 4 at (0, L, 0): fix ux, uz (can expand in y)
    solver.fix_dof(4, 0, 0.0)
    solver.fix_dof(4, 2, 0.0)
    # Node 5 at (0, 0, L): fix ux, uy (can expand in z)
    solver.fix_dof(5, 0, 0.0)
    solver.fix_dof(5, 1, 0.0)

    # Solve static equilibrium under thermal load
    conv, iters = solver.solve_step(dt=1.0, max_iters=20)
    assert conv, f"Solver failed to converge under thermal expansion (iters={iters})"
    u_sol = solver.u

    expected_disp = alpha * (T - T0) * L  # 1.5e-5 * 100 * 2.0 = 0.003 mm
    nid_map = mesh.node_id_to_index()

    # Node 7 is at (L, L, L), its displacement should be (expected_disp, expected_disp, expected_disp)
    idx7 = nid_map[7]
    u7_x = u_sol[3 * idx7 + 0]
    u7_y = u_sol[3 * idx7 + 1]
    u7_z = u_sol[3 * idx7 + 2]

    assert abs(u7_x - expected_disp) < 1e-6, f"u7_x expected {expected_disp}, got {u7_x}"
    assert abs(u7_y - expected_disp) < 1e-6, f"u7_y expected {expected_disp}, got {u7_y}"
    assert abs(u7_z - expected_disp) < 1e-6, f"u7_z expected {expected_disp}, got {u7_z}"


def test_3d_constrained_thermal_reaction_force():
    """Verify that uniaxially constrained cube generates exact reaction force R = -E * alpha * delta_T * Area."""
    L = 1.0  # mm (Area = 1.0 mm^2)
    alpha = 1.0e-5  # 1/K
    T0 = 0.0
    T = 100.0  # delta_T = 100.0 K
    E = 210000.0  # MPa
    nu = 0.0  # Set nu=0 for pure 1D uniaxial comparison

    mesh = create_single_c3d8_cube_mesh(L=L)
    mat = Material(name="Uniaxial_Thermal")
    mat.Elastic(E=E, nu=nu)
    mat.Expansion(type="ISO", table=((alpha, 0.0),), zero=T0)

    solver = DynamicSolver3D(
        mesh=mesh,
        materials={0: mat},
        time_integrator="quasistatic",
    )
    solver.set_temperature(T, ref_temp=T0)

    # Constrain both X=0 face (nodes 1, 4, 5, 8) and X=L face (nodes 2, 3, 6, 7) in X
    for nid in [1, 4, 5, 8, 2, 3, 6, 7]:
        solver.fix_dof(nid, 0, 0.0)

    # Pin node 1 in Y and Z to prevent rigid translation
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(1, 2, 0.0)

    conv, iters = solver.solve_step(dt=1.0, max_iters=20)
    assert conv, f"Solver failed to converge under constrained thermal stress (iters={iters})"
    u_sol = solver.u

    # Displacements in X must remain 0
    nid_map = mesh.node_id_to_index()
    for nid in range(1, 9):
        idx = nid_map[nid]
        assert abs(u_sol[3 * idx + 0]) < 1e-12

    # Compute reaction forces
    rf = solver.compute_reaction_forces()
    # Sum of reaction forces on X=0 face (nodes 1, 4, 5, 8)
    rf_x_face0 = sum(rf[3 * nid_map[nid] + 0] for nid in [1, 4, 5, 8])
    # Sum of reaction forces on X=L face (nodes 2, 3, 6, 7)
    rf_x_faceL = sum(rf[3 * nid_map[nid] + 0] for nid in [2, 3, 6, 7])

    # Expected thermal compressive stress: sigma = -E * alpha * delta_T = -210000 * 1e-5 * 100 = -210 MPa
    # Force on face 0 (left): pushes leftward (in -x direction on support, so reaction from wall is +210 N)
    # Force on face L (right): pushes rightward (reaction from wall is -210 N)
    expected_force = E * alpha * (T - T0) * (L * L)  # 210.0 N

    assert abs(rf_x_face0 - expected_force) < 1e-3, f"Expected {expected_force} N, got {rf_x_face0} N"
    assert abs(rf_x_faceL - (-expected_force)) < 1e-3, f"Expected {-expected_force} N, got {rf_x_faceL} N"
