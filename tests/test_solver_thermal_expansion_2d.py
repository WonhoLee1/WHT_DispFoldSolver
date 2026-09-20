"""
test_solver_thermal_expansion_2d.py
===================================
Rigorous verification of 2D plane strain thermal expansion solver capabilities:
1. Free thermal expansion test: verifies in-plane expansion under plane strain constraint:
   eps_xx = eps_yy = (1 + nu) * alpha * delta_T
   For nu=0.3, alpha=1.5e-5, delta_T=100K: eps_xx = 1.3 * 1.5e-5 * 100 = 1.95e-3.
2. Uniaxially constrained thermal reaction force:
   For nu=0, verifies exact wall reaction force R = E * alpha * delta_T * Area.
"""

import numpy as np
import pytest

from dispsolver.mesh2d.mesh2d import Mesh2D, Node2D, Element2D
from dispsolver.model.material import Material
from dispsolver.solver2d.dynamic2d import DynamicSolver2D


def create_single_cpe4_quad_mesh(L: float = 1.0) -> Mesh2D:
    """Create a single 4-node quad element mesh in [0, L] x [0, L]."""
    mesh = Mesh2D()
    # 4 nodes
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, L, 0.0)
    mesh.add_node(3, L, L)
    mesh.add_node(4, 0.0, L)

    # 1 element
    mesh.add_element(1, [1, 2, 3, 4], "CPE4", pid=0)
    return mesh


def test_2d_free_thermal_expansion():
    """Verify that a free 2D quad expands exactly according to plane-strain thermal expansion:
    eps_xx = eps_yy = (1 + nu) * alpha * delta_T.
    """
    L = 2.0  # mm
    alpha = 1.5e-5  # 1/K
    T0 = 20.0
    T = 120.0  # delta_T = 100.0 K
    E = 4000.0  # MPa
    nu = 0.3

    mesh = create_single_cpe4_quad_mesh(L=L)
    mat = Material(name="Thermal_Polymer")
    mat.Elastic(E=E, nu=nu)
    mat.Expansion(type="ISO", table=((alpha, 0.0),), zero=T0)

    solver = DynamicSolver2D(
        mesh=mesh,
        materials={0: mat},
        time_integrator="quasistatic",
    )
    solver.set_temperature(T, ref_temp=T0)

    # Minimum BCs to prevent rigid body modes (RBM):
    # Node 1 at (0, 0): fix ux, uy
    solver.fix_dof(1, 0, 0.0)
    solver.fix_dof(1, 1, 0.0)
    # Node 2 at (L, 0): fix uy (can expand in x)
    solver.fix_dof(2, 1, 0.0)
    # Node 4 at (0, L): fix ux (can expand in y)
    solver.fix_dof(4, 0, 0.0)

    # Solve static equilibrium under thermal load
    conv, iters = solver.solve_step(dt=1.0, max_iters=20)
    assert conv, f"Solver failed to converge under 2D thermal expansion (iters={iters})"
    u_sol = solver.u

    # Under plane strain: eps_xx = eps_yy = (1 + nu) * alpha * delta_T
    expected_eps = (1.0 + nu) * alpha * (T - T0)
    expected_disp = expected_eps * L

    nid_map = mesh.node_id_to_index()
    # Node 3 is at (L, L), its displacement should be (expected_disp, expected_disp)
    idx3 = nid_map[3]
    u3_x = u_sol[2 * idx3 + 0]
    u3_y = u_sol[2 * idx3 + 1]

    assert abs(u3_x - expected_disp) < 1e-6, f"u3_x expected {expected_disp}, got {u3_x}"
    assert abs(u3_y - expected_disp) < 1e-6, f"u3_y expected {expected_disp}, got {u3_y}"


def test_2d_constrained_thermal_reaction_force():
    """Verify that uniaxially constrained 2D quad generates exact reaction force
    R = -E * alpha * delta_T * Area (with nu=0, unit thickness Area = L * 1.0).
    """
    L = 1.0  # mm
    alpha = 1.0e-5  # 1/K
    T0 = 0.0
    T = 100.0  # delta_T = 100.0 K
    E = 210000.0  # MPa
    nu = 0.0

    mesh = create_single_cpe4_quad_mesh(L=L)
    mat = Material(name="Uniaxial_Thermal_2D")
    mat.Elastic(E=E, nu=nu)
    mat.Expansion(type="ISO", table=((alpha, 0.0),), zero=T0)

    solver = DynamicSolver2D(
        mesh=mesh,
        materials={0: mat},
        time_integrator="quasistatic",
    )
    solver.set_temperature(T, ref_temp=T0)

    # Constrain X=0 edge (nodes 1, 4) and X=L edge (nodes 2, 3) in X
    for nid in [1, 4, 2, 3]:
        solver.fix_dof(nid, 0, 0.0)

    # Pin node 1 in Y to prevent rigid Y translation
    solver.fix_dof(1, 1, 0.0)

    conv, iters = solver.solve_step(dt=1.0, max_iters=20)
    assert conv, f"Solver failed to converge under constrained 2D thermal stress (iters={iters})"
    u_sol = solver.u

    # Displacements in X must remain 0
    nid_map = mesh.node_id_to_index()
    for nid in [1, 2, 3, 4]:
        idx = nid_map[nid]
        assert abs(u_sol[2 * idx + 0]) < 1e-12

    # Compute reaction forces
    rf = solver.compute_reaction_forces()
    rf_x_face0 = rf[2 * nid_map[1] + 0] + rf[2 * nid_map[4] + 0]

    # Expected force: E * alpha * delta_T * (L * thickness) where thickness=1.0 mm
    expected_force = E * alpha * (T - T0) * L  # 210000 * 1e-5 * 100 * 1 = 210.0 N
    assert abs(rf_x_face0 - expected_force) < 1e-3, f"Expected {expected_force} N, got {rf_x_face0} N"
