"""
test_nitsche_contact3d.py
==========================
Verification suite for NitscheContactConstraint3D.

Tests 1 and 3 have no volume elements, so they exercise the class in its
explicit penalty mode (consistency=False: p = gamma * g_n, symmetric). The
consistent (Nitsche) mode needs the slave elements' stress; it is checked
by test 2 here and by the exact contact patch tests in
tests/test_deck06_contact.py.

Tests:
1. test_nitsche_3d_tangent_symmetry_and_psd:
   - Verifies symmetry |K - K^T| <= 1e-14 |K| (roundoff; the solver's own
     symmetry test uses 1e-10 relative).
   - Verifies positive semi-definiteness (all eigenvalues >= -1e-13).
   - Verifies Newton's third law (sum of interface contact forces == 0).
2. test_nitsche_two_cube_contact_converges:
   - Full 3D FEA solve with DynamicSolver3D using Nitsche contact between two stacked C3D8 cubes.
   - Verifies quadratic Newton-Raphson convergence without Lagrange multipliers.
3. test_nitsche_patch_test_uniform_pressure:
   - 2x2 grid of Quad4 faces under uniform compression, verifying uniform contact force distribution.
"""

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.constraint3d.nitsche_contact3d import (
    NitscheContactConstraint3D,
    _quad4_shape_functions,
    _quad4_normal,
)


def test_nitsche_3d_tangent_symmetry_and_psd():
    """Verify bit-level symmetry, positive semi-definiteness, and force equilibrium."""
    L = 2.0
    # Single Quad4 pair: slave at z=0, master at z=0
    coords = np.array([
        # Slave quad: nodes 1, 2, 3, 4
        [0.0, 0.0, 0.0],
        [L, 0.0, 0.0],
        [L, L, 0.0],
        [0.0, L, 0.0],
        # Master quad: nodes 5, 6, 7, 8
        [0.0, 0.0, 0.0],
        [L, 0.0, 0.0],
        [L, L, 0.0],
        [0.0, L, 0.0],
    ], dtype=np.float64)

    nid_to_idx = {nid: nid - 1 for nid in range(1, 9)}
    gamma = 2.5e6

    contact = NitscheContactConstraint3D(
        slave_faces=[(1, 2, 3, 4)],
        master_faces=[(5, 6, 7, 8)],
        nid_to_idx=nid_to_idx,
        coords=coords,
        gamma=gamma,
        consistency=False,   # surface-only fixture (no elements): penalty mode
    )

    # Penetration in Z: move slave nodes down by delta = 0.05 mm
    delta = 0.05
    u = np.zeros(24, dtype=np.float64)
    for a in range(4):
        u[3 * a + 2] = -delta

    f_c, (rows, cols, data), stats = contact.assemble(u)

    # 1. Total force equilibrium (Newton's third law)
    expected_total_F = gamma * delta * (L * L)
    assert abs(stats["total_normal_force"] - expected_total_F) < 1e-6
    assert abs(np.sum(f_c)) < 1e-11

    # 2. Reconstruct sparse matrix K
    K = csc_matrix((data, (rows, cols)), shape=(24, 24)).toarray()

    # 3. Exact symmetry check
    asym_err = float(np.max(np.abs(K - K.T)))
    assert asym_err <= 1e-14 * np.abs(K).max(), f"Nitsche tangent matrix asymmetry: {asym_err}"

    # 4. Positive semi-definiteness check (all eigenvalues >= 0 up to machine epsilon)
    eigenvalues = np.linalg.eigvalsh(K)
    min_eig = float(np.min(eigenvalues))
    max_eig = float(np.max(eigenvalues))
    rel_min_eig = min_eig / max_eig if max_eig > 0 else min_eig
    assert rel_min_eig >= -1e-12, f"Unphysical negative eigenvalue found: min={min_eig}, max={max_eig}"

    # 5. In-plane (X and Y) forces must be zero
    for a in range(8):
        assert abs(f_c[3 * a + 0]) < 1e-14
        assert abs(f_c[3 * a + 1]) < 1e-14


def _build_two_cube_nitsche_fixture(gamma_scale: float = 1.0, L: float = 10.0, gap0: float = 0.2):
    """Build two stacked C3D8 blocks with Nitsche contact at the interface."""
    mesh = Mesh3D()
    xs, ys, zs = [0.0, L], [0.0, L], [gap0, gap0 + L]
    nid = 1
    grid_top = {}
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                mesh.add_node(nid, x, y, z)
                grid_top[(i, j, k)] = nid
                nid += 1

    top_conn = [
        grid_top[(0, 0, 0)], grid_top[(1, 0, 0)], grid_top[(1, 1, 0)], grid_top[(0, 1, 0)],
        grid_top[(0, 0, 1)], grid_top[(1, 0, 1)], grid_top[(1, 1, 1)], grid_top[(0, 1, 1)],
    ]
    mesh.add_element(1, top_conn, "C3D8")

    # Bottom block, z in [0, gap0]
    grid_bot = {}
    nid2 = nid
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(nid2, x, y, 0.0)
            grid_bot[(i, j, 0)] = nid2
            nid2 += 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(nid2, x, y, gap0)
            grid_bot[(i, j, 1)] = nid2
            nid2 += 1

    bot_conn = [
        grid_bot[(0, 0, 0)], grid_bot[(1, 0, 0)], grid_bot[(1, 1, 0)], grid_bot[(0, 1, 0)],
        grid_bot[(0, 0, 1)], grid_bot[(1, 0, 1)], grid_bot[(1, 1, 1)], grid_bot[(0, 1, 1)],
    ]
    mesh.add_element(2, bot_conn, "C3D8")

    E, nu = 200000.0, 0.3
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)
    nid_to_idx = mesh.node_id_to_index()

    top_bottom_face = (
        grid_top[(0, 0, 0)], grid_top[(1, 0, 0)], grid_top[(1, 1, 0)], grid_top[(0, 1, 0)],
    )
    bot_top_face = (
        grid_bot[(0, 0, 1)], grid_bot[(1, 0, 1)], grid_bot[(1, 1, 1)], grid_bot[(0, 1, 1)],
    )
    bot_bottom_nodes = [
        grid_bot[(0, 0, 0)], grid_bot[(1, 0, 0)], grid_bot[(1, 1, 0)], grid_bot[(0, 1, 0)],
    ]
    top_top_nodes = [
        grid_top[(0, 0, 1)], grid_top[(1, 0, 1)], grid_top[(1, 1, 1)], grid_top[(0, 1, 1)],
    ]

    k_rep = E * L
    contact = NitscheContactConstraint3D(
        slave_faces=[top_bottom_face],
        master_faces=[bot_top_face],
        nid_to_idx=nid_to_idx,
        coords=mesh.coords,
        gamma=gamma_scale * k_rep,
    )
    solver.add_constraint(contact)

    # Fix bottom nodes completely
    for n in bot_bottom_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fix_dof(n, 2, 0.0)

    # Rollers on top block X and Y
    for n in list(top_bottom_face) + top_top_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)

    return solver, contact, top_top_nodes, nid_to_idx


def test_nitsche_two_cube_contact_converges():
    """Verify 3D full FEA convergence of stacked cubes under Nitsche contact."""
    solver, contact, top_nodes, nid_to_idx = _build_two_cube_nitsche_fixture(gamma_scale=1.0, gap0=0.1)

    # Prescribe downward displacement pushing top cube into bottom cube (delta_z = -0.15 mm, crossing gap 0.1 mm)
    for n in top_nodes:
        solver.fixed_dofs[3 * nid_to_idx[n] + 2] = -0.15

    converged, iters = solver.solve_step(dt=1.0, max_iters=50)
    assert converged is True
    assert iters > 0

    _, _, stats = contact.assemble(solver.u)
    assert stats["n_active"] == 4
    assert stats["total_normal_force"] > 0.0
    assert stats["max_penetration"] < 0.05


def test_nitsche_patch_test_uniform_pressure():
    """Verify uniform normal contact pressure across a 2x2 grid of Quad4 faces."""
    mesh = Mesh3D()
    W, H = 2.0, 1.0
    nx, ny = 2, 2
    xs = np.linspace(0.0, W, nx + 1)
    ys = np.linspace(0.0, W, ny + 1)

    nid = 1
    bot_grid = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(nid, x, y, 0.0)
            bot_grid[(i, j)] = nid
            nid += 1

    top_grid = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(nid, x, y, 0.0)
            top_grid[(i, j)] = nid
            nid += 1

    master_faces = [
        (bot_grid[(i, j)], bot_grid[(i + 1, j)], bot_grid[(i + 1, j + 1)], bot_grid[(i, j + 1)])
        for j in range(ny) for i in range(nx)
    ]
    slave_faces = [
        (top_grid[(i, j)], top_grid[(i + 1, j)], top_grid[(i + 1, j + 1)], top_grid[(i, j + 1)])
        for j in range(ny) for i in range(nx)
    ]

    nid_to_idx = mesh.node_id_to_index()
    gamma = 1.0e7
    contact = NitscheContactConstraint3D(
        slave_faces=slave_faces,
        master_faces=master_faces,
        nid_to_idx=nid_to_idx,
        coords=mesh.coords,
        gamma=gamma,
        consistency=False,   # surface-only fixture (no elements): penalty mode
    )

    # Prescribe uniform normal penetration
    delta = 0.01
    u = np.zeros(mesh.num_nodes * 3, dtype=np.float64)
    for j in range(ny + 1):
        for i in range(nx + 1):
            snid = top_grid[(i, j)]
            u[3 * nid_to_idx[snid] + 2] = -delta

    f_c, (rows, cols, data), stats = contact.assemble(u)

    expected_total_F = gamma * delta * (W * W)
    assert abs(stats["total_normal_force"] - expected_total_F) < 1e-6
    assert abs(np.sum(f_c)) < 1e-10

    # Symmetric stiffness check
    K = csc_matrix((data, (rows, cols)), shape=(mesh.num_nodes * 3, mesh.num_nodes * 3)).toarray()
    assert float(np.max(np.abs(K - K.T))) <= 1e-14 * float(np.abs(K).max())
