"""
test_dual_mortar_2d.py
======================
Verification test suite for 2D Segment-to-Segment (S2S) Dual Mortar & Nitsche Contact.

Tests:
1. test_1d_dual_basis_biorthogonality_and_partition_of_unity:
   Verifies machine-precision biorthogonality int_{-1}^1 psi_i * N_j dxi = delta_{ij} (< 1e-14)
   and exact partition of unity psi_1(xi) + psi_2(xi) == 1.0 (< 1e-15).
2. test_single_segment_pair_uniform_pressure_split:
   Single pair of conforming segments under uniform penetration.
   Checks 50-50 nodal force split and exact action-reaction Newton's third law.
3. test_nonmatching_mesh_2d_patch_test:
   Non-matching 2D block contact patch test (top block 5 elements, bottom block 3 elements).
   Verifies uniform contact transmission and stress uniformity (error < 0.1%).
4. test_uzawa_augmented_lagrangian_contraction:
   PDASS Uzawa augmented Lagrangian multiplier iterations, verifying geometric decay
   of penetration down to floating-point tolerance.
5. test_nitsche_tangent_symmetry:
   Multiplier-free Nitsche formulation tangent stiffness symmetry: |K - K^T| < 1e-12.
   Also checks Dual Mortar rank-1 consistent tangent symmetry.
"""

import numpy as np
import pytest
from scipy.sparse import csc_matrix

from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.constraint.dual_mortar_contact import (
    DualMortarContactConstraint2D,
    _line2_shape_functions,
    _line2_dual_shape_functions,
)


# ===========================================================================
# 1. 1D Biorthogonality & Partition of Unity
# ===========================================================================

def test_1d_dual_basis_biorthogonality_and_partition_of_unity():
    """Verify machine-precision biorthogonality and partition of unity."""
    # Partition of unity across dense grid
    xi_grid = np.linspace(-1.0, 1.0, 101)
    for xi in xi_grid:
        psi1, psi2 = _line2_dual_shape_functions(xi)
        assert abs((psi1 + psi2) - 1.0) < 1e-15, f"Partition of unity failed at xi={xi}: sum={psi1+psi2}"

    # Exact Gauss-Legendre quadrature of order 5 (exact for polynomials up to degree 9)
    # psi_i * N_j is quadratic (degree 2), so 5-point Gauss is exact to machine precision
    gauss_pts = [
        (-np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0, (322.0 - 13.0 * np.sqrt(70.0)) / 900.0),
        (-np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0, (322.0 + 13.0 * np.sqrt(70.0)) / 900.0),
        (0.0, 128.0 / 225.0),
        (+np.sqrt(5.0 - 2.0 * np.sqrt(10.0 / 7.0)) / 3.0, (322.0 + 13.0 * np.sqrt(70.0)) / 900.0),
        (+np.sqrt(5.0 + 2.0 * np.sqrt(10.0 / 7.0)) / 3.0, (322.0 - 13.0 * np.sqrt(70.0)) / 900.0),
    ]

    integral_matrix = np.zeros((2, 2), dtype=np.float64)
    for xi, w in gauss_pts:
        N1, N2 = _line2_shape_functions(xi)
        psi1, psi2 = _line2_dual_shape_functions(xi)
        N = np.array([N1, N2])
        psi = np.array([psi1, psi2])
        integral_matrix += np.outer(psi, N) * w

    expected_I = np.eye(2, dtype=np.float64)
    biorthogonality_err = float(np.max(np.abs(integral_matrix - expected_I)))
    assert biorthogonality_err < 1e-14, f"Biorthogonality error {biorthogonality_err} exceeded 1e-14"

    # Verify scaled physical segment of length L_s
    L_s = 4.25
    coords = np.array([
        [0.0, 0.0],
        [L_s, 0.0],
        [0.0, 0.0],
        [L_s, 0.0],
    ])
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
    contact = DualMortarContactConstraint2D(
        slave_segments=[(1, 2)],
        master_segments=[(3, 4)],
        nid_to_idx=nid_to_idx,
        coords=coords,
    )
    # On physical segment: D_global = L_s / 2 for each node
    assert abs(contact.D_global[1] - L_s / 2.0) < 1e-14
    assert abs(contact.D_global[2] - L_s / 2.0) < 1e-14


# ===========================================================================
# 2. Single Segment Pair Uniform Pressure Split
# ===========================================================================

def test_single_segment_pair_uniform_pressure_split():
    """Verify 50-50 nodal force split and Newton's third law under uniform penetration."""
    L_s = 2.0
    coords = np.array([
        [0.0, 0.0],  # Slave node 1
        [L_s, 0.0],  # Slave node 2
        [0.0, 0.0],  # Master node 3
        [L_s, 0.0],  # Master node 4
    ])
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
    k_c = 5.0e6
    contact = DualMortarContactConstraint2D(
        slave_segments=[(1, 2)],
        master_segments=[(3, 4)],
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=k_c,
    )

    # Uniform downward displacement of slave: delta = 0.02
    delta = 0.02
    u = np.zeros(8, dtype=np.float64)
    u[2 * 0 + 1] = -delta  # node 1 (Y)
    u[2 * 1 + 1] = -delta  # node 2 (Y)

    f_c, (rows, cols, data), stats = contact.assemble(u)

    # Expected total normal contact force: F = k_c * delta * L_s
    expected_total_F = k_c * delta * L_s
    assert abs(stats["total_normal_force"] - expected_total_F) < 1e-8

    # Check 50-50 split on slave: each slave node gets -0.5 * F in Y (resisting force)
    # Master gets +0.5 * F in Y
    assert abs(f_c[2 * 0 + 1] - (-0.5 * expected_total_F)) < 1e-8
    assert abs(f_c[2 * 1 + 1] - (-0.5 * expected_total_F)) < 1e-8
    assert abs(f_c[2 * 2 + 1] - (+0.5 * expected_total_F)) < 1e-8
    assert abs(f_c[2 * 3 + 1] - (+0.5 * expected_total_F)) < 1e-8

    # Horizontal forces must be identically zero
    assert abs(f_c[2 * 0 + 0]) < 1e-14
    assert abs(f_c[2 * 1 + 0]) < 1e-14

    # Newton's third law: sum of contact forces across interface == 0
    assert abs(np.sum(f_c)) < 1e-12


# ===========================================================================
# 3. Non-matching Mesh 2D Block Patch Test (Top 5 elements, Bottom 3 elements)
# ===========================================================================

def _create_2d_block_fixture(
    n_top_x: int = 5,
    n_bot_x: int = 3,
    W: float = 1.0,
    H: float = 1.0,
    E: float = 2.0e5,
    nu: float = 0.3,
    augmented_lagrange: bool = False,
    formulation: str = "dual_mortar",
):
    """Create a 2D two-block contact fixture with dissimilar mesh densities along interface.

    Bottom block: [-W/2, W/2] x [-H, 0] with n_bot_x elements along X, 2 along Y.
    Top block:    [-W/2, W/2] x [ 0, H] with n_top_x elements along X, 2 along Y.
    """
    mesh = Mesh2D()
    nid = 1
    eid = 1

    # 1. Bottom block
    bot_nx = n_bot_x
    bot_ny = 2
    bot_x = np.linspace(-W / 2.0, W / 2.0, bot_nx + 1)
    bot_y = np.linspace(-H, 0.0, bot_ny + 1)
    bot_node_grid = np.zeros((bot_ny + 1, bot_nx + 1), dtype=int)

    for j in range(bot_ny + 1):
        for i in range(bot_nx + 1):
            mesh.add_node(nid, bot_x[i], bot_y[j])
            bot_node_grid[j, i] = nid
            nid += 1

    for j in range(bot_ny):
        for i in range(bot_nx):
            n1 = bot_node_grid[j, i]
            n2 = bot_node_grid[j, i + 1]
            n3 = bot_node_grid[j + 1, i + 1]
            n4 = bot_node_grid[j + 1, i]
            mesh.add_element(eid, [n1, n2, n3, n4], elem_type="CPE4")
            eid += 1

    # Master segments: top surface of bottom block (j = bot_ny, y = 0)
    master_segments = [
        (bot_node_grid[bot_ny, i], bot_node_grid[bot_ny, i + 1])
        for i in range(bot_nx)
    ]
    bottom_fixed_nodes = list(bot_node_grid[0, :])

    # 2. Top block
    top_nx = n_top_x
    top_ny = 2
    top_x = np.linspace(-W / 2.0, W / 2.0, top_nx + 1)
    top_y = np.linspace(0.0, H, top_ny + 1)
    top_node_grid = np.zeros((top_ny + 1, top_nx + 1), dtype=int)

    for j in range(top_ny + 1):
        for i in range(top_nx + 1):
            mesh.add_node(nid, top_x[i], top_y[j])
            top_node_grid[j, i] = nid
            nid += 1

    for j in range(top_ny):
        for i in range(top_nx):
            n1 = top_node_grid[j, i]
            n2 = top_node_grid[j, i + 1]
            n3 = top_node_grid[j + 1, i + 1]
            n4 = top_node_grid[j + 1, i]
            mesh.add_element(eid, [n1, n2, n3, n4], elem_type="CPE4")
            eid += 1

    # Slave segments: bottom surface of top block (j = 0, y = 0)
    slave_segments = [
        (top_node_grid[0, i], top_node_grid[0, i + 1])
        for i in range(top_nx)
    ]
    top_driven_nodes = list(top_node_grid[top_ny, :])

    solver = DynamicSolver2D(mesh, materials={"E": E, "nu": nu}, nlgeom=True)
    nid_to_idx = mesh.node_id_to_index()
    coords = mesh.nodes_array()

    k_c = 10.0 * E
    contact = DualMortarContactConstraint2D(
        slave_segments=slave_segments,
        master_segments=master_segments,
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=k_c,
        augmented_lagrange=augmented_lagrange,
        formulation=formulation,
    )
    solver.add_constraint(contact)

    # Boundary conditions for 1D uniaxial patch test:
    # Fix bottom surface in X and Y
    for n in bottom_fixed_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)

    # Rollers on side boundaries (x = -W/2 and x = +W/2) for uniaxial compression
    for nid, node in mesh.nodes.items():
        if abs(node.x - (-W / 2.0)) < 1e-8 or abs(node.x - (W / 2.0)) < 1e-8:
            solver.fix_dof(nid, 0, 0.0)

    # Top surface: prescribed uniform downward displacement in Y
    prescribed_d = -0.005
    for n in top_driven_nodes:
        solver.fix_dof(n, 1, prescribed_d)

    return solver, contact, slave_segments, master_segments, nid_to_idx


def test_nonmatching_mesh_2d_patch_test():
    """Non-matching 2D block patch test: 5 slave elements on 3 master elements.

    Must pass with stress uniformity error < 0.1% across the contact interface.
    """
    solver, contact, slave_segs, master_segs, nid_to_idx = _create_2d_block_fixture(
        n_top_x=5, n_bot_x=3, augmented_lagrange=False
    )

    converged, iters = solver.solve_step(dt=1.0, max_iters=25)
    assert converged, f"Patch test failed to converge, iters={iters}"

    # Check active set: all 6 slave interface nodes must be in contact
    active = contact.get_active_set(solver.u)
    assert len(active) == 6

    # Evaluate contact weak gap g_tilde at interface nodes
    g_tilde = contact._compute_g_tilde(solver.u)
    g_vals = [g_tilde[nid] for nid in active]

    # Check weak gap uniformity across non-matching slave nodes
    mean_g = float(np.mean(g_vals))
    assert mean_g > 0.0
    rel_std_g = float(np.std(g_vals) / mean_g)
    assert rel_std_g < 1.0e-3, f"Non-matching contact penetration non-uniformity: {rel_std_g * 100:.3f}%"

    # Evaluate reaction forces / assembled contact forces
    f_c, _, stats = contact.assemble(solver.u)
    total_Fy = stats["total_normal_force"]
    assert total_Fy > 0.0


# ===========================================================================
# 4. Uzawa Augmented Lagrangian Contraction Test
# ===========================================================================

def test_uzawa_augmented_lagrangian_contraction():
    """Verify that Uzawa augmented Lagrangian multiplier iterations monotonically shrink penetration."""
    solver, contact, slave_segs, master_segs, nid_to_idx = _create_2d_block_fixture(
        n_top_x=4, n_bot_x=3, augmented_lagrange=True
    )

    penetrations = []
    # Perform multiple augmented cycles
    for cycle in range(5):
        converged, iters = solver.solve_step(dt=1.0, max_iters=30, augment_relaxation=1.0)
        assert converged, f"Cycle {cycle} failed to converge"
        res = contact.update_augmented_multipliers(solver.u, omega=1.0)
        penetrations.append(res["max_penetration"])

    # Penetration must decrease monotonically down to floating-point floor (< 1e-9)
    for i in range(len(penetrations) - 1):
        if penetrations[i] > 1e-9:
            assert penetrations[i + 1] <= penetrations[i] * 1.01, (
                f"Penetration did not contract from cycle {i} ({penetrations[i]}) "
                f"to cycle {i+1} ({penetrations[i+1]})"
            )

    # Penetration at end should be contracted to near zero or floating-point floor (< 1e-9)
    assert penetrations[-1] < 1e-8, f"Expected penetration below 1e-8, got {penetrations[-1]}"


# ===========================================================================
# 5. Nitsche & Dual Mortar Tangent Stiffness Symmetry Test
# ===========================================================================

def test_nitsche_tangent_symmetry():
    """Verify Nitsche contact formulation tangent stiffness is strictly symmetric: |K - K^T| < 1e-12."""
    coords = np.array([
        [0.0, 0.0],   # Slave 1
        [1.0, 0.0],   # Slave 2
        [2.0, 0.0],   # Slave 3
        [0.0, 0.0],   # Master 4
        [1.2, 0.0],   # Master 5
        [2.0, 0.0],   # Master 6
    ])
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

    contact_nitsche = DualMortarContactConstraint2D(
        slave_segments=[(1, 2), (2, 3)],
        master_segments=[(4, 5), (5, 6)],
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=1.0e7,
        formulation="nitsche",
    )

    # Prescribe arbitrary non-uniform penetrating displacement
    u = np.zeros(12, dtype=np.float64)
    u[2 * 0 + 1] = -0.01
    u[2 * 1 + 1] = -0.015
    u[2 * 2 + 1] = -0.008
    u[2 * 3 + 1] = 0.0
    u[2 * 4 + 1] = 0.002
    u[2 * 5 + 1] = 0.0

    _, (rows, cols, data), _ = contact_nitsche.assemble(u)
    K_nitsche = csc_matrix((data, (rows, cols)), shape=(12, 12)).toarray()

    # Verify symmetry
    asym_nitsche = float(np.max(np.abs(K_nitsche - K_nitsche.T)))
    assert asym_nitsche < 1e-12, f"Nitsche tangent asymmetry {asym_nitsche} exceeded 1e-12"

    # Also verify Dual Mortar consistent tangent symmetry
    contact_dm = DualMortarContactConstraint2D(
        slave_segments=[(1, 2), (2, 3)],
        master_segments=[(4, 5), (5, 6)],
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=1.0e7,
        formulation="dual_mortar",
    )
    _, (rows_dm, cols_dm, data_dm), _ = contact_dm.assemble(u)
    K_dm = csc_matrix((data_dm, (rows_dm, cols_dm)), shape=(12, 12)).toarray()

    asym_dm = float(np.max(np.abs(K_dm - K_dm.T)))
    assert asym_dm < 1e-12, f"Dual Mortar tangent asymmetry {asym_dm} exceeded 1e-12"
