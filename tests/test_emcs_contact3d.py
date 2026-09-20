"""
test_emcs_contact3d.py
======================
Verification Suite for EMCS Conserving Contact Mechanics & PDASS (Team B).

Tests:
1. Gonzalez Discrete Gradient Derivative Energy Balance Consistency:
   Delta Psi_c == f_c . (u_{n+1} - u_n) within 1e-12.
   Tested on rigid plane contact, two-body pairwise contact, and micro-increments.

2. Symmetric Consistent Tangent Stiffness Matrix K_c:
   Machine-precision symmetry (||K_c - K_c^T|| < 1e-12).
   High-accuracy agreement with Central Finite Differences (rel_err < 1e-5).

3. 3D Elastic Block High-Velocity Undamped Impact Dynamics:
   Undamped EMCS midpoint time integration of dynamic impact, maximum compression,
   and restitution/rebound against a rigid wall.
   Total Mechanical Energy H(t) = E_kin + E_strain + Psi_contact strictly conserved
   (max energy drift < 1e-8 J across entire collision cycle).

4. PDASS Semi-Smooth Newton Complementarity & Chatter-Free Hysteresis Band:
   Verification of activation threshold (g > 1e-6) and deactivation threshold (g < -1e-5).
"""

from __future__ import annotations
import numpy as np
import pytest
from scipy.sparse import csc_matrix

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.emcs_dynamic3d import EMCSSolver3D
from dispsolver.solver3d.emcs_contact import EMCSContactConstraint3D


def build_c3d8_single_cube(size: float = 1.0, offset_x: float = 0.0) -> Mesh3D:
    """Build a single C3D8 hexahedral cube mesh."""
    mesh = Mesh3D()
    # 8 nodes
    coords = [
        (offset_x, 0.0, 0.0),
        (offset_x + size, 0.0, 0.0),
        (offset_x + size, size, 0.0),
        (offset_x, size, 0.0),
        (offset_x, 0.0, size),
        (offset_x + size, 0.0, size),
        (offset_x + size, size, size),
        (offset_x, size, size),
    ]
    for nid, (x, y, z) in enumerate(coords, start=1):
        mesh.add_node(nid, x, y, z)

    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], "C3D8")
    return mesh


# =========================================================================
# TEST 1: Gonzalez Discrete Gradient Energy Balance Consistency
# =========================================================================
def test_emcs_contact_discrete_gradient_energy_balance():
    """Verify that Gonzalez discrete gradient contact force exactly balances potential energy change.
    
    Delta Psi_c == f_c . (u_{n+1} - u_n) to machine precision (< 1e-12).
    """
    mesh = build_c3d8_single_cube(size=1.0, offset_x=0.0)
    coords = mesh.nodes_array()
    num_dofs = mesh.num_nodes * 3

    # Slave nodes at x = 0 face (nodes 1, 4, 5, 8 in 1-based index -> 0, 3, 4, 7 in 0-based index)
    slave_indices = [0, 3, 4, 7]

    # Rigid plane at x = 0 with normal n = (1, 0, 0) pointing in +x direction
    contact = EMCSContactConstraint3D(
        num_dofs=num_dofs,
        node_coords=coords,
        k_c=5.0e4,
        contact_type="rigid_plane",
        slave_node_indices=slave_indices,
        plane_point=np.array([0.0, 0.0, 0.0]),
        plane_normal=np.array([1.0, 0.0, 0.0]),
        dg_tol=1e-12,
    )

    # Test cases:
    # 1. Non-contact to contact (penetration entry)
    # 2. In contact to deeper penetration
    # 3. In contact to rebound/separation
    # 4. Micro-displacement (|Delta g| < 1e-12)
    # 5. Separation to separation
    test_displacements = [
        # (u_n, u_np1)
        # Entry: u_n = 0.05, u_np1 = -0.05 (x-direction penetration: g_n = -0.05 <= 0, g_np1 = +0.05 > 0)
        (np.zeros(num_dofs), np.zeros(num_dofs)),
        (np.full(num_dofs, 0.02), np.full(num_dofs, -0.04)),
        (np.full(num_dofs, -0.03), np.full(num_dofs, -0.08)),
        (np.full(num_dofs, -0.06), np.full(num_dofs, 0.03)),
        (np.full(num_dofs, -0.05), np.full(num_dofs, -0.05 + 1e-14)),  # micro-increment
        (np.full(num_dofs, 0.1), np.full(num_dofs, 0.2)),             # separated
    ]

    # Non-uniform random-like fixed displacements
    for case_idx, (u_n_base, u_np1_base) in enumerate(test_displacements):
        u_n = u_n_base.copy()
        u_np1 = u_np1_base.copy()
        # Add lateral y and z perturbations to verify 3D invariance
        u_n[1::3] += 0.01 * case_idx
        u_np1[1::3] += 0.02 * case_idx

        psi_n = contact.compute_contact_energy(u_n)
        psi_np1 = contact.compute_contact_energy(u_np1)
        delta_psi = psi_np1 - psi_n

        f_c, _, stats = contact.assemble_emcs(u_n, u_np1, dt=0.001)
        work_c = float(np.dot(f_c, u_np1 - u_n))

        err = abs(delta_psi - work_c)
        print(f"Case {case_idx}: Delta_Psi = {delta_psi:+.8e}, W_c = {work_c:+.8e}, err = {err:.2e}")
        assert err < 1e-12, f"Gonzalez discrete gradient energy balance violated in case {case_idx}: err={err}"


def test_emcs_two_body_contact_discrete_gradient_energy_balance():
    """Verify Gonzalez discrete gradient on two-body pairwise contact."""
    mesh_a = build_c3d8_single_cube(size=1.0, offset_x=0.0)
    mesh_b = build_c3d8_single_cube(size=1.0, offset_x=1.0)
    
    # Merge coords for testing
    coords_a = mesh_a.nodes_array()
    coords_b = mesh_b.nodes_array()
    coords_all = np.vstack([coords_a, coords_b])
    num_dofs = len(coords_all) * 3

    # Pair nodes on touching interface (x = 1.0 face of A vs x = 1.0 face of B)
    # A right face: 1, 2, 5, 6 (0-based)
    # B left face: 8, 11, 12, 15 (0-based)
    contact_pairs = [(8, 1), (11, 2), (12, 5), (15, 6)]

    contact = EMCSContactConstraint3D(
        num_dofs=num_dofs,
        node_coords=coords_all,
        k_c=1.0e5,
        contact_type="two_body",
        contact_pairs=contact_pairs,
        pair_normals=np.array([[1.0, 0.0, 0.0]] * len(contact_pairs)),
        dg_tol=1e-12,
    )

    u_n = np.zeros(num_dofs)
    u_np1 = np.zeros(num_dofs)

    # Move body B to the left by 0.05 (penetrating into A)
    for s_idx, m_idx in contact_pairs:
        u_np1[3 * s_idx] = -0.05
        u_np1[3 * m_idx] = +0.02

    psi_n = contact.compute_contact_energy(u_n)
    psi_np1 = contact.compute_contact_energy(u_np1)
    delta_psi = psi_np1 - psi_n

    f_c, _, stats = contact.assemble_emcs(u_n, u_np1, dt=0.001)
    work_c = float(np.dot(f_c, u_np1 - u_n))

    err = abs(delta_psi - work_c)
    print(f"Two-Body: Delta_Psi = {delta_psi:+.8e}, W_c = {work_c:+.8e}, err = {err:.2e}")
    assert err < 1e-12, f"Two-body discrete gradient balance failed: err={err}"


# =========================================================================
# TEST 2: Tangent Stiffness Consistency with Numerical Finite Difference
# =========================================================================
def test_emcs_contact_tangent_consistency_with_finite_difference():
    """Verify symmetric consistent tangent K_c against central finite differences."""
    mesh = build_c3d8_single_cube(size=1.0, offset_x=0.0)
    coords = mesh.nodes_array()
    num_dofs = mesh.num_nodes * 3
    slave_indices = [0, 3, 4, 7]

    contact = EMCSContactConstraint3D(
        num_dofs=num_dofs,
        node_coords=coords,
        k_c=8.0e4,
        contact_type="rigid_plane",
        slave_node_indices=slave_indices,
        plane_point=np.array([0.0, 0.0, 0.0]),
        plane_normal=np.array([1.0, 0.0, 0.0]),
        dg_tol=1e-12,
    )

    u_n = np.zeros(num_dofs)
    u_np1 = np.zeros(num_dofs)
    # Penetration in x
    for s_idx in slave_indices:
        u_n[3 * s_idx] = -0.02
        u_np1[3 * s_idx] = -0.06

    # Assemble analytical tangent
    f_c0, (rows, cols, data), _ = contact.assemble_emcs(u_n, u_np1, dt=0.001)
    K_c = csc_matrix((data, (rows, cols)), shape=(num_dofs, num_dofs)).toarray()

    # 1. Machine-precision symmetry check
    symm_err = np.max(np.abs(K_c - K_c.T))
    assert symm_err < 1e-12, f"K_c matrix must be symmetric, got symm_err={symm_err}"

    # 2. Central Finite Difference check
    eps = 1e-7
    K_num = np.zeros((num_dofs, num_dofs), dtype=np.float64)

    for j in range(num_dofs):
        u_plus = u_np1.copy()
        u_minus = u_np1.copy()
        u_plus[j] += eps
        u_minus[j] -= eps

        f_plus, _, _ = contact.assemble_emcs(u_n, u_plus, dt=0.001)
        f_minus, _, _ = contact.assemble_emcs(u_n, u_minus, dt=0.001)

        # d(f_c) / d(u_{n+1})
        K_num[:, j] = (f_plus - f_minus) / (2.0 * eps)

    active_dofs = []
    for s in slave_indices:
        active_dofs.extend([3 * s, 3 * s + 1, 3 * s + 2])

    K_c_sub = K_c[np.ix_(active_dofs, active_dofs)]
    K_num_sub = K_num[np.ix_(active_dofs, active_dofs)]

    rel_err = np.linalg.norm(K_c_sub - K_num_sub) / np.linalg.norm(K_c_sub)
    print(f"Tangent verification: ||K_ana - K_num|| / ||K_ana|| = {rel_err:.4e}")
    assert rel_err < 1e-5, f"Tangent stiffness derivative mismatch: rel_err={rel_err}"


# =========================================================================
# TEST 3: 3D High-Velocity Undamped Block Impact Dynamic Simulation
# =========================================================================
def test_emcs_high_velocity_cube_impact_energy_conservation():
    """Verify total mechanical energy conservation during high-velocity 3D block impact.
    
    Theoretical Benchmark:
      3D C3D8 elastic block flying at high velocity against an analytical rigid wall,
      undergoing impact penetration, maximum compression, elastic restitution,
      and complete rebound separation.
    
    Target Metric:
      Total mechanical energy drift |H(t) - H(0)| < 1e-8 J across entire collision cycle.
    """
    mesh = build_c3d8_single_cube(size=1.0, offset_x=0.02)
    density = 7.85e-9   # tonne/mm^3
    E = 1000.0          # MPa
    nu = 0.0            # Uniaxial impact mode
    k_contact = 10000.0 # N/mm

    solver = EMCSSolver3D(
        mesh=mesh,
        density=density,
        material_params={"E": E, "nu": nu},
        alpha_m=0.0,  # Zero Rayleigh damping (undamped)
        beta_k=0.0,
        use_anderson=False,
    )

    # Constrain Y and Z to focus on pure 1D longitudinal dynamic contact-impact wave
    for nid in mesh.nodes:
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)

    # Front slave nodes facing rigid wall at x = 0
    slave_node_indices = [0, 3, 4, 7]

    contact = EMCSContactConstraint3D(
        num_dofs=solver.num_dofs,
        node_coords=mesh.nodes_array(),
        k_c=k_contact,
        contact_type="rigid_plane",
        slave_node_indices=slave_node_indices,
        plane_point=np.array([0.0, 0.0, 0.0]),
        plane_normal=np.array([1.0, 0.0, 0.0]),
        dg_tol=1e-12,
    )
    solver.add_contact_constraint(contact)

    # Initial velocity flying towards wall at v_x = -100 mm/s
    v_init = -100.0
    for nid in mesh.nodes:
        solver.state.v[3 * (nid - 1)] = v_init

    # Compute initial total energy H(0)
    E0_tot, E0_kin, E0_str = solver.compute_total_energy()
    print(f"\n[Dynamic Impact] Initial State: E0_total = {E0_tot:.10e} J (E_kin = {E0_kin:.10e}, E_str = {E0_str:.10e})")
    assert E0_tot > 0.0

    # Simulation setup: full collision cycle (flight -> impact & compression -> restitution -> separation)
    dt = 0.0001  # 0.1 ms
    n_steps = 10
    energy_history = []
    contact_occurred = False
    rebound_occurred = False

    for step in range(1, n_steps + 1):
        converged, iters, res_norm = solver.solve_step_emcs(dt=dt, tol=1e-10, max_iters=35)
        assert converged, f"Solver failed to converge at step {step} (res_norm={res_norm:.3e})"

        E_tot, E_kin, E_str, E_c = solver.compute_detailed_energy()
        drift = abs(E_tot - E0_tot)
        energy_history.append((step, E_tot, E_kin, E_str, E_c, drift))

        # Check contact state
        if E_c > 1e-12:
            contact_occurred = True

        # Check rebound state (positive x velocity after contact)
        mean_vx = np.mean(solver.state.v[0::3])
        if contact_occurred and mean_vx > 50.0:
            rebound_occurred = True

        print(
            f"Step {step:2d} | it={iters:2d} | E_tot={E_tot:.10e} | "
            f"E_kin={E_kin:.4e} | E_str={E_str:.4e} | E_c={E_c:.4e} | drift={drift:.2e} J"
        )

    max_drift = max(h[5] for h in energy_history)
    print(f"\n[Dynamic Impact Summary] Max Energy Drift across cycle: {max_drift:.4e} J")
    print(f"Contact Occurred: {contact_occurred}, Rebound Occurred: {rebound_occurred}")

    assert contact_occurred, "Impact contact must have occurred during simulation!"
    assert rebound_occurred, "Block must rebound away from wall after impact!"
    assert max_drift < 1e-8, f"EMCS total energy conservation violated: max drift = {max_drift:.4e} J >= 1e-8 J"


# =========================================================================
# TEST 4: PDASS Semi-Smooth Newton & Hysteresis Chatter Band
# =========================================================================
def test_emcs_pdass_hysteresis_chatter_band():
    """Verify PDASS hysteresis logic: activate at g > 1e-6, deactivate at g < -1e-5."""
    mesh = build_c3d8_single_cube(size=1.0, offset_x=0.0)
    coords = mesh.nodes_array()
    num_dofs = mesh.num_nodes * 3
    slave_indices = [0]

    contact = EMCSContactConstraint3D(
        num_dofs=num_dofs,
        node_coords=coords,
        k_c=1.0e4,
        contact_type="rigid_plane",
        slave_node_indices=slave_indices,
        plane_point=np.array([0.0, 0.0, 0.0]),
        plane_normal=np.array([1.0, 0.0, 0.0]),
        enable_pdass=True,
        g_act=1e-6,
        g_deact=-1e-5,
    )

    u = np.zeros(num_dofs)
    # Case 1: Initial separation (g = -1e-4) -> Inactive
    u[0] = 1e-4
    n_act, n_deact = contact.update_pdass_state(u)
    assert not contact.active_flags[0]

    # Case 2: In chatter band below activation (g = 5e-7) -> Stays Inactive
    u[0] = -5e-7
    contact.update_pdass_state(u)
    assert not contact.active_flags[0]

    # Case 3: Cross activation threshold (g = 2e-6 > g_act) -> Activates!
    u[0] = -2e-6
    contact.update_pdass_state(u)
    assert contact.active_flags[0]

    # Case 4: Retreat into chatter band (g = -5e-6, between -1e-5 and +1e-6) -> Stays ACTIVE (no chatter)!
    u[0] = 5e-6
    contact.update_pdass_state(u)
    assert contact.active_flags[0], "Hysteresis band must keep contact ACTIVE when retreating above g_deact!"

    # Case 5: Fully separate (g = -2e-5 < g_deact) -> Deactivates!
    u[0] = 2e-5
    contact.update_pdass_state(u)
    assert not contact.active_flags[0], "Contact must DEACTIVATE once separation exceeds g_deact!"
