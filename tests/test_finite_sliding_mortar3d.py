"""
test_finite_sliding_mortar3d.py
================================
Comprehensive verification test suite for FiniteSlidingDualMortar3D:
1. Wohlmuth (2000) Bi-Orthogonality Test: int(Psi_a * N_b) = D_a * delta_ab.
2. Algebraic Condensation Test: u_s = D^-1 @ M_sm @ u_m exact interpolation.
3. Opposing Normal Filter Test: n_A . n_B <= -0.3 and shared-node exclusion for self-contact.
4. Convected Tracking Large Sliding Test: Slave sliding across multiple master Quad4 elements.
5. Popp (2010) Leibniz Domain Boundary Variation Line Integral Test: K_contour consistency.
6. Numba JIT vs Python Parity Test: Machine-precision equality of assembled forces & tangent.
7. Action-Reaction Force Equilibrium Test: sum(f_slave) + sum(f_master) = 0.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from dispsolver.constraint3d.finite_sliding_mortar3d import (
    FiniteSlidingDualMortar3D,
    _quad4_shape_functions,
    _quad4_tangents_and_normal_numba,
    _project_point_to_quad4_numba,
    _GAUSS2X2,
    HAS_NUMBA,
)


def test_wohlmuth_biorthogonality():
    """Test 1: Verify Wohlmuth (2000) bi-orthogonality condition:
    int_{f} Psi_a(xi, eta) * N_b(xi, eta) dA = D_{f, a} * delta_{ab}
    """
    # 1 Quad4 face of size 2.0 x 1.5 at z = 0
    coords = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.5, 0.0],
        [0.0, 1.5, 0.0],
    ], dtype=np.float64)

    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
    slave_faces = [(1, 2, 3, 4)]
    master_faces = [(1, 2, 3, 4)]  # dummy

    contact = FiniteSlidingDualMortar3D(
        slave_faces=slave_faces,
        master_faces=master_faces,
        nid_to_idx=nid_to_idx,
        coords=coords,
    )

    A_f = contact._A_f[0]
    q_arr = coords

    # Compute numerical integral of Psi_a * N_b
    integral_Psi_N = np.zeros((4, 4), dtype=np.float64)
    for xi, eta, w_g in _GAUSS2X2:
        N = np.array(_quad4_shape_functions(xi, eta))
        Psi = A_f @ N
        _, _, _, dA = _quad4_tangents_and_normal_numba(q_arr, xi, eta)
        integral_Psi_N += np.outer(Psi, N) * (dA * w_g)

    # Expected: diagonal matrix with D_f on the diagonal
    diag_vals = np.diag(integral_Psi_N)
    off_diag = integral_Psi_N - np.diag(diag_vals)

    # Area should be 2.0 * 1.5 = 3.0, sum of D_f equals area
    assert np.isclose(np.sum(diag_vals), 3.0, atol=1e-12)
    # Off-diagonals must strictly vanish within machine precision
    assert np.max(np.abs(off_diag)) < 1e-13, f"Non-zero off-diagonals: {off_diag}"
    # Verify D_global matches
    for a, nid in enumerate(slave_faces[0]):
        assert np.isclose(contact.D_global[nid], diag_vals[a], atol=1e-12)


def test_algebraic_condensation_exactness():
    """Test 2: Verify Wohlmuth algebraic condensation:
    u_s = D^-1 @ M_sm @ u_m
    When master is displaced by an affine / rigid field, the condensed slave displacement
    must match the master displacement field exactly!
    """
    # Slave: 1 Quad4 face [0.2, 0.8] x [0.2, 0.8] at z = 0.0
    # Master: 1 Quad4 face [0.0, 1.0] x [0.0, 1.0] at z = 0.0
    coords = np.array([
        # Slave nodes 1..4
        [0.2, 0.2, 0.0],
        [0.8, 0.2, 0.0],
        [0.8, 0.8, 0.0],
        [0.2, 0.8, 0.0],
        # Master nodes 5..8 (facing slave)
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    nid_to_idx = {i + 1: i for i in range(8)}
    slave_faces = [(1, 2, 3, 4)]
    # Reversed orientation for master face to face slave (normal opposing)
    master_faces = [(5, 8, 7, 6)]

    contact = FiniteSlidingDualMortar3D(
        slave_faces=slave_faces,
        master_faces=master_faces,
        nid_to_idx=nid_to_idx,
        coords=coords,
        opposing_normal_threshold=1.0,  # allow all for geometry test
    )

    u_zero = np.zeros(24, dtype=np.float64)
    contact.update_contact_topology(u_zero)

    D_mat, M_sm, C_mat = contact.compute_algebraic_condensation_matrices()
    assert D_mat.shape[0] == 4
    assert M_sm.shape == (4, 4)

    # Linear master displacement: u_m = alpha * x + beta * y + c
    # Let u_m = 0.05 on all master nodes (rigid shift)
    u_m = np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float64)
    u_s_condensed = C_mat @ u_m

    # Condensed slave displacements must be identical to 0.05
    assert np.allclose(u_s_condensed, 0.05, atol=1e-12)


def test_opposing_normal_and_shared_node_filter():
    """Test 3: Self-contact filters:
    1. Adjacent faces sharing nodes must be automatically excluded.
    2. Co-planar faces with identical normals (n_A . n_B = +1.0) must be rejected
       by opposing normal filter (n_A . n_B <= -0.3).
    3. Deformed facing faces (n_A . n_B = -1.0) must pass.
    """
    # 2 Quad4 faces sharing an edge
    coords = np.array([
        # Face A: nodes 1, 2, 3, 4
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        # Face B: nodes 2, 5, 6, 3 (shares edge 2-3 with Face A)
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        # Face C: nodes 7, 8, 9, 10 (separate, parallel, above Face A, facing it)
        [0.0, 0.0, 0.01],
        [1.0, 0.0, 0.01],
        [1.0, 1.0, 0.01],
        [0.0, 1.0, 0.01],
    ], dtype=np.float64)

    nid_to_idx = {i + 1: i for i in range(10)}

    # Case A: Face A vs Face B (shared nodes 2, 3)
    c_shared = FiniteSlidingDualMortar3D(
        slave_faces=[(1, 2, 3, 4)],
        master_faces=[(2, 5, 6, 3)],
        nid_to_idx=nid_to_idx,
        coords=coords,
        opposing_normal_threshold=-0.3,
    )
    u = np.zeros(30, dtype=np.float64)
    c_shared.update_contact_topology(u)
    # Must have zero segments because they share nodes
    assert len(c_shared.segments) == 0

    # Case B: Face A vs Face C pointing same direction (both normals +z, dot = +1.0)
    c_same_normal = FiniteSlidingDualMortar3D(
        slave_faces=[(1, 2, 3, 4)],
        master_faces=[(7, 8, 9, 10)],
        nid_to_idx=nid_to_idx,
        coords=coords,
        opposing_normal_threshold=-0.3,
    )
    c_same_normal.update_contact_topology(u)
    # Must be filtered out because normals are not opposing (dot > -0.3)
    assert len(c_same_normal.segments) == 0

    # Case C: Face A vs Face C oriented opposite (normal -z, dot = -1.0)
    c_opposing = FiniteSlidingDualMortar3D(
        slave_faces=[(1, 2, 3, 4)],
        master_faces=[(7, 10, 9, 8)],
        nid_to_idx=nid_to_idx,
        coords=coords,
        opposing_normal_threshold=-0.3,
    )
    c_opposing.update_contact_topology(u)
    # Must successfully detect contact segment!
    assert len(c_opposing.segments) == 1


def test_finite_sliding_convected_tracking_across_elements():
    """Test 4: Convected tracking across multiple elements under large sliding:
    Slave element slides from Master Element 1 to Master Element 2.
    """
    # 2 Master faces adjacent along x:
    # M1: [0, 1] x [0, 1]
    # M2: [1, 2] x [0, 1]
    # Slave: [0.2, 0.8] x [0.2, 0.8] initially above M1
    coords = np.array([
        # Slave nodes 1..4
        [0.2, 0.2, 0.005],
        [0.8, 0.2, 0.005],
        [0.8, 0.8, 0.005],
        [0.2, 0.8, 0.005],
        # M1 nodes 5..8 (oriented CCW from top view -> reversed to face -z)
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        # M2 nodes 9..12
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float64)

    nid_to_idx = {i + 1: i for i in range(12)}
    slave_faces = [(1, 2, 3, 4)]
    # Facing -z:
    master_faces = [(5, 8, 7, 6), (9, 12, 11, 10)]

    contact = FiniteSlidingDualMortar3D(
        slave_faces=slave_faces,
        master_faces=master_faces,
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=1e6,
        opposing_normal_threshold=-0.3,
    )

    u = np.zeros(36, dtype=np.float64)

    # Configuration 1: Slave above M1, u_z = -0.01 (penetration = 0.005)
    for n in range(4):
        u[3 * n + 2] = -0.010
    f1, _, stats1 = contact.assemble(u)
    assert stats1["n_active"] > 0
    assert len(contact.segments) == 1
    assert contact.segments[0]["master_face"] == master_faces[0]

    # Configuration 2: Large finite sliding: slave shifted by +1.0 in x
    # Slave now lies completely above M2!
    for n in range(4):
        u[3 * n + 0] = 1.0
        u[3 * n + 2] = -0.010

    f2, _, stats2 = contact.assemble(u)
    assert stats2["n_active"] > 0
    assert len(contact.segments) == 1
    # Convected tracking smoothly transitioned contact pairing to M2!
    assert contact.segments[0]["master_face"] == master_faces[1]

    # Configuration 3: Boundary crossing transition: slave shifted by +0.5 in x
    # Slave straddles the boundary between M1 and M2!
    for n in range(4):
        u[3 * n + 0] = 0.5
        u[3 * n + 2] = -0.010

    f3, _, stats3 = contact.assemble(u)
    assert stats3["n_active"] > 0
    # Both M1 and M2 are simultaneously engaged across the clipped boundary!
    assert len(contact.segments) == 2


def test_popp_leibniz_boundary_variation_contour_integral():
    """Test 5: Verify Popp (2010) Leibniz boundary variation line integral.
    When a clipped segment intersects an element boundary, contour_segments must be
    populated and contribute to the tangent stiffness.
    """
    coords = np.array([
        # Slave nodes 1..4: [0, 1] x [0, 1] at z = 0.002
        [0.0, 0.0, 0.002],
        [1.0, 0.0, 0.002],
        [1.0, 1.0, 0.002],
        [0.0, 1.0, 0.002],
        # Master nodes 5..8: shifted [0.5, 1.5] x [0.5, 1.5] at z = 0.0
        [0.5, 0.5, 0.0],
        [1.5, 0.5, 0.0],
        [1.5, 1.5, 0.0],
        [0.5, 1.5, 0.0],
    ], dtype=np.float64)

    nid_to_idx = {i + 1: i for i in range(8)}
    slave_faces = [(1, 2, 3, 4)]
    master_faces = [(5, 8, 7, 6)]

    contact_with_contour = FiniteSlidingDualMortar3D(
        slave_faces=slave_faces,
        master_faces=master_faces,
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=1e6,
        use_contour_integral=True,
    )

    contact_no_contour = FiniteSlidingDualMortar3D(
        slave_faces=slave_faces,
        master_faces=master_faces,
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=1e6,
        use_contour_integral=False,
    )

    u = np.zeros(24, dtype=np.float64)
    # Impose vertical penetration u_z = -0.005 on slave
    for n in range(4):
        u[3 * n + 2] = -0.005

    f1, (r1, c1, d1), stats1 = contact_with_contour.assemble(u, use_numba=False)
    f2, (r2, c2, d2), stats2 = contact_no_contour.assemble(u, use_numba=False)

    # Force vectors must be identical (contour integral affects tangent stiffness)
    assert np.allclose(f1, f2, atol=1e-12)

    # Contour segments must be detected in contact_with_contour
    assert stats1["n_contour"] > 0
    assert stats2["n_contour"] == 0
    # Tangent triplets with contour include additional boundary variation terms
    assert len(d1) > len(d2)


def test_numba_jit_assembly_parity():
    """Test 6: Verify Numba JIT accelerated assembly matches Python reference to machine precision."""
    coords = np.array([
        # Slave nodes 1..4
        [0.1, 0.1, 0.001],
        [0.9, 0.1, 0.001],
        [0.9, 0.9, 0.001],
        [0.1, 0.9, 0.001],
        # Master nodes 5..8
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    nid_to_idx = {i + 1: i for i in range(8)}
    slave_faces = [(1, 2, 3, 4)]
    master_faces = [(5, 8, 7, 6)]

    contact = FiniteSlidingDualMortar3D(
        slave_faces=slave_faces,
        master_faces=master_faces,
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=2.5e6,
        use_contour_integral=True,
    )

    u = np.zeros(24, dtype=np.float64)
    for n in range(4):
        u[3 * n + 2] = -0.004

    # Assemble via Numba
    f_numba, (r_num, c_num, d_num), stats_num = contact.assemble(u, use_numba=True)
    # Assemble via Python
    f_py, (r_py, c_py, d_py), stats_py = contact.assemble(u, use_numba=False)

    # Verify force vector parity
    assert np.allclose(f_numba, f_py, atol=1e-13), f"Force discrepancy: {np.max(np.abs(f_numba - f_py))}"

    # Build CSR matrices to compare tangent stiffnesses
    K_numba = sp.csr_matrix((d_num, (r_num, c_num)), shape=(24, 24))
    K_py = sp.csr_matrix((d_py, (r_py, c_py)), shape=(24, 24))

    diff_K = (K_numba - K_py).toarray()
    assert np.allclose(K_numba.toarray(), K_py.toarray(), rtol=1e-12, atol=1e-9), f"Stiffness discrepancy: {np.max(np.abs(diff_K))}"


def test_action_reaction_force_equilibrium():
    """Test 7: Verify Newton's third law:
    sum(f_slave) + sum(f_master) = 0 for arbitrary finite sliding contact.
    """
    coords = np.array([
        # Slave nodes 1..4
        [0.2, 0.3, 0.002],
        [0.9, 0.3, 0.002],
        [0.9, 0.8, 0.002],
        [0.2, 0.8, 0.002],
        # Master nodes 5..8
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [1.2, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    nid_to_idx = {i + 1: i for i in range(8)}
    slave_faces = [(1, 2, 3, 4)]
    master_faces = [(5, 8, 7, 6)]

    contact = FiniteSlidingDualMortar3D(
        slave_faces=slave_faces,
        master_faces=master_faces,
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=5.0e6,
    )

    u = np.zeros(24, dtype=np.float64)
    # Combined vertical compression and horizontal sliding
    for n in range(4):
        u[3 * n + 0] = 0.05
        u[3 * n + 1] = 0.02
        u[3 * n + 2] = -0.006

    f_contact, _, stats = contact.assemble(u, use_numba=True)
    assert stats["n_active"] > 0
    assert stats["total_normal_force"] > 0.0

    # Total force on slave nodes (0..3)
    f_slave = np.zeros(3, dtype=np.float64)
    for n in range(4):
        f_slave += f_contact[3 * n : 3 * n + 3]

    # Total force on master nodes (4..7)
    f_master = np.zeros(3, dtype=np.float64)
    for n in range(4, 8):
        f_master += f_contact[3 * n : 3 * n + 3]

    sum_forces = f_slave + f_master
    # Action-reaction equilibrium to machine precision
    assert np.allclose(sum_forces, 0.0, atol=1e-11), f"Unbalanced force: {sum_forces}"
