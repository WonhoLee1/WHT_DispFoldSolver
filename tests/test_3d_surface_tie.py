"""
test_3d_surface_tie.py
======================
Unit tests for 3D Penalty-based Surface-to-Surface Tie Constraint.
"""

import numpy as np
import pytest

from dispsolver.constraint3d.surface_tie3d import SurfaceTieConstraint3D, _project_point_to_quad4


def test_3d_quad4_projection():
    """Test 3D point projection onto a Quad4 face."""
    q1 = np.array([0.0, 0.0, 0.0])
    q2 = np.array([2.0, 0.0, 0.0])
    q3 = np.array([2.0, 2.0, 0.0])
    q4 = np.array([0.0, 2.0, 0.0])

    p = np.array([1.0, 1.0, 0.5])  # Point at center, elevated in Z

    xi, eta, dist, proj_pt = _project_point_to_quad4(p, q1, q2, q3, q4)

    assert abs(xi) < 1e-5
    assert abs(eta) < 1e-5
    assert abs(dist - 0.5) < 1e-5
    assert np.allclose(proj_pt, [1.0, 1.0, 0.0], atol=1e-5)


def test_3d_surface_tie_assembly():
    """Test 3D surface tie penalty force and stiffness matrix assembly."""
    # Master quad face: 4 nodes at z = 0
    coords = np.array([
        [0.0, 0.0, 0.0], # Node 0 (M1)
        [2.0, 0.0, 0.0], # Node 1 (M2)
        [2.0, 2.0, 0.0], # Node 2 (M3)
        [0.0, 2.0, 0.0], # Node 3 (M4)
        [1.0, 1.0, 0.1], # Node 4 (S1 - Slave, initially elevated by z=0.1)
    ], dtype=np.float64)

    nid_to_idx = {i: i for i in range(5)}
    slave_nids = [4]
    master_faces = [(0, 1, 2, 3)]

    tie = SurfaceTieConstraint3D(
        slave_node_ids=slave_nids,
        master_faces=master_faces,
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=1e5,
        position_tolerance=0.5
    )

    u = np.zeros(15, dtype=np.float64)
    f_tie, (rows, cols, data), stats = tie.assemble(u)

    # Initial gap = 0.1 in Z
    assert abs(stats["max_gap"] - 0.1) < 1e-5
    # Slave force in Z should be k_tie * gap_z = 1e5 * 0.1 = 10000.0
    assert abs(f_tie[3 * 4 + 2] - 10000.0) < 1e-4

    # Rigid translation test: displace all nodes by [0.5, 0.5, 0.5]
    u_rigid = np.tile([0.5, 0.5, 0.5], 5)
    f_tie_rigid, _, stats_rigid = tie.assemble(u_rigid)
    # Gap remains 0.1 in Z, relative gap vector is unchanged
    assert abs(stats_rigid["max_gap"] - 0.1) < 1e-5
