"""
test_rigid_plate_tie.py
======================
Unit tests for Plate Builder, RigidBodyPart, and SurfaceTieConstraint.
"""

import numpy as np
import pytest

from dispsolver.mesh.plate_builder import create_folding_plate_parts
from dispsolver.part.rigid_body import RigidBodyPart
from dispsolver.constraint.surface_tie import SurfaceTieConstraint


def test_plate_builder_geometry():
    """Verify folding plate mesh generation."""
    plates = create_folding_plate_parts(
        left_x_range=(-40.0, -10.0),
        right_x_range=(10.0, 40.0),
        y_range=(-0.5, 0.0),
        nx=10,
        ny=2,
    )

    assert "left" in plates
    assert "right" in plates

    left = plates["left"]
    # base_node_id/base_elem_id default raised 10000 -> 100000 this
    # session (node-id overflow fix, see AGENTS.md / dev_log) to give the
    # display mesh 10x the headroom before colliding with the reserved
    # plate-node range.
    assert left["master_rp_id"] == 100000
    assert len(left["top_surface_nids"]) == 11  # nx + 1
    assert len(left["slave_nids"]) == 11 * 3   # (nx + 1) * (ny + 1)


def test_rigid_body_part_kinematics():
    """Verify rigid body part kinematics."""
    master_coord = np.array([0.0, 0.0])
    slave_coords = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    rb = RigidBodyPart(
        name="TEST_RIGID",
        master_id=1,
        slave_ids=[2, 3],
        master_coord=master_coord,
        slave_coords=slave_coords,
    )

    assert rb.n_slaves == 2

    # Pure translation (1, 2)
    u_slaves_trans = rb.get_slave_displacements(u_master=np.array([1.0, 2.0]), theta=0.0)
    np.testing.assert_allclose(u_slaves_trans[0:2], [1.0, 2.0])
    np.testing.assert_allclose(u_slaves_trans[2:4], [1.0, 2.0])

    # Pure 90 deg rotation around (0,0)
    # Node (1, 0) -> (0, 1) => disp (-1, 1)
    # Node (0, 1) -> (-1, 0) => disp (-1, -1)
    u_slaves_rot = rb.get_slave_displacements(u_master=np.array([0.0, 0.0]), theta=np.pi / 2.0)
    np.testing.assert_allclose(u_slaves_rot[0:2], [-1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(u_slaves_rot[2:4], [-1.0, -1.0], atol=1e-12)


def test_surface_tie_constraint():
    """Verify surface tie constraint assembly and energy calculation."""
    # Simple setup: 1 slave node at (5, 0.1), master segment from (0, 0) to (10, 0)
    coords = np.array([
        [5.0, 0.1],  # Node 0 (Slave)
        [0.0, 0.0],  # Node 1 (Master 1)
        [10.0, 0.0], # Node 2 (Master 2)
    ])
    nid_to_idx = {0: 0, 1: 1, 2: 2}

    tie = SurfaceTieConstraint(
        slave_node_ids=[0],
        master_node_ids=[1, 2],
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=1e4,
    )

    assert tie.n_active == 1

    u = np.zeros(6)
    f_int = np.zeros(6)
    K_eff = np.zeros((6, 6))

    energy = tie.apply_penalty(u, f_int, K_eff)

    # Initial gap = (5, 0.1) - (5, 0) = (0, 0.1)
    # Energy = 0.5 * k * gap^2 = 0.5 * 1e4 * 0.01 = 50.0
    assert np.isclose(energy, 50.0)
    assert np.abs(f_int[1]) > 0.0  # Vertical force on slave
