"""Unit test suite for Proximity Search and Feature Angle Flood Fill Propagation."""

import pytest
import numpy as np
from dispsolver.model import Part, GeneralSet


def test_closest_node_and_element_and_k_nearest():
    """Verify finding closest node/element and k-nearest set creation."""
    part = Part(name="ProximityPart", dim=3)
    # 4 nodes along X: 0.0, 1.0, 2.0, 3.0
    for nid, x in enumerate([0.0, 1.0, 2.0, 3.0], start=1):
        part.add_node(nid, [x, 0.0, 0.0])

    # Query point at x = 1.1 -> closest node should be node 2 (at x=1.0)
    best_nid, dist = part.get_closest_node([1.1, 0.0, 0.0])
    assert best_nid == 2
    assert np.isclose(dist, 0.1)

    # 2 elements
    part.add_element(1, "B31", [1, 2])  # centroid at x=0.5
    part.add_element(2, "B31", [2, 3])  # centroid at x=1.5

    # Query at x=1.4 -> closest element should be element 2 (centroid 1.5, dist 0.1)
    best_eid, dist_e = part.get_closest_element([1.4, 0.0, 0.0])
    assert best_eid == 2
    assert np.isclose(dist_e, 0.1)

    # k=2 nearest nodes to x=1.8 -> should be node 2 (x=1.0) and node 3 (x=2.0)
    k_nodes_set = part.create_set_from_k_nearest(name="K2_NODES", coords=[1.8, 0.0, 0.0], k=2, entity_type="NODES")
    assert set(k_nodes_set.get_nodes()) == {2, 3}


def test_angle_propagation_2d_l_bracket():
    """Verify that 2D angle propagation stops at sharp 90-degree corner."""
    part = Part(name="LBracket2D", dim=2)
    # L-bracket mesh with 3 Quad elements:
    # Elem 1: x in [0, 1], y in [0, 1]
    # Elem 2: x in [1, 2], y in [0, 1]  (horizontal wing)
    # Elem 3: x in [0, 1], y in [1, 2]  (vertical wing)
    #
    # 7 --- 8
    # |  3  |
    # 3 --- 4 --- 6
    # |  1  |  2  |
    # 0 --- 1 --- 5
    nodes_coords = {
        0: [0.0, 0.0], 1: [1.0, 0.0], 5: [2.0, 0.0],
        3: [0.0, 1.0], 4: [1.0, 1.0], 6: [2.0, 1.0],
        7: [0.0, 2.0], 8: [1.0, 2.0]
    }
    for nid, (x, y) in nodes_coords.items():
        part.add_node(nid, [x, y])

    part.add_element(1, "CPS4", [0, 1, 4, 3])
    part.add_element(2, "CPS4", [1, 5, 6, 4])
    part.add_element(3, "CPS4", [3, 4, 8, 7])

    # Horizontal step top edge is from node 4 to node 6 (y=1, x in [1, 2])
    # Seed coords at (1.5, 1.0)
    # Normal of edge (4, 6) points +Y [0, 1]
    # At node 4, the vertical wall edge (4, 8) has normal pointing +X [1, 0]
    # Angle difference between [0, 1] and [1, 0] is 90 degrees!
    # Propagation with feature_angle_deg=20.0 should STOP at the corner and NOT include edge (4, 8)!
    step_top = part.create_surface_by_angle(name="STEP_TOP", seed_coords=[1.5, 1.0], feature_angle_deg=20.0)

    # The step top should only contain edge (4, 6) with nodes {4, 6}
    assert set(step_top.get_nodes()) == {4, 6}
    assert len(step_top.faces) == 1

    # Vertical top edge at y=2 is (7, 8). Seed coords at (0.5, 2.0)
    vert_top = part.create_surface_by_angle(name="VERT_TOP", seed_coords=[0.5, 2.0], feature_angle_deg=20.0)
    assert set(vert_top.get_nodes()) == {7, 8}
    assert len(vert_top.faces) == 1


def test_angle_propagation_3d_cube():
    """Verify that 3D angle propagation stops at sharp 90-degree corners of a cube."""
    part = Part(name="Cube3D", dim=3)
    coords = [
        (1, 0, 0, 0), (2, 1, 0, 0), (3, 1, 1, 0), (4, 0, 1, 0),
        (5, 0, 0, 1), (6, 1, 0, 1), (7, 1, 1, 1), (8, 0, 1, 1)
    ]
    for nid, x, y, z in coords:
        part.add_node(nid, [x, y, z])
    part.add_element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    # Seed on Top face (+Z, z=1.0)
    top_surf = part.create_surface_by_angle(name="CUBE_TOP", seed_coords=[0.5, 0.5, 1.0], feature_angle_deg=30.0)
    # All other adjacent faces are perpendicular (90 degrees > 30 degrees)
    # So it should only contain the Top face (nodes 5, 6, 7, 8)
    assert set(top_surf.get_nodes()) == {5, 6, 7, 8}
    assert len(top_surf.faces) == 1
    assert top_surf.faces[0].face_id == 1


def test_angle_propagation_curved_smooth_surface():
    """Verify that angle propagation successfully traverses a curved surface with small inter-element angles."""
    part = Part(name="CurvedStrip", dim=2)
    # 5 quads forming a curved arc, each bent by 10 degrees (total 50 degrees bend)
    # Quad k spans angle theta_k to theta_{k+1}, radius from R_in=5 to R_out=6
    R_in = 5.0
    R_out = 6.0
    n_segs = 5
    d_theta = np.radians(10.0)  # 10 degrees between adjacent segments

    nid = 1
    node_map = {}
    for i in range(n_segs + 1):
        th = i * d_theta
        # Inner node
        xi, yi = R_in * np.cos(th), R_in * np.sin(th)
        part.add_node(nid, [xi, yi])
        node_map[(i, "in")] = nid
        nid += 1
        # Outer node
        xo, yo = R_out * np.cos(th), R_out * np.sin(th)
        part.add_node(nid, [xo, yo])
        node_map[(i, "out")] = nid
        nid += 1

    for i in range(n_segs):
        n1 = node_map[(i, "in")]
        n2 = node_map[(i + 1, "in")]
        n3 = node_map[(i + 1, "out")]
        n4 = node_map[(i, "out")]
        part.add_element(i + 1, "CPS4", [n1, n2, n3, n4])

    # Propagate along the outer circular rim (normal pointing outwards radially)
    # Between adjacent segments, normal angle diff is exactly 10 degrees!
    # If feature_angle_deg is 15.0 (which is > 10.0), it should propagate across ALL 5 segments!
    seed_x = R_out * np.cos(0.5 * d_theta)
    seed_y = R_out * np.sin(0.5 * d_theta)
    outer_arc = part.create_surface_by_angle(name="OUTER_ARC", seed_coords=[seed_x, seed_y], feature_angle_deg=15.0)

    # All 5 outer edges should be selected!
    assert len(outer_arc.faces) == 5
    # All outer nodes should be present: (n_segs + 1) = 6 nodes
    expected_outer_nodes = {node_map[(i, "out")] for i in range(n_segs + 1)}
    assert set(outer_arc.get_nodes()) == expected_outer_nodes

    # But if feature_angle_deg is 5.0 (< 10.0), it should stop at the first segment!
    single_seg = part.create_surface_by_angle(name="SINGLE_SEG", seed_coords=[seed_x, seed_y], feature_angle_deg=5.0)
    assert len(single_seg.faces) == 1
