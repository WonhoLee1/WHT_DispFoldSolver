"""Unit test suite for search tolerance, orthogonal face projection, multiple seeds, and boundary guards."""

import pytest
import numpy as np
from dispsolver.model import Part, GeneralSet


def test_search_tolerance_and_raise_if_none():
    """Verify search_tolerance bounds closest node/element and triggers raise_if_none."""
    part = Part(name="TolPart", dim=3)
    part.add_node(1, [0.0, 0.0, 0.0])
    part.add_node(2, [1.0, 0.0, 0.0])
    part.add_element(1, "B31", [1, 2])

    # Query at x=1.05 (dist=0.05 to node 2)
    # Case A: search_tolerance=0.1 (within tolerance)
    nid, dist = part.get_closest_node([1.05, 0.0, 0.0], search_tolerance=0.1)
    assert nid == 2
    assert np.isclose(dist, 0.05)

    # Case B: search_tolerance=0.01 (exceeds tolerance) -> returns None
    nid_none, dist_ex = part.get_closest_node([1.05, 0.0, 0.0], search_tolerance=0.01)
    assert nid_none is None
    assert np.isclose(dist_ex, 0.05)

    # Case C: raise_if_none=True with exceeding tolerance -> raises ValueError
    with pytest.raises(ValueError, match="No node found within search_tolerance"):
        part.get_closest_node([1.05, 0.0, 0.0], search_tolerance=0.01, raise_if_none=True)

    # Case D: get_closest_element with search_tolerance
    eid, dist_eid = part.get_closest_element([2.0, 0.0, 0.0], search_tolerance=0.5)
    assert eid is None

    # Case E: create_set_from_k_nearest with search_tolerance
    k_set = part.create_set_from_k_nearest(
        name="NEAR_NODE",
        coords=[1.05, 0.0, 0.0],
        k=2,
        search_tolerance=0.1
    )
    # Only node 2 is within 0.1 (node 1 is at dist=1.05 > 0.1)
    assert set(k_set.get_nodes()) == {2}


def test_orthogonal_face_projection_abaqus_findat():
    """Verify that a query point right in the interior of a Hex face is matched via orthogonal projection."""
    part = Part(name="BlockFindAt", dim=3)
    # 1 Hex element from (0,0,0) to (10,10,1)
    coords = [
        (1, 0, 0, 0), (2, 10, 0, 0), (3, 10, 10, 0), (4, 0, 10, 0),
        (5, 0, 0, 1), (6, 10, 0, 1), (7, 10, 10, 1), (8, 0, 10, 1)
    ]
    for nid, x, y, z in coords:
        part.add_node(nid, [x, y, z])
    part.add_element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    # Query point at (5.0, 5.0, 1.02) - right above the center of the top face (+Z, z=1.0)
    # Height offset = 0.02, exactly in the middle of face interior
    top_surf = part.create_surface_by_angle(
        name="TOP_SURF",
        seed_coords=[5.0, 5.0, 1.02],
        feature_angle_deg=20.0,
        search_tolerance=0.05
    )
    # Should find the top face (nodes 5, 6, 7, 8)
    assert set(top_surf.get_nodes()) == {5, 6, 7, 8}
    assert len(top_surf.faces) == 1
    assert top_surf.faces[0].face_id == 1


def test_multiple_seed_points():
    """Verify specifying a list of seed coordinates to select separate surfaces in one call."""
    part = Part(name="TwoBlocks", dim=2)
    # Block A: x in [0, 1], y in [0, 1]
    part.add_node(1, [0.0, 0.0]); part.add_node(2, [1.0, 0.0])
    part.add_node(3, [1.0, 1.0]); part.add_node(4, [0.0, 1.0])
    part.add_element(1, "CPS4", [1, 2, 3, 4])

    # Block B (separated): x in [5, 6], y in [0, 1]
    part.add_node(5, [5.0, 0.0]); part.add_node(6, [6.0, 0.0])
    part.add_node(7, [6.0, 1.0]); part.add_node(8, [5.0, 1.0])
    part.add_element(2, "CPS4", [5, 6, 7, 8])

    # Pass 2 seed coordinates for top surfaces of both blocks
    combined_top = part.create_surface_by_angle(
        name="BOTH_TOPS",
        seed_coords=[[0.5, 1.0], [5.5, 1.0]],
        feature_angle_deg=20.0
    )
    # Top edge of Block A is (4, 3); top edge of Block B is (8, 7)
    assert set(combined_top.get_nodes()) == {3, 4, 7, 8}
    assert len(combined_top.faces) == 2


def test_max_distance_cutoff():
    """Verify that propagation halts when distance from seed exceeds max_distance."""
    part = Part(name="LongStrip", dim=2)
    # 5 Quads in a row along X: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], y in [0, 1]
    for i in range(6):
        part.add_node(2 * i + 1, [float(i), 0.0])
        part.add_node(2 * i + 2, [float(i), 1.0])

    for i in range(5):
        n1 = 2 * i + 1
        n2 = 2 * (i + 1) + 1
        n3 = 2 * (i + 1) + 2
        n4 = 2 * i + 2
        part.add_element(i + 1, "CPS4", [n1, n2, n3, n4])

    # Propagate on top edge from seed (0.5, 1.0) with max_distance = 2.2
    # Face centroids of top edges are at x = 0.5, 1.5, 2.5, 3.5, 4.5
    # Distances from (0.5, 1.0):
    # elem 1: dist=0.0 <= 2.2
    # elem 2: dist=1.0 <= 2.2
    # elem 3: dist=2.0 <= 2.2
    # elem 4: dist=3.0 > 2.2 (excluded!)
    cutoff_surf = part.create_surface_by_angle(
        name="CUTOFF_TOP",
        seed_coords=[0.5, 1.0],
        feature_angle_deg=20.0,
        max_distance=2.2
    )
    # Only the first 3 top edges should be selected (elems 1, 2, 3)
    assert len(cutoff_surf.faces) == 3
    # Selected nodes should be along y=1 for x=0, 1, 2, 3 -> nodes 2, 4, 6, 8
    assert set(cutoff_surf.get_nodes()) == {2, 4, 6, 8}


def test_stop_at_nodes_barrier():
    """Verify that propagation is blocked at user-defined barrier nodes (partition/weld line)."""
    part = Part(name="PartitionedPlate", dim=2)
    # 4 Quads in a row along X
    for i in range(5):
        part.add_node(2 * i + 1, [float(i), 0.0])
        part.add_node(2 * i + 2, [float(i), 1.0])

    for i in range(4):
        part.add_element(i + 1, "CPS4", [2 * i + 1, 2 * i + 3, 2 * i + 4, 2 * i + 2])

    # Define barrier nodes at x=2.0: nodes 5 (bottom) and 6 (top)
    barrier_nodes = {6}

    # Start propagation from top of elem 1 (seed at x=0.5, y=1.0)
    # Even though all top faces are in a straight line (angle diff = 0),
    # propagation should STOP before crossing node 6!
    blocked_surf = part.create_surface_by_angle(
        name="BLOCKED_TOP",
        seed_coords=[0.5, 1.0],
        feature_angle_deg=20.0,
        stop_at_nodes=barrier_nodes
    )
    # Should only contain top faces for elem 1 and elem 2 (before crossing past node 6)
    assert len(blocked_surf.faces) == 2
    assert set(blocked_surf.get_nodes()) == {2, 4, 6}
    assert 8 not in set(blocked_surf.get_nodes())


def test_thin_sheet_wrap_around_guard():
    """Verify target_normal prevents picking opposite face in ultra-thin 0.1mm sheet."""
    part = Part(name="ThinSheet", dim=2)
    # Thin strip: thickness = 0.1mm, length = 10mm
    # y=0 is bottom, y=0.1 is top
    part.add_node(1, [0.0, 0.0]); part.add_node(2, [10.0, 0.0])
    part.add_node(3, [10.0, 0.1]); part.add_node(4, [0.0, 0.1])
    part.add_element(1, "CPS4", [1, 2, 3, 4])

    # Top surface has outward normal [0, 1]
    # Bottom surface has outward normal [0, -1]
    # If we specify target_normal=[0, 1] with seed at y=0.1:
    top_surf = part.create_surface_by_angle(
        name="TOP_ONLY",
        seed_coords=[5.0, 0.1],
        target_normal=[0.0, 1.0],
        feature_angle_deg=45.0
    )
    assert set(top_surf.get_nodes()) == {3, 4}
    assert len(top_surf.faces) == 1
