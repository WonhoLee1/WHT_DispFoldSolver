"""Unit test suite for Set Operations (Union, Difference, Intersection, Overlapping)
and findAt-style proximity extraction for n nearest nodes/elements.
"""

import pytest
import numpy as np

from dispsolver.model import Model, Part, GeneralSet, ElementFace


def test_set_operations_entity_independence_and_algebra():
    """Verify that GeneralSet operations (| , - , & , union, difference, intersection)
    act on nodes, elements, and faces independently ('끼리끼리 작용').
    """
    part = Part(name="TestPart", dim=2)
    # Add 6 nodes
    for nid in range(1, 7):
        part.add_node(nid, [float(nid), 0.0])

    # Set A: nodes [1, 2, 3], elements [10, 20], faces [(10, 0)]
    s_a = GeneralSet(
        name="SET_A",
        part=part,
        nodes=[1, 2, 3],
        elements=[10, 20],
        faces=[(10, 0)]
    )

    # Set B: nodes [3, 4, 5], elements [20, 30], faces=[(20, 1)]
    s_b = GeneralSet(
        name="SET_B",
        part=part,
        nodes=[3, 4, 5],
        elements=[20, 30],
        faces=[(20, 1)]
    )

    # 1. Union (s_a | s_b and s_a.union(s_b))
    u1 = s_a | s_b
    u2 = s_a.union(s_b, name="U_NAMED")
    for u in [u1, u2]:
        assert set(u.node_ids) == {1, 2, 3, 4, 5}
        assert set(u.element_ids) == {10, 20, 30}
        assert len(u.faces) == 2
        assert {(f.element_id, f.face_id) for f in u.faces} == {(10, 0), (20, 1)}

    # 2. Intersection (s_a & s_b and s_a.intersection(s_b))
    inter1 = s_a & s_b
    inter2 = s_a.intersection(s_b, name="INTER_NAMED")
    for inter in [inter1, inter2]:
        assert set(inter.node_ids) == {3}
        assert set(inter.element_ids) == {20}
        assert len(inter.faces) == 0

    # 3. Difference (s_a - s_b and s_a.difference(s_b))
    diff1 = s_a - s_b
    diff2 = s_a.difference(s_b, name="DIFF_NAMED")
    for diff in [diff1, diff2]:
        assert set(diff.node_ids) == {1, 2}
        assert set(diff.element_ids) == {10}
        assert len(diff.faces) == 1
        assert (diff.faces[0].element_id, diff.faces[0].face_id) == (10, 0)


def test_overlapping_nodes_and_elements_extraction():
    """Verify get_overlapping_nodes and get_overlapping_elements,
    including direct node intersection vs associative element node expansion.
    """
    part = Part(name="MeshPart", dim=2)
    # Two adjacent quads along X:
    # 4 --- 5 --- 6
    # |  1  |  2  |
    # 1 --- 2 --- 3
    nodes_coords = {
        1: [0.0, 0.0],
        2: [1.0, 0.0],
        3: [2.0, 0.0],
        4: [0.0, 1.0],
        5: [1.0, 1.0],
        6: [2.0, 1.0],
    }
    for nid, c in nodes_coords.items():
        part.add_node(nid, c)

    part.add_element(1, "CPS4", [1, 2, 5, 4])
    part.add_element(2, "CPS4", [2, 3, 6, 5])

    # Element sets
    s_e1 = part.create_set("E1", elements=[1])  # nodes: 1, 2, 4, 5
    s_e2 = part.create_set("E2", elements=[2])  # nodes: 2, 3, 5, 6

    # With include_elements=False:
    # Explicit node_ids are empty in element-only sets
    assert len(s_e1.get_overlapping_nodes(s_e2, include_elements=False)) == 0

    # With include_elements=True:
    # The interface nodes between elem 1 and elem 2 are nodes 2 and 5!
    overlap_interface = s_e1.get_overlapping_nodes(s_e2, include_elements=True)
    assert np.array_equal(overlap_interface, [2, 5])

    # Part level helper
    assert np.array_equal(part.get_overlapping_nodes("E1", "E2", include_elements=True), [2, 5])

    # Explicit node sets
    s_n1 = part.create_set("N1", nodes=[1, 2, 5])
    s_n2 = part.create_set("N2", nodes=[2, 3, 5])
    assert np.array_equal(s_n1.get_overlapping_nodes(s_n2), [2, 5])
    assert np.array_equal(part.get_overlapping_nodes("N1", "N2"), [2, 5])

    # Overlapping elements
    s_comb1 = part.create_set("COMB1", elements=[1, 2])
    s_comb2 = part.create_set("COMB2", elements=[2])
    assert np.array_equal(s_comb1.get_overlapping_elements(s_comb2), [2])
    assert np.array_equal(part.get_overlapping_elements("COMB1", "COMB2"), [2])


def test_part_find_closest_nodes_and_elements():
    """Verify find_closest_nodes and find_closest_elements for distance-ordered n entities."""
    part = Part(name="GridPart", dim=2)
    # Line of nodes: x = 0, 1, 2, 3, 4, 5
    for i in range(6):
        part.add_node(i + 1, [float(i), 0.0])

    # Elements centered at x = 0.5, 1.5, 2.5, 3.5, 4.5
    for i in range(5):
        part.add_element(i + 1, "B21", [i + 1, i + 2])

    # Query point at x = 2.1, y = 0.0
    # Distances to nodes:
    # node 3 (x=2.0) -> dist = 0.1
    # node 4 (x=3.0) -> dist = 0.9
    # node 2 (x=1.0) -> dist = 1.1
    # node 5 (x=4.0) -> dist = 1.9
    # node 1 (x=0.0) -> dist = 2.1
    # node 6 (x=5.0) -> dist = 2.9

    # 1. Closest 1 node
    assert part.find_closest_nodes([2.1, 0.0], n=1) == [3]

    # 2. Closest 3 nodes
    closest_3 = part.find_closest_nodes([2.1, 0.0], n=3)
    assert closest_3 == [3, 4, 2]

    # 3. With return_distances=True
    closest_3_dist = part.find_closest_nodes([2.1, 0.0], n=3, return_distances=True)
    assert len(closest_3_dist) == 3
    assert closest_3_dist[0][0] == 3
    assert pytest.approx(closest_3_dist[0][1], abs=1e-6) == 0.1
    assert closest_3_dist[1][0] == 4
    assert pytest.approx(closest_3_dist[1][1], abs=1e-6) == 0.9

    # 4. Search tolerance bounds
    # Tolerance 0.5: only node 3 is within dist 0.1 <= 0.5
    assert part.find_closest_nodes([2.1, 0.0], n=5, search_tolerance=0.5) == [3]

    # 5. Closest elements (by centroid)
    # Centroids:
    # Elem 3: centroid = 2.5 -> dist = 0.4
    # Elem 2: centroid = 1.5 -> dist = 0.6
    # Elem 4: centroid = 3.5 -> dist = 1.4
    # Elem 1: centroid = 0.5 -> dist = 1.6
    # Elem 5: centroid = 4.5 -> dist = 2.4
    assert part.find_closest_elements([2.1, 0.0], n=1) == [3]
    assert part.find_closest_elements([2.1, 0.0], n=2) == [3, 2]

    elem_2_dist = part.find_closest_elements([2.1, 0.0], n=2, return_distances=True)
    assert elem_2_dist[0][0] == 3
    assert pytest.approx(elem_2_dist[0][1], abs=1e-6) == 0.4
    assert elem_2_dist[1][0] == 2
    assert pytest.approx(elem_2_dist[1][1], abs=1e-6) == 0.6


def test_part_find_at_general_set_creation():
    """Verify Abaqus-style find_at method for single and multiple seed points."""
    part = Part(name="PlatePart", dim=2)
    # 2 Quads:
    # 4 --- 5 --- 6
    # |  1  |  2  |
    # 1 --- 2 --- 3
    part.add_node(1, [0.0, 0.0])
    part.add_node(2, [1.0, 0.0])
    part.add_node(3, [2.0, 0.0])
    part.add_node(4, [0.0, 1.0])
    part.add_node(5, [1.0, 1.0])
    part.add_node(6, [2.0, 1.0])

    part.add_element(1, "CPS4", [1, 2, 5, 4])  # centroid = (0.5, 0.5)
    part.add_element(2, "CPS4", [2, 3, 6, 5])  # centroid = (1.5, 0.5)

    # 1. Single query point near node 1
    s_near_1 = part.find_at(coords=[0.05, 0.02], name="NEAR_NODE_1", entity_type="NODES", n=1)
    assert set(s_near_1.node_ids) == {1}
    assert len(s_near_1.element_ids) == 0
    assert "NEAR_NODE_1" in part.sets

    # 2. Pick 2 nearest nodes
    s_near_2nodes = part.find_at(coords=[0.05, 0.02], entity_type="NODES", n=2)
    # Nearest are node 1 (dist~0.05) and node 2 (dist~0.95) or node 4 (dist~0.98)
    assert 1 in s_near_2nodes.node_ids
    assert len(s_near_2nodes.node_ids) == 2

    # 3. Multiple query points picking elements
    s_both_elems = part.find_at(
        coords=[[0.4, 0.4], [1.6, 0.6]],
        name="BOTH_ELEMS",
        entity_type="ELEMENTS",
        n=1
    )
    assert set(s_both_elems.element_ids) == {1, 2}
    assert len(s_both_elems.node_ids) == 0
    assert "BOTH_ELEMS" in part.sets
    assert "BOTH_ELEMS" in part.element_sets


def test_part_boolean_operations():
    """Verify part.boolean_union, boolean_intersection, and boolean_difference."""
    part = Part(name="BoolPart", dim=2)
    for nid in range(1, 10):
        part.add_node(nid, [float(nid), 0.0])

    part.create_node_set("SET1", [1, 2, 3, 4])
    part.create_node_set("SET2", [3, 4, 5, 6])

    # Boolean union
    u = part.boolean_union(name="UNION_12", set_a="SET1", set_b="SET2")
    assert set(u.node_ids) == {1, 2, 3, 4, 5, 6}
    assert "UNION_12" in part.sets

    # Boolean intersection
    inter = part.boolean_intersection(name="INTER_12", set_a="SET1", set_b="SET2")
    assert set(inter.node_ids) == {3, 4}
    assert "INTER_12" in part.sets

    # Boolean difference
    diff = part.boolean_difference(name="DIFF_12", set_a="SET1", set_b="SET2")
    assert set(diff.node_ids) == {1, 2}
    assert "DIFF_12" in part.sets
