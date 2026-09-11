"""Unit test suite for GeneralSet (Smart Unified Region) functionality."""

import pytest
import numpy as np
from dispsolver.model import Model, Part, GeneralSet, SolidSection, DisplacementBC


def test_general_set_basic_and_associative_nodes():
    """Verify that GeneralSet resolves nodes associatively from elements."""
    part = Part(name="Plate", dim=2)
    # 2 Quads side-by-side along X
    # Nodes:
    # 3 --- 4 --- 5
    # |  1  |  2  |
    # 0 --- 1 --- 2
    coords = [
        (0, 0.0, 0.0),
        (1, 1.0, 0.0),
        (2, 2.0, 0.0),
        (3, 0.0, 1.0),
        (4, 1.0, 1.0),
        (5, 2.0, 1.0),
    ]
    for nid, x, y in coords:
        part.add_node(nid, [x, y])

    part.add_element(1, "CPS4", [0, 1, 4, 3])
    part.add_element(2, "CPS4", [1, 2, 5, 4])

    # Case A: Explicit node set
    s_nodes = part.create_set("ONLY_NODES", nodes=[0, 3])
    assert np.array_equal(s_nodes.get_nodes(), [0, 3])
    assert len(s_nodes.get_elements()) == 0

    # Case B: Element set -> associative node resolution
    s_elem1 = part.create_set("ELEM_1", elements=[1])
    assert np.array_equal(s_elem1.get_elements(), [1])
    # Nodes belonging to element 1 are [0, 1, 3, 4]
    assert np.array_equal(s_elem1.get_nodes(include_elements=True), [0, 1, 3, 4])
    assert len(s_elem1.get_nodes(include_elements=False)) == 0


def test_general_set_2d_exterior_cancellation():
    """Verify that shared internal edges cancel out to yield only exterior boundary segments."""
    part = Part(name="Plate2D", dim=2)
    for nid, (x, y) in enumerate([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]):
        part.add_node(nid, [float(x), float(y)])

    part.add_element(1, "CPS4", [0, 1, 4, 3])
    part.add_element(2, "CPS4", [1, 2, 5, 4])

    all_elems = part.create_set("ALL_ELEMS", elements=[1, 2])
    
    # Exterior faces: total 2 elements * 4 edges = 8 edges.
    # The shared edge is between node 1 and node 4.
    # Interior edge (1, 4) is shared by elem 1 and elem 2 -> cancels out!
    # Expected exterior edges count = 8 - 2 = 6 edges.
    exterior_faces = all_elems.get_faces(exterior_only=True)
    assert len(exterior_faces) == 6

    # get_segments returns 2D tuples (node_a, node_b)
    segments = all_elems.get_segments()
    assert len(segments) == 6

    # The shared edge (1, 4) or (4, 1) must NOT be in the exterior segments
    sorted_segs = [tuple(sorted(seg)) for seg in segments]
    assert (1, 4) not in sorted_segs
    assert (0, 1) in sorted_segs
    assert (1, 2) in sorted_segs
    assert (2, 5) in sorted_segs
    assert (4, 5) in sorted_segs
    assert (3, 4) in sorted_segs
    assert (0, 3) in sorted_segs


def test_general_set_3d_hex_exterior_cancellation():
    """Verify that shared internal faces in 3D Hex mesh cancel out."""
    part = Part(name="Bar3D", dim=3)
    # 2 Hex8 elements along X
    node_grid = {}
    nid = 1
    for i in range(3):
        for j in range(2):
            for k in range(2):
                part.add_node(nid, [float(i), float(j), float(k)])
                node_grid[(i, j, k)] = nid
                nid += 1

    for i in range(2):
        conn = [
            node_grid[(i, 0, 0)],
            node_grid[(i + 1, 0, 0)],
            node_grid[(i + 1, 1, 0)],
            node_grid[(i, 1, 0)],
            node_grid[(i, 0, 1)],
            node_grid[(i + 1, 0, 1)],
            node_grid[(i + 1, 1, 1)],
            node_grid[(i, 1, 1)],
        ]
        part.add_element(i + 1, "C3D8", conn)

    hex_set = part.create_set("TWO_HEX", elements=[1, 2])
    # 2 Hex8 * 6 faces = 12 total faces
    # The interface at x=1 is shared between elem 1 (Right face) and elem 2 (Left face)
    # Exterior faces should be 12 - 2 = 10 faces
    ext_faces = hex_set.get_faces(exterior_only=True)
    assert len(ext_faces) == 10

    all_faces = hex_set.get_faces(exterior_only=False)
    assert len(all_faces) == 12


def test_general_set_bounding_box():
    """Verify geometric bounding-box filtering."""
    part = Part(name="GridPart", dim=2)
    # 3x3 grid of nodes from x=0..2, y=0..2
    nid = 1
    for y in [0.0, 1.0, 2.0]:
        for x in [0.0, 1.0, 2.0]:
            part.add_node(nid, [x, y])
            nid += 1

    # 4 elements
    part.add_element(1, "CPS4", [1, 2, 5, 4])  # x in [0, 1], y in [0, 1]
    part.add_element(2, "CPS4", [2, 3, 6, 5])  # x in [1, 2], y in [0, 1]
    part.add_element(3, "CPS4", [4, 5, 8, 7])  # x in [0, 1], y in [1, 2]
    part.add_element(4, "CPS4", [5, 6, 9, 8])  # x in [1, 2], y in [1, 2]

    # Filter left half: x in [0.0, 0.9]
    left_set = part.create_set_from_box(
        name="LEFT_HALF",
        x_range=(0.0, 0.9),
        y_range=(0.0, 2.0),
        entity_type="ALL"
    )
    # Nodes with x=0.0 are 1, 4, 7
    assert set(left_set.node_ids) == {1, 4, 7}
    # Centroids of elem 1 & 3 are at x=0.5 -> within [0, 0.9]
    assert set(left_set.element_ids) == {1, 3}

    # Test associative resolution of left_set:
    # Elem 1 & 3 nodes are 1, 2, 4, 5, 7, 8
    nodes_resolved = left_set.get_nodes(include_elements=True)
    assert set(nodes_resolved) == {1, 2, 4, 5, 7, 8}


def test_general_set_algebra():
    """Verify set union, intersection, and difference operators."""
    part = Part(name="AlgebraPart", dim=2)
    for nid in range(1, 11):
        part.add_node(nid, [float(nid), 0.0])

    s1 = part.create_set("SET1", nodes=[1, 2, 3, 4, 5], elements=[10, 20])
    s2 = part.create_set("SET2", nodes=[4, 5, 6, 7], elements=[20, 30])

    # Union
    union_set = s1 | s2
    assert set(union_set.node_ids) == {1, 2, 3, 4, 5, 6, 7}
    assert set(union_set.element_ids) == {10, 20, 30}

    # Intersection
    inter_set = s1 & s2
    assert set(inter_set.node_ids) == {4, 5}
    assert set(inter_set.element_ids) == {20}

    # Difference
    diff_set = s1 - s2
    assert set(diff_set.node_ids) == {1, 2, 3}
    assert set(diff_set.element_ids) == {10}


def test_general_set_integration_with_model():
    """Verify GeneralSet usage in section assignment and BC application in Model hierarchy."""
    model = Model(name="GeneralSetIntegration", dim=3)
    mat = model.Material(name="Steel", mat_type="ELASTIC")
    mat.elastic = (210000.0, 0.3)
    sec = model.SolidSection(name="SteelSec", material="Steel")

    part = model.Part(name="Bar", dim=3)
    # 1 Hex element
    coords = [
        (1, 0, 0, 0), (2, 1, 0, 0), (3, 1, 1, 0), (4, 0, 1, 0),
        (5, 0, 0, 1), (6, 1, 0, 1), (7, 1, 1, 1), (8, 0, 1, 1)
    ]
    for nid, x, y, z in coords:
        part.add_node(nid, [x, y, z])
    part.add_element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    # Create GeneralSet for element and pass the object directly to assign_section
    elem_region = part.create_set("BODY_ELEM", elements=[1])
    part.assign_section(elem_region, "SteelSec")

    # Create GeneralSet for root nodes using bounding box
    root_region = part.create_set_from_box("ROOT_NODES", x_range=(0.0, 0.01), entity_type="NODES")
    assert set(root_region.node_ids) == {1, 4, 5, 8}

    inst = model.root_assembly.create_instance("BAR_INST", part=part)

    # Step with BC referencing the GeneralSet via INSTANCE.SET
    step = model.Step(name="Step-1", procedure="STATIC")
    bc = DisplacementBC(name="FIX_X0", region="BAR_INST.ROOT_NODES", u1=0.0)
    step.add_boundary_condition(bc)

    # Build solver system
    solver, sys = model.create_solver3d("Step-1")

    # Verify that BAR_INST.ROOT_NODES is present in global_nsets
    assert "BAR_INST.ROOT_NODES" in sys.global_nsets
    assert len(sys.global_nsets["BAR_INST.ROOT_NODES"]) == 4

    # Verify that BAR_INST.BODY_ELEM is present in global_elsets
    assert "BAR_INST.BODY_ELEM" in sys.global_elsets
    assert len(sys.global_elsets["BAR_INST.BODY_ELEM"]) == 1

    # Verify that solver fixed DOFs are correctly mapped
    for nid in [1, 4, 5, 8]:
        gid = sys.node_local_to_global[("BAR_INST", nid)]
        idx = sys.nid_to_idx[gid]
        assert 3 * idx + 0 in solver.fixed_dofs
