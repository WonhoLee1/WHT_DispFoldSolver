"""Unit test suite for advanced geometric region filtering (Sphere, Cylinder, Plane, Normal, Predicate)."""

import pytest
import numpy as np
from dispsolver.model import Part, GeneralSet


def test_sphere_and_hollow_sphere_filtering():
    """Verify spherical region and hollow spherical shell filtering."""
    part = Part(name="SphereTestPart", dim=3)
    # Create grid of points at radii 0.0, 1.0, 2.0, 3.0 along X, Y, Z
    nid = 1
    for r in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        part.add_node(nid, [r, 0.0, 0.0])
        nid += 1
        if r > 0:
            part.add_node(nid, [-r, 0.0, 0.0])
            nid += 1

    # Case A: Solid sphere radius <= 1.2
    s_solid = part.create_set_from_sphere(name="SOLID_SPHERE", center=[0.0, 0.0, 0.0], radius=1.2)
    # Coordinates inside 1.2: 0.0, 0.5, -0.5, 1.0, -1.0
    nodes_solid = s_solid.get_nodes()
    for nid_val in nodes_solid:
        coords = part.nodes[nid_val]
        assert np.linalg.norm(coords) <= 1.2 + 1e-6

    # Verify that r=2.0 is NOT included
    for nid_val in nodes_solid:
        assert abs(part.nodes[nid_val][0]) < 1.2

    # Case B: Hollow spherical shell: 0.8 <= r <= 2.2
    s_hollow = part.create_set_from_sphere(
        name="HOLLOW_SPHERE",
        center=[0.0, 0.0, 0.0],
        radius=2.2,
        inner_radius=0.8
    )
    nodes_hollow = s_hollow.get_nodes()
    for nid_val in nodes_hollow:
        dist = np.linalg.norm(part.nodes[nid_val])
        assert 0.8 - 1e-6 <= dist <= 2.2 + 1e-6

    # Node at origin (r=0) must NOT be in hollow shell
    for nid_val in nodes_hollow:
        assert np.linalg.norm(part.nodes[nid_val]) >= 0.8 - 1e-6


def test_cylinder_and_hinge_axis_filtering():
    """Verify finite cylinder filtering along an arbitrary axis."""
    part = Part(name="HingeCylinderPart", dim=3)
    # Hinge axis along Z from z=0 to z=10, centered at (x=3, y=0)
    p1 = [3.0, 0.0, 0.0]
    p2 = [3.0, 0.0, 10.0]

    # Add test nodes:
    # 1. Inside cylinder: (3.5, 0, 5) -> dist from axis = 0.5 <= 1.0, z=5 in [0, 10]
    part.add_node(1, [3.5, 0.0, 5.0])
    # 2. Outside cylinder (radius too large): (5.0, 0, 5) -> dist = 2.0 > 1.0
    part.add_node(2, [5.0, 0.0, 5.0])
    # 3. Outside cylinder (z beyond endpoint): (3.2, 0, 12.0) -> z=12 > 10
    part.add_node(3, [3.2, 0.0, 12.0])
    # 4. Outside cylinder (z before startpoint): (3.2, 0, -1.0) -> z=-1 < 0
    part.add_node(4, [3.2, 0.0, -1.0])
    # 5. Inside hollow shell: (4.0, 0, 5) -> dist = 1.0
    part.add_node(5, [4.0, 0.0, 5.0])

    cyl_set = part.create_set_from_cylinder(
        name="HINGE_PIN",
        point1=p1,
        point2=p2,
        radius=1.2,
        inner_radius=0.0
    )
    nodes = set(cyl_set.get_nodes())
    assert 1 in nodes
    assert 5 in nodes
    assert 2 not in nodes
    assert 3 not in nodes
    assert 4 not in nodes

    # Hollow pipe cylinder: 0.8 <= r <= 1.2
    pipe_set = part.create_set_from_cylinder(
        name="HINGE_BUSHING",
        point1=p1,
        point2=p2,
        radius=1.2,
        inner_radius=0.8
    )
    pipe_nodes = set(pipe_set.get_nodes())
    assert 5 in pipe_nodes  # dist=1.0 is in [0.8, 1.2]
    assert 1 not in pipe_nodes  # dist=0.5 is < 0.8


def test_plane_and_half_space_filtering():
    """Verify plane cutting and half-space (symmetry BC) filtering."""
    part = Part(name="SymmetryPart", dim=3)
    # Nodes with x in [-2, -1, 0, 1, 2]
    nid = 1
    for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        for y in [0.0, 1.0]:
            part.add_node(nid, [x, y, 0.0])
            nid += 1

    # Case A: On Plane X=0 (Symmetry plane)
    sym_set = part.create_set_from_plane(
        name="XSYMM_NODES",
        point=[0.0, 0.0, 0.0],
        normal=[1.0, 0.0, 0.0],
        side="ON_PLANE",
        tol=1e-5
    )
    sym_nodes = sym_set.get_nodes()
    assert len(sym_nodes) == 2
    for n in sym_nodes:
        assert np.isclose(part.nodes[n][0], 0.0)

    # Case B: Positive Half-Space (X >= 0)
    pos_set = part.create_set_from_plane(
        name="RIGHT_HALF",
        point=[0.0, 0.0, 0.0],
        normal=[1.0, 0.0, 0.0],
        side="POSITIVE"
    )
    pos_nodes = pos_set.get_nodes()
    assert len(pos_nodes) == 6  # x=0, x=1, x=2 each has 2 nodes -> 6 nodes
    for n in pos_nodes:
        assert part.nodes[n][0] >= -1e-6

    # Case C: Negative Half-Space (X <= 0)
    neg_set = part.create_set_from_plane(
        name="LEFT_HALF",
        point=[0.0, 0.0, 0.0],
        normal=[1.0, 0.0, 0.0],
        side="NEGATIVE"
    )
    neg_nodes = neg_set.get_nodes()
    assert len(neg_nodes) == 6  # x=-2, x=-1, x=0 each has 2 nodes -> 6 nodes
    for n in neg_nodes:
        assert part.nodes[n][0] <= 1e-6


def test_surface_from_normal_2d():
    """Verify extracting 2D top (+Y) and bottom (-Y) boundary edges via outward normal."""
    part = Part(name="Strip2D", dim=2)
    # 2 Quad elements in a row: x in [0, 2], y in [0, 1]
    # 3 --- 4 --- 5
    # |  1  |  2  |
    # 0 --- 1 --- 2
    coords = [
        (0, 0.0, 0.0), (1, 1.0, 0.0), (2, 2.0, 0.0),
        (3, 0.0, 1.0), (4, 1.0, 1.0), (5, 2.0, 1.0),
    ]
    for nid, x, y in coords:
        part.add_node(nid, [x, y])

    part.add_element(1, "CPS4", [0, 1, 4, 3])
    part.add_element(2, "CPS4", [1, 2, 5, 4])

    # Top surface: normal pointing along +Y [0, 1]
    top_surf = part.create_surface_from_normal(name="TOP_SURF", direction=[0.0, 1.0], angle_tol_deg=10.0)
    # Should include edges along y=1: (3, 4) and (4, 5) -> nodes 3, 4, 5
    top_nodes = set(top_surf.get_nodes())
    assert top_nodes == {3, 4, 5}
    assert len(top_surf.faces) == 2

    # Bottom surface: normal pointing along -Y [0, -1]
    bot_surf = part.create_surface_from_normal(name="BOT_SURF", direction=[0.0, -1.0], angle_tol_deg=10.0)
    bot_nodes = set(bot_surf.get_nodes())
    assert bot_nodes == {0, 1, 2}
    assert len(bot_surf.faces) == 2


def test_surface_from_normal_3d():
    """Verify extracting 3D top (+Z) and side (-X) boundary faces via outward normal."""
    part = Part(name="Block3D", dim=3)
    # 1 Hex element: x in [0, 1], y in [0, 1], z in [0, 1]
    coords = [
        (1, 0, 0, 0), (2, 1, 0, 0), (3, 1, 1, 0), (4, 0, 1, 0),
        (5, 0, 0, 1), (6, 1, 0, 1), (7, 1, 1, 1), (8, 0, 1, 1)
    ]
    for nid, x, y, z in coords:
        part.add_node(nid, [x, y, z])
    part.add_element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])

    # Top surface (+Z): face 1 in Hex8 (nodes 5, 6, 7, 8)
    top_surf = part.create_surface_from_normal(name="TOP_Z", direction=[0.0, 0.0, 1.0])
    assert set(top_surf.get_nodes()) == {5, 6, 7, 8}
    assert len(top_surf.faces) == 1
    assert top_surf.faces[0].face_id == 1

    # Bottom surface (-Z): face 0 in Hex8 (nodes 1, 2, 3, 4)
    bot_surf = part.create_surface_from_normal(name="BOT_Z", direction=[0.0, 0.0, -1.0])
    assert set(bot_surf.get_nodes()) == {1, 2, 3, 4}
    assert len(bot_surf.faces) == 1
    assert bot_surf.faces[0].face_id == 0

    # Left surface (-X): nodes 1, 4, 5, 8
    left_surf = part.create_surface_from_normal(name="LEFT_X", direction=[-1.0, 0.0, 0.0])
    assert set(left_surf.get_nodes()) == {1, 4, 5, 8}
    assert len(left_surf.faces) == 1


def test_custom_condition_predicate():
    """Verify custom mathematical predicate function (e.g. elliptical disk)."""
    part = Part(name="CustomPredicatePart", dim=2)
    # Grid in [-3, 3] x [-3, 3]
    nid = 1
    for x in np.linspace(-3, 3, 7):
        for y in np.linspace(-3, 3, 7):
            part.add_node(nid, [x, y])
            nid += 1

    # Condition: Ellipse (x/2.0)^2 + y^2 <= 1.0
    ellipse_set = part.create_set_from_condition(
        name="ELLIPSE",
        condition_fn=lambda c: (c[0] / 2.0) ** 2 + (c[1]) ** 2 <= 1.0 + 1e-6
    )
    nodes = ellipse_set.get_nodes()
    assert len(nodes) > 0
    for n in nodes:
        coords = part.nodes[n]
        assert (coords[0] / 2.0) ** 2 + coords[1] ** 2 <= 1.0 + 1e-5
