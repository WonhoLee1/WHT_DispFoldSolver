"""
test_3d_negative_volume_guard.py
==================================
DynamicSolver3D refuses 8-node hexahedra with a zero/negative reference
Jacobian (Abaqus/Standard: element set ErrElemVolSmallNegZero) instead of
silently assembling a negative-definite element stiffness.

Found 2026-10-01: all 240 roller C3D8I elements of
benchmark_inp/phase2_methodology/01_hertz_roller_plate_3d.inp were
left-handed; the contact solve then stalled with no error pointing at the
mesh.
"""

import pytest

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

_CUBE = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def _mesh(order, etype="C3D8"):
    mesh = Mesh3D()
    for i, (x, y, z) in enumerate(_CUBE, start=1):
        mesh.add_node(i, float(x), float(y), float(z))
    mesh.add_element(1, order, etype)
    return mesh


@pytest.mark.parametrize("etype", ["C3D8", "C3D8I", "C3D8R"])
def test_right_handed_hex_constructs(etype):
    DynamicSolver3D(_mesh([1, 2, 3, 4, 5, 6, 7, 8], etype), {"E": 1000.0, "nu": 0.3})


@pytest.mark.parametrize("etype", ["C3D8", "C3D8I", "C3D8R"])
def test_left_handed_hex_is_rejected(etype):
    # top and bottom faces swapped -> det J = -1 everywhere
    with pytest.raises(ValueError, match="negative volume"):
        DynamicSolver3D(_mesh([5, 6, 7, 8, 1, 2, 3, 4], etype), {"E": 1000.0, "nu": 0.3})


def test_collapsed_hex_is_rejected():
    # node 7 dragged through the opposite face: one Gauss point goes negative
    mesh = _mesh([1, 2, 3, 4, 5, 6, 7, 8])
    mesh.nodes[7].x, mesh.nodes[7].y, mesh.nodes[7].z = -1.5, -1.5, -1.5
    with pytest.raises(ValueError, match="negative volume"):
        DynamicSolver3D(mesh, {"E": 1000.0, "nu": 0.3})


# -----------------------------------------------------------------
# 4-node and 10-node Tetrahedra (C3D4, C3D10)
# -----------------------------------------------------------------
def _tet_mesh(order, etype="C3D4"):
    mesh = Mesh3D()
    # 4 standard vertices of a regular-ish tetrahedron
    pts = [
        (0.0, 0.0, 0.0),  # 1
        (1.0, 0.0, 0.0),  # 2
        (0.0, 1.0, 0.0),  # 3
        (0.0, 0.0, 1.0),  # 4
    ]
    if etype.startswith("C3D10"):
        # 6 mid-edge nodes
        pts += [
            (0.5, 0.0, 0.0),  # 5: 1-2
            (0.5, 0.5, 0.0),  # 6: 2-3
            (0.0, 0.5, 0.0),  # 7: 3-1
            (0.0, 0.0, 0.5),  # 8: 1-4
            (0.5, 0.0, 0.5),  # 9: 2-4
            (0.0, 0.5, 0.5),  # 10: 3-4
        ]
    for i, (x, y, z) in enumerate(pts, start=1):
        mesh.add_node(i, float(x), float(y), float(z))
    mesh.add_element(1, order, etype)
    return mesh


@pytest.mark.parametrize("etype", ["C3D4", "C3D10"])
def test_right_handed_tet_constructs(etype):
    order = list(range(1, 11)) if etype == "C3D10" else [1, 2, 3, 4]
    DynamicSolver3D(_tet_mesh(order, etype), {"E": 1000.0, "nu": 0.3})


@pytest.mark.parametrize("etype", ["C3D4", "C3D10"])
def test_left_handed_tet_is_rejected(etype):
    # Swap nodes 2 and 3 -> inverts base triangle orientation -> det J < 0
    if etype == "C3D10":
        order = [1, 3, 2, 4, 7, 6, 5, 8, 10, 9]
    else:
        order = [1, 3, 2, 4]
    with pytest.raises(ValueError, match="negative volume"):
        DynamicSolver3D(_tet_mesh(order, etype), {"E": 1000.0, "nu": 0.3})


# -----------------------------------------------------------------
# 6-node Wedge / Prism (C3D6)
# -----------------------------------------------------------------
def _wedge_mesh(order, etype="C3D6"):
    mesh = Mesh3D()
    pts = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),  # Bottom: 1, 2, 3
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0),  # Top: 4, 5, 6
    ]
    for i, (x, y, z) in enumerate(pts, start=1):
        mesh.add_node(i, float(x), float(y), float(z))
    mesh.add_element(1, order, etype)
    return mesh


def test_right_handed_wedge_constructs():
    DynamicSolver3D(_wedge_mesh([1, 2, 3, 4, 5, 6], "C3D6"), {"E": 1000.0, "nu": 0.3})


def test_left_handed_wedge_is_rejected():
    # Swap top and bottom faces -> inverted
    with pytest.raises(ValueError, match="negative volume"):
        DynamicSolver3D(_wedge_mesh([4, 5, 6, 1, 2, 3], "C3D6"), {"E": 1000.0, "nu": 0.3})


# -----------------------------------------------------------------
# 20-node Hex (C3D20)
# -----------------------------------------------------------------
def test_right_handed_c3d20_constructs():
    # 8 corner nodes + 12 mid-edge dummy nodes
    mesh = _mesh([1, 2, 3, 4, 5, 6, 7, 8], "C3D20")
    for i in range(9, 21):
        mesh.add_node(i, 0.5, 0.5, 0.5)
    mesh.elements[1].node_ids = list(range(1, 21))
    DynamicSolver3D(mesh, {"E": 1000.0, "nu": 0.3})


def test_left_handed_c3d20_is_rejected():
    mesh = _mesh([1, 2, 3, 4, 5, 6, 7, 8], "C3D20")
    for i in range(9, 21):
        mesh.add_node(i, 0.5, 0.5, 0.5)
    # Swap top and bottom corner faces
    mesh.elements[1].node_ids = [5, 6, 7, 8, 1, 2, 3, 4] + list(range(9, 21))
    with pytest.raises(ValueError, match="negative volume"):
        DynamicSolver3D(mesh, {"E": 1000.0, "nu": 0.3})

