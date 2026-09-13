"""
test_surface_tie_unified.py
===========================
Tests for the unified SurfaceTie factory function (Abaqus Scripting API Parity).
"""

import pytest
import numpy as np
from dispsolver.constraint import SurfaceTie, SurfaceTieConstraint, SurfaceTieConstraint3D
from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d.mesh3d import Mesh3D


def test_surface_tie_2d_coords():
    # 2D coords test
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
    tie = SurfaceTie(
        slave=[3, 4],
        master=[1, 2],
        coords=coords,
        nid_to_idx=nid_to_idx,
        penalty_stiffness=1e5,
        name="TEST_2D"
    )
    assert isinstance(tie, SurfaceTieConstraint)
    assert tie.k_tie == 1e5
    assert tie.name == "TEST_2D"


def test_surface_tie_2d_mesh():
    mesh = Mesh2D()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(3, 0.0, 0.1)
    mesh.add_node(4, 1.0, 0.1)
    mesh.add_element(1, [1, 2, 4, 3], elem_type="CPE4")

    tie = SurfaceTie(
        slave=[3, 4],
        master=[1, 2],
        mesh=mesh,
        penalty_stiffness=5e4
    )
    assert isinstance(tie, SurfaceTieConstraint)
    assert tie.k_tie == 5e4


def test_surface_tie_3d_faces():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.1],
        [1.0, 0.0, 0.1],
    ])
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    master_faces = [(1, 2, 3, 4)]
    tie = SurfaceTie(
        slave=[5, 6],
        master=master_faces,
        coords=coords,
        nid_to_idx=nid_to_idx,
        penalty_stiffness=2e5,
        name="TEST_3D"
    )
    assert isinstance(tie, SurfaceTieConstraint3D)
    assert tie.k_tie == 2e5
    assert tie.name == "TEST_3D"


def test_surface_tie_3d_mesh_auto_faces():
    mesh = Mesh3D()
    # 8 nodes of a cube
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
        (9, 0.0, 0.0, 1.1), (10, 1.0, 0.0, 1.1)
    ]
    for nid, x, y, z in nodes:
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], elem_type="C3D8")

    # Pass master as top node IDs: [5, 6, 7, 8]
    tie = SurfaceTie(
        slave=[9, 10],
        master=[5, 6, 7, 8],
        mesh=mesh,
        penalty_stiffness=1e6
    )
    assert isinstance(tie, SurfaceTieConstraint3D)
    # Master faces should have automatically resolved (5, 6, 7, 8)
    assert (5, 6, 7, 8) in tie.master_faces
