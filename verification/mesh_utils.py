"""
mesh_utils.py
=============
Mesh builders for verification benchmarks.

All builders return a ``dispsolver.mesh.Mesh`` with deterministic node
IDs and element IDs, plus helper node sets for BC application.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List

from dispsolver.mesh import Mesh


def build_block_mesh(
    nx: int, ny: int,
    x0: float = 0.0, y0: float = 0.0,
    width: float = 1.0, height: float = 1.0,
    pid: int = 0,
) -> Tuple[Mesh, dict]:
    """Build a regular structured QUAD4 mesh for a rectangular block.

    Node IDs: 0 .. nx*ny-1,  ordered row-major (j*nx + i).
    Element IDs: 0 .. (nx-1)*(ny-1)-1.

    Returns
    -------
    mesh : Mesh
    info : dict with keys:
        'nx', 'ny', 'nid_to_idx',
        'left'   — list of node IDs on x=x0
        'right'  — list of node IDs on x=x0+width
        'bottom' — list of node IDs on y=y0
        'top'    — list of node IDs on y=y0+height
        'all'    — list of all node IDs
    """
    mesh = Mesh()
    xs = np.linspace(x0, x0 + width, nx)
    ys = np.linspace(y0, y0 + height, ny)

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            nid = j * nx + i
            mesh.add_node(nid, x, y)

    eid = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i
            n2 = j * nx + i + 1
            n3 = (j + 1) * nx + i + 1
            n4 = (j + 1) * nx + i
            mesh.add_element(eid, [n1, n2, n3, n4], "QUAD4", pid=pid)
            eid += 1

    nid_to_idx = mesh.node_id_to_index()
    left   = [j * nx + 0       for j in range(ny)]
    right  = [j * nx + (nx - 1) for j in range(ny)]
    bottom = [0 * nx + i        for i in range(nx)]
    top    = [(ny - 1) * nx + i for i in range(nx)]

    info = {
        'nx': nx, 'ny': ny,
        'nid_to_idx': nid_to_idx,
        'left': left, 'right': right,
        'bottom': bottom, 'top': top,
        'all': list(range(nx * ny)),
        'xs': xs, 'ys': ys,
    }
    return mesh, info


def build_beam_mesh(
    nx: int, ny: int,
    length: float, height: float,
    pid: int = 0,
) -> Tuple[Mesh, dict]:
    """Build a beam mesh: length × height, with x along the beam axis.

    Convenience wrapper around build_block_mesh with x0=0, y0=-height/2
    so the beam is centered on y=0 (neutral axis at y=0).

    Returns
    -------
    mesh, info — same as build_block_mesh, but:
        'left'   = clamped/support nodes at x=0
        'right'  = free/load nodes at x=length
        'top'    = top surface nodes
        'bottom' = bottom surface nodes
    """
    mesh, info = build_block_mesh(
        nx, ny, x0=0.0, y0=-height / 2.0,
        width=length, height=height, pid=pid,
    )
    return mesh, info


def build_patch_mesh_irregular() -> Tuple[Mesh, dict]:
    """Build the classic Q4 patch test mesh with an internal node.

    Node layout (irregular — tests that the element handles non-rectangular
    geometry correctly):

        7(0,1)───8(0.6,1)───9(1,1)
          |          |          |
        4(0,0.4)─5(0.5,0.5)─6(1,0.6)
          |          |          |
        1(0,0)───2(0.3,0)───3(1,0)

    Elements (CCW):
        E1: 1-2-5-4   E2: 2-3-6-5
        E3: 4-5-8-7   E4: 5-6-9-8

    Returns
    -------
    mesh, info — with 'internal_nodes' = [4, 5, 8] (non-boundary nodes)
    """
    mesh = Mesh()
    coords = {
        1: (0.0, 0.0),  2: (0.3, 0.0),  3: (1.0, 0.0),
        4: (0.0, 0.4),  5: (0.5, 0.5), 6: (1.0, 0.6),
        7: (0.0, 1.0),  8: (0.6, 1.0), 9: (1.0, 1.0),
    }
    for nid, (x, y) in coords.items():
        mesh.add_node(nid, x, y)

    mesh.add_element(1, [1, 2, 5, 4], "QUAD4")
    mesh.add_element(2, [2, 3, 6, 5], "QUAD4")
    mesh.add_element(3, [4, 5, 8, 7], "QUAD4")
    mesh.add_element(4, [5, 6, 9, 8], "QUAD4")

    nid_to_idx = mesh.node_id_to_index()
    info = {
        'nid_to_idx': nid_to_idx,
        'boundary_nodes': [1, 2, 3, 6, 9, 8, 7, 4],
        'internal_nodes': [5],
        'all': list(coords.keys()),
    }
    return mesh, info


def build_single_element_mesh(
    width: float = 1.0, height: float = 1.0,
    pid: int = 0,
) -> Tuple[Mesh, dict]:
    """Build a single Q4 element mesh (unit square by default).

    Nodes (CCW):
        0(0,0)  1(W,0)  2(W,H)  3(0,H)
    """
    mesh = Mesh()
    mesh.add_node(0, 0.0,   0.0)
    mesh.add_node(1, width, 0.0)
    mesh.add_node(2, width, height)
    mesh.add_node(3, 0.0,   height)
    mesh.add_element(0, [0, 1, 2, 3], "QUAD4", pid=pid)

    info = {
        'nid_to_idx': mesh.node_id_to_index(),
        'all': [0, 1, 2, 3],
        'bottom': [0, 1], 'top': [2, 3],
        'left': [0, 3], 'right': [1, 2],
    }
    return mesh, info
