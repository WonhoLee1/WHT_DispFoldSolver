"""
mesh_io.py
==========
Mesh import/export using meshio.

Supports reading meshio-supported formats (VTK, INP, MED, etc.)
and converting to the internal Mesh representation.
"""

from __future__ import annotations

from typing import Optional, Dict, Set
import numpy as np

from .mesh import Mesh


def read_mesh(filepath: str) -> Mesh:
    """Read a mesh from a meshio-supported file format.

    Parameters
    ----------
    filepath : str
        Path to the mesh file (VTK, INP, MED, etc.).

    Returns
    -------
    Mesh
        Internal mesh representation.
    """
    import meshio

    m = meshio.read(filepath)

    mesh = Mesh()

    # Nodes
    points = m.points
    for i, pt in enumerate(points):
        nid = i + 1  # meshio uses 0-based; convert to 1-based
        mesh.add_node(nid, float(pt[0]), float(pt[1]))

    # Elements
    eid = 1
    for cell_block in m.cells:
        block_type = cell_block.type
        block_data = cell_block.data

        # Map meshio types to internal types
        type_map = {
            "quad": "QUAD4",
            "quad4": "QUAD4",
            "triangle": "TRIA3",
            "tria3": "TRIA3",
        }
        elem_type = type_map.get(block_type.lower(), block_type.upper())

        for row in block_data:
            # meshio uses 0-based → 1-based
            node_ids = [int(n) + 1 for n in row]
            mesh.add_element(eid, node_ids, elem_type)
            eid += 1

    return mesh


def write_mesh(mesh: Mesh, filepath: str) -> None:
    """Write a mesh to a meshio-supported file format.

    Parameters
    ----------
    mesh : Mesh
        Internal mesh representation.
    filepath : str
        Output path (extension determines format).
    """
    import meshio

    nid_to_idx = mesh.node_id_to_index()
    points = mesh.nodes_array()

    # Convert internal elements to meshio cells
    cells_dict: Dict[str, np.ndarray] = {}

    # Group by element type
    elem_by_type: Dict[str, list] = {}
    for elem in mesh.elements.values():
        elem_by_type.setdefault(elem.elem_type, []).append(elem)

    # Meshio type mapping (0-based indexing)
    type_map = {
        "QUAD4": ("quad", 4),
        "TRIA3": ("triangle", 3),
    }

    for internal_type, elems in elem_by_type.items():
        mtype, nn = type_map.get(internal_type, (internal_type.lower(), 0))
        if nn == 0:
            continue
        data = np.zeros((len(elems), nn), dtype=np.int64)
        for i, elem in enumerate(elems):
            data[i] = [nid_to_idx[nid] for nid in elem.node_ids]
        cells_dict[mtype] = data

    if not cells_dict:
        return

    m = meshio.Mesh(points=points, cells=cells_dict)
    m.write(filepath)
