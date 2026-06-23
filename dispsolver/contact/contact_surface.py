"""ContactSurface and auto exterior edge detection for 2D meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..mesh.mesh import Mesh


@dataclass
class ContactSurface:
    """A surface definition for contact detection.

    Attributes
    ----------
    name : str
        Surface name (e.g. "PART1_SURF").
    surface_type : str
        "ELEMENT" (edge-based) or "NODE" (node-based).
    edges : list of (int, int)
        Exterior edges as sorted node-ID pairs (n1, n2) with n1 < n2.
    normals : list of (float, float)
        Outward unit normal for each edge.
    node_ids : set of int
        All nodes belonging to this surface.
    instance_name : str or None
        Original part instance name, if any.
    """

    name: str
    surface_type: str = "ELEMENT"
    edges: List[Tuple[int, int]] = field(default_factory=list)
    normals: List[Tuple[float, float]] = field(default_factory=list)
    node_ids: Set[int] = field(default_factory=set)
    instance_name: Optional[str] = None


# Edge topology for supported element types
# Each entry maps: elem_type -> list of (local_i, local_j) node index pairs
_ELEMENT_EDGES: Dict[str, List[Tuple[int, int]]] = {
    "QUAD4": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "TRIA3": [(0, 1), (1, 2), (2, 0)],
}


def auto_detect_exterior(
    mesh: Mesh,
    element_set_name: Optional[str] = None,
    surface_name: Optional[str] = None,
) -> ContactSurface:
    """Detect exterior edges of a 2D mesh by edge adjacency counting.

    Parameters
    ----------
    mesh : Mesh
        The finite element mesh.
    element_set_name : str or None
        Name of the element set to process.  If None, all elements are used.
    surface_name : str or None
        Name for the returned ContactSurface.  Auto-generated if not given.

    Returns
    -------
    ContactSurface
        Surface with exterior edges, outward normals, and node IDs.
    """
    # Collect element IDs
    if element_set_name is not None:
        if element_set_name not in mesh.element_sets:
            raise KeyError(
                f"Element set '{element_set_name}' not found in mesh. "
                f"Available: {list(mesh.element_sets.keys())}"
            )
        elem_ids: Set[int] = mesh.element_sets[element_set_name]
    else:
        elem_ids = set(mesh.elements.keys())

    if not elem_ids:
        raise ValueError("No elements found for exterior detection.")

    # Build edge adjacency: {(min_nid, max_nid): count}
    edge_count: Dict[Tuple[int, int], int] = {}
    # Also track which edge belongs to which element
    edge_to_elem: Dict[Tuple[int, int], int] = {}

    for eid in elem_ids:
        elem = mesh.elements[eid]
        elem_type = elem.elem_type.upper()
        if elem_type not in _ELEMENT_EDGES:
            raise ValueError(
                f"Unsupported element type '{elem.elem_type}' for exterior detection. "
                f"Supported: {list(_ELEMENT_EDGES.keys())}"
            )
        node_ids = elem.node_ids
        for i, j in _ELEMENT_EDGES[elem_type]:
            n1, n2 = node_ids[i], node_ids[j]
            key = (min(n1, n2), max(n1, n2))
            edge_count[key] = edge_count.get(key, 0) + 1
            if key not in edge_to_elem:
                edge_to_elem[key] = eid

    # Exterior edges: those appearing only once
    exterior_keys = sorted(k for k, v in edge_count.items() if v == 1)

    if not exterior_keys:
        raise ValueError("No exterior edges found (mesh may be closed or empty).")

    # Build node-to-coordinate lookup
    nid_to_coord = {
        nid: np.array([mesh.nodes[nid].x, mesh.nodes[nid].y], dtype=np.float64)
        for nid in mesh.nodes
    }

    edges: List[Tuple[int, int]] = []
    normals: List[Tuple[float, float]] = []
    surface_node_ids: Set[int] = set()

    for key in exterior_keys:
        n1, n2 = key
        edges.append((n1, n2))
        surface_node_ids.add(n1)
        surface_node_ids.add(n2)

        # Find the element that owns this edge (to determine outward normal)
        owner_eid = edge_to_elem[key]
        owner_elem = mesh.elements[owner_eid]

        # Element centroid
        centroid = np.mean(
            [nid_to_coord[nid] for nid in owner_elem.node_ids], axis=0
        )

        # Edge midpoint
        p1, p2 = nid_to_coord[n1], nid_to_coord[n2]
        midpoint = 0.5 * (p1 + p2)

        # Edge direction and outward normal candidates
        edge_vec = p2 - p1
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-30:
            # Degenerate edge – skip normal computation
            normals.append((0.0, 0.0))
            continue
        tangent = edge_vec / edge_len
        # Two perpendicular candidates: (-ty, tx) and (ty, -tx)
        n_candidate_1 = np.array([-tangent[1], tangent[0]])
        n_candidate_2 = np.array([tangent[1], -tangent[0]])

        # Choose the one pointing away from centroid
        to_centroid = centroid - midpoint
        dot1 = np.dot(n_candidate_1, to_centroid)
        dot2 = np.dot(n_candidate_2, to_centroid)

        outward = n_candidate_2 if dot1 > dot2 else n_candidate_1
        # Normalise
        n_len = np.linalg.norm(outward)
        if n_len > 1e-30:
            outward = outward / n_len
        normals.append((float(outward[0]), float(outward[1])))

    default_name = surface_name or (
        f"{element_set_name}_SURF" if element_set_name else "ALL_SURF"
    )

    return ContactSurface(
        name=default_name,
        surface_type="ELEMENT",
        edges=edges,
        normals=normals,
        node_ids=surface_node_ids,
    )
