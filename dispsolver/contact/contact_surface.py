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


@dataclass
class ContactSurface3D:
    """A 3D boundary surface definition for contact detection.

    Attributes
    ----------
    name : str
        Surface name (e.g. "PART1_EXTERIOR").
    surface_type : str
        "ELEMENT" (face-based) or "NODE" (node-based).
    faces : list of tuple of int
        Exterior faces as node-ID tuples, oriented with outward normals.
    normals : list of (float, float, float)
        Outward unit normal for each face at reference configuration.
    node_ids : set of int
        All nodes belonging to this surface.
    instance_name : str or None
        Original part instance name, if any.
    """

    name: str
    surface_type: str = "ELEMENT"
    faces: List[Tuple[int, ...]] = field(default_factory=list)
    normals: List[Tuple[float, float, float]] = field(default_factory=list)
    node_ids: Set[int] = field(default_factory=set)
    instance_name: Optional[str] = None
    face_to_elem: Dict[int, int] = field(default_factory=dict)
    elem_to_pid: Dict[int, int] = field(default_factory=dict)


# 3D Element face topology: maps element type -> list of local face node index tuples
_ELEMENT_FACES_3D: Dict[str, List[Tuple[int, ...]]] = {
    # 8-node hexahedral elements (6 Quad4 faces)
    "C3D8": [
        (0, 1, 2, 3),  # Face 0 (Bottom, -Z)
        (4, 5, 6, 7),  # Face 1 (Top, +Z)
        (0, 1, 5, 4),  # Face 2 (Front, -Y)
        (1, 2, 6, 5),  # Face 3 (Right, +X)
        (2, 3, 7, 6),  # Face 4 (Back, +Y)
        (3, 0, 4, 7),  # Face 5 (Left, -X)
    ],
    "C3D8R": [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    "C3D8I": [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    "C3D8H": [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    "C3D8_FBAR": [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    "C3D8_CR": [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    "HEX8": [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    "SOLID_SHELL": [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    "C3D8_SS": [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ],
    # 4-node tetrahedral elements (4 Tri3 faces)
    "C3D4": [
        (0, 1, 2),  # Face 0
        (0, 3, 1),  # Face 1
        (1, 3, 2),  # Face 2
        (2, 3, 0),  # Face 3
    ],
    "C3D4_ANP": [
        (0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0),
    ],
    "TET4": [
        (0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0),
    ],
    # 10-node quadratic tetrahedral elements (corner vertices)
    "C3D10": [
        (0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0),
    ],
    "C3D10M": [
        (0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0),
    ],
    # 6-node wedge/prism elements (2 Tri3 + 3 Quad4 faces)
    "C3D6": [
        (0, 1, 2),     # Face 0 (bottom tri)
        (3, 4, 5),     # Face 1 (top tri)
        (0, 1, 4, 3),  # Face 2 (quad)
        (1, 2, 5, 4),  # Face 3 (quad)
        (2, 0, 3, 5),  # Face 4 (quad)
    ],
    "WEDGE6": [
        (0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5),
    ],
}


def auto_detect_exterior_3d(
    mesh: Any,
    element_set_name: Optional[str] = None,
    surface_name: Optional[str] = None,
) -> ContactSurface3D:
    """Detect exterior 3D faces (Quad4 and Tri3) of a 3D mesh by face adjacency counting.

    Supports Mesh3D, Part, or any object with .elements and .nodes.
    Faces shared by two elements appear twice and are canceled out.
    Exterior faces appear exactly once and are oriented with outward-pointing normals.

    Parameters
    ----------
    mesh : Any
        A Mesh3D or Part instance.
    element_set_name : str or None
        Name of element set to process. If None, all elements are processed.
    surface_name : str or None
        Name of the resulting ContactSurface3D. Auto-generated if None.

    Returns
    -------
    ContactSurface3D
        Surface container with exterior face node tuples, outward normals, and node IDs.
    """
    # 1. Extract element IDs and element data
    if hasattr(mesh, "element_sets") and element_set_name is not None:
        if element_set_name not in mesh.element_sets:
            raise KeyError(
                f"Element set '{element_set_name}' not found in mesh. "
                f"Available: {list(mesh.element_sets.keys())}"
            )
        elset = mesh.element_sets[element_set_name]
        elem_ids = elset.element_ids if hasattr(elset, "element_ids") else set(elset)
    else:
        elem_ids = set(mesh.elements.keys())

    if not elem_ids:
        raise ValueError("No elements found for 3D exterior detection.")

    # 2. Build face adjacency: {tuple_of_sorted_node_ids: list of (elem_id, face_node_tuple)}
    face_counts: Dict[Tuple[int, ...], int] = {}
    face_definitions: Dict[Tuple[int, ...], Tuple[int, Tuple[int, ...]]] = {}

    for eid in elem_ids:
        elem = mesh.elements[eid]
        if hasattr(elem, "elem_type") and hasattr(elem, "node_ids"):
            elem_type = elem.elem_type.upper()
            node_ids = list(elem.node_ids)
        elif isinstance(elem, (list, tuple)) and len(elem) == 2:
            elem_type = str(elem[0]).upper()
            node_ids = list(elem[1])
        else:
            raise ValueError(f"Cannot parse element {eid}: {elem}")

        if elem_type not in _ELEMENT_FACES_3D:
            # Fallback based on node count
            if len(node_ids) == 8:
                face_templates = _ELEMENT_FACES_3D["C3D8"]
            elif len(node_ids) == 4:
                face_templates = _ELEMENT_FACES_3D["C3D4"]
            elif len(node_ids) == 6:
                face_templates = _ELEMENT_FACES_3D["C3D6"]
            elif len(node_ids) == 10:
                face_templates = _ELEMENT_FACES_3D["C3D10"]
            else:
                raise ValueError(
                    f"Unsupported 3D element type '{elem_type}' with {len(node_ids)} nodes."
                )
        else:
            face_templates = _ELEMENT_FACES_3D[elem_type]

        for local_indices in face_templates:
            face_nodes = tuple(node_ids[i] for i in local_indices)
            key = tuple(sorted(face_nodes))
            face_counts[key] = face_counts.get(key, 0) + 1
            face_definitions[key] = (eid, face_nodes)

    # 3. Exterior faces: those appearing exactly once
    exterior_keys = sorted(k for k, count in face_counts.items() if count == 1)
    if not exterior_keys:
        raise ValueError("No exterior 3D faces found (mesh may be empty or degenerate).")

    # 4. Helper to resolve node coordinates
    def _get_coord(nid: int) -> np.ndarray:
        if hasattr(mesh, "nodes"):
            node = mesh.nodes[nid]
            if hasattr(node, "x") and hasattr(node, "y") and hasattr(node, "z"):
                return np.array([node.x, node.y, node.z], dtype=np.float64)
            elif isinstance(node, (list, tuple, np.ndarray)):
                arr = np.asarray(node, dtype=np.float64)
                if len(arr) == 3:
                    return arr
                elif len(arr) == 2:
                    return np.array([arr[0], arr[1], 0.0], dtype=np.float64)
        raise KeyError(f"Cannot resolve coordinates for node {nid}")

    faces: List[Tuple[int, ...]] = []
    normals: List[Tuple[float, float, float]] = []
    surface_node_ids: Set[int] = set()
    face_to_elem: Dict[int, int] = {}
    elem_to_pid: Dict[int, int] = {}

    for face_idx, key in enumerate(exterior_keys):
        owner_eid, face_nodes = face_definitions[key]
        owner_elem = mesh.elements[owner_eid]
        elem_node_ids = owner_elem.node_ids if hasattr(owner_elem, "node_ids") else owner_elem[1]
        face_to_elem[face_idx] = owner_eid
        elem_to_pid[owner_eid] = getattr(owner_elem, "pid", 0)

        # Element centroid
        elem_coords = np.array([_get_coord(nid) for nid in elem_node_ids], dtype=np.float64)
        c_elem = np.mean(elem_coords, axis=0)

        # Face centroid
        face_coords = np.array([_get_coord(nid) for nid in face_nodes], dtype=np.float64)
        c_face = np.mean(face_coords, axis=0)

        # Face outward normal: cross product of edges
        v0 = face_coords[1] - face_coords[0]
        v1 = face_coords[2] - face_coords[0]
        normal = np.cross(v0, v1)
        norm_val = np.linalg.norm(normal)
        if norm_val < 1e-14:
            # Degenerate face
            unit_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            unit_n = normal / norm_val

        # Check orientation: vector from element centroid to face centroid
        to_face = c_face - c_elem
        if np.dot(unit_n, to_face) < 0.0:
            # Normal points inward! Reverse winding order to flip normal outwards
            if len(face_nodes) == 4:
                face_nodes = (face_nodes[0], face_nodes[3], face_nodes[2], face_nodes[1])
            elif len(face_nodes) == 3:
                face_nodes = (face_nodes[0], face_nodes[2], face_nodes[1])
            unit_n = -unit_n

        faces.append(face_nodes)
        normals.append((float(unit_n[0]), float(unit_n[1]), float(unit_n[2])))
        surface_node_ids.update(face_nodes)

    default_name = surface_name or (
        f"{element_set_name}_SURF3D" if element_set_name else "ALL_EXTERIOR_3D"
    )

    return ContactSurface3D(
        name=default_name,
        surface_type="ELEMENT",
        faces=faces,
        normals=normals,
        node_ids=surface_node_ids,
        face_to_elem=face_to_elem,
        elem_to_pid=elem_to_pid,
    )

