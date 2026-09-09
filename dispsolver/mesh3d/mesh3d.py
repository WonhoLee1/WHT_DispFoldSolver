"""
mesh3d.py
=========
3D Mesh data structure for WHT_DispFoldSolver.
Independent of 2D mesh to prevent regressions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class Node3D:
    """3D Node representation with 3 translational DOFs (u_x, u_y, u_z)."""
    id: int
    x: float
    y: float
    z: float


@dataclass
class Element3D:
    """3D Solid Element representation."""
    id: int
    node_ids: List[int]
    elem_type: str  # e.g. "C3D8", "C3D8R", "C3D8I", "C3D4", "C3D10", "C3D10M"
    pid: int = 0    # Property ID / Part ID / Elset ID


class Mesh3D:
    """3D Unstructured Finite Element Mesh Container."""

    def __init__(self):
        self.nodes: Dict[int, Node3D] = {}
        self.elements: Dict[int, Element3D] = {}
        self.node_sets: Dict[str, List[int]] = {}
        self.element_sets: Dict[str, List[int]] = {}

    def add_node(self, node_id: int, x: float, y: float, z: float) -> Node3D:
        """Add a 3D node to the mesh."""
        node = Node3D(id=node_id, x=float(x), y=float(y), z=float(z))
        self.nodes[node_id] = node
        return node

    def add_element(self, elem_id: int, node_ids: List[int], elem_type: str, pid: int = 0) -> Element3D:
        """Add a 3D element to the mesh."""
        elem = Element3D(id=elem_id, node_ids=list(node_ids), elem_type=elem_type, pid=pid)
        self.elements[elem_id] = elem
        return elem

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)

    def node_id_to_index(self) -> Dict[int, int]:
        """Map 1-based Node IDs to 0-based contiguous indices."""
        sorted_ids = sorted(self.nodes.keys())
        return {nid: idx for idx, nid in enumerate(sorted_ids)}

    @property
    def nid_to_idx(self) -> Dict[int, int]:
        return self.node_id_to_index()

    @property
    def coords(self) -> np.ndarray:
        return self.nodes_array()

    def nodes_array(self) -> np.ndarray:
        """Return (N, 3) numpy array of nodal initial coordinates."""
        sorted_ids = sorted(self.nodes.keys())
        coords = np.zeros((len(sorted_ids), 3), dtype=np.float64)
        for idx, nid in enumerate(sorted_ids):
            node = self.nodes[nid]
            coords[idx] = [node.x, node.y, node.z]
        return coords

    def elements_connectivity_array(self) -> np.ndarray:
        """Return (Ne, Nn_per_elem) array of 0-based node indices."""
        nid_map = self.node_id_to_index()
        sorted_eids = sorted(self.elements.keys())
        if not sorted_eids:
            return np.zeros((0, 0), dtype=np.int64)
        nn = len(self.elements[sorted_eids[0]].node_ids)
        conn = np.zeros((len(sorted_eids), nn), dtype=np.int64)
        for idx, eid in enumerate(sorted_eids):
            elem = self.elements[eid]
            conn[idx] = [nid_map[nid] for nid in elem.node_ids]
        return conn
