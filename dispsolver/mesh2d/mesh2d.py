"""
mesh2d.py
=========
2D Mesh data structure for WHT_DispFoldSolver.
Independent 2D mesh container mirroring Mesh3D architecture.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class Node2D:
    """2D Node representation with 2 translational DOFs (u_x, u_y)."""
    id: int
    x: float
    y: float


@dataclass
class Element2D:
    """2D Solid Element representation."""
    id: int
    node_ids: List[int]
    elem_type: str  # e.g. "CPE4", "CPE4I", "CPE4R", "CPE4H", "CPE4_FBAR", "CPE3", "CPE6", "CPE6M", "CPE8"
    pid: int = 0    # Property ID / Part ID / Section ID


class Mesh2D:
    """2D Unstructured Finite Element Mesh Container."""

    def __init__(self):
        self.nodes: Dict[int, Node2D] = {}
        self.elements: Dict[int, Element2D] = {}
        self.node_sets: Dict[str, List[int]] = {}
        self.element_sets: Dict[str, List[int]] = {}
        self.elem_controls: Optional[np.ndarray] = None

    def add_node(self, node_id: int, x: float, y: float) -> Node2D:
        """Add a 2D node to the mesh."""
        node = Node2D(id=node_id, x=float(x), y=float(y))
        self.nodes[node_id] = node
        return node

    def add_element(self, elem_id: int, node_ids: List[int], elem_type: str, pid: int = 0) -> Element2D:
        """Add a 2D element to the mesh."""
        elem = Element2D(id=elem_id, node_ids=list(node_ids), elem_type=str(elem_type).upper(), pid=pid)
        self.elements[elem_id] = elem
        return elem

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)

    def nodes_array(self) -> np.ndarray:
        """Return (N, 2) numpy array of reference nodal coordinates ordered by insertion."""
        coords = np.zeros((self.num_nodes, 2), dtype=np.float64)
        for i, (nid, node) in enumerate(self.nodes.items()):
            coords[i, 0] = node.x
            coords[i, 1] = node.y
        return coords

    def node_id_to_index(self) -> Dict[int, int]:
        """Mapping from 1-based Node ID to 0-based contiguous array index."""
        return {nid: i for i, nid in enumerate(self.nodes.keys())}

    def element_id_to_index(self) -> Dict[int, int]:
        """Mapping from Element ID to 0-based contiguous array index."""
        return {eid: i for i, eid in enumerate(self.elements.keys())}
