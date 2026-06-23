"""
mesh.py
=======
Core mesh data structures for 2D FEM.

Node, Element, NodeSet, and Mesh containers.
Node IDs can be arbitrary integers (no re-indexing requirement).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np


class Node:
    """A single 2D node with coordinates (x, y)."""

    __slots__ = ("nid", "x", "y")

    def __init__(self, nid: int, x: float, y: float):
        self.nid = nid
        self.x = float(x)
        self.y = float(y)

    @property
    def coords(self) -> np.ndarray:
        """Return (2,) coordinate array."""
        return np.array([self.x, self.y], dtype=np.float64)

    def __repr__(self) -> str:
        return f"Node({self.nid}: {self.x:.6f}, {self.y:.6f})"


class Element:
    """A single finite element.

    Parameters
    ----------
    eid : int
        Element ID (arbitrary integer).
    node_ids : list of int
        Node IDs defining the element connectivity.
        Q4: 4 nodes in CCW order; T3: 3 nodes in CCW order.
    elem_type : str
        Element type string, e.g. 'QUAD4', 'TRIA3'.
    pid : int, optional
        Property ID (cross-reference to material/section).
    """

    __slots__ = ("eid", "node_ids", "elem_type", "pid")

    def __init__(
        self,
        eid: int,
        node_ids: List[int],
        elem_type: str,
        pid: Optional[int] = None,
    ):
        self.eid = eid
        self.node_ids = list(node_ids)
        self.elem_type = elem_type.upper()
        self.pid = pid

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def __repr__(self) -> str:
        return f"Element({self.eid}: {self.elem_type} {self.node_ids})"


class NodeSet:
    """A named set of node IDs."""

    __slots__ = ("name", "node_ids")

    def __init__(self, name: str, node_ids: Optional[Set[int]] = None):
        self.name = name
        self.node_ids: Set[int] = set(node_ids) if node_ids is not None else set()

    def add(self, nid: int) -> None:
        self.node_ids.add(nid)

    def __contains__(self, nid: int) -> bool:
        return nid in self.node_ids

    def __repr__(self) -> str:
        return f"NodeSet({self.name}: {len(self.node_ids)} nodes)"


class Mesh:
    """FEM mesh container.

    Stores nodes, elements, node sets, and element sets.
    Node IDs are arbitrary integers; the mesh does not require contiguous IDs.

    Attributes
    ----------
    nodes : dict[int, Node]
    elements : dict[int, Element]
    node_sets : dict[str, NodeSet]
    element_sets : dict[str, set[int]]
    """

    def __init__(self):
        self.nodes: Dict[int, Node] = {}
        self.elements: Dict[int, Element] = {}
        self.node_sets: Dict[str, NodeSet] = {}
        self.element_sets: Dict[str, Set[int]] = {}

    # ------------------------------------------------------------------
    # Node API
    # ------------------------------------------------------------------

    def add_node(self, nid: int, x: float, y: float) -> Node:
        """Add a node. If nid exists it is overwritten."""
        node = Node(nid, x, y)
        self.nodes[nid] = node
        return node

    def get_node(self, nid: int) -> Node:
        return self.nodes[nid]

    def node_count(self) -> int:
        return len(self.nodes)

    def node_ids(self) -> List[int]:
        """Return sorted node ID list."""
        return sorted(self.nodes.keys())

    def nodes_array(self) -> np.ndarray:
        """Return (N, 2) coordinate array sorted by node ID."""
        ids = self.node_ids()
        arr = np.zeros((len(ids), 2), dtype=np.float64)
        for i, nid in enumerate(ids):
            arr[i, 0] = self.nodes[nid].x
            arr[i, 1] = self.nodes[nid].y
        return arr

    def node_id_to_index(self) -> Dict[int, int]:
        """Return mapping from node ID to 0-based row index in nodes_array()."""
        return {nid: idx for idx, nid in enumerate(self.node_ids())}

    # ------------------------------------------------------------------
    # Element API
    # ------------------------------------------------------------------

    def add_element(
        self,
        eid: int,
        node_ids: List[int],
        elem_type: str,
        pid: Optional[int] = None,
    ) -> Element:
        """Add an element. If eid exists it is overwritten."""
        elem = Element(eid, node_ids, elem_type, pid)
        self.elements[eid] = elem
        return elem

    def get_element(self, eid: int) -> Element:
        return self.elements[eid]

    def element_count(self) -> int:
        return len(self.elements)

    def elements_by_type(self, elem_type: str) -> List[Element]:
        """Return elements matching a given type (case-insensitive)."""
        t = elem_type.upper()
        return [e for e in self.elements.values() if e.elem_type == t]

    def connectivity_array(
        self, elem_type: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return connectivity data for assembly.

        Parameters
        ----------
        elem_type : str, optional
            If given, filter to only this element type.

        Returns
        -------
        conn : (n_elems, n_node_per_elem) int32 array — node IDs per element
        nid_to_idx : dict mapping node ID → 0-based index
        sorted_nids : sorted list of all node IDs
        elem_ids : (n_elems,) int32 array — element IDs
        """
        if elem_type:
            elems = self.elements_by_type(elem_type)
        else:
            elems = list(self.elements.values())

        if not elems:
            npe = 0
        else:
            npe = elems[0].n_nodes

        sorted_nids = self.node_ids()
        nid_to_idx = self.node_id_to_index()

        conn = np.zeros((len(elems), npe), dtype=np.int32)
        eids = np.zeros(len(elems), dtype=np.int32)
        for i, elem in enumerate(elems):
            conn[i] = [nid_to_idx[nid] for nid in elem.node_ids]
            eids[i] = elem.eid

        return conn, nid_to_idx, sorted_nids, eids

    # ------------------------------------------------------------------
    # NodeSet API
    # ------------------------------------------------------------------

    def add_nodeset(self, name: str, node_ids: Optional[Set[int]] = None) -> NodeSet:
        ns = NodeSet(name, node_ids)
        self.node_sets[name] = ns
        return ns

    def get_nodeset(self, name: str) -> NodeSet:
        return self.node_sets[name]

    # ------------------------------------------------------------------
    # ElementSet API
    # ------------------------------------------------------------------

    def add_elementset(self, name: str, elem_ids: Optional[Set[int]] = None) -> None:
        self.element_sets[name] = set(elem_ids) if elem_ids else set()

    # ------------------------------------------------------------------
    # Boundary conditions helper
    # ------------------------------------------------------------------

    def apply_spc(
        self,
        node_ids: Sequence[int],
        dofs: Sequence[int],
    ) -> np.ndarray:
        """Generate SPC (Single Point Constraint) DOF indices.

        Parameters
        ----------
        node_ids : sequence of int
            Nodes to constrain.
        dofs : sequence of int
            Local DOFs to constrain (0=ux, 1=uy).

        Returns
        -------
        bc_dofs : (n_bc,) int32 array — global DOF indices.
        """
        nid_to_idx = self.node_id_to_index()
        bc_list = []
        for nid in node_ids:
            idx = nid_to_idx[nid]
            base = idx * 2  # 2 DOF/node: ux, uy
            for d in dofs:
                bc_list.append(base + d)
        return np.array(bc_list, dtype=np.int32)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"Mesh: {self.node_count()} nodes, {self.element_count()} elements",
        ]
        if self.node_sets:
            names = ", ".join(self.node_sets)
            lines.append(f"  NodeSets: {names}")
        if self.element_sets:
            names = ", ".join(self.element_sets)
            lines.append(f"  ElementSets: {names}")
        # Count by element type
        type_counts: Dict[str, int] = {}
        for e in self.elements.values():
            type_counts[e.elem_type] = type_counts.get(e.elem_type, 0) + 1
        for t, c in sorted(type_counts.items()):
            lines.append(f"  {t}: {c}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
