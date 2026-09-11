"""First-class Set, Surface, and Smart GeneralSet definitions for CAE model hierarchy."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, List, Sequence, Optional, Tuple, Any, Dict, Union
import numpy as np


class SetScope:
    """Scope indicator for Sets and Surfaces."""
    PART = "PART"
    ASSEMBLY = "ASSEMBLY"


@dataclass
class NodeSet:
    """Named collection of Node IDs (part-local or assembly-global)."""
    name: str
    node_ids: Set[int] = field(default_factory=set)
    scope: str = SetScope.PART

    def add(self, nid: int) -> None:
        self.node_ids.add(int(nid))

    def add_many(self, nids: Sequence[int]) -> None:
        for nid in nids:
            self.node_ids.add(int(nid))

    def to_array(self) -> np.ndarray:
        return np.array(sorted(self.node_ids), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.node_ids)

    def __contains__(self, nid: int) -> bool:
        return int(nid) in self.node_ids


@dataclass
class ElementSet:
    """Named collection of Element IDs."""
    name: str
    element_ids: Set[int] = field(default_factory=set)
    scope: str = SetScope.PART

    def add(self, eid: int) -> None:
        self.element_ids.add(int(eid))

    def add_many(self, eids: Sequence[int]) -> None:
        for eid in eids:
            self.element_ids.add(int(eid))

    def to_array(self) -> np.ndarray:
        return np.array(sorted(self.element_ids), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.element_ids)

    def __contains__(self, eid: int) -> bool:
        return int(eid) in self.element_ids


@dataclass(frozen=True)
class ElementFace:
    """Identifies a specific face/edge of an element.
    
    face_id:
    - 2D Quad (4 edges): 0=(0,1), 1=(1,2), 2=(2,3), 3=(3,0)
    - 3D Hex (6 faces): 0=Bottom(0,1,2,3), 1=Top(4,5,6,7), 2=Front(0,1,5,4),
                        3=Right(1,2,6,5), 4=Back(2,3,7,6), 5=Left(3,0,4,7)
    """
    element_id: int
    face_id: int


@dataclass
class Surface:
    """Geometric boundary surface composed of element faces or nodes."""
    name: str
    faces: List[ElementFace] = field(default_factory=list)
    node_ids: Optional[Set[int]] = None
    scope: str = SetScope.PART

    def add_face(self, element_id: int, face_id: int) -> None:
        self.faces.append(ElementFace(element_id=int(element_id), face_id=int(face_id)))

    def add_node(self, nid: int) -> None:
        if self.node_ids is None:
            self.node_ids = set()
        self.node_ids.add(int(nid))

    def __len__(self) -> int:
        return len(self.faces) if self.faces else (len(self.node_ids) if self.node_ids else 0)


# Standard canonical face topologies for 2D and 3D solid elements
_QUAD4_EDGE_NODES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
]

_HEX8_FACE_NODES = [
    (0, 1, 2, 3),  # Face 0 (Bottom)
    (4, 5, 6, 7),  # Face 1 (Top)
    (0, 1, 5, 4),  # Face 2 (Front)
    (1, 2, 6, 5),  # Face 3 (Right)
    (2, 3, 7, 6),  # Face 4 (Back)
    (3, 0, 4, 7),  # Face 5 (Left)
]


class GeneralSet:
    """Smart Unified Set (Region) container supporting associative resolution of
    elements, nodes, and boundary faces/segments.
    
    Capabilities:
    1. Holds arbitrary combination of Node IDs, Element IDs, and ElementFaces.
    2. get_elements(): Returns sorted element IDs.
    3. get_nodes(include_elements=True): Returns sorted node IDs, automatically
       expanding to include all nodes belonging to the elements in this set.
    4. get_faces(exterior_only=True): Returns ElementFaces. If not explicitly defined,
       automatically computes the exterior boundary faces by canceling shared internal faces!
    5. get_segments(): Returns 2D edge segments (node_a, node_b) for surface tie / contact.
    6. from_bounding_box(): Factory method filtering entities by 2D/3D coordinate ranges.
    7. Supports Python set algebra: | (union), & (intersection), - (difference).
    """

    def __init__(
        self,
        name: str,
        part: Optional[Any] = None,
        nodes: Optional[Sequence[int]] = None,
        elements: Optional[Sequence[int]] = None,
        faces: Optional[Sequence[Union[ElementFace, Tuple[int, int]]]] = None,
        scope: str = SetScope.PART
    ):
        self.name = str(name)
        self.part = part
        self.scope = str(scope)
        self.node_ids: Set[int] = set(nodes) if nodes is not None else set()
        self.element_ids: Set[int] = set(elements) if elements is not None else set()
        
        self.faces: List[ElementFace] = []
        if faces is not None:
            for f in faces:
                if isinstance(f, ElementFace):
                    self.faces.append(f)
                else:
                    self.faces.append(ElementFace(int(f[0]), int(f[1])))

    # ─── CORE EXTRACTION & ASSOCIATIVE RESOLUTION ──────────────────────

    def get_elements(self) -> np.ndarray:
        """Return sorted array of Element IDs contained in this set."""
        return np.array(sorted(self.element_ids), dtype=np.int64)

    def get_nodes(self, include_elements: bool = True) -> np.ndarray:
        """Return sorted array of Node IDs.
        
        If include_elements is True and a part reference is available,
        automatically resolves and includes all nodes constituting the elements in this set.
        """
        all_nids = set(self.node_ids)
        if include_elements and self.part is not None and len(self.element_ids) > 0:
            for eid in self.element_ids:
                if eid in self.part.elements:
                    _, conn = self.part.elements[eid]
                    all_nids.update(conn)
        return np.array(sorted(all_nids), dtype=np.int64)

    def get_faces(self, exterior_only: bool = True) -> List[ElementFace]:
        """Return list of ElementFaces.
        
        If explicit faces exist, returns them.
        Otherwise, if elements exist in this set and part is set, automatically
        computes boundary faces by canceling shared internal interfaces.
        """
        if len(self.faces) > 0:
            return list(self.faces)

        if self.part is None or len(self.element_ids) == 0:
            return []

        # Count occurrences of sorted node tuples defining each face
        # Shared internal faces will appear >= 2 times. Exterior faces appear exactly 1 time.
        face_registry: Dict[Tuple[int, ...], List[ElementFace]] = {}

        for eid in self.element_ids:
            if eid not in self.part.elements:
                continue
            etype, conn = self.part.elements[eid]
            n_nodes = len(conn)

            if n_nodes == 4:  # 2D Quad
                for fid, (i0, i1) in enumerate(_QUAD4_EDGE_NODES):
                    key = tuple(sorted([conn[i0], conn[i1]]))
                    face_registry.setdefault(key, []).append(ElementFace(eid, fid))
            elif n_nodes == 8:  # 3D Hex
                for fid, face_indices in enumerate(_HEX8_FACE_NODES):
                    key = tuple(sorted([conn[idx] for idx in face_indices]))
                    face_registry.setdefault(key, []).append(ElementFace(eid, fid))

        result: List[ElementFace] = []
        for key, face_list in face_registry.items():
            if exterior_only:
                if len(face_list) == 1:
                    result.append(face_list[0])
            else:
                result.extend(face_list)

        return result

    def get_segments(self) -> List[Tuple[int, int]]:
        """Return list of 2D line segments (node_a, node_b) on the exterior boundary."""
        faces = self.get_faces(exterior_only=True)
        segments: List[Tuple[int, int]] = []
        if self.part is None:
            return segments

        for ef in faces:
            if ef.element_id not in self.part.elements:
                continue
            etype, conn = self.part.elements[ef.element_id]
            if len(conn) == 4 and ef.face_id < 4:
                i0, i1 = _QUAD4_EDGE_NODES[ef.face_id]
                segments.append((conn[i0], conn[i1]))
        return segments

    # ─── GEOMETRIC BOUNDING-BOX FILTERING ─────────────────────────────

    @classmethod
    def from_bounding_box(
        cls,
        part: Any,
        name: str,
        x_bounds: Optional[Tuple[float, float]] = None,
        y_bounds: Optional[Tuple[float, float]] = None,
        z_bounds: Optional[Tuple[float, float]] = None,
        entity_type: str = "ALL"  # "ALL", "NODES", "ELEMENTS"
    ) -> GeneralSet:
        """Create a GeneralSet by filtering nodes and elements within coordinate ranges.
        
        Parameters
        ----------
        part : Part
            Owner part to search within
        name : str
            Set name
        x_bounds : tuple of (xmin, xmax), optional
        y_bounds : tuple of (ymin, ymax), optional
        z_bounds : tuple of (zmin, zmax), optional
        entity_type : str
            "ALL", "NODES", or "ELEMENTS"
        """
        gset = cls(name=name, part=part)
        tol = 1e-6

        def in_box(coords: np.ndarray) -> bool:
            x = coords[0]
            y = coords[1]
            z = coords[2] if len(coords) >= 3 else 0.0
            if x_bounds is not None and (x < x_bounds[0] - tol or x > x_bounds[1] + tol):
                return False
            if y_bounds is not None and (y < y_bounds[0] - tol or y > y_bounds[1] + tol):
                return False
            if z_bounds is not None and (z < z_bounds[0] - tol or z > z_bounds[1] + tol):
                return False
            return True

        # 1. Filter Nodes
        matching_nodes = set()
        for nid, coords in part.nodes.items():
            if in_box(coords):
                matching_nodes.add(nid)

        if entity_type.upper() in ["ALL", "NODES"]:
            gset.node_ids = matching_nodes

        # 2. Filter Elements (All element nodes must be in the box, or element centroid)
        if entity_type.upper() in ["ALL", "ELEMENTS"]:
            for eid, (_, conn) in part.elements.items():
                elem_coords = np.array([part.nodes[n] for n in conn])
                centroid = np.mean(elem_coords, axis=0)
                if in_box(centroid):
                    gset.element_ids.add(eid)

        return gset

    # ─── SET ALGEBRA OPERATORS ────────────────────────────────────────

    def __or__(self, other: GeneralSet) -> GeneralSet:
        """Union of two GeneralSets (self | other)."""
        p = self.part or other.part
        new_set = GeneralSet(
            name=f"{self.name}_OR_{other.name}",
            part=p,
            nodes=self.node_ids | other.node_ids,
            elements=self.element_ids | other.element_ids,
            faces=list(set(self.faces) | set(other.faces))
        )
        return new_set

    def __and__(self, other: GeneralSet) -> GeneralSet:
        """Intersection of two GeneralSets (self & other)."""
        p = self.part or other.part
        new_set = GeneralSet(
            name=f"{self.name}_AND_{other.name}",
            part=p,
            nodes=self.node_ids & other.node_ids,
            elements=self.element_ids & other.element_ids,
            faces=list(set(self.faces) & set(other.faces))
        )
        return new_set

    def __sub__(self, other: GeneralSet) -> GeneralSet:
        """Difference of two GeneralSets (self - other)."""
        p = self.part or other.part
        new_set = GeneralSet(
            name=f"{self.name}_SUB_{other.name}",
            part=p,
            nodes=self.node_ids - other.node_ids,
            elements=self.element_ids - other.element_ids,
            faces=list(set(self.faces) - set(other.faces))
        )
        return new_set

    def __len__(self) -> int:
        return len(self.element_ids) if len(self.element_ids) > 0 else len(self.node_ids)

    def __repr__(self) -> str:
        return (
            f"GeneralSet(name='{self.name}', num_nodes={len(self.node_ids)}, "
            f"num_elements={len(self.element_ids)}, num_faces={len(self.faces)})"
        )
