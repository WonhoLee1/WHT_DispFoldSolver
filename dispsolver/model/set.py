"""First-class Set, Surface, and Smart GeneralSet definitions for CAE model hierarchy."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, List, Sequence, Optional, Tuple, Any, Dict, Union, Callable
from collections import deque
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
        eids = set(self.element_ids)
        if len(eids) == 0 and len(self.faces) > 0:
            eids = {ef.element_id for ef in self.faces}
        return np.array(sorted(eids), dtype=np.int64)

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

    @classmethod
    def from_condition(
        cls,
        part: Any,
        name: str,
        condition_fn: Callable[[np.ndarray], bool],
        entity_type: str = "ALL",  # "ALL", "NODES", "ELEMENTS"
        element_selection: str = "CENTROID"  # "CENTROID", "ALL_NODES", "ANY_NODE"
    ) -> GeneralSet:
        """Create a GeneralSet by filtering nodes and elements using a custom predicate function.
        
        Parameters
        ----------
        part : Part
            Owner part
        name : str
            Set name
        condition_fn : callable
            Function taking coordinates np.ndarray and returning bool
        entity_type : str
            "ALL", "NODES", or "ELEMENTS"
        element_selection : str
            "CENTROID" (element centroid satisfies condition),
            "ALL_NODES" (all nodes satisfy condition),
            "ANY_NODE" (at least one node satisfies condition)
        """
        gset = cls(name=name, part=part)

        # 1. Filter Nodes
        matching_nodes = set()
        for nid, coords in part.nodes.items():
            if condition_fn(coords):
                matching_nodes.add(nid)

        if entity_type.upper() in ["ALL", "NODES"]:
            gset.node_ids = matching_nodes

        # 2. Filter Elements
        if entity_type.upper() in ["ALL", "ELEMENTS"]:
            for eid, (_, conn) in part.elements.items():
                elem_coords = np.array([part.nodes[n] for n in conn])
                sel = element_selection.upper()
                if sel == "CENTROID":
                    centroid = np.mean(elem_coords, axis=0)
                    if condition_fn(centroid):
                        gset.element_ids.add(eid)
                elif sel == "ALL_NODES":
                    if all(n in matching_nodes for n in conn):
                        gset.element_ids.add(eid)
                elif sel == "ANY_NODE":
                    if any(n in matching_nodes for n in conn):
                        gset.element_ids.add(eid)
                else:
                    centroid = np.mean(elem_coords, axis=0)
                    if condition_fn(centroid):
                        gset.element_ids.add(eid)

        return gset

    @classmethod
    def from_sphere(
        cls,
        part: Any,
        name: str,
        center: Sequence[float],
        radius: float,
        inner_radius: float = 0.0,
        entity_type: str = "ALL",
        element_selection: str = "CENTROID"
    ) -> GeneralSet:
        """Create a GeneralSet filtering entities within a spherical (or cylindrical 2D) volume.
        
        Parameters
        ----------
        center : Sequence[float]
            Center coordinates (x, y) or (x, y, z)
        radius : float
            Outer radius
        inner_radius : float, optional
            Inner radius for hollow sphere/shell filtering (default 0.0)
        """
        c = np.array(center, dtype=np.float64)
        r_out_sq = float(radius) ** 2
        r_in_sq = float(inner_radius) ** 2
        tol = 1e-7

        def in_sphere(coords: np.ndarray) -> bool:
            dim = min(len(coords), len(c))
            diff = coords[:dim] - c[:dim]
            extra_sq = 0.0
            if len(coords) > dim:
                extra_sq += np.sum(coords[dim:] ** 2)
            elif len(c) > dim:
                extra_sq += np.sum(c[dim:] ** 2)
            dist_sq = np.sum(diff ** 2) + extra_sq
            return (r_in_sq - tol) <= dist_sq <= (r_out_sq + tol)

        return cls.from_condition(
            part=part,
            name=name,
            condition_fn=in_sphere,
            entity_type=entity_type,
            element_selection=element_selection
        )

    @classmethod
    def from_cylinder(
        cls,
        part: Any,
        name: str,
        point1: Sequence[float],
        point2: Sequence[float],
        radius: float,
        inner_radius: float = 0.0,
        entity_type: str = "ALL",
        element_selection: str = "CENTROID"
    ) -> GeneralSet:
        """Create a GeneralSet filtering entities within a finite cylinder (hinge axis, shaft, roller).
        
        Parameters
        ----------
        point1 : Sequence[float]
            First endpoint of the cylinder axis
        point2 : Sequence[float]
            Second endpoint of the cylinder axis
        radius : float
            Outer radius from the axis
        inner_radius : float, optional
            Inner radius for hollow cylinder filtering (default 0.0)
        """
        p1 = np.array(point1, dtype=np.float64)
        p2 = np.array(point2, dtype=np.float64)
        v = p2 - p1
        L = np.linalg.norm(v)
        if L < 1e-12:
            return cls.from_sphere(part, name, center=p1, radius=radius, inner_radius=inner_radius,
                                  entity_type=entity_type, element_selection=element_selection)

        axis = v / L
        r_out_sq = float(radius) ** 2
        r_in_sq = float(inner_radius) ** 2
        tol = 1e-6

        def in_cylinder(coords: np.ndarray) -> bool:
            # Match dimension up to 3D
            c3 = np.zeros(3, dtype=np.float64)
            c3[:len(coords)] = coords
            pt1_3 = np.zeros(3, dtype=np.float64)
            pt1_3[:len(p1)] = p1
            axis_3 = np.zeros(3, dtype=np.float64)
            axis_3[:len(axis)] = axis

            w = c3 - pt1_3
            t = np.dot(w, axis_3)
            if t < -tol or t > L + tol:
                return False
            d_sq = np.dot(w, w) - t * t
            return (r_in_sq - tol) <= d_sq <= (r_out_sq + tol)

        return cls.from_condition(
            part=part,
            name=name,
            condition_fn=in_cylinder,
            entity_type=entity_type,
            element_selection=element_selection
        )

    @classmethod
    def from_plane(
        cls,
        part: Any,
        name: str,
        point: Sequence[float],
        normal: Sequence[float],
        side: str = "ON_PLANE",  # "ON_PLANE", "POSITIVE", "NEGATIVE"
        tol: float = 1e-4,
        entity_type: str = "ALL",
        element_selection: str = "CENTROID"
    ) -> GeneralSet:
        """Create a GeneralSet filtering entities on or to one side of a plane (e.g. symmetry BC).
        
        Parameters
        ----------
        point : Sequence[float]
            Point on the plane
        normal : Sequence[float]
            Plane normal vector
        side : str
            "ON_PLANE" (|dist| <= tol), "POSITIVE" (dist >= -tol), "NEGATIVE" (dist <= tol)
        tol : float
            Geometric distance tolerance
        """
        p = np.array(point, dtype=np.float64)
        n = np.array(normal, dtype=np.float64)
        n_len = np.linalg.norm(n)
        if n_len < 1e-12:
            raise ValueError("Normal vector length cannot be zero.")
        n_hat = n / n_len
        side_upper = side.upper()

        def in_plane(coords: np.ndarray) -> bool:
            c3 = np.zeros(3, dtype=np.float64)
            c3[:len(coords)] = coords
            p3 = np.zeros(3, dtype=np.float64)
            p3[:len(p)] = p
            n3 = np.zeros(3, dtype=np.float64)
            n3[:len(n_hat)] = n_hat

            dist = np.dot(c3 - p3, n3)
            if side_upper in ["ON", "ON_PLANE"]:
                return abs(dist) <= tol
            elif side_upper in ["POS", "POSITIVE"]:
                return dist >= -tol
            elif side_upper in ["NEG", "NEGATIVE"]:
                return dist <= tol
            else:
                raise ValueError(f"Unknown side '{side}', expected 'ON_PLANE', 'POSITIVE', or 'NEGATIVE'")

        return cls.from_condition(
            part=part,
            name=name,
            condition_fn=in_plane,
            entity_type=entity_type,
            element_selection=element_selection
        )

    @classmethod
    def from_surface_normal(
        cls,
        part: Any,
        name: str,
        direction: Sequence[float],
        angle_tol_deg: float = 15.0
    ) -> GeneralSet:
        """Create a GeneralSet containing exterior boundary faces whose outward normal matches direction.
        
        Useful for extracting Top/Bottom display surfaces, load application boundaries, etc.
        
        Parameters
        ----------
        direction : Sequence[float]
            Target outward normal vector (e.g. [0, 1, 0] for +Y top surface)
        angle_tol_deg : float
            Maximum allowable angular deviation in degrees (default 15.0)
        """
        gset = cls(name=name, part=part)
        d = np.array(direction, dtype=np.float64)
        d_len = np.linalg.norm(d)
        if d_len < 1e-12:
            raise ValueError("Direction vector length cannot be zero.")
        d_hat = d / d_len
        cos_tol = np.cos(np.radians(angle_tol_deg))

        # 1. Compute all exterior faces for this part
        all_part_set = cls(name="_temp_all", part=part, elements=list(part.elements.keys()))
        exterior_faces = all_part_set.get_faces(exterior_only=True)

        matching_faces: List[ElementFace] = []
        matching_nodes: Set[int] = set()
        matching_elements: Set[int] = set()

        for ef in exterior_faces:
            if ef.element_id not in part.elements:
                continue
            etype, conn = part.elements[ef.element_id]
            n_nodes = len(conn)

            # Node coordinates for the element
            elem_nodes = [part.nodes[n] for n in conn]
            elem_centroid = np.mean(elem_nodes, axis=0)

            face_normal = None
            face_nodes_idx = []

            if n_nodes == 4 and ef.face_id < 4:  # 2D Quad edge
                i0, i1 = _QUAD4_EDGE_NODES[ef.face_id]
                p0 = part.nodes[conn[i0]]
                p1 = part.nodes[conn[i1]]
                face_nodes_idx = [conn[i0], conn[i1]]
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                # CCW normal is (dy, -dx)
                normal_2d = np.array([dy, -dx], dtype=np.float64)
                mid = 0.5 * (p0 + p1)
                # Ensure normal points outward away from element centroid
                if np.dot(normal_2d, mid[:2] - elem_centroid[:2]) < 0:
                    normal_2d = -normal_2d
                norm_len = np.linalg.norm(normal_2d)
                if norm_len > 1e-12:
                    face_normal = np.zeros(len(d_hat))
                    face_normal[:2] = normal_2d / norm_len

            elif n_nodes == 8 and ef.face_id < 6:  # 3D Hex face
                f_indices = _HEX8_FACE_NODES[ef.face_id]
                face_nodes_idx = [conn[idx] for idx in f_indices]
                p0 = part.nodes[face_nodes_idx[0]]
                p1 = part.nodes[face_nodes_idx[1]]
                p3 = part.nodes[face_nodes_idx[3]]
                v1 = p1 - p0
                v2 = p3 - p0
                normal_3d = np.cross(v1, v2)
                mid = np.mean([part.nodes[n] for n in face_nodes_idx], axis=0)
                if np.dot(normal_3d, mid - elem_centroid) < 0:
                    normal_3d = -normal_3d
                norm_len = np.linalg.norm(normal_3d)
                if norm_len > 1e-12:
                    face_normal = normal_3d / norm_len

            if face_normal is not None:
                dim_cmp = min(len(face_normal), len(d_hat))
                cos_angle = np.dot(face_normal[:dim_cmp], d_hat[:dim_cmp])
                if cos_angle >= cos_tol - 1e-6:
                    matching_faces.append(ef)
                    matching_nodes.update(face_nodes_idx)
                    matching_elements.add(ef.element_id)

        gset.faces = matching_faces
        gset.node_ids = matching_nodes
        gset.element_ids = set()
        return gset

    @classmethod
    def from_k_nearest(
        cls,
        part: Any,
        name: str,
        coords: Sequence[float],
        k: int = 1,
        entity_type: str = "NODES",  # "NODES", "ELEMENTS"
        search_tolerance: Optional[float] = None,
        raise_if_none: bool = False
    ) -> GeneralSet:
        """Create a GeneralSet containing the k nearest nodes or elements to a query coordinate.
        
        Parameters
        ----------
        coords : Sequence[float]
            Query coordinates (x, y) or (x, y, z)
        k : int
            Number of nearest entities to return (default 1)
        entity_type : str
            "NODES" or "ELEMENTS"
        search_tolerance : float, optional
            Maximum allowable distance from query coordinate
        raise_if_none : bool
            Raise ValueError if no entity is found within search_tolerance (default False)
        """
        gset = cls(name=name, part=part)
        q = np.array(coords, dtype=np.float64)
        k = max(1, int(k))
        tol_sq = float(search_tolerance) ** 2 if search_tolerance is not None else float("inf")

        if entity_type.upper() in ["ALL", "NODES"]:
            dist_list = []
            for nid, c in part.nodes.items():
                dim = min(len(q), len(c))
                d_sq = np.sum((c[:dim] - q[:dim]) ** 2)
                if len(q) > dim:
                    d_sq += np.sum(q[dim:] ** 2)
                elif len(c) > dim:
                    d_sq += np.sum(c[dim:] ** 2)
                if d_sq <= tol_sq + 1e-12:
                    dist_list.append((d_sq, nid))
            dist_list.sort(key=lambda x: x[0])
            if len(dist_list) == 0 and raise_if_none and search_tolerance is not None:
                raise ValueError(f"No node found within search_tolerance={search_tolerance} for coords={coords}")
            gset.node_ids = {nid for _, nid in dist_list[:k]}

        if entity_type.upper() in ["ALL", "ELEMENTS"]:
            dist_list = []
            for eid, (_, conn) in part.elements.items():
                elem_nodes = [part.nodes[n] for n in conn]
                centroid = np.mean(elem_nodes, axis=0)
                dim = min(len(q), len(centroid))
                d_sq = np.sum((centroid[:dim] - q[:dim]) ** 2)
                if len(q) > dim:
                    d_sq += np.sum(q[dim:] ** 2)
                elif len(centroid) > dim:
                    d_sq += np.sum(centroid[dim:] ** 2)
                if d_sq <= tol_sq + 1e-12:
                    dist_list.append((d_sq, eid))
            dist_list.sort(key=lambda x: x[0])
            if len(dist_list) == 0 and raise_if_none and search_tolerance is not None:
                raise ValueError(f"No element found within search_tolerance={search_tolerance} for coords={coords}")
            gset.element_ids = {eid for _, eid in dist_list[:k]}

        return gset

    @classmethod
    def from_angle_propagation(
        cls,
        part: Any,
        name: str,
        seed_coords: Union[Sequence[float], Sequence[Sequence[float]]],
        feature_angle_deg: float = 20.0,
        search_tolerance: Optional[float] = None,
        max_distance: Optional[float] = None,
        stop_at_nodes: Optional[Union[Set[int], Sequence[int]]] = None,
        target_normal: Optional[Sequence[float]] = None,
        raise_if_none: bool = False
    ) -> GeneralSet:
        """Create a GeneralSet containing a contiguous smooth surface starting from one or more seed locations.
        
        Uses outward normal flood-fill propagation (BFS). Stops propagation at sharp edges
        where the angle between adjacent face normals exceeds feature_angle_deg.
        
        Features:
        - Orthogonal Projection to face plane for finding accurate seed face (Abaqus findAt style).
        - Multiple seeds support (pass list of coordinates to merge separate surfaces).
        - search_tolerance: Max distance for seed lookup to prevent accidental wrong selections.
        - max_distance: Radial cutoff from seed coordinates.
        - stop_at_nodes: Boundary barrier (e.g. partition lines, welds).
        - target_normal: Thin-sheet wrap-around prevention filter.
        """
        gset = cls(name=name, part=part)
        cos_threshold = np.cos(np.radians(feature_angle_deg))

        # Parse seed coordinates into list of 1D float arrays
        if len(seed_coords) > 0 and isinstance(seed_coords[0], (list, tuple, np.ndarray)):
            query_pts = [np.array(pt, dtype=np.float64) for pt in seed_coords]
        else:
            query_pts = [np.array(seed_coords, dtype=np.float64)]

        # 1. Get all exterior boundary faces for this part
        all_part_set = cls(name="_temp_all", part=part, elements=list(part.elements.keys()))
        exterior_faces = all_part_set.get_faces(exterior_only=True)
        if len(exterior_faces) == 0:
            return gset

        # 2. Extract normals, centroids, face node coords, and node lists for each exterior face
        face_normals: List[np.ndarray] = []
        face_centroids: List[np.ndarray] = []
        face_nodes_list: List[List[int]] = []
        face_coords_list: List[List[np.ndarray]] = []

        for ef in exterior_faces:
            _, conn = part.elements[ef.element_id]
            n_nodes = len(conn)
            elem_nodes = [part.nodes[n] for n in conn]
            elem_centroid = np.mean(elem_nodes, axis=0)

            face_normal = np.zeros(part.dim, dtype=np.float64)
            face_mid = np.zeros(part.dim, dtype=np.float64)
            face_nids: List[int] = []
            face_crds: List[np.ndarray] = []

            if n_nodes == 4 and ef.face_id < 4:  # 2D Quad edge
                i0, i1 = _QUAD4_EDGE_NODES[ef.face_id]
                p0 = part.nodes[conn[i0]]
                p1 = part.nodes[conn[i1]]
                face_nids = [conn[i0], conn[i1]]
                face_crds = [p0, p1]
                dx = p1[0] - p0[0]
                dy = p1[1] - p0[1]
                normal_2d = np.array([dy, -dx], dtype=np.float64)
                mid = 0.5 * (p0 + p1)
                if np.dot(normal_2d, mid[:2] - elem_centroid[:2]) < 0:
                    normal_2d = -normal_2d
                norm_len = np.linalg.norm(normal_2d)
                if norm_len > 1e-12:
                    face_normal[:2] = normal_2d / norm_len
                face_mid = mid

            elif n_nodes == 8 and ef.face_id < 6:  # 3D Hex face
                f_indices = _HEX8_FACE_NODES[ef.face_id]
                face_nids = [conn[idx] for idx in f_indices]
                face_crds = [part.nodes[n] for n in face_nids]
                p0 = face_crds[0]
                p1 = face_crds[1]
                p3 = face_crds[3]
                v1 = p1 - p0
                v2 = p3 - p0
                normal_3d = np.cross(v1, v2)
                mid = np.mean(face_crds, axis=0)
                if np.dot(normal_3d, mid - elem_centroid) < 0:
                    normal_3d = -normal_3d
                norm_len = np.linalg.norm(normal_3d)
                if norm_len > 1e-12:
                    face_normal = normal_3d / norm_len
                face_mid = mid

            face_normals.append(face_normal)
            face_centroids.append(face_mid)
            face_nodes_list.append(face_nids)
            face_coords_list.append(face_crds)

        # 3. Find seed faces using Orthogonal Projection + Tolerance
        seed_face_indices: Set[int] = set()
        t_norm = np.array(target_normal, dtype=np.float64) if target_normal is not None else None
        if t_norm is not None:
            t_norm_len = np.linalg.norm(t_norm)
            if t_norm_len > 1e-12:
                t_norm = t_norm / t_norm_len

        for q in query_pts:
            best_idx = None
            best_proj_dist = float("inf")
            best_euc_dist = float("inf")
            best_euc_idx = None

            for idx, (fn, fc, fcrds) in enumerate(zip(face_normals, face_centroids, face_coords_list)):
                # If target_normal is specified, filter out opposing faces (cos < 0.2)
                if t_norm is not None:
                    dim_cmp = min(len(fn), len(t_norm))
                    if np.dot(fn[:dim_cmp], t_norm[:dim_cmp]) < 0.2:
                        continue

                dim = min(len(q), len(fc))
                # Euclidean distance to face centroid
                euc_d = np.linalg.norm(fc[:dim] - q[:dim])
                if euc_d < best_euc_dist:
                    best_euc_dist = euc_d
                    best_euc_idx = idx

                # Orthogonal Projection Check
                if part.dim == 2 and len(fcrds) == 2:
                    p0, p1 = fcrds[0][:dim], fcrds[1][:dim]
                    v = p1 - p0
                    v_sq = np.dot(v, v)
                    if v_sq > 1e-14:
                        t = np.dot(q[:dim] - p0, v) / v_sq
                        if -1e-4 <= t <= 1.0 + 1e-4:
                            p_proj = p0 + np.clip(t, 0.0, 1.0) * v
                            dist_ortho = np.linalg.norm(q[:dim] - p_proj)
                            if dist_ortho < best_proj_dist:
                                best_proj_dist = dist_ortho
                                best_idx = idx
                elif part.dim == 3 and len(fcrds) == 4:
                    p0 = fcrds[0][:dim]
                    n_hat = fn[:dim]
                    h = abs(np.dot(q[:dim] - p0, n_hat))
                    p_proj = q[:dim] - np.dot(q[:dim] - p0, n_hat) * n_hat
                    # Point in planar convex quadrilateral check
                    in_poly = True
                    for i in range(4):
                        pa = fcrds[i][:dim]
                        pb = fcrds[(i + 1) % 4][:dim]
                        edge = pb - pa
                        to_pt = p_proj - pa
                        cross_prod = np.cross(edge, to_pt)
                        if np.dot(cross_prod, n_hat) < -1e-6:
                            in_poly = False
                            break
                    if in_poly:
                        if h < best_proj_dist:
                            best_proj_dist = h
                            best_idx = idx

            # Choose best orthogonal face if found, otherwise fallback to closest centroid
            chosen_seed_idx = best_idx if best_idx is not None else best_euc_idx
            chosen_dist = best_proj_dist if best_idx is not None else best_euc_dist

            # Tolerance check
            if search_tolerance is not None and chosen_dist > search_tolerance:
                if raise_if_none:
                    raise ValueError(
                        f"Seed point {q} is too far ({chosen_dist:.4f}) from any boundary face; "
                        f"search_tolerance is {search_tolerance}."
                    )
                continue

            if chosen_seed_idx is not None:
                seed_face_indices.add(chosen_seed_idx)

        if len(seed_face_indices) == 0:
            return gset

        # 4. Build adjacency graph between exterior faces
        adjacency: Dict[int, List[int]] = {i: [] for i in range(len(exterior_faces))}
        if part.dim == 2:
            node_to_faces: Dict[int, List[int]] = {}
            for f_idx, nids in enumerate(face_nodes_list):
                for nid in nids:
                    node_to_faces.setdefault(nid, []).append(f_idx)
            for f_idx, nids in enumerate(face_nodes_list):
                for nid in nids:
                    for neighbor_idx in node_to_faces[nid]:
                        if neighbor_idx != f_idx and neighbor_idx not in adjacency[f_idx]:
                            adjacency[f_idx].append(neighbor_idx)
        else:  # 3D Hex
            edge_to_faces: Dict[Tuple[int, int], List[int]] = {}
            for f_idx, nids in enumerate(face_nodes_list):
                n_f = len(nids)
                for i in range(n_f):
                    na, nb = nids[i], nids[(i + 1) % n_f]
                    edge_key = tuple(sorted([na, nb]))
                    edge_to_faces.setdefault(edge_key, []).append(f_idx)
            for edge_key, face_indices in edge_to_faces.items():
                for i in range(len(face_indices)):
                    for j in range(i + 1, len(face_indices)):
                        fa = face_indices[i]
                        fb = face_indices[j]
                        if fb not in adjacency[fa]:
                            adjacency[fa].append(fb)
                        if fa not in adjacency[fb]:
                            adjacency[fb].append(fa)

        # 5. BFS Flood Fill starting from all resolved seed faces
        visited: Set[int] = set(seed_face_indices)
        queue: deque[int] = deque(seed_face_indices)
        stop_nodes_set = set(stop_at_nodes) if stop_at_nodes is not None else set()

        while queue:
            curr = queue.popleft()
            curr_norm = face_normals[curr]

            # Barrier check on current face: if all nodes are barrier, do not propagate from here
            if stop_nodes_set and set(face_nodes_list[curr]).issubset(stop_nodes_set):
                continue

            for nxt in adjacency[curr]:
                if nxt in visited:
                    continue

                # Stop barrier check: do not cross boundary if the shared interface contains a stop node
                if stop_nodes_set:
                    shared_nids = set(face_nodes_list[curr]) & set(face_nodes_list[nxt])
                    if any(nid in stop_nodes_set for nid in shared_nids):
                        continue

                # Max distance cutoff check from query points
                if max_distance is not None:
                    nxt_center = face_centroids[nxt]
                    min_dist_to_any_seed = min(
                        np.linalg.norm(nxt_center[:len(q)] - q[:len(nxt_center)]) for q in query_pts
                    )
                    if min_dist_to_any_seed > max_distance:
                        continue

                nxt_norm = face_normals[nxt]
                # Target normal guard (e.g. wrap-around to opposite side of thin plate)
                if t_norm is not None:
                    dim_cmp = min(len(nxt_norm), len(t_norm))
                    if np.dot(nxt_norm[:dim_cmp], t_norm[:dim_cmp]) < 0.2:
                        continue

                dim_cmp = min(len(curr_norm), len(nxt_norm))
                cos_ang = np.dot(curr_norm[:dim_cmp], nxt_norm[:dim_cmp])
                if cos_ang >= cos_threshold - 1e-6:
                    visited.add(nxt)
                    queue.append(nxt)

        # 6. Populate GeneralSet with visited faces and nodes
        matching_faces: List[ElementFace] = [exterior_faces[idx] for idx in visited]
        matching_nodes: Set[int] = set()
        for idx in visited:
            matching_nodes.update(face_nodes_list[idx])

        gset.faces = matching_faces
        gset.node_ids = matching_nodes
        gset.element_ids = set()
        return gset

    # ─── SET ALGEBRA OPERATORS & BOOLEAN METHODS ──────────────────────

    def union(self, other: GeneralSet, name: Optional[str] = None) -> GeneralSet:
        """Return the union of two GeneralSets (self | other).
        
        Performs entity-wise union: nodes with nodes, elements with elements,
        and faces with faces.
        """
        p = self.part or other.part
        set_name = name if name is not None else f"{self.name}_OR_{other.name}"
        return GeneralSet(
            name=set_name,
            part=p,
            nodes=self.node_ids | other.node_ids,
            elements=self.element_ids | other.element_ids,
            faces=list(set(self.faces) | set(other.faces))
        )

    def intersection(self, other: GeneralSet, name: Optional[str] = None) -> GeneralSet:
        """Return the intersection of two GeneralSets (self & other).
        
        Performs entity-wise intersection: nodes with nodes, elements with elements,
        and faces with faces.
        """
        p = self.part or other.part
        set_name = name if name is not None else f"{self.name}_AND_{other.name}"
        return GeneralSet(
            name=set_name,
            part=p,
            nodes=self.node_ids & other.node_ids,
            elements=self.element_ids & other.element_ids,
            faces=list(set(self.faces) & set(other.faces))
        )

    def difference(self, other: GeneralSet, name: Optional[str] = None) -> GeneralSet:
        """Return the difference of two GeneralSets (self - other).
        
        Performs entity-wise difference: nodes minus nodes, elements minus elements,
        and faces minus faces.
        """
        p = self.part or other.part
        set_name = name if name is not None else f"{self.name}_SUB_{other.name}"
        return GeneralSet(
            name=set_name,
            part=p,
            nodes=self.node_ids - other.node_ids,
            elements=self.element_ids - other.element_ids,
            faces=list(set(self.faces) - set(other.faces))
        )

    def get_overlapping_nodes(self, other: GeneralSet, include_elements: bool = False) -> np.ndarray:
        """Return sorted array of Node IDs overlapping between self and other.
        
        Parameters
        ----------
        other : GeneralSet
            The other set to compare with.
        include_elements : bool
            If True, expands elements in each set to their constituent nodes before
            computing the intersection. If False, only compares explicit node_ids.
        """
        if include_elements:
            nodes_self = set(self.get_nodes(include_elements=True))
            nodes_other = set(other.get_nodes(include_elements=True))
            overlap = nodes_self & nodes_other
        else:
            overlap = self.node_ids & other.node_ids
        return np.array(sorted(overlap), dtype=np.int64)

    def get_overlapping_elements(self, other: GeneralSet) -> np.ndarray:
        """Return sorted array of Element IDs overlapping between self and other."""
        overlap = self.element_ids & other.element_ids
        return np.array(sorted(overlap), dtype=np.int64)

    def get_overlapping_faces(self, other: GeneralSet) -> List[ElementFace]:
        """Return list of ElementFaces overlapping between self and other."""
        return list(set(self.faces) & set(other.faces))

    def __or__(self, other: GeneralSet) -> GeneralSet:
        """Union of two GeneralSets (self | other)."""
        return self.union(other)

    def __and__(self, other: GeneralSet) -> GeneralSet:
        """Intersection of two GeneralSets (self & other)."""
        return self.intersection(other)

    def __sub__(self, other: GeneralSet) -> GeneralSet:
        """Difference of two GeneralSets (self - other)."""
        return self.difference(other)

    def __len__(self) -> int:
        return len(self.element_ids) if len(self.element_ids) > 0 else len(self.node_ids)

    def __repr__(self) -> str:
        return (
            f"GeneralSet(name='{self.name}', num_nodes={len(self.node_ids)}, "
            f"num_elements={len(self.element_ids)}, num_faces={len(self.faces)})"
        )

