"""Part and SectionAssignment definitions for CAE model hierarchy."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional, Union, Callable
import numpy as np

from dispsolver.model.set import NodeSet, ElementSet, Surface, ElementFace, SetScope, GeneralSet


@dataclass
class SectionAssignment:
    """Binds an ElementSet to a Section within a Part."""
    region: str  # ElementSet name
    section_name: str
    offset: float = 0.0


class Part:
    """Part representation defined in its own local coordinate system."""

    def __init__(self, name: str, dim: int = 3):
        self.name = str(name)
        self.dim = int(dim)
        # nodes: local_nid -> np.ndarray([x, y]) or ([x, y, z])
        self.nodes: Dict[int, np.ndarray] = {}
        # elements: local_eid -> (element_type_str, list_of_local_node_ids)
        self.elements: Dict[int, Tuple[str, List[int]]] = {}
        # sets & surfaces
        self.sets: Dict[str, GeneralSet] = {}
        self.node_sets: Dict[str, NodeSet] = {}
        self.element_sets: Dict[str, ElementSet] = {}
        self.surfaces: Dict[str, Surface] = {}
        # section assignments
        self.section_assignments: List[SectionAssignment] = []

    def add_node(self, nid: int, coords: Sequence[float]) -> np.ndarray:
        """Add a node with local coordinates."""
        arr = np.array(coords[:self.dim], dtype=np.float64)
        self.nodes[int(nid)] = arr
        return arr

    def Node(self, nid: int, x: float, y: float, z: float = 0.0) -> np.ndarray:
        """Abaqus-compatible Node factory on Part."""
        return self.add_node(nid, (x, y, z))

    def add_element(self, eid: int, elem_type: str, node_ids: Sequence[int]) -> Tuple[str, List[int]]:
        """Add an element with local connectivity."""
        item = (str(elem_type).upper(), [int(n) for n in node_ids])
        self.elements[int(eid)] = item
        return item

    def Element(self, eid: int, elem_type: str, node_ids: Sequence[int]) -> Tuple[str, List[int]]:
        """Abaqus-compatible Element factory on Part."""
        return self.add_element(eid, elem_type, node_ids)

    def create_set(
        self,
        name: str,
        nodes: Optional[Sequence[int]] = None,
        elements: Optional[Sequence[int]] = None,
        faces: Optional[Sequence[Union[ElementFace, Tuple[int, int]]]] = None,
    ) -> GeneralSet:
        """Create a smart GeneralSet holding nodes, elements, and/or faces."""
        gset = GeneralSet(
            name=str(name),
            part=self,
            nodes=nodes,
            elements=elements,
            faces=faces,
            scope=SetScope.PART
        )
        self.sets[str(name)] = gset
        # Maintain legacy element_sets and node_sets for backward compatibility
        if elements is not None and len(elements) > 0:
            self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(elements), scope=SetScope.PART)
        if nodes is not None and len(nodes) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(nodes), scope=SetScope.PART)
        return gset

    def Set(self, name: str, nodes: Optional[Sequence[int]] = None, elements: Optional[Sequence[int]] = None) -> GeneralSet:
        """Abaqus-compatible Set factory on Part."""
        return self.create_set(name=name, nodes=nodes, elements=elements)

    def NodeSet(self, name: str, nodes: Sequence[int]) -> GeneralSet:
        """Abaqus-compatible NodeSet factory on Part."""
        return self.create_set(name=name, nodes=nodes)

    def ElementSet(self, name: str, elements: Sequence[int]) -> GeneralSet:
        """Abaqus-compatible ElementSet factory on Part."""
        return self.create_set(name=name, elements=elements)

    def create_set_from_box(
        self,
        name: str,
        x_range: Optional[Tuple[float, float]] = None,
        y_range: Optional[Tuple[float, float]] = None,
        z_range: Optional[Tuple[float, float]] = None,
        entity_type: str = "ALL"
    ) -> GeneralSet:
        """Create a GeneralSet by filtering nodes and elements within coordinate ranges."""
        gset = GeneralSet.from_bounding_box(
            part=self,
            name=str(name),
            x_bounds=x_range,
            y_bounds=y_range,
            z_bounds=z_range,
            entity_type=entity_type
        )
        self.sets[str(name)] = gset
        if len(gset.element_ids) > 0:
            self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(gset.element_ids), scope=SetScope.PART)
        if len(gset.node_ids) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(gset.node_ids), scope=SetScope.PART)
        return gset

    def create_set_from_sphere(
        self,
        name: str,
        center: Sequence[float],
        radius: float,
        inner_radius: float = 0.0,
        entity_type: str = "ALL",
        element_selection: str = "CENTROID"
    ) -> GeneralSet:
        """Create a GeneralSet filtering entities within a spherical (or cylindrical 2D) volume."""
        gset = GeneralSet.from_sphere(
            part=self,
            name=str(name),
            center=center,
            radius=radius,
            inner_radius=inner_radius,
            entity_type=entity_type,
            element_selection=element_selection
        )
        self.sets[str(name)] = gset
        if len(gset.element_ids) > 0:
            self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(gset.element_ids), scope=SetScope.PART)
        if len(gset.node_ids) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(gset.node_ids), scope=SetScope.PART)
        return gset

    def create_set_from_cylinder(
        self,
        name: str,
        point1: Sequence[float],
        point2: Sequence[float],
        radius: float,
        inner_radius: float = 0.0,
        entity_type: str = "ALL",
        element_selection: str = "CENTROID"
    ) -> GeneralSet:
        """Create a GeneralSet filtering entities within a finite cylinder (hinge axis, shaft, roller)."""
        gset = GeneralSet.from_cylinder(
            part=self,
            name=str(name),
            point1=point1,
            point2=point2,
            radius=radius,
            inner_radius=inner_radius,
            entity_type=entity_type,
            element_selection=element_selection
        )
        self.sets[str(name)] = gset
        if len(gset.element_ids) > 0:
            self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(gset.element_ids), scope=SetScope.PART)
        if len(gset.node_ids) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(gset.node_ids), scope=SetScope.PART)
        return gset

    def create_set_from_plane(
        self,
        name: str,
        point: Sequence[float],
        normal: Sequence[float],
        side: str = "ON_PLANE",
        tol: float = 1e-4,
        entity_type: str = "ALL",
        element_selection: str = "CENTROID"
    ) -> GeneralSet:
        """Create a GeneralSet filtering entities on or to one side of a plane (e.g. symmetry BC)."""
        gset = GeneralSet.from_plane(
            part=self,
            name=str(name),
            point=point,
            normal=normal,
            side=side,
            tol=tol,
            entity_type=entity_type,
            element_selection=element_selection
        )
        self.sets[str(name)] = gset
        if len(gset.element_ids) > 0:
            self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(gset.element_ids), scope=SetScope.PART)
        if len(gset.node_ids) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(gset.node_ids), scope=SetScope.PART)
        return gset

    def create_set_from_condition(
        self,
        name: str,
        condition_fn: Callable[[np.ndarray], bool],
        entity_type: str = "ALL",
        element_selection: str = "CENTROID"
    ) -> GeneralSet:
        """Create a GeneralSet by filtering nodes and elements using a custom predicate function."""
        gset = GeneralSet.from_condition(
            part=self,
            name=str(name),
            condition_fn=condition_fn,
            entity_type=entity_type,
            element_selection=element_selection
        )
        self.sets[str(name)] = gset
        if len(gset.element_ids) > 0:
            self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(gset.element_ids), scope=SetScope.PART)
        if len(gset.node_ids) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(gset.node_ids), scope=SetScope.PART)
        return gset

    def create_surface_from_normal(
        self,
        name: str,
        direction: Sequence[float],
        angle_tol_deg: float = 15.0
    ) -> GeneralSet:
        """Create a GeneralSet containing exterior boundary faces whose outward normal matches direction."""
        gset = GeneralSet.from_surface_normal(
            part=self,
            name=str(name),
            direction=direction,
            angle_tol_deg=angle_tol_deg
        )
        self.sets[str(name)] = gset
        if len(gset.node_ids) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(gset.node_ids), scope=SetScope.PART)
        if len(gset.element_ids) > 0:
            self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(gset.element_ids), scope=SetScope.PART)
        if len(gset.faces) > 0:
            self.surfaces[str(name)] = Surface(name=str(name), faces=list(gset.faces), node_ids=set(gset.node_ids), scope=SetScope.PART)
        return gset

    def get_closest_node(
        self,
        coords: Sequence[float],
        search_tolerance: Optional[float] = None,
        raise_if_none: bool = False
    ) -> Tuple[Optional[int], float]:
        """Find the single closest node to the query coordinates, optionally bounded by search_tolerance."""
        q = np.array(coords, dtype=np.float64)
        best_nid = None
        min_d_sq = float("inf")
        for nid, c in self.nodes.items():
            dim = min(len(q), len(c))
            d_sq = np.sum((c[:dim] - q[:dim]) ** 2)
            if len(q) > dim:
                d_sq += np.sum(q[dim:] ** 2)
            elif len(c) > dim:
                d_sq += np.sum(c[dim:] ** 2)
            if d_sq < min_d_sq:
                min_d_sq = d_sq
                best_nid = nid
        if best_nid is None:
            raise ValueError(f"Part '{self.name}' has no nodes.")

        dist = float(np.sqrt(min_d_sq))
        if search_tolerance is not None and dist > search_tolerance:
            if raise_if_none:
                raise ValueError(f"No node found within search_tolerance={search_tolerance} for coords={coords} (closest dist={dist:.4f})")
            return None, dist
        return best_nid, dist

    def get_closest_element(
        self,
        coords: Sequence[float],
        search_tolerance: Optional[float] = None,
        raise_if_none: bool = False
    ) -> Tuple[Optional[int], float]:
        """Find the single closest element (by centroid) to the query coordinates, optionally bounded by search_tolerance."""
        q = np.array(coords, dtype=np.float64)
        best_eid = None
        min_d_sq = float("inf")
        for eid, (_, conn) in self.elements.items():
            elem_nodes = [self.nodes[n] for n in conn]
            centroid = np.mean(elem_nodes, axis=0)
            dim = min(len(q), len(centroid))
            d_sq = np.sum((centroid[:dim] - q[:dim]) ** 2)
            if len(q) > dim:
                d_sq += np.sum(q[dim:] ** 2)
            elif len(centroid) > dim:
                d_sq += np.sum(centroid[dim:] ** 2)
            if d_sq < min_d_sq:
                min_d_sq = d_sq
                best_eid = eid
        if best_eid is None:
            raise ValueError(f"Part '{self.name}' has no elements.")

        dist = float(np.sqrt(min_d_sq))
        if search_tolerance is not None and dist > search_tolerance:
            if raise_if_none:
                raise ValueError(f"No element found within search_tolerance={search_tolerance} for coords={coords} (closest dist={dist:.4f})")
            return None, dist
        return best_eid, dist

    def find_closest_nodes(
        self,
        coords: Sequence[float],
        n: int = 1,
        search_tolerance: Optional[float] = None,
        return_distances: bool = False
    ) -> Union[List[int], List[Tuple[int, float]]]:
        """Find the n closest nodes to query coordinates, ordered by distance (ascending).
        
        Parameters
        ----------
        coords : Sequence[float]
            Query coordinates (x, y) or (x, y, z)
        n : int
            Number of closest nodes to return (default 1)
        search_tolerance : float, optional
            Maximum allowable distance from query coordinate. Only nodes within this
            tolerance will be included.
        return_distances : bool
            If True, returns list of (node_id, distance). If False, returns list of node_id.
        """
        q = np.array(coords, dtype=np.float64)
        tol_sq = float(search_tolerance) ** 2 if search_tolerance is not None else float("inf")
        candidates = []
        for nid, c in self.nodes.items():
            dim = min(len(q), len(c))
            d_sq = np.sum((c[:dim] - q[:dim]) ** 2)
            if len(q) > dim:
                d_sq += np.sum(q[dim:] ** 2)
            elif len(c) > dim:
                d_sq += np.sum(c[dim:] ** 2)
            if d_sq <= tol_sq + 1e-12:
                candidates.append((d_sq, nid))

        candidates.sort(key=lambda x: x[0])
        selected = candidates[:max(1, int(n))]
        if return_distances:
            return [(nid, float(np.sqrt(d_sq))) for d_sq, nid in selected]
        return [nid for _, nid in selected]

    def find_closest_elements(
        self,
        coords: Sequence[float],
        n: int = 1,
        search_tolerance: Optional[float] = None,
        return_distances: bool = False
    ) -> Union[List[int], List[Tuple[int, float]]]:
        """Find the n closest elements (by centroid) to query coordinates, ordered by distance.
        
        Parameters
        ----------
        coords : Sequence[float]
            Query coordinates (x, y) or (x, y, z)
        n : int
            Number of closest elements to return (default 1)
        search_tolerance : float, optional
            Maximum allowable distance from query coordinate.
        return_distances : bool
            If True, returns list of (element_id, distance). If False, returns list of element_id.
        """
        q = np.array(coords, dtype=np.float64)
        tol_sq = float(search_tolerance) ** 2 if search_tolerance is not None else float("inf")
        candidates = []
        for eid, (_, conn) in self.elements.items():
            elem_nodes = [self.nodes[node] for node in conn]
            centroid = np.mean(elem_nodes, axis=0)
            dim = min(len(q), len(centroid))
            d_sq = np.sum((centroid[:dim] - q[:dim]) ** 2)
            if len(q) > dim:
                d_sq += np.sum(q[dim:] ** 2)
            elif len(centroid) > dim:
                d_sq += np.sum(centroid[dim:] ** 2)
            if d_sq <= tol_sq + 1e-12:
                candidates.append((d_sq, eid))

        candidates.sort(key=lambda x: x[0])
        selected = candidates[:max(1, int(n))]
        if return_distances:
            return [(eid, float(np.sqrt(d_sq))) for d_sq, eid in selected]
        return [eid for _, eid in selected]

    def find_at(
        self,
        coords: Union[Sequence[float], Sequence[Sequence[float]]],
        name: Optional[str] = None,
        entity_type: str = "ALL",  # "ALL", "NODES", "ELEMENTS"
        n: int = 1,
        search_tolerance: Optional[float] = None
    ) -> GeneralSet:
        """Abaqus-style findAt function to extract n nearest nodes and/or elements as a GeneralSet.
        
        Parameters
        ----------
        coords : Sequence[float] or Sequence[Sequence[float]]
            One query point (x, y[, z]) or a sequence of query points [(x1, y1), (x2, y2), ...]
        name : str, optional
            Set name. If provided, registers the set in part.sets and part.node_sets/element_sets.
            If None, returns an unregistered GeneralSet.
        entity_type : str
            "ALL", "NODES", or "ELEMENTS"
        n : int
            Number of nearest entities to pick per query point (default 1)
        search_tolerance : float, optional
            Max distance cutoff.
        """
        pts: List[Sequence[float]]
        if len(coords) > 0 and isinstance(coords[0], (int, float, np.floating, np.integer)):
            pts = [coords]  # single coordinate vector
        else:
            pts = list(coords)

        collected_nodes: Set[int] = set()
        collected_elements: Set[int] = set()

        for pt in pts:
            if entity_type.upper() in ["ALL", "NODES"]:
                nids = self.find_closest_nodes(pt, n=n, search_tolerance=search_tolerance)
                collected_nodes.update(nids)
            if entity_type.upper() in ["ALL", "ELEMENTS"]:
                eids = self.find_closest_elements(pt, n=n, search_tolerance=search_tolerance)
                collected_elements.update(eids)

        set_name = str(name) if name is not None else f"FIND_AT_{len(self.sets) + 1}"
        gset = GeneralSet(
            name=set_name,
            part=self,
            nodes=collected_nodes,
            elements=collected_elements
        )
        if name is not None:
            self.sets[str(name)] = gset
            if len(gset.node_ids) > 0:
                self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(gset.node_ids), scope=SetScope.PART)
            if len(gset.element_ids) > 0:
                self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(gset.element_ids), scope=SetScope.PART)
        return gset

    def create_set_from_k_nearest(
        self,
        name: str,
        coords: Sequence[float],
        k: int = 1,
        entity_type: str = "NODES",
        search_tolerance: Optional[float] = None,
        raise_if_none: bool = False
    ) -> GeneralSet:
        """Create a GeneralSet containing the k nearest nodes or elements to query coordinates."""
        gset = GeneralSet.from_k_nearest(
            part=self,
            name=str(name),
            coords=coords,
            k=k,
            entity_type=entity_type,
            search_tolerance=search_tolerance,
            raise_if_none=raise_if_none
        )
        self.sets[str(name)] = gset
        if len(gset.node_ids) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(gset.node_ids), scope=SetScope.PART)
        if len(gset.element_ids) > 0:
            self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(gset.element_ids), scope=SetScope.PART)
        return gset

    def create_surface_by_angle(
        self,
        name: str,
        seed_coords: Union[Sequence[float], Sequence[Sequence[float]]],
        feature_angle_deg: float = 20.0,
        search_tolerance: Optional[float] = None,
        max_distance: Optional[float] = None,
        stop_at_nodes: Optional[Union[Set[int], Sequence[int]]] = None,
        target_normal: Optional[Sequence[float]] = None,
        raise_if_none: bool = False
    ) -> GeneralSet:
        """Create a continuous boundary surface starting from one or more seed points by propagating across adjacent faces within feature_angle_deg."""
        gset = GeneralSet.from_angle_propagation(
            part=self,
            name=str(name),
            seed_coords=seed_coords,
            feature_angle_deg=feature_angle_deg,
            search_tolerance=search_tolerance,
            max_distance=max_distance,
            stop_at_nodes=stop_at_nodes,
            target_normal=target_normal,
            raise_if_none=raise_if_none
        )
        self.sets[str(name)] = gset
        if len(gset.node_ids) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(gset.node_ids), scope=SetScope.PART)
        if len(gset.faces) > 0:
            self.surfaces[str(name)] = Surface(name=str(name), faces=list(gset.faces), node_ids=set(gset.node_ids), scope=SetScope.PART)
        return gset

    def create_set_by_angle_propagation(
        self,
        name: str,
        seed_coords: Union[Sequence[float], Sequence[Sequence[float]]],
        feature_angle_deg: float = 20.0,
        search_tolerance: Optional[float] = None,
        max_distance: Optional[float] = None,
        stop_at_nodes: Optional[Union[Set[int], Sequence[int]]] = None,
        target_normal: Optional[Sequence[float]] = None,
        raise_if_none: bool = False
    ) -> GeneralSet:
        """Alias for create_surface_by_angle."""
        return self.create_surface_by_angle(
            name=name,
            seed_coords=seed_coords,
            feature_angle_deg=feature_angle_deg,
            search_tolerance=search_tolerance,
            max_distance=max_distance,
            stop_at_nodes=stop_at_nodes,
            target_normal=target_normal,
            raise_if_none=raise_if_none
        )

    def create_node_set(self, name: str, node_ids: Sequence[int]) -> NodeSet:
        """Create a named NodeSet within the Part."""
        nset = NodeSet(name=str(name), node_ids=set(node_ids), scope=SetScope.PART)
        self.node_sets[str(name)] = nset
        # Also register in self.sets
        if str(name) not in self.sets:
            self.sets[str(name)] = GeneralSet(name=str(name), part=self, nodes=node_ids)
        return nset

    def create_element_set(self, name: str, element_ids: Sequence[int]) -> ElementSet:
        """Create a named ElementSet within the Part."""
        elset = ElementSet(name=str(name), element_ids=set(element_ids), scope=SetScope.PART)
        self.element_sets[str(name)] = elset
        # Also register in self.sets
        if str(name) not in self.sets:
            self.sets[str(name)] = GeneralSet(name=str(name), part=self, elements=element_ids)
        return elset

    def create_surface(self, name: str, faces: Sequence[Tuple[int, int]]) -> Surface:
        """Create a named Surface from (element_id, face_id) pairs."""
        surf = Surface(
            name=str(name),
            faces=[ElementFace(int(eid), int(fid)) for eid, fid in faces],
            scope=SetScope.PART
        )
        self.surfaces[str(name)] = surf
        return surf

    def _resolve_set(self, s: Union[str, GeneralSet]) -> GeneralSet:
        """Resolve a set identifier (name or GeneralSet) to a GeneralSet in this Part."""
        if isinstance(s, GeneralSet):
            return s
        if str(s) in self.sets:
            return self.sets[str(s)]
        if str(s) in self.node_sets:
            ns = self.node_sets[str(s)]
            return GeneralSet(name=ns.name, part=self, nodes=ns.node_ids)
        if str(s) in self.element_sets:
            es = self.element_sets[str(s)]
            return GeneralSet(name=es.name, part=self, elements=es.element_ids)
        raise KeyError(f"Set '{s}' not found in Part '{self.name}'.")

    def boolean_union(
        self,
        name: str,
        set_a: Union[str, GeneralSet],
        set_b: Union[str, GeneralSet]
    ) -> GeneralSet:
        """Perform union of two sets (set_a | set_b) and register the result in this Part."""
        sa = self._resolve_set(set_a)
        sb = self._resolve_set(set_b)
        res = sa.union(sb, name=str(name))
        self.sets[str(name)] = res
        if len(res.node_ids) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(res.node_ids), scope=SetScope.PART)
        if len(res.element_ids) > 0:
            self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(res.element_ids), scope=SetScope.PART)
        return res

    def boolean_intersection(
        self,
        name: str,
        set_a: Union[str, GeneralSet],
        set_b: Union[str, GeneralSet]
    ) -> GeneralSet:
        """Perform intersection of two sets (set_a & set_b) and register the result in this Part."""
        sa = self._resolve_set(set_a)
        sb = self._resolve_set(set_b)
        res = sa.intersection(sb, name=str(name))
        self.sets[str(name)] = res
        if len(res.node_ids) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(res.node_ids), scope=SetScope.PART)
        if len(res.element_ids) > 0:
            self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(res.element_ids), scope=SetScope.PART)
        return res

    def boolean_difference(
        self,
        name: str,
        set_a: Union[str, GeneralSet],
        set_b: Union[str, GeneralSet]
    ) -> GeneralSet:
        """Perform difference of two sets (set_a - set_b) and register the result in this Part."""
        sa = self._resolve_set(set_a)
        sb = self._resolve_set(set_b)
        res = sa.difference(sb, name=str(name))
        self.sets[str(name)] = res
        if len(res.node_ids) > 0:
            self.node_sets[str(name)] = NodeSet(name=str(name), node_ids=set(res.node_ids), scope=SetScope.PART)
        if len(res.element_ids) > 0:
            self.element_sets[str(name)] = ElementSet(name=str(name), element_ids=set(res.element_ids), scope=SetScope.PART)
        return res

    def get_overlapping_nodes(
        self,
        set_a: Union[str, GeneralSet],
        set_b: Union[str, GeneralSet],
        include_elements: bool = False
    ) -> np.ndarray:
        """Extract overlapping nodes between two sets.
        
        Parameters
        ----------
        set_a, set_b : str or GeneralSet
            The sets to compare.
        include_elements : bool
            If True, resolves nodes belonging to elements in each set before computing overlap.
        """
        sa = self._resolve_set(set_a)
        sb = self._resolve_set(set_b)
        return sa.get_overlapping_nodes(sb, include_elements=include_elements)

    def get_overlapping_elements(
        self,
        set_a: Union[str, GeneralSet],
        set_b: Union[str, GeneralSet]
    ) -> np.ndarray:
        """Extract overlapping elements between two sets."""
        sa = self._resolve_set(set_a)
        sb = self._resolve_set(set_b)
        return sa.get_overlapping_elements(sb)

    def assign_section(
        self,
        region: Union[str, GeneralSet, ElementSet],
        section_name: str,
        offset: float = 0.0
    ) -> SectionAssignment:
        """Assign a Section to an ElementSet or GeneralSet within this Part."""
        if hasattr(region, "name"):
            region_str = str(region.name)
        else:
            region_str = str(region)

        if region_str not in self.element_sets and region_str not in self.sets:
            raise KeyError(f"Region '{region_str}' does not exist in Part '{self.name}'")

        # If it's a GeneralSet with elements but not yet in element_sets
        if region_str not in self.element_sets and region_str in self.sets:
            gset = self.sets[region_str]
            self.element_sets[region_str] = ElementSet(name=region_str, element_ids=set(gset.element_ids), scope=SetScope.PART)

        assignment = SectionAssignment(region=region_str, section_name=str(section_name), offset=offset)
        self.section_assignments.append(assignment)
        return assignment

    def SectionAssignment(self, region: Union[str, GeneralSet, ElementSet], sectionName: str, offset: float = 0.0) -> SectionAssignment:
        """Abaqus-compatible SectionAssignment method on Part."""
        if region == "ALL" and "ALL" not in self.sets and "ALL" not in self.element_sets:
            self.create_set("ALL", elements=list(self.elements.keys()), nodes=list(self.nodes.keys()))
        return self.assign_section(region=region, section_name=sectionName, offset=offset)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)
