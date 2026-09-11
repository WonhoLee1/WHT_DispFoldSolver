"""Part and SectionAssignment definitions for CAE model hierarchy."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional, Union
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

    def add_element(self, eid: int, elem_type: str, node_ids: Sequence[int]) -> Tuple[str, List[int]]:
        """Add an element with local connectivity."""
        item = (str(elem_type).upper(), [int(n) for n in node_ids])
        self.elements[int(eid)] = item
        return item

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

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_elements(self) -> int:
        return len(self.elements)
