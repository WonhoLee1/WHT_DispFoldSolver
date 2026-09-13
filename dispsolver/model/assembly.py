"""Assembly definition and high-performance global flattening engine."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Sequence, Optional, Any, Union
import numpy as np

from dispsolver.model.part import Part
from dispsolver.model.instance import Instance
from dispsolver.model.set import NodeSet, ElementSet, Surface, SetScope


@dataclass
class FlattenedSolverSystem:
    """Production-ready Flat Solver Bundle directly consumable by DynamicSolver / DynamicSolver3D."""
    # 1. Global Coordinates & Connectivity
    coords: np.ndarray                      # (N_global, dim) float64
    elem_conn_0based: np.ndarray            # (Ne_global, nodes_per_elem) int64
    elem_types: List[str]                   # (Ne_global,) element type string per element
    dim: int                                # 2 or 3
    
    # 2. Material & Property IDs
    elem_pids: np.ndarray                   # (Ne_global,) int32
    materials_by_pid: Dict[int, Any]        # pid -> Material object or Numba props dict
    material_names_by_pid: Dict[int, str]   # pid -> Deck Material name
    
    # 3. ID Mappings (Bidirectional Traceability)
    node_local_to_global: Dict[Tuple[str, int], int]  # (instance_name, local_nid) -> global_nid
    elem_local_to_global: Dict[Tuple[str, int], int]  # (instance_name, local_eid) -> global_eid
    nid_to_idx: Dict[int, int]                        # global_nid -> 0-based contiguous index
    
    # 4. Global Qualified Sets & Surfaces ("INSTANCE_NAME.SET_NAME")
    global_nsets: Dict[str, np.ndarray]     # qualified name -> 0-based global node indices
    global_elsets: Dict[str, np.ndarray]    # qualified name -> 0-based global element indices
    
    # 5. Precomputed Sparse Matrix Topology (Instant CSR/CSC, zero-overhead in Numba)
    rows_topo: np.ndarray                   # Flat int32 row indices
    cols_topo: np.ndarray                   # Flat int32 col indices
    num_dofs: int                           # Total system DOFs (N_nodes * dim)
    
    # 6. Constraints mapped to Global DOFs
    constraints: List[Any] = field(default_factory=list)

    # 7. Predefined Fields (Initial Conditions)
    predefined_fields: List[Any] = field(default_factory=list)

    def to_mesh3d(self):
        """Construct a Mesh3D instance for DynamicSolver3D."""
        from dispsolver.mesh3d import Mesh3D
        mesh = Mesh3D()
        for gid, idx in self.nid_to_idx.items():
            c = self.coords[idx]
            z = c[2] if self.dim >= 3 else 0.0
            mesh.add_node(gid, c[0], c[1], z)
            
        for e_idx in range(len(self.elem_conn_0based)):
            gid = e_idx + 1
            # 1-based node IDs
            node_ids = [gid_val for gid_val, idx_val in self.nid_to_idx.items() if idx_val in self.elem_conn_0based[e_idx]]
            # Ensure correct connectivity ordering
            conn_0 = self.elem_conn_0based[e_idx]
            inv_nid = {idx: g for g, idx in self.nid_to_idx.items()}
            ordered_conn = [inv_nid[idx] for idx in conn_0]
            mesh.add_element(gid, ordered_conn, self.elem_types[e_idx], pid=int(self.elem_pids[e_idx]))
            
        return mesh

    def to_mesh2d(self):
        """Construct a modern Mesh2D instance for DynamicSolver2D."""
        from dispsolver.mesh2d import Mesh2D
        mesh = Mesh2D()
        inv_nid = {idx: g for g, idx in self.nid_to_idx.items()}
        for gid, idx in self.nid_to_idx.items():
            c = self.coords[idx]
            mesh.add_node(gid, c[0], c[1])
            
        for e_idx in range(len(self.elem_conn_0based)):
            gid = e_idx + 1
            conn_0 = self.elem_conn_0based[e_idx]
            ordered_conn = [inv_nid[idx] for idx in conn_0]
            mesh.add_element(gid, ordered_conn, self.elem_types[e_idx], pid=int(self.elem_pids[e_idx]))
            
        return mesh

    def to_legacy_mesh2d(self):
        """Construct a legacy 2D Mesh instance for DynamicSolver."""
        from dispsolver.mesh import Mesh, Node, Element
        mesh = Mesh()
        inv_nid = {idx: g for g, idx in self.nid_to_idx.items()}
        for gid, idx in self.nid_to_idx.items():
            c = self.coords[idx]
            mesh.nodes[gid] = Node(gid, c[0], c[1])
            mesh.coords[idx] = c[:2]
            mesh.nid_to_idx[gid] = idx
            
        for e_idx in range(len(self.elem_conn_0based)):
            gid = e_idx + 1
            conn_0 = self.elem_conn_0based[e_idx]
            ordered_conn = [inv_nid[idx] for idx in conn_0]
            mesh.elements[gid] = Element(gid, ordered_conn, self.elem_types[e_idx], pid=int(self.elem_pids[e_idx]))
            
        return mesh


class Assembly:
    """Root Assembly managing Part Instances, interactions, and global flattening."""

    def __init__(self, name: str = "Assembly", dim: int = 3):
        self.name = str(name)
        self.dim = int(dim)
        self.instances: Dict[str, Instance] = {}
        self.node_sets: Dict[str, NodeSet] = {}
        self.element_sets: Dict[str, ElementSet] = {}
        self.surfaces: Dict[str, Surface] = {}
        self.ties: List[Dict[str, Any]] = []
        self.rigid_bodies: List[Dict[str, Any]] = []

    def create_instance(self, name: str, part: Part, dependent: bool = True) -> Instance:
        """Instantiate a Part in this Assembly."""
        inst = Instance(name=str(name), part=part, dependent=dependent)
        self.instances[str(name)] = inst
        return inst

    def Instance(self, name: str, part: Part, dependent: bool = True) -> Instance:
        """Abaqus-style Instance factory alias."""
        return self.create_instance(name=name, part=part, dependent=dependent)

    def Set(
        self,
        name: str,
        nodes: Optional[Sequence[int]] = None,
        elements: Optional[Sequence[int]] = None
    ) -> Any:
        """Abaqus-compatible Set factory on Assembly."""
        if nodes is not None:
            nset = NodeSet(name=str(name), node_ids=[int(n) for n in nodes], scope=SetScope.ASSEMBLY)
            self.node_sets[str(name)] = nset
            return nset
        elif elements is not None:
            elset = ElementSet(name=str(name), element_ids=[int(e) for e in elements], scope=SetScope.ASSEMBLY)
            self.element_sets[str(name)] = elset
            return elset
        return None

    def add_tie(self, name: str, master: str, slave: str, position_tolerance: float = 0.0) -> None:
        """Define a surface tie constraint between two surfaces."""
        self.ties.append({
            "name": str(name),
            "master": str(master),
            "slave": str(slave),
            "position_tolerance": float(position_tolerance)
        })

    def add_rigid_body(self, name: str, ref_node: Union[int, Tuple[str, int]], region: str) -> None:
        """Bind a region to move rigidly with a reference node."""
        self.rigid_bodies.append({
            "name": str(name),
            "ref_node": ref_node,
            "region": str(region)
        })

    def build_solver_system(self, model: Any) -> FlattenedSolverSystem:
        """Flatten instances into global contiguous solver arrays and CSR sparsity topology.
        
        Guarantees:
        1. Zero-collision global node/element ID reassignment.
        2. Exact Section/Material -> PID resolution with sharing optimization.
        3. 3D affine transformation applied to all node coordinates.
        4. Instant CSR Sparsity Topology pre-calculation (Numba-compatible).
        """
        dim = self.dim
        dof_per_node = dim

        global_nid_counter = 1
        global_eid_counter = 1

        node_local_to_global: Dict[Tuple[str, int], int] = {}
        elem_local_to_global: Dict[Tuple[str, int], int] = {}

        all_coords_list: List[np.ndarray] = []
        all_elem_conn: List[List[int]] = []
        all_elem_types: List[str] = []
        all_elem_pids: List[int] = []

        # Section -> PID cache to ensure elements sharing section/material share PID
        section_to_pid: Dict[str, int] = {}
        materials_by_pid: Dict[int, Any] = {}
        material_names_by_pid: Dict[int, str] = {}
        current_pid = 0

        # 1. Flatten Instances
        for inst_name, inst in self.instances.items():
            part = inst.part
            local_nids = sorted(part.nodes.keys())
            
            # (A) Transform coordinates to world space
            raw_coords = np.array([part.nodes[nid] for nid in local_nids], dtype=np.float64)
            world_coords = inst.transform.apply(raw_coords)
            all_coords_list.append(world_coords)

            # (B) Map Node IDs
            inst_nid_map: Dict[int, int] = {}
            for idx, local_nid in enumerate(local_nids):
                gid = global_nid_counter
                global_nid_counter += 1
                node_local_to_global[(inst_name, local_nid)] = gid
                inst_nid_map[local_nid] = gid

            # (C) Build Element -> Section lookup for this Part
            elem_to_section: Dict[int, str] = {}
            for sa in part.section_assignments:
                elset = part.element_sets.get(sa.region)
                if elset:
                    for eid in elset.element_ids:
                        elem_to_section[eid] = sa.section_name

            # (D) Map Elements and Assign PIDs
            for local_eid, (etype, conn_local) in part.elements.items():
                gid = global_eid_counter
                global_eid_counter += 1
                elem_local_to_global[(inst_name, local_eid)] = gid

                conn_global = [inst_nid_map[nid] for nid in conn_local]
                all_elem_conn.append(conn_global)
                all_elem_types.append(etype)

                sec_name = elem_to_section.get(local_eid, "__DEFAULT__")
                if sec_name not in section_to_pid:
                    section_to_pid[sec_name] = current_pid
                    current_pid += 1
                    assigned_pid = section_to_pid[sec_name]
                    
                    sec_obj = model.sections.get(sec_name)
                    mat_name = sec_obj.material_name if sec_obj else "DEFAULT"
                    mat_obj = model.materials.get(mat_name)
                    
                    if mat_obj is not None:
                        # Convert to Numba DOD props if available
                        if hasattr(mat_obj, "to_numba_props"):
                            materials_by_pid[assigned_pid] = mat_obj.to_numba_props()
                        else:
                            materials_by_pid[assigned_pid] = mat_obj
                    else:
                        materials_by_pid[assigned_pid] = {"E": 4000.0, "nu": 0.3}
                    material_names_by_pid[assigned_pid] = mat_name

                all_elem_pids.append(section_to_pid[sec_name])

        # 2. Finalize Global Arrays
        global_coords = np.vstack(all_coords_list) if all_coords_list else np.zeros((0, dim), dtype=np.float64)
        n_nodes = len(global_coords)
        num_dofs = n_nodes * dof_per_node

        nid_to_idx = {gid: idx for idx, gid in enumerate(range(1, n_nodes + 1))}

        nodes_per_elem = len(all_elem_conn[0]) if all_elem_conn else 0
        elem_conn_0based = np.zeros((len(all_elem_conn), nodes_per_elem), dtype=np.int64)
        for e_idx, conn in enumerate(all_elem_conn):
            elem_conn_0based[e_idx, :] = [nid_to_idx[nid] for nid in conn]

        # 3. Resolve Qualified Sets (e.g. "INST_NAME.SET_NAME")
        global_nsets: Dict[str, np.ndarray] = {}
        for inst_name, inst in self.instances.items():
            for sname, nset in inst.part.node_sets.items():
                qual_name = f"{inst_name}.{sname}"
                gids = [node_local_to_global[(inst_name, nid)] for nid in nset.node_ids if (inst_name, nid) in node_local_to_global]
                global_nsets[qual_name] = np.array(sorted([nid_to_idx[g] for g in gids]), dtype=np.int64)

        # Assembly-level node sets
        for sname, nset in self.node_sets.items():
            gids = [nid for nid in nset.node_ids if nid in nid_to_idx]
            global_nsets[sname] = np.array(sorted([nid_to_idx[g] for g in gids]), dtype=np.int64)

        global_elsets: Dict[str, np.ndarray] = {}
        for inst_name, inst in self.instances.items():
            for sname, elset in inst.part.element_sets.items():
                qual_name = f"{inst_name}.{sname}"
                gids = [elem_local_to_global[(inst_name, eid)] for eid in elset.element_ids if (inst_name, eid) in elem_local_to_global]
                global_elsets[qual_name] = np.array(sorted([g - 1 for g in gids]), dtype=np.int64)

        # Assembly-level element sets
        for sname, elset in self.element_sets.items():
            gids = [eid for eid in elset.element_ids if eid <= len(elem_conn_0based)]
            global_elsets[sname] = np.array(sorted([g - 1 for g in gids]), dtype=np.int64)

        # Also resolve GeneralSets with associative resolution
        for inst_name, inst in self.instances.items():
            for sname, gset in inst.part.sets.items():
                qual_name = f"{inst_name}.{sname}"
                # Nodes (associative resolution from elements if needed)
                resolved_nodes = gset.get_nodes(include_elements=True)
                if len(resolved_nodes) > 0 and qual_name not in global_nsets:
                    gids = [node_local_to_global[(inst_name, nid)] for nid in resolved_nodes if (inst_name, nid) in node_local_to_global]
                    global_nsets[qual_name] = np.array(sorted([nid_to_idx[g] for g in gids]), dtype=np.int64)
                # Elements
                resolved_elems = gset.get_elements()
                if len(resolved_elems) > 0 and qual_name not in global_elsets:
                    gids = [elem_local_to_global[(inst_name, eid)] for eid in resolved_elems if (inst_name, eid) in elem_local_to_global]
                    global_elsets[qual_name] = np.array(sorted([g - 1 for g in gids]), dtype=np.int64)

        # 4. Precompute Global CSR/CSC Sparsity Topology (rows_topo, cols_topo)
        n_elems = len(elem_conn_0based)
        if n_elems > 0 and nodes_per_elem > 0:
            elem_dofs_count = nodes_per_elem * dof_per_node
            n_entries_per_elem = elem_dofs_count * elem_dofs_count

            rows_topo = np.zeros(n_elems * n_entries_per_elem, dtype=np.int32)
            cols_topo = np.zeros(n_elems * n_entries_per_elem, dtype=np.int32)

            idx = 0
            dofs_e = np.zeros(elem_dofs_count, dtype=np.int32)
            for e in range(n_elems):
                conn_e = elem_conn_0based[e]
                for a in range(nodes_per_elem):
                    n = conn_e[a]
                    for d in range(dof_per_node):
                        dofs_e[a * dof_per_node + d] = n * dof_per_node + d
                for i in range(elem_dofs_count):
                    r = dofs_e[i]
                    for j in range(elem_dofs_count):
                        rows_topo[idx] = r
                        cols_topo[idx] = dofs_e[j]
                        idx += 1
        else:
            rows_topo = np.zeros(0, dtype=np.int32)
            cols_topo = np.zeros(0, dtype=np.int32)

        # 5. Compile Constraints from Model and Assembly
        compiled_constraints: List[Any] = []
        if hasattr(model, "constraints"):
            for cname, constraint in model.constraints.items():
                compiled_constraints.append(constraint)

        # 6. Compile Predefined Fields from InitialStep
        compiled_predefined_fields = []
        if hasattr(model, "initial_step"):
            compiled_predefined_fields = list(model.initial_step.predefined_fields.values())

        return FlattenedSolverSystem(
            coords=global_coords,
            elem_conn_0based=elem_conn_0based,
            elem_types=all_elem_types,
            dim=dim,
            elem_pids=np.array(all_elem_pids, dtype=np.int32),
            materials_by_pid=materials_by_pid,
            material_names_by_pid=material_names_by_pid,
            node_local_to_global=node_local_to_global,
            elem_local_to_global=elem_local_to_global,
            nid_to_idx=nid_to_idx,
            global_nsets=global_nsets,
            global_elsets=global_elsets,
            rows_topo=rows_topo,
            cols_topo=cols_topo,
            num_dofs=num_dofs,
            constraints=compiled_constraints,
            predefined_fields=compiled_predefined_fields
        )
