"""
legacy_adapter.py
=================
Adapter bridging the new Abaqus reader's `Model` object to the legacy
`ModelBuilderResult` expected by downstream solver callers (ex12, ex13, etc.).

This allows the new declarative reader pipeline to completely replace
`AbaqusParser` + `AbaqusModel` + `ModelBuilder`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import warnings
import numpy as np

from dispsolver.mesh import Mesh, Node, Element
from dispsolver.load.amplitude import Amplitude
from dispsolver.constraint.surface_tie import SurfaceTieConstraint
from dispsolver.constraint.rbe2_condensed import KinematicRBE2Constraint
from dispsolver.constraint.rbe2 import RBE2HingeConstraint
from dispsolver.contact.contact_surface import ContactSurface, auto_detect_exterior
from dispsolver.contact.contact_solver import ContactPair
from dispsolver.material import NeoHookean, Yeoh, ArrudaBoyce, J2Plasticity, ViscoelasticMaterial
from dispsolver.material.type_tags import J2_PLASTIC, ARRUDA_BOYCE_VISCO, NEOHOOKEAN
from dispsolver.model.model import Model
from dispsolver.model.constraint import Tie, RigidBody, MPC
from dispsolver.model.bc import DisplacementBC

_ABAQUS_TO_SOLVER_ELEM_TYPE = {
    "CPE4H": "CPE4H",
    "CPE4RH": "CPE4RH",
    "CPE4IH": "CPE4IH",
    "CPE4I": "CPE4I",
    "CAX4": "CAX4",
    "CAX4I": "CAX4I",
    "CAX4H": "CAX4H",
    "CAX4IH": "CAX4IH",
    "CAX4R": "CAX4R",
    "CAX4RH": "CAX4RH",
    "CAX8": "CAX8",
    "CAX8H": "CAX8H",
    "CAX8R": "CAX8R",
    "CAX8RH": "CAX8RH",
}
ABAQUS_ELEMENT_MAP = _ABAQUS_TO_SOLVER_ELEM_TYPE



class AbaqusBoundary:
    """Abaqus boundary condition specification (legacy-compatible)."""

    def __init__(self, nset: str, dof1: int, dof2: int, value: float = 0.0,
                 amplitude: Optional[str] = None):
        self.nset = str(nset)
        self.dof1 = int(dof1)
        self.dof2 = int(dof2)
        self.value = float(value)
        self.amplitude = amplitude

    def __repr__(self) -> str:
        return (f"AbaqusBoundary(nset={self.nset!r}, dof1={self.dof1}, "
                f"dof2={self.dof2}, value={self.value}, amplitude={self.amplitude!r})")


class ModelBuilderResult:
    """Structured result holding what `read_abaqus_input` returns today.

    Fields
    ------
    mesh : Mesh
        Flat mesh with all nodes, elements, sets.
    materials : dict[int, object]
        pid -> material model instance.
    material_params : dict[int, dict]
        pid -> material constructor kwargs.
    material_names : dict[int, str]
        pid -> *MATERIAL name from the deck.
    material_types : dict[int, str]
        pid -> canonical type tag (e.g. J2_PLASTIC, ARRUDA_BOYCE_VISCO).
    part_names : dict[int, str]
        pid -> human display-name override for non-layer parts.
    constraints : list
        List of solver constraint objects (RBE2HingeConstraint, TieConstraint).
    solver_config : dict
        Solver configuration fields.
    amplitudes : dict[str, Amplitude]
        name -> Amplitude instance.
    dload_configs : list[dict]
        List of surface traction config dicts.
    contact_pairs : list[dict]
        List of contact pair configs.
    contact_surfaces : list[dict]
        List of surface definitions.
    penalty_constraints : list
        SurfaceTieConstraint objects.
    rbe2_constraints : list
        KinematicRBE2Constraint objects.
    rbe2_elements : list
        Legacy RBE2 element objects.
    boundaries : list[AbaqusBoundary]
        Prescribed boundary conditions.
    contact_pair_objects : list
        ContactPair solver objects.
    """

    def __init__(self):
        self.mesh: Optional[Mesh] = None
        self.materials: Dict[int, object] = {}
        self.material_params: Dict[int, dict] = {}
        self.material_names: Dict[int, str] = {}
        self.material_types: Dict[int, str] = {}
        self.part_names: Dict[int, str] = {}
        self.constraints: List = []
        self.solver_config: Dict[str, Any] = {}
        self.amplitudes: Dict[str, Amplitude] = {}
        self.dload_configs: List[dict] = []
        self.contact_pairs: List[dict] = []
        self.contact_surfaces: List[dict] = []
        self.penalty_constraints: List = []
        self.rbe2_elements: List = []
        self.rbe2_constraints: List = []
        self.boundaries: List[AbaqusBoundary] = []
        self.contact_pair_objects: List = []

    def __repr__(self) -> str:
        n_mat = len(self.materials)
        n_con = len(self.constraints)
        n_pen = len(self.penalty_constraints)
        n_rbe_el = len(self.rbe2_elements)
        n_rbe_co = len(self.rbe2_constraints)
        n_amp = len(self.amplitudes)
        n_dl = len(self.dload_configs)
        n_cp = len(self.contact_pairs)
        n_cs = len(self.contact_surfaces)
        n_cpo = len(self.contact_pair_objects)
        return (
            f"ModelBuilderResult("
            f"mesh={'yes' if self.mesh else 'no'}, "
            f"materials={n_mat}, constraints={n_con}, penalty_constraints={n_pen}, "
            f"rbe2_elements={n_rbe_el}, rbe2_constraints={n_rbe_co}, "
            f"amplitudes={n_amp}, dloads={n_dl}, "
            f"contact_pairs={n_cp}, surfaces={n_cs}, "
            f"contact_objects={n_cpo})"
        )


def _build_material_instance(mat_name: str, mat_obj: Any) -> Tuple[object, dict, str]:
    """Convert a Model.Material object into a runtime material model instance,
    its params dict, and canonical type tag."""
    if mat_obj is None:
        mat = NeoHookean()
        return mat, {"E": 1000.0, "nu": 0.3}, NEOHOOKEAN

    has_visco = bool(getattr(mat_obj, "viscoelastic_prony", None))
    has_hyper = bool(getattr(mat_obj, "hyperelastic", None))
    has_plastic = bool(getattr(mat_obj, "plastic", None))
    has_elastic = bool(getattr(mat_obj, "elastic", None))

    mat = None
    mat_params = {}
    type_tag = "unknown"

    if has_hyper:
        hyper = mat_obj.hyperelastic
        model_type = hyper.get("model", "").upper()
        if "ARRUDA" in model_type:
            mu = float(hyper.get("mu", 1.0))
            lambda_m = float(hyper.get("lambda_m", 3.0))
            K = float(hyper.get("K", 0.0))
            if K <= 0.0 and hyper.get("d") is not None and float(hyper.get("d")) > 0:
                K = 2.0 / float(hyper.get("d"))
            elif K <= 0.0:
                K = 100.0
            mat = ArrudaBoyce()
            mat_params = {"mu": mu, "lambda_m": lambda_m, "K": K}
            type_tag = ARRUDA_BOYCE_VISCO if has_visco else "arruda-boyce"
        elif "YEOH" in model_type:
            C10 = float(hyper.get("C10", 0.5))
            C20 = float(hyper.get("C20", 0.0))
            C30 = float(hyper.get("C30", 0.0))
            mat = Yeoh()
            mat_params = {"C10": C10, "C20": C20, "C30": C30}
            type_tag = "yeoh"
        elif "NEO HOOKE" in model_type:
            C10 = float(hyper.get("C10", 0.5))
            D1 = float(hyper.get("D1", 0.02))
            mu = 2.0 * C10
            K = 2.0 / D1 if D1 > 0 else 100.0
            lam = K - 2.0 * mu / 3.0
            mat = NeoHookean()
            mat_params = {"mu": mu, "lambda": lam}
            type_tag = NEOHOOKEAN
        elif "POLYNOMIAL" in model_type or "MOONEY" in model_type or hyper.get("table"):
            table = hyper.get("table", ())
            if table and len(table[0]) >= 1:
                C10 = float(table[0][0])
                C01 = float(table[0][1]) if len(table[0]) >= 2 else 0.0
                D1 = float(table[0][2]) if len(table[0]) >= 3 else 0.0
            else:
                C10 = float(hyper.get("C10", 0.5))
                C01 = float(hyper.get("C01", 0.0))
                D1 = float(hyper.get("D1", 0.0))
            mu = 2.0 * (C10 + C01)
            nu = 0.4999
            E = 2.0 * (1.0 + nu) * mu
            mat = NeoHookean()
            mat_params = {"E": E, "nu": nu, "mu": mu, "lambda": 2.0 / D1 if D1 > 0 else 100.0 * mu}
            type_tag = NEOHOOKEAN
        else:
            mat = NeoHookean()
            mat_params = {"E": 1000.0, "nu": 0.3}
            type_tag = NEOHOOKEAN

    elif has_elastic and has_plastic:
        E, nu = mat_obj.elastic
        ht = mat_obj.plastic
        sigma_y0 = float(ht[0][0])
        if len(ht) >= 2:
            delta_sigma = float(ht[-1][0] - ht[0][0])
            delta_eps = float(ht[-1][1] - ht[0][1])
            H = delta_sigma / max(delta_eps, 1e-30)
        else:
            H = 0.0
        mat = J2Plasticity(E=float(E), nu=float(nu), sigma_y0=sigma_y0, H=H)
        mat_params = {"E": float(E), "nu": float(nu), "sigma_y0": sigma_y0, "H": H}
        type_tag = J2_PLASTIC

    elif has_elastic:
        E, nu = mat_obj.elastic
        mat = NeoHookean()
        mat_params = {"E": float(E), "nu": float(nu)}
        type_tag = NEOHOOKEAN

    else:
        mat = NeoHookean()
        mat_params = {"E": 1000.0, "nu": 0.3}
        type_tag = NEOHOOKEAN

    if has_visco:
        base_mat = mat
        base_params = dict(mat_params)
        prony = mat_obj.viscoelastic_prony
        gi_list = [float(p[0]) for p in prony]
        tau_list = [float(p[2]) for p in prony]
        trs = getattr(mat_obj, "extra_params", {}).get("trs") or {}
        wlf_params = trs if trs.get("definition", "").upper() == "WLF" else None
        mat = ViscoelasticMaterial(base_mat, gi_list, tau_list, wlf_params=wlf_params)
        mat_params = {
            **base_params,
            "base": base_mat,
            "prony": [(g, 0.0, tau) for g, tau in zip(gi_list, tau_list)],
            "wlf": wlf_params,
        }
        if isinstance(base_mat, ArrudaBoyce):
            type_tag = ARRUDA_BOYCE_VISCO
        else:
            type_tag = "viscoelastic"

    return mat, mat_params, type_tag


def model_to_builder_result(model: Model) -> ModelBuilderResult:
    """Build a legacy `ModelBuilderResult` from a modern `dispsolver.model.Model`.

    Parameters
    ----------
    model : Model
        Populated Model object from `AbaqusInputReader`.

    Returns
    -------
    ModelBuilderResult
        Complete legacy bundle matching what `read_abaqus_input` produced.
    """
    result = ModelBuilderResult()
    # topology / pid flattening only: the materials are built below by _build_material_instance
    sys = model.root_assembly.build_solver_system(model, resolve_materials=False)

    single_instance = (len(model.root_assembly.instances) <= 1)

    mesh = Mesh()
    local_to_mesh_nid: Dict[Tuple[str, int], int] = {}
    local_to_mesh_eid: Dict[Tuple[str, int], int] = {}

    if single_instance and len(model.root_assembly.instances) == 1:
        # Preserve original node and element IDs for single-instance / flat decks
        inst_name, inst = next(iter(model.root_assembly.instances.items()))
        part = inst.part
        dim = model.dim

        # Compute world coordinates
        local_nids = sorted(part.nodes.keys())
        raw_coords = np.array([part.nodes[nid] for nid in local_nids], dtype=np.float64)
        shifted = raw_coords + np.asarray(inst.transform.translation[:dim], dtype=np.float64)
        rot_only = inst.transform.copy()
        rot_only.translation = np.zeros_like(rot_only.translation)
        if (rot_only.rotation_axis is not None
                and abs(rot_only.rotation_angle_rad) > 1e-15
                and float(np.linalg.norm(rot_only.rotation_axis)) < 1e-12):
            rot_only.rotation_axis = None
        world_coords = rot_only.apply(shifted)

        for idx, local_nid in enumerate(local_nids):
            c = world_coords[idx]
            mesh.add_node(local_nid, float(c[0]), float(c[1]))
            local_to_mesh_nid[(inst_name, local_nid)] = local_nid

        # Section lookup for PIDs
        elem_to_section: Dict[int, str] = {}
        for sa in part.section_assignments:
            elset = part.element_sets.get(sa.region)
            if elset is not None:
                for eid in elset.element_ids:
                    elem_to_section[eid] = sa.section_name

        section_names_in_order = list(model.sections.keys())

        for local_eid, (etype, conn_local) in part.elements.items():
            sec_name = elem_to_section.get(local_eid, "__DEFAULT__")
            if sec_name in section_names_in_order:
                pid_1based = section_names_in_order.index(sec_name) + 1
            else:
                pid_1based = 1
            mesh.add_element(local_eid, list(conn_local), etype, pid=pid_1based)
            local_to_mesh_eid[(inst_name, local_eid)] = local_eid

        # Node sets and element sets from part
        for sname, nset in part.node_sets.items():
            mesh.add_nodeset(sname, set(nset.node_ids))
        for sname, elset in part.element_sets.items():
            mesh.add_elementset(sname, set(elset.element_ids))
        # Also include any assembly-level sets
        for sname, nset in model.root_assembly.node_sets.items():
            if sname not in mesh.node_sets:
                mesh.add_nodeset(sname, set(nset.node_ids))
        for sname, elset in model.root_assembly.element_sets.items():
            if sname not in mesh.element_sets:
                mesh.add_elementset(sname, set(elset.element_ids))

    else:
        # Multi-instance assembly: use global flattened indexing
        inv_nid = {idx: gid for gid, idx in sys.nid_to_idx.items()}
        for gid, idx in sys.nid_to_idx.items():
            c = sys.coords[idx]
            mesh.add_node(gid, float(c[0]), float(c[1]))

        for e_idx in range(len(sys.elem_conn_0based)):
            gid = e_idx + 1
            conn_0 = sys.elem_conn_0based[e_idx]
            valid_conn = conn_0[conn_0 != -1]
            ordered_conn = [inv_nid[idx] for idx in valid_conn]
            pid_1based = int(sys.elem_pids[e_idx]) + 1
            mesh.add_element(gid, ordered_conn, sys.elem_types[e_idx], pid=pid_1based)

        for sname, n_indices in sys.global_nsets.items():
            nids = set(inv_nid[i] for i in n_indices if i in inv_nid)
            mesh.add_nodeset(sname, nids)

        for sname, e_indices in sys.global_elsets.items():
            eids = set(int(i) + 1 for i in e_indices)
            mesh.add_elementset(sname, eids)

    result.mesh = mesh

    # 2. Materials, material_params, material_names, material_types, part_names
    material_cache: Dict[str, Tuple[object, dict, str]] = {}
    pids_0based = sorted(sys.materials_by_pid.keys())

    # Build section to material mapping in section definition order
    sec_items = list(model.sections.items())
    for idx, (sec_name, sec_obj) in enumerate(sec_items):
        pid_1based = idx + 1
        mat_name = sec_obj.material_name if sec_obj else "DEFAULT"
        result.material_names[pid_1based] = mat_name

        if mat_name not in material_cache:
            m_obj = model.materials.get(mat_name)
            material_cache[mat_name] = _build_material_instance(mat_name, m_obj)

        mat, mat_params, type_tag = material_cache[mat_name]
        result.materials[pid_1based] = mat
        result.material_params[pid_1based] = mat_params
        result.material_types[pid_1based] = type_tag

        # Plate labeling
        if "PLATE_" in sec_name.upper():
            suffix = sec_name.upper().split("PLATE_")[-1]
            result.part_names[pid_1based] = f"Plate {suffix.title()}"

    # Fallback if no sections defined
    if not result.materials and pids_0based:
        for p0 in pids_0based:
            pid_1based = p0 + 1
            mat_name = sys.material_names_by_pid.get(p0, "DEFAULT")
            result.material_names[pid_1based] = mat_name
            if mat_name not in material_cache:
                m_obj = model.materials.get(mat_name)
                material_cache[mat_name] = _build_material_instance(mat_name, m_obj)
            mat, mat_params, type_tag = material_cache[mat_name]
            result.materials[pid_1based] = mat
            result.material_params[pid_1based] = mat_params
            result.material_types[pid_1based] = type_tag

    # 3. Constraints (SurfaceTieConstraint, KinematicRBE2Constraint, MPC)
    def _resolve_nodes(name_or_set: Any) -> List[int]:
        if isinstance(name_or_set, (int, float)):
            return [int(name_or_set)]
        s_str = str(name_or_set)
        if s_str.isdigit():
            return [int(s_str)]
        if s_str in mesh.node_sets:
            return list(mesh.node_sets[s_str].node_ids)
        if s_str in mesh.element_sets:
            node_ids = set()
            for eid in mesh.element_sets[s_str].element_ids:
                if eid in mesh.elements:
                    node_ids.update(mesh.elements[eid].node_ids)
            return list(node_ids)
        # Check part node sets directly
        for part in model.parts.values():
            if s_str in part.node_sets:
                return list(part.node_sets[s_str].node_ids)
        return []

    # Model and assembly constraints
    all_constraints = list(model.constraints.values())
    for tie_dict in getattr(model.root_assembly, "ties", []):
        all_constraints.append(Tie(
            name=tie_dict.get("name", "TIE"),
            master=tie_dict.get("master"),
            slave=tie_dict.get("slave"),
            position_tolerance=tie_dict.get("position_tolerance", 0.0),
        ))
    for rb_dict in getattr(model.root_assembly, "rigid_bodies", []):
        all_constraints.append(RigidBody(
            name=rb_dict.get("name", "RB"),
            ref_point=rb_dict.get("ref_node"),
            tie_nodes=rb_dict.get("region"),
        ))

    nid_to_idx = mesh.node_id_to_index()
    coords = mesh.nodes_array()

    for c in all_constraints:
        if isinstance(c, Tie):
            slave_name = c.secondary if hasattr(c, "secondary") and c.secondary else getattr(c, "slave", None)
            master_name = c.main if hasattr(c, "main") and c.main else getattr(c, "master", None)
            slave_nodes = _resolve_nodes(slave_name)
            master_nodes = _resolve_nodes(master_name)
            if slave_nodes and master_nodes:
                penalty = float(c.position_tolerance) if c.position_tolerance and float(c.position_tolerance) > 1e3 else 1e6
                tie_name = f"{slave_name}_{master_name}_TIE"
                tie_con = SurfaceTieConstraint(
                    slave_node_ids=slave_nodes,
                    master_node_ids=master_nodes,
                    nid_to_idx=nid_to_idx,
                    coords=coords,
                    penalty_stiffness=penalty,
                    name=tie_name,
                )
                result.penalty_constraints.append(tie_con)

        elif isinstance(c, RigidBody):
            ref_node = int(c.ref_point)
            slave_target = c.tie_nodes if c.tie_nodes else c.pin_nodes
            slave_nodes = _resolve_nodes(slave_target)
            if slave_nodes and ref_node in nid_to_idx:
                rb_con = KinematicRBE2Constraint(
                    mesh=mesh,
                    master_id=ref_node,
                    slave_ids=slave_nodes,
                )
                result.rbe2_constraints.append(rb_con)

        elif isinstance(c, MPC):
            mpc_type = getattr(c, "mpc_type", "").upper()
            if mpc_type in ("RBE2", "BEAM"):
                nodes = _resolve_nodes(c.first_point) + _resolve_nodes(c.second_point)
                if len(nodes) >= 2:
                    ext_offset = sum(con.n_extra_primal() for con in result.constraints)
                    con = RBE2HingeConstraint(
                        mesh, master_id=nodes[0],
                        slave_ids=nodes[1:],
                        extra_primal_offset=ext_offset,
                    )
                    result.constraints.append(con)

    # 4. Amplitudes
    for amp_name, amp_obj in model.amplitudes.items():
        smooth = getattr(amp_obj, "smooth", False)
        time_type = getattr(amp_obj, "time_type", "STEP")
        data = getattr(amp_obj, "data", [])
        result.amplitudes[amp_name] = Amplitude(
            name=amp_name, smooth=smooth, time_type=time_type, data=data
        )

    # 5. Boundary conditions from Steps
    for sname, step in model.steps.items():
        for bc_entry in getattr(step, "boundary_conditions", {}).values():
            bc = bc_entry.entity if hasattr(bc_entry, "entity") else bc_entry
            if isinstance(bc, DisplacementBC):
                reg = str(bc.region)
                u1 = getattr(bc, "u1", None)
                u2 = getattr(bc, "u2", None)
                ur3 = getattr(bc, "ur3", None)
                amp = getattr(bc, "amplitude", None)
                if u1 is not None and u2 is not None and float(u1) == float(u2):
                    result.boundaries.append(
                        AbaqusBoundary(nset=reg, dof1=1, dof2=2, value=float(u1), amplitude=amp)
                    )
                else:
                    if u1 is not None:
                        result.boundaries.append(
                            AbaqusBoundary(nset=reg, dof1=1, dof2=1, value=float(u1), amplitude=amp)
                        )
                    if u2 is not None:
                        result.boundaries.append(
                            AbaqusBoundary(nset=reg, dof1=2, dof2=2, value=float(u2), amplitude=amp)
                        )
                if ur3 is not None:
                    result.boundaries.append(
                        AbaqusBoundary(nset=reg, dof1=6, dof2=6, value=float(ur3), amplitude=amp)
                    )

    # 6. Solver config
    density = 1e-9
    for m in model.materials.values():
        if getattr(m, "density", 0.0) > 0:
            density = float(m.density)
            break

    solver_cfg: Dict[str, Any] = {
        "density": density,
        "max_iter": 20,
        "tol": 1e-8,
        "static_mode": False,
        "alpha": 0.0,
        "amplitude_map": {},
        "bc_amp_map": {},
        "load_amp_map": {},
    }

    non_initial_steps = [s for sname, s in model.steps.items() if sname != "Initial"]
    target_step = non_initial_steps[0] if non_initial_steps else model.initial_step
    if target_step is not None:
        proc = getattr(target_step, "procedure", "DYNAMIC").upper()
        solver_cfg["procedure"] = proc
        if proc == "STATIC":
            solver_cfg["static_mode"] = True
            solver_cfg["dt_init"] = getattr(target_step, "dt_init", 0.1)
            solver_cfg["t_total"] = getattr(target_step, "time_period", 1.0)
            solver_cfg["dt_min"] = getattr(target_step, "dt_min", 1e-10)
            solver_cfg["dt_max"] = getattr(target_step, "dt_max", solver_cfg["t_total"])
        else:
            solver_cfg["dt_init"] = getattr(target_step, "dt_init", 0.01)
            solver_cfg["t_total"] = getattr(target_step, "time_period", 1.0)
            solver_cfg["dt_min"] = getattr(target_step, "dt_min", 1e-10)
            solver_cfg["dt_max"] = getattr(target_step, "dt_max", solver_cfg["t_total"])
            # *DYNAMIC's resolved HHT operator (io/keywords/steps.py); Step has no
            # 'alpha' attribute, so this used to be 0.0 (Newmark) whatever the deck said
            ti = getattr(target_step, "time_integration", None) or {}
            if "alpha" in ti:
                solver_cfg["alpha"] = float(ti["alpha"])
                solver_cfg["beta"] = float(ti["beta"])
                solver_cfg["gamma"] = float(ti["gamma"])
                solver_cfg["time_integrator"] = ti.get("name")
                solver_cfg["initial_acceleration"] = bool(ti.get("initial_acceleration", True))

    if mesh.elements:
        first_elem = next(iter(mesh.elements.values()))
        solver_cfg["element_type"] = first_elem.elem_type

    result.solver_config = solver_cfg

    # 7. Surfaces and Contact
    for sname in mesh.node_sets:
        result.contact_surfaces.append({
            "name": sname,
            "surface_type": "NODE",
            "definitions": [(sname, "")],
        })

    return result
