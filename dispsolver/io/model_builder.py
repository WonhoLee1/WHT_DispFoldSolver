"""
model_builder.py
================
Convert AbaqusModel → dispsolver native objects (Mesh, Material, Constraints, etc.).

Part of the Abaqus .inp import pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import warnings

import numpy as np

from .abaqus_model import (
    AbaqusAmplitude,
    AbaqusBoundary,
    AbaqusContactPair,
    AbaqusContactProperty,
    AbaqusDload,
    AbaqusElement,
    AbaqusLoad,
    AbaqusMaterial,
    AbaqusMpc,
    AbaqusNode,
    AbaqusModel,
    AbaqusSection,
    AbaqusStep,
    AbaqusSurface,
    AbaqusTie,
)
from ..mesh import Mesh
from ..load.amplitude import Amplitude
from ..contact.contact_surface import ContactSurface, auto_detect_exterior
from ..contact.contact_solver import ContactPair


# ──────────────────────────────────────────────
# Material model imports (lazy)
# ──────────────────────────────────────────────

def _material_info(mat_type: str, params: dict) -> str:
    """Return human-readable material type summary."""
    if mat_type == "ELASTIC":
        return "NeoHookean"
    elif mat_type == "HYPERELASTIC":
        return params.get("model", "UNKNOWN")
    elif mat_type == "PLASTIC":
        return "J2Plasticity"
    elif mat_type == "VISCOELASTIC":
        return "ViscoelasticMaterial"
    else:
        return mat_type


# ──────────────────────────────────────────────
# Builder result
# ──────────────────────────────────────────────

class ModelBuilderResult:
    """Structured result from ModelBuilder.build().

    Fields
    ------
    mesh : Mesh
        Flat mesh with all nodes, elements, sets.
    materials : dict[int, object]
        pid → material model instance.
    material_params : dict[int, dict]
        pid → material constructor kwargs.
    constraints : list
        List of solver constraint objects (RBE2HingeConstraint, TieConstraint).
    solver_config : dict
        Solver configuration fields (density, dt_init, t_total, dt_min, dt_max,
        static_mode, alpha, max_iter, tol, amplitude_map, bc_amp_map, load_amp_map).
    amplitudes : dict[str, Amplitude]
        name → Amplitude instance.
    dload_configs : list[dict]
        List of surface traction config dicts: {elset, face, pressure, amplitude_name}.
    contact_pairs : list[dict]
        List of contact pair configs: {name, slave_surface, master_surface,
        penalty_stiffness, friction_coeff}.
    contact_surfaces : list[dict]
        List of surface definitions: {name, surface_type, definitions}.
    contact_pair_objects : list[ContactPair]
        ContactPair solver objects (ready to pass to DynamicSolver).
    """

    def __init__(self):
        self.mesh: Optional[Mesh] = None
        self.materials: Dict[int, object] = {}
        self.material_params: Dict[int, dict] = {}
        self.constraints: List = []
        self.solver_config: Dict[str, Any] = {}
        self.amplitudes: Dict[str, Amplitude] = {}
        self.dload_configs: List[dict] = []
        self.contact_pairs: List[dict] = []
        self.contact_surfaces: List[dict] = []
        self.contact_pair_objects: List[ContactPair] = []

    def __repr__(self) -> str:
        n_mat = len(self.materials)
        n_con = len(self.constraints)
        n_amp = len(self.amplitudes)
        n_dl = len(self.dload_configs)
        n_cp = len(self.contact_pairs)
        n_cs = len(self.contact_surfaces)
        n_cpo = len(self.contact_pair_objects)
        return (
            f"ModelBuilderResult("
            f"mesh={'yes' if self.mesh else 'no'}, "
            f"materials={n_mat}, constraints={n_con}, "
            f"amplitudes={n_amp}, dloads={n_dl}, "
            f"contact_pairs={n_cp}, surfaces={n_cs}, "
            f"contact_objects={n_cpo})"
        )


# ──────────────────────────────────────────────
# Abaqus element type → dispsolver type
# ──────────────────────────────────────────────

_ABAQUS_TO_DISPSOLVER_ELEM = {
    "CPE4": "QUAD4",
    "CPE4R": "QUAD4",
    "CPE3": "TRIA3",
    "CPS4": "QUAD4",   # plane stress (handled as QUAD4 for mesh)
    "CPS3": "TRIA3",
}

_UNSUPPORTED_3D_ELEMENTS = {"C3D8", "C3D20", "C3D8R", "C3D20R", "C3D4", "C3D10"}

# Face number → node index mapping for Q4 element (Abaqus convention)
# Face 1: nodes 1-2 (edge 0), Face 2: nodes 2-3 (edge 1),
# Face 3: nodes 3-4 (edge 2), Face 4: nodes 4-1 (edge 3)
_Q4_FACE_NODES = {
    1: (0, 1),
    2: (1, 2),
    3: (2, 3),
    4: (3, 0),
    "P1": (0, 1),
    "P2": (1, 2),
    "P3": (2, 3),
    "P4": (3, 0),
    "S1": (0, 1),  # SPOS/S
    "S2": (1, 2),
    "S3": (2, 3),
    "S4": (3, 0),
}

# T3 face → node indices
_T3_FACE_NODES = {
    1: (0, 1),
    2: (1, 2),
    3: (2, 0),
    "P1": (0, 1),
    "P2": (1, 2),
    "P3": (2, 0),
}


# ──────────────────────────────────────────────
# ModelBuilder
# ──────────────────────────────────────────────

class ModelBuilder:
    """Convert AbaqusModel into dispsolver-native objects.

    Usage
    -----
    builder = ModelBuilder(abaqus_model)
    result = builder.build()
    mesh = result.mesh
    """

    def __init__(self, abq_model: AbaqusModel):
        self._abq = abq_model
        self._result = ModelBuilderResult()
        self._pid_counter: int = 0  # auto PID for sections without explicit section def

    def build(self) -> ModelBuilderResult:
        """Run all build phases and return the result."""
        self._flatten_parts()
        self._build_mesh()
        self._build_sets()
        self._build_sections()
        self._build_materials()
        self._build_amplitudes()
        self._build_constraints()
        self._build_dloads()
        self._build_contact()
        self._build_solver_config()
        return self._result

    # ------------------------------------------------------------------
    # Phase 0: Part/Assembly flattening
    # ------------------------------------------------------------------

    def _flatten_parts(self):
        """Flatten *PART + *ASSEMBLY + *INSTANCE into single flat model.

        Flat models (no PART/ASSEMBLY) pass through unchanged.
        """
        if not self._abq.parts:
            return  # already flat

        # We'll collect flattened nodes/elements/sets here.
        flat_nodes: Dict[int, AbaqusNode] = {}
        flat_elements: List[AbaqusElement] = []
        flat_nsets: Dict[str, set[int]] = {}
        flat_elsets: Dict[str, set[int]] = {}
        flat_materials: Dict[str, AbaqusMaterial] = {}
        next_node_id = max((n.id for n in self._abq.nodes), default=0) + 1
        next_elem_id = max((e.eid for e in self._abq.elements), default=0) + 1

        # Copy existing flat nodes/elements (outside PART/ASSEMBLY)
        for n in self._abq.nodes:
            flat_nodes[n.id] = n
        for e in self._abq.elements:
            flat_elements.append(e)

        def _apply_transform(nodes: Dict[int, AbaqusNode],
                             dx: float, dy: float,
                             angle_deg: float) -> Dict[int, AbaqusNode]:
            """Apply TRANSLATION and/or ROTATION to a node set."""
            if abs(angle_deg) > 1e-12:
                angle_rad = np.deg2rad(angle_deg)
                c, s = np.cos(angle_rad), np.sin(angle_rad)
                transformed: Dict[int, AbaqusNode] = {}
                for nid, n in nodes.items():
                    xn = n.x * c - n.y * s + dx
                    yn = n.x * s + n.y * c + dy
                    transformed[nid] = AbaqusNode(id=nid, x=xn, y=yn)
                return transformed
            else:
                transformed = {}
                for nid, n in nodes.items():
                    transformed[nid] = AbaqusNode(id=nid, x=n.x + dx, y=n.y + dy)
                return transformed

        # Process each assembly
        for asm_name, asm_data in self._abq.assemblies.items():
            for inst_name, inst_data in asm_data.get("instances", {}).items():
                part_name = inst_data.get("part", "")
                part_data = self._abq.parts.get(part_name)
                if part_data is None:
                    warnings.warn(f"Part '{part_name}' referenced by instance '{inst_name}' not found")
                    continue

                dx = inst_data.get("translation", (0.0, 0.0))[0]
                dy = inst_data.get("translation", (0.0, 0.0))[1]
                angle = inst_data.get("rotation", 0.0)

                # Get part nodes/elements/sets
                part_nodes = part_data.get("nodes", {})
                part_elems = part_data.get("elements", [])

                # Apply transform
                xformed = _apply_transform(part_nodes, dx, dy, angle)

                # Merge with prefix IDs (re-map to avoid collisions)
                prefix = f"{inst_name}_"
                nid_map: Dict[int, int] = {}
                for nid, node in xformed.items():
                    new_nid = next_node_id
                    next_node_id += 1
                    nid_map[nid] = new_nid
                    flat_nodes[new_nid] = AbaqusNode(id=new_nid, x=node.x, y=node.y)

                for elem in part_elems:
                    new_eid = next_elem_id
                    next_elem_id += 1
                    new_node_ids = [nid_map.get(n, n) for n in elem.node_ids]
                    new_elem = AbaqusElement(eid=new_eid, node_ids=new_node_ids,
                                             etype=elem.etype)
                    flat_elements.append(new_elem)

                # Merge part sets (prefixed)
                for ps_name, ps_ids in part_data.get("nsets", {}).items():
                    flat_name = f"{prefix}{ps_name}"
                    if flat_name not in flat_nsets:
                        flat_nsets[flat_name] = set()
                    flat_nsets[flat_name].update(nid_map.get(n, n) for n in ps_ids)
                for ps_name, ps_ids in part_data.get("elsets", {}).items():
                    flat_name = f"{prefix}{ps_name}"
                    if flat_name not in flat_elsets:
                        flat_elsets[flat_name] = set()
                    # Element IDs may shift; we need an element ID map
                    # Since element IDs are re-assigned, we cannot easily map
                    # old elset IDs. This is best-effort.
                    # TODO: persistent element ID mapping for instance elsets
                    pass

        # If we have instances, also process assembly-level sets
        for asm_data in self._abq.assemblies.values():
            for s_name, s_ids in asm_data.get("nsets", {}).items():
                if s_name not in flat_nsets:
                    flat_nsets[s_name] = set(s_ids)
                else:
                    flat_nsets[s_name].update(s_ids)

        # Replace model data with flattened data
        if flat_nodes:
            self._abq.nodes = list(flat_nodes.values())
            self._abq.elements = flat_elements
            # Merge into model's nsets/elsets
            for s_name, s_ids in flat_nsets.items():
                existing = self._abq.nsets.get(s_name, set())
                existing.update(s_ids)
                self._abq.nsets[s_name] = existing
            for s_name, s_ids in flat_elsets.items():
                existing = self._abq.elsets.get(s_name, set())
                existing.update(s_ids)
                self._abq.elsets[s_name] = existing

    # ------------------------------------------------------------------
    # Phase 1: Mesh construction
    # ------------------------------------------------------------------

    def _build_mesh(self):
        """Convert nodes and elements into dispsolver Mesh."""
        mesh = Mesh()

        # Nodes
        for abq_node in self._abq.nodes:
            mesh.add_node(abq_node.id, abq_node.x, abq_node.y)

        # Elements — convert Abaqus type → dispsolver type
        for abq_elem in self._abq.elements:
            etype_upper = abq_elem.etype.upper()

            # Reject 3D elements
            if etype_upper in _UNSUPPORTED_3D_ELEMENTS:
                raise NotImplementedError(
                    f"3D element {abq_elem.etype} is not supported "
                    f"(2D solver only). Use CPE4/CPE3/CPE4R instead."
                )

            disp_type = _ABAQUS_TO_DISPSOLVER_ELEM.get(etype_upper)
            if disp_type is None:
                warnings.warn(
                    f"Unknown element type '{abq_elem.etype}' — treating as QUAD4"
                )
                disp_type = "QUAD4"

            mesh.add_element(abq_elem.eid, abq_elem.node_ids, disp_type)

        self._result.mesh = mesh

    def _build_sets(self):
        """Add node sets and element sets to the mesh."""
        mesh = self._result.mesh
        for name, ids in self._abq.nsets.items():
            mesh.add_nodeset(name, set(ids))
        for name, ids in self._abq.elsets.items():
            mesh.add_elementset(name, set(ids))

    # ------------------------------------------------------------------
    # Phase 1b: Section → PID assignment
    # ------------------------------------------------------------------

    def _build_sections(self):
        """Assign PIDs to elements from *SOLID SECTION definitions.

        Iterates sections in order; each unique material gets the next PID.
        Elements in the section's ELSET receive that PID.
        Builds self._material_name_to_pid for _build_materials to use.
        """
        self._material_name_to_pid: Dict[str, int] = {}
        if not self._abq.sections:
            return
        mesh = self._result.mesh
        pid = 0
        seen_materials: set = set()
        for sec in self._abq.sections.values():
            mat_name = sec.material
            if mat_name not in seen_materials:
                pid += 1
                seen_materials.add(mat_name)
                self._material_name_to_pid[mat_name] = pid
            else:
                pid = self._material_name_to_pid[mat_name]
            # Assign PID to elements in this section's ELSET
            elem_ids = mesh.element_sets.get(sec.elset)
            if elem_ids:
                for eid in elem_ids:
                    elem = mesh.elements.get(eid)
                    if elem is not None:
                        elem.pid = pid

    # ------------------------------------------------------------------
    # Phase 2: Material construction
    # ------------------------------------------------------------------

    def _build_materials(self):
        """Convert Abaqus material definitions into dispsolver material models.

        Uses lazy imports to avoid circular dependencies.
        Handles:
        - Legacy flat materials (mat_type="ELASTIC", "PLASTIC", ...)
        - COMPOSITE materials from *MATERIAL scoping
        """
        from ..material import NeoHookean, Yeoh, ArrudaBoyce
        from ..material import J2Plasticity, ViscoelasticMaterial

        # Collect material configurations; PID assignment happens in the loop.
        pid = 0
        for mat_name, abq_mat in self._abq.materials.items():

            # ── COMPOSITE: accumulated from *MATERIAL + sub-keywords ──
            if abq_mat.mat_type == "COMPOSITE":
                has_hyper = abq_mat.hyperelastic is not None
                has_elastic = abq_mat.elastic is not None
                has_plastic = abq_mat.plastic_table is not None
                has_visco = abq_mat.viscoelastic_prony is not None

                if has_hyper:
                    hyper = abq_mat.hyperelastic
                    model_type = hyper.get("model", "").upper()
                    if "NEO HOOKE" in model_type:
                        C10 = hyper.get("C10", 0.5)
                        D1 = hyper.get("D1", 0.02)
                        mu = 2.0 * C10
                        K = 2.0 / D1 if D1 > 0 else 100.0
                        lam = K - 2.0 * mu / 3.0
                        mat = NeoHookean()
                        mat_params = {"mu": mu, "lambda": lam}
                    elif "YEOH" in model_type:
                        C10 = hyper.get("C10", 0.5)
                        C20 = hyper.get("C20", 0.0)
                        C30 = hyper.get("C30", 0.0)
                        mat = Yeoh()
                        mat_params = {"C10": C10, "C20": C20, "C30": C30}
                    elif "ARRUDA-BOYCE" in model_type or "ARRUDA BOYCE" in model_type:
                        mu = hyper.get("mu", 1.0)
                        lambda_m = hyper.get("lambda_m", 3.0)
                        K_val = hyper.get("K", 100.0)
                        mat = ArrudaBoyce()
                        mat_params = {"mu": mu, "lambda_m": lambda_m, "K": K_val}
                    else:
                        warnings.warn(f"Unknown hyperelastic model '{model_type}' — using NeoHookean")
                        mat = NeoHookean()
                        mat_params = {"E": 1000.0, "nu": 0.3}

                elif has_elastic and has_plastic:
                    E, nu = abq_mat.elastic
                    ht = abq_mat.plastic_table
                    sigma_y0 = ht[0][0]
                    if len(ht) >= 2:
                        delta_sigma = ht[-1][0] - ht[0][0]
                        delta_eps = ht[-1][1] - ht[0][1]
                        H = delta_sigma / max(delta_eps, 1e-30)
                    else:
                        H = 0.0
                    mat = J2Plasticity(E=E, nu=nu, sigma_y0=sigma_y0, H=H)
                    mat_params = {"E": E, "nu": nu, "sigma_y0": sigma_y0, "H": H}

                elif has_elastic:
                    E, nu = abq_mat.elastic
                    mat = NeoHookean()
                    mat_params = {"E": E, "nu": nu}

                else:
                    warnings.warn(f"Material '{mat_name}' has no physical properties — skipping")
                    continue

                # Wrap in ViscoelasticMaterial if viscoelastic data present
                if has_visco:
                    gi_list = [p[0] for p in abq_mat.viscoelastic_prony]
                    tau_list = [p[2] for p in abq_mat.viscoelastic_prony]
                    wlf_params = abq_mat.trs
                    mat = ViscoelasticMaterial(
                        mat, gi_list, tau_list, wlf_params=wlf_params,
                    )
                    mat_params = {
                        "base": mat, "prony": abq_mat.viscoelastic_prony,
                        "wlf": wlf_params,
                    }

                # Copy density to model-level field for _build_solver_config
                if abq_mat.density is not None:
                    self._abq.density = abq_mat.density

                # PID from sections, or fallback sequential
                sec_pid = self._material_name_to_pid.get(mat_name) if hasattr(self, '_material_name_to_pid') else None
                if sec_pid is not None:
                    pid = sec_pid
                else:
                    pid += 1
                self._result.materials[pid] = mat
                self._result.material_params[pid] = mat_params
                continue  # ← COMPOSITE done

            # ── Legacy flat materials ──
            pid += 1
            mat_type = abq_mat.mat_type.upper()

            if mat_type == "ELASTIC":
                # *ELASTIC: E, nu — params passed as material_params dict
                E = abq_mat.params.get("E", 1000.0)
                nu = abq_mat.params.get("nu", 0.3)
                mat = NeoHookean()
                mat_params = {"E": E, "nu": nu}
                self._result.materials[pid] = mat
                self._result.material_params[pid] = mat_params

            elif mat_type == "HYPERELASTIC":
                model_type = abq_mat.params.get("model", "").upper()
                if "NEO HOOKE" in model_type:
                    C10 = abq_mat.params.get("C10", 0.5)
                    D1 = abq_mat.params.get("D1", 0.02)
                    mu = 2.0 * C10
                    K = 2.0 / D1 if D1 > 0 else 100.0
                    lam = K - 2.0 * mu / 3.0
                    mat = NeoHookean()
                    mat_params = {"mu": mu, "lambda": lam}
                elif "YEOH" in model_type:
                    C10 = abq_mat.params.get("C10", 0.5)
                    C20 = abq_mat.params.get("C20", 0.0)
                    C30 = abq_mat.params.get("C30", 0.0)
                    mat = Yeoh()
                    mat_params = {"C10": C10, "C20": C20, "C30": C30}
                elif "ARRUDA-BOYCE" in model_type or "ARRUDA BOYCE" in model_type:
                    mu = abq_mat.params.get("mu", 1.0)
                    lambda_m = abq_mat.params.get("lambda_m", 3.0)
                    K = abq_mat.params.get("K", 100.0)
                    mat = ArrudaBoyce()
                    mat_params = {"mu": mu, "lambda_m": lambda_m, "K": K}
                else:
                    warnings.warn(f"Unknown hyperelastic model '{model_type}' — using NeoHookean")
                    mat = NeoHookean()
                    mat_params = {"E": 1000.0, "nu": 0.3}
                self._result.materials[pid] = mat
                self._result.material_params[pid] = mat_params

            elif mat_type == "PLASTIC":
                E = abq_mat.params.get("E", 1000.0)
                nu = abq_mat.params.get("nu", 0.3)
                sigma_y0 = abq_mat.params.get("sigma_y0", 100.0)
                hardening_table = abq_mat.params.get("hardening_table")
                if hardening_table is not None:
                    ht = np.array(hardening_table, dtype=np.float64)
                    mat = J2Plasticity(E=E, nu=nu, sigma_y0=sigma_y0,
                                       hardening_table=ht)
                    mat_params = {"E": E, "nu": nu, "sigma_y0": sigma_y0,
                                  "hardening_table": ht.tolist()}
                else:
                    H = abq_mat.params.get("H", 0.0)
                    mat = J2Plasticity(E=E, nu=nu, sigma_y0=sigma_y0, H=H)
                    mat_params = {"E": E, "nu": nu, "sigma_y0": sigma_y0, "H": H}
                self._result.materials[pid] = mat
                self._result.material_params[pid] = mat_params

            elif mat_type == "VISCOELASTIC":
                # Base material must be defined first in the .inp
                base_mat_key = mat_name  # VISCO uses same name as base
                base_mat = self._result.materials.get(pid)
                if base_mat is None:
                    # Fallback: use elastic base
                    warnings.warn("Viscoelastic material without preceding elastic — "
                                  "creating default NeoHookean base")
                    base_mat = NeoHookean(K=100.0, mu=10.0)
                prony = abq_mat.params.get("prony", [])
                gi_list = [p[0] for p in prony]
                tau_list = [p[1] for p in prony]
                # Check for WLF
                wlf_params = abq_mat.params.get("wlf")
                mat = ViscoelasticMaterial(
                    base_mat, gi_list, tau_list,
                    wlf_params=wlf_params,
                )
                mat_params = {"base": base_mat, "prony": prony,
                              "wlf": wlf_params}
                # VISCO replaces the pid entry
                self._result.materials[pid] = mat
                self._result.material_params[pid] = mat_params

            elif mat_type == "DENSITY":
                # Density is stored in solver_config via self._abq.density
                # in _build_solver_config; nothing to build here.
                pass

            else:
                warnings.warn(f"Unknown material type '{mat_type}' — skipping")
                self._result.material_params[pid] = {}

    # ------------------------------------------------------------------
    # Phase 3: Amplitude conversion
    # ------------------------------------------------------------------

    def _build_amplitudes(self):
        """Convert AbaqusAmplitude → dispsolver Amplitude."""
        for amp_name, abq_amp in self._abq.amplitudes.items():
            # Determine amplitude type
            smooth = abq_amp.smooth
            time_type = abq_amp.time_type  # "STEP" or "TIME"
            data = abq_amp.data  # list of (t, a) tuples

            if smooth:
                # Use smooth step interpolation
                amp = Amplitude(name=amp_name, smooth=smooth,
                                time_type=time_type, data=data)
            else:
                amp = Amplitude(name=amp_name, time_type=time_type, data=data)
            self._result.amplitudes[amp_name] = amp

    # ------------------------------------------------------------------
    # Phase 4: Constraints
    # ------------------------------------------------------------------

    def _build_constraints(self):
        """Convert TIE and MPC into solver constraint objects."""
        mesh = self._result.mesh

        # TIE constraints
        for tie in self._abq.ties:
            try:
                from ..constraint.tie import TieConstraint
            except ImportError:
                warnings.warn("TieConstraint not available — skipping TIE")
                continue

            slave_nodes = set()
            for s_name in tie.slave_surfaces:
                ns = mesh.node_sets.get(s_name)
                if ns:
                    slave_nodes.update(ns.node_ids)
            master_nodes = set()
            for m_name in tie.master_surfaces:
                ns = mesh.node_sets.get(m_name)
                if ns:
                    master_nodes.update(ns.node_ids)

            if slave_nodes and master_nodes:
                constraint = TieConstraint(
                    slave_nodes=list(slave_nodes),
                    master_nodes=list(master_nodes),
                    position_tolerance=tie.position_tolerance,
                )
                self._result.constraints.append(constraint)

        # MPC constraints
        for mpc in self._abq.mpcs:
            if mpc.mpc_type.upper() in ("RBE2", "BEAM"):
                try:
                    from ..constraint.rbe2 import RBE2HingeConstraint
                except ImportError:
                    warnings.warn("RBE2HingeConstraint not available — skipping MPC")
                    continue
                # Abaqus convention: first node = master, rest = slaves
                if len(mpc.nodes) >= 2:
                    constraint = RBE2HingeConstraint(
                        master_node=mpc.nodes[0],
                        slave_nodes=list(mpc.nodes[1:]),
                    )
                    self._result.constraints.append(constraint)

        # Rigid body constraints (from *RIGID BODY)
        for rb in self._abq.rigid_bodies:
            try:
                from ..constraint.rbe2 import RBE2HingeConstraint
            except ImportError:
                warnings.warn("RBE2HingeConstraint not available — skipping RIGID BODY")
                continue
            if len(rb.node_ids) > 0:
                ext_offset = self._next_extra_primal_offset()
                constraint = RBE2HingeConstraint(
                    mesh, master_id=rb.ref_node,
                    slave_ids=rb.node_ids,
                    extra_primal_offset=ext_offset,
                )
                self._result.constraints.append(constraint)

    # ------------------------------------------------------------------
    # Phase 5: Distributed loads
    # ------------------------------------------------------------------

    def _build_dloads(self):
        """Convert DLOAD definitions into surface traction configs."""
        for dload in self._abq.dloads:
            # dload.face is a string like "P3", face number 3, etc.
            config = {
                "elset": dload.elset,
                "face": str(dload.face),
                "pressure": float(dload.magnitude),
                "amplitude_name": getattr(dload, "amplitude", None),
            }
            self._result.dload_configs.append(config)

    # ------------------------------------------------------------------
    # Phase 6: Contact surfaces & pairs
    # ------------------------------------------------------------------

    def _build_contact(self):
        """Convert contact surfaces and pairs into solver configs and objects."""
        # Build a mapping from surface name to ContactSurface objects
        surface_map: Dict[str, ContactSurface] = {}
        mesh = self._result.mesh

        # Contact surfaces: create ContactSurface objects via auto_detect_exterior
        for surf_name, abq_surf in self._abq.surfaces.items():
            surf_config = {
                "name": surf_name,
                "surface_type": abq_surf.surface_type,
                "definitions": abq_surf.definitions,
            }
            self._result.contact_surfaces.append(surf_config)

            # Try to build a ContactSurface from definitions
            if abq_surf.surface_type.upper() == "ELEMENT" and mesh is not None:
                for elset_name, side in abq_surf.definitions:
                    # elset_name could be an instance name or set name
                    try:
                        cs = auto_detect_exterior(mesh, elset_name)
                        cs.name = surf_name
                        surface_map[surf_name] = cs
                    except Exception:
                        warnings.warn(f"Could not auto-detect exterior for surface {surf_name} "
                                       f"using set {elset_name}")

        # Contact pairs: create ContactPair solver objects
        for cp in self._abq.contact_pairs:
            # Look up contact property
            eps = 1e6  # default penalty stiffness
            mu = 0.0   # default friction coefficient
            if cp.interaction and cp.interaction in self._abq.contact_properties:
                prop = self._abq.contact_properties[cp.interaction]
                eps = prop.penalty_stiffness
                mu = prop.friction_coeff

            cp_config = {
                "name": cp.name if cp.name else "",
                "slave_surface": cp.slave_surface,
                "master_surface": cp.master_surface,
                "penalty_stiffness": eps,
                "friction_coeff": mu,
                "interaction": cp.interaction,
            }
            self._result.contact_pairs.append(cp_config)

            # Create ContactPair object if both surfaces are available and mesh exists
            if mesh is not None:
                slave_surf = surface_map.get(cp.slave_surface)
                master_surf = surface_map.get(cp.master_surface)
                if slave_surf is not None and master_surf is not None:
                    cp_obj = ContactPair(slave_surf, master_surf, mesh, eps=eps)
                    self._result.contact_pair_objects.append(cp_obj)

    # ------------------------------------------------------------------
    # Phase 7: Solver configuration
    # ------------------------------------------------------------------

    def _build_solver_config(self):
        """Extract solver configuration from AbaqusModel."""
        config: Dict[str, Any] = {
            "density": self._abq.density if self._abq.density is not None else 1e-9,
            "max_iter": 20,
            "tol": 1e-8,
            "static_mode": False,
            "alpha": 0.0,
            "amplitude_map": {},
            "bc_amp_map": {},
            "load_amp_map": {},
        }

        # Step parameters
        if self._abq.steps:
            step = self._abq.steps[0]
            config["procedure"] = step.procedure
            tp = step.bounds if step.bounds else {}

            if step.procedure.upper() == "STATIC":
                config["static_mode"] = True
                # *STATIC: dt_init, t_total, dt_min, dt_max
                config["dt_init"] = tp.get("dt_init", 0.1)
                config["t_total"] = tp.get("t_total", 1.0)
                config["dt_min"] = tp.get("dt_min", 1e-10)
                config["dt_max"] = tp.get("dt_max", config["t_total"])
            else:
                # *DYNAMIC (default)
                config["dt_init"] = tp.get("dt_init", 0.01)
                config["t_total"] = tp.get("t_total", 1.0)
                config["dt_min"] = tp.get("dt_min", 1e-10)
                config["dt_max"] = tp.get("dt_max", config["t_total"])
                # HHT-alpha
                config["alpha"] = tp.get("alpha", 0.0)

        # Amplitude mappings for BC and loads
        # (set by parser if AMPLITUDE= keyword param was used)
        for bc in self._abq.boundaries:
            if bc.amplitude:
                config["bc_amp_map"][bc.amplitude] = bc.amplitude
        for load in self._abq.loads:
            if load.amplitude:
                config["load_amp_map"][load.amplitude] = load.amplitude

        self._result.solver_config = config

    # ------------------------------------------------------------------
    # Helper: DOF conversion (Abaqus 1-based → 0-based)
    # ------------------------------------------------------------------

    @staticmethod
    def dof_abaqus_to_global(dof_abaqus: int) -> int:
        """Convert Abaqus 1-based DOF to 0-based local DOF.

        Abaqus DOF1=ux → 0, DOF2=uy → 1.
        """
        return dof_abaqus - 1

    def _next_extra_primal_offset(self) -> int:
        """Return the cumulative extra primal DOF offset for the next constraint."""
        return sum(c.n_extra_primal() for c in self._result.constraints)

    @staticmethod
    def compute_bc_dofs(
        mesh: Mesh,
        node_ids: list[int],
        dofs: list[int],
    ) -> np.ndarray:
        """Compute global DOF indices for Abaqus boundary conditions.

        Abaqus DOF1 (1) → local DOF 0 → global: node_idx * 2 + 0
        Abaqus DOF2 (2) → local DOF 1 → global: node_idx * 2 + 1

        Parameters
        ----------
        mesh : Mesh
        node_ids : list of int
            Node IDs to constrain.
        dofs : list of int
            Abaqus DOF numbers (1-based) to constrain.

        Returns
        -------
        bc_dofs : np.ndarray, shape (n_bc,)
        """
        nid_to_idx = mesh.node_id_to_index()
        bc_list = []
        for nid in node_ids:
            idx = nid_to_idx[nid]
            base = idx * 2
            for d in dofs:
                local_dof = ModelBuilder.dof_abaqus_to_global(d)
                bc_list.append(base + local_dof)
        return np.array(bc_list, dtype=np.int32)
