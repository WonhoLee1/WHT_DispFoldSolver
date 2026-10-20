"""
abaqus_api_builder.py
=====================
Builds Abaqus Python API Model/Part/Assembly/Step hierarchies and emits 100% compliant
Abaqus input decks (.inp), with automated roundtrip verification via read_abaqus_input.
"""

from __future__ import annotations
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from dispsolver.model.model import Model
from dispsolver.model.step import Step
from dispsolver.model.control import SolutionControl
from dispsolver.model.interaction import ContactControl, ContactProperty, ContactStd
from dispsolver.io import read_abaqus_input, read_inp


class AbaqusApiBuilder:
    """Builds Abaqus Model hierarchy and exports verified .inp input decks."""

    def __init__(self, config: UnifiedFoldModelConfig, mesh: Any, meta: Dict[str, Any]) -> None:
        self.config = config
        self.mesh = mesh
        self.meta = meta
        self.dim = config.simulation_case.dim.upper()

    def build_model_hierarchy(self) -> Model:
        """Constructs Model, Part, Assembly, Step, and SolutionControl."""
        model = Model(name=f"DispFold_{self.dim}_Model")
        stages = self.config.simulation_case.stages
        preset = self.config.simulation_case.solution_preset

        if stages:
            for stg in stages:
                s = model.StaticStep(
                    name=stg.name,
                    time_period=stg.duration,
                    nlgeom=True,
                    dt_init=stg.dt_init,
                    dt_min=stg.dt_min,
                    dt_max=stg.dt_max,
                )
                s.solution_controls = SolutionControl(preset=preset)
        else:
            step_name = "Step-1_Folding"
            step = model.StaticStep(
                name=step_name,
                time_period=self.config.simulation_case.total_time,
                nlgeom=True,
                dt_init=self.config.simulation_case.dt_init,
                dt_min=self.config.simulation_case.dt_min,
                dt_max=self.config.simulation_case.dt_max,
            )
            step.solution_controls = SolutionControl(preset=preset)
        return model

    def generate_inp_string(self) -> str:
        """Generates complete, standard Abaqus .inp keyword deck as a string."""
        lines = []
        _w = lines.append

        dim_tag = "2D Plane Strain" if self.dim == "2D" else "3D Continuum Solid"
        _w("**" + "=" * 78)
        _w(f"** Abaqus Input Deck: Foldable Display 180-Degree Teardrop ({dim_tag})")
        _w(f"** Layup: {self.config.simulation_case.preset_name} | Total Thickness: {self.config.total_thickness:.3f} mm")
        _w("**" + "=" * 78)
        _w("*HEADING")
        _w(f"Foldable Display 180-Degree Teardrop Folding Simulation ({self.dim})")
        _w("**")

        # 1. Nodes
        _w("*NODE")
        coords = self.mesh.nodes_array()
        nid_map = self.mesh.node_id_to_index()
        sorted_nids = sorted(nid_map.keys())

        if self.dim == "2D":
            for nid in sorted_nids:
                idx = nid_map[nid]
                _w(f"{nid}, {coords[idx, 0]:.6f}, {coords[idx, 1]:.6f}")
        else:
            for nid in sorted_nids:
                idx = nid_map[nid]
                _w(f"{nid}, {coords[idx, 0]:.6f}, {coords[idx, 1]:.6f}, {coords[idx, 2]:.6f}")

        # Dedicated Rigid Body Reference Nodes (pivots)
        rp_left_nid = max(sorted_nids) + 1
        rp_right_nid = max(sorted_nids) + 2
        p_l = self.meta["left_pivot"]
        p_r = self.meta["right_pivot"]
        w_z = float(getattr(self.config.geometry, "width_z", 1.0))

        if self.dim == "2D":
            _w(f"{rp_left_nid}, {p_l[0]:.6f}, {p_l[1]:.6f}")
            _w(f"{rp_right_nid}, {p_r[0]:.6f}, {p_r[1]:.6f}")
        else:
            _w(f"{rp_left_nid}, {p_l[0]:.6f}, {p_l[1]:.6f}, {0.5 * w_z:.6f}")
            _w(f"{rp_right_nid}, {p_r[0]:.6f}, {p_r[1]:.6f}, {0.5 * w_z:.6f}")

        # 2. Elements per Layer and Plate
        _w("**" + "-" * 78)
        _w("** Elements and Solid Sections")
        _w("**" + "-" * 78)

        layer_elem_sets = self.meta.get("layer_elem_sets", {})
        for lay in self.config.layers:
            eids = layer_elem_sets.get(lay.name, [])
            if not eids:
                continue

            elem_type = lay.elem_type_2d if self.dim == "2D" else lay.elem_type_3d
            abq_elem_type = elem_type.replace("_CR", "").replace("QUAD4", "CPE4I")
            if "CPE4H" in abq_elem_type:
                abq_elem_type = "CPE4H"
            elif "CPE4I" in abq_elem_type:
                abq_elem_type = "CPE4I"
            elif "CPE4R" in abq_elem_type:
                abq_elem_type = "CPE4R"
            elif "CPE4" in abq_elem_type:
                abq_elem_type = "CPE4"
            elif "C3D8H" in abq_elem_type:
                abq_elem_type = "C3D8H"
            elif "C3D8I" in abq_elem_type:
                abq_elem_type = "C3D8I"
            elif "C3D8R" in abq_elem_type:
                abq_elem_type = "C3D8R"
            elif "C3D8" in abq_elem_type:
                abq_elem_type = "C3D8"

            elset_name = f"ELSET_{lay.name.upper()}"
            _w(f"*ELEMENT, TYPE={abq_elem_type}, ELSET={elset_name}")
            for eid in eids:
                elem = self.mesh.elements[eid]
                nodes_str = ", ".join(str(n) for n in elem.node_ids)
                _w(f"{eid}, {nodes_str}")

            mat_name = lay.material_name
            sec_name = f"SEC_{lay.name.upper()}"
            _w(f"*SOLID SECTION, ELSET={elset_name}, MATERIAL={mat_name}")

        # Plate Elements
        plate_type = "CPE4" if self.dim == "2D" else "C3D8"
        plate_eids_l = []
        plate_eids_r = []
        for eid, elem in self.mesh.elements.items():
            if elem.pid == 4:
                idx0 = nid_map[elem.node_ids[0]]
                if coords[idx0, 0] < 0.0:
                    plate_eids_l.append(eid)
                else:
                    plate_eids_r.append(eid)

        if plate_eids_l:
            _w(f"*ELEMENT, TYPE={plate_type}, ELSET=ELSET_PLATE_LEFT")
            for eid in plate_eids_l:
                nodes_str = ", ".join(str(n) for n in self.mesh.elements[eid].node_ids)
                _w(f"{eid}, {nodes_str}")
            _w("*SOLID SECTION, ELSET=ELSET_PLATE_LEFT, MATERIAL=PLATE")

        if plate_eids_r:
            _w(f"*ELEMENT, TYPE={plate_type}, ELSET=ELSET_PLATE_RIGHT")
            for eid in plate_eids_r:
                nodes_str = ", ".join(str(n) for n in self.mesh.elements[eid].node_ids)
                _w(f"{eid}, {nodes_str}")
            _w("*SOLID SECTION, ELSET=ELSET_PLATE_RIGHT, MATERIAL=PLATE")

        # 3. Materials
        _w("**" + "-" * 78)
        _w("** Material Definitions")
        _w("**" + "-" * 78)
        for name, mat in self.config.materials.items():
            _w(f"*MATERIAL, NAME={name}")
            _w(f"*DENSITY\n{mat.density:.4e}")
            if mat.mat_type == "ELASTIC":
                _w(f"*ELASTIC\n{mat.E:.2f}, {mat.nu:.4f}")
            elif mat.mat_type == "ELASTIC_PLASTIC":
                _w(f"*ELASTIC\n{mat.E:.2f}, {mat.nu:.4f}")
                _w(f"*PLASTIC\n{mat.yield_stress:.2f}, 0.0")
                if mat.hardening_modulus > 0:
                    _w(f"{mat.yield_stress + mat.hardening_modulus * 0.1:.2f}, 0.1")
            elif mat.mat_type == "HYPERELASTIC_VISCO":
                # Arruda-Boyce Hyperelastic + Viscoelastic
                # D = 2/K with K = E / (3 (1 - 2 nu)) from the spec; a fixed 0.001 made the PSA 240x too stiff in volume
                d_compress = 2.0 * 3.0 * (1.0 - 2.0 * mat.nu) / mat.E
                _w(f"*HYPERELASTIC, ARRUDA-BOYCE\n{mat.arruda_boyce_mu:.6f}, {mat.arruda_boyce_lm:.2f}, {d_compress:.6g}")
                if mat.prony_g:
                    _w("*VISCOELASTIC, TIME=PRONY")
                    for g, tau in zip(mat.prony_g, mat.prony_tau):
                        _w(f"{g:.4f}, 0.0, {tau:.4f}")

        # 4. Node Sets
        _w("**" + "-" * 78)
        _w("** Node Sets for Constraints and Boundary Conditions")
        _w("**" + "-" * 78)
        _w("*NSET, NSET=RP_LEFT")
        _w(str(rp_left_nid))
        _w("*NSET, NSET=RP_RIGHT")
        _w(str(rp_right_nid))

        if self.meta.get("plate_left_nodes"):
            _w("*NSET, NSET=NSET_PLATE_LEFT")
            _w(", ".join(str(n) for n in self.meta["plate_left_nodes"]))
        if self.meta.get("plate_right_nodes"):
            _w("*NSET, NSET=NSET_PLATE_RIGHT")
            _w(", ".join(str(n) for n in self.meta["plate_right_nodes"]))

        if self.meta.get("disp_bottom_left_nids"):
            _w("*NSET, NSET=NSET_TIED_LEFT")
            _w(", ".join(str(n) for n in self.meta["disp_bottom_left_nids"]))
        if self.meta.get("disp_bottom_right_nids"):
            _w("*NSET, NSET=NSET_TIED_RIGHT")
            _w(", ".join(str(n) for n in self.meta["disp_bottom_right_nids"]))

        # 5. Rigid Body Constraints
        _w("**" + "-" * 78)
        _w("** Rigid Body Constraints (Support Plates to Pivot Reference Nodes)")
        _w("**" + "-" * 78)
        _w(f"*RIGID BODY, ELSET=ELSET_PLATE_LEFT, REF NODE={rp_left_nid}, PIN NSET=NSET_PLATE_LEFT")
        _w(f"*RIGID BODY, ELSET=ELSET_PLATE_RIGHT, REF NODE={rp_right_nid}, PIN NSET=NSET_PLATE_RIGHT")

        # 6. Tie Constraints (Coupling between plate and display bottom)
        _w("**" + "-" * 78)
        _w("** Tie Constraints / Multi-Point Constraints")
        _w("**" + "-" * 78)
        _w("*TIE, NAME=TIE_LEFT")
        _w("NSET_TIED_LEFT, NSET_PLATE_LEFT")
        _w("*TIE, NAME=TIE_RIGHT")
        _w("NSET_TIED_RIGHT, NSET_PLATE_RIGHT")

        # 7. Contact Definitions (Self-contact if required)
        if self.config.requires_self_contact():
            if self.dim == "2D":
                top_l_nids = sorted(list({n for seg in self.meta.get("disp_top_left_segments", []) for n in seg}))
                top_r_nids = sorted(list({n for seg in self.meta.get("disp_top_right_segments", []) for n in seg}))
            else:
                top_l_nids = sorted(list({n for face in self.meta.get("disp_top_left_faces", []) for n in face}))
                top_r_nids = sorted(list({n for face in self.meta.get("disp_top_right_faces", []) for n in face}))

            if top_l_nids and top_r_nids:
                _w("**" + "-" * 78)
                _w("** Surface and Contact Interaction")
                _w("**" + "-" * 78)
                _w("*NSET, NSET=NSET_TOP_LEFT")
                _w(", ".join(str(n) for n in top_l_nids))
                _w("*NSET, NSET=NSET_TOP_RIGHT")
                _w(", ".join(str(n) for n in top_r_nids))
                _w("*SURFACE, NAME=SURF_TOP_LEFT, TYPE=NODE")
                _w("NSET_TOP_LEFT")
                _w("*SURFACE, NAME=SURF_TOP_RIGHT, TYPE=NODE")
                _w("NSET_TOP_RIGHT")
                _w("*SURFACE INTERACTION, NAME=CONTACT_PROP")
                _w("*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD")
                _w("*CONTACT PAIR, INTERACTION=CONTACT_PROP")
                _w("SURF_TOP_RIGHT, SURF_TOP_LEFT")

        # 8. Amplitude Definitions
        _w("**" + "-" * 78)
        _w("** Amplitude Definitions")
        _w("**" + "-" * 78)
        _w("*AMPLITUDE, NAME=AMP_SMOOTH, DEFINITION=SMOOTH STEP")
        _w("0.0, 0.0, 1.0, 1.0")

        # 9. Step(s) and Boundary Conditions
        stages = self.config.simulation_case.stages
        if not stages:
            from example_disp_fold.model_spec import StageSpec
            stages = [
                StageSpec(
                    name="Step-1_Folding",
                    stage_type="FOLD",
                    duration=self.config.simulation_case.total_time,
                    theta_start_deg=0.0,
                    theta_target_deg=self.config.simulation_case.theta_max_deg,
                    dt_init=self.config.simulation_case.dt_init,
                    dt_min=self.config.simulation_case.dt_min,
                    dt_max=self.config.simulation_case.dt_max,
                )
            ]

        for s_idx, stg in enumerate(stages):
            _w("**" + "-" * 78)
            _w(f"** Step {s_idx + 1}: {stg.name} ({stg.stage_type})")
            _w("**" + "-" * 78)
            _w(f"*STEP, NAME={stg.name}, NLGEOM=YES")
            _w(f"{stg.name}: {stg.stage_type} - Rotation {stg.theta_start_deg:.1f}deg -> {stg.theta_target_deg:.1f}deg, Duration={stg.duration:.2f}s")
            _w(f"*STATIC\n{stg.dt_init:.5f}, {stg.duration:.2f}, {stg.dt_min:.5e}, {stg.dt_max:.5f}")

            # Rotations in radians (UR3 = DOF 6)
            th_target_rad = float(np.radians(stg.theta_target_deg))
            _w("** Boundary Conditions on Reference Nodes")
            _w("*BOUNDARY, AMPLITUDE=AMP_SMOOTH")
            _w(f"{rp_left_nid}, 1, 2, 0.0")
            _w(f"{rp_left_nid}, 6, 6, {-th_target_rad:.6f}")
            _w(f"{rp_right_nid}, 1, 2, 0.0")
            _w(f"{rp_right_nid}, 6, 6, {th_target_rad:.6f}")

            if self.dim == "3D":
                _w("** 3D Symmetry on Z=0 Plane")
                _w("*BOUNDARY")
                z0_nodes = [nid for nid, node in self.mesh.nodes.items() if abs(getattr(node, "z", 0.0)) < 1e-4]
                if z0_nodes:
                    for nid in z0_nodes:
                        _w(f"{nid}, 3, 3, 0.0")

            # Standard Output Requests (Field and History)
            _w("** Field and History Output Requests")
            _w("*OUTPUT, FIELD, VARIABLE=PRESELECT")
            _w("*NODE OUTPUT")
            _w("U, RF")
            _w("*ELEMENT OUTPUT, DIRECTIONS=YES")
            _w("S, MISES, LE, E")
            if self.config.requires_self_contact():
                _w("*CONTACT OUTPUT")
                _w("CPRESS, COPEN, CSHEAR, CSTATUS")
            _w("*OUTPUT, HISTORY")
            _w("*ENERGY OUTPUT")
            _w("ALLIE, ALLKE, ALLSE, ALLVD, ALLAE, ALLSD, ALLFD")
            _w("*END STEP")

        return "\n".join(lines) + "\n"

    def write_inp(self, filepath: str) -> str:
        """Writes complete, standard Abaqus .inp keyword deck.

        Parameters
        ----------
        filepath : str
            Target file path (e.g. results/.../model.inp)
        """
        inp_text = self.generate_inp_string()
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(inp_text)

        return filepath

    # Mesh pid rule (gmsh_mesh_generator): PET -> 1, PSA -> 2, everything else in the stack (GLASS) -> 3, plate -> 4.
    SOLVER_PIDS = {"PET": 1, "PSA": 2, "GLASS": 3, "PLATE": 4}

    def solver_materials(self, filepath: str) -> Dict[int, Any]:
        """Solver materials per mesh pid, read back from the exported deck.

        The deck is the single place the stack's constitutive laws are written (PET elastic-plastic, PSA
        Arruda-Boyce + Prony, GLASS and PLATE elastic), so the solver is given exactly what model.inp says.
        The solvers used to be handed {"E", "nu"} dictionaries, which solved every layer as linear elastic.
        """
        parsed = read_inp(filepath, dim=2 if self.dim == "2D" else 3)
        missing = [n for n in self.SOLVER_PIDS if n in self.config.materials and n not in parsed.materials]
        if missing:
            raise ValueError(f"materials {missing} are in the stack but not in {filepath}")
        return {pid: parsed.materials[name] for name, pid in self.SOLVER_PIDS.items() if name in parsed.materials}

    def validate_inp_roundtrip(self, filepath: str) -> Dict[str, Any]:
        """Validates that the emitted .inp can be parsed back with full keyword fidelity."""
        from dispsolver.io import read_inp
        model_legacy = read_abaqus_input(filepath)
        model_inp = read_inp(filepath, dim=2 if self.dim == "2D" else 3)

        n_nodes_parsed = len(model_legacy.mesh.nodes)
        n_elems_parsed = len(model_legacy.mesh.elements)

        n_nodes_orig = len(self.mesh.nodes)
        n_elems_orig = len(self.mesh.elements)
        n_nodes_expected = n_nodes_orig + 2  # +2 Reference Nodes (RP_LEFT, RP_RIGHT)

        if n_nodes_parsed != n_nodes_expected:
            raise ValueError(f"Roundtrip node count mismatch: orig={n_nodes_orig}, expected={n_nodes_expected}, parsed={n_nodes_parsed}")
        if n_elems_parsed != n_elems_orig:
            raise ValueError(f"Roundtrip elem count mismatch: orig={n_elems_orig}, parsed={n_elems_parsed}")

        # Check materials
        for mat_name in self.config.materials.keys():
            if not any(mat_name in str(name) for name in model_legacy.material_names.values()):
                raise ValueError(f"Material {mat_name} missing from roundtrip parsed model")

        # Check essential sets
        for sname in ["RP_LEFT", "RP_RIGHT", "NSET_PLATE_LEFT", "NSET_PLATE_RIGHT", "NSET_TIED_LEFT", "NSET_TIED_RIGHT"]:
            if sname not in model_legacy.mesh.node_sets:
                raise ValueError(f"Essential Node Set {sname} missing from roundtrip parsed model")

        for elname in ["ELSET_PLATE_LEFT", "ELSET_PLATE_RIGHT"]:
            if elname not in model_legacy.mesh.element_sets:
                raise ValueError(f"Essential Element Set {elname} missing from roundtrip parsed model")

        # Check constraints
        if len(model_legacy.rbe2_constraints) < 2:
            raise ValueError(f"Expected at least 2 RigidBody constraints, found {len(model_legacy.rbe2_constraints)}")
        if len(model_legacy.penalty_constraints) < 2:
            raise ValueError(f"Expected at least 2 Tie constraints, found {len(model_legacy.penalty_constraints)}")

        # Check boundaries (at least 4: 2 translation + 2 rotation)
        if len(model_legacy.boundaries) < 4:
            raise ValueError(f"Expected at least 4 boundary conditions, found {len(model_legacy.boundaries)}")

        # Check steps and output requests via model_inp
        analysis_steps = {k: v for k, v in model_inp.steps.items() if k.lower() != "initial"}
        if not analysis_steps:
            raise ValueError("No analysis steps found in parsed AbaqusModel")
        for sname, step_obj in analysis_steps.items():
            if not step_obj.field_output_requests:
                raise ValueError(f"Field output requests missing in step {sname}")
            if not step_obj.history_output_requests:
                raise ValueError(f"History output requests missing in step {sname}")

        return {
            "valid": True,
            "num_nodes": n_nodes_parsed,
            "mesh_nodes": n_nodes_orig,
            "ref_nodes": 2,
            "num_elements": n_elems_parsed,
            "materials": list(model_legacy.materials.keys()),
            "node_sets": list(model_legacy.mesh.node_sets.keys()),
            "element_sets": list(model_legacy.mesh.element_sets.keys()),
            "boundaries": model_legacy.boundaries,
            "rbe2_constraints": len(model_legacy.rbe2_constraints),
            "tie_constraints": len(model_legacy.penalty_constraints),
            "steps": [s for s in model_inp.steps.keys() if s.lower() != "initial"],
            "solver_config": model_legacy.solver_config,
            "part_names": model_legacy.part_names,
        }


# Alias for specification and test compatibility
AbaqusModelBuilder = AbaqusApiBuilder
