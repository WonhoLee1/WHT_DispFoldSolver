"""
run_fistr_teardrop.py
=====================
Builds and executes the 2D (Plane-Strain 361) and 3D (Continuum 361) Foldable Display
180-Degree Teardrop Folding models in FrontISTR (fistr1.exe), parses the resulting
displacements, stresses, and reaction moments, and generates verification plots.

Usage:
    python example_disp_fold/run_fistr_teardrop.py --dim 2D --preset compact_4
    python example_disp_fold/run_fistr_teardrop.py --dim 3D --preset compact_4
    python example_disp_fold/run_fistr_teardrop.py --dim both --preset teardown
"""

from __future__ import annotations
import os
import sys
import re
import time
import json
import math
import subprocess
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator

# FrontISTR SMP binary path
FISTR_SMP_DIR = r"D:\PythonCodeStudy\FrontISTR\FrontISTR-latest.thread_openblas"
FISTR_EXE = os.path.join(FISTR_SMP_DIR, "fistr1.exe")


class FrontISTRTeardropBuilder:
    """
    Converts UnifiedFoldModelConfig + GmshMeshGenerator (2D or 3D) into
    FrontISTR input files (hecmw_ctrl.dat, fistr_teardrop.msh, fistr_teardrop.cnt).

    Note on 2D Plane Strain in FrontISTR NLSTATIC:
    ----------------------------------------------
    FrontISTR's native 2D elements (231, 232, 241, 242) only support INFINITESIMAL
    (small strain) kinematics in fstr_CreateMatrix.f90 (line 371).
    Therefore, for 2D Plane-Strain large-rotation (180-deg folding) analysis,
    the 2D Quad4 mesh is extruded by 1 element in Z (z in [-0.5, +0.5] mm) as
    TYPE=361 hexahedra with u_z = 0 constrained on ALL nodes, which is mathematically
    100% identical to 2D Plane Strain while unlocking FrontISTR's 3D finite-strain
    element formulations (FORM361=IC / FBAR / UP) and Rodrigues rotational BCs
    (!BOUNDARY, ROT_CENTER=...).
    """

    def __init__(
        self,
        config: UnifiedFoldModelConfig,
        psa_form361: str = "FBAR",
        solid_form361: str = "IC",
        substeps: int = 90,
        solver_method: str = "MUMPS",
        mesh_spec: Optional[Any] = None,
    ) -> None:
        self.config = config
        if mesh_spec is not None:
            self.config.mesh = mesh_spec
        self.dim = config.simulation_case.dim.upper()
        self.psa_form361 = psa_form361.upper()
        self.solid_form361 = solid_form361.upper()
        self.substeps = substeps
        self.solver_method = solver_method.upper()
        self.res_write_freq = max(1, substeps // 10) if substeps >= 10 else 1

        mesher = GmshMeshGenerator(config)
        self.raw_mesh, self.raw_meta = mesher.build()

        # Prepared 3D nodes, elements, and groups for FrontISTR
        self.nodes_3d: Dict[int, Tuple[float, float, float]] = {}
        self.elements_3d: Dict[int, List[int]] = {}
        self.egrp_map: Dict[str, List[int]] = {}
        self.egrp_mat: Dict[str, str] = {}
        self.egrp_form361: Dict[str, str] = {}

        self.left_driven_nids: List[int] = []
        self.right_driven_nids: List[int] = []
        self.z_fixed_nids: List[int] = []
        self.pivot_dummy_nids: List[int] = []
        self.pivot_l_nid: int = -1
        self.pivot_r_nid: int = -1
        self.real_max_nid: int = 0
        self.real_max_eid: int = 0

        self._prepare_fistr_mesh()

    def _prepare_fistr_mesh(self) -> None:
        """Prepares 3D node/element dictionaries for either 2D (extruded nz=1) or 3D (nz>=2)."""
        w_z = float(getattr(self.config.geometry, "width_z", 1.0))
        z0, z1 = -0.5 * w_z, +0.5 * w_z

        if self.dim == "2D":
            # Extrude 2D mesh by 1 layer in Z:
            # Front plane (z = z0): original nid
            # Back plane  (z = z1): nid + n_2d_offset
            max_nid_2d = max(self.raw_mesh.nodes.keys())
            n_offset = max_nid_2d

            for nid, node in self.raw_mesh.nodes.items():
                self.nodes_3d[nid] = (float(node.x), float(node.y), z0)
                self.nodes_3d[nid + n_offset] = (float(node.x), float(node.y), z1)

            # Extrude elements
            layer_elem_sets = self.raw_meta["layer_elem_sets"]
            for lay in self.config.layers:
                egrp_name = f"EL_{lay.name.upper()}"
                eids = layer_elem_sets.get(lay.name, [])
                if not eids:
                    continue
                self.egrp_map[egrp_name] = []
                self.egrp_mat[egrp_name] = lay.material_name.upper()
                self.egrp_form361[egrp_name] = (
                    self.psa_form361 if "PSA" in lay.material_name.upper() else self.solid_form361
                )
                for eid in eids:
                    el2d = self.raw_mesh.elements[eid]
                    n1, n2, n3, n4 = el2d.node_ids
                    self.elements_3d[eid] = [
                        n1, n2, n3, n4,
                        n1 + n_offset, n2 + n_offset, n3 + n_offset, n4 + n_offset,
                    ]
                    self.egrp_map[egrp_name].append(eid)

            # Support plates (pid == 4)
            self.egrp_map["EL_PLATE"] = []
            self.egrp_mat["EL_PLATE"] = "PLATE"
            self.egrp_form361["EL_PLATE"] = self.solid_form361
            for eid, el2d in self.raw_mesh.elements.items():
                if el2d.pid == 4:
                    n1, n2, n3, n4 = el2d.node_ids
                    self.elements_3d[eid] = [
                        n1, n2, n3, n4,
                        n1 + n_offset, n2 + n_offset, n3 + n_offset, n4 + n_offset,
                    ]
                    self.egrp_map["EL_PLATE"].append(eid)

            # Driven node groups (both Z planes)
            base_l = sorted(set(self.raw_meta["plate_left_nodes"] + self.raw_meta["disp_bottom_left_nids"]))
            base_r = sorted(set(self.raw_meta["plate_right_nodes"] + self.raw_meta["disp_bottom_right_nids"]))
            self.left_driven_nids = sorted(base_l + [n + n_offset for n in base_l])
            self.right_driven_nids = sorted(base_r + [n + n_offset for n in base_r])

            # In 2D Plane Strain, ALL non-driven nodes have u_z = 0 (driven nodes already have u_z = 0 via ROT_CENTER)
            driven_set = set(self.left_driven_nids) | set(self.right_driven_nids)
            self.z_fixed_nids = sorted([n for n in self.nodes_3d.keys() if n not in driven_set])

        else:
            # 3D mesh directly from Mesh3D
            for nid, node in self.raw_mesh.nodes.items():
                self.nodes_3d[nid] = (float(node.x), float(node.y), float(node.z))

            layer_elem_sets = self.raw_meta["layer_elem_sets"]
            for lay in self.config.layers:
                egrp_name = f"EL_{lay.name.upper()}"
                eids = layer_elem_sets.get(lay.name, [])
                if not eids:
                    continue
                self.egrp_map[egrp_name] = []
                self.egrp_mat[egrp_name] = lay.material_name.upper()
                self.egrp_form361[egrp_name] = (
                    self.psa_form361 if "PSA" in lay.material_name.upper() else self.solid_form361
                )
                for eid in eids:
                    el3d = self.raw_mesh.elements[eid]
                    self.elements_3d[eid] = list(el3d.node_ids)
                    self.egrp_map[egrp_name].append(eid)

            self.egrp_map["EL_PLATE"] = []
            self.egrp_mat["EL_PLATE"] = "PLATE"
            self.egrp_form361["EL_PLATE"] = self.solid_form361
            for eid, el3d in self.raw_mesh.elements.items():
                if el3d.pid == 4:
                    self.elements_3d[eid] = list(el3d.node_ids)
                    self.egrp_map["EL_PLATE"].append(eid)

            self.left_driven_nids = sorted(
                set(self.raw_meta["plate_left_nodes"] + self.raw_meta["disp_bottom_left_nids"])
            )
            self.right_driven_nids = sorted(
                set(self.raw_meta["plate_right_nodes"] + self.raw_meta["disp_bottom_right_nids"])
            )
            driven_set = set(self.left_driven_nids) | set(self.right_driven_nids)
            # In 3D, only Z=0 symmetry plane nodes have u_z = 0
            self.z_fixed_nids = sorted([
                nid for nid, (x, y, z) in self.nodes_3d.items()
                if abs(z) < 1e-4 and nid not in driven_set
            ])

        self.real_max_nid = max(self.nodes_3d.keys())
        self.real_max_eid = max(self.elements_3d.keys())

        # Add 1 disconnected reference Hex8 element to host stationary pivot nodes PIVOT_L and PIVOT_R
        p_l = self.raw_meta["left_pivot"]   # (-1.4, 0.0)
        p_r = self.raw_meta["right_pivot"]  # (+1.4, 0.0)
        y_ref_bot = -2.0
        base_nid = self.real_max_nid
        # 8 nodes of reference hex: [-1.4, +1.4] x [-2.0, 0.0] x [z0, z1]
        ref_coords = [
            (p_l[0], y_ref_bot, z0),  # base_nid + 1
            (p_r[0], y_ref_bot, z0),  # base_nid + 2
            (p_r[0], p_r[1],    z0),  # base_nid + 3  <-- PIVOT_R
            (p_l[0], p_l[1],    z0),  # base_nid + 4  <-- PIVOT_L
            (p_l[0], y_ref_bot, z1),  # base_nid + 5
            (p_r[0], y_ref_bot, z1),  # base_nid + 6
            (p_r[0], p_r[1],    z1),  # base_nid + 7
            (p_l[0], p_l[1],    z1),  # base_nid + 8
        ]
        self.pivot_dummy_nids = []
        for idx, pt in enumerate(ref_coords, start=1):
            rnid = base_nid + idx
            self.nodes_3d[rnid] = pt
            self.pivot_dummy_nids.append(rnid)

        self.pivot_r_nid = base_nid + 3
        self.pivot_l_nid = base_nid + 4

        ref_eid = self.real_max_eid + 1
        self.elements_3d[ref_eid] = list(self.pivot_dummy_nids)
        self.egrp_map["EL_PIVOT_REF"] = [ref_eid]
        self.egrp_mat["EL_PIVOT_REF"] = "PLATE"
        self.egrp_form361["EL_PIVOT_REF"] = self.solid_form361

    @staticmethod
    def _format_id_list(ids: List[int], per_line: int = 10) -> List[str]:
        lines = []
        for i in range(0, len(ids), per_line):
            chunk = ids[i : i + per_line]
            lines.append(" " + ", ".join(str(x) for x in chunk))
        return lines

    def write_fistr_deck(self, run_dir: str) -> Dict[str, str]:
        """Writes hecmw_ctrl.dat, fistr_teardrop.msh, and fistr_teardrop.cnt into run_dir."""
        os.makedirs(run_dir, exist_ok=True)
        ctrl_path = os.path.join(run_dir, "hecmw_ctrl.dat")
        msh_path = os.path.join(run_dir, "fistr_teardrop.msh")
        cnt_path = os.path.join(run_dir, "fistr_teardrop.cnt")

        # 1. Write hecmw_ctrl.dat
        ctrl_lines = [
            "!MESH, NAME=fstrMSH, TYPE=HECMW-ENTIRE",
            "fistr_teardrop.msh",
            "!CONTROL, NAME=fstrCNT",
            "fistr_teardrop.cnt",
            "!RESULT, NAME=fstrRES, IO=OUT",
            "fistr_teardrop.res",
            "!RESULT, NAME=vis_out, IO=OUT",
            "fistr_teardrop_vis",
        ]
        with open(ctrl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ctrl_lines) + "\n")

        # 2. Write fistr_teardrop.msh
        msh_lines = []
        _m = msh_lines.append
        _m("!HEADER")
        _m(f"FrontISTR {self.dim} Foldable Display 180-deg Teardrop Folding Mesh")
        _m("!NODE")
        for nid in sorted(self.nodes_3d.keys()):
            x, y, z = self.nodes_3d[nid]
            _m(f" {nid}, {x:.8e}, {y:.8e}, {z:.8e}")

        for egrp_name, eids in self.egrp_map.items():
            if not eids:
                continue
            _m(f"!ELEMENT, TYPE=361, EGRP={egrp_name}")
            for eid in eids:
                nstr = ", ".join(str(n) for n in self.elements_3d[eid])
                _m(f" {eid}, {nstr}")

        # Node groups
        _m("!NGROUP, NGRP=PIVOT_L")
        _m(f" {self.pivot_l_nid}")
        _m("!NGROUP, NGRP=PIVOT_R")
        _m(f" {self.pivot_r_nid}")
        _m("!NGROUP, NGRP=PIVOT_DUMMY")
        msh_lines.extend(self._format_id_list(self.pivot_dummy_nids))
        _m("!NGROUP, NGRP=LEFT_DRIVEN")
        msh_lines.extend(self._format_id_list(self.left_driven_nids))
        _m("!NGROUP, NGRP=RIGHT_DRIVEN")
        msh_lines.extend(self._format_id_list(self.right_driven_nids))
        if self.z_fixed_nids:
            _m("!NGROUP, NGRP=Z_FIXED")
            msh_lines.extend(self._format_id_list(self.z_fixed_nids))

        # Materials in .msh (placeholder items, actual nonlinear properties in .cnt)
        mat_Order = ["PET", "PSA", "GLASS", "PLATE"]
        for mname in mat_Order:
            _m(f"!MATERIAL, NAME={mname}, ITEM=1")
            _m("!ITEM=1, SUBITEM=2")
            if mname == "PET":
                _m(" 4000.0, 0.30")
            elif mname == "PSA":
                _m(" 0.05, 0.499")
            elif mname == "GLASS":
                _m(" 70000.0, 0.22")
            else:
                _m(" 200000.0, 0.30")

        # Sections in .msh (order defines SECNUM = 1, 2, ... in .cnt)
        sec_forms: List[Tuple[int, str, str]] = []
        sec_idx = 0
        for egrp_name, eids in self.egrp_map.items():
            if not eids:
                continue
            sec_idx += 1
            mat_name = self.egrp_mat[egrp_name]
            form_name = self.egrp_form361[egrp_name]
            _m(f"!SECTION, TYPE=SOLID, EGRP={egrp_name}, MATERIAL={mat_name}")
            _m(" 1.0")
            sec_forms.append((sec_idx, egrp_name, form_name))

        _m("!END")
        with open(msh_path, "w", encoding="utf-8") as f:
            f.write("\n".join(msh_lines) + "\n")

        # 3. Write fistr_teardrop.cnt
        theta_rad = math.radians(self.config.simulation_case.theta_max_deg)
        cnt_lines = []
        _c = cnt_lines.append
        _c("!VERSION")
        _c(" 5")
        _c("!SOLUTION, TYPE=NLSTATIC")
        _c("!WRITE, LOG, FREQUENCY=1")
        _c(f"!WRITE, RESULT, FREQUENCY={self.res_write_freq}")
        _c(f"!WRITE, VISUAL, FREQUENCY={self.res_write_freq}")
        _c("!OUTPUT_RES")
        _c(" DISP, ON")
        _c(" REACTION, ON")
        _c(" NSTRESS, ON")
        _c(" NMISES, ON")
        _c(" ESTRESS, ON")
        _c(" EMISES, ON")
        _c("!OUTPUT_VIS")
        _c(" DISP, ON")
        _c(" REACTION, ON")
        _c(" NSTRESS, ON")
        _c(" NMISES, ON")

        # Solver configuration
        if self.solver_method == "MUMPS":
            _c("!SOLVER, METHOD=MUMPS, TIMELOG=YES, ITERLOG=NO")
        else:
            _c(f"!SOLVER, METHOD={self.solver_method}, PRECOND=1, TIMELOG=YES, ITERLOG=NO")
            _c(" 10000, 1")
            _c(" 1.0e-8, 1.0, 0.0")

        # Boundary conditions
        _c("!BOUNDARY, GRPID=1")
        _c(" PIVOT_DUMMY, 1, 3, 0.0")
        if self.z_fixed_nids:
            _c(" Z_FIXED, 3, 3, 0.0")

        # Exact Rodrigues finite-rotation BCs around pivot_L and pivot_R
        # Left wing rotates clockwise (-theta_rad around Z) -> folds UPWARD (+Y)
        _c("!BOUNDARY, ROT_CENTER=PIVOT_L, GRPID=1")
        _c(" LEFT_DRIVEN, 1, 1, 0.0")
        _c(" LEFT_DRIVEN, 2, 2, 0.0")
        _c(f" LEFT_DRIVEN, 3, 3, {-theta_rad:.12f}")

        # Right wing rotates counter-clockwise (+theta_rad around Z) -> folds UPWARD (+Y)
        _c("!BOUNDARY, ROT_CENTER=PIVOT_R, GRPID=1")
        _c(" RIGHT_DRIVEN, 1, 1, 0.0")
        _c(" RIGHT_DRIVEN, 2, 2, 0.0")
        _c(f" RIGHT_DRIVEN, 3, 3, {theta_rad:.12f}")

        # Materials in .cnt (Total Lagrangian Green-Lagrange finite strain by default)
        _c("!MATERIAL, NAME=PET")
        _c("!ELASTIC")
        _c(" 4000.0, 0.30")
        _c("!MATERIAL, NAME=PSA")
        _c("!HYPERELASTIC, TYPE=NEOHOOKE")
        _c(" 8.33889260e-03, 2.40000000e-01")
        _c("!MATERIAL, NAME=GLASS")
        _c("!ELASTIC")
        _c(" 70000.0, 0.22")
        _c("!MATERIAL, NAME=PLATE")
        _c("!ELASTIC")
        _c(" 200000.0, 0.30")

        # Section element formulations (FORM361=IC / FBAR / UP)
        for sec_id, egrp_name, form_str in sec_forms:
            _c(f"!SECTION, SECNUM={sec_id}, FORM361={form_str}")

        # Step control
        _c(f"!STEP, SUBSTEPS={self.substeps}, MAXITER=30, CONVERG=1.0e-5")
        _c(" BOUNDARY, 1")

        # Visualizer VTK output
        _c("!VISUAL, method=PSR")
        _c("!surface_num=1")
        _c("!surface 1")
        _c("!output_type=VTK")
        _c("!END")

        with open(cnt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cnt_lines) + "\n")

        return {
            "run_dir": run_dir,
            "ctrl": ctrl_path,
            "msh": msh_path,
            "cnt": cnt_path,
        }


def parse_fistr_res_file(res_path: str, max_real_nid: int) -> Dict[str, np.ndarray]:
    """
    Parses a FrontISTR sequential result file (fistr_teardrop.res.0.<step>)
    and returns nodal DISP (N, 3), REACTION (N, 3), NSTRESS (N, 6), NMISES (N,).
    Excludes the 8 dummy reference nodes (nid > max_real_nid).
    """
    with open(res_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f]

    # Locate *data section
    idx = 0
    n_lines = len(lines)
    while idx < n_lines and not lines[idx].startswith("*data"):
        idx += 1
    if idx >= n_lines:
        raise ValueError(f"Could not find *data in {res_path}")
    idx += 1

    # Line after *data: n_node, n_elem
    parts = lines[idx].split()
    n_node_total = int(parts[0])
    n_elem_total = int(parts[1])
    idx += 1

    # Next line: n_node_comp, n_elem_comp
    parts = lines[idx].split()
    n_node_comp = int(parts[0])
    n_elem_comp = int(parts[1])
    idx += 1

    # Component sizes (may span multiple lines)
    comp_sizes: List[int] = []
    while len(comp_sizes) < n_node_comp:
        comp_sizes.extend(int(x) for x in lines[idx].split())
        idx += 1

    # Component names
    comp_names: List[str] = []
    for _ in range(n_node_comp):
        comp_names.append(lines[idx].strip())
        idx += 1

    total_vals_per_node = sum(comp_sizes)
    node_data: Dict[int, List[float]] = {}

    for _ in range(n_node_total):
        nid = int(lines[idx].split()[0])
        idx += 1
        vals: List[float] = []
        while len(vals) < total_vals_per_node:
            vals.extend(float(x) for x in lines[idx].split())
            idx += 1
        if nid <= max_real_nid:
            node_data[nid] = vals

    sorted_nids = sorted(node_data.keys())
    mat = np.array([node_data[n] for n in sorted_nids], dtype=np.float64)

    alias_map = {
        "DISPLACEMENT": "DISP",
        "REACTION_FORCE": "REACTION",
        "NodalSTRESS": "NSTRESS",
        "NodalMISES": "NMISES",
    }
    result: Dict[str, np.ndarray] = {"nids": np.array(sorted_nids, dtype=np.int32)}
    col_offset = 0
    for cname, csize in zip(comp_names, comp_sizes):
        norm_name = alias_map.get(cname, cname)
        result[norm_name] = mat[:, col_offset : col_offset + csize]
        col_offset += csize

    return result


def parse_fistr_0_log(log_path: str, stdout_text: str = "") -> Dict[str, Any]:
    """Parses 0.log and stdout to verify completion and extract substep iteration history."""
    text = ""
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

    combined = text + "\n" + stdout_text
    completed = "FrontISTR Completed !!" in combined
    step_blocks = re.findall(r"step\s*=\s*(\d+)\s*,\s*substep\s*=\s*(\d+)", combined, flags=re.IGNORECASE)
    if not step_blocks:
        step_blocks = re.findall(r"Step\s*=\s*(\d+)\s+Substep\s*=\s*(\d+)", combined, flags=re.IGNORECASE)
    return {
        "completed": completed,
        "num_logged_substeps": len(step_blocks),
        "raw_tail": "\n".join((stdout_text.strip() or text.strip()).splitlines()[-25:]),
    }


def compute_reaction_moment_from_res(
    nodes_3d: Dict[int, Tuple[float, float, float]],
    res_data: Dict[str, np.ndarray],
    driven_nids: List[int],
    pivot_xy: Tuple[float, float],
) -> Dict[str, float]:
    """
    Computes net reaction forces (Fx, Fy, Fz) and driving moment Mz about the pivot
    in the DEFORMED configuration x = X + u (matching LESSONS_LEARNED.md & DynamicSolver).
    """
    if "REACTION" not in res_data or "DISP" not in res_data:
        return {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0, "Mz": 0.0}

    nid_to_row = {int(n): i for i, n in enumerate(res_data["nids"])}
    disp = res_data["DISP"]
    reac = res_data["REACTION"]

    fx_sum = 0.0
    fy_sum = 0.0
    fz_sum = 0.0
    mz_sum = 0.0
    px, py = pivot_xy

    for nid in driven_nids:
        if nid not in nid_to_row:
            continue
        row = nid_to_row[nid]
        X, Y, Z = nodes_3d[nid]
        ux, uy, uz = disp[row, 0], disp[row, 1], disp[row, 2]
        rx, ry, rz = reac[row, 0], reac[row, 1], reac[row, 2]

        x_def = X + ux
        y_def = Y + uy
        dx = x_def - px
        dy = y_def - py

        fx_sum += rx
        fy_sum += ry
        fz_sum += rz
        mz_sum += dx * ry - dy * rx

    return {
        "Fx": float(fx_sum),
        "Fy": float(fy_sum),
        "Fz": float(fz_sum),
        "Mz": float(mz_sum),
    }


def run_fistr_case(
    dim: str = "2D",
    preset: str = "compact_4",
    psa_form361: str = "FI",
    solid_form361: str = "FI",
    substeps: int = 90,
    threads: int = 4,
    out_base_dir: Optional[str] = None,
    skip_run: bool = False,
    mesh_spec: Optional[Any] = None,
    case_name_override: Optional[str] = None,
    width_z: Optional[float] = None,
    theta_max_deg: Optional[float] = None,
) -> Dict[str, Any]:
    """Builds and runs a single 2D or 3D FrontISTR teardrop folding analysis."""
    if out_base_dir is None:
        out_base_dir = os.path.join(REPO_ROOT, "example_disp_fold", "results_fistr")
    case_name = case_name_override or f"FISTR_{dim.upper()}_{preset}_{solid_form361}_{psa_form361}"
    run_dir = os.path.join(out_base_dir, case_name)
    os.makedirs(run_dir, exist_ok=True)

    config = UnifiedFoldModelConfig(preset=preset, dim=dim)
    if width_z is not None:
        config.geometry.width_z = float(width_z)
    if theta_max_deg is not None:
        config.simulation_case.theta_max_deg = float(theta_max_deg)
    builder = FrontISTRTeardropBuilder(
        config=config,
        psa_form361=psa_form361,
        solid_form361=solid_form361,
        substeps=substeps,
        solver_method="MUMPS",
        mesh_spec=mesh_spec,
    )

    if not skip_run:
        paths = builder.write_fistr_deck(run_dir)

        print("=" * 100)
        print(f"[*] Running FrontISTR ({dim.upper()}) Teardrop Folding: {case_name}")
        print(f"    Run Directory : {run_dir}")
        print(f"    Nodes (3D)    : {builder.real_max_nid} (+8 pivot ref) | Elements (361): {builder.real_max_eid} (+1 ref)")
        print(f"    Formulations  : Solid={solid_form361}, PSA={psa_form361} | Substeps={substeps} | Solver=MUMPS")
        print("=" * 100)

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["PATH"] = FISTR_SMP_DIR + os.pathsep + env.get("PATH", "")

        t0 = time.perf_counter()
        proc = subprocess.run(
            [FISTR_EXE, "-t", str(threads)],
            cwd=run_dir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        wall_time = time.perf_counter() - t0

        stdout_path = os.path.join(run_dir, "fistr_stdout.log")
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(proc.stdout + "\n" + proc.stderr)

        log_info = parse_fistr_0_log(os.path.join(run_dir, "0.log"), proc.stdout)
        completed = log_info["completed"] and (proc.returncode == 0)

        if not completed:
            print(f"[ERROR] FrontISTR run failed (returncode={proc.returncode}, elapsed={wall_time:.2f}s)")
            print("--- Last 25 lines of 0.log / stdout ---")
            print(log_info.get("raw_tail", "") or "\n".join(proc.stdout.splitlines()[-25:]))
            return {
                "case_name": case_name,
                "dim": dim.upper(),
                "preset": preset,
                "completed": False,
                "wall_time": wall_time,
                "run_dir": run_dir,
            }
    else:
        summary_json_path = os.path.join(run_dir, "fistr_summary.json")
        wall_time = 0.0
        if os.path.exists(summary_json_path):
            with open(summary_json_path, "r", encoding="utf-8") as f:
                wall_time = float(json.load(f).get("wall_time_s", 0.0))

    # Parse available result steps: fistr_teardrop.res.0.<step>
    res_files = []
    for fname in os.listdir(run_dir):
        m = re.match(r"^fistr_teardrop\.res\.0\.(\d+)$", fname)
        if m:
            res_files.append((int(m.group(1)), os.path.join(run_dir, fname)))
    res_files.sort(key=lambda x: x[0])

    if not res_files:
        raise RuntimeError(f"FrontISTR completed but no fistr_teardrop.res.0.* files found in {run_dir}")

    pivot_L = builder.raw_meta["left_pivot"]
    pivot_R = builder.raw_meta["right_pivot"]

    history_steps = []
    history_theta = []
    history_M_L = []
    history_M_R = []
    history_Fx_L = []
    history_Fx_R = []
    step_snapshots = {}

    for step_num, rpath in res_files:
        rdata = parse_fistr_res_file(rpath, builder.real_max_nid)
        theta_deg = (step_num / float(substeps)) * config.simulation_case.theta_max_deg
        reac_L = compute_reaction_moment_from_res(
            builder.nodes_3d, rdata, builder.left_driven_nids, pivot_L
        )
        reac_R = compute_reaction_moment_from_res(
            builder.nodes_3d, rdata, builder.right_driven_nids, pivot_R
        )
        history_steps.append(step_num)
        history_theta.append(theta_deg)
        history_M_L.append(reac_L["Mz"])
        history_M_R.append(reac_R["Mz"])
        history_Fx_L.append(reac_L["Fx"])
        history_Fx_R.append(reac_R["Fx"])
        step_snapshots[step_num] = rdata

    final_step, _ = res_files[-1]
    final_res = step_snapshots[final_step]
    disp_final = final_res["DISP"]
    mises_final = final_res["NMISES"][:, 0] if "NMISES" in final_res else np.zeros(len(disp_final))

    # Deformed coordinates of real nodes
    real_nids = final_res["nids"]
    X0 = np.array([builder.nodes_3d[int(n)] for n in real_nids], dtype=np.float64)
    X_def = X0 + disp_final

    # Teardrop geometry metrics at 180-deg fold
    hinge_mask = np.abs(X0[:, 0]) <= config.geometry.hinge_half_gap + 1e-4
    loop_width_x = float(np.max(X_def[hinge_mask, 0]) - np.min(X_def[hinge_mask, 0]))
    apex_y = float(np.min(X_def[hinge_mask, 1]))
    max_y = float(np.max(X_def[:, 1]))
    max_uz = float(np.max(np.abs(disp_final[:, 2])))
    max_mises = float(np.max(mises_final))

    M_L_final = history_M_L[-1]
    M_R_final = history_M_R[-1]
    avg_M = 0.5 * (abs(M_L_final) + abs(M_R_final))
    symm_err = (abs(abs(M_L_final) - abs(M_R_final)) / max(avg_M, 1e-9)) * 100.0

    print(f"[OK] FrontISTR Completed !! in {wall_time:.2f}s (Substeps={final_step}/{substeps})")
    print(f"     Deformed Teardrop Width (Delta X_hinge): {loop_width_x:.4f} mm | Max Height Y: {max_y:.4f} mm | Apex Y_min: {apex_y:.4f} mm")
    print(f"     Max |u_z|: {max_uz:.6e} mm | Max Nodal Mises Stress: {max_mises:.2f} MPa")
    print(f"     Reaction Moments: M_L = {M_L_final:+.4f} N*mm, M_R = {M_R_final:+.4f} N*mm (Symmetry Error: {symm_err:.4f}%)")
    print(f"     Reaction Forces : Fx_L = {history_Fx_L[-1]:+.4f} N, Fx_R = {history_Fx_R[-1]:+.4f} N (Sum Fx = {history_Fx_L[-1]+history_Fx_R[-1]:+.3e} N)")

    summary = {
        "case_name": case_name,
        "dim": dim.upper(),
        "preset": preset,
        "solid_form361": solid_form361,
        "psa_form361": psa_form361,
        "substeps": substeps,
        "completed": True,
        "wall_time_s": wall_time,
        "num_real_nodes": int(builder.real_max_nid),
        "num_real_elements": int(builder.real_max_eid),
        "loop_width_x_mm": loop_width_x,
        "apex_y_mm": apex_y,
        "max_y_mm": max_y,
        "max_abs_uz_mm": max_uz,
        "max_mises_mpa": max_mises,
        "M_L_final_Nmm": M_L_final,
        "M_R_final_Nmm": M_R_final,
        "Fx_L_final_N": history_Fx_L[-1],
        "Fx_R_final_N": history_Fx_R[-1],
        "moment_symmetry_err_pct": symm_err,
        "history_theta_deg": history_theta,
        "history_M_L": history_M_L,
        "history_M_R": history_M_R,
        "history_Fx_L": history_Fx_L,
        "history_Fx_R": history_Fx_R,
        "run_dir": run_dir,
    }

    with open(os.path.join(run_dir, "fistr_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "summary": summary,
        "builder": builder,
        "X0": X0,
        "X_def": X_def,
        "step_snapshots": step_snapshots,
    }


def generate_fistr_comparison_plots(
    res_2d: Dict[str, Any],
    res_3d: Dict[str, Any],
    out_plot_path: str,
) -> None:
    """Generates a 4-panel Matplotlib comparison plot of FrontISTR 2D vs 3D Teardrop Folding."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import koreanize_matplotlib  # noqa: F401
    except ImportError:
        pass

    plt.rcParams["font.size"] = 9
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=180)

    # Panel 1: Progressive & Final Deformed Teardrop Shape (2D vs 3D)
    ax1 = axes[0, 0]
    for res_obj, label_prefix, color_main, ls in [
        (res_2d, "2D (평면변형률 361)", "#1f77b4", "-"),
        (res_3d, "3D (연속체 솔리드 361)", "#d62728", "--"),
    ]:
        X0 = res_obj["X0"]
        snaps = res_obj["step_snapshots"]
        substeps = res_obj["summary"]["substeps"]
        z_min = float(np.min(X0[:, 2]))
        y_cont_bot = float(np.min(X0[np.abs(X0[:, 0]) < 1e-3, 1]))
        # Plot intermediate steps lightly for 2D
        if label_prefix.startswith("2D"):
            for stp, rdata in sorted(snaps.items()):
                if stp == substeps:
                    continue
                deg = (stp / float(substeps)) * 90.0
                X_s = X0 + rdata["DISP"]
                bot_mask = (np.abs(X0[:, 1] - y_cont_bot) < 1e-5) & (np.abs(X0[:, 2] - z_min) < 1e-5)
                order = np.argsort(X0[bot_mask, 0])
                ax1.plot(
                    X_s[bot_mask, 0][order],
                    X_s[bot_mask, 1][order],
                    color="#aec7e8",
                    lw=0.9,
                    alpha=0.7,
                    label=f"중간 폴딩 궤적 ({deg:.0f}°)" if stp in (30, 60) else None,
                )

        # Plot final 180-deg deformed bottom & top neutral profiles
        X_def = res_obj["X_def"]
        bot_mask = (np.abs(X0[:, 1] - y_cont_bot) < 1e-5) & (np.abs(X0[:, 2] - z_min) < 1e-5)
        order = np.argsort(X0[bot_mask, 0])
        ax1.plot(
            X_def[bot_mask, 0][order],
            X_def[bot_mask, 1][order],
            color=color_main,
            linestyle=ls,
            lw=2.0,
            label=f"{label_prefix} 최종 180° 폴딩",
        )

    ax1.set_title("1. FrontISTR 2D vs 3D 180° Tear-Drop 변형 형상 (전체)")
    ax1.set_xlabel("X [mm]")
    ax1.set_ylabel("Y [mm]")
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", frameon=True)

    # Panel 2: Hinge Zoom-In Teardrop Loop Profile (2D vs 3D)
    ax2 = axes[0, 1]
    for res_obj, label_prefix, color_main, marker in [
        (res_2d, "2D 평면변형률", "#1f77b4", "o"),
        (res_3d, "3D 연속체", "#d62728", "x"),
    ]:
        X0 = res_obj["X0"]
        X_def = res_obj["X_def"]
        z_min = float(np.min(X0[:, 2]))
        y_cont_bot = float(np.min(X0[np.abs(X0[:, 0]) < 1e-3, 1]))
        y_max0 = float(np.max(X0[:, 1]))
        hinge_bot = (np.abs(X0[:, 0]) <= 12.0) & (np.abs(X0[:, 1] - y_cont_bot) < 1e-5) & (np.abs(X0[:, 2] - z_min) < 1e-5)
        hinge_top = (np.abs(X0[:, 0]) <= 12.0) & (np.abs(X0[:, 1] - y_max0) < 1e-5) & (np.abs(X0[:, 2] - z_min) < 1e-5)
        ord_b = np.argsort(X0[hinge_bot, 0])
        ord_t = np.argsort(X0[hinge_top, 0])
        ax2.plot(
            X_def[hinge_bot, 0][ord_b],
            X_def[hinge_bot, 1][ord_b],
            color=color_main,
            lw=1.8,
            label=f"{label_prefix} (연속 하면)",
        )
        ax2.plot(
            X_def[hinge_top, 0][ord_t],
            X_def[hinge_top, 1][ord_t],
            color=color_main,
            linestyle=":",
            lw=1.5,
            label=f"{label_prefix} (상면)",
        )

    ax2.axvline(-1.4, color="gray", linestyle="--", lw=0.8, label="피벗 X = ±1.4 mm")
    ax2.axvline(+1.4, color="gray", linestyle="--", lw=0.8)
    ax2.set_title("2. 힌지 영역 Tear-Drop 루프 확대 비교 (X ∈ [-5, +5] mm)")
    ax2.set_xlabel("변형 후 X [mm]")
    ax2.set_ylabel("변형 후 Y [mm]")
    ax2.set_xlim(-5.5, 5.5)
    ax2.set_ylim(-1.0, 10.0)
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", frameon=True)

    # Panel 3: Driving Reaction Moment vs Folding Angle
    ax3 = axes[1, 0]
    for res_obj, label_prefix, c_l, c_r, ls in [
        (res_2d, "2D", "#1f77b4", "#2ca02c", "-"),
        (res_3d, "3D", "#d62728", "#ff7f0e", "--"),
    ]:
        s = res_obj["summary"]
        th = s["history_theta_deg"]
        ml = np.abs(s["history_M_L"])
        mr = np.abs(s["history_M_R"])
        ax3.plot(th, ml, color=c_l, linestyle=ls, marker="o", ms=4, lw=1.6, label=f"{label_prefix} |M_L| [N·mm]")
        ax3.plot(th, mr, color=c_r, linestyle=ls, marker="s", ms=3, lw=1.2, label=f"{label_prefix} |M_R| [N·mm]")

    ax3.set_title("3. 폴딩 각도별 피벗 구동 반력 모멘트 (|M_drive|)")
    ax3.set_xlabel("플레이트 회전각 θ [deg] (총 폴딩각 = 2θ)")
    ax3.set_ylabel("구동 모멘트 |M_z| [N·mm]")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="best", frameon=True)

    # Panel 4: 3D Out-of-Plane Z-Displacement (Poisson Bulging / Anticlastic Effect) vs 2D Plane Strain
    ax4 = axes[1, 1]
    for res_obj, label_prefix, color_main in [
        (res_2d, "2D 평면변형률 (모든 절점 u_z = 0 구속)", "#1f77b4"),
        (res_3d, "3D 연속체 자유단 (Z = +0.5 mm 외측면 u_z)", "#d62728"),
    ]:
        X0 = res_obj["X0"]
        final_step = res_obj["summary"]["substeps"]
        disp = res_obj["step_snapshots"][final_step]["DISP"]
        z_max = np.max(X0[:, 2])
        y_max0 = np.max(X0[:, 1])
        top_zmax_mask = (np.abs(X0[:, 2] - z_max) < 1e-5) & (np.abs(X0[:, 1] - y_max0) < 1e-5)
        ord_x = np.argsort(X0[top_zmax_mask, 0])
        ax4.plot(
            X0[top_zmax_mask, 0][ord_x],
            disp[top_zmax_mask, 2][ord_x] * 1000.0,  # convert mm to um
            color=color_main,
            lw=1.8,
            label=label_prefix,
        )

    ax4.set_title("4. 폭 방향(Z) 면외 변위 u_z 분포 비교 (2D 평면변형률 vs 3D 자유단)")
    ax4.set_xlabel("초기 좌표 X_0 [mm]")
    ax4.set_ylabel("면외 변위 u_z [μm]")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="best", frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_plot_path), exist_ok=True)
    plt.savefig(out_plot_path, dpi=180)
    plt.close(fig)
    print(f"[OK] Saved 2D vs 3D FrontISTR comparison plot: {out_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FrontISTR 2D & 3D Foldable Display Teardrop Runner")
    parser.add_argument("--dim", type=str, default="both", choices=["2D", "3D", "both"])
    parser.add_argument("--preset", type=str, default="compact_4", choices=["compact_4", "teardrop"])
    parser.add_argument("--solid_form", type=str, default="FI", choices=["FI", "BBAR", "IC", "FBAR"])
    parser.add_argument("--psa_form", type=str, default="FI", choices=["FI", "BBAR", "FBAR", "UP", "IC"])
    parser.add_argument("--substeps", type=int, default=90)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--skip_run", action="store_true", help="Re-parse existing results without re-running fistr1.exe")
    args = parser.parse_args()

    out_base_dir = os.path.join(REPO_ROOT, "example_disp_fold", "results_fistr")
    os.makedirs(out_base_dir, exist_ok=True)

    if args.dim.lower() == "both":
        res_2d = run_fistr_case(
            dim="2D",
            preset=args.preset,
            psa_form361=args.psa_form,
            solid_form361=args.solid_form,
            substeps=args.substeps,
            threads=args.threads,
            out_base_dir=out_base_dir,
            skip_run=args.skip_run,
        )
        res_3d = run_fistr_case(
            dim="3D",
            preset=args.preset,
            psa_form361=args.psa_form,
            solid_form361=args.solid_form,
            substeps=args.substeps,
            threads=args.threads,
            out_base_dir=out_base_dir,
            skip_run=args.skip_run,
        )
        if res_2d.get("summary", {}).get("completed") and res_3d.get("summary", {}).get("completed"):
            plot_path = os.path.join(out_base_dir, f"fistr_2d_vs_3d_{args.preset}_comparison.png")
            generate_fistr_comparison_plots(res_2d, res_3d, plot_path)
    else:
        run_fistr_case(
            dim=args.dim.upper(),
            preset=args.preset,
            psa_form361=args.psa_form,
            solid_form361=args.solid_form,
            substeps=args.substeps,
            threads=args.threads,
            out_base_dir=out_base_dir,
            skip_run=args.skip_run,
        )


if __name__ == "__main__":
    main()
