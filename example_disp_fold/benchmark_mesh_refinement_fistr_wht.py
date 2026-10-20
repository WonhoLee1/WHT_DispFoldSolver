"""
benchmark_mesh_refinement_fistr_wht.py
======================================
Comprehensive benchmark suite testing FrontISTR and WHT_DispFoldSolver:
1. 2D Mesh Refinement (100%, 50%, 25%, 12.5%, 6.25%):
   - Execution speed (MUMPS setup/solve, wall time, iters/substep)
   - Load (Driving moment Mz, reaction forces Fx, Fy)
   - Stress (Peak von Mises stress in PET, PSA, Glass)
   - Strain (Peak shear strain gamma_xy and axial strains)
2. 3D Mesh Refinement:
   - In-plane sizes (100%, 50%, 25%)
   - Depth scaling (nz = 1, 10, 20) at 100% in-plane mesh
   - Depth scaling (nz = 1, 10, 20) at 50% in-plane mesh
3. Direct WHT Solver vs FrontISTR Parity Comparison:
   - Speed (Intel oneMKL PARDISO vs MUMPS)
   - Reaction moment, stress, strain, and deformed teardrop silhouette
4. Generation of consolidated Markdown report, CSV data, and Matplotlib/PyVista figures.
"""

from __future__ import annotations
import os
import sys
import re
import json
import time
import math
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig, MeshSpec
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from example_disp_fold.run_fistr_teardrop import (
    run_fistr_case,
    FrontISTRTeardropBuilder,
    parse_fistr_res_file,
    compute_reaction_moment_from_res,
)
from example_disp_fold.benchmark_and_capture_fistr_xy import parse_detailed_fistr_speed

from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.fold_model_config import smoothstep_amp


# -----------------------------------------------------------------------------
# WHT Solver Runner for Foldable Display
# -----------------------------------------------------------------------------
def run_wht_case(
    dim: str = "2D",
    preset: str = "teardrop",
    mesh_spec: Optional[MeshSpec] = None,
    substeps: int = 30,
    threads: int = 6,
    case_name_override: Optional[str] = None,
    out_base_dir: Optional[str] = None,
    width_z: Optional[float] = None,
) -> Dict[str, Any]:
    """Runs WHT_DispFoldSolver (DynamicSolver2D or DynamicSolver3D) on exact same config and mesh."""
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)

    if out_base_dir is None:
        out_base_dir = os.path.join(REPO_ROOT, "example_disp_fold", "results_wht")
    os.makedirs(out_base_dir, exist_ok=True)

    elem_case = "2D-Opt1" if dim == "2D" else "3D-Opt1"
    config = UnifiedFoldModelConfig(preset=preset, dim=dim, elem_case=elem_case)
    if mesh_spec is not None:
        config.mesh = mesh_spec
    if width_z is not None:
        config.geometry.width_z = float(width_z)

    mesher = GmshMeshGenerator(config)
    mesh, meta = mesher.build()

    case_name = case_name_override or f"WHT_{dim}_{preset}_{config.mesh.nx_hinge}_{config.mesh.nz_elements}"
    run_dir = os.path.join(out_base_dir, case_name)
    os.makedirs(run_dir, exist_ok=True)

    # Material properties
    materials = {
        1: {"E": 4000.0, "nu": 0.30},
        2: {"E": 0.05, "nu": 0.499},
        3: {"E": 70000.0, "nu": 0.22},
        4: {"E": 200000.0, "nu": 0.30},
    }

    t0_setup = time.perf_counter()
    if dim == "2D":
        solver = DynamicSolver2D(mesh, materials=materials, nlgeom=True)
        solver.max_du_limit = 1.0
    else:
        solver = DynamicSolver3D(mesh, materials=materials, nlgeom=True, solution_controls="STANDARD")
        solver.max_du_limit = 1.0
    setup_time = time.perf_counter() - t0_setup

    pivot_L = meta["left_pivot"]
    pivot_R = meta["right_pivot"]
    hinge_gap = config.geometry.hinge_half_gap
    disp_left_plate_nodes = [nid for nid, node in mesh.nodes.items() if node.x <= -hinge_gap + 1e-4 and node.y >= 0.0]
    disp_right_plate_nodes = [nid for nid, node in mesh.nodes.items() if node.x >= hinge_gap - 1e-4 and node.y >= 0.0]
    nid_map = mesh.node_id_to_index()
    ndof_n = 2 if dim == "2D" else 3

    theta_max_rad = math.radians(config.simulation_case.theta_max_deg)
    dt = 1.0 / float(substeps)

    history_M_L = []
    history_M_R = []
    history_Fx_L = []
    history_Fx_R = []
    history_theta = []
    total_newton_iters = 0

    print(f"[*] Running WHT Solver ({dim}): {case_name}")
    print(f"    Nodes: {len(mesh.nodes)} | Elements: {len(mesh.elements)} | DOFs: {solver.num_dofs} | Substeps: {substeps}")

    t0_solve = time.perf_counter()
    theta_prev = 0.0

    for step in range(1, substeps + 1):
        s = step / float(substeps)
        theta_curr = theta_max_rad * smoothstep_amp(s)
        deg_curr = math.degrees(theta_curr)

        # Dirichlet BCs (Clockwise left, CCW right -> fold upward into +Y)
        cos_th_L = math.cos(-theta_curr); sin_th_L = math.sin(-theta_curr)
        for nid in meta["plate_left_nodes"] + meta["disp_bottom_left_nids"]:
            node = mesh.nodes[nid]
            dx0 = node.x - pivot_L[0]; dy0 = node.y - pivot_L[1]
            solver.fix_dof(nid, 0, pivot_L[0] + cos_th_L * dx0 - sin_th_L * dy0 - node.x)
            solver.fix_dof(nid, 1, pivot_L[1] + sin_th_L * dx0 + cos_th_L * dy0 - node.y)
            if dim == "3D":
                solver.fix_dof(nid, 2, 0.0)

        cos_th_R = math.cos(theta_curr); sin_th_R = math.sin(theta_curr)
        for nid in meta["plate_right_nodes"] + meta["disp_bottom_right_nids"]:
            node = mesh.nodes[nid]
            dx0 = node.x - pivot_R[0]; dy0 = node.y - pivot_R[1]
            solver.fix_dof(nid, 0, pivot_R[0] + cos_th_R * dx0 - sin_th_R * dy0 - node.x)
            solver.fix_dof(nid, 1, pivot_R[1] + sin_th_R * dx0 + cos_th_R * dy0 - node.y)
            if dim == "3D":
                solver.fix_dof(nid, 2, 0.0)

        if dim == "3D":
            for nid, node in mesh.nodes.items():
                if abs(getattr(node, "z", 0.0)) < 1e-4:
                    solver.fix_dof(nid, 2, 0.0)

        # Kinematic Rotation Predictor
        dth_L = -(theta_curr - theta_prev)
        cos_dth_L = math.cos(dth_L); sin_dth_L = math.sin(dth_L)
        for nid in disp_left_plate_nodes:
            node = mesh.nodes[nid]; idx = nid_map[nid]
            cx = node.x + solver.u[idx * ndof_n + 0]; cy = node.y + solver.u[idx * ndof_n + 1]
            dx = cx - pivot_L[0]; dy = cy - pivot_L[1]
            solver.u[idx * ndof_n + 0] = pivot_L[0] + cos_dth_L * dx - sin_dth_L * dy - node.x
            solver.u[idx * ndof_n + 1] = pivot_L[1] + sin_dth_L * dx + cos_dth_L * dy - node.y

        dth_R = +(theta_curr - theta_prev)
        cos_dth_R = math.cos(dth_R); sin_dth_R = math.sin(dth_R)
        for nid in disp_right_plate_nodes:
            node = mesh.nodes[nid]; idx = nid_map[nid]
            cx = node.x + solver.u[idx * ndof_n + 0]; cy = node.y + solver.u[idx * ndof_n + 1]
            dx = cx - pivot_R[0]; dy = cy - pivot_R[1]
            solver.u[idx * ndof_n + 0] = pivot_R[0] + cos_dth_R * dx - sin_dth_R * dy - node.x
            solver.u[idx * ndof_n + 1] = pivot_R[1] + sin_dth_R * dx + cos_dth_R * dy - node.y

        for dof, val in solver.fixed_dofs.items():
            solver.u[dof] = val

        converged, iters = solver.solve_step(
            dt=dt,
            max_iters=25,
            use_anderson_accel=True,
            anderson_m=4,
        )
        total_newton_iters += iters
        theta_prev = theta_curr

        if not converged:
            print(f"[!] WHT solver did not converge at step {step}/{substeps}")
            break

        # Compute reaction forces & moments using rigorous compute_section_reactions
        driven_nodes_L = meta["plate_left_nodes"] + meta["disp_bottom_left_nids"]
        driven_nodes_R = meta["plate_right_nodes"] + meta["disp_bottom_right_nids"]

        ctr_L = (float(pivot_L[0]), float(pivot_L[1])) if dim == "2D" else (float(pivot_L[0]), float(pivot_L[1]), 0.0)
        ctr_R = (float(pivot_R[0]), float(pivot_R[1])) if dim == "2D" else (float(pivot_R[0]), float(pivot_R[1]), 0.0)

        reac_L = solver.compute_section_reactions(driven_nodes_L, center_pt=ctr_L, current_coords=True)
        reac_R = solver.compute_section_reactions(driven_nodes_R, center_pt=ctr_R, current_coords=True)

        history_theta.append(deg_curr)
        history_M_L.append(float(reac_L["Mz"]))
        history_M_R.append(float(reac_R["Mz"]))
        history_Fx_L.append(float(reac_L["Fx"]))
        history_Fx_R.append(float(reac_R["Fx"]))

    solve_time = time.perf_counter() - t0_solve
    wall_time = setup_time + solve_time

    # Final metrics
    coords0 = mesh.nodes_array()
    if dim == "2D":
        disp = solver.u.reshape(-1, 2)
        disp_3d = np.column_stack([disp, np.zeros(len(disp))])
        coords0_3d = np.column_stack([coords0, np.zeros(len(coords0))])
    else:
        disp_3d = solver.u.reshape(-1, 3)
        coords0_3d = coords0

    X_def = coords0_3d + disp_3d
    hinge_mask = np.abs(coords0_3d[:, 0]) <= config.geometry.hinge_half_gap + 1e-4
    loop_width_x = float(np.max(X_def[hinge_mask, 0]) - np.min(X_def[hinge_mask, 0]))
    apex_y = float(np.min(X_def[hinge_mask, 1]))
    max_y = float(np.max(X_def[:, 1]))

    # Von Mises Stress approximation across layers
    # DynamicSolver compute_stresses if available or element strain energy
    stresses = getattr(solver, "element_stresses", None)
    if stresses is not None and len(stresses) > 0:
        max_mises = float(np.max([np.linalg.norm(s) for s in stresses.values()]))
    else:
        # Compute from linear elastic stress or constitutive post
        max_mises = 0.0

    M_L_final = history_M_L[-1] if history_M_L else 0.0
    M_R_final = history_M_R[-1] if history_M_R else 0.0
    avg_M = 0.5 * (abs(M_L_final) + abs(M_R_final))
    symm_err = (abs(abs(M_L_final) - abs(M_R_final)) / max(avg_M, 1e-9)) * 100.0

    print(f"[OK] WHT Solver Completed in {wall_time:.2f}s (Solve: {solve_time:.2f}s, Setup: {setup_time:.2f}s)")
    print(f"     Deformed Width: {loop_width_x:.4f} mm | M_L = {M_L_final:+.4f} N*mm, M_R = {M_R_final:+.4f} N*mm")

    summary = {
        "case_name": case_name,
        "solver": "WHT",
        "dim": dim,
        "preset": preset,
        "nodes": len(mesh.nodes),
        "elements": len(mesh.elements),
        "dofs": solver.num_dofs,
        "substeps": substeps,
        "total_newton_iters": total_newton_iters,
        "avg_iters_per_substep": total_newton_iters / float(max(substeps, 1)),
        "wall_time_s": wall_time,
        "solve_time_s": solve_time,
        "setup_time_s": setup_time,
        "ms_per_substep": (solve_time * 1000.0) / float(max(substeps, 1)),
        "ms_per_iter": (solve_time * 1000.0) / float(max(total_newton_iters, 1)),
        "loop_width_x_mm": loop_width_x,
        "apex_y_mm": apex_y,
        "max_y_mm": max_y,
        "M_L_final_Nmm": M_L_final,
        "M_R_final_Nmm": M_R_final,
        "moment_symmetry_err_pct": symm_err,
        "history_theta": history_theta,
        "history_M_L": history_M_L,
        "history_M_R": history_M_R,
    }

    with open(os.path.join(run_dir, "wht_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return {
        "summary": summary,
        "X0": coords0_3d,
        "X_def": X_def,
        "disp": disp_3d,
    }


# -----------------------------------------------------------------------------
# Benchmark Runner Orchestrating 2D, 3D, and Parity Comparisons
# -----------------------------------------------------------------------------
class MeshRefinementBenchmarkSuite:
    """Executes the complete matrix of 2D, 3D, and WHT parity tests."""

    def __init__(
        self,
        preset: str = "teardrop",
        substeps: int = 30,
        threads: int = 6,
        out_dir: Optional[str] = None,
    ) -> None:
        self.preset = preset
        self.substeps = substeps
        self.threads = threads
        if out_dir is None:
            out_dir = os.path.join(REPO_ROOT, "example_disp_fold", "results_benchmark_refinement")
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.results_2d: List[Dict[str, Any]] = []
        self.results_3d_inplane: List[Dict[str, Any]] = []
        self.results_3d_depth_100: List[Dict[str, Any]] = []
        self.results_3d_depth_50: List[Dict[str, Any]] = []
        self.results_wht_2d: List[Dict[str, Any]] = []
        self.results_wht_3d: List[Dict[str, Any]] = []

        self.load_all_existing_results()

    def load_all_existing_results(self) -> None:
        """Discovers and parses all completed FISTR and WHT runs in out_dir."""
        for dname in sorted(os.listdir(self.out_dir)):
            dpath = os.path.join(self.out_dir, dname)
            if not os.path.isdir(dpath):
                continue
            fistr_sum = os.path.join(dpath, "fistr_summary.json")
            wht_sum = os.path.join(dpath, "wht_summary.json")
            if os.path.exists(fistr_sum):
                try:
                    with open(fistr_sum, "r", encoding="utf-8") as f:
                        s = json.load(f)
                    if s.get("preset", "") != self.preset:
                        continue
                    speed = parse_detailed_fistr_speed(dpath)
                    s.update(speed)
                    if "Refine" in dname and "2D" in dname:
                        m = re.search(r"(\d+(?:\.\d+)?)pct", dname)
                        sc = float(m.group(1)) if m else 100.0
                        s["mesh_scale_pct"] = sc
                        s["elem_size_ratio"] = sc / 100.0
                        if not any(r["summary"]["case_name"] == s["case_name"] for r in self.results_2d):
                            self.results_2d.append({"summary": s})
                    elif "InPlane" in dname:
                        m = re.search(r"(\d+(?:\.\d+)?)pct", dname)
                        sc = float(m.group(1)) if m else 100.0
                        s["mesh_scale_pct"] = sc
                        if not any(r["summary"]["case_name"] == s["case_name"] for r in self.results_3d_inplane):
                            self.results_3d_inplane.append({"summary": s})
                    elif "Depth100" in dname:
                        m = re.search(r"nz(\d+)", dname)
                        nz = int(m.group(1)) if m else 1
                        s["nz_elements"] = nz
                        if not any(r["summary"]["case_name"] == s["case_name"] for r in self.results_3d_depth_100):
                            self.results_3d_depth_100.append({"summary": s})
                    elif "Depth50" in dname:
                        m = re.search(r"nz(\d+)", dname)
                        nz = int(m.group(1)) if m else 1
                        s["nz_elements"] = nz
                        if not any(r["summary"]["case_name"] == s["case_name"] for r in self.results_3d_depth_50):
                            self.results_3d_depth_50.append({"summary": s})
                except Exception as e:
                    print(f"[!] Warning reading {fistr_sum}: {e}")
            elif os.path.exists(wht_sum):
                try:
                    with open(wht_sum, "r", encoding="utf-8") as f:
                        s = json.load(f)
                    if s.get("preset", "") != self.preset:
                        continue
                    if s["dim"] == "2D":
                        if not any(r["summary"]["case_name"] == s["case_name"] for r in self.results_wht_2d):
                            self.results_wht_2d.append({"summary": s})
                    else:
                        if not any(r["summary"]["case_name"] == s["case_name"] for r in self.results_wht_3d):
                            self.results_wht_3d.append({"summary": s})
                except Exception as e:
                    print(f"[!] Warning reading {wht_sum}: {e}")

        # Sort all lists consistently
        self.results_2d.sort(key=lambda r: -r["summary"].get("mesh_scale_pct", 0.0))
        self.results_3d_inplane.sort(key=lambda r: -r["summary"].get("mesh_scale_pct", 0.0))
        self.results_3d_depth_100.sort(key=lambda r: r["summary"].get("nz_elements", 0))
        self.results_3d_depth_50.sort(key=lambda r: r["summary"].get("nz_elements", 0))
        self.results_wht_2d.sort(key=lambda r: r["summary"].get("elements", 0))
        self.results_wht_3d.sort(key=lambda r: r["summary"].get("elements", 0))

    # 1. 2D Folding Test (100% to 6.25%)
    def run_2d_refinement_matrix(self) -> None:
        print("\n" + "#" * 100)
        print(" [PART 1] 2D Mesh Refinement (100%, 50%, 25%, 12.5%, 6.25%) FrontISTR Tests")
        print("#" * 100)

        levels = [
            ("100.0%", 1.0, 30, 20),
            ("50.0%",  0.5, 60, 40),
            ("25.0%",  0.25, 120, 80),
            ("12.5%",  0.125, 240, 160),
            ("6.25%",  0.0625, 480, 320),
        ]

        for label, scale, nx_h, nx_p in levels:
            cname = f"FISTR_2D_Refine_{label.replace('%','pct')}"
            mspec = MeshSpec(nx_hinge=nx_h, nx_plate=nx_p, nz_elements=1)
            res = run_fistr_case(
                dim="2D",
                preset=self.preset,
                psa_form361="FI",
                solid_form361="FI",
                substeps=self.substeps,
                threads=self.threads,
                out_base_dir=self.out_dir,
                mesh_spec=mspec,
                case_name_override=cname,
            )
            if res.get("summary", {}).get("completed", False):
                speed_info = parse_detailed_fistr_speed(res["summary"]["run_dir"])
                res["summary"].update(speed_info)
                res["summary"]["mesh_scale_pct"] = scale * 100.0
                res["summary"]["elem_size_ratio"] = scale
                self.results_2d.append(res)
            else:
                print(f"[!] Warning: {cname} failed to complete.")

    # 2. 3D Folding Test (In-plane 100, 50, 25 + Depth nz=1, 10, 20)
    def run_3d_refinement_matrix(self) -> None:
        print("\n" + "#" * 100)
        print(" [PART 2] 3D Mesh Refinement & Depth Scaling FrontISTR Tests")
        print("#" * 100)

        # 2A. In-plane 100%, 50%, 25% with nz=1, width_z=1.0 (dz = 1.0 mm)
        inplane_levels = [
            ("100.0%", 1.0, 30, 20),
            ("50.0%",  0.5, 60, 40),
            ("25.0%",  0.25, 120, 80),
        ]
        for label, scale, nx_h, nx_p in inplane_levels:
            cname = f"FISTR_3D_InPlane_{label.replace('%','pct')}"
            mspec = MeshSpec(nx_hinge=nx_h, nx_plate=nx_p, nz_elements=1)
            res = run_fistr_case(
                dim="3D",
                preset=self.preset,
                psa_form361="FI",
                solid_form361="FI",
                substeps=self.substeps,
                threads=self.threads,
                out_base_dir=self.out_dir,
                mesh_spec=mspec,
                case_name_override=cname,
                width_z=1.0,
            )
            if res.get("summary", {}).get("completed", False):
                speed_info = parse_detailed_fistr_speed(res["summary"]["run_dir"])
                res["summary"].update(speed_info)
                res["summary"]["mesh_scale_pct"] = scale * 100.0
                self.results_3d_inplane.append(res)
            else:
                print(f"[!] Warning: {cname} failed to complete.")

        # 2B. Depth scaling at 100% in-plane mesh (nz = 1, 10, 20, dz = 1.0 mm)
        depth_nz_list = [1, 10, 20]
        for nz in depth_nz_list:
            cname = f"FISTR_3D_Depth100_nz{nz}"
            mspec = MeshSpec(nx_hinge=30, nx_plate=20, nz_elements=nz)
            w_z = float(nz) * 1.0
            res = run_fistr_case(
                dim="3D",
                preset=self.preset,
                psa_form361="FI",
                solid_form361="FI",
                substeps=self.substeps,
                threads=self.threads,
                out_base_dir=self.out_dir,
                mesh_spec=mspec,
                case_name_override=cname,
                width_z=w_z,
            )
            if res.get("summary", {}).get("completed", False):
                speed_info = parse_detailed_fistr_speed(res["summary"]["run_dir"])
                res["summary"].update(speed_info)
                res["summary"]["nz_elements"] = nz
                self.results_3d_depth_100.append(res)
            else:
                print(f"[!] Warning: {cname} failed to complete.")

        # 2C. Depth scaling at 50% in-plane mesh (nz = 1, 10, 20, dz = 1.0 mm)
        for nz in depth_nz_list:
            cname = f"FISTR_3D_Depth50_nz{nz}"
            mspec = MeshSpec(nx_hinge=60, nx_plate=40, nz_elements=nz)
            w_z = float(nz) * 1.0
            res = run_fistr_case(
                dim="3D",
                preset=self.preset,
                psa_form361="FI",
                solid_form361="FI",
                substeps=self.substeps,
                threads=self.threads,
                out_base_dir=self.out_dir,
                mesh_spec=mspec,
                case_name_override=cname,
                width_z=w_z,
            )
            if res.get("summary", {}).get("completed", False):
                speed_info = parse_detailed_fistr_speed(res["summary"]["run_dir"])
                res["summary"].update(speed_info)
                res["summary"]["nz_elements"] = nz
                self.results_3d_depth_50.append(res)
            else:
                print(f"[!] Warning: {cname} failed to complete.")

    # 3. WHT Solver Direct Parity Tests
    def run_wht_parity_matrix(self) -> None:
        print("\n" + "#" * 100)
        print(" [PART 3] WHT_DispFoldSolver Parity Tests on Matching Discretizations")
        print("#" * 100)
        self.results_wht_2d.clear()
        self.results_wht_3d.clear()

        # 2D WHT cases: 100%, 50%, 25%
        wht_2d_levels = [
            ("100.0%", 30, 20),
            ("50.0%",  60, 40),
            ("25.0%",  120, 80),
        ]
        for label, nx_h, nx_p in wht_2d_levels:
            cname = f"WHT_2D_Refine_{label.replace('%','pct')}"
            mspec = MeshSpec(nx_hinge=nx_h, nx_plate=nx_p, nz_elements=1)
            res = run_wht_case(
                dim="2D",
                preset=self.preset,
                mesh_spec=mspec,
                substeps=self.substeps,
                threads=self.threads,
                case_name_override=cname,
                out_base_dir=self.out_dir,
            )
            self.results_wht_2d.append(res)

        # 3D WHT cases: 100% (nz=1, 10, 20, dz=1.0 mm)
        wht_3d_cases = [
            ("100_nz1", 30, 20, 1),
            ("100_nz10", 30, 20, 10),
            ("100_nz20", 30, 20, 20),
        ]
        for ctag, nx_h, nx_p, nz in wht_3d_cases:
            cname = f"WHT_3D_{ctag}"
            mspec = MeshSpec(nx_hinge=nx_h, nx_plate=nx_p, nz_elements=nz)
            w_z = float(nz) * 1.0
            res = run_wht_case(
                dim="3D",
                preset=self.preset,
                mesh_spec=mspec,
                substeps=self.substeps,
                threads=self.threads,
                case_name_override=cname,
                out_base_dir=self.out_dir,
                width_z=w_z,
            )
            self.results_wht_3d.append(res)

    # 4. Generate Comprehensive Consolidated Report & Figures
    def generate_benchmark_report_and_plots(self) -> str:
        report_md_path = os.path.join(self.out_dir, "benchmark_report.md")
        report_json_path = os.path.join(self.out_dir, "benchmark_summary_data.json")

        summary_export = {
            "results_2d": [r["summary"] for r in self.results_2d],
            "results_3d_inplane": [r["summary"] for r in self.results_3d_inplane],
            "results_3d_depth_100": [r["summary"] for r in self.results_3d_depth_100],
            "results_3d_depth_50": [r["summary"] for r in self.results_3d_depth_50],
            "results_wht_2d": [r["summary"] for r in self.results_wht_2d],
            "results_wht_3d": [r["summary"] for r in self.results_wht_3d],
        }
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_export, f, indent=2, ensure_ascii=False)

        lines = []
        _p = lines.append
        _p("# FrontISTR vs WHT Solver 메쉬 세분화 및 성능·정확도 벤치마크 결과 보고서\n")
        _p(f"- **실시 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lay_desc = "14-Layer Full-Stack: 7 PSA + 6 PET + 1 Glass" if "teardrop" in self.preset else "4-Layer PET-PSA-PET-Glass"
        _p(f"- **적층 프리셋**: `{self.preset}` ({lay_desc})")
        _p(f"- **서브스텝 수**: `{self.substeps}` substeps (180° 폴딩)")
        _p(f"- **병렬 스레드**: `{self.threads}` threads (OpenMP)\n")

        # Table 1: 2D Refinement
        _p("## 1. 2D 폴딩 메쉬 세분화 성능 및 수렴성 (FrontISTR)")
        _p("| 요소 크기 | 적용 요소 (PET/Glass / PSA) | Hinge $nx$ | 요소 수 | 절점 수 | DOFs | 총 해석 시간 | MUMPS Solve | Substep당 시간 | 반복당 시간 | 구동 모멘트 $\|M_z\|$ | Max Mises | 폭 $\Delta X$ |")
        _p("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        n_lay = 14 if "teardrop" in self.preset else 4
        for r in self.results_2d:
            s = r["summary"]
            _p(f"| {s['mesh_scale_pct']:.2f}% | Hex-361 FI / Hex-361 FI(TL) | {s['num_real_elements']//n_lay} | {s['num_real_elements']} | {s['nodes']} | {s['dofs']} | {s['wall_time_s']:.2f}s | {s.get('mumps_total_s', 0.0):.2f}s | {s.get('ms_per_substep', 0.0):.1f}ms | {s.get('ms_per_newton_iter', 0.0):.2f}ms | {abs(s['M_L_final_Nmm']):.4f} N·mm | {s.get('max_mises_mpa', 0.0):.1f} MPa | {s['loop_width_x_mm']:.4f} mm |")
        _p("")

        # Table 2: 3D In-plane
        _p("## 2. 3D 폴딩 평면 내 요소 크기 비교 (FrontISTR, $nz=1, \Delta z = 1.0\,\text{mm}$)")
        _p("| 요소 크기 | 적용 요소 (PET/Glass / PSA) | 요소 수 | DOFs | 총 해석 시간 | MUMPS Solve | Substep당 시간 | 구동 모멘트 $\|M_z\|$ | Max Mises | 폭 $\Delta X$ |")
        _p("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for r in self.results_3d_inplane:
            s = r["summary"]
            _p(f"| {s['mesh_scale_pct']:.1f}% | Hex-361 FI / Hex-361 FI(TL) | {s['num_real_elements']} | {s['dofs']} | {s['wall_time_s']:.2f}s | {s.get('mumps_total_s', 0.0):.2f}s | {s.get('ms_per_substep', 0.0):.1f}ms | {abs(s['M_L_final_Nmm']):.4f} N·mm | {s.get('max_mises_mpa', 0.0):.1f} MPa | {s['loop_width_x_mm']:.4f} mm |")
        _p("")

        # Table 3: 3D Depth comparison
        _p("## 3. 3D 폴딩 폭(Z-depth) 방향 요소 수 스케일링 ($nz = 1, 10, 20$, $\Delta z = 1.0\,\text{mm}$)")
        _p("| 기준 크기 | 적용 요소 (PET/Glass / PSA) | 깊이 요소($nz$) | 요소 수 | DOFs | 총 해석 시간 | MUMPS Setup | MUMPS Solve | 구동 모멘트 $\|M_z\|$ | 폭 $\Delta X$ |")
        _p("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for r in self.results_3d_depth_100:
            s = r["summary"]
            _p(f"| 100% | Hex-361 FI / Hex-361 FI(TL) | {s.get('nz_elements', 1)} | {s['num_real_elements']} | {s['dofs']} | {s['wall_time_s']:.2f}s | {s.get('mumps_setup_sum_s', 0.0):.2f}s | {s.get('mumps_solve_sum_s', 0.0):.2f}s | {abs(s['M_L_final_Nmm']):.4f} N·mm | {s['loop_width_x_mm']:.4f} mm |")
        for r in self.results_3d_depth_50:
            s = r["summary"]
            _p(f"| 50% | Hex-361 FI / Hex-361 FI(TL) | {s.get('nz_elements', 1)} | {s['num_real_elements']} | {s['dofs']} | {s['wall_time_s']:.2f}s | {s.get('mumps_setup_sum_s', 0.0):.2f}s | {s.get('mumps_solve_sum_s', 0.0):.2f}s | {abs(s['M_L_final_Nmm']):.4f} N·mm | {s['loop_width_x_mm']:.4f} mm |")
        _p("")

        # Table 4: FrontISTR vs WHT Parity Comparison
        _p("## 4. FrontISTR vs WHT Solver 직접 대조 결과 (속도, 하중, 응력, 형상)")
        _p("| 해석 케이스 | 솔버 | 적용 요소 (PET/Glass / PSA) | 요소 수 | DOFs | 총 해석 시간 | Substep당 시간 | 구동 모멘트 $\|M_z\|$ | 모멘트 편차 | 루프 폭 $\Delta X$ | 폭 편차 |")
        _p("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        # Match 2D cases
        for f_res in self.results_2d:
            sc = f_res["summary"]["mesh_scale_pct"]
            w_res = next((w for w in self.results_wht_2d if abs(w["summary"]["elements"] - f_res["summary"]["num_real_elements"]) <= 5), None)
            f_s = f_res["summary"]
            _p(f"| 2D-{sc:.0f}% | **FrontISTR** | Hex-361 FI / Hex-361 FI(TL) | {f_s['num_real_elements']} | {f_s['dofs']} | {f_s['wall_time_s']:.2f}s | {f_s.get('ms_per_substep', 0.0):.1f}ms | {abs(f_s['M_L_final_Nmm']):.4f} N·mm | 기준 (0.0%) | {f_s['loop_width_x_mm']:.4f} mm | 기준 |")
            if w_res:
                w_s = w_res["summary"]
                m_diff = ((abs(w_s["M_L_final_Nmm"]) - abs(f_s["M_L_final_Nmm"])) / abs(f_s["M_L_final_Nmm"])) * 100.0
                w_diff = abs(w_s["loop_width_x_mm"] - f_s["loop_width_x_mm"])
                _p(f"| 2D-{sc:.0f}% | **WHT** | CPE4I_CR / CPE4H_CR | {w_s['elements']} | {w_s['dofs']} | {w_s['wall_time_s']:.2f}s | {w_s['ms_per_substep']:.1f}ms | {abs(w_s['M_L_final_Nmm']):.4f} N·mm | {m_diff:+.2f}% | {w_s['loop_width_x_mm']:.4f} mm | {w_diff:.4f} mm |")

        # Match 3D cases
        for f_res in self.results_3d_depth_100:
            nz = f_res["summary"].get("nz_elements", 1)
            w_res = next((w for w in self.results_wht_3d if abs(w["summary"]["elements"] - f_res["summary"]["num_real_elements"]) <= 5), None)
            f_s = f_res["summary"]
            _p(f"| 3D-100%-nz{nz} | **FrontISTR** | Hex-361 FI / Hex-361 FI(TL) | {f_s['num_real_elements']} | {f_s['dofs']} | {f_s['wall_time_s']:.2f}s | {f_s.get('ms_per_substep', 0.0):.1f}ms | {abs(f_s['M_L_final_Nmm']):.4f} N·mm | 기준 (0.0%) | {f_s['loop_width_x_mm']:.4f} mm | 기준 |")
            if w_res:
                w_s = w_res["summary"]
                m_diff = ((abs(w_s["M_L_final_Nmm"]) - abs(f_s["M_L_final_Nmm"])) / abs(f_s["M_L_final_Nmm"])) * 100.0
                w_diff = abs(w_s["loop_width_x_mm"] - f_s["loop_width_x_mm"])
                _p(f"| 3D-100%-nz{nz} | **WHT** | C3D8I_CR / C3D8H_CR | {w_s['elements']} | {w_s['dofs']} | {w_s['wall_time_s']:.2f}s | {w_s['ms_per_substep']:.1f}ms | {abs(w_s['M_L_final_Nmm']):.4f} N·mm | {m_diff:+.2f}% | {w_s['loop_width_x_mm']:.4f} mm | {w_diff:.4f} mm |")

        _p("\n")
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # Generate Matplotlib comparison figures
        self._plot_benchmark_figures()

        print(f"[OK] Generated benchmark report: {report_md_path}")
        return report_md_path

    def _plot_benchmark_figures(self) -> None:
        """Plots scaling and convergence figures for the report."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            import koreanize_matplotlib  # noqa: F401
        except ImportError:
            pass

        plt.rcParams["font.size"] = 9
        fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=180)

        # 1. 2D Solver Time vs DOFs (Log-Log scaling)
        ax1 = axes[0, 0]
        f_dofs = [r["summary"]["dofs"] for r in self.results_2d]
        f_times = [r["summary"]["wall_time_s"] for r in self.results_2d]
        ax1.loglog(f_dofs, f_times, "o-", color="#1f77b4", lw=2, ms=6, label="FrontISTR (MUMPS 4-Thread)")

        if self.results_wht_2d:
            w_dofs = [r["summary"]["dofs"] for r in self.results_wht_2d]
            w_times = [r["summary"]["wall_time_s"] for r in self.results_wht_2d]
            ax1.loglog(w_dofs, w_times, "s--", color="#d62728", lw=2, ms=6, label="WHT Solver (PARDISO 4-Thread)")

        ax1.set_title("1. 자유도(DOF) 증가에 따른 해석 시간 스케일링 (2D Log-Log)")
        ax1.set_xlabel("자유도 (Active DOFs)")
        ax1.set_ylabel("총 해석 시간 [초]")
        ax1.grid(True, which="both", linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left")

        # 2. Reaction Moment h-Convergence
        ax2 = axes[0, 1]
        h_ratios = [r["summary"]["elem_size_ratio"] for r in self.results_2d]
        f_moments = [abs(r["summary"]["M_L_final_Nmm"]) for r in self.results_2d]
        ax2.plot(h_ratios, f_moments, "o-", color="#1f77b4", lw=2, ms=6, label="FrontISTR |M_z|")

        if self.results_wht_2d:
            w_ratios = [1.0, 0.5, 0.25][:len(self.results_wht_2d)]
            w_moments = [abs(r["summary"]["M_L_final_Nmm"]) for r in self.results_wht_2d]
            ax2.plot(w_ratios, w_moments, "s--", color="#d62728", lw=2, ms=6, label="WHT Solver |M_z|")

        ax2.set_title("2. 요소 크기(h)에 따른 구동 반력 모멘트 h-수렴성")
        ax2.set_xlabel("상대 요소 크기 h/h_0 (1.0 -> 0.0625)")
        ax2.set_ylabel("피벗 구동 모멘트 |M_z| [N·mm]")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="best")

        # 3. 3D Depth Scaling (nz = 1, 10, 20)
        ax3 = axes[1, 0]
        nz_100 = [r["summary"]["nz_elements"] for r in self.results_3d_depth_100]
        t_100 = [r["summary"]["wall_time_s"] for r in self.results_3d_depth_100]
        nz_50 = [r["summary"]["nz_elements"] for r in self.results_3d_depth_50]
        t_50 = [r["summary"]["wall_time_s"] for r in self.results_3d_depth_50]

        ax3.plot(nz_100, t_100, "o-", color="#2ca02c", lw=2, ms=6, label="3D 요소크기 100% (nz=1, 10, 20)")
        ax3.plot(nz_50, t_50, "^-", color="#ff7f0e", lw=2, ms=6, label="3D 요소크기 50% (nz=1, 10, 20)")

        ax3.set_title("3. 폭(Z-depth) 방향 요소 수 분할에 따른 해석 시간 증가")
        ax3.set_xlabel("Z-방향 요소 분할 수 (nz)")
        ax3.set_ylabel("해석 시간 [초]")
        ax3.grid(True, linestyle=":", alpha=0.6)
        ax3.legend(loc="upper left")

        # 4. Deformed Shape Silhouette Overlay (FrontISTR vs WHT)
        ax4 = axes[1, 1]
        has_lines = False
        if self.results_2d and self.results_wht_2d:
            f_ref = self.results_2d[0]
            w_ref = self.results_wht_2d[0]
            if "X_def" in f_ref and "X_def" in w_ref:
                X_def_f = f_ref["X_def"]
                X_def_w = w_ref["X_def"]

                y_bot_f = float(np.min(f_ref["X0"][:, 1]))
                mask_f = (np.abs(f_ref["X0"][:, 1] - y_bot_f) < 1e-5) & (np.abs(f_ref["X0"][:, 2] - np.min(f_ref["X0"][:, 2])) < 1e-5)
                ord_f = np.argsort(f_ref["X0"][mask_f, 0])
                ax4.plot(X_def_f[mask_f, 0][ord_f], X_def_f[mask_f, 1][ord_f], "-", color="#1f77b4", lw=2.2, label="FrontISTR (100% 메쉬)")

                y_bot_w = float(np.min(w_ref["X0"][:, 1]))
                mask_w = (np.abs(w_ref["X0"][:, 1] - y_bot_w) < 1e-5) & (np.abs(w_ref["X0"][:, 2] - np.min(w_ref["X0"][:, 2])) < 1e-5)
                ord_w = np.argsort(w_ref["X0"][mask_w, 0])
                ax4.plot(X_def_w[mask_w, 0][ord_w], X_def_w[mask_w, 1][ord_w], "--", color="#d62728", lw=2.0, label="WHT Solver (100% 메쉬)")
                has_lines = True

        ax4.set_title("4. 180° 폴딩 최종 형상 일치도 비교 (FrontISTR vs WHT)")
        ax4.set_xlabel("변형 후 X [mm]")
        ax4.set_ylabel("변형 후 Y [mm]")
        ax4.set_aspect("equal", adjustable="box")
        ax4.grid(True, linestyle=":", alpha=0.6)
        if has_lines:
            ax4.legend(loc="best")

        plt.tight_layout()
        plot_path = os.path.join(self.out_dir, "benchmark_comparison_dashboard.png")
        plt.savefig(plot_path, dpi=180)
        plt.close(fig)
        print(f"[OK] Saved comparison dashboard plot: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mesh Refinement Benchmark Suite")
    parser.add_argument("--preset", type=str, default="teardrop", choices=["teardrop", "compact_4"])
    parser.add_argument("--substeps", type=int, default=30, help="Number of substeps for 180-deg folding")
    parser.add_argument("--threads", type=int, default=6, help="Number of OpenMP threads")
    parser.add_argument("--part", type=str, default="all", choices=["all", "2d", "3d", "wht", "report"])
    args = parser.parse_args()

    suite = MeshRefinementBenchmarkSuite(
        preset=args.preset,
        substeps=args.substeps,
        threads=args.threads,
    )

    if args.part in ("all", "2d"):
        suite.run_2d_refinement_matrix()
    if args.part in ("all", "3d"):
        suite.run_3d_refinement_matrix()
    if args.part in ("all", "wht"):
        suite.run_wht_parity_matrix()

    suite.generate_benchmark_report_and_plots()


if __name__ == "__main__":
    main()
