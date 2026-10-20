"""
run_20deg_parity_study.py
=========================
14-Layer Full-Stack Foldable Display 20-Degree Early Folding Parity Study.
Directly compares FrontISTR (Ground Truth, 6-Core MUMPS) vs WHT Solver (6-Core oneMKL PARDISO)
at theta = 20.0 degrees with Unit Depth / Width Normalization (bar_M = M_z / W_z, bar_F = F / W_z).

Compares:
1. FrontISTR 2D (Plane-Strain Hex 361 FI)
2. FrontISTR 3D (Continuum Hex 361 FI, nz=1, W_z=1.0mm)
3. WHT 2D (CPE4I + CPE4H, Pure Elastic PET)
4. WHT 2D (CPE4I + CPE4H, Elastoplastic PET)
5. WHT 2D (CPE4R + CPE4H, Pure Elastic PET - Opt2-H)
6. WHT 3D (C3D8I + C3D8H, nz=1, W_z=1.0mm, Pure Elastic PET)
7. WHT 3D (C3D8R + C3D8H, nz=1, W_z=1.0mm, Pure Elastic PET - Opt2-H)

Generates:
- 1:1 Comparison Table with Unit-Depth normalized moments and reaction forces.
- High-resolution comparison plot saved in dev_log/figures/.
"""

import os
import sys
import math
import time
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koreanize_matplotlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig, MaterialSpec
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from example_disp_fold.run_fistr_teardrop import run_fistr_case
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


def smoothstep_amp(s: float) -> float:
    s_clamped = max(0.0, min(1.0, s))
    return s_clamped * s_clamped * (3.0 - 2.0 * s_clamped)


def run_wht_20deg(
    dim: str = "2D",
    preset: str = "teardrop",
    elem_case: str = "2D-Opt1",
    pet_mat_type: str = "ELASTIC",
    target_deg: float = 20.0,
    substeps: int = 5,
    threads: int = 6,
    width_z: float = 1.0,
    nz: int = 1,
) -> Dict[str, Any]:
    """Runs WHT Solver up to target_deg (e.g. 20.0 deg) and returns step metrics."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    config = UnifiedFoldModelConfig(preset=preset, dim=dim)
    config.simulation_case.theta_max_deg = float(target_deg)
    if dim == "3D":
        config.geometry.width_z = float(width_z)

    # Configure materials dict
    materials = {
        1: {"E": 4000.0, "nu": 0.30},
        2: {"E": 0.05, "nu": 0.499},
        3: {"E": 70000.0, "nu": 0.22},
        4: {"E": 200000.0, "nu": 0.30},
    }
    if pet_mat_type == "ELASTIC_PLASTIC":
        materials[1]["mat_type"] = "ELASTIC_PLASTIC"
        materials[1]["yield_stress"] = 80.0
        materials[1]["hardening_modulus"] = 400.0

    t0_setup = time.perf_counter()
    mesher = GmshMeshGenerator(config)
    mesh, meta = mesher.build()

    is_opt2_h = "Opt2" in elem_case
    if is_opt2_h:
        # Override element types to reduced integration + u-P
        for el in mesh.elements.values():
            if dim == "2D":
                if el.pid in (1, 3):  # PET, Glass
                    el.elem_type = "CPE4R"
                elif el.pid == 2:     # PSA
                    el.elem_type = "CPE4H"
                elif el.pid == 4:     # Plate
                    el.elem_type = "CPE4"
            else:
                if el.pid in (1, 3):  # PET, Glass
                    el.elem_type = "C3D8R"
                elif el.pid == 2:     # PSA
                    el.elem_type = "C3D8H"
                elif el.pid == 4:     # Plate
                    el.elem_type = "C3D8"

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

    theta_max_rad = math.radians(target_deg)
    dt = 1.0 / float(substeps)

    history_M_L = []
    history_M_R = []
    history_Fx_L = []
    history_Fx_R = []
    history_Fy_L = []
    history_Fy_R = []
    history_theta = []
    total_newton_iters = 0

    print(f"[*] Running WHT Solver ({dim}): {elem_case} (PET={pet_mat_type}, target={target_deg} deg)")
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
            tol=1e-4,
            use_anderson_accel=True,
            anderson_m=4,
        )
        total_newton_iters += iters
        theta_prev = theta_curr

        if not converged:
            print(f"[!] WHT solver did not converge at step {step}/{substeps}")
            break

        # Compute reaction forces & moments using compute_section_reactions
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
        history_Fy_L.append(float(reac_L["Fy"]))
        history_Fy_R.append(float(reac_R["Fy"]))

    solve_time = time.perf_counter() - t0_solve
    wall_time = setup_time + solve_time

    # Normalize by physical width W_z
    eff_wz = float(width_z) if dim == "3D" else 1.0

    M_L_final = history_M_L[-1] if history_M_L else 0.0
    M_R_final = history_M_R[-1] if history_M_R else 0.0
    bar_M_L = M_L_final / eff_wz
    bar_M_R = M_R_final / eff_wz
    bar_avg_M = 0.5 * (abs(bar_M_L) + abs(bar_M_R))
    symm_err = (abs(abs(bar_M_L) - abs(bar_M_R)) / max(bar_avg_M, 1e-9)) * 100.0

    Fx_L_final = history_Fx_L[-1] if history_Fx_L else 0.0
    Fx_R_final = history_Fx_R[-1] if history_Fx_R else 0.0
    bar_Fx_L = Fx_L_final / eff_wz
    bar_Fx_R = Fx_R_final / eff_wz

    Fy_L_final = history_Fy_L[-1] if history_Fy_L else 0.0
    Fy_R_final = history_Fy_R[-1] if history_Fy_R else 0.0
    bar_Fy_L = Fy_L_final / eff_wz
    bar_Fy_R = Fy_R_final / eff_wz

    # Deformed apex coordinates
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

    return {
        "case_name": f"WHT_{dim}_{elem_case}_{pet_mat_type}",
        "solver": "WHT",
        "dim": dim,
        "elem_case": elem_case,
        "pet_mat_type": pet_mat_type,
        "width_z": eff_wz,
        "nodes": len(mesh.nodes),
        "elements": len(mesh.elements),
        "dofs": solver.num_dofs,
        "substeps": substeps,
        "total_iters": total_newton_iters,
        "wall_time_s": wall_time,
        "solve_time_s": solve_time,
        "target_deg": target_deg,
        "bar_M_L": bar_M_L,
        "bar_M_R": bar_M_R,
        "bar_avg_M": bar_avg_M,
        "symm_err_pct": symm_err,
        "bar_Fx_L": bar_Fx_L,
        "bar_Fx_R": bar_Fx_R,
        "bar_Fy_L": bar_Fy_L,
        "bar_Fy_R": bar_Fy_R,
        "loop_width_x_mm": loop_width_x,
        "apex_y_mm": apex_y,
        "history_theta": history_theta,
        "history_bar_M_L": [m / eff_wz for m in history_M_L],
        "history_bar_M_R": [m / eff_wz for m in history_M_R],
        "history_bar_Fx_L": [f / eff_wz for f in history_Fx_L],
        "history_bar_Fx_R": [f / eff_wz for f in history_Fx_R],
    }


def run_fistr_20deg(
    dim: str = "2D",
    preset: str = "teardrop",
    target_deg: float = 20.0,
    substeps: int = 5,
    threads: int = 6,
    width_z: float = 1.0,
) -> Dict[str, Any]:
    """Runs FrontISTR up to target_deg (e.g. 20.0 deg) and returns step metrics."""
    out_dir = os.path.join(REPO_ROOT, "example_disp_fold", "results_20deg_parity")
    case_name = f"FISTR_{dim}_target{int(target_deg)}deg_wz{int(width_z)}"

    res = run_fistr_case(
        dim=dim,
        preset=preset,
        psa_form361="FI",
        solid_form361="FI",
        substeps=substeps,
        threads=threads,
        out_base_dir=out_dir,
        case_name_override=case_name,
        width_z=width_z if dim == "3D" else 1.0,
        theta_max_deg=target_deg,
    )
    s = res["summary"]

    eff_wz = float(width_z) if dim == "3D" else 1.0
    bar_M_L = s["M_L_final_Nmm"] / eff_wz
    bar_M_R = s["M_R_final_Nmm"] / eff_wz
    bar_avg_M = 0.5 * (abs(bar_M_L) + abs(bar_M_R))
    symm_err = s["moment_symmetry_err_pct"]

    bar_Fx_L = s["Fx_L_final_N"] / eff_wz
    bar_Fx_R = s["Fx_R_final_N"] / eff_wz

    return {
        "case_name": case_name,
        "solver": "FrontISTR",
        "dim": dim,
        "elem_case": "Hex-361 FI",
        "pet_mat_type": "ELASTIC",
        "width_z": eff_wz,
        "nodes": s["num_real_nodes"],
        "elements": s["num_real_elements"],
        "dofs": s["num_real_nodes"] * 3,
        "substeps": substeps,
        "total_iters": s.get("total_newton_iters", substeps * 3),
        "wall_time_s": s["wall_time_s"],
        "solve_time_s": s.get("fistr_solve_s", s["wall_time_s"]),
        "target_deg": target_deg,
        "bar_M_L": bar_M_L,
        "bar_M_R": bar_M_R,
        "bar_avg_M": bar_avg_M,
        "symm_err_pct": symm_err,
        "bar_Fx_L": bar_Fx_L,
        "bar_Fx_R": bar_Fx_R,
        "bar_Fy_L": 0.0,
        "bar_Fy_R": 0.0,
        "loop_width_x_mm": s["loop_width_x_mm"],
        "apex_y_mm": s["apex_y_mm"],
        "max_mises_mpa": s.get("max_mises_mpa", 0.0),
        "history_theta": s["history_theta_deg"],
        "history_bar_M_L": [m / eff_wz for m in s["history_M_L"]],
        "history_bar_M_R": [m / eff_wz for m in s["history_M_R"]],
        "history_bar_Fx_L": [f / eff_wz for f in s["history_Fx_L"]],
        "history_bar_Fx_R": [f / eff_wz for f in s["history_Fx_R"]],
    }


def main():
    print("=" * 110)
    print(" 14-Layer Full-Stack 20-Degree Early Folding Parity Study")
    print(" Ground Truth: FrontISTR (6-Core MUMPS) vs WHT Solver (6-Core oneMKL PARDISO)")
    print(" Unit Depth Normalization: bar_M = M / W_z [N*mm/mm], bar_F = F / W_z [N/mm]")
    print("=" * 110)

    target_deg = 20.0
    substeps = 5
    threads = 6
    preset = "teardrop"

    results = []

    # 1. FrontISTR 2D (Ground Truth)
    print("\n>>> [1/6] Running FrontISTR 2D (Ground Truth)...")
    try:
        f_2d = run_fistr_20deg(dim="2D", preset=preset, target_deg=target_deg, substeps=substeps, threads=threads)
        results.append(f_2d)
    except Exception as e:
        print(f"[!] FrontISTR 2D failed: {e}")

    # 2. FrontISTR 3D (Ground Truth nz=1, W_z=1.0mm)
    print("\n>>> [2/6] Running FrontISTR 3D nz=1 (Ground Truth)...")
    try:
        f_3d = run_fistr_20deg(dim="3D", preset=preset, target_deg=target_deg, substeps=substeps, threads=threads, width_z=1.0)
        results.append(f_3d)
    except Exception as e:
        print(f"[!] FrontISTR 3D failed: {e}")

    # 3. WHT 2D (Opt1: CPE4I + CPE4H, Pure Elastic PET - Parity Match)
    print("\n>>> [3/6] Running WHT 2D-Opt1 (CPE4I + CPE4H)...")
    try:
        w_2d_elast = run_wht_20deg(
            dim="2D", preset=preset, elem_case="2D-Opt1", pet_mat_type="ELASTIC",
            target_deg=target_deg, substeps=substeps, threads=threads
        )
        results.append(w_2d_elast)
    except Exception as e:
        print(f"[!] WHT 2D-Opt1 (Elastic) failed: {e}")

    # 4. WHT 2D (Opt2-H: CPE4R + CPE4H, Pure Elastic PET - Fast Alternative)
    print("\n>>> [4/6] Running WHT 2D-Opt2-H (CPE4R + CPE4H, Reduced Integration)...")
    try:
        w_2d_opt2 = run_wht_20deg(
            dim="2D", preset=preset, elem_case="2D-Opt2-H", pet_mat_type="ELASTIC",
            target_deg=target_deg, substeps=substeps, threads=threads
        )
        results.append(w_2d_opt2)
    except Exception as e:
        print(f"[!] WHT 2D-Opt2-H failed: {e}")

    # 5. WHT 3D (Opt1: C3D8I + C3D8H, nz=1, Pure Elastic PET)
    print("\n>>> [5/6] Running WHT 3D-Opt1 (C3D8I + C3D8H, nz=1)...")
    try:
        w_3d_opt1 = run_wht_20deg(
            dim="3D", preset=preset, elem_case="3D-Opt1", pet_mat_type="ELASTIC",
            target_deg=target_deg, substeps=substeps, threads=threads, width_z=1.0, nz=1
        )
        results.append(w_3d_opt1)
    except Exception as e:
        print(f"[!] WHT 3D-Opt1 failed: {e}")

    # 6. WHT 3D (Opt2-H: C3D8R + C3D8H, nz=1, Pure Elastic PET - Fast Alternative)
    print("\n>>> [6/6] Running WHT 3D-Opt2-H (C3D8R + C3D8H, nz=1)...")
    try:
        w_3d_opt2 = run_wht_20deg(
            dim="3D", preset=preset, elem_case="3D-Opt2-H", pet_mat_type="ELASTIC",
            target_deg=target_deg, substeps=substeps, threads=threads, width_z=1.0, nz=1
        )
        results.append(w_3d_opt2)
    except Exception as e:
        print(f"[!] WHT 3D-Opt2-H failed: {e}")

    # Summary Table
    print("\n" + "=" * 130)
    print(f" 20-DEGREE EARLY FOLDING PARITY SUMMARY TABLE (theta = {target_deg} deg)")
    print("=" * 130)
    print(f"{'Case Name':<28} | {'Solver':<8} | {'Dim':<3} | {'Element Case':<14} | {'PET Mat':<8} | {'Time (s)':<8} | {'bar_M_L (N)':<12} | {'bar_M_R (N)':<12} | {'Symm Err %':<10} | {'bar_Fx_L (N/mm)':<15}")
    print("-" * 130)

    fistr_ref_M = None
    for r in results:
        if r["solver"] == "FrontISTR" and fistr_ref_M is None:
            fistr_ref_M = r["bar_avg_M"]

        print(
            f"{r['case_name']:<28} | "
            f"{r['solver']:<8} | "
            f"{r['dim']:<3} | "
            f"{r['elem_case']:<14} | "
            f"{r['pet_mat_type']:<8} | "
            f"{r['wall_time_s']:<8.2f} | "
            f"{r['bar_M_L']:<+12.4f} | "
            f"{r['bar_M_R']:<+12.4f} | "
            f"{r['symm_err_pct']:<10.4f} | "
            f"{r['bar_Fx_L']:<+15.4f}"
        )
    print("=" * 130)

    # Save summary json
    out_dir = os.path.join(REPO_ROOT, "example_disp_fold", "results_20deg_parity")
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "parity_study_20deg_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Summary JSON saved to: {summary_path}")

    # Plot 4-panel comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "FISTR_2D": "#1f77b4",
        "FISTR_3D": "#d62728",
        "WHT_2D_2D-Opt1_ELASTIC": "#2ca02c",
        "WHT_2D_2D-Opt1_ELASTIC_PLASTIC": "#ff7f0e",
        "WHT_2D_2D-Opt2-H_ELASTIC": "#9467bd",
        "WHT_3D_3D-Opt1_ELASTIC": "#8c564b",
        "WHT_3D_3D-Opt2-H_ELASTIC": "#e377c2",
    }

    # Panel 1: Unit Moment vs Theta
    for r in results:
        c = colors.get(r["case_name"], "#333333")
        th = r["history_theta"]
        m_vals = [abs(m) for m in r["history_bar_M_L"]]
        label = f"{r['solver']} {r['dim']} ({r['elem_case']}, {r['pet_mat_type']})"
        ls = "-" if r["solver"] == "FrontISTR" else "--"
        ax1.plot(th, m_vals, marker="o", ms=4, lw=1.8, color=c, linestyle=ls, label=label)
    ax1.set_xlabel("회전 각도 θ [deg]", fontsize=10)
    ax1.set_ylabel("단위폭 구동 모멘트 |bar_M_L| [N·mm/mm]", fontsize=10)
    ax1.set_title("1. 단위폭 구동 모멘트 진행 곡선 (0° -> 20°)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8, loc="best")

    # Panel 2: Unit Horizontal Reaction Force Fx vs Theta
    for r in results:
        c = colors.get(r["case_name"], "#333333")
        th = r["history_theta"]
        fx_vals = [abs(f) for f in r["history_bar_Fx_L"]]
        label = f"{r['solver']} {r['dim']} ({r['elem_case']})"
        ls = "-" if r["solver"] == "FrontISTR" else "--"
        ax2.plot(th, fx_vals, marker="s", ms=4, lw=1.8, color=c, linestyle=ls, label=label)
    ax2.set_xlabel("회전 각도 θ [deg]", fontsize=10)
    ax2.set_ylabel("단위폭 수평 반력 |bar_Fx_L| [N/mm]", fontsize=10)
    ax2.set_title("2. 단위폭 수평 반력 진행 곡선 (0° -> 20°)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8, loc="best")

    # Panel 3: Wall Time Comparison Bar Chart
    case_labels = [f"{r['solver']}\n{r['dim']}-{r['elem_case']}\n({r['pet_mat_type'][:4]})" for r in results]
    times = [r["wall_time_s"] for r in results]
    bar_cols = [colors.get(r["case_name"], "#333333") for r in results]
    bars = ax3.bar(range(len(results)), times, color=bar_cols, width=0.55, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, h + 0.05 * max(times), f"{h:.2f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax3.set_xticks(range(len(results)))
    ax3.set_xticklabels(case_labels, fontsize=8)
    ax3.set_ylabel("총 연산 시간 [s]", fontsize=10)
    ax3.set_title("3. 20° 해석 총 소요 시간 비교", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", axis="y", alpha=0.6)

    # Panel 4: Moment Deviation % from FrontISTR Ground Truth
    if fistr_ref_M is not None and fistr_ref_M > 1e-6:
        devs = [((r["bar_avg_M"] - fistr_ref_M) / fistr_ref_M) * 100.0 for r in results]
        bars4 = ax4.bar(range(len(results)), devs, color=bar_cols, width=0.55, edgecolor="black", linewidth=0.8)
        for bar in bars4:
            h = bar.get_height()
            va = "bottom" if h >= 0 else "top"
            ax4.text(bar.get_x() + bar.get_width() / 2.0, h, f"{h:+.1f}%", ha="center", va=va, fontsize=8, fontweight="bold")
        ax4.axhline(0, color="black", linestyle="-", lw=1.0)
        ax4.set_xticks(range(len(results)))
        ax4.set_xticklabels(case_labels, fontsize=8)
        ax4.set_ylabel("FrontISTR 대비 모멘트 편차 [%]", fontsize=10)
        ax4.set_title(f"4. FrontISTR 정답 대비 모멘트 편차 (기준: {fistr_ref_M:.4f} N)", fontsize=11, fontweight="bold")
        ax4.grid(True, linestyle=":", axis="y", alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(REPO_ROOT, "dev_log", "figures", "parity_study_20deg_comparison.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[OK] Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
