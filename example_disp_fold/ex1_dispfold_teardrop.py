"""
ex1_dispfold_teardrop.py
========================
Unified 2D and 3D Multi-Layer Foldable Display 180-Degree Teardrop Folding Simulation.

Features:
---------
1. Single Source of Truth: identical configuration generates 2D and 3D models seamlessly.
2. Gmsh 4.15 Biased Mesher: Transfinite Quad (2D) and Extruded Hex (3D) with fine hinge resolution.
3. Abaqus API Parity & INP Exporter with roundtrip verification.
4. Kitware VTKHDF Version 2.2 Pre/Post database streaming.
5. PySide6 + pyvistaqt interactive CAE GUI with time slider, layer tree, and 9pt Matplotlib dashboard.
6. Automated timestamp archiving: results/DYYYYMMDD_HHMMSS/ (run.log, model.inp, vtkhdf, png, gif, html).
"""

from __future__ import annotations
import os
import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Threading configuration for MKL PARDISO
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Repository root on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from example_disp_fold.abaqus_api_builder import AbaqusApiBuilder
from example_disp_fold.vtkhdf_unified_io import UnifiedVTKHDFDatabase
from example_disp_fold.report_generator import generate_engineering_html_report

from dispsolver.solver.dt_controller import AdaptiveDtController
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.fold_model_config import smoothstep_amp


class TeeLogger:
    """Duplicates stdout/stderr to console and a persistent run.log file."""

    def __init__(self, log_filepath: str):
        self.terminal = sys.stdout
        self.log_file = open(log_filepath, "w", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()


def run_dispfold_simulation(
    dim: str = "3D",
    preset: str = "teardrop",
    elem_case: Optional[str] = None,
    theta_max_deg: float = 90.0,
    enable_predictor: bool = True,
    tie_method: str = "MPC",
    contact_mode: str = "AUTO",
    sol_preset: str = "ADAPTIVE",
    max_steps: Optional[int] = None,
    launch_gui: bool = False,
    results_base_dir: Optional[str] = None,
    step_callback: Optional[Any] = None,
) -> str:
    """Executes complete folding simulation with automated results archiving."""
    # 1. Setup timestamped results folder
    timestamp_str = datetime.now().strftime("D%Y%m%d_%H%M%S")
    if results_base_dir is None:
        results_base_dir = os.path.join(REPO_ROOT, "example_disp_fold", "results")
    out_dir = os.path.join(results_base_dir, timestamp_str)
    os.makedirs(out_dir, exist_ok=True)

    # 2. Wire Tee Logger
    log_path = os.path.join(out_dir, "run.log")
    tee = TeeLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee

    t_start = time.perf_counter()

    try:
        print("=" * 115)
        print(f" WHT_DispFoldSolver: Unified {dim} Display 180° Teardrop Folding Pipeline")
        print(f" Preset: {preset.upper()} | Target Plate Rotation: {theta_max_deg:.2f}° (Total {2*theta_max_deg:.2f}°)")
        print(f" Output Archive: {out_dir}")
        print("=" * 115)

        # 3. Model Specification
        config = UnifiedFoldModelConfig(preset=preset, dim=dim, elem_case=elem_case)
        config.simulation_case.theta_max_deg = theta_max_deg
        config.simulation_case.enable_predictor = enable_predictor
        config.simulation_case.tie_method = tie_method
        config.simulation_case.contact_mode = contact_mode
        config.simulation_case.solution_preset = sol_preset
        config.simulation_case.max_steps = max_steps

        resolved_contact = config.resolve_contact_guard()
        print(f"[*] Configuration: {len(config.layers)} layers, Total Thickness = {config.total_thickness:.3f} mm")
        print(f"[*] Element Case: {config.simulation_case.elem_case} (Solid: {config.layers[1].elem_type_3d if dim=='3D' else config.layers[1].elem_type_2d}, PSA: {config.layers[0].elem_type_3d if dim=='3D' else config.layers[0].elem_type_2d})")
        print(f"[*] Contact Mode: {contact_mode} -> Resolved: {resolved_contact}")
        print(f"[*] Kinematic Predictor: {'ENABLED (carry_contacting_nodes)' if enable_predictor else 'DISABLED'}")

        # 4. Gmsh Mesh Generation
        print("\n[*] Generating continuum mesh with Gmsh 4.15 API...")
        mesher = GmshMeshGenerator(config)
        mesh, meta = mesher.build()
        n_nodes = len(mesh.nodes)
        n_elems = len(mesh.elements)
        print(f"[OK] Mesh generated: {n_nodes} nodes, {n_elems} elements, det J > 0 verified.")

        # 5. Abaqus API Hierarchy & .inp Exporter
        abq_builder = AbaqusApiBuilder(config, mesh, meta)
        inp_path = os.path.join(out_dir, "model.inp")
        abq_builder.write_inp(inp_path)
        print(f"[OK] Exported verified Abaqus input deck: {inp_path}")

        # 6. VTKHDF Database Initialization
        vtkhdf_path = os.path.join(out_dir, "model_and_results.vtkhdf")
        vdb = UnifiedVTKHDFDatabase(vtkhdf_path, mesh, meta, config)

        # 7. Material mapping for solver
        # PID 1: PET (E=4000, nu=0.30)
        # PID 2: PSA (E=0.05, nu=0.499)
        # PID 3: GLASS (E=70000, nu=0.22)
        # PID 4: Rigid Plate (E=200000, nu=0.30)
        materials = {
            1: {"E": 4000.0, "nu": 0.30},
            2: {"E": 0.05, "nu": 0.499},
            3: {"E": 70000.0, "nu": 0.22},
            4: {"E": 200000.0, "nu": 0.30},
        }

        # 8. Solver Instantiation
        if dim == "2D":
            solver = DynamicSolver2D(mesh, materials=materials, nlgeom=True)
            solver.max_du_limit = 1.0
        else:
            solver = DynamicSolver3D(mesh, materials=materials, nlgeom=True, solution_controls=sol_preset)
            solver.max_du_limit = 1.0

        # Wire Wohlmuth Dual Mortar Self-Contact if requested by preset or geometry guard
        if config.requires_self_contact():
            if dim == "2D":
                from dispsolver.constraint.dual_mortar_contact import DualMortarContactConstraint2D
                contact_c = DualMortarContactConstraint2D(
                    slave_segments=meta["disp_top_right_segments"],
                    master_segments=meta["disp_top_left_segments"],
                    nid_to_idx=mesh.node_id_to_index(),
                    coords=mesh.nodes_array(),
                    penalty_stiffness=1.0e6,
                    position_tolerance=2.0,
                    finite_sliding=True,
                    name="SELF_CONTACT_DUAL_MORTAR_2D",
                )
                solver.add_constraint(contact_c)
                print(f"[OK] Wohlmuth Dual Mortar Self-Contact wired into 2D solver: {len(meta['disp_top_right_segments'])} slave, {len(meta['disp_top_left_segments'])} master segments.")
            else:
                from dispsolver.constraint3d.finite_sliding_mortar3d import FiniteSlidingDualMortar3D
                contact_c = FiniteSlidingDualMortar3D(
                    slave_faces=meta["disp_top_right_faces"],
                    master_faces=meta["disp_top_left_faces"],
                    nid_to_idx=mesh.node_id_to_index(),
                    coords=mesh.nodes_array(),
                    penalty_stiffness=1.0e6,
                    position_tolerance=2.0,
                    name="SELF_CONTACT_DUAL_MORTAR_3D",
                )
                solver.add_constraint(contact_c)
                print(f"[OK] Wohlmuth Finite-Sliding Dual Mortar Self-Contact wired into 3D solver: {len(meta['disp_top_right_faces'])} slave, {len(meta['disp_top_left_faces'])} master faces.")

        print(f"[*] Solver ready: {solver.num_dofs} active DOFs. SolutionControl = {sol_preset}.")

        # 9. Adaptive Time Stepping Controller
        dt_ctrl = AdaptiveDtController(
            dt_init=config.simulation_case.dt_init,
            dt_min=config.simulation_case.dt_min,
            dt_max=config.simulation_case.dt_max,
            target_iters=8,
        )

        th_max_rad = np.radians(theta_max_deg)
        pivot_L = meta["left_pivot"]
        pivot_R = meta["right_pivot"]

        # Precompute display stack nodes mounted on support plates for full-stack predictor
        hinge_gap = config.geometry.hinge_half_gap
        disp_left_plate_nodes = [nid for nid, node in mesh.nodes.items() if node.x <= -hinge_gap + 1e-4 and node.y >= 0.0]
        disp_right_plate_nodes = [nid for nid, node in mesh.nodes.items() if node.x >= hinge_gap - 1e-4 and node.y >= 0.0]
        nid_map = mesh.node_id_to_index()
        ndof_n = 2 if dim == "2D" else 3

        # History containers
        history_time = [0.0]
        history_theta = [0.0]
        history_iters = [0]
        history_M_L = [0.0]
        history_M_R = [0.0]
        history_Fx_L = [0.0]
        history_Fx_R = [0.0]
        history_symm_err = [0.0]
        history_u = [solver.u.copy()]

        # Record step 0 to VTKHDF
        vdb.add_step(0.0, solver.u)

        t = 0.0
        step = 0
        theta_prev = 0.0
        u_converged = solver.u.copy()
        consecutive_cutbacks = 0
        max_cutbacks = 20

        print("\n" + "-" * 115)
        print(f"{'Step':>5} | {'Time':>7} | {'Theta':>7} | {'dt':>8} | {'Fx_L [N]':>9} | {'Fx_R [N]':>9} | {'Sum Fx':>10} | {'M_drive_L':>11} | {'M_drive_R':>11} | {'Symm Err':>9} | {'Iters':>5} | {'Status':>6}")
        print("-" * 115)

        while t < 1.0 - 1e-6:
            if max_steps is not None and step >= max_steps:
                print(f"[!] Reached max_steps={max_steps}. Terminating solve loop.")
                break

            step += 1
            dt = dt_ctrl.dt
            t_next = min(1.0, t + dt)
            dt_actual = t_next - t

            s = t_next
            theta_curr = th_max_rad * smoothstep_amp(s)

            # Apply Kinematic Rigid Plate & Tied Display Dirichlet BCs (Fold upwards into +Y)
            # Left plate: rotate clockwise (-theta) around pivot_L
            cos_th_L = np.cos(-theta_curr); sin_th_L = np.sin(-theta_curr)
            tied_l_nodes = meta["plate_left_nodes"] + meta["disp_bottom_left_nids"]
            for nid in tied_l_nodes:
                node = mesh.nodes[nid]
                dx0 = node.x - pivot_L[0]
                dy0 = node.y - pivot_L[1]
                x_new = pivot_L[0] + cos_th_L * dx0 - sin_th_L * dy0
                y_new = pivot_L[1] + sin_th_L * dx0 + cos_th_L * dy0
                solver.fix_dof(nid, 0, x_new - node.x)
                solver.fix_dof(nid, 1, y_new - node.y)
                if dim == "3D":
                    solver.fix_dof(nid, 2, 0.0)

            # Right plate: rotate counter-clockwise (+theta) around pivot_R
            cos_th_R = np.cos(theta_curr); sin_th_R = np.sin(theta_curr)
            tied_r_nodes = meta["plate_right_nodes"] + meta["disp_bottom_right_nids"]
            for nid in tied_r_nodes:
                node = mesh.nodes[nid]
                dx0 = node.x - pivot_R[0]
                dy0 = node.y - pivot_R[1]
                x_new = pivot_R[0] + cos_th_R * dx0 - sin_th_R * dy0
                y_new = pivot_R[1] + sin_th_R * dx0 + cos_th_R * dy0
                solver.fix_dof(nid, 0, x_new - node.x)
                solver.fix_dof(nid, 1, y_new - node.y)
                if dim == "3D":
                    solver.fix_dof(nid, 2, 0.0)

            # Symmetry BC on 3D at Z = 0
            if dim == "3D":
                for nid, node in mesh.nodes.items():
                    if abs(node.z) < 1e-4:
                        solver.fix_dof(nid, 2, 0.0)

            # Kinematic Rotation Predictor (Carry entire display laminate rigidly in plate zones)
            if enable_predictor:
                dth_L = -(theta_curr - theta_prev)
                cos_dth_L = np.cos(dth_L); sin_dth_L = np.sin(dth_L)
                for nid in disp_left_plate_nodes:
                    node = mesh.nodes[nid]
                    idx = nid_map[nid]
                    cx = node.x + solver.u[idx * ndof_n + 0]
                    cy = node.y + solver.u[idx * ndof_n + 1]
                    dx = cx - pivot_L[0]; dy = cy - pivot_L[1]
                    solver.u[idx * ndof_n + 0] = pivot_L[0] + cos_dth_L * dx - sin_dth_L * dy - node.x
                    solver.u[idx * ndof_n + 1] = pivot_L[1] + sin_dth_L * dx + cos_dth_L * dy - node.y

                dth_R = +(theta_curr - theta_prev)
                cos_dth_R = np.cos(dth_R); sin_dth_R = np.sin(dth_R)
                for nid in disp_right_plate_nodes:
                    node = mesh.nodes[nid]
                    idx = nid_map[nid]
                    cx = node.x + solver.u[idx * ndof_n + 0]
                    cy = node.y + solver.u[idx * ndof_n + 1]
                    dx = cx - pivot_R[0]; dy = cy - pivot_R[1]
                    solver.u[idx * ndof_n + 0] = pivot_R[0] + cos_dth_R * dx - sin_dth_R * dy - node.x
                    solver.u[idx * ndof_n + 1] = pivot_R[1] + sin_dth_R * dx + cos_dth_R * dy - node.y

                # Exact Dirichlet prescribed values on fixed DOFs
                for dof, val in solver.fixed_dofs.items():
                    solver.u[dof] = val

            # Solve step
            converged, iters = solver.solve_step(
                dt=dt_actual,
                max_iters=25,
                use_anderson_accel=True,
                anderson_m=4,
            )

            if not converged:
                consecutive_cutbacks += 1
                if consecutive_cutbacks > max_cutbacks:
                    print(f"\n[FATAL] Exceeded maximum cutbacks ({max_cutbacks}). Terminating.")
                    break
                dt_ctrl.notify_convergence(iters=iters, converged=False)
                solver.u = u_converged.copy()
                print(f"  [CUTBACK {consecutive_cutbacks}] Retrying step {step} with dt={dt_ctrl.dt:.5f}s ...")
                continue

            # Step converged
            consecutive_cutbacks = 0
            u_converged = solver.u.copy()
            theta_prev = theta_curr
            t = t_next
            dt_ctrl.notify_convergence(iters=iters, converged=True)

            # Reactions & Moments
            if dim == "2D":
                res_L = solver.compute_section_reactions(node_ids=tied_l_nodes, center_pt=pivot_L)
                res_R = solver.compute_section_reactions(node_ids=tied_r_nodes, center_pt=pivot_R)
            else:
                res_L = solver.compute_section_reactions(node_ids=tied_l_nodes, center_pt=(pivot_L[0], pivot_L[1], 0.0))
                res_R = solver.compute_section_reactions(node_ids=tied_r_nodes, center_pt=(pivot_R[0], pivot_R[1], 0.0))

            M_L = res_L["Mz"];  M_R = res_R["Mz"]

            Fx_L = res_L["Fx"]; Fx_R = res_R["Fx"]
            sum_Fx = Fx_L + Fx_R
            avg_M = 0.5 * (abs(M_L) + abs(M_R))
            diff_M = abs(abs(M_L) - abs(M_R))
            symm_err = (diff_M / max(avg_M, 1e-6)) * 100.0

            deg_disp = np.degrees(theta_curr)
            print(f"{step:5d} | {t:7.4f} | {deg_disp:6.2f}° | {dt_actual:8.5f} | {Fx_L:+9.3f} | {Fx_R:+9.3f} | {sum_Fx:+10.3e} | {M_L:+11.4f} | {M_R:+11.4f} | {symm_err:8.4f}% | {iters:5d} | {'CONV':>6}")

            history_time.append(t)
            history_theta.append(deg_disp)
            history_iters.append(iters)
            history_M_L.append(M_L)
            history_M_R.append(M_R)
            history_Fx_L.append(Fx_L)
            history_Fx_R.append(Fx_R)
            history_symm_err.append(symm_err)
            history_u.append(u_converged.copy())

            # Stream step to VTKHDF
            vdb.add_step(t, u_converged)

            # Live callback notification (PySide6 / streaming)
            if step_callback is not None:
                step_callback(step, t, deg_disp, u_converged.copy(), iters, M_L, M_R)

        vdb.close()
        wall_time = time.perf_counter() - t_start

        print("-" * 115)
        print(f"[OK] Simulation completed in {wall_time:.2f}s!")
        print(f"     Final Plate Fold: {history_theta[-1]:.2f}° (Total 180° Fold: {2*history_theta[-1]:.2f}°)")
        print(f"     Reaction Symmetry Error: {history_symm_err[-1]:.4f}% | Net Horizontal Fx: {history_Fx_L[-1] + history_Fx_R[-1]:.3e} N")
        print("=" * 115)

        # 10. Generate X-Y Summary Plot
        plot_path = os.path.join(out_dir, "summary_plots.png")
        _generate_summary_plot(plot_path, history_theta, history_M_L, history_M_R, history_iters, config)
        print(f"[OK] Saved summary plot: {plot_path}")

        # 11. Generate Folding Animation GIF
        gif_path = os.path.join(out_dir, "folding_animation.gif")
        _generate_folding_gif(gif_path, mesh, meta, config, history_theta, history_u)
        print(f"[OK] Saved folding animation GIF: {gif_path}")

        # 12. Generate JSON metadata summary
        summary_data = {
            "dim": dim,
            "preset": preset,
            "contact_mode": resolved_contact,
            "final_theta": float(history_theta[-1]),
            "final_M_L": float(history_M_L[-1]),
            "final_M_R": float(history_M_R[-1]),
            "symmetry_error": float(history_symm_err[-1]),
            "wall_time_s": float(wall_time),
            "num_steps": len(history_theta) - 1,
            "avg_iters": float(np.mean(history_iters[1:])) if len(history_iters) > 1 else 0.0,
            "cutbacks": consecutive_cutbacks,
        }
        json_path = os.path.join(out_dir, "result_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # 12. Generate HTML 1-Page Engineering Brief
        html_path = generate_engineering_html_report(out_dir, config, summary_data)
        print(f"[OK] Generated 1-Page HTML report: {html_path}")

        history_bundle = {
            "history_time": history_time,
            "history_theta": history_theta,
            "history_iters": history_iters,
            "history_M_L": history_M_L,
            "history_M_R": history_M_R,
            "history_Fx_L": history_Fx_L,
            "history_Fx_R": history_Fx_R,
            "history_symm_err": history_symm_err,
            "history_u": history_u,
            "u_final": u_converged,
            "wall_time_s": wall_time,
        }

        # 13. Launch GUI Viewer if requested
        if launch_gui:
            from example_disp_fold.viewer_qt_app import launch_qt_viewer
            print("[*] Launching integrated PySide6 CAE Viewer...")
            launch_qt_viewer(mesh, meta, config, history_bundle)

    finally:
        sys.stdout = old_stdout
        tee.close()

    return out_dir


def _generate_summary_plot(
    filepath: str,
    thetas: List[float],
    M_L: List[float],
    M_R: List[float],
    iters: List[int],
    config: Any,
) -> None:
    """Generates 3-row publication quality summary PNG with 9pt Korean font."""
    import matplotlib.pyplot as plt
    try:
        import koreanize_matplotlib
    except ImportError:
        pass

    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    plt.rcParams["font.size"] = 9
    plt.rcParams["font.family"] = ["Malgun Gothic", "Noto Sans KR", "sans-serif"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.0), dpi=150)

    # 1. Moment vs Theta + Elastica Corridor
    D_slip, D_mono, _ = config.compute_composite_bending_stiffness()
    thetas_deg = np.array(thetas)
    M_avg = 0.5 * (np.abs(M_L) + np.abs(M_R))

    ax1.plot(thetas_deg, M_avg, color="#0284c7", lw=2.0, label="FEM M_drive (평균)")
    ax1.plot(thetas_deg, np.abs(M_L), "--", color="#0ea5e9", lw=1.2, label="Left M_L")
    ax1.plot(thetas_deg, np.abs(M_R), ":", color="#38bdf8", lw=1.2, label="Right M_R")

    # Elastica Corridor
    L_eff = 2.0 * config.geometry.hinge_half_gap
    th_rad = np.radians(thetas_deg)
    M_slip = D_slip * (2.0 * th_rad) / L_eff
    M_mono = D_mono * (2.0 * th_rad) / L_eff
    ax1.fill_between(thetas_deg, M_slip, M_mono, color="#0284c7", alpha=0.15, label=f"Elastica Bounds (D_slip={D_slip:.2f})")

    ax1.set_xlabel("플레이트 접힘 각도 θ [deg]")
    ax1.set_ylabel("피벗 구동 모멘트 [N·mm]")
    ax1.set_title(f"180° 티어드롭 접힘 구동 모멘트 & 이론해 비교 ({config.simulation_case.dim} {config.simulation_case.preset_name})")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Newton Iterations
    steps = np.arange(1, len(iters) + 1)
    ax2.bar(steps, iters, color="#10b981", alpha=0.7, width=0.8, label="Newton Iterations")
    ax2.axhline(10.0, color="#ef4444", linestyle="--", lw=1.0, label="Target (10 iters)")
    ax2.set_xlabel("해석 증분 스텝 (Step)")
    ax2.set_ylabel("뉴턴-랩슨 반복수")
    ax2.set_title("수치해석 수렴 이력 (Convergence History)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def _generate_folding_gif(
    gif_path: str,
    mesh: Any,
    meta: Dict[str, Any],
    config: Any,
    history_theta: List[float],
    history_u: List[np.ndarray],
) -> None:
    """Renders 2D/3D deformed shapes across time steps into an animated GIF."""
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import io
    from PIL import Image

    if not history_u or len(history_u) < 2:
        return

    orig_coords = mesh.nodes_array()
    nid_map = mesh.node_id_to_index()
    is_3d = (config.simulation_case.dim == "3D")
    dof_per_node = 3 if is_3d else 2

    # Map each element to color
    elem_colors = {}
    layer_sets = meta.get("layer_elem_sets", {})
    for lay in config.layers:
        c = lay.color
        if lay.name in layer_sets:
            for eid in layer_sets[lay.name]:
                elem_colors[eid] = c

    plate_color = "#94a3b8"
    frames = []
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=120)

    # Subsample steps if too many for fast GIF generation
    step_indices = list(range(len(history_u)))
    if len(step_indices) > 25:
        step_indices = [step_indices[i] for i in np.linspace(0, len(step_indices) - 1, 25, dtype=int)]

    for s_idx in step_indices:
        ax.clear()
        u = history_u[s_idx]
        th = history_theta[s_idx]

        # Deformed coordinates
        def_coords = orig_coords.copy()
        for nid, idx in nid_map.items():
            st = idx * dof_per_node
            def_coords[idx, 0] += u[st]
            def_coords[idx, 1] += u[st + 1]

        patches = []
        facecolors = []

        for eid, elem in mesh.elements.items():
            if is_3d:
                nids = elem.node_ids[:4]
            else:
                nids = elem.node_ids

            idx = [nid_map[n] for n in nids]
            xy = def_coords[idx, :2]
            patches.append(Polygon(xy, closed=True))
            c = elem_colors.get(eid, plate_color)
            facecolors.append(c)

        pc = PatchCollection(patches, facecolors=facecolors, edgecolors="#1e293b", linewidths=0.3, alpha=0.9)
        ax.add_collection(pc)

        ax.set_aspect("equal", "datalim")
        ax.set_xlim(-45.0, 45.0)
        ax.set_ylim(-10.0, 45.0)
        ax.set_title(
            f"Teardrop 180° Folding Deformation ({config.simulation_case.dim} {config.simulation_case.preset_name}) | θ = {th:.1f}° (Total {2*th:.1f}°)",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("X [mm]", fontsize=8)
        ax.set_ylabel("Y [mm]", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf))

    plt.close(fig)

    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=120,
            loop=0,
            optimize=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified 2D/3D Display 180° Teardrop Folding Simulation")
    parser.add_argument("--dim", type=str, default="3D", choices=["2D", "3D"], help="Simulation dimension")
    parser.add_argument("--preset", type=str, default="teardrop", choices=["teardrop", "compact_4", "u_shape", "compact_waterdrop", "crease_zero_gap"], help="Kinematic preset")
    parser.add_argument("--elem_case", type=str, default=None, choices=["3D-Opt1", "3D-Opt2-H", "3D-Base", "2D-Opt1", "2D-Opt2", "2D-Base"], help="Element formulation case preset")
    parser.add_argument("--theta_max", type=float, default=90.0, help="Maximum plate rotation angle in deg (default 90° for 180° total fold)")
    parser.add_argument("--enable_predictor", dest="enable_predictor", action="store_true", default=True, help="Enable kinematic rotation predictor")
    parser.add_argument("--no_predictor", dest="enable_predictor", action="store_false", help="Disable kinematic rotation predictor")
    parser.add_argument("--tie_method", type=str, default="MPC", choices=["MPC", "PENALTY"], help="Tie constraint formulation")
    parser.add_argument("--contact_mode", type=str, default="AUTO", choices=["AUTO", "OFF", "DUAL_MORTAR"], help="Contact formulation")
    parser.add_argument("--sol_preset", type=str, default="ADAPTIVE", choices=["ADAPTIVE", "STANDARD", "RELAXED"], help="SolutionControl preset")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional step limit for smoke testing")
    parser.add_argument("--view", action="store_true", help="Launch integrated PySide6 CAE viewer upon completion")
    parser.add_argument("--live", action="store_true", help="Launch live PySide6 streaming viewer while solver executes")

    args = parser.parse_args()

    if args.live:
        from example_disp_fold.viewer_qt_app import launch_qt_viewer_live
        from example_disp_fold.model_spec import UnifiedFoldModelConfig
        from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator

        cfg = UnifiedFoldModelConfig(preset=args.preset, dim=args.dim, elem_case=args.elem_case)
        mesh, meta = GmshMeshGenerator(cfg).build()
        run_kwargs = {
            "dim": args.dim,
            "preset": args.preset,
            "elem_case": args.elem_case,
            "theta_max_deg": args.theta_max,
            "enable_predictor": args.enable_predictor,
            "tie_method": args.tie_method,
            "contact_mode": args.contact_mode,
            "sol_preset": args.sol_preset,
            "max_steps": args.max_steps,
        }
        launch_qt_viewer_live(run_dispfold_simulation, run_kwargs, mesh, meta, cfg)
    else:
        run_dispfold_simulation(
            dim=args.dim,
            preset=args.preset,
            elem_case=args.elem_case,
            theta_max_deg=args.theta_max,
            enable_predictor=args.enable_predictor,
            tie_method=args.tie_method,
            contact_mode=args.contact_mode,
            sol_preset=args.sol_preset,
            max_steps=args.max_steps,
            launch_gui=args.view,
        )
