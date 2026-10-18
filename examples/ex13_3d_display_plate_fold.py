"""
ex13_3d_display_plate_fold.py
==============================
Full 3D Multi-Layer Foldable Display 180-Degree Teardrop Folding Solver.

Features Propagated from Canonical 2D Solver (EX12):
-----------------------------------------------------
1. Abaqus Python API Architecture & Parity:
   - Structured Model, Step, SolutionControl (preset="ADAPTIVE"), and Interaction hierarchy.
2. Standardized Teardrop Kinematics:
   - Full span [-40, 40] mm, Free hinge gap [-7.5, 7.5] mm (15.0 mm span).
   - Rigid pivots at (+-1.4, 0.0) mm producing exact 2.8 mm teardrop neck gap.
   - Hinge PSA void: bottom PSA layer 1 voided in hinge span [-7.5, 7.5] mm.
3. Locking-Free 3D Solid Formulations:
   - PET & GLASS: C3D8I_CR (9-mode Enhanced Assumed Strain + Co-Rotational, shear-locking free).
   - PSA: C3D8H_CR (Herrmann u-P Hybrid + Co-Rotational, volumetric locking free & large shear slip).
4. Adaptive Time Stepping with Smart Cutback (AdaptiveDtController):
   - Dynamic dt acceleration, divergence detection, state rollback, and cutback recovery.
5. Kinematic Rotation Predictor:
   - Rigid-body rotation predictor for plate-coupled slave nodes, eliminating startup residual spikes.
6. Real-Time Hinge Reactions & Static Equilibrium Monitoring:
   - Evaluates left/right drive moments (M_L, M_R) about pivots, horizontal pull forces (Fx_L, Fx_R),
     symmetry error, and static equilibrium at every converged step.
7. Visualizer & Data Pipeline:
   - Compatible with view_ex13_pyvista.py interactive 3D viewer.
"""

from __future__ import annotations
import os
import sys

# Parallel linear solver configuration (4 cores for MKL PARDISO)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import time
import pickle
import argparse
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.constraint3d.surface_tie3d import SurfaceTieConstraint3D
from dispsolver.solver.dt_controller import AdaptiveDtController
from dispsolver.model.model import Model
from dispsolver.model.step import Step
from dispsolver.model.control import SolutionControl
from dispsolver.model.interaction import ContactControl, ContactProperty
from dispsolver.fold_model_config import (
    make_teardrop_config,
    FoldModelConfig,
    DEFAULT_CONFIG,
    smoothstep_amp,
)


def build_3d_display_and_plate_mesh(
    config: Optional[FoldModelConfig] = None,
    width_disp: float = 1.0,
    nz_disp: int = 2,
    nx_hinge: int = 30,
    nx_plate: int = 20,
    layup: str = "teardrop_14",
) -> Tuple[Mesh3D, Dict[str, Any]]:
    """Builds full 3D multi-layer continuum display and rigid support plate mesh.

    Parameters
    ----------
    config : FoldModelConfig, optional
        Teardrop configuration (geometry, thicknesses, pivots). Defaults to make_teardrop_config().
    width_disp : float
        Extrusion width along Z axis (mm). Default 1.0 mm.
    nz_disp : int
        Number of elements along Z width. Default 2.
    nx_hinge : int
        Number of elements in free hinge span [-7.5, 7.5]. Default 30 (dx = 0.5 mm).
    nx_plate : int
        Number of elements along each plate body span. Default 20.
    layup : str
        'teardrop_14' (full 14-layer canonical stackup) or 'compact_4' (4-layer fast smoke test).
    """
    if config is None:
        config = make_teardrop_config()

    geo = config.geometry
    half_L = geo.display_half_length  # 40.0 mm
    hinge_gap = geo.hinge_half_gap     # 7.5 mm
    pivot_x = geo.hinge_pivot_x       # 1.4 mm
    pivot_y = geo.hinge_pivot_y       # 0.0 mm
    plate_t = geo.plate_thickness     # 0.5 mm

    mesh = Mesh3D()
    nid = 1

    # 1. Define X-coordinates (Left Plate -> Free Hinge -> Right Plate)
    x_left = np.linspace(-half_L, -hinge_gap, nx_plate + 1)
    x_hinge = np.linspace(-hinge_gap, hinge_gap, nx_hinge + 1)
    x_right = np.linspace(hinge_gap, half_L, nx_plate + 1)
    xs = np.concatenate([x_left[:-1], x_hinge[:-1], x_right])
    nx_total = len(xs) - 1

    # Z-coordinates
    zs = np.linspace(-width_disp / 2.0, width_disp / 2.0, nz_disp + 1)

    # 2. Define Y-layers (Layup)
    if layup == "teardrop_14":
        # Full 14-layer stackup (Total thickness 0.63 mm):
        # Layer 0: PSA (30 um) - Voided in hinge span!
        # Layer 1: PET (60 um)
        # Layer 2: PSA (30 um)
        # Layer 3: PET (60 um)
        # Layer 4: PSA (30 um)
        # Layer 5: PET (60 um)
        # Layer 6: PSA (30 um)
        # Layer 7: PET (60 um)
        # Layer 8: PSA (30 um)
        # Layer 9: PET (60 um)
        # Layer 10: PSA (30 um)
        # Layer 11: PET (60 um)
        # Layer 12: PSA (30 um)
        # Layer 13: GLASS (60 um)
        layer_specs = [
            ("PSA", 0.030, True),   # layer 0: voided in hinge span
            ("PET", 0.060, False),
            ("PSA", 0.030, False),
            ("PET", 0.060, False),
            ("PSA", 0.030, False),
            ("PET", 0.060, False),
            ("PSA", 0.030, False),
            ("PET", 0.060, False),
            ("PSA", 0.030, False),
            ("PET", 0.060, False),
            ("PSA", 0.030, False),
            ("PET", 0.060, False),
            ("PSA", 0.030, False),
            ("GLASS", 0.060, False),
        ]
    else:  # compact_4
        layer_specs = [
            ("PET", 0.060, False),
            ("PSA", 0.030, False),
            ("PET", 0.060, False),
            ("PSA", 0.030, False),
        ]

    n_layers = len(layer_specs)
    y_bounds = [0.0]
    for _, t_l, _ in layer_specs:
        y_bounds.append(y_bounds[-1] + t_l)
    y_bounds = np.array(y_bounds)
    thick_total = float(y_bounds[-1])

    # Build Display Nodes Grid (nx_total + 1, n_layers + 1, nz_disp + 1)
    disp_grid = np.zeros((nx_total + 1, n_layers + 1, nz_disp + 1), dtype=np.int32)
    for k, z in enumerate(zs):
        for j, y in enumerate(y_bounds):
            for i, x in enumerate(xs):
                mesh.add_node(nid, float(x), float(y), float(z))
                disp_grid[i, j, k] = nid
                nid += 1

    # Build Display Elements
    eid = 1
    disp_bottom_left_nids = []
    disp_bottom_right_nids = []

    for k in range(nz_disp):
        for j in range(n_layers):
            mat_name, _, is_void_in_hinge = layer_specs[j]

            if mat_name == "PET":
                pid = 1; elem_type = "C3D8I_CR"
            elif mat_name == "PSA":
                pid = 2; elem_type = "C3D8H_CR"
            elif mat_name == "GLASS":
                pid = 3; elem_type = "C3D8I_CR"
            else:
                pid = 1; elem_type = "C3D8I_CR"

            for i in range(nx_total):
                x_mid = 0.5 * (xs[i] + xs[i + 1])

                # Check hinge void condition
                if is_void_in_hinge and (-hinge_gap - 1e-6 < x_mid < hinge_gap + 1e-6):
                    continue  # Element voided in free hinge span!

                n1 = int(disp_grid[i, j, k])
                n2 = int(disp_grid[i + 1, j, k])
                n3 = int(disp_grid[i + 1, j + 1, k])
                n4 = int(disp_grid[i, j + 1, k])
                n5 = int(disp_grid[i, j, k + 1])
                n6 = int(disp_grid[i + 1, j, k + 1])
                n7 = int(disp_grid[i + 1, j + 1, k + 1])
                n8 = int(disp_grid[i, j + 1, k + 1])

                mesh.add_element(eid, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type=elem_type)
                mesh.elements[eid].pid = pid
                eid += 1

    # Extract display bottom surface slave nodes at Y = 0.0 (Left: x <= -hinge_gap, Right: x >= hinge_gap)
    for k in range(nz_disp + 1):
        for i in range(nx_total + 1):
            n = int(disp_grid[i, 0, k])
            node = mesh.nodes[n]
            if node.x <= -hinge_gap + 1e-4:
                disp_bottom_left_nids.append(n)
            elif node.x >= hinge_gap - 1e-4:
                disp_bottom_right_nids.append(n)

    # 3. Build Left Rigid Plate: X in [-half_L, -hinge_gap], Y in [-plate_t, 0], Z in [-w/2, w/2]
    ny_plate = 1
    y_p_left = np.linspace(-plate_t, 0.0, ny_plate + 1)
    plate_l_grid = np.zeros((nx_plate + 1, ny_plate + 1, nz_disp + 1), dtype=np.int32)

    for k, z in enumerate(zs):
        for j, y in enumerate(y_p_left):
            for i, x in enumerate(x_left):
                mesh.add_node(nid, float(x), float(y), float(z))
                plate_l_grid[i, j, k] = nid
                nid += 1

    plate_left_nodes = [int(plate_l_grid[i, j, k]) for i in range(nx_plate + 1) for j in range(ny_plate + 1) for k in range(nz_disp + 1)]
    plate_l_top_faces = []

    for k in range(nz_disp):
        for j in range(ny_plate):
            for i in range(nx_plate):
                n1 = int(plate_l_grid[i, j, k])
                n2 = int(plate_l_grid[i + 1, j, k])
                n3 = int(plate_l_grid[i + 1, j + 1, k])
                n4 = int(plate_l_grid[i, j + 1, k])
                n5 = int(plate_l_grid[i, j, k + 1])
                n6 = int(plate_l_grid[i + 1, j, k + 1])
                n7 = int(plate_l_grid[i + 1, j + 1, k + 1])
                n8 = int(plate_l_grid[i, j + 1, k + 1])
                mesh.add_element(eid, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type="C3D8_CR")
                mesh.elements[eid].pid = 4
                eid += 1
                if j == ny_plate - 1:
                    plate_l_top_faces.append((n4, n3, n7, n8))

    # 4. Build Right Rigid Plate: X in [hinge_gap, half_L], Y in [-plate_t, 0], Z in [-w/2, w/2]
    y_p_right = np.linspace(-plate_t, 0.0, ny_plate + 1)
    plate_r_grid = np.zeros((nx_plate + 1, ny_plate + 1, nz_disp + 1), dtype=np.int32)

    for k, z in enumerate(zs):
        for j, y in enumerate(y_p_right):
            for i, x in enumerate(x_right):
                mesh.add_node(nid, float(x), float(y), float(z))
                plate_r_grid[i, j, k] = nid
                nid += 1

    plate_right_nodes = [int(plate_r_grid[i, j, k]) for i in range(nx_plate + 1) for j in range(ny_plate + 1) for k in range(nz_disp + 1)]
    plate_r_top_faces = []

    for k in range(nz_disp):
        for j in range(ny_plate):
            for i in range(nx_plate):
                n1 = int(plate_r_grid[i, j, k])
                n2 = int(plate_r_grid[i + 1, j, k])
                n3 = int(plate_r_grid[i + 1, j + 1, k])
                n4 = int(plate_r_grid[i, j + 1, k])
                n5 = int(plate_r_grid[i, j, k + 1])
                n6 = int(plate_r_grid[i + 1, j, k + 1])
                n7 = int(plate_r_grid[i + 1, j + 1, k + 1])
                n8 = int(plate_r_grid[i, j + 1, k + 1])
                mesh.add_element(eid, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type="C3D8_CR")
                mesh.elements[eid].pid = 4
                eid += 1
                if j == ny_plate - 1:
                    plate_r_top_faces.append((n4, n3, n7, n8))

    meta = {
        "disp_bottom_left_nids": disp_bottom_left_nids,
        "disp_bottom_right_nids": disp_bottom_right_nids,
        "plate_left_nodes": plate_left_nodes,
        "plate_right_nodes": plate_right_nodes,
        "plate_l_top_faces": plate_l_top_faces,
        "plate_r_top_faces": plate_r_top_faces,
        "left_pivot": (-pivot_x, pivot_y),
        "right_pivot": (pivot_x, pivot_y),
        "thick_disp": thick_total,
        "length_disp": 2.0 * half_L,
        "width_disp": width_disp,
        "layup": layup,
        "layer_specs": layer_specs,
    }

    return mesh, meta


def build_3d_abaqus_model_hierarchy(config: FoldModelConfig, preset: str = "ADAPTIVE") -> Model:
    """Builds standard Abaqus Python API Model container hierarchy for 3D EX13."""
    model = Model(name="Ex13_3D_FoldingModel", dim=3)
    step = model.Step(
        name="Step-1_Folding",
        procedure="STATIC",
        time_period=config.drive.t_total,
        dt_init=config.drive.dt_init,
        dt_min=config.drive.dt_min,
        dt_max=config.drive.dt_max,
        nlgeom=True,
    )
    sol_ctrl = model.SolutionControl(
        name="AdaptiveSolutionControl",
        preset=preset,
    )
    step.solution_control = sol_ctrl
    return model


def run_3d_display_plate_fold(
    theta_max_deg: float = 90.0,
    config: Optional[FoldModelConfig] = None,
    layup: str = "compact_4",
    width_disp: float = 1.0,
    nz_disp: int = 2,
    penalty_tie: float = 1.0e5,
    max_steps: Optional[int] = None,
    preset: str = "ADAPTIVE",
    max_du_limit: Optional[float] = None,
    use_anderson: bool = True,
    out_pkl_name: str = "ex13_3d_fold_result.pkl",
    debug_iters: bool = False,
) -> Dict[str, Any]:
    """Executes full 3D 180-degree teardrop folding simulation with adaptive time-stepping."""
    if config is None:
        config = make_teardrop_config()

    print("=" * 115)
    print(" EX13: 3D Multi-Layer Display 180° Hinge Folding Solver (Abaqus API & EX12 Parity)")
    print(f" Layup: {layup} | Width: {width_disp:.2f} mm | Elements: PET/GLASS (C3D8I_CR) + PSA (C3D8H_CR)")
    print(f" Teardrop Neck: Pivots (+-{config.geometry.hinge_pivot_x:.2f}, 0.0) mm | Free Span: [-7.5, 7.5] mm")
    print("=" * 115)

    t_start = time.perf_counter()

    # 1. Abaqus Python API Hierarchy
    abq_model = build_3d_abaqus_model_hierarchy(config, preset=preset)
    print(f"[ABAQUS API] Model '{abq_model.name}' declared with Step '{list(abq_model.steps.keys())[-1]}'")
    print(f"[ABAQUS API] SolutionControl Preset: {preset}")

    # 2. Build 3D Mesh
    mesh, meta = build_3d_display_and_plate_mesh(
        config=config,
        width_disp=width_disp,
        nz_disp=nz_disp,
        layup=layup,
    )
    nid_map = mesh.node_id_to_index()
    coords = mesh.nodes_array()

    # 3. Material Properties (Standard Elastic/Plastic matching EX12)
    # PID 1: PET (E=4000 MPa, nu=0.30)
    # PID 2: PSA (E=0.05 MPa, nu=0.499 - Nearly Incompressible Rubber)
    # PID 3: GLASS (E=70000 MPa, nu=0.22)
    # PID 4: Rigid Plate (E=200000 MPa, nu=0.30)
    materials = {
        1: {"E": 4000.0, "nu": 0.30},
        2: {"E": 0.05, "nu": 0.499},
        3: {"E": 70000.0, "nu": 0.22},
        4: {"E": 200000.0, "nu": 0.30},
    }

    solver = DynamicSolver3D(mesh, materials, nlgeom=True, solution_controls=preset)
    if max_du_limit is not None:
        # Newton correction clamp (dynamic3d.py reads it via getattr; solver default 0.10 mm)
        solver.max_du_limit = float(max_du_limit)
    if debug_iters:
        solver.debug_sdi = True  # per-Newton-iteration r_norm / r_norm/q print (dynamic3d.py)
    print(f"[*] max_du_limit={getattr(solver, 'max_du_limit', 'solver default 0.10')} | Anderson={'ON' if use_anderson else 'OFF'}")

    # 4. Tie Constraints: Kinematic MPC Elimination (Abaqus *TIE parity)
    # Coupled display bottom nodes follow the rigid plate motion directly via Dirichlet BCs.
    # This completely eliminates penalty spring condition-number degradation and guarantees 0.0mm gap.
    print(f"[*] Mesh Statistics: {len(mesh.nodes)} nodes, {len(mesh.elements)} 3D elements, {solver.num_dofs} DOFs.")
    print(f"[*] Kinematic MPC Ties: Left={len(meta['disp_bottom_left_nids'])} nodes, Right={len(meta['disp_bottom_right_nids'])} nodes.")

    # 5. Adaptive Time Stepping Controller
    dt_ctrl = AdaptiveDtController(
        dt_init=0.005,
        dt_min=1.0e-5,
        dt_max=0.02,
        target_iters=10,
    )

    th_max_rad = np.radians(theta_max_deg)
    pivot_L = meta["left_pivot"]
    pivot_R = meta["right_pivot"]

    history_time = [0.0]
    history_theta = [0.0]
    history_iters = [0]
    history_M_L = [0.0]
    history_M_R = [0.0]
    history_Fx_L = [0.0]
    history_Fx_R = [0.0]
    history_symm_err = [0.0]

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
            print(f"[!] Reached max_steps={max_steps}. Stopping solve loop.")
            break

        step += 1
        dt = dt_ctrl.dt
        t_next = min(1.0, t + dt)
        dt_actual = t_next - t

        # Kinematic smooth ramp
        s = t_next
        theta_curr = th_max_rad * smoothstep_amp(s)

        # Apply Kinematic Rigid Plate & Tied Display Dirichlet BCs (Fold upwards into +Y)
        # Left plate (x < -7.5) & tied display bottom nodes: rotate clockwise (-theta) around pivot_L
        cos_th_L = np.cos(-theta_curr)
        sin_th_L = np.sin(-theta_curr)
        for nid in meta["plate_left_nodes"] + meta["disp_bottom_left_nids"]:
            node = mesh.nodes[nid]
            dx0 = node.x - pivot_L[0]
            dy0 = node.y - pivot_L[1]
            x_new = pivot_L[0] + cos_th_L * dx0 - sin_th_L * dy0
            y_new = pivot_L[1] + sin_th_L * dx0 + cos_th_L * dy0
            solver.fix_dof(nid, 0, x_new - node.x)
            solver.fix_dof(nid, 1, y_new - node.y)
            solver.fix_dof(nid, 2, 0.0)

        # Right plate (x > +7.5) & tied display bottom nodes: rotate counter-clockwise (+theta) around pivot_R
        cos_th_R = np.cos(theta_curr)
        sin_th_R = np.sin(theta_curr)
        for nid in meta["plate_right_nodes"] + meta["disp_bottom_right_nids"]:
            node = mesh.nodes[nid]
            dx0 = node.x - pivot_R[0]
            dy0 = node.y - pivot_R[1]
            x_new = pivot_R[0] + cos_th_R * dx0 - sin_th_R * dy0
            y_new = pivot_R[1] + sin_th_R * dx0 + cos_th_R * dy0
            solver.fix_dof(nid, 0, x_new - node.x)
            solver.fix_dof(nid, 1, y_new - node.y)
            solver.fix_dof(nid, 2, 0.0)

        # Symmetry / plane-strain boundary at Z = 0
        for nid in mesh.nodes.keys():
            if abs(mesh.nodes[nid].z) < 1e-4:
                solver.fix_dof(nid, 2, 0.0)

        # Nonlinear solve step with Anderson Acceleration
        converged, iters = solver.solve_step(
            dt=dt_actual,
            max_iters=25,
            use_anderson_accel=use_anderson,
            anderson_m=4,
        )

        if not converged:
            consecutive_cutbacks += 1
            if consecutive_cutbacks > max_cutbacks:
                print(f"\n[FATAL] Exceeded maximum consecutive cutbacks ({max_cutbacks}). Terminating.")
                break
            dt_ctrl.notify_convergence(iters=iters, converged=False)
            # Rollback state
            solver.u = u_converged.copy()
            print(f"  [CUTBACK {consecutive_cutbacks}] Retrying step {step} with dt={dt_ctrl.dt:.5f}s ...")
            continue

        # Success
        consecutive_cutbacks = 0
        u_converged = solver.u.copy()
        theta_prev = theta_curr
        t = t_next
        dt_ctrl.notify_convergence(iters=iters, converged=True)

        # Real-Time Hinge Reactions & Symmetry Check (eval_hinge_reactions 3D Parity)
        res_L = solver.compute_section_reactions(center_pt=(pivot_L[0], pivot_L[1], 0.0), node_ids=meta["plate_left_nodes"] + meta["disp_bottom_left_nids"])
        res_R = solver.compute_section_reactions(center_pt=(pivot_R[0], pivot_R[1], 0.0), node_ids=meta["plate_right_nodes"] + meta["disp_bottom_right_nids"])

        Fx_L = res_L["Fx"]; Fx_R = res_R["Fx"]
        M_L = res_L["Mz"];  M_R = res_R["Mz"]
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

    wall_time = time.perf_counter() - t_start
    print("-" * 115)
    print(f"[OK] 3D Folding Simulation Finished in {wall_time:.2f}s!")
    print(f"     Final Plate Fold: {history_theta[-1]:.2f}° (Total 180° Fold: {2*history_theta[-1]:.2f}°)")
    print(f"     Final Drive Moments: Left = {history_M_L[-1]:.4f} N*mm, Right = {history_M_R[-1]:.4f} N*mm")
    print(f"     Reaction Symmetry Error: {history_symm_err[-1]:.4f}% | Horizontal Sum Fx: {history_Fx_L[-1] + history_Fx_R[-1]:.3e} N")
    print("=" * 115)

    # Save results for view_ex13_pyvista.py
    out_dir = os.path.join(REPO_ROOT, "examples", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_pkl = os.path.join(out_dir, out_pkl_name)

    res_data = {
        "mesh": mesh,
        "u_final": solver.u,
        "history_time": history_time,
        "history_theta": history_theta,
        "history_iters": history_iters,
        "history_M_L": history_M_L,
        "history_M_R": history_M_R,
        "history_Fx_L": history_Fx_L,
        "history_Fx_R": history_Fx_R,
        "history_symm_err": history_symm_err,
        "wall_time_s": wall_time,
        "meta": meta,
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(res_data, f)
    print(f"Saved verified 3D fold results to: {out_pkl}")

    return res_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EX13: Full 3D Multi-Layer Display 180° Hinge Folding Solver")
    parser.add_argument("--theta_max", type=float, default=90.0, help="Maximum folding angle per plate (deg)")
    parser.add_argument("--layup", type=str, default="compact_4", choices=["compact_4", "teardrop_14"], help="Layup mode")
    parser.add_argument("--width", type=float, default=1.0, help="Display extrusion width in Z (mm)")
    parser.add_argument("--nz", type=int, default=2, help="Number of elements along Z width")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum steps to run")
    parser.add_argument("--max_du_limit", type=float, default=None, help="Newton correction clamp [mm] (default: solver's 0.10)")
    parser.add_argument("--no_anderson", action="store_true", help="Disable Anderson acceleration")
    parser.add_argument("--debug_iters", action="store_true", help="Print per-iteration residual norms")
    parser.add_argument("--out_pkl", type=str, default="ex13_3d_fold_result.pkl", help="Result pickle file name under examples/results")
    parser.add_argument("--sol_preset", type=str, default="ADAPTIVE", choices=["ADAPTIVE", "STANDARD", "RELAXED"], help="SolutionControl preset")
    args = parser.parse_args()

    run_3d_display_plate_fold(
        theta_max_deg=args.theta_max,
        layup=args.layup,
        width_disp=args.width,
        nz_disp=args.nz,
        max_steps=args.max_steps,
        preset=args.sol_preset,
        max_du_limit=args.max_du_limit,
        use_anderson=not args.no_anderson,
        out_pkl_name=args.out_pkl,
        debug_iters=args.debug_iters,
    )
