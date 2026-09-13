"""
ex13_3d_display_plate_fold.py
==============================
Full 3D Multi-layer Foldable Display 180-Degree Hinge Folding Solver.

Features:
1. Full 3D Solid Continuum Mesh:
   - Alternating multi-layer stackup:
     * PET layers: C3D8I_CR (9-mode EAS + Co-Rotational, locking-free)
     * PSA layers: C3D8H_CR (Herrmann u-P Hybrid + Co-Rotational, locking-free & large shear slip)
2. Dual Rigid Support Plates:
   - Left plate (X in [-40, -10]) driven about Left Pivot (-3.0, 0.0)
   - Right plate (X in [10, 40]) driven about Right Pivot (+3.0, 0.0)
   - Exact 3D kinematic rigid-body Dirichlet BCs per step (no artificial penalty gap).
3. 3D Surface Tie Coupling:
   - SurfaceTieConstraint3D couples plate top Quad4 faces to display bottom surface nodes.
4. Target:
   - Full 90 deg / side (180 deg combined face-to-face fold, teardrop U-loop).
"""

from __future__ import annotations
import os
import sys
import time
import pickle
from typing import Dict, List, Tuple, Any
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.constraint3d.surface_tie3d import SurfaceTieConstraint3D
from dispsolver.solver.dt_controller import AdaptiveDtController


def build_3d_display_and_plate_mesh(
    length_disp: float = 80.0,
    thick_disp: float = 0.5,
    width_disp: float = 1.0,
    nx_disp: int = 40,
    ny_layers: int = 4,
    nz_disp: int = 2,
    plate_thickness: float = 0.5,
    nx_plate: int = 15,
    ny_plate: int = 1,
) -> Tuple[Mesh3D, Dict[str, Any]]:
    """Builds a complete 3D composite mesh containing:
    1. Multi-layer Deformable Display (PET + PSA alternating)
    2. Left and Right Rigid Support Plates
    """
    mesh = Mesh3D()
    nid = 1

    # Coordinate ranges
    x_disp = np.linspace(-length_disp / 2.0, length_disp / 2.0, nx_disp + 1)
    y_disp = np.linspace(0.0, thick_disp, ny_layers + 1)
    z_disp = np.linspace(-width_disp / 2.0, width_disp / 2.0, nz_disp + 1)

    # 1. Build Display Nodes
    disp_grid = np.zeros((nx_disp + 1, ny_layers + 1, nz_disp + 1), dtype=np.int32)
    for k, z in enumerate(z_disp):
        for j, y in enumerate(y_disp):
            for i, x in enumerate(x_disp):
                mesh.add_node(nid, x, y, z)
                disp_grid[i, j, k] = nid
                nid += 1

    # Display Elements (Alternating PET / PSA)
    # Layer 0 (bottom): PET (pid=1, C3D8I_CR)
    # Layer 1: PSA (pid=2, C3D8H_CR)
    # Layer 2: PET (pid=1, C3D8I_CR)
    # Layer 3 (top): PSA (pid=2, C3D8H_CR)
    eid = 1
    for k in range(nz_disp):
        for j in range(ny_layers):
            is_psa = (j % 2 == 1)
            elem_type = "C3D8H_CR" if is_psa else "C3D8I_CR"
            pid = 2 if is_psa else 1
            for i in range(nx_disp):
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

    # Extract display bottom surface slave nodes (y = 0.0)
    # Left region: x <= -10.0, Right region: x >= 10.0
    disp_bottom_left_nids = []
    disp_bottom_right_nids = []
    for k in range(nz_disp + 1):
        for i in range(nx_disp + 1):
            n = int(disp_grid[i, 0, k])
            node = mesh.nodes[n]
            if node.x <= -10.0 + 1e-4:
                disp_bottom_left_nids.append(n)
            elif node.x >= 10.0 - 1e-4:
                disp_bottom_right_nids.append(n)

    # 2. Build Left Rigid Plate (X in [-40, -10], Y in [-plate_thick, 0], Z in [-width/2, width/2])
    x_p_left = np.linspace(-length_disp / 2.0, -10.0, nx_plate + 1)
    y_p_left = np.linspace(-plate_thickness, 0.0, ny_plate + 1)
    plate_l_grid = np.zeros((nx_plate + 1, ny_plate + 1, nz_disp + 1), dtype=np.int32)

    for k, z in enumerate(z_disp):
        for j, y in enumerate(y_p_left):
            for i, x in enumerate(x_p_left):
                mesh.add_node(nid, x, y, z)
                plate_l_grid[i, j, k] = nid
                nid += 1

    plate_left_nodes = [int(plate_l_grid[i, j, k]) for i in range(nx_plate + 1) for j in range(ny_plate + 1) for k in range(nz_disp + 1)]

    # Plate Left Elements and Top Master Faces (y = 0.0, j = ny_plate)
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
                mesh.elements[eid].pid = 3
                eid += 1

                if j == ny_plate - 1:
                    # Top surface face (Y=0): [n4, n3, n7, n8]
                    plate_l_top_faces.append((n4, n3, n7, n8))

    # 3. Build Right Rigid Plate (X in [10, 40], Y in [-plate_thick, 0], Z in [-width/2, width/2])
    x_p_right = np.linspace(10.0, length_disp / 2.0, nx_plate + 1)
    y_p_right = np.linspace(-plate_thickness, 0.0, ny_plate + 1)
    plate_r_grid = np.zeros((nx_plate + 1, ny_plate + 1, nz_disp + 1), dtype=np.int32)

    for k, z in enumerate(z_disp):
        for j, y in enumerate(y_p_right):
            for i, x in enumerate(x_p_right):
                mesh.add_node(nid, x, y, z)
                plate_r_grid[i, j, k] = nid
                nid += 1

    plate_right_nodes = [int(plate_r_grid[i, j, k]) for i in range(nx_plate + 1) for j in range(ny_plate + 1) for k in range(nz_disp + 1)]

    # Plate Right Elements and Top Master Faces (y = 0.0, j = ny_plate)
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
                mesh.elements[eid].pid = 3
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
        "left_pivot": (-3.0, 0.0),
        "right_pivot": (3.0, 0.0),
        "thick_disp": thick_disp,
        "length_disp": length_disp,
    }

    return mesh, meta


def run_3d_display_plate_fold(
    theta_max_deg: float = 90.0,
    n_steps: int = 40,
    penalty_tie: float = 1.0e5,
) -> Dict[str, Any]:
    """Executes full 3D 180-degree folding solve with multi-layer display and dual rigid plates."""
    print("=" * 90)
    print(f" EX13: 3D Multi-Layer Display 180° Hinge Folding Solver (90° per side)")
    print(f" Materials: PET (C3D8I_CR) + PSA (C3D8H_CR) | 3D Surface Tie Constraint")
    print("=" * 90)

    t_start = time.perf_counter()

    # 1. Build Mesh
    mesh, meta = build_3d_display_and_plate_mesh(
        length_disp=80.0,
        thick_disp=0.5,
        width_disp=1.0,
        nx_disp=40,
        ny_layers=4,
        nz_disp=2,
        plate_thickness=0.5,
        nx_plate=15,
        ny_plate=1,
    )
    nid_map = mesh.node_id_to_index()
    coords = mesh.nodes_array()

    # 2. Material Definitions
    # PID 1: PET (E=4000 MPa, nu=0.3) -> C3D8I_CR
    # PID 2: PSA (E=1.0 MPa, nu=0.499) -> C3D8H_CR (nearly incompressible)
    # PID 3: Rigid Plate (E=2.0e5 MPa, nu=0.3)
    materials = {
        1: {"E": 4000.0, "nu": 0.3},
        2: {"E": 1.0, "nu": 0.499},
        3: {"E": 2.0e5, "nu": 0.3},
    }

    solver = DynamicSolver3D(mesh, materials, nlgeom=True)

    # 3. Setup 3D Surface Tie Constraints
    # Left Tie
    tie_left = SurfaceTieConstraint3D(
        slave_node_ids=meta["disp_bottom_left_nids"],
        master_faces=meta["plate_l_top_faces"],
        nid_to_idx=nid_map,
        coords=coords,
        penalty_stiffness=penalty_tie,
        position_tolerance=0.2,
        name="Tie_Left"
    )
    # Right Tie
    tie_right = SurfaceTieConstraint3D(
        slave_node_ids=meta["disp_bottom_right_nids"],
        master_faces=meta["plate_r_top_faces"],
        nid_to_idx=nid_map,
        coords=coords,
        penalty_stiffness=penalty_tie,
        position_tolerance=0.2,
        name="Tie_Right"
    )

    solver.add_constraint(tie_left)
    solver.add_constraint(tie_right)

    print(f"[*] Mesh statistics: {len(mesh.nodes)} nodes, {len(mesh.elements)} 3D solid elements, {solver.num_dofs} DOFs.")
    print(f"[*] Surface Ties: Left has {len(tie_left.pairs)} coupled pairs, Right has {len(tie_right.pairs)} coupled pairs.")

    # 4. Kinematic Rotation Ramp: Smootherstep
    # s in [0, 1] -> theta(s) = theta_max * (10s^3 - 15s^4 + 6s^5)
    def smooth_ramp(s: float) -> float:
        s = np.clip(s, 0.0, 1.0)
        return float(10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5)

    th_max_rad = np.radians(theta_max_deg)
    pivot_L = meta["left_pivot"]
    pivot_R = meta["right_pivot"]

    # History recording
    history_time = []
    history_theta = []
    history_iters = []
    history_tie_gap_L = []
    history_tie_gap_R = []

    dt = 1.0 / float(n_steps)
    t = 0.0
    step = 0

    while t < 1.0 - 1e-6:
        step += 1
        t_next = min(1.0, t + dt)
        s = t_next
        theta_curr = th_max_rad * smooth_ramp(s)

        # Apply Exact Kinematic Rigid-Body Dirichlet BCs to Plate Nodes
        # Left Plate: rotates by +theta about Left Pivot (-3.0, 0.0) (folds upward)
        # R_z(theta): [cos, -sin; sin, cos]
        cos_th_L = np.cos(theta_curr)
        sin_th_L = np.sin(theta_curr)
        for nid in meta["plate_left_nodes"]:
            node = mesh.nodes[nid]
            dx0 = node.x - pivot_L[0]
            dy0 = node.y - pivot_L[1]
            x_new = pivot_L[0] + cos_th_L * dx0 - sin_th_L * dy0
            y_new = pivot_L[1] + sin_th_L * dx0 + cos_th_L * dy0
            ux_target = x_new - node.x
            uy_target = y_new - node.y

            solver.fix_dof(nid, 0, ux_target)
            solver.fix_dof(nid, 1, uy_target)
            solver.fix_dof(nid, 2, 0.0)

        # Right Plate: rotates by -theta about Right Pivot (+3.0, 0.0)
        cos_th_R = np.cos(-theta_curr)
        sin_th_R = np.sin(-theta_curr)
        for nid in meta["plate_right_nodes"]:
            node = mesh.nodes[nid]
            dx0 = node.x - pivot_R[0]
            dy0 = node.y - pivot_R[1]
            x_new = pivot_R[0] + cos_th_R * dx0 - sin_th_R * dy0
            y_new = pivot_R[1] + sin_th_R * dx0 + cos_th_R * dy0
            ux_target = x_new - node.x
            uy_target = y_new - node.y

            solver.fix_dof(nid, 0, ux_target)
            solver.fix_dof(nid, 1, uy_target)
            solver.fix_dof(nid, 2, 0.0)

        # Fix display Z=0 symmetry / plane strain
        for nid in mesh.nodes.keys():
            if abs(mesh.nodes[nid].z) < 1e-4:
                solver.fix_dof(nid, 2, 0.0)

        # Solve nonlinear step
        converged, iters = solver.solve_step(dt=dt, max_iters=25)

        # Monitor tie gap
        _, _, stats_L = tie_left.assemble(solver.u)
        _, _, stats_R = tie_right.assemble(solver.u)
        gap_L = stats_L.get("max_gap", 0.0)
        gap_R = stats_R.get("max_gap", 0.0)

        history_time.append(t_next)
        history_theta.append(np.degrees(theta_curr))
        history_iters.append(iters)
        history_tie_gap_L.append(gap_L)
        history_tie_gap_R.append(gap_R)

        deg_disp = np.degrees(theta_curr)
        print(f"  Step {step:2d} | t={t_next:.3f} | Theta={deg_disp:5.1f}° | Iters={iters:2d} | Conv={converged} | TieGap_L={gap_L*1e3:6.2f}um | TieGap_R={gap_R*1e3:6.2f}um")

        if not converged:
            print(f"[!] Warning: Step {step} did not converge within max iterations.")

        t = t_next

    wall_time = time.perf_counter() - t_start
    print("=" * 90)
    print(f"[OK] Full 3D Multi-Layer Folding Completed in {wall_time:.2f}s!")
    print(f"     Final Rotation: {history_theta[-1]:.1f}° per plate (Total Fold: {2*history_theta[-1]:.1f}°)")
    print(f"     Max Surface Tie Gap: Left = {max(history_tie_gap_L)*1e3:.2f} um, Right = {max(history_tie_gap_R)*1e3:.2f} um")
    print("=" * 90)

    # Save results
    out_dir = os.path.join(REPO_ROOT, "examples", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_pkl = os.path.join(out_dir, "ex13_3d_fold_result.pkl")

    res_data = {
        "mesh": mesh,
        "u_final": solver.u,
        "history_time": history_time,
        "history_theta": history_theta,
        "history_iters": history_iters,
        "history_tie_gap_L": history_tie_gap_L,
        "history_tie_gap_R": history_tie_gap_R,
        "wall_time_s": wall_time,
        "meta": meta
    }

    with open(out_pkl, "wb") as f:
        pickle.dump(res_data, f)
    print(f"Saved 3D fold results to: {out_pkl}")

    return res_data


if __name__ == "__main__":
    run_3d_display_plate_fold(theta_max_deg=90.0, n_steps=40)
