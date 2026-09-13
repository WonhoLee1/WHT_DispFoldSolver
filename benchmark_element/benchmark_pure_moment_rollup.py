"""
benchmark_pure_moment_rollup.py
===============================
Geometrically Nonlinear Pure Bending Moment Roll-Up Benchmark for 5-Layer Composite.

Rolls up a 40 mm x 0.21 mm 5-layer composite beam (PET-PSA-PET-PSA-PET) into a
perfect 180-degree circular arc (half circle / U-shape with R = L/pi = 12.732 mm)
under pure kinematic moment rotation.

Compares:
1. 2D Candidates:
   - 2D-Opt1: CPE4I + CPE4H (Conformal)
   - 2D-Opt2: CPE4R + CPE4H (Conformal)
   - 2D-Base: CPE4  + CPE4  (Baseline)
2. 3D Candidates:
   - 3D-Opt1: C3D8I + C3D8H (Conformal)
   - 3D-Opt2: C3D8R + C3D8R (Conformal)
   - 3D-Base: C3D8  + C3D8  (Baseline)

Metrics:
- Maximum reached rotation angle theta_max (target 180 deg)
- Circularity RMSE % (deviation from exact circle R = 12.732 mm)
- Mid-span interlayer shear slip Delta u_mid [um] (at x = 20 mm)
- Total wall-clock time [s] and Newton iterations
"""

from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

# Problem Constants
L_BEAM = 40.0       # mm
T_PET = 0.05        # mm (50 um)
T_PSA = 0.03        # mm (30 um)
TOTAL_H = 3 * T_PET + 2 * T_PSA  # 0.21 mm (210 um)
Z_MID = TOTAL_H / 2.0            # 0.105 mm
WIDTH = 1.0         # mm

E_PET = 4000.0      # MPa
NU_PET = 0.3

MU_PSA = 0.05       # MPa
NU_PSA = 0.499
E_PSA = 2.0 * MU_PSA * (1.0 + NU_PSA)  # ~0.15 MPa

PID_PET = 1
PID_PSA = 2

# Theoretical Radius for theta = 180 deg (pi rad)
THETA_MAX = np.pi   # 180 degrees
R_THEORY = L_BEAM / np.pi  # 12.732395 mm


def save_case_mesh_cache(label: str, mesh: Any, u: np.ndarray, is_2d: bool):
    """Saves final deformed mesh and displacement state as a lightweight NPZ for PyVista visualization."""
    if not label:
        return
    out_dir = Path(__file__).resolve().parent / "results" / "meshes"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = label.split()[0].replace("-", "_")
    cache_path = out_dir / f"{safe_name}.npz"

    nid_map = mesh.node_id_to_index()
    n_nodes = len(mesh.nodes)
    if is_2d:
        coords = np.zeros((n_nodes, 2), dtype=np.float64)
        for nid, idx in nid_map.items():
            coords[idx, 0] = mesh.nodes[nid].x
            coords[idx, 1] = mesh.nodes[nid].y
    else:
        coords = np.zeros((n_nodes, 3), dtype=np.float64)
        for nid, idx in nid_map.items():
            coords[idx, 0] = mesh.nodes[nid].x
            coords[idx, 1] = mesh.nodes[nid].y
            coords[idx, 2] = mesh.nodes[nid].z

    elem_conn = []
    elem_pids = []
    elem_types = []
    for eid, elem in mesh.elements.items():
        conn_0 = [nid_map[n] for n in elem.node_ids]
        elem_conn.append(conn_0)
        elem_pids.append(getattr(elem, "pid", 0))
        elem_types.append(elem.elem_type)

    np.savez_compressed(
        cache_path,
        label=label,
        is_2d=is_2d,
        coords=coords,
        u_disp=u,
        elem_conn=np.array(elem_conn, dtype=object),
        elem_pids=np.array(elem_pids, dtype=np.int32),
        elem_types=np.array(elem_types, dtype=object)
    )


def make_2d_mesh(pet_type: str, psa_type: str, nx: int = 40) -> Tuple[Mesh2D, List[int], List[int], np.ndarray]:
    mesh = Mesh2D()
    layer_thicknesses = [T_PET, T_PSA, T_PET, T_PSA, T_PET]
    layer_types = [pet_type, psa_type, pet_type, psa_type, pet_type]
    layer_pids = [PID_PET, PID_PSA, PID_PET, PID_PSA, PID_PET]

    dx = L_BEAM / nx
    y_coords = [0.0]
    for t in layer_thicknesses:
        y_coords.append(y_coords[-1] + t)

    node_id = 1
    grid = np.zeros((nx + 1, 6), dtype=np.int32)
    for j in range(6):
        y_val = y_coords[j]
        for i in range(nx + 1):
            mesh.add_node(node_id, i * dx, y_val)
            grid[i, j] = node_id
            node_id += 1

    elem_id = 1
    for j in range(5):
        etype = layer_types[j]
        pid = layer_pids[j]
        for i in range(nx):
            n1 = int(grid[i, j])
            n2 = int(grid[i + 1, j])
            n3 = int(grid[i + 1, j + 1])
            n4 = int(grid[i, j + 1])
            mesh.add_element(elem_id, [n1, n2, n3, n4], elem_type=etype, pid=pid)
            elem_id += 1

    root_nodes = sorted(list(grid[0, :]))
    tip_nodes = sorted(list(grid[-1, :]))
    return mesh, root_nodes, tip_nodes, grid


def make_3d_mesh(pet_type: str, psa_type: str, nx: int = 40) -> Tuple[Mesh3D, List[int], List[int], np.ndarray]:
    mesh = Mesh3D()
    layer_thicknesses = [T_PET, T_PSA, T_PET, T_PSA, T_PET]
    layer_types = [pet_type, psa_type, pet_type, psa_type, pet_type]
    layer_pids = [PID_PET, PID_PSA, PID_PET, PID_PSA, PID_PET]

    dx = L_BEAM / nx
    z_coords = [0.0]
    for t in layer_thicknesses:
        z_coords.append(z_coords[-1] + t)

    node_id = 1
    grid = np.zeros((nx + 1, 2, 6), dtype=np.int32)
    for k in range(6):
        z_val = z_coords[k]
        for j in range(2):
            y_val = j * WIDTH
            for i in range(nx + 1):
                mesh.add_node(node_id, i * dx, y_val, z_val)
                grid[i, j, k] = node_id
                node_id += 1

    elem_id = 1
    for k in range(5):
        etype = layer_types[k]
        pid = layer_pids[k]
        for i in range(nx):
            n1 = int(grid[i, 0, k])
            n2 = int(grid[i + 1, 0, k])
            n3 = int(grid[i + 1, 1, k])
            n4 = int(grid[i, 1, k])
            n5 = int(grid[i, 0, k + 1])
            n6 = int(grid[i + 1, 0, k + 1])
            n7 = int(grid[i + 1, 1, k + 1])
            n8 = int(grid[i, 1, k + 1])
            mesh.add_element(elem_id, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type=etype, pid=pid)
            elem_id += 1

    root_nodes = sorted(list(grid[0, :, :].ravel()))
    tip_nodes = sorted(list(grid[-1, :, :].ravel()))
    return mesh, root_nodes, tip_nodes, grid


def compute_roll_up_displacements(z_val: float, theta: float) -> Tuple[float, float]:
    """Exact kinematic RBE2 cross-section roll-up displacement for pure moment."""
    if abs(theta) < 1e-7:
        ux = -(z_val - Z_MID) * theta
        uz = 0.0
    else:
        R = L_BEAM / theta
        ux = (R - z_val + Z_MID) * np.sin(theta) - L_BEAM
        uz = R * (1.0 - np.cos(theta)) + (z_val - Z_MID) * (np.cos(theta) - 1.0)
    return ux, uz


def run_rollup_2d(pet_type: str, psa_type: str, n_substeps: int = 25, label: str = "") -> Dict[str, Any]:
    if pet_type.upper() == "CPE6M":
        from dispsolver.mesh2d.conformal_cpe6m_cpe4h_builder import create_5layer_cpe6m_cpe4h_conformal_mesh
        mesh, sets = create_5layer_cpe6m_cpe4h_conformal_mesh(
            length=L_BEAM, t_pet=T_PET, t_psa=T_PSA, nx=40, pid_pet=PID_PET, pid_psa=PID_PSA
        )
        root_nodes = sets["LEFT_NODES"]
        tip_nodes = sets["RIGHT_NODES"]
        centerline_nodes = sorted(sets["CENTERLINE_NODES"], key=lambda n: mesh.nodes[n].x)
        grid = None
    else:
        mesh, root_nodes, tip_nodes, grid = make_2d_mesh(pet_type, psa_type)
        centerline_nodes = None

    materials = {
        PID_PET: {"E": E_PET, "nu": NU_PET},
        PID_PSA: {"E": E_PSA, "nu": NU_PSA}
    }
    solver = DynamicSolver2D(mesh, materials=materials, nlgeom=True)

    # Clamped Root
    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)

    nid_map = mesh.node_id_to_index()
    t_start = time.perf_counter()
    total_iters = 0
    max_theta_reached = 0.0

    hist_theta = [0.0]
    hist_M_root = [0.0]
    hist_M_tip = [0.0]
    hist_Fx_root = [0.0]
    hist_Fy_root = [0.0]
    hist_U_strain = [0.0]

    for s in range(1, n_substeps + 1):
        frac = float(s) / float(n_substeps)
        ramp = frac * frac * (3.0 - 2.0 * frac)
        theta_s = ramp * THETA_MAX

        # Update tip cross-section kinematic rotation BCs
        for nid in tip_nodes:
            node = mesh.nodes[nid]
            ux_t, uy_t = compute_roll_up_displacements(node.y, theta_s)
            solver.fix_dof(nid, 0, ux_t)
            solver.fix_dof(nid, 1, uy_t)

        conv, iters = solver.solve_step(dt=1.0, max_iters=35)
        total_iters += iters
        if conv:
            max_theta_reached = theta_s
            root_rx = solver.compute_section_reactions(root_nodes, center_pt=(0.0, Z_MID))
            tip_x_mean = float(np.mean([mesh.nodes[n].x + solver.u[2 * nid_map[n]] for n in tip_nodes]))
            tip_y_mean = float(np.mean([mesh.nodes[n].y + solver.u[2 * nid_map[n] + 1] for n in tip_nodes]))
            tip_rx = solver.compute_section_reactions(tip_nodes, center_pt=(tip_x_mean, tip_y_mean))

            hist_theta.append(float(np.degrees(theta_s)))
            hist_M_root.append(float(root_rx["Mz"]))
            hist_M_tip.append(float(tip_rx["Mz"]))
            hist_Fx_root.append(float(root_rx["Fx"]))
            hist_Fy_root.append(float(root_rx["Fy"]))

            u_str = solver.compute_strain_energy() * 1000.0  # mJ
            hist_U_strain.append(float(u_str))
        else:
            print(f"    [2D {pet_type}+{psa_type}] Cutback/Diverged at step {s}/{n_substeps} (theta = {np.degrees(theta_s):.1f} deg)")
            break

    t_wall = time.perf_counter() - t_start

    # Extract deformed centerline profile along length
    tau_x_coords = []
    tau_xy_lower = []
    tau_xy_upper = []

    if grid is not None:
        nx = grid.shape[0] - 1
        prof_x = []
        prof_y = []
        for i in range(nx + 1):
            mid_nodes = [int(grid[i, 2]), int(grid[i, 3])]
            cx = np.mean([mesh.nodes[n].x + solver.u[2 * nid_map[n]] for n in mid_nodes])
            cy = np.mean([mesh.nodes[n].y + solver.u[2 * nid_map[n] + 1] for n in mid_nodes])
            prof_x.append(float(cx))
            prof_y.append(float(cy))

            # Shear stress in PSA layers
            x_val = i * (L_BEAM / nx)
            n_bot1, n_top1 = int(grid[i, 1]), int(grid[i, 2])
            ux_b1, ux_t1 = solver.u[2 * nid_map[n_bot1]], solver.u[2 * nid_map[n_top1]]
            tau_l = MU_PSA * (ux_t1 - ux_b1) / T_PSA

            n_bot2, n_top2 = int(grid[i, 3]), int(grid[i, 4])
            ux_b2, ux_t2 = solver.u[2 * nid_map[n_bot2]], solver.u[2 * nid_map[n_top2]]
            tau_u = MU_PSA * (ux_t2 - ux_b2) / T_PSA

            tau_x_coords.append(float(x_val))
            tau_xy_lower.append(float(tau_l))
            tau_xy_upper.append(float(tau_u))

        mid_i = nx // 2
        bot_node = int(grid[mid_i, 0])
        top_node = int(grid[mid_i, -1])
        ux_bot = solver.u[2 * nid_map[bot_node]]
        ux_top = solver.u[2 * nid_map[top_node]]
        shear_slip_mid_um = abs(ux_top - ux_bot) * 1000.0  # um

        tip_mid_nodes = [int(grid[-1, 2]), int(grid[-1, 3])]
        final_tip_x = float(np.mean([mesh.nodes[n].x + solver.u[2 * nid_map[n]] for n in tip_mid_nodes]))
        final_tip_y = float(np.mean([mesh.nodes[n].y + solver.u[2 * nid_map[n] + 1] for n in tip_mid_nodes]))
    else:
        prof_x = [float(mesh.nodes[n].x + solver.u[2 * nid_map[n]]) for n in centerline_nodes]
        prof_y = [float(mesh.nodes[n].y + solver.u[2 * nid_map[n] + 1]) for n in centerline_nodes]

        bot_node = min(mesh.nodes.keys(), key=lambda n: abs(mesh.nodes[n].x - 20.0) + abs(mesh.nodes[n].y - 0.0))
        top_node = min(mesh.nodes.keys(), key=lambda n: abs(mesh.nodes[n].x - 20.0) + abs(mesh.nodes[n].y - TOTAL_H))
        ux_bot = solver.u[2 * nid_map[bot_node]]
        ux_top = solver.u[2 * nid_map[top_node]]
        shear_slip_mid_um = abs(ux_top - ux_bot) * 1000.0

        tip_c = centerline_nodes[-1]
        final_tip_x = float(mesh.nodes[tip_c].x + solver.u[2 * nid_map[tip_c]])
        final_tip_y = float(mesh.nodes[tip_c].y + solver.u[2 * nid_map[tip_c] + 1])

    # Circularity RMSE % against theoretical circle center (0, Z_MID + R_THEORY)
    r_errors = []
    for px, py in zip(prof_x, prof_y):
        dist = np.sqrt(px**2 + (py - (Z_MID + R_THEORY))**2)
        r_errors.append(dist - R_THEORY)
    circularity_rmse_pct = (np.sqrt(np.mean(np.square(r_errors))) / R_THEORY) * 100.0

    is_u_shape = bool(np.mean(prof_x) > 0.3 * R_THEORY)
    shape_verdict = "True U-Shape Arc" if is_u_shape else "Collapsed (Linear Squashing)"

    # Final section reaction and energy summary
    final_M_root = hist_M_root[-1]
    final_M_tip = hist_M_tip[-1]
    final_Fx_root = hist_Fx_root[-1]
    final_Fy_root = hist_Fy_root[-1]
    final_U_strain = hist_U_strain[-1]

    save_case_mesh_cache(label, mesh, solver.u, is_2d=True)

    return {
        "max_theta_deg": float(np.degrees(max_theta_reached)),
        "completed_180": bool(abs(max_theta_reached - THETA_MAX) < 1e-4),
        "is_u_shape": is_u_shape,
        "shape_verdict": shape_verdict,
        "circularity_rmse_pct": round(float(circularity_rmse_pct), 3),
        "shear_slip_mid_um": round(float(shear_slip_mid_um), 2),
        "final_tip_x": round(final_tip_x, 3),
        "final_tip_y": round(final_tip_y, 3),
        "t_wall_sec": round(t_wall, 3),
        "total_iters": total_iters,
        "final_M_root": round(final_M_root, 6),
        "final_M_tip": round(final_M_tip, 6),
        "final_Fx_root": round(final_Fx_root, 6),
        "final_Fy_root": round(final_Fy_root, 6),
        "final_U_strain_mJ": round(final_U_strain, 4),
        "history_theta": hist_theta,
        "history_M_root": hist_M_root,
        "history_M_tip": hist_M_tip,
        "history_Fx_root": hist_Fx_root,
        "history_Fy_root": hist_Fy_root,
        "history_U_strain": hist_U_strain,
        "tau_x_coords": tau_x_coords,
        "tau_xy_lower": tau_xy_lower,
        "tau_xy_upper": tau_xy_upper,
        "profile_x": prof_x,
        "profile_y": prof_y
    }



def run_rollup_3d(pet_type: str, psa_type: str, n_substeps: int = 25, label: str = "") -> Dict[str, Any]:
    mesh, root_nodes, tip_nodes, grid = make_3d_mesh(pet_type, psa_type)
    materials = {
        PID_PET: {"E": E_PET, "nu": NU_PET},
        PID_PSA: {"E": E_PSA, "nu": NU_PSA}
    }
    solver = DynamicSolver3D(mesh, material_params={"E": E_PET, "nu": NU_PET}, nlgeom=True)
    solver.materials = materials
    solver._setup_numba_topology()

    # Clamped Root
    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)

    # 1-element strip plane-strain condition (fix uy)
    for nid in mesh.nodes.keys():
        solver.fix_dof(nid, 1, 0.0)

    nid_map = mesh.node_id_to_index()
    t_start = time.perf_counter()
    total_iters = 0
    max_theta_reached = 0.0

    hist_theta = [0.0]
    hist_M_root = [0.0]
    hist_M_tip = [0.0]
    hist_Fx_root = [0.0]
    hist_Fz_root = [0.0]
    hist_U_strain = [0.0]

    for s in range(1, n_substeps + 1):
        frac = float(s) / float(n_substeps)
        ramp = frac * frac * (3.0 - 2.0 * frac)
        theta_s = ramp * THETA_MAX

        # Update tip cross-section kinematic rotation BCs
        for nid in tip_nodes:
            node = mesh.nodes[nid]
            ux_t, uz_t = compute_roll_up_displacements(node.z, theta_s)
            solver.fix_dof(nid, 0, ux_t)
            solver.fix_dof(nid, 2, uz_t)

        conv, iters = solver.solve_step(dt=1.0, max_iters=35)
        total_iters += iters
        if conv:
            max_theta_reached = theta_s
            root_rx = solver.compute_section_reactions(root_nodes, center_pt=(0.0, 0.5 * WIDTH, Z_MID))
            tip_x_mean = float(np.mean([mesh.nodes[n].x + solver.u[3 * nid_map[n]] for n in tip_nodes]))
            tip_z_mean = float(np.mean([mesh.nodes[n].z + solver.u[3 * nid_map[n] + 2] for n in tip_nodes]))
            tip_rx = solver.compute_section_reactions(tip_nodes, center_pt=(tip_x_mean, 0.5 * WIDTH, tip_z_mean))

            hist_theta.append(float(np.degrees(theta_s)))
            # My is the bending moment about thickness; normalize by WIDTH
            hist_M_root.append(float(root_rx["My"] / WIDTH))
            hist_M_tip.append(float(tip_rx["My"] / WIDTH))
            hist_Fx_root.append(float(root_rx["Fx"] / WIDTH))
            hist_Fz_root.append(float(root_rx["Fz"] / WIDTH))

            u_str = (solver.compute_strain_energy() / WIDTH) * 1000.0  # mJ per mm width
            hist_U_strain.append(float(u_str))
        else:
            print(f"    [3D {pet_type}+{psa_type}] Cutback/Diverged at step {s}/{n_substeps} (theta = {np.degrees(theta_s):.1f} deg)")
            break

    t_wall = time.perf_counter() - t_start

    # Extract deformed centerline profile along length (center in y and z)
    nx = grid.shape[0] - 1
    prof_x = []
    prof_z = []
    tau_x_coords = []
    tau_xy_lower = []
    tau_xy_upper = []

    for i in range(nx + 1):
        mid_nodes = [int(grid[i, 0, 2]), int(grid[i, 0, 3]), int(grid[i, 1, 2]), int(grid[i, 1, 3])]
        cx = np.mean([mesh.nodes[n].x + solver.u[3 * nid_map[n]] for n in mid_nodes])
        cz = np.mean([mesh.nodes[n].z + solver.u[3 * nid_map[n] + 2] for n in mid_nodes])
        prof_x.append(float(cx))
        prof_z.append(float(cz))

        # Shear stress in PSA layers (tau_xz)
        x_val = i * (L_BEAM / nx)
        # Lower PSA: between k=1 and k=2
        ux_b1 = 0.5 * (solver.u[3 * nid_map[int(grid[i, 0, 1])]] + solver.u[3 * nid_map[int(grid[i, 1, 1])]])
        ux_t1 = 0.5 * (solver.u[3 * nid_map[int(grid[i, 0, 2])]] + solver.u[3 * nid_map[int(grid[i, 1, 2])]])
        tau_l = MU_PSA * (ux_t1 - ux_b1) / T_PSA

        # Upper PSA: between k=3 and k=4
        ux_b2 = 0.5 * (solver.u[3 * nid_map[int(grid[i, 0, 3])]] + solver.u[3 * nid_map[int(grid[i, 1, 3])]])
        ux_t2 = 0.5 * (solver.u[3 * nid_map[int(grid[i, 0, 4])]] + solver.u[3 * nid_map[int(grid[i, 1, 4])]])
        tau_u = MU_PSA * (ux_t2 - ux_b2) / T_PSA

        tau_x_coords.append(float(x_val))
        tau_xy_lower.append(float(tau_l))
        tau_xy_upper.append(float(tau_u))

    # Circularity RMSE % against theoretical circle center (0, Z_MID + R_THEORY)
    r_errors = []
    for px, pz in zip(prof_x, prof_z):
        dist = np.sqrt(px**2 + (pz - (Z_MID + R_THEORY))**2)
        r_errors.append(dist - R_THEORY)
    circularity_rmse_pct = (np.sqrt(np.mean(np.square(r_errors))) / R_THEORY) * 100.0

    # Mid-span shear slip: between top PET and bottom PET at x = 20 mm
    mid_i = nx // 2
    bot_nodes = [int(grid[mid_i, 0, 0]), int(grid[mid_i, 1, 0])]
    top_nodes = [int(grid[mid_i, 0, -1]), int(grid[mid_i, 1, -1])]
    ux_bot = np.mean([solver.u[3 * nid_map[n]] for n in bot_nodes])
    ux_top = np.mean([solver.u[3 * nid_map[n]] for n in top_nodes])
    shear_slip_mid_um = abs(ux_top - ux_bot) * 1000.0  # um

    # Final tip position
    tip_mid_nodes = [int(grid[-1, 0, 2]), int(grid[-1, 0, 3]), int(grid[-1, 1, 2]), int(grid[-1, 1, 3])]
    final_tip_x = float(np.mean([mesh.nodes[n].x + solver.u[3 * nid_map[n]] for n in tip_mid_nodes]))
    final_tip_z = float(np.mean([mesh.nodes[n].z + solver.u[3 * nid_map[n] + 2] for n in tip_mid_nodes]))

    is_u_shape = bool(np.mean(prof_x) > 0.3 * R_THEORY)
    shape_verdict = "True U-Shape Arc" if is_u_shape else "Collapsed (Linear Squashing)"

    final_M_root = hist_M_root[-1]
    final_M_tip = hist_M_tip[-1]
    final_Fx_root = hist_Fx_root[-1]
    final_Fz_root = hist_Fz_root[-1]
    final_U_strain = hist_U_strain[-1]

    save_case_mesh_cache(label, mesh, solver.u, is_2d=False)

    return {
        "max_theta_deg": float(np.degrees(max_theta_reached)),
        "completed_180": bool(abs(max_theta_reached - THETA_MAX) < 1e-4),
        "is_u_shape": is_u_shape,
        "shape_verdict": shape_verdict,
        "circularity_rmse_pct": round(float(circularity_rmse_pct), 3),
        "shear_slip_mid_um": round(float(shear_slip_mid_um), 2),
        "final_tip_x": round(final_tip_x, 3),
        "final_tip_y": round(final_tip_z, 3),
        "t_wall_sec": round(t_wall, 3),
        "total_iters": total_iters,
        "final_M_root": round(final_M_root, 6),
        "final_M_tip": round(final_M_tip, 6),
        "final_Fx_root": round(final_Fx_root, 6),
        "final_Fy_root": round(final_Fz_root, 6),
        "final_U_strain_mJ": round(final_U_strain, 4),
        "history_theta": hist_theta,
        "history_M_root": hist_M_root,
        "history_M_tip": hist_M_tip,
        "history_Fx_root": hist_Fx_root,
        "history_Fy_root": hist_Fz_root,
        "history_U_strain": hist_U_strain,
        "tau_x_coords": tau_x_coords,
        "tau_xy_lower": tau_xy_lower,
        "tau_xy_upper": tau_xy_upper,
        "profile_x": prof_x,
        "profile_y": prof_z
    }



def run_all_rollups():
    print("=" * 80, flush=True)
    print("5-LAYER COMPOSITE PURE BENDING MOMENT ROLL-UP BENCHMARK (180 DEG U-TURN)", flush=True)
    print("L=40.0mm, H=0.21mm, Target R=12.732mm, End Rotation theta=180 deg", flush=True)
    print("=" * 80, flush=True)

    results = {"2D": {}, "3D": {}}

    # 2D Candidate Suite
    suite_2d = [
        ("2D-CR   (CPE4_CR+CPE4_CR)", "CPE4_CR", "CPE4_CR"),
        ("2D-Opt1 (CPE4I+CPE4H)",     "CPE4I",   "CPE4H"),
        ("2D-CPE6M (CPE6M+CPE4H)",   "CPE6M",   "CPE4H"),
        ("2D-Opt2 (CPE4R+CPE4H)",     "CPE4R",   "CPE4H"),
        ("2D-Base (CPE4+CPE4)",       "CPE4",    "CPE4"),
    ]

    print(f"\n>>> Running 2D Candidate Roll-Up Suite ({len(suite_2d)} Cases) ...", flush=True)
    for label, pet_e, psa_e in suite_2d:
        print(f"\n[2D] {label} ...", flush=True)
        res = run_rollup_2d(pet_e, psa_e, n_substeps=25, label=label)
        results["2D"][label] = res
        print(f"     Max Theta: {res['max_theta_deg']:.1f} deg | Verdict: {res['shape_verdict']} | Circularity RMSE: {res['circularity_rmse_pct']:.3f}% | Mid Slip: {res['shear_slip_mid_um']:.1f} um", flush=True)
        print(f"     M_root: {res['final_M_root']:.6f} N*mm | M_tip: {res['final_M_tip']:.6f} N*mm | U_strain: {res['final_U_strain_mJ']:.2f} mJ | Fx: {res['final_Fx_root']:.2e} N", flush=True)
        print(f"     Tip Position: (x={res['final_tip_x']:.2f}, y={res['final_tip_y']:.2f}) mm [Theory: (0.0, {Z_MID + 2*R_THEORY:.2f})]", flush=True)
        print(f"     Time: {res['t_wall_sec']:.3f} s | Total Iters: {res['total_iters']}", flush=True)

    # 3D Candidate Suite
    suite_3d = [
        ("3D-CR   (C3D8_CR+C3D8_CR)", "C3D8_CR", "C3D8_CR"),
        ("3D-Opt1 (C3D8I+C3D8H)",     "C3D8I",   "C3D8H"),
        ("3D-Opt2-H (C3D8R+C3D8H)",   "C3D8R",   "C3D8H"),
        ("3D-Opt2 (C3D8R+C3D8R)",     "C3D8R",   "C3D8R"),
        ("3D-Base (C3D8+C3D8)",       "C3D8",    "C3D8"),
    ]

    print(f"\n>>> Running 3D Candidate Roll-Up Suite ({len(suite_3d)} Cases) ...", flush=True)
    for label, pet_e, psa_e in suite_3d:
        print(f"\n[3D] {label} ...", flush=True)
        res = run_rollup_3d(pet_e, psa_e, n_substeps=25, label=label)
        results["3D"][label] = res
        print(f"     Max Theta: {res['max_theta_deg']:.1f} deg | Verdict: {res['shape_verdict']} | Circularity RMSE: {res['circularity_rmse_pct']:.3f}% | Mid Slip: {res['shear_slip_mid_um']:.1f} um", flush=True)
        print(f"     M_root: {res['final_M_root']:.6f} N*mm | M_tip: {res['final_M_tip']:.6f} N*mm | U_strain: {res['final_U_strain_mJ']:.2f} mJ | Fx: {res['final_Fx_root']:.2e} N", flush=True)
        print(f"     Tip Position: (x={res['final_tip_x']:.2f}, y={res['final_tip_y']:.2f}) mm [Theory: (0.0, {Z_MID + 2*R_THEORY:.2f})]", flush=True)
        print(f"     Time: {res['t_wall_sec']:.3f} s | Total Iters: {res['total_iters']}", flush=True)


    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "benchmark_pure_moment_rollup.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Roll-up benchmark completed and saved to {json_path}", flush=True)
    return results


if __name__ == "__main__":
    run_all_rollups()
