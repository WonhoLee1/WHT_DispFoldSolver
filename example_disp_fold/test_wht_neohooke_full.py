import math
import os
import sys
import time
import json
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from dispsolver.solver2d.dynamic2d import DynamicSolver2D

def test_wht_2d_neohooke(substeps=45):
    config = UnifiedFoldModelConfig(preset="teardrop", dim="2D", elem_case="2D-Opt1")
    config.geometry.width_z = 1.0
    config.mesh.nz_elements = 1

    mesher = GmshMeshGenerator(config)
    mesh, meta = mesher.build()

    materials = {
        1: {"E": 4000.0, "nu": 0.30},
        2: {"type": "NEOHOOKEAN", "c10": 8.3388926e-3, "d1": 0.24, "nu": 0.499},
        3: {"E": 70000.0, "nu": 0.22},
        4: {"E": 200000.0, "nu": 0.30},
    }

    solver = DynamicSolver2D(mesh, materials=materials, nlgeom=True)
    solver.max_du_limit = 1.0

    pivot_L = meta["left_pivot"]
    pivot_R = meta["right_pivot"]
    hinge_gap = config.geometry.hinge_half_gap
    disp_left_plate_nodes = [nid for nid, node in mesh.nodes.items() if node.x <= -hinge_gap + 1e-4 and node.y >= 0.0]
    disp_right_plate_nodes = [nid for nid, node in mesh.nodes.items() if node.x >= hinge_gap - 1e-4 and node.y >= 0.0]
    nid_map = mesh.node_id_to_index()

    theta_max_rad = math.radians(90.0)
    theta_prev = 0.0

    print(f"[*] Starting WHT 2D Neo-Hookean Folding (Substeps={substeps}, 2 deg/step, Total DOFs={solver.num_dofs})...", flush=True)
    t0 = time.perf_counter()

    history = []

    for step in range(1, substeps + 1):
        theta_curr = theta_max_rad * (step / float(substeps))
        deg_curr = math.degrees(theta_curr)

        # 1. Left plate Dirichlet BCs
        cos_th_L = math.cos(-theta_curr)
        sin_th_L = math.sin(-theta_curr)
        for nid in meta["plate_left_nodes"] + meta["disp_bottom_left_nids"]:
            node = mesh.nodes[nid]
            dx0 = node.x - pivot_L[0]
            dy0 = node.y - pivot_L[1]
            solver.fix_dof(nid, 0, pivot_L[0] + cos_th_L * dx0 - sin_th_L * dy0 - node.x)
            solver.fix_dof(nid, 1, pivot_L[1] + sin_th_L * dx0 + cos_th_L * dy0 - node.y)

        # 2. Right plate Dirichlet BCs
        cos_th_R = math.cos(theta_curr)
        sin_th_R = math.sin(theta_curr)
        for nid in meta["plate_right_nodes"] + meta["disp_bottom_right_nids"]:
            node = mesh.nodes[nid]
            dx0 = node.x - pivot_R[0]
            dy0 = node.y - pivot_R[1]
            solver.fix_dof(nid, 0, pivot_R[0] + cos_th_R * dx0 - sin_th_R * dy0 - node.x)
            solver.fix_dof(nid, 1, pivot_R[1] + sin_th_R * dx0 + cos_th_R * dy0 - node.y)

        # 3. Kinematic rotation predictor
        dth_L = -(theta_curr - theta_prev)
        cos_dth_L = math.cos(dth_L)
        sin_dth_L = math.sin(dth_L)
        for nid in disp_left_plate_nodes:
            node = mesh.nodes[nid]
            idx = nid_map[nid]
            cx = node.x + solver.u[idx * 2 + 0]
            cy = node.y + solver.u[idx * 2 + 1]
            dx = cx - pivot_L[0]
            dy = cy - pivot_L[1]
            solver.u[idx * 2 + 0] = pivot_L[0] + cos_dth_L * dx - sin_dth_L * dy - node.x
            solver.u[idx * 2 + 1] = pivot_L[1] + sin_dth_L * dx + cos_dth_L * dy - node.y

        dth_R = +(theta_curr - theta_prev)
        cos_dth_R = math.cos(dth_R)
        sin_dth_R = math.sin(dth_R)
        for nid in disp_right_plate_nodes:
            node = mesh.nodes[nid]
            idx = nid_map[nid]
            cx = node.x + solver.u[idx * 2 + 0]
            cy = node.y + solver.u[idx * 2 + 1]
            dx = cx - pivot_R[0]
            dy = cy - pivot_R[1]
            solver.u[idx * 2 + 0] = pivot_R[0] + cos_dth_R * dx - sin_dth_R * dy - node.x
            solver.u[idx * 2 + 1] = pivot_R[1] + sin_dth_R * dx + cos_dth_R * dy - node.y

        for dof, val in solver.fixed_dofs.items():
            solver.u[dof] = val

        converged, iters = solver.solve_step(1.0, max_iters=25, tol=1e-4)
        reac_L = solver.compute_section_reactions(meta["plate_left_nodes"] + meta["disp_bottom_left_nids"], center_pt=pivot_L, current_coords=True)
        reac_R = solver.compute_section_reactions(meta["plate_right_nodes"] + meta["disp_bottom_right_nids"], center_pt=pivot_R, current_coords=True)

        history.append({
            "step": step,
            "theta_deg": deg_curr,
            "converged": converged,
            "iters": iters,
            "M_L": reac_L["Mz"],
            "M_R": reac_R["Mz"],
            "Fx_L": reac_L["Fx"],
            "Fx_R": reac_R["Fx"],
        })

        if step % 5 == 0 or step == substeps:
            print(f"Step {step:2d}/{substeps} ({deg_curr:5.1f} deg): Iters={iters:2d} | M_L={reac_L['Mz']:+.4f} N*mm, M_R={reac_R['Mz']:+.4f} N*mm | Fx_L={reac_L['Fx']:+.4f} N", flush=True)

        theta_prev = theta_curr
        if not converged:
            print(f"[!] Diverged at step {step}", flush=True)
            return False, history

    t_total = time.perf_counter() - t0
    print(f"[*] Complete! Wall Time = {t_total:.2f}s | Final M_L = {reac_L['Mz']:+.4f} N*mm", flush=True)

    # Compute loop width
    nodes_x_curr = [mesh.nodes[n].x + solver.u[2*nid_map[n]] for n in mesh.nodes]
    loop_width = max(nodes_x_curr) - min(nodes_x_curr)

    out_data = {
        "completed": True,
        "wall_time_s": t_total,
        "substeps": substeps,
        "M_L_final": reac_L["Mz"],
        "M_R_final": reac_R["Mz"],
        "Fx_L_final": reac_L["Fx"],
        "Fx_R_final": reac_R["Fx"],
        "loop_width_x_mm": loop_width,
        "history": history,
    }
    with open("example_disp_fold/results_14layer_ny1_vs_ny2/wht_2d_neohooke_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    return True, history

if __name__ == "__main__":
    test_wht_2d_neohooke(45)
