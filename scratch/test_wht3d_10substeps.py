"""
scratch/test_wht3d_10substeps.py
Tests WHT 3D with 10 substeps (2.0 deg / step) to ensure 100% convergence up to 20.0 deg.
"""

import os
import sys
import math
import numpy as np
from scipy.linalg import polar

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from scratch.run_mesh_refinement_study_20pct import compute_3d_elem_F, smoothstep_amp

def main():
    config = UnifiedFoldModelConfig(preset="teardrop", dim="3D")
    config.simulation_case.theta_max_deg = 20.0
    config.mesh.nx_hinge = 38
    config.mesh.nx_plate = 25
    config.geometry.width_z = 1.0

    materials = {
        1: {"E": 4000.0, "nu": 0.30},
        2: {"E": 0.05, "nu": 0.499},
        3: {"E": 70000.0, "nu": 0.22},
        4: {"E": 200000.0, "nu": 0.30},
    }

    mesher = GmshMeshGenerator(config)
    mesh, meta = mesher.build()
    solver = DynamicSolver3D(mesh, materials=materials, nlgeom=True, solution_controls="STANDARD")
    solver.max_du_limit = 1.0

    pivot_L = meta["left_pivot"]
    pivot_R = meta["right_pivot"]
    hinge_gap = config.geometry.hinge_half_gap
    disp_left_plate_nodes = [nid for nid, node in mesh.nodes.items() if node.x <= -hinge_gap + 1e-4 and node.y >= 0.0]
    disp_right_plate_nodes = [nid for nid, node in mesh.nodes.items() if node.x >= hinge_gap - 1e-4 and node.y >= 0.0]
    nid_map = mesh.node_id_to_index()
    ndof_n = 3

    substeps = 10  # 10 substeps (2 deg / step)
    theta_max_rad = math.radians(20.0)
    dt = 1.0 / float(substeps)
    theta_prev = 0.0

    history_M_L = []
    history_Fx_L = []

    print("Running WHT 3D (38 hinge elements) with 10 substeps...")
    all_converged = True
    for step in range(1, substeps + 1):
        s = step / float(substeps)
        theta_curr = theta_max_rad * smoothstep_amp(s)
        deg_curr = math.degrees(theta_curr)
        solver.fixed_dofs.clear()

        cos_th_L = math.cos(-theta_curr); sin_th_L = math.sin(-theta_curr)
        for nid in meta["plate_left_nodes"] + meta["disp_bottom_left_nids"]:
            node = mesh.nodes[nid]
            dx0 = node.x - pivot_L[0]; dy0 = node.y - pivot_L[1]
            solver.fix_dof(nid, 0, pivot_L[0] + cos_th_L * dx0 - sin_th_L * dy0 - node.x)
            solver.fix_dof(nid, 1, pivot_L[1] + sin_th_L * dx0 + cos_th_L * dy0 - node.y)
            solver.fix_dof(nid, 2, 0.0)

        cos_th_R = math.cos(theta_curr); sin_th_R = math.sin(theta_curr)
        for nid in meta["plate_right_nodes"] + meta["disp_bottom_right_nids"]:
            node = mesh.nodes[nid]
            dx0 = node.x - pivot_R[0]; dy0 = node.y - pivot_R[1]
            solver.fix_dof(nid, 0, pivot_R[0] + cos_th_R * dx0 - sin_th_R * dy0 - node.x)
            solver.fix_dof(nid, 1, pivot_R[1] + sin_th_R * dx0 + cos_th_R * dy0 - node.y)
            solver.fix_dof(nid, 2, 0.0)

        for nid, node in mesh.nodes.items():
            if abs(getattr(node, "z", 0.0)) < 1e-4:
                solver.fix_dof(nid, 2, 0.0)

        # Predictor
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

        converged, iters = solver.solve_step(dt=dt, max_iters=25, tol=1e-4, use_anderson_accel=True, anderson_m=4)
        theta_prev = theta_curr
        print(f"Step {step:2d} (deg={deg_curr:5.2f}): converged={converged}, iters={iters}")
        if not converged:
            all_converged = False

        driven_nodes_L = meta["plate_left_nodes"] + meta["disp_bottom_left_nids"]
        ctr_L = (float(pivot_L[0]), float(pivot_L[1]), 0.0)
        reac_L = solver.compute_section_reactions(driven_nodes_L, center_pt=ctr_L, current_coords=True)
        history_M_L.append(float(reac_L["Mz"]))
        history_Fx_L.append(float(reac_L["Fx"]))

    print(f"\nAll converged: {all_converged}")
    print(f"Final 20 deg Moment |M_L| = {abs(history_M_L[-1]):.4f} N*mm")
    print(f"Final 20 deg Force  |F_x| = {abs(history_Fx_L[-1]):.4f} N")

    # Layerwise stresses
    u_vec = solver.u
    nodes = mesh.nodes
    mat_lookup = {1: "PET", 2: "PSA", 3: "GLASS"}

    for lname, elem_ids in meta["layer_elem_sets"].items():
        canon_name = lname.replace("_Voided", "")
        zone_mises = []; zone_sig_tt = []; zone_eps_eng = []

        for eid in elem_ids:
            el = mesh.elements[eid]
            enids = el.node_ids
            ref_pts = np.array([[nodes[nid].x, nodes[nid].y, getattr(nodes[nid], 'z', 0.0)] for nid in enids])
            cur_pts = np.zeros_like(ref_pts)
            for i, nid in enumerate(enids):
                idx = nid_map[nid]
                cur_pts[i, 0] = ref_pts[i, 0] + u_vec[idx * ndof_n + 0]
                cur_pts[i, 1] = ref_pts[i, 1] + u_vec[idx * ndof_n + 1]
                cur_pts[i, 2] = ref_pts[i, 2] + u_vec[idx * ndof_n + 2]

            if abs(np.mean(ref_pts[:, 0])) > 2.0:
                continue

            try:
                F = compute_3d_elem_F(ref_pts, cur_pts)
            except:
                continue

            R, U = polar(F)
            detF = np.linalg.det(F)
            C = F.T @ F
            eigenvals = np.linalg.eigvalsh(C)
            stretches = np.sqrt(np.maximum(eigenvals, 1e-12))
            eps_eng = np.max(np.abs(stretches - 1.0))
            E_mat = 0.5 * (C - np.eye(3))

            mat_type = mat_lookup.get(el.pid, "PET")
            if mat_type == "PSA":
                C10 = 0.0083388926; D1 = 0.24; K = 2.0 / D1; J = detF
                b = F @ F.T; b_bar = J**(-2.0/3.0) * b
                dev_b_bar = b_bar - (1.0/3.0) * np.trace(b_bar) * np.eye(3)
                p = K * (J - 1.0)
                sigma_cauchy = (2.0 * C10 / J) * dev_b_bar + p * np.eye(3)
            else:
                E_mod = materials[el.pid]["E"]; nu_mod = materials[el.pid]["nu"]
                lam = E_mod * nu_mod / ((1.0 + nu_mod) * (1.0 - 2.0 * nu_mod))
                mu_mod = E_mod / (2.0 * (1.0 + nu_mod))
                trE = np.trace(E_mat)
                S_mat = lam * trE * np.eye(3) + 2.0 * mu_mod * E_mat
                sigma_cauchy = (1.0 / detF) * F @ S_mat @ F.T

            sigma_local = R.T @ sigma_cauchy @ R
            sig_tt = sigma_local[0, 0]
            dev_sig = sigma_cauchy - np.trace(sigma_cauchy)/3.0 * np.eye(3)
            mises = np.sqrt(1.5 * np.sum(dev_sig * dev_sig))

            zone_mises.append(mises)
            zone_sig_tt.append(abs(sig_tt))
            zone_eps_eng.append(eps_eng * 100.0)

        if zone_mises and ("GLASS" in canon_name or "PET_1" in canon_name or "PSA_1" in canon_name or "PET_6" in canon_name):
            print(f"  {canon_name:<10}: Mises = {np.max(zone_mises):6.2f} MPa, sig_tt = {np.max(zone_sig_tt):6.2f} MPa, eps_eng = {np.max(zone_eps_eng):.2f}%")

if __name__ == "__main__":
    main()
