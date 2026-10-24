"""
scratch/extract_wht_layerwise_20deg.py
Accurately runs WHT 2D and WHT 3D using the exact formulation from run_20deg_parity_study.py,
and extracts layerwise stresses and strains at 20.0 degrees.
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
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

def smoothstep_amp(s: float) -> float:
    s_clamped = max(0.0, min(1.0, s))
    return s_clamped * s_clamped * (3.0 - 2.0 * s_clamped)

def compute_2d_elem_F(ref_pts, cur_pts):
    dN_dxi = 0.25 * np.array([-1.0, 1.0, 1.0, -1.0])
    dN_deta = 0.25 * np.array([-1.0, -1.0, 1.0, 1.0])
    dN = np.vstack([dN_dxi, dN_deta])
    J0 = dN @ ref_pts[:, :2]
    invJ0 = np.linalg.inv(J0)
    dN_dX = invJ0 @ dN
    F2 = cur_pts[:, :2].T @ dN_dX.T
    F3 = np.eye(3)
    F3[:2, :2] = F2
    return F3

def compute_3d_elem_F(ref_pts, cur_pts):
    corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=float)
    dN = np.zeros((3, 8))
    for i in range(8):
        c = corners[i]
        dN[0, i] = 0.125 * c[0]
        dN[1, i] = 0.125 * c[1]
        dN[2, i] = 0.125 * c[2]
    J0 = dN @ ref_pts
    invJ0 = np.linalg.inv(J0)
    dN_dX = invJ0 @ dN
    F = cur_pts.T @ dN_dX.T
    return F

def solve_wht(dim="2D"):
    config = UnifiedFoldModelConfig(preset="teardrop", dim=dim)
    config.simulation_case.theta_max_deg = 20.0
    if dim == "3D":
        config.geometry.width_z = 1.0

    materials = {
        1: {"E": 4000.0, "nu": 0.30},
        2: {"E": 0.05, "nu": 0.499},
        3: {"E": 70000.0, "nu": 0.22},
        4: {"E": 200000.0, "nu": 0.30},
    }

    mesher = GmshMeshGenerator(config)
    mesh, meta = mesher.build()

    if dim == "2D":
        solver = DynamicSolver2D(mesh, materials=materials, nlgeom=True)
        solver.max_du_limit = 1.0
    else:
        solver = DynamicSolver3D(mesh, materials=materials, nlgeom=True, solution_controls="STANDARD")
        solver.max_du_limit = 1.0

    pivot_L = meta["left_pivot"]
    pivot_R = meta["right_pivot"]
    hinge_gap = config.geometry.hinge_half_gap
    disp_left_plate_nodes = [nid for nid, node in mesh.nodes.items() if node.x <= -hinge_gap + 1e-4 and node.y >= 0.0]
    disp_right_plate_nodes = [nid for nid, node in mesh.nodes.items() if node.x >= hinge_gap - 1e-4 and node.y >= 0.0]
    nid_map = mesh.node_id_to_index()
    ndof_n = 2 if dim == "2D" else 3

    substeps = 5
    theta_max_rad = math.radians(20.0)
    dt = 1.0 / float(substeps)
    theta_prev = 0.0

    for step in range(1, substeps + 1):
        s = step / float(substeps)
        theta_curr = theta_max_rad * smoothstep_amp(s)
        solver.fixed_dofs.clear()

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
        print(f"[{dim} step {step}] converged={converged}, iters={iters}")

    # Extract layer metrics
    u_vec = solver.u
    nodes = mesh.nodes
    results = {}
    mat_lookup = {1: "PET", 2: "PSA", 3: "GLASS"}

    for lname, elem_ids in meta["layer_elem_sets"].items():
        canon_name = lname.replace("_Voided", "")
        zone_mises = []
        zone_sig_tt = []
        zone_tau_tn = []
        zone_eps_eng = []
        zone_eps_true = []
        zone_gamma_tn = []

        for eid in elem_ids:
            el = mesh.elements[eid]
            enids = el.node_ids
            ref_pts = np.array([[nodes[nid].x, nodes[nid].y, getattr(nodes[nid], 'z', 0.0)] for nid in enids])
            cur_pts = np.zeros_like(ref_pts)
            for i, nid in enumerate(enids):
                idx = nid_map[nid]
                cur_pts[i, 0] = ref_pts[i, 0] + u_vec[idx * ndof_n + 0]
                cur_pts[i, 1] = ref_pts[i, 1] + u_vec[idx * ndof_n + 1]
                if dim == "3D":
                    cur_pts[i, 2] = ref_pts[i, 2] + u_vec[idx * ndof_n + 2]

            x_mid = np.mean(ref_pts[:, 0])
            if abs(x_mid) > 2.0:
                continue

            try:
                if dim == "2D":
                    F = compute_2d_elem_F(ref_pts, cur_pts)
                else:
                    F = compute_3d_elem_F(ref_pts, cur_pts)
            except:
                continue

            R, U = polar(F)
            detF = np.linalg.det(F)
            C = F.T @ F
            eigenvals = np.linalg.eigvalsh(C)
            stretches = np.sqrt(np.maximum(eigenvals, 1e-12))
            eps_eng = np.max(np.abs(stretches - 1.0))
            eps_true = np.max(np.abs(np.log(stretches)))
            E_mat = 0.5 * (C - np.eye(3))
            gamma_tn = 2.0 * abs(E_mat[0, 1])

            mat_type = mat_lookup.get(el.pid, "PET")
            if mat_type == "PSA":
                C10 = 0.0083388926
                D1 = 0.24
                K = 2.0 / D1
                J = detF
                b = F @ F.T
                b_bar = J**(-2.0/3.0) * b
                dev_b_bar = b_bar - (1.0/3.0) * np.trace(b_bar) * np.eye(3)
                p = K * (J - 1.0)
                sigma_cauchy = (2.0 * C10 / J) * dev_b_bar + p * np.eye(3)
            else:
                E_mod = materials[el.pid]["E"]
                nu_mod = materials[el.pid]["nu"]
                lam = E_mod * nu_mod / ((1.0 + nu_mod) * (1.0 - 2.0 * nu_mod))
                mu_mod = E_mod / (2.0 * (1.0 + nu_mod))
                trE = np.trace(E_mat)
                S_mat = lam * trE * np.eye(3) + 2.0 * mu_mod * E_mat
                sigma_cauchy = (1.0 / detF) * F @ S_mat @ F.T

            sigma_local = R.T @ sigma_cauchy @ R
            sig_tt = sigma_local[0, 0]
            tau_tn = sigma_local[0, 1]
            dev_sig = sigma_cauchy - np.trace(sigma_cauchy)/3.0 * np.eye(3)
            mises = np.sqrt(1.5 * np.sum(dev_sig * dev_sig))

            zone_mises.append(mises)
            zone_sig_tt.append(sig_tt)
            zone_tau_tn.append(tau_tn)
            zone_eps_eng.append(eps_eng)
            zone_eps_true.append(eps_true)
            zone_gamma_tn.append(gamma_tn)

        if zone_mises:
            results[canon_name] = {
                "max_mises": float(np.max(zone_mises)),
                "max_sig_tt": float(np.max(np.abs(zone_sig_tt))),
                "max_tau_tn": float(np.max(np.abs(zone_tau_tn))),
                "max_eps_eng": float(np.max(zone_eps_eng) * 100.0),
                "max_eps_true": float(np.max(zone_eps_true) * 100.0),
                "max_gamma_tn": float(np.max(zone_gamma_tn) * 100.0),
            }

    return results

def main():
    print("--- Running WHT Solver 2D (Opt1) ---")
    res_2d = solve_wht("2D")
    print("--- Running WHT Solver 3D (Opt1) ---")
    res_3d = solve_wht("3D")

    import json
    with open("scratch/wht_20deg_extracted_stresses.json", "w") as f:
        json.dump({"WHT_2D": res_2d, "WHT_3D": res_3d}, f, indent=2)
    print("Saved results to scratch/wht_20deg_extracted_stresses.json")

if __name__ == "__main__":
    main()
