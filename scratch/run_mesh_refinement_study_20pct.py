"""
scratch/run_mesh_refinement_study_20pct.py
Runs the complete 20% element size reduction benchmark at theta = 20.0 deg across:
1. WHT Solver 2D (CPE4I + CPE4H)
2. WHT Solver 3D (C3D8I + C3D8H)
3. FrontISTR Conforming 3D (Hex 362 + Hex 361 UP)
4. FrontISTR Baseline 3D (Hex 361 FI)

Refined mesh parameters:
- nx_hinge = 38 (was 30, Delta_X reduced from 0.500mm to 0.395mm: -21.1%)
- nx_plate = 25 (was 20, -20.0%)
"""

import os
import sys
import math
import time
import json
import numpy as np
from scipy.linalg import polar

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig, MeshSpec
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from example_disp_fold.run_fistr_conforming_teardrop import (
    ConformingTeardropMeshBuilder,
    FrontISTRConformingTeardropRunner,
    parse_fistr_res_file,
    compute_reaction_moment_from_res,
    STACKUP_14,
)
from example_disp_fold.run_fistr_teardrop import run_fistr_case
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from scratch.compute_true_layer_strain import compute_elem_F
from scratch.extract_localized_local_frame_stress_strain import compute_local_frame_stresses_and_strains

OUT_DIR = os.path.join(REPO_ROOT, "example_disp_fold", "results_mesh_refine_20pct")
os.makedirs(OUT_DIR, exist_ok=True)

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

# =====================================================================
# 1. WHT Solver Runner (Refined)
# =====================================================================
def run_refined_wht(dim="2D"):
    config = UnifiedFoldModelConfig(preset="teardrop", dim=dim)
    config.simulation_case.theta_max_deg = 20.0
    config.mesh.nx_hinge = 38
    config.mesh.nx_plate = 25
    if dim == "3D":
        config.geometry.width_z = 1.0

    materials = {
        1: {"E": 4000.0, "nu": 0.30},
        2: {"E": 0.05, "nu": 0.499},
        3: {"E": 70000.0, "nu": 0.22},
        4: {"E": 200000.0, "nu": 0.30},
    }

    t0_setup = time.perf_counter()
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

    substeps = 10
    theta_max_rad = math.radians(20.0)
    dt = 1.0 / float(substeps)
    theta_prev = 0.0

    history_M_L = []
    history_Fx_L = []
    t0_solve = time.perf_counter()

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
        print(f"      [{dim}] Step {step:2d}/{substeps} (deg={math.degrees(theta_curr):5.2f}): converged={converged}, iters={iters}", flush=True)
        if not converged:
            raise RuntimeError(f"WHT {dim} failed to converge at step {step} (deg={math.degrees(theta_curr):.2f})")
        theta_prev = theta_curr

        driven_nodes_L = meta["plate_left_nodes"] + meta["disp_bottom_left_nids"]
        ctr_L = (float(pivot_L[0]), float(pivot_L[1])) if dim == "2D" else (float(pivot_L[0]), float(pivot_L[1]), 0.0)
        reac_L = solver.compute_section_reactions(driven_nodes_L, center_pt=ctr_L, current_coords=True)
        history_M_L.append(float(reac_L["Mz"]))
        history_Fx_L.append(float(reac_L["Fx"]))

    solve_time = time.perf_counter() - t0_solve

    # Extract layer metrics at apex |X| <= 2.0mm
    u_vec = solver.u
    nodes = mesh.nodes
    mat_lookup = {1: "PET", 2: "PSA", 3: "GLASS"}
    layer_metrics = {}

    for lname, elem_ids in meta["layer_elem_sets"].items():
        canon_name = lname.replace("_Voided", "")
        zone_mises = []
        zone_sig_tt = []
        zone_eps_eng = []
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
                F = compute_2d_elem_F(ref_pts, cur_pts) if dim == "2D" else compute_3d_elem_F(ref_pts, cur_pts)
            except:
                continue

            R, U = polar(F)
            detF = np.linalg.det(F)
            C = F.T @ F
            eigenvals = np.linalg.eigvalsh(C)
            stretches = np.sqrt(np.maximum(eigenvals, 1e-12))
            eps_eng = np.max(np.abs(stretches - 1.0))
            E_mat = 0.5 * (C - np.eye(3))
            gamma_tn = 2.0 * abs(E_mat[0, 1])

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
            zone_gamma_tn.append(gamma_tn * 100.0)

        if zone_mises:
            layer_metrics[canon_name] = {
                "max_mises": float(np.max(zone_mises)),
                "max_sig_tt": float(np.max(zone_sig_tt)),
                "max_eps_eng": float(np.max(zone_eps_eng)),
                "max_gamma_tn": float(np.max(zone_gamma_tn)),
            }

    # Apex Y
    coords0 = mesh.nodes_array()
    disp_all = u_vec.reshape(-1, ndof_n)
    cur_y = coords0[:, 1] + disp_all[:, 1]
    apex_y = float(np.min(cur_y))

    return {
        "case_name": f"WHT_{dim}_Refined_80pct",
        "bar_M_L": abs(history_M_L[-1]),
        "bar_Fx_L": abs(history_Fx_L[-1]),
        "apex_y_mm": apex_y,
        "wall_time_s": solve_time,
        "layers": layer_metrics,
    }

# =====================================================================
# 2. FrontISTR Conforming 3D Runner (Refined)
# =====================================================================
def run_refined_fistr_conforming():
    run_dir = os.path.join(OUT_DIR, "FISTR_Conforming_Refined_80pct")
    os.makedirs(run_dir, exist_ok=True)

    config = UnifiedFoldModelConfig(preset="teardrop", dim="3D")
    builder = ConformingTeardropMeshBuilder(
        config=config,
        nx_hinge=38,
        nx_plate=25,
        width_z=1.0,
    )
    builder.build()

    runner = FrontISTRConformingTeardropRunner(
        builder=builder,
        run_dir=run_dir,
        substeps=20,          # 20 substeps for 20 deg (1 deg/substep)
        theta_max_deg=20.0,
    )
    summary = runner.run()

    # Extract layer metrics at 20 deg
    res_path = os.path.join(run_dir, "fistr_teardrop.res.0.20")
    local_res = compute_local_frame_stresses_and_strains(res_path, builder)
    apex_res = local_res["apex"]

    layer_metrics = {}
    for lyr in STACKUP_14:
        lname = lyr["name"]
        d = apex_res.get(lname)
        if d:
            layer_metrics[lname] = {
                "max_mises": float(d["max_mises"]),
                "max_sig_tt": float(d["max_sig_tt"]),
                "max_eps_eng": float(d["max_eps_eng"]),
                "max_gamma_tn": float(d["max_gamma_tn"]),
            }

    return {
        "case_name": "FISTR_Conforming_3D_Refined_80pct",
        "bar_M_L": abs(summary["M_L_final_Nmm"]),
        "bar_Fx_L": abs(summary["Fx_L_final_N"]),
        "apex_y_mm": summary["apex_y_mm"],
        "wall_time_s": summary["wall_time_s"],
        "layers": layer_metrics,
    }

# =====================================================================
# 3. FrontISTR Baseline 3D Runner (Refined)
# =====================================================================
def run_refined_fistr_baseline():
    run_dir = os.path.join(OUT_DIR, "FISTR_Baseline_Refined_80pct")
    os.makedirs(run_dir, exist_ok=True)

    mesh_spec = MeshSpec(
        use_gmsh=False,
        nx_hinge=38,
        nx_plate=25,
        nz_elements=1,
    )

    summary = run_fistr_case(
        dim="3D",
        preset="teardrop",
        psa_form361="FI",
        solid_form361="FI",
        substeps=5,
        threads=6,
        out_base_dir=OUT_DIR,
        case_name_override="FISTR_Baseline_Refined_80pct",
        width_z=1.0,
        theta_max_deg=20.0,
        mesh_spec=mesh_spec,
    )

    # Extract layer metrics
    res_path = os.path.join(run_dir, "fistr_teardrop.res.0.5")
    msh_path = os.path.join(run_dir, "fistr_teardrop.msh")

    nodes_dict = {}
    elems_by_grp = {}
    curr_grp = None

    with open(msh_path, "r") as f:
        reading_nodes = False; reading_elems = False
        for line in f:
            line_str = line.strip()
            if line_str.startswith("!NODE"):
                reading_nodes = True; reading_elems = False; continue
            elif line_str.startswith("!ELEMENT"):
                reading_nodes = False; reading_elems = True
                for p in line_str.split(","):
                    if "EGRP=" in p: curr_grp = p.split("=")[1].strip()
                if curr_grp not in elems_by_grp: elems_by_grp[curr_grp] = []
                continue
            elif line_str.startswith("!"):
                reading_nodes = False; reading_elems = False; continue

            if reading_nodes:
                parts = line_str.split(",")
                if len(parts) >= 4:
                    nodes_dict[int(parts[0])] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            elif reading_elems:
                parts = line_str.split(",")
                if len(parts) >= 9:
                    elems_by_grp[curr_grp].append((int(parts[0]), [int(p) for p in parts[1:9]]))

    data = parse_fistr_res_file(res_path, max_real_nid=max(nodes_dict.keys()))
    disp = data["DISP"]
    nids = data["nids"]
    nid_to_disp = {nids[i]: disp[i] for i in range(len(nids))}

    materials = {
        "PET": {"E": 4000.0, "nu": 0.30},
        "PSA": {"C10": 0.0083388926, "D1": 0.24, "mu": 0.016677785, "K": 8.3333333},
        "GLASS": {"E": 70000.0, "nu": 0.22},
    }

    grp_map = {
        "EL_GLASS_TOP": ("GLASS_Top", "GLASS"),
        "EL_PSA_6": ("PSA_6", "PSA"),
        "EL_PET_6": ("PET_6", "PET"),
        "EL_PSA_5": ("PSA_5", "PSA"),
        "EL_PET_5": ("PET_5", "PET"),
        "EL_PSA_4": ("PSA_4", "PSA"),
        "EL_PET_4": ("PET_4", "PET"),
        "EL_PSA_3": ("PSA_3", "PSA"),
        "EL_PET_3": ("PET_3", "PET"),
        "EL_PSA_2": ("PSA_2", "PSA"),
        "EL_PET_2": ("PET_2", "PET"),
        "EL_PSA_1": ("PSA_1", "PSA"),
        "EL_PET_1": ("PET_1", "PET"),
        "EL_PSA_0_VOIDED": ("PSA_0", "PSA"),
    }

    layer_metrics = {}
    for grp, (lname, mat) in grp_map.items():
        elems = elems_by_grp.get(grp, [])
        zm = []; ztt = []; ze = []; zg = []
        for eid, enids in elems:
            ref_pts = np.array([nodes_dict[nid] for nid in enids])
            cur_pts = np.array([nodes_dict[nid] + nid_to_disp.get(nid, np.zeros(3)) for nid in enids])
            if abs(np.mean(ref_pts[:, 0])) > 2.0: continue
            try:
                F = compute_elem_F(ref_pts, cur_pts, 0.0, 0.0, 0.0)
            except:
                continue
            R, U = polar(F)
            detF = np.linalg.det(F)
            C = F.T @ F
            eigenvals = np.linalg.eigvalsh(C)
            stretches = np.sqrt(np.maximum(eigenvals, 1e-12))
            eps_eng = np.max(np.abs(stretches - 1.0))
            E_mat = 0.5 * (C - np.eye(3))
            gamma_tn = 2.0 * abs(E_mat[0, 1])

            if mat == "PSA":
                C10 = materials["PSA"]["C10"]; D1 = materials["PSA"]["D1"]; K = 2.0 / D1; J = detF
                b = F @ F.T; b_bar = J**(-2.0/3.0) * b
                dev_b_bar = b_bar - (1.0/3.0) * np.trace(b_bar) * np.eye(3)
                p = K * (J - 1.0)
                sigma_cauchy = (2.0 * C10 / J) * dev_b_bar + p * np.eye(3)
            else:
                E_mod = materials[mat]["E"]; nu_mod = materials[mat]["nu"]
                lam = E_mod * nu_mod / ((1.0 + nu_mod) * (1.0 - 2.0 * nu_mod))
                mu_mod = E_mod / (2.0 * (1.0 + nu_mod))
                trE = np.trace(E_mat)
                S_mat = lam * trE * np.eye(3) + 2.0 * mu_mod * E_mat
                sigma_cauchy = (1.0 / detF) * F @ S_mat @ F.T

            sigma_local = R.T @ sigma_cauchy @ R
            sig_tt = sigma_local[0, 0]
            dev_sig = sigma_cauchy - np.trace(sigma_cauchy)/3.0 * np.eye(3)
            mises = np.sqrt(1.5 * np.sum(dev_sig * dev_sig))

            zm.append(mises); ztt.append(abs(sig_tt)); ze.append(eps_eng * 100.0); zg.append(gamma_tn * 100.0)

        if zm:
            layer_metrics[lname] = {
                "max_mises": float(np.max(zm)),
                "max_sig_tt": float(np.max(ztt)),
                "max_eps_eng": float(np.max(ze)),
                "max_gamma_tn": float(np.max(zg)),
            }

    return {
        "case_name": "FISTR_Baseline_3D_Refined_80pct",
        "bar_M_L": abs(summary.get("M_L_final_Nmm", summary.get("bar_M_L", 0.0))),
        "bar_Fx_L": abs(summary.get("Fx_L_final_N", summary.get("bar_Fx_L", 0.0))),
        "apex_y_mm": summary["apex_y_mm"],
        "wall_time_s": summary["wall_time_s"],
        "layers": layer_metrics,
    }

def main():
    wht_only = "--wht-only" in sys.argv
    out_json = os.path.join(OUT_DIR, "mesh_refinement_20pct_results.json")
    existing_results = {}
    if os.path.exists(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
        except Exception:
            pass

    print("=" * 100)
    print("STARTING 20% ELEMENT SIZE REDUCTION (REFINED MESH) STUDY AT 20 DEGREES")
    print(f"Mode: {'WHT Only (Reusing FrontISTR results)' if wht_only else 'Full 4-Solver Study'}")
    print("=" * 100)

    print("\n>>> [1/4] Running WHT Solver 2D (Refined: nx_hinge=38, nx_plate=25, 10 substeps)...")
    res_wht2d = run_refined_wht("2D")
    print(f"    WHT 2D Done: |M_L| = {res_wht2d['bar_M_L']:.4f} N*mm, Time = {res_wht2d['wall_time_s']:.2f}s")

    print("\n>>> [2/4] Running WHT Solver 3D (Refined: nx_hinge=38, nx_plate=25, 10 substeps)...")
    res_wht3d = run_refined_wht("3D")
    print(f"    WHT 3D Done: |M_L| = {res_wht3d['bar_M_L']:.4f} N*mm, Time = {res_wht3d['wall_time_s']:.2f}s")

    if wht_only and "FISTR_Conforming_Refined" in existing_results and "FISTR_Baseline_Refined" in existing_results:
        print("\n>>> Reusing existing FrontISTR Conforming 3D results...")
        res_conf = existing_results["FISTR_Conforming_Refined"]
        print("\n>>> Reusing existing FrontISTR Baseline 3D results...")
        res_base = existing_results["FISTR_Baseline_Refined"]
    else:
        print("\n>>> [3/4] Running FrontISTR Conforming 3D (Refined: nx_hinge=38, nx_plate=25)...")
        res_conf = run_refined_fistr_conforming()
        print(f"    FrontISTR Conforming Done: |M_L| = {res_conf['bar_M_L']:.4f} N*mm, Time = {res_conf['wall_time_s']:.2f}s")

        print("\n>>> [4/4] Running FrontISTR Baseline 3D (Refined: nx_hinge=38, nx_plate=25)...")
        res_base = run_refined_fistr_baseline()
        print(f"    FrontISTR Baseline Done: |M_L| = {res_base['bar_M_L']:.4f} N*mm, Time = {res_base['wall_time_s']:.2f}s")

    all_results = {
        "WHT_2D_Refined": res_wht2d,
        "WHT_3D_Refined": res_wht3d,
        "FISTR_Conforming_Refined": res_conf,
        "FISTR_Baseline_Refined": res_base,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved all refined results to {out_json}")

if __name__ == "__main__":
    main()
