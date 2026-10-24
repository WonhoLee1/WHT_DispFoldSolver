"""
scratch/parse_and_generate_refinement_report.py
Loads refined results from disk (or runs WHT if needed) and produces a comprehensive 1:1 comparison.
"""

import os
import sys
import json
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.run_fistr_conforming_teardrop import (
    ConformingTeardropMeshBuilder,
    parse_fistr_res_file,
    STACKUP_14,
)
from scratch.run_mesh_refinement_study_20pct import run_refined_wht
from scratch.extract_localized_local_frame_stress_strain import compute_local_frame_stresses_and_strains
from scratch.compute_true_layer_strain import compute_elem_F
from scipy.linalg import polar

OUT_DIR = os.path.join(REPO_ROOT, "example_disp_fold", "results_mesh_refine_20pct")

def parse_fistr_conf():
    run_dir = os.path.join(OUT_DIR, "FISTR_Conforming_Refined_80pct")
    sum_path = os.path.join(run_dir, "fistr_summary.json")
    with open(sum_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    config = UnifiedFoldModelConfig(preset="teardrop", dim="3D")
    builder = ConformingTeardropMeshBuilder(config=config, nx_hinge=38, nx_plate=25, width_z=1.0)
    builder.build()

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
        "apex_y_mm": summary.get("apex_y_mm", 0.0),
        "wall_time_s": summary.get("wall_time_s", 0.0),
        "layers": layer_metrics,
    }

def parse_fistr_base():
    run_dir = os.path.join(OUT_DIR, "FISTR_Baseline_Refined_80pct")
    sum_path = os.path.join(run_dir, "fistr_summary.json")
    with open(sum_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    res_path = os.path.join(run_dir, "fistr_teardrop.res.0.5")
    msh_path = os.path.join(run_dir, "fistr_teardrop.msh")

    nodes_dict = {}
    elems_by_grp = {}
    curr_grp = None

    with open(msh_path, "r", encoding="utf-8") as f:
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
        "bar_M_L": abs(summary.get("M_L_final_Nmm", 0.0)),
        "bar_Fx_L": abs(summary.get("Fx_L_final_N", 0.0)),
        "apex_y_mm": summary.get("apex_y_mm", 0.0),
        "wall_time_s": summary.get("wall_time_s", 0.0),
        "layers": layer_metrics,
    }

def main():
    print("--- Running/Loading WHT 2D Refined ---")
    res_wht2d = run_refined_wht("2D")
    print("--- Running/Loading WHT 3D Refined ---")
    res_wht3d = run_refined_wht("3D")
    print("--- Parsing FrontISTR Conforming Refined ---")
    res_conf = parse_fistr_conf()
    print("--- Parsing FrontISTR Baseline Refined ---")
    res_base = parse_fistr_base()

    all_results = {
        "WHT_2D_Refined": res_wht2d,
        "WHT_3D_Refined": res_wht3d,
        "FISTR_Conforming_Refined": res_conf,
        "FISTR_Baseline_Refined": res_base,
    }

    out_json = os.path.join(OUT_DIR, "mesh_refinement_20pct_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[+] Saved refined results to {out_json}")

if __name__ == "__main__":
    main()
