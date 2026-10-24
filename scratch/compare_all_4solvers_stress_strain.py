"""
scratch/compare_all_4solvers_stress_strain.py
Extracts and compares localized stresses and strains at theta = 20.0 degrees across:
1. WHT Solver 2D (CPE4I + CPE4H)
2. WHT Solver 3D (C3D8I + C3D8H)
3. FrontISTR Conforming 3D (Hex 362 + Hex 361 UP)
4. FrontISTR Baseline 3D (Hex 361 FI)
"""

import os
import sys
import numpy as np
from scipy.linalg import polar

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.run_fistr_conforming_teardrop import (
    ConformingTeardropMeshBuilder,
    parse_fistr_res_file,
    STACKUP_14,
)
from scratch.compute_true_layer_strain import compute_elem_F
from scratch.extract_localized_local_frame_stress_strain import compute_local_frame_stresses_and_strains

def extract_fistr_conforming():
    config = UnifiedFoldModelConfig(preset="teardrop", dim="3D")
    builder = ConformingTeardropMeshBuilder(config, width_z=1.0)
    builder.build()
    res_path = 'example_disp_fold/results_conforming_teardrop/01_conforming_hybrid_14layer/fistr_teardrop.res.0.20'
    res = compute_local_frame_stresses_and_strains(res_path, builder)
    return res

def extract_fistr_baseline():
    res_path = "example_disp_fold/results_20deg_parity/FISTR_3D_target20deg_wz1/fistr_teardrop.res.0.5"
    msh_path = "example_disp_fold/results_20deg_parity/FISTR_3D_target20deg_wz1/fistr_teardrop.msh"

    nodes_dict = {}
    elems_by_grp = {}
    reading_nodes = False
    reading_elems = False
    curr_grp = None

    with open(msh_path, "r") as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("!NODE"):
                reading_nodes = True
                reading_elems = False
                continue
            elif line_str.startswith("!ELEMENT"):
                reading_nodes = False
                reading_elems = True
                parts = line_str.split(",")
                for p in parts:
                    if "EGRP=" in p:
                        curr_grp = p.split("=")[1].strip()
                if curr_grp not in elems_by_grp:
                    elems_by_grp[curr_grp] = []
                continue
            elif line_str.startswith("!"):
                reading_nodes = False
                reading_elems = False
                continue

            if reading_nodes:
                parts = line_str.split(",")
                if len(parts) >= 4:
                    nid = int(parts[0])
                    nodes_dict[nid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            elif reading_elems:
                parts = line_str.split(",")
                if len(parts) >= 9:
                    eid = int(parts[0])
                    nids = [int(p) for p in parts[1:9]]
                    elems_by_grp[curr_grp].append((eid, nids))

    data = parse_fistr_res_file(res_path, max_real_nid=max(nodes_dict.keys()))
    disp = data["DISP"]
    nids = data["nids"]
    nid_to_disp = {nids[i]: disp[i] for i in range(len(nids))}

    materials = {
        "PET": {"E": 4000.0, "nu": 0.30},
        "PSA": {"C10": 0.0083388926, "D1": 0.24, "mu": 0.016677785, "K": 8.3333333},
        "GLASS": {"E": 70000.0, "nu": 0.22},
    }

    results = {}
    # Map EGRP to layer name
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

    for grp, (lname, mat) in grp_map.items():
        elems = elems_by_grp.get(grp, [])
        zone_mises = []
        zone_sig_tt = []
        zone_tau_tn = []
        zone_eps_eng = []
        zone_eps_true = []
        zone_gamma_tn = []

        for eid, enids in elems:
            ref_pts = np.array([nodes_dict[nid] for nid in enids])
            cur_pts = np.array([nodes_dict[nid] + nid_to_disp.get(nid, np.zeros(3)) for nid in enids])
            x_mid = np.mean(ref_pts[:, 0])
            if abs(x_mid) > 2.0:
                continue
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
            eps_true = np.max(np.abs(np.log(stretches)))
            E_mat = 0.5 * (C - np.eye(3))
            gamma_tn = 2.0 * abs(E_mat[0, 1])

            if mat == "PSA":
                C10 = materials["PSA"]["C10"]
                D1 = materials["PSA"]["D1"]
                K = 2.0 / D1
                J = detF
                b = F @ F.T
                b_bar = J**(-2.0/3.0) * b
                dev_b_bar = b_bar - (1.0/3.0) * np.trace(b_bar) * np.eye(3)
                p = K * (J - 1.0)
                sigma_cauchy = (2.0 * C10 / J) * dev_b_bar + p * np.eye(3)
            else:
                E_mod = materials[mat]["E"]
                nu_mod = materials[mat]["nu"]
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
            results[lname] = {
                "max_mises": np.max(zone_mises),
                "max_sig_tt": np.max(np.abs(zone_sig_tt)),
                "max_tau_tn": np.max(np.abs(zone_tau_tn)),
                "max_eps_eng": np.max(zone_eps_eng) * 100.0,
                "max_eps_true": np.max(zone_eps_true) * 100.0,
                "max_gamma_tn": np.max(zone_gamma_tn) * 100.0,
            }

    return results

def main():
    print("Extracting FrontISTR Conforming 3D...")
    conf_res = extract_fistr_conforming()
    conf_apex = conf_res["apex"]

    print("Extracting FrontISTR Baseline 3D...")
    base_apex = extract_fistr_baseline()

    print("\n" + "=" * 125)
    print("FRONTISTR CONFORMING 3D vs FRONTISTR BASELINE 3D (20 DEGREES, APEX ZONE |X| <= 2.0mm)")
    print("=" * 125)
    header = f"{'Layer Name':<12} | {'Conf Mises':<11} | {'Base Mises':<11} | {'Conf sig_tt':<12} | {'Base sig_tt':<12} | {'Conf eps_eng':<13} | {'Base eps_eng':<13} | {'Conf gamma':<11} | {'Base gamma':<11}"
    print(header)
    print("-" * 125)

    for lyr in STACKUP_14:
        lname = lyr["name"]
        c = conf_apex.get(lname, {})
        b = base_apex.get(lname, {})
        c_mises = f"{c.get('max_mises', 0.0):.2f}"
        b_mises = f"{b.get('max_mises', 0.0):.2f}"
        c_sig = f"{c.get('max_sig_tt', 0.0):.2f}"
        b_sig = f"{b.get('max_sig_tt', 0.0):.2f}"
        c_eps = f"{c.get('max_eps_eng', 0.0):.2f}%"
        b_eps = f"{b.get('max_eps_eng', 0.0):.2f}%"
        c_gam = f"{c.get('max_gamma_tn', 0.0):.2f}%"
        b_gam = f"{b.get('max_gamma_tn', 0.0):.2f}%"
        print(f"{lname:<12} | {c_mises:<11} | {b_mises:<11} | {c_sig:<12} | {b_sig:<12} | {c_eps:<13} | {b_eps:<13} | {c_gam:<11} | {b_gam:<11}")

if __name__ == "__main__":
    main()
