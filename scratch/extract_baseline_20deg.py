import os
import sys
import numpy as np
from scipy.linalg import polar

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.run_fistr_teardrop import parse_fistr_res_file
from scratch.compute_true_layer_strain import compute_elem_F

def main():
    res_path = "example_disp_fold/results_20deg_parity/FISTR_3D_target20deg_wz1/fistr_teardrop.res.0.5"
    msh_path = "example_disp_fold/results_20deg_parity/FISTR_3D_target20deg_wz1/fistr_teardrop.msh"

    # parse msh
    nodes_dict = {}
    elems_by_pid = {}
    reading_nodes = False
    reading_elems = False
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
                if len(parts) >= 10:
                    eid = int(parts[0])
                    elem_type = int(parts[1])
                    pid = int(parts[2])
                    nids = [int(p) for p in parts[3:11]]
                    if pid not in elems_by_pid:
                        elems_by_pid[pid] = []
                    elems_by_pid[pid].append((eid, nids))

    data = parse_fistr_res_file(res_path, max_real_nid=max(nodes_dict.keys()))
    disp = data["DISP"]
    nids = data["nids"]
    nid_to_disp = {nids[i]: disp[i] for i in range(len(nids))}

    print("=== FRONTISTR BASELINE 3D (HEX 361 FI) AT 20 DEGREES ===")
    # pid 1: PET, pid 2: PSA, pid 3: GLASS, pid 4: PLATE
    for pid, name in [(3, "GLASS"), (1, "PET"), (2, "PSA")]:
        elems = elems_by_pid.get(pid, [])
        apex_mises = []
        apex_eps_eng = []
        apex_gamma = []
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
            C = F.T @ F
            eigenvals = np.linalg.eigvalsh(C)
            stretches = np.sqrt(np.maximum(eigenvals, 1e-12))
            eps_eng = np.max(np.abs(stretches - 1.0))
            E_mat = 0.5 * (C - np.eye(3))
            gamma_tn = 2.0 * abs(E_mat[0, 1])

            # stress from res if available
            apex_eps_eng.append(eps_eng)
            apex_gamma.append(gamma_tn)

        if apex_eps_eng:
            print(f"{name:<10}: max_eps_eng={np.max(apex_eps_eng)*100:6.2f}%, max_gamma_tn={np.max(apex_gamma)*100:6.2f}%")

if __name__ == "__main__":
    main()
