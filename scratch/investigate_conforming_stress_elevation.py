"""
scratch/investigate_conforming_stress_elevation.py
Analyzes why FrontISTR Conforming 3D stresses appear elevated compared to analytical beam theory and WHT Solver.
"""

import os
import sys
import numpy as np
from scipy.linalg import polar

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.run_fistr_conforming_teardrop import (
    ConformingTeardropMeshBuilder,
    parse_fistr_res_file,
    STACKUP_14,
)
from scratch.compute_true_layer_strain import compute_elem_F

def main():
    config = UnifiedFoldModelConfig(preset="teardrop", dim="3D")
    builder = ConformingTeardropMeshBuilder(config, width_z=1.0)
    builder.build()

    res_path = 'example_disp_fold/results_conforming_teardrop/01_conforming_hybrid_14layer/fistr_teardrop.res.0.20'
    data = parse_fistr_res_file(res_path, max_real_nid=builder.real_max_nid)
    disp = data["DISP"]
    nids = data["nids"]
    nid_to_disp = {nids[i]: disp[i] for i in range(len(nids))}
    nodes_dict = {nid: np.array([x, y, z]) for nid, x, y, z in builder.node_list}

    # Curvature of the actual deformed shape at apex
    # Let's measure the actual radius of curvature R_actual at X = 0
    # Centerline nodes of PET_1 or Glass
    top_nodes = [nid for nid, pt in nodes_dict.items() if abs(pt[1] - 0.63) < 1e-4 and abs(pt[2]) < 1e-4 and abs(pt[0]) <= 2.0]
    top_pts = np.array([nodes_dict[nid] + nid_to_disp[nid] for nid in sorted(top_nodes, key=lambda n: nodes_dict[n][0])])

    # Fit circle to top_pts
    # (x - xc)^2 + (y - yc)^2 = R^2
    xs = top_pts[:, 0]
    ys = top_pts[:, 1]
    # Linear least squares for circle fit: x^2 + y^2 = 2 x xc + 2 y yc + (R^2 - xc^2 - yc^2)
    A = np.column_stack([2 * xs, 2 * ys, np.ones_like(xs)])
    b = xs**2 + ys**2
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = sol[0], sol[1]
    R_fit = np.sqrt(sol[2] + xc**2 + yc**2)

    print("=" * 80)
    print(f"ACTUAL DEFORMED CURVATURE RADIUS AT APEX (20 deg):")
    print(f"  Uniform theoretical radius: R_theory = 15.0 / (40 * pi / 180) = 21.486 mm")
    print(f"  Actual fitted radius at apex: R_actual = {R_fit:.3f} mm")
    print(f"  Curvature concentration factor: kappa_actual / kappa_theory = {21.486 / R_fit:.2f}x")
    print("=" * 80)

    # Now let's inspect the stress tensor components in Glass and PET_1
    materials = {
        "PET": {"E": 4000.0, "nu": 0.30},
        "PSA": {"C10": 0.0083388926, "D1": 0.24, "mu": 0.016677785, "K": 8.3333333},
        "GLASS": {"E": 70000.0, "nu": 0.22},
    }

    print(f"{'Layer':<10} | {'Mises':<8} | {'|sig_tt|':<9} | {'|sig_nn|':<9} | {'|sig_zz|':<9} | {'|tau_tn|':<9} | {'eps_eng':<8} | {'eps_theory':<10} | {'Ratio(%)'}")
    print("-" * 95)

    for lyr in STACKUP_14:
        lname = lyr["name"]
        mat = lyr["mat"]
        t = lyr["t"]
        if mat == "PSA":
            continue
        elems = builder.layer_elements[lname]
        mises_list = []
        sig_tt_list = []
        sig_nn_list = []
        sig_zz_list = []
        tau_tn_list = []
        eps_list = []

        E_mod = materials[mat]["E"]
        nu_mod = materials[mat]["nu"]
        E_prime = E_mod / (1.0 - nu_mod**2)
        sig_theory = E_prime * (t / 2.0) / 21.486
        eps_theory = (t / 2.0) / 21.486 * 100.0

        for elem in elems:
            ref_pts = np.array([nodes_dict[nid] for nid in elem[:8]])
            cur_pts = np.array([nodes_dict[nid] + nid_to_disp.get(nid, np.zeros(3)) for nid in elem[:8]])
            x_mid = np.mean(ref_pts[:, 0])
            if abs(x_mid) > 1.0: # look at pure center |X| <= 1.0mm
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

            E_mat = 0.5 * (C - np.eye(3))
            lam = E_mod * nu_mod / ((1.0 + nu_mod) * (1.0 - 2.0 * nu_mod))
            mu_mod = E_mod / (2.0 * (1.0 + nu_mod))
            trE = np.trace(E_mat)
            S_mat = lam * trE * np.eye(3) + 2.0 * mu_mod * E_mat
            sigma_cauchy = (1.0 / detF) * F @ S_mat @ F.T

            sigma_local = R.T @ sigma_cauchy @ R
            sig_tt = sigma_local[0, 0]
            sig_nn = sigma_local[1, 1]
            sig_zz = sigma_local[2, 2]
            tau_tn = sigma_local[0, 1]
            dev_sig = sigma_cauchy - np.trace(sigma_cauchy)/3.0 * np.eye(3)
            mises = np.sqrt(1.5 * np.sum(dev_sig * dev_sig))

            mises_list.append(mises)
            sig_tt_list.append(abs(sig_tt))
            sig_nn_list.append(abs(sig_nn))
            sig_zz_list.append(abs(sig_zz))
            tau_tn_list.append(abs(tau_tn))
            eps_list.append(eps_eng * 100.0)

        if mises_list:
            m_max = np.max(mises_list)
            tt_max = np.max(sig_tt_list)
            nn_max = np.max(sig_nn_list)
            zz_max = np.max(sig_zz_list)
            tn_max = np.max(tau_tn_list)
            eps_max = np.max(eps_list)
            ratio = (tt_max / sig_theory - 1.0) * 100.0 if sig_theory > 0 else 0
            print(f"{lname:<10} | {m_max:8.2f} | {tt_max:9.2f} | {nn_max:9.2f} | {zz_max:9.2f} | {tn_max:9.2f} | {eps_max:7.2f}% | {eps_theory:9.3f}% | {ratio:+6.1f}%")

if __name__ == "__main__":
    main()
