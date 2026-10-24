import os
import sys
import numpy as np

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.run_fistr_conforming_teardrop import ConformingTeardropMeshBuilder, STACKUP_14
from scratch.extract_localized_local_frame_stress_strain import compute_local_frame_stresses_and_strains

def main():
    config = UnifiedFoldModelConfig(preset="teardrop", dim="3D")
    builder = ConformingTeardropMeshBuilder(config, width_z=1.0)
    builder.build()

    res_path_conf = 'example_disp_fold/results_conforming_teardrop/01_conforming_hybrid_14layer/fistr_teardrop.res.0.20'
    res_conf = compute_local_frame_stresses_and_strains(res_path_conf, builder)

    print("=== FRONTISTR CONFORMING 3D AT 20 DEGREES (APEX: |X| <= 2.0 mm) ===")
    print(f"{'Layer Name':<12} | {'Mises (MPa)':<12} | {'|sig_tt| (MPa)':<14} | {'|tau_tn| (MPa)':<14} | {'eps_eng (%)':<12} | {'gamma_tn (%)':<12}")
    print("-" * 85)
    for lyr in STACKUP_14:
        lname = lyr["name"]
        d = res_conf["apex"].get(lname)
        if d:
            print(f"{lname:<12} | {d['max_mises']:12.2f} | {d['max_sig_tt']:14.2f} | {d['max_tau_tn']:14.4f} | {d['max_eps_eng']:12.2f}% | {d['max_gamma_tn']:12.2f}%")

    print("\n=== FRONTISTR CONFORMING 3D AT 20 DEGREES (ENDS: |X| >= 7.5 mm) ===")
    print(f"{'Layer Name':<12} | {'Mises (MPa)':<12} | {'|sig_tt| (MPa)':<14} | {'|tau_tn| (MPa)':<14} | {'eps_eng (%)':<12} | {'gamma_tn (%)':<12}")
    print("-" * 85)
    for lyr in STACKUP_14:
        lname = lyr["name"]
        d = res_conf["ends"].get(lname)
        if d:
            print(f"{lname:<12} | {d['max_mises']:12.2f} | {d['max_sig_tt']:14.2f} | {d['max_tau_tn']:14.4f} | {d['max_eps_eng']:12.2f}% | {d['max_gamma_tn']:12.2f}%")

if __name__ == "__main__":
    main()
