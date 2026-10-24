"""
scratch/run_fistr_pure_plane_strain_90deg.py
Runs 14-layer 90-degree teardrop folding simulation with FrontISTR Conforming Hybrid model
under PURE PLANE STRAIN (u_z = 0 on ALL nodes across entire mesh).
Compares with mid-plane-only fixed conforming model, baseline 361 FI, and WHT 2D.
"""

import os
import sys
import time
import math
import json
import subprocess
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.run_fistr_conforming_teardrop import (
    ConformingTeardropMeshBuilder,
    FrontISTRConformingTeardropRunner,
    STACKUP_14,
    FISTR_EXE,
)
from scratch.extract_localized_local_frame_stress_strain import compute_local_frame_stresses_and_strains

class PurePlaneStrain90MeshBuilder(ConformingTeardropMeshBuilder):
    def build(self):
        super().build()
        driven_set = set(self.left_driven_nids) | set(self.right_driven_nids)
        # Constrain ALL non-driven nodes with u_z = 0 (Pure Plane Strain)
        self.z_fixed_nids = sorted([
            nid for nid, x, y, z in self.node_list[:self.real_max_nid]
            if nid not in driven_set
        ])
        print(f"[*] Pure Plane Strain active: z_fixed_nids count = {len(self.z_fixed_nids)} / {self.real_max_nid} real nodes")

class Refined90PurePlaneStrainRunner(FrontISTRConformingTeardropRunner):
    def write_deck(self):
        super().write_deck()
        cnt_path = os.path.join(self.run_dir, "fistr_teardrop.cnt")
        with open(cnt_path, "r", encoding="utf-8") as f:
            cnt_text = f.read()
        cnt_text = cnt_text.replace("MAXITER=50, CONVERG=2.0e-4", "MAXITER=80, CONVERG=5.0e-3, CONVERG_DDISP=2.0e-3")
        with open(cnt_path, "w", encoding="utf-8") as f:
            f.write(cnt_text)

OUT_DIR_90_PPS = os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop_pure_plane_strain_90deg")

def main():
    os.environ["OMP_NUM_THREADS"] = "8"
    os.makedirs(OUT_DIR_90_PPS, exist_ok=True)

    print("=" * 100)
    print("STARTING 14-LAYER 90° TEARDROP FOLDING: PURE PLANE STRAIN CONFORMING HYBRID (ALL u_z = 0)")
    print(f"Hinge Element Size: ~{15.0/55.0:.4f} mm (nx_hinge=55, nx_plate=35)")
    print("=" * 100)

    config = UnifiedFoldModelConfig(preset="teardrop", dim="2D")
    builder = PurePlaneStrain90MeshBuilder(
        config=config,
        nx_hinge=55,
        nx_plate=35,
        width_z=1.0,
        ny_stiff=1,
    )
    builder.build()

    runner = Refined90PurePlaneStrainRunner(
        builder=builder,
        run_dir=OUT_DIR_90_PPS,
        substeps=90,          # 90 substeps (1.0 deg / substep)
        theta_max_deg=90.0,
    )

    t0 = time.time()
    summary = runner.run()
    elapsed = time.time() - t0

    print("\n" + "=" * 100)
    print("FRONTISTR 90° PURE PLANE STRAIN SOLVE SUMMARY")
    print(f"  Completed   : {summary.get('completed')}")
    print(f"  Wall Time   : {elapsed:.2f} s")
    print(f"  Final Moment: {abs(summary.get('M_L_final_Nmm', 0.0)):.4f} N*mm")
    print(f"  Final Force : {abs(summary.get('Fx_L_final_N', 0.0)):.4f} N")
    print(f"  Loop Width  : {summary.get('loop_width_x_mm', 0.0):.4f} mm")
    print(f"  Apex Y      : {summary.get('apex_y_mm', 0.0):.4f} mm")
    print("=" * 100)

    latest_step = summary.get("latest_substep", 90)
    res_path = os.path.join(OUT_DIR_90_PPS, f"fistr_teardrop.res.0.{latest_step}")
    if not os.path.exists(res_path):
        print(f"[!] Warning: Result file {res_path} not found.")
        return

    print(f"[*] Extracting local frame layerwise metrics from step {latest_step}: {res_path}")
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

    pure_ps_90_result = {
        "case_name": "FISTR_Conforming_90deg_PurePlaneStrain_All_uz0",
        "nx_hinge": 55,
        "nx_plate": 35,
        "h_hinge_mm": 15.0 / 55.0,
        "num_real_nodes": builder.real_max_nid,
        "num_real_elements": builder.real_max_eid,
        "bar_M_L": abs(summary.get("M_L_final_Nmm", 0.0)),
        "bar_Fx_L": abs(summary.get("Fx_L_final_N", 0.0)),
        "loop_width_x_mm": summary.get("loop_width_x_mm", 0.0),
        "apex_y_mm": summary.get("apex_y_mm", 0.0),
        "wall_time_s": elapsed,
        "completed": summary.get("completed", False),
        "latest_substep": latest_step,
        "layers": layer_metrics,
    }

    out_json = os.path.join(OUT_DIR_90_PPS, "pure_plane_strain_conforming_90deg_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pure_ps_90_result, f, indent=2)
    print(f"\n[+] Saved pure plane strain 90-deg results to {out_json}")

if __name__ == "__main__":
    main()
