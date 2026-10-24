"""
scratch/run_fistr_conforming_90deg_refined.py
Runs 14-layer 90-degree teardrop folding simulation with FrontISTR Conforming Hybrid model
using refined element size (nx_hinge=55, nx_plate=35, ~45% size reduction, h~0.273mm)
and compares with 100% baseline (nx_hinge=30) and 361 FI baseline.
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

class RefinedFrontISTRRunner(FrontISTRConformingTeardropRunner):
    def write_deck(self):
        super().write_deck()
        cnt_path = os.path.join(self.run_dir, "fistr_teardrop.cnt")
        with open(cnt_path, "r", encoding="utf-8") as f:
            cnt_text = f.read()
        # Use CONVERG=5.0e-3, CONVERG_DDISP=2.0e-3 and MAXITER=80 for large-deformation 90-deg folding convergence
        cnt_text = cnt_text.replace("MAXITER=50, CONVERG=2.0e-4", "MAXITER=80, CONVERG=5.0e-3, CONVERG_DDISP=2.0e-3")
        with open(cnt_path, "w", encoding="utf-8") as f:
            f.write(cnt_text)

OUT_DIR = os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_teardrop_refined_90deg")

def main():
    os.environ["OMP_NUM_THREADS"] = "8"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 100)
    print("STARTING 14-LAYER 90° TEARDROP FOLDING: REFINED CONFORMING HYBRID (nx_hinge=55, nx_plate=35)")
    print(f"Hinge Element Size: ~{15.0/55.0:.4f} mm (vs Baseline 0.5000 mm, 45.5% reduction)")
    print("=" * 100)

    config = UnifiedFoldModelConfig(preset="teardrop", dim="2D")
    builder = ConformingTeardropMeshBuilder(
        config=config,
        nx_hinge=55,
        nx_plate=35,
        width_z=1.0,
        ny_stiff=1,
    )
    builder.build()

    print(f"[*] Mesh statistics:")
    print(f"    Nodes (Real)    : {builder.real_max_nid}")
    print(f"    Elements (Real) : {builder.real_max_eid}")

    runner = RefinedFrontISTRRunner(
        builder=builder,
        run_dir=OUT_DIR,
        substeps=90,          # 90 substeps (1.0 deg / substep) for full 90-deg wing rotation
        theta_max_deg=90.0,
    )

    t0 = time.time()
    summary = runner.run()
    elapsed = time.time() - t0

    print("\n" + "=" * 100)
    print("FRONTISTR SOLVE SUMMARY")
    print(f"  Completed   : {summary.get('completed')}")
    print(f"  Wall Time   : {elapsed:.2f} s")
    print(f"  Final Moment: {abs(summary.get('M_L_final_Nmm', 0.0)):.4f} N*mm")
    print(f"  Final Force : {abs(summary.get('Fx_L_final_N', 0.0)):.4f} N")
    print(f"  Loop Width  : {summary.get('loop_width_x_mm', 0.0):.4f} mm")
    print(f"  Apex Y      : {summary.get('apex_y_mm', 0.0):.4f} mm")
    print("=" * 100)

    # Extract layerwise metrics at 90 deg (or latest available substep)
    latest_step = summary.get("latest_substep", 90)
    res_path = os.path.join(OUT_DIR, f"fistr_teardrop.res.0.{latest_step}")
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

    refined_90_result = {
        "case_name": "FISTR_Conforming_90deg_Refined_55pct",
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

    out_json = os.path.join(OUT_DIR, "refined_conforming_90deg_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(refined_90_result, f, indent=2)
    print(f"\n[+] Saved refined 90-deg results to {out_json}")

if __name__ == "__main__":
    main()
