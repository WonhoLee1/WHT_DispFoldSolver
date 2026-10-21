import os
import sys
import time
import json
import math
import subprocess
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig
from example_disp_fold.run_fistr_conforming_teardrop import (
    ConformingTeardropMeshBuilder,
    FrontISTRConformingTeardropRunner,
    parse_fistr_res_file,
    compute_reaction_moment_from_res,
)
from example_disp_fold.benchmark_14layer_wht_vs_fistr_ny1_vs_ny2 import FISTR_EXE

# Setup 8 cores
NUM_THREADS = 8
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)

def run_fistr_8core_fast(substeps=30, theta_max_deg=90.0):
    run_dir = os.path.join("example_disp_fold", "results_14layer_ny1_vs_ny2", "fistr_fast_90deg_8core")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 80)
    print(f"[*] Launching FrontISTR 3D Fast 90-Deg Teardrop Folding Benchmark")
    print(f"    Threads       : {NUM_THREADS} OpenMP Cores")
    print(f"    Substeps      : {substeps} ({theta_max_deg / substeps:.2f} deg / step)")
    print(f"    Target Theta  : {theta_max_deg} deg (Total Folding = {2 * theta_max_deg} deg)")
    print(f"    Run Directory : {run_dir}")
    print("=" * 80)

    config = UnifiedFoldModelConfig(preset="teardrop", dim="3D", elem_case="3D-Opt1")
    config.geometry.width_z = 1.0

    builder = ConformingTeardropMeshBuilder(
        config=config,
        nx_hinge=30,
        nx_plate=20,
        width_z=1.0,
        ny_stiff=1,
    )

    runner = FrontISTRConformingTeardropRunner(
        builder=builder,
        run_dir=run_dir,
        substeps=substeps,
        theta_max_deg=theta_max_deg,
    )

    t0 = time.time()
    result = runner.run()
    elapsed = time.time() - t0

    print("=" * 80)
    print(f"[*] FrontISTR 3D Run Complete!")
    print(f"    Completed     : {result.get('completed', False)}")
    print(f"    Elapsed Time  : {elapsed:.2f} s")
    print(f"    Final M_L     : {result.get('M_L_final', 0.0):+.4f} N*mm")
    print(f"    Final M_R     : {result.get('M_R_final', 0.0):+.4f} N*mm")
    print(f"    Moment Balance: {abs(result.get('M_L_final', 0.0) + result.get('M_R_final', 0.0)):.4e} N*mm")
    print(f"    Loop Width    : {result.get('loop_width_x_mm', 0.0):.4f} mm")
    print(f"    Max Mises     : {result.get('max_mises_MPa', 0.0):.2f} MPa")
    print("=" * 80)

    summary_file = os.path.join(run_dir, "fast_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result

if __name__ == "__main__":
    run_fistr_8core_fast(substeps=90, theta_max_deg=90.0)
