"""
scratch/test_fistr_pure_plane_strain_20deg.py
Runs FrontISTR Conforming Hybrid model to 20 degrees with pure plane strain (u_z = 0 on ALL nodes).
Compares moment |M_L| directly against WHT 2D (0.5133 N*mm) and WHT 3D (0.5113 N*mm).
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

class PurePlaneStrainMeshBuilder(ConformingTeardropMeshBuilder):
    def build(self):
        super().build()
        driven_set = set(self.left_driven_nids) | set(self.right_driven_nids)
        # Constrain ALL non-driven nodes with u_z = 0 (Pure Plane Strain)
        self.z_fixed_nids = sorted([
            nid for nid, x, y, z in self.node_list[:self.real_max_nid]
            if nid not in driven_set
        ])
        print(f"[*] Pure Plane Strain active: z_fixed_nids count = {len(self.z_fixed_nids)} / {self.real_max_nid} real nodes")

class Refined20DegRunner(FrontISTRConformingTeardropRunner):
    def write_deck(self):
        super().write_deck()
        cnt_path = os.path.join(self.run_dir, "fistr_teardrop.cnt")
        with open(cnt_path, "r", encoding="utf-8") as f:
            cnt_text = f.read()
        cnt_text = cnt_text.replace("MAXITER=50, CONVERG=2.0e-4", "MAXITER=60, CONVERG=2.0e-3, CONVERG_DDISP=2.0e-3")
        with open(cnt_path, "w", encoding="utf-8") as f:
            f.write(cnt_text)

OUT_DIR_20 = os.path.join(REPO_ROOT, "example_disp_fold", "results_conforming_pure_plane_strain_20deg")

def main():
    os.environ["OMP_NUM_THREADS"] = "8"
    os.makedirs(OUT_DIR_20, exist_ok=True)

    print("=" * 100)
    print("RUNNING 20° FRONTISTR CONFORMING WITH PURE PLANE STRAIN (ALL NODES u_z = 0)")
    print("=" * 100)

    config = UnifiedFoldModelConfig(preset="teardrop", dim="2D")
    builder = PurePlaneStrainMeshBuilder(
        config=config,
        nx_hinge=55,
        nx_plate=35,
        width_z=1.0,
        ny_stiff=1,
    )
    builder.build()

    runner = Refined20DegRunner(
        builder=builder,
        run_dir=OUT_DIR_20,
        substeps=10,          # 10 substeps (2.0 deg / substep) to 20 deg
        theta_max_deg=20.0,
    )

    t0 = time.time()
    summary = runner.run()
    elapsed = time.time() - t0

    m_l = abs(summary.get("M_L_final_Nmm", 0.0))
    fx_l = abs(summary.get("Fx_L_final_N", 0.0))

    print("\n" + "=" * 100)
    print("20° PURE PLANE STRAIN RESULT COMPARISON")
    print("=" * 100)
    print(f"FrontISTR Conforming (Pure Plane Strain, all u_z=0): Moment = {m_l:.4f} N*mm | Force = {fx_l:.4f} N | Wall Time = {elapsed:.2f} s")
    print(f"WHT Solver 2D (Opt1, Pure Plane Strain)           : Moment = 0.5133 N*mm | Force = 0.0210 N")
    print(f"WHT Solver 3D (Opt1, Pure Plane Strain, all u_z=0): Moment = 0.5113 N*mm | Force = 0.0209 N")
    print(f"FrontISTR Baseline 3D (Hex 361 FI, locking)       : Moment = 4.8872 N*mm | Force = 0.5925 N")
    diff_wht2d = (m_l - 0.5133) / 0.5133 * 100.0
    diff_wht3d = (m_l - 0.5113) / 0.5113 * 100.0
    print(f"Difference vs WHT 2D: {diff_wht2d:+.2f}%")
    print(f"Difference vs WHT 3D: {diff_wht3d:+.2f}%")
    print("=" * 100)

if __name__ == "__main__":
    main()
