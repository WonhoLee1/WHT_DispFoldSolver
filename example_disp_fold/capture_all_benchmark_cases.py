"""
capture_all_benchmark_cases.py
==============================
Renders high-resolution PyVista ParaView-dark style screenshots (Black background,
White font size 12, Dark Gray mesh edges, Coordinate axes, Single colorbar) for all
benchmark cases, saving individual PNGs to dev_log/figures/cases/ and a comparison montage.
"""

from __future__ import annotations
import os
import sys
import json
import re
from typing import Dict, Any, List
import numpy as np
import pyvista as pv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig, MeshSpec
from example_disp_fold.run_fistr_teardrop import FrontISTRTeardropBuilder, parse_fistr_res_file
from example_disp_fold.benchmark_and_capture_fistr_xy import render_pyvista_xy_capture

CASE_DIR = os.path.join(REPO_ROOT, "example_disp_fold", "results_benchmark_refinement")
OUT_FIG_DIR = os.path.join(REPO_ROOT, "dev_log", "figures", "cases")
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def capture_fistr_case_pyvista(case_name: str) -> str:
    """Loads mesh and final step result from case directory, then renders PyVista capture."""
    run_dir = os.path.join(CASE_DIR, case_name)
    summary_path = os.path.join(run_dir, "fistr_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    dim = summary["dim"]
    preset = summary["preset"]
    substeps = summary["substeps"]

    # Reconstruct exact MeshSpec from case name
    w_z = 1.0
    if "Refine" in case_name:
        m = re.search(r"(\d+(?:\.\d+)?)pct", case_name)
        sc = float(m.group(1)) / 100.0 if m else 1.0
        nx_h = int(round(30 / sc))
        nx_p = int(round(20 / sc))
        mspec = MeshSpec(nx_hinge=nx_h, nx_plate=nx_p, nz_elements=1)
    elif "InPlane" in case_name:
        m = re.search(r"(\d+(?:\.\d+)?)pct", case_name)
        sc = float(m.group(1)) / 100.0 if m else 1.0
        nx_h = int(round(30 / sc))
        nx_p = int(round(20 / sc))
        mspec = MeshSpec(nx_hinge=nx_h, nx_plate=nx_p, nz_elements=1)
    elif "Depth100" in case_name:
        m = re.search(r"nz(\d+)", case_name)
        nz = int(m.group(1)) if m else 1
        mspec = MeshSpec(nx_hinge=30, nx_plate=20, nz_elements=nz)
        w_z = float(nz) * 1.0
    elif "Depth50" in case_name:
        m = re.search(r"nz(\d+)", case_name)
        nz = int(m.group(1)) if m else 1
        mspec = MeshSpec(nx_hinge=60, nx_plate=40, nz_elements=nz)
        w_z = float(nz) * 1.0
    else:
        mspec = MeshSpec(nx_hinge=30, nx_plate=20, nz_elements=1)

    config = UnifiedFoldModelConfig(preset=preset, dim=dim)
    config.geometry.width_z = w_z
    builder = FrontISTRTeardropBuilder(
        config=config,
        psa_form361="FI",
        solid_form361="FI",
        substeps=substeps,
        solver_method="MUMPS",
        mesh_spec=mspec,
    )

    res_path = os.path.join(run_dir, f"fistr_teardrop.res.0.{substeps}")
    if not os.path.exists(res_path):
        # Find highest res step
        candidates = [f for f in os.listdir(run_dir) if f.startswith("fistr_teardrop.res.0.")]
        if not candidates:
            raise FileNotFoundError(f"No result files in {run_dir}")
        candidates.sort(key=lambda x: int(x.split(".")[-1]))
        res_path = os.path.join(run_dir, candidates[-1])

    res_data = parse_fistr_res_file(res_path, builder.real_max_nid)
    out_png_path = os.path.join(OUT_FIG_DIR, f"{case_name}.png")

    dofs = summary.get("dofs", summary.get("num_real_nodes", 0) * 3)
    wall_time = summary.get("wall_time_s", 0.0)
    title_label = f"FrontISTR {dim} Foldable Display 180° Fold\n[{case_name}] (DOFs={dofs}, WallTime={wall_time:.2f}s)"
    render_pyvista_xy_capture(builder, res_data, title_label, out_png_path)
    return out_png_path


def main():
    cases_to_capture = [
        # 2D Refinement 5 steps
        "FISTR_2D_Refine_100.0pct",
        "FISTR_2D_Refine_50.0pct",
        "FISTR_2D_Refine_25.0pct",
        "FISTR_2D_Refine_12.5pct",
        "FISTR_2D_Refine_6.25pct",
        # 3D In-plane
        "FISTR_3D_InPlane_100.0pct",
        "FISTR_3D_InPlane_50.0pct",
        "FISTR_3D_InPlane_25.0pct",
        # 3D Depth
        "FISTR_3D_Depth100_nz1",
        "FISTR_3D_Depth100_nz10",
        "FISTR_3D_Depth100_nz20",
        "FISTR_3D_Depth50_nz1",
        "FISTR_3D_Depth50_nz10",
        "FISTR_3D_Depth50_nz20",
    ]

    print("=" * 80)
    print(f"[*] Capturing {len(cases_to_capture)} Benchmark Cases using PyVista...")
    print("=" * 80)

    for cname in cases_to_capture:
        try:
            png = capture_fistr_case_pyvista(cname)
            print(f"[OK] {cname} -> {png}")
        except Exception as e:
            print(f"[ERROR] {cname}: {e}")

    # Generate 5-step 2D Refinement Comparison Montage
    from PIL import Image
    montage_cases_2d = [
        "FISTR_2D_Refine_100.0pct",
        "FISTR_2D_Refine_50.0pct",
        "FISTR_2D_Refine_25.0pct",
        "FISTR_2D_Refine_12.5pct",
        "FISTR_2D_Refine_6.25pct",
    ]
    imgs = [Image.open(os.path.join(OUT_FIG_DIR, f"{c}.png")) for c in montage_cases_2d if os.path.exists(os.path.join(OUT_FIG_DIR, f"{c}.png"))]
    if len(imgs) == 5:
        # Stack vertically: each resized to 1920 x 1000
        total_h = sum(im.height for im in imgs)
        montage = Image.new("RGB", (imgs[0].width, total_h))
        y_off = 0
        for im in imgs:
            montage.paste(im, (0, y_off))
            y_off += im.height
        montage_path = os.path.join(REPO_ROOT, "dev_log", "figures", "montage_2d_refinement_5steps.png")
        montage.save(montage_path, quality=90)
        print(f"[OK] Saved 2D Refinement Montage: {montage_path}")


if __name__ == "__main__":
    main()
