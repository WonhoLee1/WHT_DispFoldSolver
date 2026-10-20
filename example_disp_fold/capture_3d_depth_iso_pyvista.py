"""
capture_3d_depth_iso_pyvista.py
===============================
Renders high-resolution 3D Isometric (Iso) view captures for FrontISTR 3D Depth Scaling cases
(nz = 1, 10, 20), clearly showing the 3D extrusion width, element divisions through thickness,
and anticlastic edge effects in ParaView dark style (Black background, white font size 12,
dark gray edges, coordinate axes, single colorbar).
"""

from __future__ import annotations
import os
import sys
import json
import re
from typing import Dict, Any, List
import numpy as np
import pyvista as pv
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.model_spec import UnifiedFoldModelConfig, MeshSpec
from example_disp_fold.run_fistr_teardrop import FrontISTRTeardropBuilder, parse_fistr_res_file
from example_disp_fold.benchmark_and_capture_fistr_xy import build_pyvista_deformed_grid

CASE_DIR = os.path.join(REPO_ROOT, "example_disp_fold", "results_benchmark_refinement")
OUT_FIG_DIR = os.path.join(REPO_ROOT, "dev_log", "figures", "cases")
os.makedirs(OUT_FIG_DIR, exist_ok=True)


def render_pyvista_iso_capture(
    builder: FrontISTRTeardropBuilder,
    res_data: Dict[str, np.ndarray],
    title_label: str,
    out_png_path: str,
) -> None:
    """
    Renders a 3D Isometric ParaView-style capture:
    - Left Viewport: Full 3D Isometric View (Material Coloring + dark gray edges)
    - Right Viewport: Hinge Tear-Drop Loop 3D Isometric Zoom-In (von Mises + 1 colorbar)
    """
    pv.OFF_SCREEN = True
    grid = build_pyvista_deformed_grid(builder, res_data)

    pl = pv.Plotter(shape=(1, 2), window_size=(1920, 1000), off_screen=True)

    # --- Left: Full 3D Isometric View ---
    pl.subplot(0, 0)
    pl.set_background("black")
    pl.add_mesh(
        grid,
        scalars="Material_ID",
        cmap=["#7f7f7f", "#17becf", "#1f77b4", "#ff7f0e"],
        show_edges=True,
        edge_color="#444444",
        line_width=1.0,
        show_scalar_bar=False,
        lighting=True,
    )
    pl.view_isometric()
    pl.camera.focal_point = (0.0, 18.0, 0.0)
    pl.camera.position = (55.0, 48.0, 60.0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.zoom(1.15)
    pl.add_axes(color="white", line_width=2)
    pl.add_text(
        f"{title_label}\n[Full 3D Isometric View]\nGray=Plate, Cyan=Glass, Blue=PET, Orange=PSA",
        position="upper_left",
        font_size=12,
        color="white",
    )

    # --- Right: Hinge Tear-Drop Bulb 3D Isometric Zoom-In ---
    pl.subplot(0, 1)
    pl.set_background("black")
    sbar_args = dict(
        title="Von Mises [MPa]",
        color="white",
        title_font_size=12,
        label_font_size=12,
        n_labels=5,
        position_x=0.84,
        position_y=0.15,
        width=0.04,
        height=0.70,
        vertical=True,
    )
    pl.add_mesh(
        grid,
        scalars="Mises_MPa",
        cmap="turbo",
        clim=[0.0, 2800.0],
        show_edges=True,
        edge_color="#444444",
        line_width=1.0,
        show_scalar_bar=True,
        scalar_bar_args=sbar_args,
        lighting=True,
    )
    pl.view_isometric()
    pl.camera.focal_point = (0.0, 3.5, 0.0)
    pl.camera.position = (18.0, 16.0, 22.0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.zoom(1.25)
    pl.add_axes(color="white", line_width=2)
    pl.add_text(
        f"{title_label}\n[3D Hinge Bulb Isometric Zoom-In]\nZ-Depth Mesh Lines & 3D Stress Concentration",
        position="upper_left",
        font_size=12,
        color="white",
    )

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    pl.screenshot(out_png_path)
    pl.close()
    print(f"[OK] Saved PyVista 3D Iso Capture: {out_png_path}")


def capture_single_3d_depth_case(case_name: str) -> str:
    run_dir = os.path.join(CASE_DIR, case_name)
    summary_path = os.path.join(run_dir, "fistr_summary.json")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    dim = summary["dim"]
    preset = summary["preset"]
    substeps = summary["substeps"]

    m_nz = re.search(r"nz(\d+)", case_name)
    nz = int(m_nz.group(1)) if m_nz else 2

    if "Depth50" in case_name:
        nx_h, nx_p = 60, 40
        base_desc = "50% Mesh"
    else:
        nx_h, nx_p = 30, 20
        base_desc = "100% Mesh"

    mspec = MeshSpec(nx_hinge=nx_h, nx_plate=nx_p, nz_elements=nz)
    config = UnifiedFoldModelConfig(preset=preset, dim=dim)
    config.geometry.width_z = float(nz) * 1.0
    builder = FrontISTRTeardropBuilder(
        config=config,
        psa_form361="FI",
        solid_form361="FI",
        substeps=substeps,
        solver_method="MUMPS",
        mesh_spec=mspec,
    )

    res_path = os.path.join(run_dir, f"fistr_teardrop.res.0.{substeps}")
    res_data = parse_fistr_res_file(res_path, builder.real_max_nid)

    out_png_path = os.path.join(OUT_FIG_DIR, f"{case_name}_iso.png")
    dofs = summary.get("dofs", summary.get("num_real_nodes", 0) * 3)
    wall_time = summary.get("wall_time_s", 0.0)

    title_label = (
        f"FrontISTR 3D Foldable Display 180° Fold (ISO 3D)\n"
        f"[{case_name}] ({base_desc}, nz={nz} Depth Layers | Elements={summary['num_real_elements']}, DOFs={dofs}, Time={wall_time:.2f}s)"
    )

    render_pyvista_iso_capture(builder, res_data, title_label, out_png_path)
    return out_png_path


def main():
    depth_cases = [
        "FISTR_3D_Depth100_nz1",
        "FISTR_3D_Depth100_nz10",
        "FISTR_3D_Depth100_nz20",
        "FISTR_3D_Depth50_nz1",
        "FISTR_3D_Depth50_nz10",
        "FISTR_3D_Depth50_nz20",
    ]

    print("=" * 80)
    print(f"[*] Capturing {len(depth_cases)} 3D Depth Cases in Isometric (Iso) View...")
    print("=" * 80)

    for cname in depth_cases:
        try:
            png = capture_single_3d_depth_case(cname)
            print(f"[OK] {cname} -> {png}")
        except Exception as e:
            print(f"[ERROR] {cname}: {e}")

    # Build 3D Depth nz=1, 10, 20 Comparison Montage (100% mesh)
    montage_cases_100 = [
        "FISTR_3D_Depth100_nz1_iso.png",
        "FISTR_3D_Depth100_nz10_iso.png",
        "FISTR_3D_Depth100_nz20_iso.png",
    ]
    imgs_100 = [Image.open(os.path.join(OUT_FIG_DIR, f)) for f in montage_cases_100 if os.path.exists(os.path.join(OUT_FIG_DIR, f))]
    if len(imgs_100) == 3:
        total_h = sum(im.height for im in imgs_100)
        montage = Image.new("RGB", (imgs_100[0].width, total_h))
        y_off = 0
        for im in imgs_100:
            montage.paste(im, (0, y_off))
            y_off += im.height
        montage_path = os.path.join(REPO_ROOT, "dev_log", "figures", "montage_3d_depth_iso_nz1_10_20.png")
        montage.save(montage_path, quality=90)
        print(f"[OK] Saved 3D Depth ISO Montage: {montage_path}")


if __name__ == "__main__":
    main()
