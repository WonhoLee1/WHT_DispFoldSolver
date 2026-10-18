"""
benchmark_and_capture_fistr_xy.py
=================================
1. Analyzes FrontISTR (fistr1.exe) execution speed in detail:
   - Total wall time, fistr1 internal TOTAL/pre/solve time
   - MUMPS sparse factorization (setup) time vs triangular solve time vs FE assembly/IO time
   - Total Newton iterations, ms/substep, ms/iteration
   - Substep scaling speed test (90 substeps vs 30 substeps for 180-deg folding)
2. Captures high-resolution X-Y Plane Folded State visualizations using PyVista (ParaView dark style:
   black background, white font size 12, dark gray mesh edges, coordinate axes, single colorbar)
   and Matplotlib X-Y progressive folding cross-sections.
"""

from __future__ import annotations
import os
import sys
import re
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from example_disp_fold.run_fistr_teardrop import (
    FrontISTRTeardropBuilder,
    UnifiedFoldModelConfig,
    parse_fistr_res_file,
    run_fistr_case,
)


def parse_detailed_fistr_speed(run_dir: str) -> Dict[str, Any]:
    """Extracts detailed timing metrics from fistr_stdout.log and fistr_summary.json."""
    stdout_path = os.path.join(run_dir, "fistr_stdout.log")
    summary_path = os.path.join(run_dir, "fistr_summary.json")

    with open(stdout_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Extract MUMPS setup & solve times per Newton iteration
    setup_times = [float(x) for x in re.findall(r"setup time\s*:\s*([0-9.E+-]+)", text)]
    solve_times = [float(x) for x in re.findall(r"solve time\s*:\s*([0-9.E+-]+)", text)]
    iters = re.findall(r"iter:\s*(\d+),\s*res\(force\):\s*([0-9.E+-]+)", text)

    m_tot = re.search(r"TOTAL TIME \(sec\)\s*:\s*([0-9.E+-]+)", text)
    m_pre = re.search(r"pre \(sec\)\s*:\s*([0-9.E+-]+)", text)
    m_slv = re.search(r"solve \(sec\)\s*:\s*([0-9.E+-]+)", text)

    fistr_total_s = float(m_tot.group(1)) if m_tot else summary["wall_time_s"]
    fistr_pre_s = float(m_pre.group(1)) if m_pre else 0.0
    fistr_solve_s = float(m_slv.group(1)) if m_slv else fistr_total_s

    n_newton_total = len(setup_times)
    sum_mumps_setup_s = float(np.sum(setup_times))
    sum_mumps_solve_s = float(np.sum(solve_times))
    sum_mumps_total_s = sum_mumps_setup_s + sum_mumps_solve_s
    assembly_and_io_s = max(0.0, fistr_solve_s - sum_mumps_total_s)

    substeps = int(summary["substeps"])
    n_dof = int(summary["num_real_nodes"]) * 3

    return {
        "case_name": summary["case_name"],
        "dim": summary["dim"],
        "preset": summary["preset"],
        "nodes": int(summary["num_real_nodes"]),
        "elements": int(summary["num_real_elements"]),
        "dofs": n_dof,
        "substeps": substeps,
        "total_newton_iters": n_newton_total,
        "avg_iters_per_substep": n_newton_total / float(substeps),
        "wall_time_s": float(summary["wall_time_s"]),
        "fistr_total_s": fistr_total_s,
        "fistr_pre_s": fistr_pre_s,
        "fistr_solve_s": fistr_solve_s,
        "mumps_setup_sum_s": sum_mumps_setup_s,
        "mumps_solve_sum_s": sum_mumps_solve_s,
        "mumps_total_s": sum_mumps_total_s,
        "assembly_update_io_s": assembly_and_io_s,
        "ms_per_substep": (fistr_solve_s * 1000.0) / float(substeps),
        "ms_per_newton_iter": (fistr_solve_s * 1000.0) / float(max(n_newton_total, 1)),
        "M_L_Nmm": float(summary["M_L_final_Nmm"]),
        "M_R_Nmm": float(summary["M_R_final_Nmm"]),
        "loop_width_mm": float(summary["loop_width_x_mm"]),
    }


def build_pyvista_deformed_grid(
    builder: FrontISTRTeardropBuilder,
    res_data: Dict[str, np.ndarray],
):
    """Builds a PyVista UnstructuredGrid in the deformed configuration with layer IDs & stress."""
    import pyvista as pv
    import vtk

    real_nids = res_data["nids"]
    nid_to_idx = {int(nid): i for i, nid in enumerate(real_nids)}

    pts0 = np.array([builder.nodes_3d[int(n)] for n in real_nids], dtype=np.float64)
    disp = res_data["DISP"]
    pts_def = pts0 + disp

    # Map material/layer ID per real element
    mat_id_map = {"PLATE": 0, "GLASS": 1, "PET": 2, "PSA": 3}
    elem_eids = sorted(eid for eid in builder.elements_3d.keys() if eid <= builder.real_max_eid)

    eid_to_mat_id: Dict[int, int] = {}
    eid_to_layer_idx: Dict[int, int] = {}
    for l_idx, (egrp_name, eids) in enumerate(builder.egrp_map.items()):
        if egrp_name == "EL_PIVOT_REF":
            continue
        mname = builder.egrp_mat.get(egrp_name, "PET")
        mid = mat_id_map.get(mname, 2)
        for eid in eids:
            eid_to_mat_id[eid] = mid
            eid_to_layer_idx[eid] = l_idx

    cells = []
    celltypes = []
    cell_mat_ids = []
    cell_layer_ids = []

    for eid in elem_eids:
        nids = builder.elements_3d[eid]
        idxs = [nid_to_idx[n] for n in nids]
        cells.append(8)
        cells.extend(idxs)
        celltypes.append(vtk.VTK_HEXAHEDRON)
        cell_mat_ids.append(eid_to_mat_id.get(eid, 2))
        cell_layer_ids.append(eid_to_layer_idx.get(eid, 0))

    grid = pv.UnstructuredGrid(cells, celltypes, pts_def)
    grid.point_data["DISP"] = disp
    grid.point_data["U_mag"] = np.linalg.norm(disp, axis=1)
    if "NMISES" in res_data:
        grid.point_data["Mises_MPa"] = res_data["NMISES"][:, 0]
    grid.cell_data["Material_ID"] = np.array(cell_mat_ids, dtype=np.int32)
    grid.cell_data["Layer_ID"] = np.array(cell_layer_ids, dtype=np.int32)
    return grid


def render_pyvista_xy_capture(
    builder: FrontISTRTeardropBuilder,
    res_data: Dict[str, np.ndarray],
    title_label: str,
    out_png_path: str,
) -> None:
    """
    Renders a ParaView-style (black background, white text, font size 12, dark gray mesh edges,
    coordinate axes, single colorbar) X-Y plane screenshot of the 180-deg folded Tear-Drop state.
    Left viewport : Full X-Y Plane Folded Structure (Material ID coloring)
    Right viewport: Hinge Tear-Drop Loop X-Y Plane Zoom-In (von Mises Stress + 1 Colorbar)
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    grid = build_pyvista_deformed_grid(builder, res_data)

    pl = pv.Plotter(shape=(1, 2), window_size=(1920, 1000), off_screen=True)

    # --- Left Viewport: Full X-Y Plane View (Material Coloring, no colorbar so total colorbar = 1) ---
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
        lighting=False,
    )
    pl.view_xy()
    pl.enable_parallel_projection()
    pl.camera.focal_point = (0.0, 19.0, 0.0)
    pl.camera.position = (0.0, 19.0, 100.0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = 22.5
    pl.add_axes(color="white", line_width=2)
    pl.add_text(
        f"{title_label}\n[Full X-Y View: 180° Folded State]\nGray=Plate, Cyan=Glass, Blue=PET, Orange=PSA",
        position="upper_left",
        font_size=12,
        color="white",
    )

    # --- Right Viewport: Hinge Tear-Drop Loop Zoom-In X-Y View (Mises Stress + Single Colorbar) ---
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
        clim=[0.0, 2500.0],
        show_edges=True,
        edge_color="#444444",
        line_width=1.2,
        show_scalar_bar=True,
        scalar_bar_args=sbar_args,
        lighting=False,
    )
    pl.view_xy()
    pl.enable_parallel_projection()
    pl.camera.focal_point = (0.0, 3.0, 0.0)
    pl.camera.position = (0.0, 3.0, 50.0)
    pl.camera.up = (0.0, 1.0, 0.0)
    pl.camera.parallel_scale = 4.6
    pl.add_axes(color="white", line_width=2)
    pl.add_text(
        f"{title_label}\n[X-Y Hinge Tear-Drop Bulb Zoom-In]\nOmega-Loop Width = {np.max(grid.points[:,0])-np.min(grid.points[:,0]):.3f} mm",
        position="upper_left",
        font_size=12,
        color="white",
    )

    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    pl.screenshot(out_png_path)
    pl.close()
    print(f"[OK] Saved PyVista X-Y Plane Folded Capture: {out_png_path}")


def render_matplotlib_xy_progression(
    builder_4: FrontISTRTeardropBuilder,
    run_dir_4: str,
    builder_14: FrontISTRTeardropBuilder,
    run_dir_14: str,
    out_png_path: str,
) -> None:
    """
    Renders a 2x2 high-definition 2D X-Y cross-section element patch view showing:
    - Top Row: compact_4 (4-layer) X-Y folding progression (0°, 60°, 120°, 180°) & X-Y Hinge Zoom-In
    - Bottom Row: teardrop (14-layer) X-Y folding progression (0°, 60°, 120°, 180°) & X-Y Hinge Zoom-In
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    import matplotlib.patches as mpatches
    try:
        import koreanize_matplotlib  # noqa: F401
    except ImportError:
        pass

    plt.rcParams["font.size"] = 9
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=180)

    mat_colors = {
        "PLATE": "#7f7f7f",
        "GLASS": "#17becf",
        "PET": "#1f77b4",
        "PSA": "#ff7f0e",
    }

    def _draw_xy_elements(ax, builder: FrontISTRTeardropBuilder, pts_def: np.ndarray, nid_to_idx: Dict[int, int], alpha: float = 0.9, edge_lw: float = 0.35):
        polys = []
        fcolors = []
        # Draw front Z-plane quad faces (n1, n2, n3, n4) of all real hex elements at z_min
        z_min = min(pt[2] for nid, pt in builder.nodes_3d.items() if nid <= builder.real_max_nid)
        for egrp_name, eids in builder.egrp_map.items():
            if egrp_name == "EL_PIVOT_REF":
                continue
            mname = builder.egrp_mat.get(egrp_name, "PET")
            col = mat_colors.get(mname, "#1f77b4")
            for eid in eids:
                nids = builder.elements_3d[eid]
                # Check if front face is on z_min
                if abs(builder.nodes_3d[nids[0]][2] - z_min) > 1e-4:
                    continue
                q_idxs = [nid_to_idx[nids[0]], nid_to_idx[nids[1]], nid_to_idx[nids[2]], nid_to_idx[nids[3]]]
                poly_xy = pts_def[q_idxs, :2]
                polys.append(poly_xy)
                fcolors.append(col)

        pc = PolyCollection(
            polys,
            facecolors=fcolors,
            edgecolors="#222222" if alpha > 0.5 else "none",
            linewidths=edge_lw,
            alpha=alpha,
        )
        ax.add_collection(pc)

    for row_idx, (builder, run_dir, model_title) in enumerate([
        (builder_4, run_dir_4, "4층 모델 (compact_4)"),
        (builder_14, run_dir_14, "14층 OLED 전체 적층 모델 (teardrop)"),
    ]):
        ax_full = axes[row_idx, 0]
        ax_zoom = axes[row_idx, 1]

        # Plot intermediate steps (30, 60) lightly and final step (90) solidly
        for step_num, alpha_val in [(30, 0.25), (60, 0.35), (90, 0.95)]:
            res_path = os.path.join(run_dir, f"fistr_teardrop.res.0.{step_num}")
            if not os.path.exists(res_path):
                continue
            rdata = parse_fistr_res_file(res_path, builder.real_max_nid)
            real_nids = rdata["nids"]
            nid_to_idx = {int(n): i for i, n in enumerate(real_nids)}
            pts0 = np.array([builder.nodes_3d[int(n)] for n in real_nids], dtype=np.float64)
            pts_def = pts0 + rdata["DISP"]

            _draw_xy_elements(ax_full, builder, pts_def, nid_to_idx, alpha=alpha_val, edge_lw=0.25)
            if step_num == 90:
                _draw_xy_elements(ax_zoom, builder, pts_def, nid_to_idx, alpha=0.95, edge_lw=0.45)

        ax_full.plot([-1.4, 1.4], [0.0, 0.0], "ro", ms=5, label="회전 피벗 (±1.4, 0)")
        ax_full.set_title(f"{model_title} — X-Y 평면 폴딩 과정 (60° → 120° → 180° 완접)")
        ax_full.set_xlabel("X [mm]")
        ax_full.set_ylabel("Y [mm]")
        ax_full.set_xlim(-38.0, 38.0)
        ax_full.set_ylim(-2.0, 41.0)
        ax_full.set_aspect("equal", adjustable="box")
        ax_full.grid(True, linestyle=":", alpha=0.6)

        ax_zoom.axvline(-1.4, color="red", linestyle="--", lw=0.9, label="피벗 X = ±1.4 mm")
        ax_zoom.axvline(+1.4, color="red", linestyle="--", lw=0.9)
        ax_zoom.plot([-1.4, 1.4], [0.0, 0.0], "ro", ms=5)
        ax_zoom.set_title(f"{model_title} — X-Y 평면 Tear-Drop (물방울) 루프 요소망 확대")
        ax_zoom.set_xlabel("변형 후 X [mm]")
        ax_zoom.set_ylabel("변형 후 Y [mm]")
        ax_zoom.set_xlim(-3.2, 3.2)
        ax_zoom.set_ylim(-0.8, 7.2)
        ax_zoom.set_aspect("equal", adjustable="box")
        ax_zoom.grid(True, linestyle=":", alpha=0.6)

        patches = [
            mpatches.Patch(color="#7f7f7f", label="지지 플레이트 (PLATE)"),
            mpatches.Patch(color="#17becf", label="UTG 강화유리 (GLASS)"),
            mpatches.Patch(color="#1f77b4", label="보호필름/폴리이미드 (PET)"),
            mpatches.Patch(color="#ff7f0e", label="초연성 점착층 (PSA, Neo-Hookean)"),
        ]
        ax_zoom.legend(handles=patches, loc="upper right", frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=180)
    plt.close(fig)
    print(f"[OK] Saved Matplotlib X-Y Folding Element Cross-Section Capture: {out_png_path}")


def main() -> None:
    base_dir = os.path.join(REPO_ROOT, "example_disp_fold", "results_fistr")
    cases = [
        ("2D", "compact_4", os.path.join(base_dir, "FISTR_2D_compact_4_FI_FI")),
        ("3D", "compact_4", os.path.join(base_dir, "FISTR_3D_compact_4_FI_FI")),
        ("2D", "teardrop", os.path.join(base_dir, "FISTR_2D_teardrop_FI_FI")),
        ("3D", "teardrop", os.path.join(base_dir, "FISTR_3D_teardrop_FI_FI")),
    ]

    speed_records = []
    for dim, preset, rdir in cases:
        if os.path.exists(os.path.join(rdir, "fistr_stdout.log")):
            speed_records.append(parse_detailed_fistr_speed(rdir))

    # Also test fast 30-substep (3 deg/step) execution for 2D & 3D compact_4 to measure speed scaling
    for dim in ["2D", "3D"]:
        fast_dir = os.path.join(base_dir, "fast_30substeps")
        res_fast = run_fistr_case(
            dim=dim,
            preset="compact_4",
            psa_form361="FI",
            solid_form361="FI",
            substeps=30,
            threads=4,
            out_base_dir=fast_dir,
        )
        if res_fast.get("summary", {}).get("completed"):
            rec = parse_detailed_fistr_speed(res_fast["summary"]["run_dir"])
            rec["case_name"] += "_30steps"
            speed_records.append(rec)

    print("\n" + "=" * 115)
    print("FrontISTR (fistr1.exe + MUMPS) Execution Speed Breakdown")
    print("=" * 115)
    header = (
        f"{'Case':<32} | {'DOFs':>6} | {'Steps':>5} | {'Iters':>5} | "
        f"{'Total(s)':>8} | {'MUMPS Setup':>11} | {'MUMPS Solve':>11} | {'Assem+IO':>8} | {'ms/Step':>7} | {'ms/Iter':>7}"
    )
    print(header)
    print("-" * 115)
    for r in speed_records:
        print(
            f"{r['case_name']:<32} | {r['dofs']:>6d} | {r['substeps']:>5d} | {r['total_newton_iters']:>5d} | "
            f"{r['fistr_total_s']:>8.2f} | {r['mumps_setup_sum_s']:>10.2f}s | {r['mumps_solve_sum_s']:>10.2f}s | "
            f"{r['assembly_update_io_s']:>7.2f}s | {r['ms_per_substep']:>7.1f} | {r['ms_per_newton_iter']:>7.1f}"
        )
    print("=" * 115)

    with open(os.path.join(base_dir, "fistr_speed_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(speed_records, f, indent=2, ensure_ascii=False)

    # Generate PyVista X-Y captures for 3D compact_4 and 3D teardrop
    cfg_3d_4 = UnifiedFoldModelConfig(preset="compact_4", dim="3D")
    bld_3d_4 = FrontISTRTeardropBuilder(cfg_3d_4, psa_form361="FI", solid_form361="FI", substeps=90)
    res_3d_4 = parse_fistr_res_file(
        os.path.join(base_dir, "FISTR_3D_compact_4_FI_FI", "fistr_teardrop.res.0.90"),
        bld_3d_4.real_max_nid,
    )
    render_pyvista_xy_capture(
        bld_3d_4,
        res_3d_4,
        "FrontISTR 3D Teardrop (compact_4, 4-Layer)",
        os.path.join(base_dir, "fistr_xy_folded_compact_4_pyvista.png"),
    )

    cfg_3d_14 = UnifiedFoldModelConfig(preset="teardrop", dim="3D")
    bld_3d_14 = FrontISTRTeardropBuilder(cfg_3d_14, psa_form361="FI", solid_form361="FI", substeps=90)
    res_3d_14 = parse_fistr_res_file(
        os.path.join(base_dir, "FISTR_3D_teardrop_FI_FI", "fistr_teardrop.res.0.90"),
        bld_3d_14.real_max_nid,
    )
    render_pyvista_xy_capture(
        bld_3d_14,
        res_3d_14,
        "FrontISTR 3D Teardrop (14-Layer OLED Stack)",
        os.path.join(base_dir, "fistr_xy_folded_teardrop_14layer_pyvista.png"),
    )

    # Generate Matplotlib X-Y Element Patch Progression Capture
    render_matplotlib_xy_progression(
        bld_3d_4,
        os.path.join(base_dir, "FISTR_3D_compact_4_FI_FI"),
        bld_3d_14,
        os.path.join(base_dir, "FISTR_3D_teardrop_FI_FI"),
        os.path.join(base_dir, "fistr_xy_folding_progression_montage.png"),
    )


if __name__ == "__main__":
    main()
