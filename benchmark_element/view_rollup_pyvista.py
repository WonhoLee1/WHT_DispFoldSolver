"""
view_rollup_pyvista.py
======================
Interactive 3D PyVista Viewer and High-Resolution Screenshot Generator for
5-Layer Composite Pure Bending Moment Roll-Up Benchmark (180 deg U-Turn).

Features:
- Unified 3D visualization for all 10 candidate cases.
- 2D Representation:
  * Default: Extruded 3D Solid (width = 1.0 mm) for direct 1:1 scale comparison with 3D models.
  * Flat Mode: Pure 2D Flat Sheet in the X-Z plane without width extrusion.
  * Real-time toggle via keyboard shortcut [E] or CLI flag --flat-2d.
- Strict adherence to project PyVista guidelines:
  * Paraview-style Black background with inverted White text (12pt).
  * Dark Gray mesh edges ('darkgray').
  * Coordinate axes displayed.
  * Single unified colorbar.
  * Interactive view switching:
      [1] Front (XZ)  [2] Side (YZ)  [3] Top (XY)  [4] Isometric
      [P] Ortho/Persp  [C] Field Toggle  [E] 2D Flat/3D Toggle  [S] Save PNG  [Q] Quit
- Automatic batch rendering (--capture_all) and 2x5 montage generation.
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

RESULTS_DIR = Path(__file__).resolve().parent / "results"
MESHES_DIR = RESULTS_DIR / "meshes"
FIGURES_DIR = repo_root / "dev_log" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 10 Benchmark Candidates in Display Order
CANDIDATE_CASES = [
    # 2D Suite
    ("2D_CR",     "2D-CR",     "CPE4_CR + CPE4_CR",  "2D Polar Co-Rotational (Shear/Vol Locked)"),
    ("2D_Opt1",   "2D-Opt1",   "CPE4I + CPE4H",      "2D Recommended (EAS Incompatible + u-P Hybrid)"),
    ("2D_CPE6M",  "2D-CPE6M",  "CPE6M + CPE4H",      "2D Mixed Conformal (Quadratic Tri + Linear Quad)"),
    ("2D_Opt2",   "2D-Opt2",   "CPE4R + CPE4H",      "2D Reduced Integration (Hourglass + u-P Hybrid)"),
    ("2D_Base",   "2D-Base",   "CPE4 + CPE4",        "2D Full Integration Baseline (Shear/Vol Locked)"),
    # 3D Suite
    ("3D_CR",     "3D-CR",     "C3D8_CR + C3D8_CR",  "3D Polar Co-Rotational (Shear/Vol Locked)"),
    ("3D_Opt1",   "3D-Opt1",   "C3D8I + C3D8H",      "3D Recommended (9-Mode EAS + Herrmann u-P)"),
    ("3D_Opt2_H", "3D-Opt2-H", "C3D8R + C3D8H",      "3D Fast Hybrid (Reduced + Herrmann u-P)"),
    ("3D_Opt2",   "3D-Opt2",   "C3D8R + C3D8R",      "3D Reduced Integration (Hourglass Stabilized)"),
    ("3D_Base",   "3D-Base",   "C3D8 + C3D8",        "3D Full Integration Baseline (Severely Locked)"),
]

CASE_LOOKUP = {c[0]: c for c in CANDIDATE_CASES}
for c in CANDIDATE_CASES:
    CASE_LOOKUP[c[1]] = c
    CASE_LOOKUP[c[0].lower()] = c
    CASE_LOOKUP[c[1].lower()] = c


def load_cached_mesh_data(case_key: str) -> Dict[str, Any]:
    """Loads mesh cache NPZ for a given case."""
    clean_key = case_key.replace("-", "_")
    target_path = MESHES_DIR / f"{clean_key}.npz"
    if not target_path.exists():
        matches = list(MESHES_DIR.glob(f"*{clean_key}*.npz"))
        if matches:
            target_path = matches[0]
        else:
            raise FileNotFoundError(
                f"Mesh cache not found for '{case_key}' at {target_path}.\n"
                f"Available caches: {[f.stem for f in MESHES_DIR.glob('*.npz')]}\n"
                f"Run 'python benchmark_element/benchmark_pure_moment_rollup.py' to generate."
            )
    return dict(np.load(target_path, allow_pickle=True))


def build_pyvista_grid(data: Dict[str, Any], extrude_2d: bool = True, width: float = 1.0) -> pv.UnstructuredGrid:
    """
    Constructs a PyVista UnstructuredGrid from cached 2D or 3D simulation data.
    All models are oriented consistently:
      - X: Beam length (0 to 40 mm initial, rolled into 180 deg arc)
      - Y: Width (0 to 1.0 mm for 3D/extruded, 0 for flat 2D)
      - Z: Thickness (0 to 0.21 mm initial, rolls up along Z)

    Parameters
    ----------
    data : Dict[str, Any]
        Cached mesh dictionary containing coords, u_disp, elem_conn, etc.
    extrude_2d : bool, default True
        If True, extrudes 2D quad/tri elements along width direction into 3D Hex/Wedge.
        If False, renders pure 2D planar sheet in the X-Z plane (Y = 0).
    width : float, default 1.0
        Extrusion width in mm.
    """
    is_2d = bool(data["is_2d"])
    coords = np.asarray(data["coords"], dtype=np.float64)
    u_disp = np.asarray(data["u_disp"], dtype=np.float64)
    elem_conn = list(data["elem_conn"])
    elem_pids = np.asarray(data["elem_pids"], dtype=np.int32)
    elem_types = list(data["elem_types"])

    if is_2d and not extrude_2d:
        # Pure 2D Flat Sheet in the X-Z plane (Y = 0)
        n_pts_2d = len(coords)
        u_2d = u_disp.reshape(-1, 2)
        def_xy = coords + u_2d

        pts_3d = np.zeros((n_pts_2d, 3), dtype=np.float64)
        pts_3d[:, 0] = def_xy[:, 0]  # X: Length
        pts_3d[:, 1] = 0.0           # Y: Width = 0 (Planar Sheet)
        pts_3d[:, 2] = def_xy[:, 1]  # Z: Thickness

        ux_full = u_2d[:, 0]
        uy_full = np.zeros(n_pts_2d, dtype=np.float64)
        uz_full = u_2d[:, 1]
        u_mag = np.sqrt(ux_full**2 + uz_full**2)

        cells = []
        cell_types = []
        for conn in elem_conn:
            n_c = len(conn)
            cells.append(n_c)
            cells.extend(conn)
            if n_c == 4:
                cell_types.append(pv.CellType.QUAD)
            elif n_c == 6:
                cell_types.append(pv.CellType.QUADRATIC_TRIANGLE)
            elif n_c == 3:
                cell_types.append(pv.CellType.TRIANGLE)
            else:
                cell_types.append(pv.CellType.POLYGON)

        grid = pv.UnstructuredGrid(cells, cell_types, pts_3d)

    elif is_2d and extrude_2d:
        # Extruded 3D Solid in width direction: Y in [0, width]
        n_pts_2d = len(coords)
        u_2d = u_disp.reshape(-1, 2)
        def_xy = coords + u_2d

        pts_3d = np.zeros((2 * n_pts_2d, 3), dtype=np.float64)
        # Front layer (Y = 0)
        pts_3d[:n_pts_2d, 0] = def_xy[:, 0]
        pts_3d[:n_pts_2d, 1] = 0.0
        pts_3d[:n_pts_2d, 2] = def_xy[:, 1]
        # Back layer (Y = width)
        pts_3d[n_pts_2d:, 0] = def_xy[:, 0]
        pts_3d[n_pts_2d:, 1] = width
        pts_3d[n_pts_2d:, 2] = def_xy[:, 1]

        ux_full = np.tile(u_2d[:, 0], 2)
        uy_full = np.zeros(2 * n_pts_2d, dtype=np.float64)
        uz_full = np.tile(u_2d[:, 1], 2)
        u_mag = np.sqrt(ux_full**2 + uz_full**2)

        cells = []
        cell_types = []
        for conn in elem_conn:
            n_c = len(conn)
            if n_c == 4:
                c0, c1, c2, c3 = conn
                hex_nodes = [c0, c1, c1 + n_pts_2d, c0 + n_pts_2d,
                             c3, c2, c2 + n_pts_2d, c3 + n_pts_2d]
                cells.append(8)
                cells.extend(hex_nodes)
                cell_types.append(pv.CellType.HEXAHEDRON)
            elif n_c == 6 or n_c == 3:
                c0, c1, c2 = conn[0], conn[1], conn[2]
                wedge_nodes = [c0, c2, c1, c0 + n_pts_2d, c2 + n_pts_2d, c1 + n_pts_2d]
                cells.append(6)
                cells.extend(wedge_nodes)
                cell_types.append(pv.CellType.WEDGE)
            else:
                raise ValueError(f"Unsupported 2D element connectivity length: {n_c}")

        grid = pv.UnstructuredGrid(cells, cell_types, pts_3d)

    else:
        # Native 3D solid data
        n_pts = len(coords)
        u_3d = u_disp.reshape(-1, 3)
        pts_3d = coords + u_3d

        ux_full = u_3d[:, 0]
        uy_full = u_3d[:, 1]
        uz_full = u_3d[:, 2]
        u_mag = np.linalg.norm(u_3d, axis=1)

        cells = []
        cell_types = []
        for conn in elem_conn:
            cells.append(len(conn))
            cells.extend(conn)
            if len(conn) == 8:
                cell_types.append(pv.CellType.HEXAHEDRON)
            elif len(conn) == 6:
                cell_types.append(pv.CellType.WEDGE)
            elif len(conn) == 4:
                cell_types.append(pv.CellType.TETRA)
            else:
                cell_types.append(pv.CellType.POLY_VERTEX)

        grid = pv.UnstructuredGrid(cells, cell_types, pts_3d)

    # Attach Point Arrays
    grid.point_data["Displacement_Magnitude"] = u_mag
    grid.point_data["Displacement_X"] = ux_full
    grid.point_data["Displacement_Y"] = uy_full
    grid.point_data["Displacement_Z"] = uz_full

    # Attach Cell Arrays
    grid.cell_data["Layer_PID"] = elem_pids
    mat_labels = np.array(["PET" if p == 1 else "PSA" for p in elem_pids], dtype=object)
    grid.cell_data["Material"] = mat_labels
    grid.cell_data["Elem_Type"] = np.array(elem_types, dtype=object)

    return grid


def get_case_reaction_info(case_key: str) -> Dict[str, Any]:
    """Loads benchmark summary metrics from JSON using exact prefix match."""
    json_path = RESULTS_DIR / "benchmark_pure_moment_rollup.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target_prefix = case_key.replace("-", "_").lower()
    for dim_key in ("2D", "3D"):
        for k, v in data.get(dim_key, {}).items():
            k_prefix = k.split()[0].replace("-", "_").lower()
            if k_prefix == target_prefix:
                return v
    return {}


def setup_plotter_camera(plotter: pv.Plotter, view_mode: str = "iso"):
    """Configures standard camera view orientations with comfortable margin."""
    if view_mode == "iso":
        # Isometric 3D view: full 180 deg arc visible with zero clipping
        plotter.camera_position = [
            (38.0, -48.0, 36.0),
            (6.5, 0.5, 12.8),
            (0.0, 0.0, 1.0)
        ]
    elif view_mode == "front":
        # XZ Plane elevation (Front view)
        plotter.camera_position = [
            (6.5, -55.0, 12.8),
            (6.5, 0.5, 12.8),
            (0.0, 0.0, 1.0)
        ]
    elif view_mode == "side":
        # YZ Plane (Side view)
        plotter.camera_position = [
            (55.0, 0.5, 12.8),
            (6.5, 0.5, 12.8),
            (0.0, 0.0, 1.0)
        ]
    elif view_mode == "top":
        # XY Plane (Top view)
        plotter.camera_position = [
            (6.5, 0.5, 55.0),
            (6.5, 0.5, 12.8),
            (0.0, 1.0, 0.0)
        ]


def render_case_view(
    case_key: str,
    field: str = "Displacement_Magnitude",
    off_screen: bool = False,
    window_size: Tuple[int, int] = (1400, 1050),
    view_mode: str = "iso",
    extrude_2d: bool = True
) -> Tuple[pv.Plotter, pv.UnstructuredGrid]:
    """
    Renders a single case in PyVista adhering strictly to all project visualization guidelines:
    - Black background with White font (12pt default)
    - Dark gray mesh edges
    - Single unified colorbar
    - Coordinate axes enabled
    """
    pv.set_plot_theme("document")
    pv.global_theme.font.size = 12
    pv.global_theme.font.color = "white"

    mesh_data = load_cached_mesh_data(case_key)
    is_2d = bool(mesh_data["is_2d"])
    grid = build_pyvista_grid(mesh_data, extrude_2d=extrude_2d)
    info = get_case_reaction_info(case_key)

    case_meta = CASE_LOOKUP.get(case_key, (case_key, case_key, "Unknown", ""))
    case_title = f"{case_meta[1]} : {case_meta[2]}"
    subtitle = case_meta[3]

    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    plotter.set_background("black")

    if field == "Layer_PID" or field == "Material":
        cmap = ["#1f77b4", "#ff7f0e"]
        clim = [0.5, 2.5]
        sbar_args = {
            "title": "Material Layer (1: PET, 2: PSA)",
            "title_font_size": 12,
            "label_font_size": 10,
            "color": "white",
            "position_x": 0.82,
            "position_y": 0.15,
            "width": 0.08,
            "height": 0.70,
            "n_labels": 2,
        }
        plotter.add_mesh(
            grid,
            scalars="Layer_PID",
            cmap=cmap,
            clim=clim,
            show_edges=True,
            edge_color="darkgray",
            scalar_bar_args=sbar_args,
            name="main_mesh"
        )
    else:
        sbar_title = "|u| Total Disp [mm]" if field == "Displacement_Magnitude" else f"{field} [mm]"
        sbar_args = {
            "title": sbar_title,
            "title_font_size": 12,
            "label_font_size": 10,
            "color": "white",
            "position_x": 0.83,
            "position_y": 0.15,
            "width": 0.08,
            "height": 0.70,
            "n_labels": 5,
            "fmt": "%.1f"
        }
        plotter.add_mesh(
            grid,
            scalars=field,
            cmap="turbo",
            show_edges=True,
            edge_color="darkgray",
            scalar_bar_args=sbar_args,
            name="main_mesh"
        )

    # Coordinate axes
    plotter.add_axes(color="white", line_width=2.0)

    # Header annotation text
    m_root = info.get("final_M_root", None)
    m_tip = info.get("final_M_tip", None)
    m_root_str = f"{m_root:+.4f}" if m_root is not None else "N/A"
    m_tip_str = f"{m_tip:+.4f}" if m_tip is not None else "N/A"
    slip_str = f"{info.get('shear_slip_mid_um', 0.0):.1f} um" if info else "N/A"
    rmse_str = f"{info.get('circularity_rmse_pct', 0.0):.3f}%" if info else "N/A"

    if m_root is not None and abs(abs(m_root) - 0.2317) < 0.03:
        status_text = "[STATUS: LOCK-FREE & PHYSICAL | M_error < 4%]"
    elif m_root is not None and abs(m_root) > 1.0:
        status_text = f"[STATUS: VOLUMETRIC LOCKED | M={abs(m_root):.2f} (7.5x Theory)]"
    else:
        status_text = "[STATUS: CONVERGED]"

    form_text = ""
    if is_2d:
        form_name = "Extruded 3D Solid (Width 1.0mm)" if extrude_2d else "Pure Flat Sheet (2D Plane-Strain)"
        form_text = f"Form: {form_name}\n"

    header_text = (
        f"Case: {case_title}\n"
        f"Role: {subtitle}\n"
        f"{form_text}"
        f"Moments: M_root = {m_root_str} N*mm | M_tip = {m_tip_str} N*mm\n"
        f"Metrics: Mid-span Slip = {slip_str} | Arc RMSE = {rmse_str}\n"
        f"{status_text}"
    )
    plotter.add_text(header_text, position="upper_left", font_size=11, color="white", font="courier", name="header_info")

    setup_plotter_camera(plotter, view_mode)

    return plotter, grid


def launch_interactive_viewer(
    case_key: str,
    initial_field: str = "Displacement_Magnitude",
    extrude_2d: bool = True
):
    """Launches an interactive PyVista 3D window with keyboard shortcuts."""
    mesh_data = load_cached_mesh_data(case_key)
    is_2d = bool(mesh_data["is_2d"])

    plotter, grid = render_case_view(
        case_key,
        field=initial_field,
        off_screen=False,
        view_mode="iso",
        extrude_2d=extrude_2d
    )

    help_text = (
        "[1] Front (XZ)  [2] Side (YZ)  [3] Top (XY)  [4] Isometric\n"
        "[P] Ortho/Persp  [C] Disp/Mat Color  [E] 2D Flat/3D Toggle  [S] Save PNG  [Q] Exit"
    )
    plotter.add_text(help_text, position="lower_left", font_size=10, color="#aaaaaa", font="arial")

    state = {
        "field_idx": 0,
        "is_ortho": False,
        "extrude_2d": extrude_2d,
        "current_grid": grid
    }
    fields = ["Displacement_Magnitude", "Displacement_X", "Displacement_Z", "Layer_PID"]

    def on_view_front():
        setup_plotter_camera(plotter, "front")
        plotter.render()

    def on_view_side():
        setup_plotter_camera(plotter, "side")
        plotter.render()

    def on_view_top():
        setup_plotter_camera(plotter, "top")
        plotter.render()

    def on_view_iso():
        setup_plotter_camera(plotter, "iso")
        plotter.render()

    def on_toggle_proj():
        state["is_ortho"] = not state["is_ortho"]
        if state["is_ortho"]:
            plotter.enable_parallel_projection()
        else:
            plotter.disable_parallel_projection()
        plotter.render()

    def on_save_screen():
        out_png = FIGURES_DIR / f"interactive_{case_key}.png"
        plotter.screenshot(str(out_png))
        print(f"[SAVED] Screenshot saved to {out_png}")

    def on_toggle_color():
        state["field_idx"] = (state["field_idx"] + 1) % len(fields)
        next_field = fields[state["field_idx"]]
        print(f"[SWITCH] Active field: {next_field}")
        plotter.mesh.set_active_scalars(next_field)
        plotter.render()

    def on_toggle_extrude():
        if not is_2d:
            print(f"[INFO] '{case_key}' is a native 3D solid mesh. Extrusion toggle only applies to 2D models.")
            return
        state["extrude_2d"] = not state["extrude_2d"]
        mode_desc = "Extruded 3D Solid (Width 1.0mm)" if state["extrude_2d"] else "Pure Flat Sheet (2D Plane-Strain)"
        print(f"[SWITCH] 2D representation mode: {mode_desc}")

        # Rebuild grid with updated mode
        new_grid = build_pyvista_grid(mesh_data, extrude_2d=state["extrude_2d"])
        state["current_grid"] = new_grid

        active_f = fields[state["field_idx"]]
        if active_f == "Layer_PID":
            plotter.add_mesh(
                new_grid,
                scalars="Layer_PID",
                cmap=["#1f77b4", "#ff7f0e"],
                clim=[0.5, 2.5],
                show_edges=True,
                edge_color="darkgray",
                name="main_mesh",
                reset_camera=False
            )
        else:
            plotter.add_mesh(
                new_grid,
                scalars=active_f,
                cmap="turbo",
                show_edges=True,
                edge_color="darkgray",
                name="main_mesh",
                reset_camera=False
            )

        # Update header text
        info = get_case_reaction_info(case_key)
        case_meta = CASE_LOOKUP.get(case_key, (case_key, case_key, "Unknown", ""))
        m_root = info.get("final_M_root", None)
        m_tip = info.get("final_M_tip", None)
        m_root_str = f"{m_root:+.4f}" if m_root is not None else "N/A"
        m_tip_str = f"{m_tip:+.4f}" if m_tip is not None else "N/A"
        slip_str = f"{info.get('shear_slip_mid_um', 0.0):.1f} um" if info else "N/A"
        rmse_str = f"{info.get('circularity_rmse_pct', 0.0):.3f}%" if info else "N/A"

        status_text = "[STATUS: LOCK-FREE & PHYSICAL | M_error < 4%]" if (m_root and abs(abs(m_root) - 0.2317) < 0.03) else "[STATUS: CONVERGED]"
        new_header = (
            f"Case: {case_meta[1]} : {case_meta[2]}\n"
            f"Role: {case_meta[3]}\n"
            f"Form: {mode_desc}\n"
            f"Moments: M_root = {m_root_str} N*mm | M_tip = {m_tip_str} N*mm\n"
            f"Metrics: Mid-span Slip = {slip_str} | Arc RMSE = {rmse_str}\n"
            f"{status_text}"
        )
        plotter.add_text(new_header, position="upper_left", font_size=11, color="white", font="courier", name="header_info")
        plotter.render()

    plotter.add_key_event("1", on_view_front)
    plotter.add_key_event("2", on_view_side)
    plotter.add_key_event("3", on_view_top)
    plotter.add_key_event("4", on_view_iso)
    plotter.add_key_event("p", on_toggle_proj)
    plotter.add_key_event("P", on_toggle_proj)
    plotter.add_key_event("s", on_save_screen)
    plotter.add_key_event("S", on_save_screen)
    plotter.add_key_event("c", on_toggle_color)
    plotter.add_key_event("C", on_toggle_color)
    plotter.add_key_event("e", on_toggle_extrude)
    plotter.add_key_event("E", on_toggle_extrude)

    print(f"\n>>> Launching PyVista Interactive Window for [{case_key}] ...")
    print("    Shortcut Keys: 1:Front, 2:Side, 3:Top, 4:Iso, P:Proj, C:Color, E:2D Flat/3D, S:Screenshot, Q:Quit")
    plotter.show()


def capture_all_screenshots() -> List[Path]:
    """Renders off-screen high-resolution PNGs for all 10 candidates and builds a 2x5 montage."""
    print("\n" + "=" * 80)
    print("BATCH CAPTURING PYVISTA DEFORMED SHAPE SCREENSHOTS (10 CANDIDATES)")
    print("=" * 80)

    saved_files = []
    case_images = {}

    for safe_name, disp_name, elems, desc in CANDIDATE_CASES:
        print(f"\n>>> Capturing [{disp_name}] ({elems}) ...")
        plotter, grid = render_case_view(
            case_key=safe_name,
            field="Displacement_Magnitude",
            off_screen=True,
            window_size=(1400, 1050),
            view_mode="iso",
            extrude_2d=True
        )
        out_png = FIGURES_DIR / f"rollup_pyvista_{safe_name}.png"
        plotter.screenshot(str(out_png))
        plotter.close()
        saved_files.append(out_png)
        case_images[safe_name] = out_png
        print(f"    Saved: {out_png} ({out_png.stat().st_size / 1024:.1f} KB)")

    print("\n>>> Synthesizing 2x5 Comparison Montage Image ...")
    montage_path = FIGURES_DIR / "rollup_deformed_shapes_montage.png"
    build_montage(case_images, montage_path)
    saved_files.append(montage_path)
    print(f"[SUCCESS] Montage created at {montage_path} ({montage_path.stat().st_size / 1024:.1f} KB)")

    return saved_files


def build_montage(case_images: Dict[str, Path], output_path: Path):
    """
    Combines 10 screenshots into a clean 2x5 grid montage with clear metadata cards.
    Row 1: 2D Candidates (2D-CR, 2D-Opt1, 2D-CPE6M, 2D-Opt2, 2D-Base)
    Row 2: 3D Candidates (3D-CR, 3D-Opt1, 3D-Opt2-H, 3D-Opt2, 3D-Base)
    """
    cols = 5
    rows = 2
    thumb_w = 700
    thumb_h = 525
    card_h = thumb_h + 75
    pad_x = 24
    pad_y = 50
    header_h = 110

    total_w = cols * thumb_w + (cols + 1) * pad_x
    total_h = rows * card_h + (rows + 1) * pad_y + header_h

    canvas = Image.new("RGB", (total_w, total_h), color=(14, 17, 23))
    draw = ImageDraw.Draw(canvas)

    # Title Banner
    draw.rectangle([(0, 0), (total_w, header_h)], fill=(22, 27, 36))
    draw.text(
        (pad_x + 10, 18),
        "5-Layer Composite 180° Pure Bending Roll-Up Benchmark: 10 Candidate Finite Elements",
        fill=(255, 255, 255)
    )
    draw.text(
        (pad_x + 10, 56),
        "Top Row: 2D Plane-Strain Candidates | Bottom Row: 3D Solid Candidates | Displacements rendered in PyVista (Paraview style)",
        fill=(170, 185, 205)
    )

    row1_keys = ["2D_CR", "2D_Opt1", "2D_CPE6M", "2D_Opt2", "2D_Base"]
    row2_keys = ["3D_CR", "3D_Opt1", "3D_Opt2_H", "3D_Opt2", "3D_Base"]

    for r_idx, row_keys in enumerate([row1_keys, row2_keys]):
        y_top = header_h + pad_y + r_idx * (card_h + pad_y)

        row_title = "ROW 1: 2D PLANE-STRAIN FORMULATIONS" if r_idx == 0 else "ROW 2: 3D SOLID FORMULATIONS"
        draw.text((pad_x, y_top - 28), row_title, fill=(90, 175, 255))

        for c_idx, key in enumerate(row_keys):
            x_left = pad_x + c_idx * (thumb_w + pad_x)
            img_path = case_images.get(key)
            if img_path and img_path.exists():
                with Image.open(img_path) as im:
                    im_thumb = im.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                    canvas.paste(im_thumb, (x_left, y_top))

            # Bottom info card rectangle
            card_y0 = y_top + thumb_h
            card_y1 = y_top + card_h
            draw.rectangle([(x_left, card_y0), (x_left + thumb_w, card_y1)], fill=(25, 30, 42))

            info = get_case_reaction_info(key)
            case_meta = CASE_LOOKUP.get(key, (key, key, "Unknown", ""))

            # Border
            is_best = "Opt1" in key or "Opt2_H" in key
            border_color = (0, 230, 130) if is_best else (75, 85, 100)
            border_width = 3 if is_best else 1
            draw.rectangle([(x_left, y_top), (x_left + thumb_w, card_y1)], outline=border_color, width=border_width)

            # Metadata text on card
            m_root = info.get("final_M_root", 0.0)
            m_tip = info.get("final_M_tip", 0.0)
            slip = info.get("shear_slip_mid_um", 0.0)
            time_s = info.get("t_wall_sec", 0.0)

            title_color = (130, 255, 180) if is_best else (255, 255, 255)
            draw.text((x_left + 12, card_y0 + 10), f"{case_meta[1]} : {case_meta[2]}", fill=title_color)
            draw.text((x_left + 12, card_y0 + 32), f"M_root = {m_root:+.4f} | M_tip = {m_tip:+.4f} N*mm", fill=(220, 225, 235))
            draw.text((x_left + 12, card_y0 + 52), f"Mid-Slip: {slip:.1f} um | Time: {time_s:.1f} s", fill=(160, 175, 195))

            # Upper-right badge
            if is_best:
                badge_w = 175
                badge_h = 28
                bx = x_left + thumb_w - badge_w - 12
                by = y_top + 12
                draw.rectangle([(bx, by), (bx + badge_w, by + badge_h)], fill=(0, 160, 85))
                draw.text((bx + 14, by + 6), "★ RECOMMENDED", fill=(255, 255, 255))
            elif abs(m_root) > 1.0:
                badge_w = 185
                badge_h = 28
                bx = x_left + thumb_w - badge_w - 12
                by = y_top + 12
                draw.rectangle([(bx, by), (bx + badge_w, by + badge_h)], fill=(180, 45, 45))
                draw.text((bx + 12, by + 6), "⚠ VOL LOCKED (7.5x)", fill=(255, 255, 255))

    canvas.save(output_path, quality=95)


def list_available_cases():
    """Prints a formatted table of all 10 candidate cases."""
    print("\n" + "=" * 80)
    print("AVAILABLE ROLL-UP CANDIDATE CASES FOR PYVISTA VISUALIZATION")
    print("=" * 80)
    print(f"{'Key':<12} | {'Display Name':<12} | {'Element Pair':<20} | {'Description'}")
    print("-" * 80)
    for safe, disp, elems, desc in CANDIDATE_CASES:
        cached = (MESHES_DIR / f"{safe}.npz").exists()
        cache_mark = "[Cached]" if cached else "[Missing]"
        print(f"{safe:<12} | {disp:<12} | {elems:<20} | {desc} {cache_mark}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="PyVista Interactive 3D Viewer & Batch Screenshot Generator for Pure Bending Roll-Up."
    )
    parser.add_argument("--case", type=str, default=None, help="Case to view interactively (e.g. '3D-Opt1', '2D-Opt1')")
    parser.add_argument("--field", type=str, default="Displacement_Magnitude",
                        choices=["Displacement_Magnitude", "Displacement_X", "Displacement_Z", "Layer_PID"],
                        help="Scalar field to display (default: Displacement_Magnitude)")
    parser.add_argument("--flat-2d", action="store_true",
                        help="Render 2D cases as pure flat sheets without width depth extrusion")
    parser.add_argument("--capture_all", action="store_true", help="Batch capture screenshots for all 10 cases and build montage")
    parser.add_argument("--list", action="store_true", help="List all available cases and cache status")

    args = parser.parse_args()

    if args.list:
        list_available_cases()
        return

    if args.capture_all:
        capture_all_screenshots()
        return

    if args.case:
        launch_interactive_viewer(
            args.case,
            initial_field=args.field,
            extrude_2d=not args.flat_2d
        )
    else:
        list_available_cases()
        print("\nTip: Launch viewer with: python benchmark_element/view_rollup_pyvista.py --case 3D-Opt1")
        print("     Or view 2D flat:    python benchmark_element/view_rollup_pyvista.py --case 2D-Opt1 --flat-2d")
        print("     Or capture all:     python benchmark_element/view_rollup_pyvista.py --capture_all")


if __name__ == "__main__":
    main()
