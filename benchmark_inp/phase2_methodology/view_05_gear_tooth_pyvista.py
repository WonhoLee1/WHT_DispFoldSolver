"""
view_05_gear_tooth_pyvista.py
=============================
Interactive 2D/3D PyVista viewer for Deck #05: 2D Gear Tooth Finite Sliding & Friction Contact.
Renders transient sliding animation and displacement/stress contours from VTKHDF (.vtkhdf) files.

Features & Controls (user_rules compliant):
- Guaranteed interactive frame updates via PyVista named actor swap (name='main_mesh', reset_camera=False)
- Interactive time-step Slider Widget (Mouse drag forward / backward)
- Console real-time step status feedback
- Background: Black (Paraview style), Text: White
- Mesh edges: Dark Gray (#404040)
- Show coordinate axes
- Single unified colorbar
- Hotkeys:
    'Left' / 'Right' Arrow: Step backward / forward
    '[' / ']': Step backward / forward
    'A' / 'D': Step backward / forward
    'Space': Toggle automatic playback animation
    '1': Front (XY) projection view
    '2': Side (YZ) view
    '3': Top (ZX / Plan) view
    '4': Isometric view
    'P': Toggle Perspective / Orthographic projection
    'S': Save high-resolution PNG screenshot
    'Q': Quit viewer
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import h5py
import pyvista as pv


def load_vtkhdf_dataset(filepath: str):
    """Load VTKHDF file (supports both Version 1.0 snapshot and Version 2.2 transient)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"VTKHDF file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        if "VTKHDF" not in f:
            raise ValueError(f"Invalid VTKHDF format in {filepath}: 'VTKHDF' root group missing.")
        g = f["VTKHDF"]

        # Read static topology
        types = np.array(g["Types"], dtype=np.uint8)
        offsets = np.array(g["Offsets"], dtype=np.int64)
        connectivity = np.array(g["Connectivity"], dtype=np.int64)

        n_cells = len(types)

        # Build PyVista cells array: [n_pts, id0, id1, ..., n_pts, id0, id1, ...]
        cell_list = []
        for i in range(n_cells):
            start = offsets[i]
            end = offsets[i + 1]
            c_nodes = connectivity[start:end]
            cell_list.append(len(c_nodes))
            cell_list.extend(c_nodes)
        pv_cells = np.array(cell_list, dtype=np.int64)

        # Check for transient Steps
        is_transient = "Steps" in g and "Values" in g["Steps"]
        if is_transient:
            steps_g = g["Steps"]
            times = np.array(steps_g["Values"], dtype=np.float64)
            n_steps = len(times)
            pts_dataset = np.array(g["Points"], dtype=np.float64)
            n_pts_per_step = len(pts_dataset) // n_steps

            # Extract step grids
            step_meshes = []
            for s in range(n_steps):
                p_start = s * n_pts_per_step
                p_end = (s + 1) * n_pts_per_step
                pts_s = pts_dataset[p_start:p_end]

                grid = pv.UnstructuredGrid(pv_cells, types, pts_s)

                # Extract PointData
                if "PointData" in g:
                    pd_g = g["PointData"]
                    for name in pd_g.keys():
                        arr = np.array(pd_g[name])
                        if len(arr) == len(pts_dataset):
                            grid.point_data[name] = arr[p_start:p_end]

                step_meshes.append(grid)

            return {
                "is_transient": True,
                "times": times,
                "grids": step_meshes,
                "n_steps": n_steps,
            }
        else:
            # Static snapshot
            pts = np.array(g["Points"], dtype=np.float64)
            grid = pv.UnstructuredGrid(pv_cells, types, pts)

            if "PointData" in g:
                for k in g["PointData"].keys():
                    grid.point_data[k] = np.array(g["PointData"][k])
            if "CellData" in g:
                for k in g["CellData"].keys():
                    grid.cell_data[k] = np.array(g["CellData"][k])

            return {
                "is_transient": False,
                "times": np.array([0.0]),
                "grids": [grid],
                "n_steps": 1,
            }


def view_gear_tooth(vtkhdf_path: str, screenshot_path: str = None, off_screen: bool = False):
    """Launch interactive PyVista visualization window with guaranteed frame swapping."""
    data = load_vtkhdf_dataset(vtkhdf_path)
    grids = data["grids"]
    times = data["times"]
    n_steps = data["n_steps"]

    # Compute global displacement magnitude range for unified colorbar
    disp_max = 0.0
    for g in grids:
        if "Displacement" in g.point_data:
            d_mag = np.linalg.norm(g.point_data["Displacement"][:, :2], axis=1)
            g.point_data["Disp_Mag"] = d_mag
            disp_max = max(disp_max, float(np.max(d_mag)))
        else:
            g.point_data["Disp_Mag"] = np.zeros(g.n_points)

    if disp_max < 1e-12:
        disp_max = 0.25

    clim = [0.0, disp_max]

    # Create Plotter
    plotter = pv.Plotter(off_screen=off_screen, window_size=[1200, 900])
    plotter.set_background("black")

    # Start at step 0 (initial unengaged position) so user clearly sees the motion from flat to sliding!
    init_step = 0
    current_step = [init_step]
    anim_running = [False]

    # Colorbar definition (single unified colorbar)
    sbar_kwargs = {
        "title": "Displacement Mag (mm)",
        "title_font_size": 12,
        "label_font_size": 10,
        "color": "white",
        "vertical": True,
        "position_x": 0.86,
        "position_y": 0.25,
        "height": 0.5,
        "width": 0.07,
    }

    # Add initial mesh
    plotter.add_mesh(
        grids[init_step],
        scalars="Disp_Mag",
        cmap="turbo",
        clim=clim,
        show_edges=True,
        edge_color="#404040",
        scalar_bar_args=sbar_kwargs,
        name="main_mesh",
        reset_camera=True,
    )

    header_text = (
        f"Deck #05: 2D Gear Tooth Finite Sliding & Friction Contact\n"
        f"Step {init_step+1}/{n_steps} | Time: {times[init_step]:.4f}s"
    )
    plotter.add_text(header_text, name="step_info", position="upper_left", color="white", font_size=12)

    # Reliable mesh replacement function (using PyVista's actor swap by name)
    def set_step(step_idx: float | int):
        idx = int(round(step_idx))
        idx = max(0, min(idx, n_steps - 1))
        current_step[0] = idx

        # Re-add with identical name='main_mesh' and reset_camera=False
        # This replaces the actor in VTK without touching camera or colorbar!
        plotter.add_mesh(
            grids[idx],
            scalars="Disp_Mag",
            cmap="turbo",
            clim=clim,
            show_edges=True,
            edge_color="#404040",
            name="main_mesh",
            reset_camera=False,
            show_scalar_bar=False,
        )

        t_val = times[idx]
        title_text = (
            f"Deck #05: 2D Gear Tooth Finite Sliding & Friction Contact\n"
            f"Step {idx+1}/{n_steps} | Time: {t_val:.4f}s"
        )
        plotter.add_text(title_text, name="step_info", position="upper_left", color="white", font_size=12)
        print(f"  -> Displaying Step {idx+1:2d}/{n_steps} (Time = {t_val:.4f}s)", flush=True)

    # Navigation wrappers
    def step_forward():
        target = min(current_step[0] + 1, n_steps - 1)
        if target != current_step[0]:
            set_step(target)
            if slider_widget is not None:
                slider_widget.GetRepresentation().SetValue(target)

    def step_backward():
        target = max(current_step[0] - 1, 0)
        if target != current_step[0]:
            set_step(target)
            if slider_widget is not None:
                slider_widget.GetRepresentation().SetValue(target)

    # Interactive slider widget
    slider_widget = None
    if n_steps > 1 and not off_screen:
        slider_widget = plotter.add_slider_widget(
            callback=set_step,
            rng=[0, n_steps - 1],
            value=init_step,
            title="Drag Slider to Change Time Step",
            pointa=(0.15, 0.06),
            pointb=(0.85, 0.06),
            style="modern",
            color="cyan",
            title_color="white",
            event_type="always",
        )

    # Animation timer callback
    def animation_loop():
        if anim_running[0]:
            next_step = (current_step[0] + 1) % n_steps
            set_step(next_step)
            if slider_widget is not None:
                slider_widget.GetRepresentation().SetValue(next_step)

    def toggle_animation():
        anim_running[0] = not anim_running[0]
        state_str = "STARTED" if anim_running[0] else "PAUSED"
        print(f"[*] Animation {state_str} (Press Space to toggle)", flush=True)

    # Camera setup
    plotter.show_axes()
    plotter.view_xy()

    # Keyboard hotkey bindings
    plotter.add_key_event("Left", step_backward)
    plotter.add_key_event("Right", step_forward)
    plotter.add_key_event("a", step_backward)
    plotter.add_key_event("d", step_forward)
    plotter.add_key_event("A", step_backward)
    plotter.add_key_event("D", step_forward)
    plotter.add_key_event("[", step_backward)
    plotter.add_key_event("]", step_forward)
    plotter.add_key_event("space", toggle_animation)

    # View orientations
    plotter.add_key_event("1", lambda: plotter.view_xy())
    plotter.add_key_event("2", lambda: plotter.view_zy())
    plotter.add_key_event("3", lambda: plotter.view_xz())
    plotter.add_key_event("4", lambda: plotter.view_isometric())
    plotter.add_key_event("p", lambda: plotter.camera.SetParallelProjection(not plotter.camera.GetParallelProjection()))
    plotter.add_key_event("P", lambda: plotter.camera.SetParallelProjection(not plotter.camera.GetParallelProjection()))

    if screenshot_path:
        os.makedirs(os.path.dirname(os.path.abspath(screenshot_path)), exist_ok=True)
        plotter.screenshot(screenshot_path)
        print(f"[+] Saved high-resolution screenshot to: {screenshot_path}")

    if not off_screen:
        print("\n" + "=" * 80)
        print("  DECK #05 PYVISTA VIEWER CONTROLS")
        print("=" * 80)
        print("  [Step Navigation]")
        print("    - Mouse: Drag the Cyan Slider Bar at bottom of screen")
        print("    - Keyboard Left / Right Arrows:  Step backward / forward")
        print("    - Keyboard 'A' / 'D' keys:        Step backward / forward")
        print("    - Keyboard '[' / ']' keys:        Step backward / forward")
        print("    - Spacebar:                       Play / Pause continuous animation")
        print("  [Views]")
        print("    - '1': Front (XY) | '2': Side (YZ) | '3': Top (ZX) | '4': Isometric")
        print("    - 'P': Perspective / Ortho toggle | 'S': Screenshot | 'Q': Quit")
        print("=" * 80 + "\n", flush=True)

        # Set timer callback for continuous animation playback
        plotter.create_timer(50, animation_loop)
        plotter.show()


def main():
    parser = argparse.ArgumentParser(description="PyVista Interactive Viewer for Deck #05 Gear Tooth Contact")
    parser.add_argument("--file", type=str, default=None, help="Path to .vtkhdf file")
    parser.add_argument("--method", type=str, default="dual_mortar", choices=["dual_mortar", "nitsche", "penalty"], help="Contact formulation")
    parser.add_argument("--screenshot", type=str, default=None, help="Path to save screenshot PNG")
    parser.add_argument("--off_screen", action="store_true", help="Render off-screen without opening GUI window")

    args = parser.parse_args()

    vtkhdf_path = args.file
    if vtkhdf_path is None:
        vtkhdf_path = os.path.join(
            "output_05_gear_tooth_finite_sliding_2d",
            f"05_gear_tooth_{args.method}_transient.vtkhdf"
        )
        if not os.path.exists(vtkhdf_path):
            vtkhdf_path = os.path.join(
                "output_05_gear_tooth_finite_sliding_2d",
                f"05_gear_tooth_{args.method}_final.vtkhdf"
            )

    print(f"[*] Loading VTKHDF: {vtkhdf_path}")
    view_gear_tooth(vtkhdf_path, screenshot_path=args.screenshot, off_screen=args.off_screen)


if __name__ == "__main__":
    main()
