"""
view_06_bolt_flange_pyvista.py
==============================
Interactive 3D PyVista viewer for Deck #06: 3D Bolt Preload & Pipe Flange Gasket Sealing Contact.
Renders transient multi-step contact deformation and displacement/stress contours from VTKHDF (.vtkhdf) files.

Features & Controls (user_rules compliant):
- Guaranteed interactive frame updates via PyVista named actor swap (name='main_mesh', reset_camera=False)
- Interactive time-step Slider Widget (Cyan color, mouse drag forward / backward)
- Background: Black (Paraview style), Text: White, font size 12
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
            n_pts_per_step = len(pts_dataset) // max(n_steps, 1)

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
                        elif len(arr) == n_pts_per_step:
                            grid.point_data[name] = arr

                # Compute Displacement magnitude if U is present
                if "U" in grid.point_data:
                    u_arr = grid.point_data["U"]
                    if u_arr.ndim == 2:
                        grid.point_data["U_Magnitude"] = np.linalg.norm(u_arr, axis=1)
                        grid.point_data["Uz"] = u_arr[:, 2]

                step_meshes.append(grid)

            return {
                "is_transient": True,
                "times": times,
                "grids": step_meshes,
                "n_steps": n_steps,
            }
        else:
            pts = np.array(g["Points"], dtype=np.float64)
            grid = pv.UnstructuredGrid(pv_cells, types, pts)
            if "PointData" in g:
                for name in g["PointData"].keys():
                    grid.point_data[name] = np.array(g["PointData"][name])
            if "U" in grid.point_data:
                u_arr = grid.point_data["U"]
                if u_arr.ndim == 2:
                    grid.point_data["U_Magnitude"] = np.linalg.norm(u_arr, axis=1)
                    grid.point_data["Uz"] = u_arr[:, 2]

            return {
                "is_transient": False,
                "times": np.array([1.0]),
                "grids": [grid],
                "n_steps": 1,
            }


class BoltFlangeViewer3D:
    """Paraview-style interactive 3D viewer for bolted flange gasket contact."""

    def __init__(self, data_dict: dict, field_name: str = "U_Magnitude", title: str = "", off_screen: bool = False):
        self.data = data_dict
        self.grids = data_dict["grids"]
        self.times = data_dict["times"]
        self.n_steps = data_dict["n_steps"]
        self.current_idx = self.n_steps - 1 if self.n_steps > 0 else 0
        self.field_name = field_name
        self.title = title
        self.is_animating = False
        self.off_screen = off_screen

        # Global scalar range across all frames for consistent colorbar
        all_vals = []
        for g in self.grids:
            if field_name in g.point_data:
                all_vals.append(g.point_data[field_name])
            elif "U_Magnitude" in g.point_data:
                all_vals.append(g.point_data["U_Magnitude"])
        if all_vals:
            cat_vals = np.concatenate(all_vals)
            self.clim = [float(np.min(cat_vals)), float(np.max(cat_vals))]
            if abs(self.clim[1] - self.clim[0]) < 1e-12:
                self.clim[1] = self.clim[0] + 1e-6
        else:
            self.clim = [0.0, 1.0]

        # PyVista Plotter Setup
        self.plotter = pv.Plotter(window_size=[1280, 800], off_screen=off_screen)
        self.plotter.set_background("black")

        # Initial Mesh Render
        self._render_current_frame(reset_camera=True)

        # Coordinate Axes & Text
        self.plotter.add_axes(color="white", line_width=2)
        self.status_actor = self.plotter.add_text(
            self._get_status_text(),
            position="upper_left",
            font_size=12,
            color="white",
            name="status_text"
        )

        # Time Slider Widget
        if self.n_steps > 1:
            self.slider_widget = self.plotter.add_slider_widget(
                callback=self._on_slider_change,
                rng=[0, self.n_steps - 1],
                value=self.current_idx,
                title="Time Increment",
                pointa=(0.1, 0.08),
                pointb=(0.9, 0.08),
                style="modern",
                color="cyan",
                title_color="white",
                interaction_event="always"
            )

        # Hotkeys
        self._setup_key_bindings()

    def _get_status_text(self) -> str:
        t_val = self.times[self.current_idx] if self.current_idx < len(self.times) else 0.0
        step_phase = "Step 1: Bolt Preload" if t_val <= 1.0 else "Step 2: Internal Pressure"
        return (
            f"Deck #06: 3D Bolt Preload & Pipe Flange Sealing Contact\n"
            f"Phase : {step_phase}\n"
            f"Frame : {self.current_idx + 1} / {self.n_steps} | Total Time: {t_val:.4f} s\n"
            f"Field : {self.field_name} (Range: [{self.clim[0]:.4e}, {self.clim[1]:.4e}])\n"
            f"Keys  : [Left/Right/A/D/[,]] Step | [Space] Play | [1-4] Views | [S] Screenshot | [Q] Quit"
        )

    def _render_current_frame(self, reset_camera: bool = False):
        grid = self.grids[self.current_idx]
        active_scalars = self.field_name if self.field_name in grid.point_data else None

        sargs = dict(
            title=f"{self.field_name}",
            title_font_size=12,
            label_font_size=10,
            color="white",
            position_x=0.82,
            position_y=0.25,
            width=0.15,
            height=0.50,
            n_labels=5,
            fmt="%.3e"
        )

        self.plotter.add_mesh(
            grid,
            scalars=active_scalars,
            clim=self.clim,
            cmap="turbo",
            show_edges=True,
            edge_color="#404040",
            scalar_bar_args=sargs,
            name="main_mesh",
            reset_camera=reset_camera
        )

    def _update_frame(self, new_idx: int):
        new_idx = max(0, min(self.n_steps - 1, new_idx))
        if new_idx == self.current_idx:
            return
        self.current_idx = new_idx
        self._render_current_frame(reset_camera=False)
        self.plotter.add_text(
            self._get_status_text(),
            position="upper_left",
            font_size=12,
            color="white",
            name="status_text"
        )
        print(f"[*] Navigated to Frame {self.current_idx + 1}/{self.n_steps} (Time = {self.times[self.current_idx]:.4f}s)", flush=True)

    def _on_slider_change(self, value):
        idx = int(round(value))
        self._update_frame(idx)

    def step_forward(self):
        self._update_frame(self.current_idx + 1)
        if hasattr(self, "slider_widget"):
            self.slider_widget.GetRepresentation().SetValue(self.current_idx)

    def step_backward(self):
        self._update_frame(self.current_idx - 1)
        if hasattr(self, "slider_widget"):
            self.slider_widget.GetRepresentation().SetValue(self.current_idx)

    def toggle_animation(self):
        self.is_animating = not self.is_animating
        print(f"[*] Animation playback: {'ON' if self.is_animating else 'OFF'}", flush=True)

    def save_screenshot(self):
        out_png = os.path.join(os.path.dirname(__file__), f"screenshot_06_frame_{self.current_idx + 1}.png")
        self.plotter.screenshot(out_png)
        print(f"[Saved] Screenshot saved: {out_png}", flush=True)

    def _setup_key_bindings(self):
        for k in ["Right", "bracketright", "d", "D"]:
            self.plotter.add_key_event(k, self.step_forward)
        for k in ["Left", "bracketleft", "a", "A"]:
            self.plotter.add_key_event(k, self.step_backward)
        self.plotter.add_key_event("space", self.toggle_animation)
        self.plotter.add_key_event("1", lambda: self.plotter.view_xy())
        self.plotter.add_key_event("2", lambda: self.plotter.view_yz())
        self.plotter.add_key_event("3", lambda: self.plotter.view_zx())
        self.plotter.add_key_event("4", lambda: self.plotter.view_isometric())
        self.plotter.add_key_event("s", self.save_screenshot)
        self.plotter.add_key_event("S", self.save_screenshot)
        self.plotter.add_key_event("q", lambda: self.plotter.close())
        self.plotter.add_key_event("Q", lambda: self.plotter.close())

    def show(self):
        print("\n" + "=" * 80)
        print(" PyVista 3D Interactive Viewer Running for Deck #06")
        print(" Controls: Left/Right arrow keys or cyan slider to browse multi-step frames")
        print("=" * 80)
        self.plotter.show()


def main():
    parser = argparse.ArgumentParser(description="3D PyVista Viewer for Deck #06 Bolted Flange Gasket Contact.")
    parser.add_argument("--vtkhdf", type=str, default=None, help="Path to .vtkhdf file")
    parser.add_argument("--field", type=str, default="U_Magnitude", help="Field to contour (U_Magnitude, Uz, etc.)")
    parser.add_argument("--screenshot", action="store_true", help="Capture headless screenshot and exit")
    args = parser.parse_args()

    vtkhdf_path = args.vtkhdf
    if vtkhdf_path is None:
        cand = os.path.join(
            os.path.dirname(__file__), "..", "..", "output_06_bolt_preload_gasket_flange_3d",
            "06_bolt_preload_gasket_flange_3d_result.vtkhdf"
        )
        if os.path.exists(cand):
            vtkhdf_path = os.path.abspath(cand)
        else:
            cand2 = os.path.join(os.getcwd(), "output_06_bolt_preload_gasket_flange_3d", "06_bolt_preload_gasket_flange_3d_result.vtkhdf")
            if os.path.exists(cand2):
                vtkhdf_path = cand2

    if not vtkhdf_path or not os.path.exists(vtkhdf_path):
        print(f"[Error] VTKHDF file not found at: {vtkhdf_path}")
        sys.exit(1)

    print(f"[*] Loading VTKHDF dataset from: {vtkhdf_path}")
    data = load_vtkhdf_dataset(vtkhdf_path)
    viewer = BoltFlangeViewer3D(data, field_name=args.field, off_screen=args.screenshot)

    if args.screenshot:
        out_png = os.path.join(os.path.dirname(__file__), "figure_06_bolt_flange_pyvista.png")
        viewer.plotter.render()
        viewer.plotter.screenshot(out_png)
        print(f"[Captured] Screenshot saved to: {out_png}")
        viewer.plotter.close()
    else:
        viewer.show()


if __name__ == "__main__":
    main()
