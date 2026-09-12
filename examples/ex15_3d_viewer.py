"""3D Visualization of 4-Layer (PET-PSA) Thin Bar Folding Simulation using PyVista.

Adheres strictly to project rules:
- Background color: Black with inverted font color (White, Paraview style)
- Mesh edge color: Dark Gray (#555555)
- Common font size: 12
- Show coordinate axes
- Single colorbar
- High-efficiency animation via internal value (grid.points) in-place updates
- Interactive view toggles (XY, YZ, ZX projections, Perspective On/Off, Step Slider)
"""

import os
import sys
import pickle
import numpy as np
import pyvista as pv
import vtk

def build_pyvista_grid(mesh, u_disp=None):
    """Convert Mesh3D and displacement vector to a PyVista UnstructuredGrid."""
    nid_map = mesh.node_id_to_index()
    coords = mesh.coords.copy()
    
    if u_disp is not None:
        pts = coords.copy()
        for nid, idx in nid_map.items():
            pts[idx] += u_disp[3*idx : 3*idx + 3]
    else:
        pts = coords.copy()
        
    cells = []
    cell_types = []
    pids = []
    for eid, elem in mesh.elements.items():
        elem_indices = [nid_map[nid] for nid in elem.node_ids]
        cells.append(8)
        cells.extend(elem_indices)
        cell_types.append(vtk.VTK_HEXAHEDRON)
        pids.append(elem.pid)
        
    cells = np.array(cells, dtype=np.int64)
    cell_types = np.array(cell_types, dtype=np.uint8)
    
    grid = pv.UnstructuredGrid(cells, cell_types, pts)
    grid.cell_data['Layer'] = np.array(pids, dtype=np.int32)
    
    if u_disp is not None:
        u_mat = u_disp.reshape(-1, 3)
        grid.point_data['Displacement_mm'] = np.linalg.norm(u_mat, axis=1)
        grid.point_data['Ux_mm'] = u_mat[:, 0]
        grid.point_data['Uy_mm'] = u_mat[:, 1]
        grid.point_data['Uz_mm'] = u_mat[:, 2]
    else:
        grid.point_data['Displacement_mm'] = np.zeros(len(pts), dtype=np.float64)
        
    return grid, coords, nid_map

def render_static_3d(mesh, u_disp, out_png, view_mode='isometric', color_by='disp', title="4-Layer Thin Bar 180° Fold (3D)"):
    """Render a high-resolution 3D deformed plot with optimal framing."""
    grid, _, _ = build_pyvista_grid(mesh, u_disp)
    
    plotter = pv.Plotter(off_screen=True, window_size=(1600, 1000))
    plotter.set_background('black')
    
    sbar_args = dict(
        title="Displacement (mm)",
        title_font_size=12,
        label_font_size=10,
        color="white",
        position_x=0.84,
        position_y=0.25,
        width=0.07,
        height=0.5,
        shadow=False
    )
    
    if color_by == 'disp':
        plotter.add_mesh(
            grid,
            scalars="Displacement_mm",
            cmap="turbo",
            edge_color="darkgray",
            show_edges=True,
            line_width=1.0,
            scalar_bar_args=sbar_args
        )
    else:
        # Categorical material colors: PET (0) = RoyalBlue, PSA (1) = Gold
        cmap_mat = ["#4169E1", "#FFD700"]
        sbar_mat = dict(
            title="Layer Material",
            title_font_size=12,
            label_font_size=10,
            color="white",
            position_x=0.84,
            position_y=0.35,
            width=0.07,
            height=0.3,
            n_labels=2,
            fmt="%.0f"
        )
        plotter.add_mesh(
            grid,
            scalars="Layer",
            cmap=cmap_mat,
            edge_color="darkgray",
            show_edges=True,
            line_width=1.0,
            scalar_bar_args=sbar_mat
        )
        plotter.add_legend([("PET Layer (PID 0)", "#4169E1"), ("PSA Layer (PID 1)", "#FFD700")], face="circle", bcolor=None, size=(0.2, 0.1), loc='upper right')
    
    plotter.add_axes(
        xlabel="X (mm)",
        ylabel="Y (mm)",
        zlabel="Z (mm)",
        color="white",
        box=False
    )
    
    plotter.add_text(title, position='upper_left', font_size=12, color='white')
    
    if view_mode == 'isometric':
        plotter.view_isometric()
        plotter.camera.azimuth += 20
        plotter.camera.elevation += 10
        plotter.reset_camera()
        plotter.camera.zoom(1.4)
    elif view_mode == 'oblique':
        plotter.view_xy()
        plotter.camera.azimuth += 25
        plotter.camera.elevation += 20
        plotter.reset_camera()
        plotter.camera.zoom(1.35)
    elif view_mode == 'front':
        plotter.view_xy()
        plotter.reset_camera()
        plotter.camera.zoom(1.2)
        
    plotter.screenshot(out_png)
    plotter.close()
    print(f"Saved 3D render: {out_png}")

def render_3d_animation(mesh, history, out_gif, fps=10):
    """Render 3D folding animation via in-place internal value updates (strictly efficient)."""
    grid, ref_coords, nid_map = build_pyvista_grid(mesh, history[0])
    
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
    plotter.set_background('black')
    
    sbar_args = dict(
        title="Displacement (mm)",
        title_font_size=12,
        label_font_size=10,
        color="white",
        position_x=0.84,
        position_y=0.25,
        width=0.07,
        height=0.5,
        shadow=False
    )
    
    max_disp = max(np.max(np.linalg.norm(u.reshape(-1, 3), axis=1)) for u in history)
    
    plotter.add_mesh(
        grid,
        scalars="Displacement_mm",
        clim=[0.0, max_disp],
        cmap="turbo",
        edge_color="darkgray",
        show_edges=True,
        line_width=0.8,
        scalar_bar_args=sbar_args
    )
    
    plotter.add_axes(xlabel="X", ylabel="Y", zlabel="Z", color="white")
    
    # Camera setup focused on the full U-fold envelope
    plotter.view_isometric()
    plotter.camera.azimuth += 20
    plotter.camera.elevation += 10
    plotter.reset_camera()
    plotter.camera.zoom(1.25)
    
    plotter.open_gif(out_gif, fps=fps)
    
    n_frames = len(history)
    print(f"Generating 3D animation ({n_frames} frames)...")
    for frame_idx, u_vec in enumerate(history):
        # IN-PLACE UPDATE of coordinates and scalars (NO recreation)
        pts_new = ref_coords.copy()
        for nid, idx in nid_map.items():
            pts_new[idx] += u_vec[3*idx : 3*idx + 3]
            
        grid.points = pts_new
        grid.point_data['Displacement_mm'] = np.linalg.norm(u_vec.reshape(-1, 3), axis=1)
        
        plotter.add_text(
            f"3D Multilayer Folding (Step {frame_idx+1}/{n_frames}) | 180° Roller Draw-In",
            position='upper_left',
            font_size=12,
            color='white',
            name='anim_title'
        )
        
        plotter.write_frame()
        
    plotter.close()
    print(f"Saved 3D animation: {out_gif}")

def launch_interactive_viewer(mesh, history):
    """Launch interactive PyVista 3D GUI window with projection toggles and step slider."""
    grid, ref_coords, nid_map = build_pyvista_grid(mesh, history[-1])
    
    plotter = pv.Plotter(window_size=(1600, 1000))
    plotter.set_background('black')
    
    sbar_args = dict(
        title="Displacement (mm)",
        title_font_size=12,
        label_font_size=10,
        color="white",
        position_x=0.88,
        position_y=0.20,
        width=0.05,
        height=0.6
    )
    
    max_disp = max(np.max(np.linalg.norm(u.reshape(-1, 3), axis=1)) for u in history)
    
    plotter.add_mesh(
        grid,
        scalars="Displacement_mm",
        clim=[0.0, max_disp],
        cmap="turbo",
        edge_color="darkgray",
        show_edges=True,
        line_width=1.0,
        scalar_bar_args=sbar_args
    )
    
    plotter.add_axes(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)", color="white")
    
    # View toggles
    def view_xy():
        plotter.view_xy()
        plotter.add_text("View: Front (XY Projection)", position='lower_left', font_size=10, color='yellow', name='vinfo')
    def view_yz():
        plotter.view_yz()
        plotter.add_text("View: Side (YZ Projection)", position='lower_left', font_size=10, color='yellow', name='vinfo')
    def view_zx():
        plotter.view_zx()
        plotter.add_text("View: Top (ZX Projection)", position='lower_left', font_size=10, color='yellow', name='vinfo')
    def view_iso():
        plotter.view_isometric()
        plotter.add_text("View: Isometric 3D", position='lower_left', font_size=10, color='yellow', name='vinfo')
    def toggle_perspective():
        curr = plotter.camera.parallel_projection
        plotter.camera.parallel_projection = not curr
        mode = "Parallel (Orthographic)" if not curr else "Perspective"
        plotter.add_text(f"Projection: {mode}", position='lower_left', font_size=10, color='yellow', name='vinfo')
        
    plotter.add_key_event('1', view_xy)
    plotter.add_key_event('2', view_yz)
    plotter.add_key_event('3', view_zx)
    plotter.add_key_event('4', view_iso)
    plotter.add_key_event('p', toggle_perspective)
    
    # Step Slider Callback for real-time deformation scrubbing
    def on_step_change(value):
        step = int(np.clip(round(value), 0, len(history) - 1))
        u_vec = history[step]
        pts_new = ref_coords.copy()
        for nid, idx in nid_map.items():
            pts_new[idx] += u_vec[3*idx : 3*idx + 3]
        grid.points = pts_new
        grid.point_data['Displacement_mm'] = np.linalg.norm(u_vec.reshape(-1, 3), axis=1)
        plotter.add_text(
            f"Step: {step + 1}/{len(history)}",
            position='upper_left',
            font_size=12,
            color='white',
            name='step_info'
        )

    plotter.add_slider_widget(
        callback=on_step_change,
        rng=[0, len(history) - 1],
        value=len(history) - 1,
        title="Simulation Step",
        color="white",
        pointa=(0.25, 0.08),
        pointb=(0.75, 0.08)
    )
    
    help_text = (
        "Controls:\n"
        "  [1]: XY Front View   [2]: YZ Side View   [3]: ZX Top View   [4]: Isometric 3D\n"
        "  [P]: Toggle Perspective/Parallel   [Slider]: Scrub Simulation Steps\n"
        "  [Left Click + Drag]: Rotate   [Middle / Shift+Left]: Pan   [Scroll]: Zoom"
    )
    plotter.add_text(help_text, position='upper_right', font_size=9, color='lightgray')
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.camera.zoom(1.3)
    
    print("Launching PyVista Interactive 3D Viewer...")
    print(help_text)
    plotter.show()

def main():
    res_path = 'examples/ex15_result.pkl'
    if not os.path.exists(res_path):
        print(f"Error: {res_path} not found. Please run examples/ex15_multilayer_thin_bar_fold.py first.")
        return
        
    print(f"Loading {res_path}...")
    with open(res_path, 'rb') as f:
        res = pickle.load(f)
        
    mesh = res['mesh']
    u_disp = res['displacement']
    history = res.get('history', [u_disp])
    
    # 1. Render High-Resolution Static 3D Images
    out_iso = 'examples/ex15_3d_deformed_isometric.png'
    out_oblique = 'examples/ex15_3d_oblique_front.png'
    out_mat = 'examples/ex15_3d_materials_isometric.png'
    
    render_static_3d(mesh, u_disp, out_iso, view_mode='isometric', color_by='disp', title="4-Layer Thin Bar: 180° Fold Isometric 3D View (Displacement)")
    render_static_3d(mesh, u_disp, out_oblique, view_mode='oblique', color_by='disp', title="4-Layer Thin Bar: 3D Oblique View (U-Fold & Draw-In)")
    render_static_3d(mesh, u_disp, out_mat, view_mode='isometric', color_by='material', title="4-Layer Thin Bar: Multilayer Materials (PET & PSA)")
    
    # 2. Render 3D Animation GIF
    out_gif = 'examples/ex15_3d_folding_animation.gif'
    if len(history) > 1:
        render_3d_animation(mesh, history, out_gif, fps=10)
        
    # 3. If interactive flag passed or run directly in windowed env
    if "--interactive" in sys.argv or "-i" in sys.argv:
        launch_interactive_viewer(mesh, history)
    else:
        print("\nNote: To launch the interactive real-time 3D GUI window, run:")
        print("  python examples/ex15_3d_viewer.py --interactive")

if __name__ == '__main__':
    main()
