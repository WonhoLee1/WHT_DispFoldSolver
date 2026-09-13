"""
view_ex13_pyvista.py
====================
Interactive 3D PyVista Viewer for EX13 3D Multi-Layer Display 180-Degree Folding.

Keybindings:
- '1': Front view (XZ plane, 180° folding profile)
- '2': Side cross-section view (YZ plane)
- '3': Top view (XY plane)
- '4': Isometric 3D view
- 'P': Toggle Perspective / Orthographic projection
- 'C': Toggle Displacement magnitude contour / Material layer coloring
- 'S': Save high-resolution screenshot to dev_log/figures/
- 'Q': Quit
"""

from __future__ import annotations
import os
import sys
import pickle
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import pyvista as pv
except ImportError:
    print("[ERROR] PyVista is not installed. Please install via 'pip install pyvista'.")
    sys.exit(1)


def visualize_3d_fold(result_path: str = None):
    if result_path is None:
        result_path = os.path.join(REPO_ROOT, "examples", "results", "ex13_3d_fold_result.pkl")

    if not os.path.exists(result_path):
        print(f"[!] Result file '{result_path}' not found. Run 'python examples/ex13_3d_display_plate_fold.py' first.")
        return

    with open(result_path, "rb") as f:
        data = pickle.load(f)

    mesh = data["mesh"]
    u_final = data["u_final"]
    nid_map = mesh.node_id_to_index()

    # Create PyVista UnstructuredGrid for deformed mesh
    n_nodes = len(mesh.nodes)
    orig_coords = mesh.nodes_array()
    deformed_coords = orig_coords.copy()

    for nid, idx in nid_map.items():
        deformed_coords[idx, 0] += u_final[3 * idx + 0]
        deformed_coords[idx, 1] += u_final[3 * idx + 1]
        deformed_coords[idx, 2] += u_final[3 * idx + 2]

    # Cell definitions for Hexahedron (VTK_HEXAHEDRON = 12)
    cells = []
    cell_types = []
    mat_ids = []

    for eid, elem in mesh.elements.items():
        node_indices = [nid_map[nid] for nid in elem.node_ids]
        if len(node_indices) == 8:
            cells.append(8)
            cells.extend(node_indices)
            cell_types.append(pv.CellType.HEXAHEDRON)
            mat_ids.append(getattr(elem, "pid", 1))

    cells = np.array(cells)
    grid = pv.UnstructuredGrid(cells, cell_types, deformed_coords)

    # Displacement magnitude field
    u_mags = np.zeros(n_nodes)
    for nid, idx in nid_map.items():
        ux = u_final[3 * idx + 0]
        uy = u_final[3 * idx + 1]
        uz = u_final[3 * idx + 2]
        u_mags[idx] = np.sqrt(ux**2 + uy**2 + uz**2)

    grid.point_data["Displacement"] = u_mags
    grid.cell_data["Material"] = np.array(mat_ids)

    # PyVista Plotter Setup (Paraview Style Dark Background)
    offscreen = "--screenshot" in sys.argv
    plotter = pv.Plotter(title="WHT_DispFoldSolver - 3D Multi-Layer Display 180° Folding", off_screen=offscreen)
    plotter.set_background("black")

    # Add Coordinate Axes
    plotter.add_axes(interactive=True, line_width=2)

    # Initial Mesh Display
    actor = plotter.add_mesh(
        grid,
        scalars="Displacement",
        cmap="turbo",
        show_edges=True,
        edge_color="#333333",
        line_width=1.0,
        scalar_bar_args={
            "title": "Displacement Magnitude (mm)",
            "color": "white",
            "title_font_size": 12,
            "label_font_size": 10,
        }
    )

    # Global State
    state = {
        "mode": "displacement",
        "perspective": True
    }

    def set_front_view():
        plotter.view_xy()
        plotter.camera.up = (0, 1, 0)
        plotter.reset_camera()
        print("[View] Front View (XY plane - 180° U-Shape folding profile)")

    def set_side_view():
        plotter.view_yz()
        plotter.reset_camera()
        print("[View] Side View (YZ cross section)")

    def set_top_view():
        plotter.view_xz()
        plotter.camera.up = (0, 0, 1)
        plotter.reset_camera()
        print("[View] Top View (XZ plane)")

    def set_iso_view():
        plotter.view_isometric()
        plotter.reset_camera()
        print("[View] Isometric 3D View")

    def toggle_perspective():
        state["perspective"] = not state["perspective"]
        if state["perspective"]:
            plotter.enable_parallel_projection()
            print("[Camera] Orthographic Projection ON")
        else:
            plotter.disable_parallel_projection()
            print("[Camera] Perspective Projection ON")

    def toggle_color():
        if state["mode"] == "displacement":
            state["mode"] = "material"
            actor.mapper.set_scalars(grid.cell_data["Material"], "Material")
            actor.mapper.lookup_table.cmap = "viridis"
            print("[Color] Material Layers View (PET=1, PSA=2, Rigid Plate=3)")
        else:
            state["mode"] = "displacement"
            actor.mapper.set_scalars(grid.point_data["Displacement"], "Displacement")
            actor.mapper.lookup_table.cmap = "turbo"
            print("[Color] Displacement Contour View")

    def save_screenshot():
        out_dir = os.path.join(REPO_ROOT, "dev_log", "figures")
        os.makedirs(out_dir, exist_ok=True)
        img_path = os.path.join(out_dir, "ex13_3d_folding_pyvista_screenshot.png")
        plotter.screenshot(img_path)
        print(f"[Screenshot] Saved high-resolution screenshot to: {img_path}")

    # Keybindings
    plotter.add_key_event("1", set_front_view)
    plotter.add_key_event("2", set_side_view)
    plotter.add_key_event("3", set_top_view)
    plotter.add_key_event("4", set_iso_view)
    plotter.add_key_event("p", toggle_perspective)
    plotter.add_key_event("P", toggle_perspective)
    plotter.add_key_event("c", toggle_color)
    plotter.add_key_event("C", toggle_color)
    plotter.add_key_event("s", save_screenshot)
    plotter.add_key_event("S", save_screenshot)

    # Initial camera view
    set_front_view()
    print("=" * 80)
    print(" 3D PyVista Viewer Launched!")
    print(" Controls: '1'=Front(XY), '2'=Side(YZ), '3'=Top(XZ), '4'=3D Iso, 'C'=Contour/Mat, 'S'=Save PNG")
    print("=" * 80)

    if "--screenshot" in sys.argv:
        out_dir = os.path.join(REPO_ROOT, "dev_log", "figures")
        os.makedirs(out_dir, exist_ok=True)
        # 1. Front View (U-shape)
        set_front_view()
        plotter.screenshot(os.path.join(out_dir, "ex13_3d_folding_pyvista_front.png"))
        # 2. Isometric 3D View
        set_iso_view()
        plotter.screenshot(os.path.join(out_dir, "ex13_3d_folding_pyvista_iso.png"))
        # 3. Material Layers View
        toggle_color()
        set_iso_view()
        plotter.screenshot(os.path.join(out_dir, "ex13_3d_folding_pyvista_material.png"))
        print("[Screenshot] Successfully generated front, iso, and material 3D views!")
        plotter.close()
    else:
        plotter.show()


if __name__ == "__main__":
    visualize_3d_fold()
