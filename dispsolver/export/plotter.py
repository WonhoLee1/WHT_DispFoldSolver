"""
plotter.py
==========
2D Matplotlib visualization and GIF animation generator for display panel folding.
Uses PatchCollection with linewidth 0.0 and edgecolor none for 100% clean solid contour rendering.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation, PillowWriter
import koreanize_matplotlib

plt.rcParams.update({'font.size': 9})

def generate_folding_gif(
    mesh,
    history_u: list[np.ndarray],
    history_stress: list[np.ndarray],
    history_strain: list[np.ndarray],
    history_angles: list[float],
    gif_path: str = "output/display_fold_animation.gif",
    fps: int = 8
):
    """Generate GIF animation of 90° folding with clean solid contour PatchCollection rendering."""
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    nodes = mesh.nodes_array()
    elem_conn = [elem.node_ids for elem in mesh.elements.values()]
    elem_indices = np.array([[mesh.node_id_to_index()[nid] for nid in conn] for conn in elem_conn])

    all_def_x = [nodes[:, 0] + u[0::2] for u in history_u]
    all_def_y = [nodes[:, 1] + u[1::2] for u in history_u]

    global_xmin = min(np.min(x) for x in all_def_x) - 3.0
    global_xmax = max(np.max(x) for x in all_def_x) + 3.0
    global_ymin = min(np.min(y) for y in all_def_y) - 3.0
    global_ymax = max(np.max(y) for y in all_def_y) + 3.0

    stress_min = min(np.min(s) for s in history_stress)
    stress_max = max(np.max(s) for s in history_stress)
    strain_min = min(np.min(e) for e in history_strain)
    strain_max = max(np.max(e) for e in history_strain)

    def update(frame):
        u = history_u[frame]
        stress = history_stress[frame]
        strain = history_strain[frame]
        angle = history_angles[frame]

        ax1.clear()
        ax2.clear()

        def_x = nodes[:, 0] + u[0::2]
        def_y = nodes[:, 1] + u[1::2]

        patches1 = []
        patches2 = []
        for conn in elem_indices:
            poly_coords = np.column_stack([def_x[conn], def_y[conn]])
            patches1.append(Polygon(poly_coords, closed=True))
            patches2.append(Polygon(poly_coords, closed=True))

        pcol1 = PatchCollection(patches1, cmap='jet', array=stress, match_original=False)
        pcol1.set_clim(stress_min, stress_max)
        pcol1.set_edgecolor('none')
        pcol1.set_linewidth(0.0)

        pcol2 = PatchCollection(patches2, cmap='viridis', array=strain, match_original=False)
        pcol2.set_clim(strain_min, strain_max)
        pcol2.set_edgecolor('none')
        pcol2.set_linewidth(0.0)

        # Left Subplot: Stress
        ax1.add_collection(pcol1)
        ax1.set_title(f"유효 응력 (Mises Stress) - 날개 회전각 {angle:.1f}° (-90°/+90°)")
        ax1.set_xlabel("X 좌표 (mm)")
        ax1.set_ylabel("Y 좌표 (mm)")
        ax1.set_xlim(global_xmin, global_xmax)
        ax1.set_ylim(global_ymin, global_ymax)
        ax1.set_aspect('equal', 'box')
        ax1.grid(True, linestyle=':', alpha=0.5)

        # Right Subplot: Strain
        ax2.add_collection(pcol2)
        ax2.set_title(f"등가 변형률 (Equivalent Strain) - 날개 회전각 {angle:.1f}° (-90°/+90°)")
        ax2.set_xlabel("X 좌표 (mm)")
        ax2.set_ylabel("Y 좌표 (mm)")
        ax2.set_xlim(global_xmin, global_xmax)
        ax2.set_ylim(global_ymin, global_ymax)
        ax2.set_aspect('equal', 'box')
        ax2.grid(True, linestyle=':', alpha=0.5)

        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=len(history_u), interval=1000 // fps)
    writer = PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)
    plt.close(fig)
    print(f"[GIF] Animation saved with clean solid contour PatchCollection: {gif_path}")


def plot_and_save_deformed_shape_png(
    mesh,
    u: np.ndarray,
    save_path: str = "output/ex11_final_folding_shape.png",
    title_prefix: str = "Abaqus 90° Display Folding Final State",
    rbe2_elements: list = None,
    tie_constraints: list = None,
    artifact_dir: str = None,
    dpi: int = 300,
):
    """
    Plots and saves 2D publication-quality PNG of initial vs final deformed shape,
    displacement contours (|u|, Uy), and rigid body / surface tie boundaries.
    """
    plt.rcParams.update({'font.size': 9})
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    nodes = mesh.nodes_array()
    n_nodes = len(nodes)
    elem_conn = [elem.node_ids for elem in mesh.elements.values()]
    nid_to_idx = mesh.node_id_to_index()
    elem_indices = np.array([[nid_to_idx[nid] for nid in conn] for conn in elem_conn])

    # Undeformed & Deformed Coordinates
    x0, y0 = nodes[:, 0], nodes[:, 1]
    ux, uy = u[0::2], u[1::2]
    x_def = x0 + ux
    y_def = y0 + uy
    u_mag = np.sqrt(ux**2 + uy**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1: Total Displacement Magnitude |u| & Tied Pair Overlay ---
    patches_mag = []
    patches_init = []
    for conn in elem_indices:
        poly_def = Polygon(np.column_stack([x_def[conn], y_def[conn]]), closed=True)
        poly_init = Polygon(np.column_stack([x0[conn], y0[conn]]), closed=True)
        patches_mag.append(poly_def)
        patches_init.append(poly_init)

    # Initial shape wireframe
    pcol_init = PatchCollection(patches_init, facecolor='none', edgecolor='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax1.add_collection(pcol_init)

    # Deformed shape filled contour
    pcol_mag = PatchCollection(patches_mag, cmap='turbo', array=u_mag, match_original=False)
    pcol_mag.set_edgecolor('black')
    pcol_mag.set_linewidth(0.2)
    ax1.add_collection(pcol_mag)
    cb1 = fig.colorbar(pcol_mag, ax=ax1, fraction=0.046, pad=0.04)
    cb1.set_label("변위 크기 |u| (mm)")

    # Plot Surface Tie connections if available
    if tie_constraints:
        for tie in tie_constraints:
            slave_ids = getattr(tie, 'slave_node_ids', [])
            master_ids = getattr(tie, 'master_node_ids', [])
            for s_nid in slave_ids:
                if s_nid in nid_to_idx:
                    s_idx = nid_to_idx[s_nid]
                    ax1.plot(x_def[s_idx], y_def[s_idx], 'g.', markersize=3, alpha=0.6)

    # Plot Hinge Reference Points if available
    if rbe2_elements:
        for rbe2 in rbe2_elements:
            m_id = getattr(rbe2, 'master_id', getattr(rbe2, 'master_node_id', None))
            if m_id is not None and m_id in nid_to_idx:
                rp_idx = nid_to_idx[m_id]
                ax1.plot(x_def[rp_idx], y_def[rp_idx], 'r*', markersize=10, label=f'RP N{m_id}')

    all_x = np.concatenate([x0, x_def])
    all_y = np.concatenate([y0, y_def])
    margin = 3.0
    ax1.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
    ax1.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)
    ax1.set_aspect('equal', 'box')
    ax1.set_title(f"{title_prefix}\n최종 변형 형상 및 변위 크기 (|u| max = {np.max(u_mag):.2f} mm)")
    ax1.set_xlabel("X 좌표 (mm)")
    ax1.set_ylabel("Y 좌표 (mm)")
    ax1.grid(True, linestyle=':', alpha=0.5)

    # --- Subplot 2: Vertical Displacement (Uy) Contour ---
    patches_uy = []
    for conn in elem_indices:
        poly_def = Polygon(np.column_stack([x_def[conn], y_def[conn]]), closed=True)
        patches_uy.append(poly_def)

    pcol_uy = PatchCollection(patches_uy, cmap='plasma', array=uy, match_original=False)
    pcol_uy.set_edgecolor('black')
    pcol_uy.set_linewidth(0.2)
    ax2.add_collection(pcol_uy)
    cb2 = fig.colorbar(pcol_uy, ax=ax2, fraction=0.046, pad=0.04)
    cb2.set_label("수직 변위 Uy (mm)")

    ax2.set_xlim(np.min(all_x) - margin, np.max(all_x) + margin)
    ax2.set_ylim(np.min(all_y) - margin, np.max(all_y) + margin)
    ax2.set_aspect('equal', 'box')
    ax2.set_title(f"수직 변위 (Uy) 분포\n(Uy min = {np.min(uy):.2f} mm, max = {np.max(uy):.2f} mm)")
    ax2.set_xlabel("X 좌표 (mm)")
    ax2.set_ylabel("Y 좌표 (mm)")
    ax2.grid(True, linestyle=':', alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[PNG] Deformed shape saved: {save_path}")

    if artifact_dir:
        art_path = os.path.join(artifact_dir, os.path.basename(save_path))
        fig.savefig(art_path, dpi=dpi, bbox_inches='tight')
        print(f"[PNG] Copy saved to artifact: {art_path}")

    plt.close(fig)
    return save_path

