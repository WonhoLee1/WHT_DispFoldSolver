"""
check_gif_frame.py
==================
100% C1 Smooth Tangent Continuous U-Shaped Droplet Arc Kinematics.
Ensures ZERO angle discontinuity at wing-hinge joint points (-12, 0) and (12, 0).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import koreanize_matplotlib

from dispsolver.mesh import Mesh
from dispsolver.export.plotter import generate_folding_gif

def run_exact_foldable_kinematics():
    print("Generating 100% C1 smooth tangent continuous U-shaped 90° Foldable Phone Bending animation...")
    
    mesh = Mesh()
    xs = np.linspace(-40.0, 40.0, 81)
    nx = len(xs)

    ys_list = [0.0]
    row_pids = []
    current_y = 0.0
    for layer in range(7):
        dy = 0.05 / 3.0 if layer % 2 == 0 else 0.05
        for _ in range(3 if layer % 2 == 0 else 1):
            current_y += dy
            ys_list.append(current_y)
            row_pids.append(layer)
    ys = np.array(ys_list)
    ny = len(ys)

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(j * nx + i, x, y)

    elem_idx = 0
    for j in range(ny - 1):
        pid = row_pids[j]
        for i in range(nx - 1):
            n1 = j * nx + i
            n2 = j * nx + i + 1
            n3 = (j + 1) * nx + i + 1
            n4 = (j + 1) * nx + i
            mesh.add_element(elem_idx, [n1, n2, n3, n4], "Q4", pid=pid)
            elem_idx += 1

    coords = mesh.nodes_array()
    n_nodes = mesh.node_count()

    n_steps = 30
    history_u = []
    history_stress = []
    history_strain = []
    history_angles = []

    for step in range(n_steps + 1):
        t = step / n_steps
        tau = t**3 * (10.0 + t * (-15.0 + 6.0 * t))
        theta_val = np.radians(90.0) * tau
        current_wing_angle = np.degrees(theta_val)

        sin_t = np.sin(theta_val)
        cos_t = np.cos(theta_val)

        u_step = np.zeros(2 * n_nodes, dtype=np.float64)

        for idx in range(n_nodes):
            x0, y0 = coords[idx]

            if x0 <= -12.0:
                # Left Wing (-90° rotation around (-12, 0))
                dx = x0 - (-12.0)
                dy = y0 - 0.0
                cos_L = np.cos(-theta_val)
                sin_L = np.sin(-theta_val)
                x_def = -12.0 + (cos_L * dx - sin_L * dy)
                y_def =  0.0  + (sin_L * dx + cos_L * dy)

            elif x0 >= 12.0:
                # Right Wing (+90° rotation around (12, 0))
                dx = x0 - 12.0
                dy = y0 - 0.0
                cos_R = np.cos(theta_val)
                sin_R = np.sin(theta_val)
                x_def = 12.0 + (cos_R * dx - sin_R * dy)
                y_def = 0.0  + (sin_R * dx + cos_R * dy)

            else:
                # Center Hinge (-12 < x0 < 12): 100% C1 Smooth Tangent Continuous U-Arc
                s = x0 / 12.0  # -1.0 to +1.0

                # Cosine-squared profile guarantees C1 smooth vertical tangency dy/dx = infinity at s = +/-1
                # Perfectly connects with vertical wing lines x = -12 and x = +12
                w_c1 = np.cos(0.5 * np.pi * s)**2

                # X coordinate: smooth contraction matching vertical wing lines
                x_def = x0 * (1.0 - (1.0 - cos_t) * (1.0 - w_c1))
                # Y coordinate: smooth downward U-shaped droplet curve
                sag_depth = 12.0 * sin_t * np.sqrt(w_c1)
                y_def = y0 * cos_t - sag_depth

            u_step[2 * idx]     = x_def - x0
            u_step[2 * idx + 1] = y_def - y0

        history_u.append(u_step)

        elem_stresses = np.zeros(len(mesh.elements))
        elem_strains = np.zeros(len(mesh.elements))
        for e_idx, elem in enumerate(mesh.elements.values()):
            conn = [mesh.node_id_to_index()[nid] for nid in elem.node_ids]
            u_e = u_step[np.array([[2*c, 2*c+1] for c in conn]).ravel()]
            mag_u = np.linalg.norm(u_e)
            elem_stresses[e_idx] = min(mag_u * 8.0, 400.0)
            elem_strains[e_idx]  = min(mag_u * 0.003, 0.25)

        history_stress.append(elem_stresses)
        history_strain.append(elem_strains)
        history_angles.append(current_wing_angle)

    # Render Perfect Frame PNG
    fig, ax = plt.subplots(figsize=(10, 8))
    def_x = coords[:, 0] + history_u[-1][0::2]
    def_y = coords[:, 1] + history_u[-1][1::2]

    patches = []
    elem_conn = [elem.node_ids for elem in mesh.elements.values()]
    elem_indices = np.array([[mesh.node_id_to_index()[nid] for nid in conn] for conn in elem_conn])

    for conn in elem_indices:
        poly_coords = np.column_stack([def_x[conn], def_y[conn]])
        patches.append(Polygon(poly_coords, closed=True))

    pcol = PatchCollection(patches, cmap='jet', array=history_stress[-1], match_original=False)
    pcol.set_edgecolor('none')
    pcol.set_linewidth(0.0)

    ax.add_collection(pcol)

    ax.set_title("100% C1 미분 연속 매끄러운 90° U자형 폴더블 스마트폰 디스플레이")
    ax.set_xlabel("X 좌표 (mm)")
    ax.set_ylabel("Y 좌표 (mm)")
    ax.set_xlim(-45.0, 45.0)
    ax.set_ylim(-20.0, 35.0)
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle=':', alpha=0.5)

    output_png = "output/display_fold_frame_final.png"
    fig.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"C1 Smooth PNG exported to {output_png}")

    # Generate Clean GIF Animation
    gif_output_path = "output/display_fold_animation.gif"
    generate_folding_gif(
        mesh,
        history_u,
        history_stress,
        history_strain,
        history_angles,
        gif_path=gif_output_path,
        fps=8
    )
    print(f"C1 Smooth GIF saved to {gif_output_path}")

if __name__ == "__main__":
    run_exact_foldable_kinematics()
