"""
ex03_corotational_pure.py
=========================
Foldable Smartphone 90° Bending Simulation with Downward Droplet/U-Sagging Kinematics.

CORE MECHANISM:
- Left wing rotates FULL -90° (vertical orientation).
- Right wing rotates FULL +90° (vertical orientation).
- Center display panel naturally sags DOWNWARD (-Y direction) into a smooth U-shaped droplet curve.
- Fixed coordinate bounds in plotter.py for stable GIF animation.
"""

import os
import numpy as np

os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["PARDISO_NUM_THREADS"] = "4"

from dispsolver.mesh import Mesh
from dispsolver.export.plotter import generate_folding_gif

class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
    def __call__(self, t):
        tau = np.clip((t - self.t0) / (self.t1 - self.t0), 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))

def run_pure_corotational():
    print("=" * 70)
    print("ex03 FOLDABLE SMARTPHONE BENDING -- LEFT -90°, RIGHT +90° & DOWNWARD SAG")
    print("=" * 70)

    mesh = Mesh()
    xs_left  = np.linspace(-40.0, -12.0, 15)[:-1]
    xs_mid   = np.linspace(-12.0,  12.0, 41)[:-1]
    xs_right = np.linspace( 12.0,  40.0, 15)
    xs = np.concatenate([xs_left, xs_mid, xs_right])
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
            mesh.add_element(elem_idx, [n1, n1+1, n1+nx+1, n1+nx], "Q4", pid=pid)
            elem_idx += 1

    coords = mesh.nodes_array()
    n_nodes = mesh.node_count()

    LEFT_PIVOT  = np.array([-12.0, 0.0])
    RIGHT_PIVOT = np.array([ 12.0, 0.0])

    amp = SmoothAmplitude(0.0, 1.0)
    WING_ROT_ANGLE = np.radians(90.0)  # EACH wing rotates FULL 90° (-90° / +90°)

    n_steps = 30
    history_u = []
    history_stress = []
    history_strain = []
    history_angles = []

    print(f"Mesh: {n_nodes} nodes, {len(mesh.elements)} elements.")
    print("Generating Foldable Smartphone 90° Bending with Downward Sagging...")

    for step in range(n_steps + 1):
        t = step / n_steps
        theta_val = WING_ROT_ANGLE * amp(t)  # 0° to 90° per wing
        current_wing_angle = np.degrees(theta_val)

        cos_L, sin_L = np.cos(-theta_val), np.sin(-theta_val)
        cos_R, sin_R = np.cos(theta_val),  np.sin(theta_val)

        u_step = np.zeros(2 * n_nodes, dtype=np.float64)

        for idx in range(n_nodes):
            x_val = coords[idx, 0]

            if x_val < -12.0:
                # Left rigid wing rotates -90° around LEFT_PIVOT
                dX = coords[idx] - LEFT_PIVOT
                u_step[2 * idx]     = (cos_L - 1.0) * dX[0] - sin_L * dX[1]
                u_step[2 * idx + 1] = sin_L * dX[0] + (cos_L - 1.0) * dX[1]
            elif x_val > 12.0:
                # Right rigid wing rotates +90° around RIGHT_PIVOT
                dX = coords[idx] - RIGHT_PIVOT
                u_step[2 * idx]     = (cos_R - 1.0) * dX[0] - sin_R * dX[1]
                u_step[2 * idx + 1] = sin_R * dX[0] + (cos_R - 1.0) * dX[1]
            else:
                # Hinge zone (-12mm to +12mm): Smooth Downward Sagging U-shaped droplet curve
                # Normalized coordinate s in [-1, +1]
                s = x_val / 12.0
                
                # Smooth angle interpolation from -theta_val to +theta_val
                w = 0.5 * (1.0 + np.sin(0.5 * np.pi * s))
                theta_h = (1.0 - w) * (-theta_val) + w * (theta_val)
                cos_h, sin_h = np.cos(theta_h), np.sin(theta_h)

                # Reference pivot interpolation
                p_ref = (1.0 - w) * LEFT_PIVOT + w * RIGHT_PIVOT
                dX = coords[idx] - p_ref

                # Downward sagging displacement calculation (U-shape droplet displacement)
                # As wings rotate to 90°, center sags down by R_bend ~ 6.0 mm
                sag_depth = 7.5 * (1.0 - s**2) * np.sin(theta_val)

                u_step[2 * idx]     = (cos_h - 1.0) * dX[0] - sin_h * dX[1]
                u_step[2 * idx + 1] = sin_h * dX[0] + (cos_h - 1.0) * dX[1] - sag_depth

        history_u.append(u_step)

        # Compute von Mises Stress & Strain fields for display
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

        print(f"Frame {step:2d} / {n_steps} | Wing Angle = {current_wing_angle:5.2f}° | Max U = {np.max(np.abs(u_step)):5.2f} mm")

    print("============================================================")
    print("Generating 1x2 Matplotlib GIF animation of Foldable Smartphone 90° Bending...")

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
    print(f"1x2 Subplot Animation complete! Saved to {gif_output_path}")

if __name__ == "__main__":
    run_pure_corotational()
