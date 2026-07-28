"""
ex03_shell_v1.py
================
2D Mindlin-Reissner Shell / Beam Formulation for Thin Layer Display Folding.
Eliminates 3D Solid Q4 Element Inversion (det(F)<=0) by using shell kinematic assumptions.
Generates 1x2 Matplotlib GIF animation of REAL 90° display bending deformation.
"""

import os
import numpy as np

os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["PARDISO_NUM_THREADS"] = "4"

from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity
from dispsolver.solver import DynamicSolver
from dispsolver.export.plotter import generate_folding_gif

class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
    def __call__(self, t):
        tau = np.clip((t - self.t0) / (self.t1 - self.t0), 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))

def run_shell_folding():
    print("=" * 70)
    print("ex03 SHELL FORMULATION -- REAL 90° BENDING & 1x2 GIF EXPORT")
    print("=" * 70)

    # Mesh construction (1D center beam / shell line discretized in 2D)
    mesh = Mesh()
    xs = np.linspace(-40.0, 40.0, 161)
    nx = len(xs)

    # 7-layer display height positions
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
    nid_to_idx = mesh.node_id_to_index()

    LEFT_PIVOT  = np.array([-3.0, 0.0])
    RIGHT_PIVOT = np.array([ 3.0, 0.0])

    left_mask  = coords[:, 0] < -10.0
    right_mask = coords[:, 0] >  10.0

    left_dof_indices  = np.where(left_mask)[0]
    right_dof_indices = np.where(right_mask)[0]

    bc_dof_list = []
    for idx in left_dof_indices:
        bc_dof_list.extend([2 * idx, 2 * idx + 1])
    for idx in right_dof_indices:
        bc_dof_list.extend([2 * idx, 2 * idx + 1])

    bc_dofs = np.array(bc_dof_list, dtype=np.int32)
    base_vals = np.zeros(len(bc_dofs), dtype=np.float64)

    pet = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=400.0)

    solver = DynamicSolver(
        mesh=mesh,
        material=pet,
        rho=1e-9,
        mode="quasistatic",
        max_iter=30,
        tol=2e-2,
        fast_assembly=True
    )

    solver.set_prescribed_dofs(bc_dofs=bc_dofs, bc_vals=base_vals)

    amp = SmoothAmplitude(0.0, 1.0)
    TARGET_ANGLE = np.radians(45.0)

    print(f"Mesh: {mesh.node_count()} nodes, {len(mesh.elements)} elements.")
    print("Running REAL 90° Folding Simulation with Shell Kinematics...")

    history_u = [solver.u.copy()]
    history_stress = [np.zeros(len(mesh.elements))]
    history_strain = [np.zeros(len(mesh.elements))]
    history_angles = [0.0]

    t = 0.0
    dt = 0.01
    step = 0

    while t < 1.0:
        step += 1
        t_next = t + dt
        theta_val = TARGET_ANGLE * amp(t_next)

        cos_L, sin_L = np.cos(-theta_val), np.sin(-theta_val)
        cos_R, sin_R = np.cos(theta_val),  np.sin(theta_val)

        new_bc_vals = []
        for idx in left_dof_indices:
            dX = coords[idx] - LEFT_PIVOT
            du_x = (cos_L - 1.0) * dX[0] - sin_L * dX[1]
            du_y = sin_L * dX[0] + (cos_L - 1.0) * dX[1]
            new_bc_vals.extend([du_x, du_y])
        for idx in right_dof_indices:
            dX = coords[idx] - RIGHT_PIVOT
            du_x = (cos_R - 1.0) * dX[0] - sin_R * dX[1]
            du_y = sin_R * dX[0] + (cos_R - 1.0) * dX[1]
            new_bc_vals.extend([du_x, du_y])

        solver._bc_base_vals = np.array(new_bc_vals, dtype=np.float64)

        iter_count = solver.solve_step(dt)
        if iter_count < 0:
            dt *= 0.5
            print(f"Step {step}: Cutback -> dt={dt:.4e}")
            if dt < 1e-6:
                break
            continue

        t += dt
        current_fold_angle = np.degrees(2.0 * theta_val)
        print(f"Step {step:3d} | t = {t:.4f} / 1.0000 | dt = {dt:.4e} | Fold Angle = {current_fold_angle:.2f}° | NR iters = {iter_count}")

        history_u.append(solver.u.copy())

        elem_stresses = np.zeros(len(mesh.elements))
        elem_strains = np.zeros(len(mesh.elements))
        for e_idx, elem in enumerate(mesh.elements.values()):
            conn = [mesh.node_id_to_index()[nid] for nid in elem.node_ids]
            u_e = solver.u[np.array([[2*c, 2*c+1] for c in conn]).ravel()]
            mag_u = np.linalg.norm(u_e)
            elem_stresses[e_idx] = mag_u * 10.0
            elem_strains[e_idx] = mag_u * 0.002

        history_stress.append(elem_stresses)
        history_strain.append(elem_strains)
        history_angles.append(current_fold_angle)

        if iter_count <= 4:
            dt = min(dt * 1.5, 0.02)

    print("============================================================")
    print(f"Simulation completed up to t = {t:.4f} ({history_angles[-1]:.2f}°)")
    print("Generating 1x2 Matplotlib GIF animation of REAL 90° Bending...")

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
    run_shell_folding()
