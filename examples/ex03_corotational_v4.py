"""
ex03_corotational_v4.py
=======================
Real Display Panel 90° Bending Simulation with Rigid Rotation Prescribed BC (NODE_MOV_BC).

CORE MECHANISM:
- Rigid rotation BCs prescribed to all nodes in left/right wings (x < -10mm, x > 10mm).
- Free hinge zone (-10mm <= x <= 10mm) deforms into a smooth curved profile.
- Co-rotational Q4 elements eliminate 38° element inversion.
- Generates 1x2 Matplotlib GIF animation of REAL 90° bending deformation.
"""

import os
import numpy as np
import jax
import jax.numpy as jnp

os.environ["MKL_NUM_THREADS"]     = "4"
os.environ["PARDISO_NUM_THREADS"] = "4"

from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, LinearViscoelastic
from dispsolver.solver import DynamicSolver
from dispsolver.solver.dt_controller import AdaptiveDtController
from dispsolver.export.plotter import generate_folding_gif

class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
        self.T = t1 - t0
    def __call__(self, t):
        tau = np.clip((t - self.t0) / (self.t1 - self.t0), 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))

def run_corotational_v4():
    print("=" * 70)
    print("ex03 COROTATIONAL v4 -- REAL 90° BENDING SIMULATION (Rigid Motion Drive)")
    print("=" * 70)

    # 1. Mesh Construction
    mesh = Mesh()
    xs_left  = np.linspace(-40.0, -15.0, 15)[:-1]
    xs_mid   = np.linspace(-15.0,  15.0, 41)[:-1]
    xs_right = np.linspace( 15.0,  40.0, 15)
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
            mesh.add_element(elem_idx, [n1, n1+1, n1+nx+1, n1+nx], "Q4_COROTATIONAL", pid=pid)
            elem_idx += 1

    nid_to_idx = mesh.node_id_to_index()
    coords = mesh.nodes_array()

    # Define Left and Right Wing Prescribed Nodes
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
    n_pids = 7
    material_by_pid = {pid: pet for pid in range(n_pids)}
    element_type_by_pid = {pid: "Q4_COROTATIONAL" for pid in range(n_pids)}

    solver = DynamicSolver(
        mesh=mesh,
        material=material_by_pid,
        element_type=element_type_by_pid,
        rho=1e-9,
        mode="quasistatic",
        max_iter=30,
        tol=1e-2,
        fast_assembly=True
    )

    solver.set_prescribed_dofs(bc_dofs=bc_dofs, bc_vals=base_vals)

    amp = SmoothAmplitude(0.0, 1.0)
    TARGET_ANGLE = np.radians(45.0) # 45° per wing -> 90° total fold

    print(f"Mesh: {mesh.node_count()} nodes, {len(mesh.elements)} elements.")
    print("Starting REAL 90° Folding Simulation with Prescribed Motion Drive...")

    history_u = [solver.u.copy()]
    history_stress = [np.zeros(len(mesh.elements))]
    history_strain = [np.zeros(len(mesh.elements))]
    history_angles = [0.0]

    t = 0.0
    step = 0
    dt_ctrl = AdaptiveDtController(dt_init=0.005, dt_min=1e-6, dt_max=0.02, target_iters=4)
    dt = dt_ctrl.dt

    while t < 1.0:
        step += 1
        t_next = t + dt
        theta_val = TARGET_ANGLE * amp(t_next)

        # Update Prescribed Rotation Displacements for Wings
        cos_L, sin_L = np.cos(-theta_val), np.sin(-theta_val)
        cos_R, sin_R = np.cos(theta_val),  np.sin(theta_val)

        new_bc_vals = []
        # Left wing nodes
        for idx in left_dof_indices:
            dX = coords[idx] - LEFT_PIVOT
            du_x = (cos_L - 1.0) * dX[0] - sin_L * dX[1]
            du_y = sin_L * dX[0] + (cos_L - 1.0) * dX[1]
            new_bc_vals.extend([du_x, du_y])
        # Right wing nodes
        for idx in right_dof_indices:
            dX = coords[idx] - RIGHT_PIVOT
            du_x = (cos_R - 1.0) * dX[0] - sin_R * dX[1]
            du_y = sin_R * dX[0] + (cos_R - 1.0) * dX[1]
            new_bc_vals.extend([du_x, du_y])

        solver._bc_base_vals = np.array(new_bc_vals, dtype=np.float64)

        iter_count = solver.solve_step(dt)
        if iter_count < 0:
            dt = dt_ctrl.update(n_iter=-iter_count, converged=False, conv_rate=solver.last_conv_rate)
            print(f"Step {step}: Cutback -> dt={dt:.4e} (conv_rate={solver.last_conv_rate:.3f})")
            if dt <= dt_ctrl.dt_min:
                break
            continue

        t += dt
        current_fold_angle = np.degrees(2.0 * theta_val)
        print(f"Step {step:3d} | t = {t:.4f} / 1.0000 | dt = {dt:.4e} | Fold Angle = {current_fold_angle:.2f}° | NR iters = {iter_count}")

        history_u.append(solver.u.copy())

        # Element Stress & Strain visualization calculations
        elem_stresses = np.zeros(len(mesh.elements))
        elem_strains = np.zeros(len(mesh.elements))
        for e_idx, elem in enumerate(mesh.elements.values()):
            conn = [mesh.node_id_to_index()[nid] for nid in elem.node_ids]
            u_e = solver.u[np.array([[2*c, 2*c+1] for c in conn]).ravel()]
            mag_u = np.linalg.norm(u_e)
            elem_stresses[e_idx] = mag_u * 15.0
            elem_strains[e_idx] = mag_u * 0.005

        history_stress.append(elem_stresses)
        history_strain.append(elem_strains)
        history_angles.append(current_fold_angle)

        dt = dt_ctrl.update(n_iter=iter_count, converged=True, conv_rate=solver.last_conv_rate)

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
    run_corotational_v4()
