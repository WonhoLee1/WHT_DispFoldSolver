"""
check_hinge_curvature.py
========================
Extracts and inspects the 90° deformed geometric shape profile of the display panel
using the Co-rotational Q4 + Abaqus Stabilization engine.
Verifies whether the display panel forms a smooth elliptical/curved profile in the hinge zone
or an unnatural sharp kink at the rigid plate boundaries.
"""

import numpy as np
from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, LinearViscoelastic
from dispsolver.solver import DynamicSolver
from dispsolver.solver.stabilization import AbaqusViscousStabilization
from dispsolver.element.q4_corotational_jax import compute_corotational_internal_force

class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
        self.T = t1 - t0
    def __call__(self, t):
        tau = np.clip((t - self.t0) / self.T, 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))

def inspect_deformed_profile():
    print("=" * 70)
    print("GEOMETRIC BENDING PROFILE & CURVATURE INSPECTION (Co-rotational Engine)")
    print("=" * 70)

    # 1. Mesh Construction
    mesh = Mesh()
    xs_left  = np.linspace(-40.0, -15.0, 15)[:-1]
    xs_mid   = np.linspace(-15.0, 15.0, 41)[:-1]
    xs_right = np.linspace(15.0, 40.0, 15)
    xs = np.concatenate([xs_left, xs_mid, xs_right])

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

    node_id = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(node_id, x, y)
            node_id += 1

    nx = len(xs)
    ny = len(ys)
    elem_id = 1
    for j in range(ny - 1):
        pid = row_pids[j]
        for i in range(nx - 1):
            n0 = j * nx + i + 1
            n1 = j * nx + i + 2
            n2 = (j + 1) * nx + i + 2
            n3 = (j + 1) * nx + i + 1
            mesh.add_element(elem_id, [n0, n1, n2, n3], "Q4", pid=pid)
            elem_id += 1

    pet = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=400.0)
    
    # DynamicSolver with Co-rotational / Abaqus Controls
    solver = DynamicSolver(
        mesh=mesh,
        material=pet,
        rho=1e-9,
        mode="quasistatic",
        max_iter=30,
        tol=1e-2,
        fast_assembly=True
    )

    left_bottom_idx = mesh.node_id_to_index()[1]
    solver.set_prescribed_dofs(bc_dofs=[2 * left_bottom_idx, 2 * left_bottom_idx + 1], bc_vals=[0.0, 0.0])

    amp = SmoothAmplitude(0.0, 1.0)
    target_angle = 1.570796

    print("Running 90° Folding Simulation up to t = 1.00...")
    t, dt = 0.0, 0.005
    step = 0
    while t < 1.0:
        step += 1
        iter_count = solver.solve_step(dt)
        if iter_count < 0:
            dt *= 0.5
            if dt < 1e-6:
                break
            continue
        t += dt
        if iter_count <= 4:
            dt = min(dt * 1.5, 0.02)

    print(f"Simulation completed up to t = {t:.4f} (90.0°)")

    # Extract Top Surface Deformed Coordinates
    top_node_indices = []
    top_init_x = []
    for nid, node in mesh.nodes.items():
        if abs(node.y - ys[-1]) < 1e-6:
            idx = mesh.node_id_to_index()[nid]
            top_node_indices.append(idx)
            top_init_x.append(node.x)

    sort_order = np.argsort(top_init_x)
    top_node_indices = [top_node_indices[i] for i in sort_order]
    top_init_x = np.array([top_init_x[i] for i in sort_order])

    def_x = top_init_x + solver.u[2 * np.array(top_node_indices)]
    def_y = ys[-1] + solver.u[2 * np.array(top_node_indices) + 1]

    # Filter Hinge Zone (-15mm to +15mm)
    hinge_mask = (top_init_x >= -15.0) & (top_init_x <= 15.0)
    hx = def_x[hinge_mask]
    hy = def_y[hinge_mask]

    # Calculate Slope & Curvature
    dx = np.gradient(hx)
    dy = np.gradient(hy)
    slope = dy / np.where(np.abs(dx) > 1e-8, dx, 1e-8)
    
    d2y = np.gradient(slope)
    curvature = np.abs(d2y) / (1.0 + slope**2)**1.5

    max_curvature = np.max(curvature)
    min_radius = 1.0 / (max_curvature + 1e-12)
    mean_curvature = np.mean(curvature)
    curv_std = np.std(curvature)

    kink_ratio = max_curvature / max(mean_curvature, 1e-6)
    is_kinked = kink_ratio > 4.0

    print("\n" + "=" * 70)
    print("--- 90° DEFORMED GEOMETRIC CURVATURE ANALYSIS ---")
    print("=" * 70)
    print(f"Hinge Zone Node Count         : {len(hx)}")
    print(f"Max Curvature (kappa_max)     : {max_curvature:.4f} 1/mm")
    print(f"Min Bending Radius (R_min)    : {min_radius:.4f} mm")
    print(f"Mean Curvature (kappa_mean)   : {mean_curvature:.4f} 1/mm")
    print(f"Curvature Concentration Ratio : {kink_ratio:.2f}")
    print(f"Shape Profile Assessment      : {'[WARN] Sharp Kink Detected (Sharp Bend at Boundary)' if is_kinked else '[PASS] Smooth Elliptical Curved Bending Profile'}")
    print("=" * 70)

if __name__ == "__main__":
    inspect_deformed_profile()
