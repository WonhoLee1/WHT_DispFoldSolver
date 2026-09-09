"""
ex11_rigid_plate_display_fold.py
================================
90-Degree Display Folding Simulation using Rigid Body Plates + Surface Tie.

Architecture
------------
1. Display Subsystem:
   - Deformable multi-layer beam (x in [-40, 40] mm, total thickness = 0.5 mm)
   - Discretized into Q4_EAS elements.
   - NO direct RBE2 constraint applied to display nodes.

2. Rigid Plate Subsystem:
   - Left Plate:  x in [-40, -10] mm, y in [-0.5, 0.0] mm, RP at (-3.0, 0.0) mm
   - Right Plate: x in [10, 40] mm,   y in [-0.5, 0.0] mm, RP at (+3.0, 0.0) mm
   - Created via `create_folding_plate_parts()` and `RigidBodyPart`.

3. Surface Tie Coupling:
   - Penalty-based `SurfaceTieConstraint` connects the bottom surface of the display
     to the top surface of the rigid plates.
   - Allows Poisson expansion/contraction and interlaminar slip without boundary stress concentrations.

4. Kinematic Drive:
   - Master RP rotations prescribed: theta_L(t) = -90° * amp(t), theta_R(t) = +90° * amp(t).
"""

import time
import numpy as np

from dispsolver.mesh import Mesh
from dispsolver.mesh.plate_builder import create_folding_plate_parts
from dispsolver.part.rigid_body import RigidBodyPart
from dispsolver.constraint.surface_tie import SurfaceTieConstraint
from dispsolver.element.rbe2 import RBE2HingeElement
from dispsolver.material.neohookean import NeoHookean
from dispsolver.solver.dynamic import DynamicSolver
from dispsolver.solver.dt_controller import AdaptiveDtController
from dispsolver.export.plotter import plot_and_save_deformed_shape_png


def run_rigid_plate_display_fold():
    print("=" * 100)
    print(" EX11: 90-Degree Display Folding with Rigid Plate + Surface Tie System")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. Build Display Mesh (Deformable)
    # ------------------------------------------------------------------
    # Length 80 mm [-40, 40], height 0.5 mm [0.0, 0.5]
    nx_disp = 80
    ny_disp = 4
    x_coords = np.linspace(-40.0, 40.0, nx_disp + 1)
    y_coords = np.linspace(0.0, 0.5, ny_disp + 1)

    mesh = Mesh()
    disp_nodes = {}
    nid = 1

    grid_nids = np.zeros((ny_disp + 1, nx_disp + 1), dtype=int)
    bottom_surface_nids = []

    for j, y_val in enumerate(y_coords):
        for i, x_val in enumerate(x_coords):
            grid_nids[j, i] = nid
            mesh.add_node(nid, x_val, y_val)
            disp_nodes[nid] = (x_val, y_val)
            if j == 0:
                bottom_surface_nids.append(nid)
            nid += 1

    eid = 1
    for j in range(ny_disp):
        for i in range(nx_disp):
            n1 = grid_nids[j, i]
            n2 = grid_nids[j, i + 1]
            n3 = grid_nids[j + 1, i + 1]
            n4 = grid_nids[j + 1, i]
            mesh.add_element(eid, [n1, n2, n3, n4], "Q4", pid=1)
            eid += 1

    # ------------------------------------------------------------------
    # 2. Build Rigid Folding Plates
    # ------------------------------------------------------------------
    plates = create_folding_plate_parts(
        left_x_range=(-40.0, -10.0),
        right_x_range=(10.0, 40.0),
        y_range=(-0.5, 0.0),
        left_pivot=(-3.0, 0.0),
        right_pivot=(3.0, 0.0),
        nx=30,
        ny=2,
        base_node_id=100000,
        base_elem_id=100000,
    )

    left_plate = plates["left"]
    right_plate = plates["right"]

    # Merge plate nodes/elements into main mesh
    for p_name, p_data in plates.items():
        for nid_p, coord in p_data["nodes_dict"].items():
            mesh.add_node(nid_p, coord[0], coord[1])
        for eid_p, conn in p_data["elements_dict"].items():
            mesh.add_element(eid_p, conn, "Q4", pid=2)

    # ------------------------------------------------------------------
    # 3. Create Constraints (RBE2 Condensation for Plates + Surface Tie)
    # ------------------------------------------------------------------
    nid_to_idx = mesh.node_id_to_index()
    coords = mesh.nodes_array()

    # RBE2 Rigid Body Constraints for Left/Right plates (mapped to 0-based DOF indices)
    left_master_idx = nid_to_idx[left_plate["master_rp_id"]]
    left_slave_indices = [nid_to_idx[sid] for sid in left_plate["slave_nids"]]
    right_master_idx = nid_to_idx[right_plate["master_rp_id"]]
    right_slave_indices = [nid_to_idx[sid] for sid in right_plate["slave_nids"]]

    rbe2_left = RBE2HingeElement(left_master_idx, left_slave_indices, coords)
    rbe2_right = RBE2HingeElement(right_master_idx, right_slave_indices, coords)

    # Store master node ID for reference mapping in dynamic.py
    rbe2_left.master_id = left_plate["master_rp_id"]
    rbe2_left.slave_ids = left_plate["slave_nids"]
    rbe2_right.master_id = right_plate["master_rp_id"]
    rbe2_right.slave_ids = right_plate["slave_nids"]

    rbe2_left.set_prescribed_theta(0.0)
    rbe2_right.set_prescribed_theta(0.0)

    # Surface Tie Constraints connecting Display Bottom -> Plate Tops
    left_display_bottom = [nid for nid in bottom_surface_nids if mesh.nodes[nid].x <= -10.0]
    right_display_bottom = [nid for nid in bottom_surface_nids if mesh.nodes[nid].x >= 10.0]

    tie_left = SurfaceTieConstraint(
        slave_node_ids=left_display_bottom,
        master_node_ids=left_plate["top_surface_nids"],
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=1e5,
        name="TIE_LEFT",
    )

    tie_right = SurfaceTieConstraint(
        slave_node_ids=right_display_bottom,
        master_node_ids=right_plate["top_surface_nids"],
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=1e5,
        name="TIE_RIGHT",
    )

    # ------------------------------------------------------------------
    # 4. Initialize Solver
    # ------------------------------------------------------------------
    mat_display = NeoHookean()
    mat_plate = NeoHookean()

    materials = {1: mat_display, 2: mat_plate}
    mat_params = {
        1: {"E": 50.0, "nu": 0.45},    # Display elastic material
        2: {"E": 20000.0, "nu": 0.3},  # Rigid plate material
    }

    solver = DynamicSolver(
        mesh=mesh,
        material=materials,
        material_params=mat_params,
        rho=1e-9,
        mode="quasistatic",
        rtol=1e-4,
        atol=1e-6,
        max_iter=25,
        rbe2_elements=[rbe2_left, rbe2_right],
        penalty_constraints=[tie_left, tie_right],
        ul_mode=True,
        verbose=True,
    )

    # Strictly fix translation of both hinge reference points (UX=0, UY=0)
    left_rp_idx = nid_to_idx[left_plate["master_rp_id"]]
    right_rp_idx = nid_to_idx[right_plate["master_rp_id"]]
    rp_bc_dofs = np.array([left_rp_idx * 2, left_rp_idx * 2 + 1, right_rp_idx * 2, right_rp_idx * 2 + 1], dtype=np.int32)
    solver.set_prescribed_dofs(rp_bc_dofs, bc_vals=np.zeros(len(rp_bc_dofs)))

    # ------------------------------------------------------------------
    # 5. Time Stepping Loop with Adaptive Dt Controller
    # ------------------------------------------------------------------
    T_TOTAL = 1.0
    THETA_MAX = np.radians(90.0)

    # Smooth time stepping for 90° folding (dt_max=0.01s ensures <= 0.9° rotation per step)
    dt_ctrl = AdaptiveDtController(dt_init=0.005, dt_min=1e-5, dt_max=0.01, target_iters=5)
    t_start_total = time.time()
    step = 0

    print(f"Setup Complete: {mesh.node_count()} nodes, {mesh.element_count()} elements.")
    print(f"Surface Tie Active: Left={tie_left.n_active} pairs, Right={tie_right.n_active} pairs.")
    print("-" * 100)

    dt = dt_ctrl.dt
    while solver.time < T_TOTAL - 1e-12:
        step += 1
        dt_step = min(dt, T_TOTAL - solver.time)
        t_next = solver.time + dt_step

        # Smooth ramp amplitude (C2 continuous)
        s = t_next / T_TOTAL
        amp = s * s * (3.0 - 2.0 * s)

        theta_L = -THETA_MAX * amp
        theta_R = THETA_MAX * amp

        rbe2_left.set_prescribed_theta(theta_L)
        rbe2_right.set_prescribed_theta(theta_R)

        n_iter = solver.solve_step(dt_step)

        if n_iter < 0:
            print(f"  STEP {step}: CUTBACK at t={solver.time:.4f}")
            dt = dt_ctrl.update(n_iter=25, converged=False)
            continue

        max_u = float(np.max(np.abs(solver.u)))
        print(f"  STEP {step:2d} | t={solver.time:.4f}s | dt={dt_step:.3f}s | "
              f"Iters={n_iter+1:2d} | θ=[{np.degrees(theta_L):6.1f}°, {np.degrees(theta_R):6.1f}°] | "
              f"max|u|={max_u:6.2f}mm")

        dt = dt_ctrl.update(n_iter=n_iter, converged=True)

    t_elapsed = time.time() - t_start_total
    print("-" * 100)
    print(f" SIMULATION COMPLETED IN {t_elapsed:.2f} SECONDS.")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 6. Export before (flat, t=0) / after (final) Shape PNGs
    # ------------------------------------------------------------------
    artifact_dir = r"C:\Users\GOODMAN\.gemini\antigravity-cli\brain\17751c1b-5d61-4d49-980f-528b2d8cf463"
    before_png_path = plot_and_save_deformed_shape_png(
        mesh=mesh,
        u=np.zeros_like(solver.u),
        save_path="output/ex11_before_folding_shape.png",
        title_prefix="EX11: 90° Display Folding Before (Flat)",
        rbe2_elements=[rbe2_left, rbe2_right],
        tie_constraints=[tie_left, tie_right],
        artifact_dir=artifact_dir,
    )
    print(f"Before (flat) shape PNG exported successfully to {before_png_path}")

    png_path = plot_and_save_deformed_shape_png(
        mesh=mesh,
        u=solver.u,
        save_path="output/ex11_final_folding_shape.png",
        title_prefix="EX11: 90° Display Folding Final State",
        rbe2_elements=[rbe2_left, rbe2_right],
        tie_constraints=[tie_left, tie_right],
        artifact_dir=artifact_dir,
    )
    print(f"Final deformed shape PNG exported successfully to {png_path}")


if __name__ == "__main__":
    run_rigid_plate_display_fold()
