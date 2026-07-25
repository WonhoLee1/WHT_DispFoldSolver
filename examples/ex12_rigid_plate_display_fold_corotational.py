"""
ex12_rigid_plate_display_fold_corotational.py
==============================================
Rigid-plate + surface-tie architecture for U-shape display folding, using
the co-rotational Q4 + J2 plasticity element (see AGENTS.md section 4) for
the display.

Architecture (corrected -- see AGENTS.md section 4.10)
--------------------------------------------------------
Earlier revisions of this example drove the plates via `RBE2HingeElement`
(a penalty + Augmented-Lagrangian rigid link). That element was already
tried, historically, as the mechanism for rotating the display and found
NOT to work -- which is *why* `RigidBodyPart` (exact kinematic rigid-body
motion) and `SurfaceTieConstraint` were introduced in the first place.
Using `RBE2HingeElement` here was a regression back to the already-
abandoned approach. This revision removes it entirely:

1. Display Subsystem (pid=1, free DOFs):
   - Deformable multi-layer beam (x in [-40, 40] mm, total thickness = 0.5 mm)
   - Q4_COROTATIONAL elements + J2Plasticity material (same params as
     ex03_corotational_v4.py: E=4000 MPa, nu=0.3, sigma_y0=80 MPa, H=400 MPa).

2. Rigid Plate Subsystem (pid=2, EXACTLY prescribed, not free DOFs):
   - Every plate node's displacement is computed directly from
     `RigidBodyPart.get_slave_displacements(u_master, theta)`
     (u_s = u_m + (R(theta) - I) @ d0) and applied as a Dirichlet BC each
     step. There is no penalty/Lagrange gap on the plate-rigidity side by
     construction -- it's exact rigid-body kinematics, the same mechanism
     ex03_corotational_v4.py already validated (there applied directly to
     display nodes; here applied to a separate plate part).

3. Surface Tie Coupling: penalty-based `SurfaceTieConstraint` connects the
   free display's bottom surface to the now-exactly-driven plate top
   surface. This is the *only* compliant coupling left in the model.
"""

import os
import time
import numpy as np

from dispsolver.mesh import Mesh
from dispsolver.mesh.plate_builder import create_folding_plate_parts
from dispsolver.part.rigid_body import RigidBodyPart
from dispsolver.constraint.surface_tie import SurfaceTieConstraint
from dispsolver.material.neohookean import NeoHookean
from dispsolver.material.plastic import J2Plasticity
from dispsolver.solver.dynamic import DynamicSolver
from dispsolver.solver.dt_controller import AdaptiveDtController
from dispsolver.solver.diagnostics import sanity_report, fold_success_verdict, check_hinge_rotation
from dispsolver.export.plotter import plot_and_save_deformed_shape_png


def run_rigid_plate_display_fold_corotational():
    print("=" * 100)
    print(" EX12: 90-Degree Display Folding, EXACT Rigid Plate BC + Tie, CO-ROTATIONAL Q4 + J2 Display")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. Build Display Mesh (Deformable)
    # ------------------------------------------------------------------
    nx_disp = 80
    ny_disp = 4
    x_coords = np.linspace(-40.0, 40.0, nx_disp + 1)
    y_coords = np.linspace(0.0, 0.5, ny_disp + 1)

    mesh = Mesh()
    nid = 1

    grid_nids = np.zeros((ny_disp + 1, nx_disp + 1), dtype=int)
    bottom_surface_nids = []

    for j, y_val in enumerate(y_coords):
        for i, x_val in enumerate(x_coords):
            grid_nids[j, i] = nid
            mesh.add_node(nid, x_val, y_val)
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
            mesh.add_element(eid, [n1, n2, n3, n4], "Q4_COROTATIONAL", pid=1)
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
        base_node_id=10000,
        base_elem_id=10000,
    )

    left_plate = plates["left"]
    right_plate = plates["right"]

    for p_name, p_data in plates.items():
        for nid_p, coord in p_data["nodes_dict"].items():
            mesh.add_node(nid_p, coord[0], coord[1])
        for eid_p, conn in p_data["elements_dict"].items():
            mesh.add_element(eid_p, conn, "Q4", pid=2)

    nid_to_idx = mesh.node_id_to_index()
    coords = mesh.nodes_array()

    # ------------------------------------------------------------------
    # 3. Exact rigid-body kinematics for each plate (NOT RBE2HingeElement
    #    -- see module docstring / AGENTS.md section 4.10)
    # ------------------------------------------------------------------
    left_body = RigidBodyPart(
        name="PLATE_LEFT",
        master_id=left_plate["master_rp_id"],
        slave_ids=left_plate["slave_nids"],
        master_coord=left_plate["master_rp_coord"],
        slave_coords=np.array([coords[nid_to_idx[sid]] for sid in left_plate["slave_nids"]]),
    )
    right_body = RigidBodyPart(
        name="PLATE_RIGHT",
        master_id=right_plate["master_rp_id"],
        slave_ids=right_plate["slave_nids"],
        master_coord=right_plate["master_rp_coord"],
        slave_coords=np.array([coords[nid_to_idx[sid]] for sid in right_plate["slave_nids"]]),
    )

    # Every plate DOF (master RP + all slave grid nodes) is prescribed --
    # order must match how bc_vals is built in the time loop below.
    left_plate_node_ids = [left_plate["master_rp_id"]] + left_plate["slave_nids"]
    right_plate_node_ids = [right_plate["master_rp_id"]] + right_plate["slave_nids"]
    plate_bc_dof_list = []
    for nid_p in left_plate_node_ids + right_plate_node_ids:
        idx = nid_to_idx[nid_p]
        plate_bc_dof_list.extend([2 * idx, 2 * idx + 1])
    plate_bc_dofs = np.array(plate_bc_dof_list, dtype=np.int32)

    # ------------------------------------------------------------------
    # 4. Surface Tie Coupling: Display Bottom -> Plate Top (the only
    #    compliant link left in the model)
    # ------------------------------------------------------------------
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
    # 5. Initialize Solver (no rbe2_elements -- plate motion is a plain
    #    Dirichlet BC now, not a solved constraint)
    # ------------------------------------------------------------------
    mat_display = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=400.0)
    mat_plate = NeoHookean()

    materials = {1: mat_display, 2: mat_plate}
    element_types = {1: "Q4_COROTATIONAL", 2: "Q4"}
    mat_params = {
        2: {"E": 20000.0, "nu": 0.3},
    }

    solver = DynamicSolver(
        mesh=mesh,
        material=materials,
        element_type=element_types,
        material_params=mat_params,
        rho=1e-9,
        mode="quasistatic",
        rtol=1e-4,
        atol=1e-6,
        max_iter=25,
        penalty_constraints=[tie_left, tie_right],
        ul_mode=True,
        verbose=True,
    )

    solver.set_prescribed_dofs(plate_bc_dofs, bc_vals=np.zeros(len(plate_bc_dofs)))

    # DOF groups for the display<->plate tracking sanity check (see
    # diagnostics.sanity_report / AGENTS.md section 4.8/4.9).
    display_node_ids = list(range(1, (ny_disp + 1) * (nx_disp + 1) + 1))
    display_idx = [nid_to_idx[n] for n in display_node_ids]
    display_dofs = np.array([d for idx in display_idx for d in (2 * idx, 2 * idx + 1)], dtype=np.int32)
    plate_dofs = plate_bc_dofs  # same DOFs, already prescribed -- used as the "driving" group

    # Hinge free-zone bottom-row node IDs (x in [-10,10]) for the
    # AGENTS.md section 1.0 "smooth U, not a sharp crease" curvature check.
    hinge_node_ids = [n for n in bottom_surface_nids if -10.0 <= mesh.nodes[n].x <= 10.0]

    # ------------------------------------------------------------------
    # 6. Time Stepping Loop with Adaptive Dt Controller
    # ------------------------------------------------------------------
    T_TOTAL = 1.0
    THETA_MAX = np.radians(90.0)

    # target_iters=8: with the exact-BC plate drive, steady-state Newton
    # iteration count settles around 6-7 once the plate motion is genuinely
    # rigid (see AGENTS.md 4.10). target_iters=5 caused dt to shrink every
    # single step (growth ratio target/(iters+1) = 5/7 < 1 forever), never
    # recovering -- geometric decay toward dt_min with no way back up.
    dt_ctrl = AdaptiveDtController(dt_init=0.005, dt_min=1e-5, dt_max=0.01, target_iters=8)
    t_start_total = time.time()
    step = 0

    print(f"Setup Complete: {mesh.node_count()} nodes, {mesh.element_count()} elements.")
    print(f"Surface Tie Active: Left={tie_left.n_active} pairs, Right={tie_right.n_active} pairs.")
    print(f"Plate DOFs (exactly prescribed): {len(plate_bc_dofs)} / {solver.n_dofs} total")
    print("-" * 100)

    # Optional wall-clock preview cutoff (WHT_MAX_WALL_SECONDS env var) so a
    # quick PNG preview doesn't have to wait for the full multi-minute run.
    # Unset -> runs to full completion as normal.
    max_wall = os.environ.get("WHT_MAX_WALL_SECONDS")
    max_wall = float(max_wall) if max_wall else None

    u_master_zero = np.zeros(2)
    theta_L = 0.0
    theta_R = 0.0

    dt = dt_ctrl.dt
    while solver.time < T_TOTAL - 1e-12:
        if max_wall is not None and (time.time() - t_start_total) > max_wall:
            print(f"  [PREVIEW CUTOFF] wall-clock budget {max_wall:.0f}s reached at t={solver.time:.4f} - stopping early for PNG export.")
            break
        step += 1
        dt_step = min(dt, T_TOTAL - solver.time)
        t_next = solver.time + dt_step

        s = t_next / T_TOTAL
        amp = s * s * (3.0 - 2.0 * s)

        theta_L = -THETA_MAX * amp
        theta_R = THETA_MAX * amp

        # Exact rigid-body BC values for every plate DOF, in the same
        # [master, slaves...] order used to build plate_bc_dofs above.
        u_left_slaves = left_body.get_slave_displacements(u_master_zero, theta_L)
        u_right_slaves = right_body.get_slave_displacements(u_master_zero, theta_R)
        new_bc_vals = np.concatenate([u_master_zero, u_left_slaves, u_master_zero, u_right_slaves])
        solver._bc_base_vals = new_bc_vals

        n_iter = solver.solve_step(dt_step)

        if n_iter < 0:
            print(f"  STEP {step}: CUTBACK at t={solver.time:.4f}")
            dt = dt_ctrl.update(n_iter=25, converged=False, conv_rate=solver.last_conv_rate)
            continue

        max_u = float(np.max(np.abs(solver.u)))
        print(f"  STEP {step:2d} | t={solver.time:.4f}s | dt={dt_step:.3f}s | "
              f"Iters={n_iter+1:2d} | theta=[{np.degrees(theta_L):6.1f}deg, {np.degrees(theta_R):6.1f}deg] | "
              f"max|u|={max_u:6.2f}mm")

        # Autonomous physical sanity check every step: Newton "converged"
        # is necessary but not sufficient -- verify the display is
        # actually tracking the (now exactly-driven) plate and the ties
        # are actually holding.
        physically_ok = sanity_report(
            step, solver.u, coords, nid_to_idx,
            tie_constraints=[tie_left, tie_right],
            driving_dofs=plate_dofs, driven_dofs=display_dofs,
            driving_label="plate", driven_label="display",
        )
        if not physically_ok:
            print(f"  [ABORT] step {step}: physical sanity check failed -- stopping "
                  f"before the run wastes more time on a physically-wrong state.")
            break

        dt = dt_ctrl.update(n_iter=n_iter, converged=True, conv_rate=solver.last_conv_rate)

    t_elapsed = time.time() - t_start_total
    print("-" * 100)
    _status = "COMPLETED" if solver.time >= T_TOTAL - 1e-12 else f"STOPPED EARLY (t={solver.time:.4f}/{T_TOTAL})"
    print(f" SIMULATION {_status} IN {t_elapsed:.2f} SECONDS.")
    print("=" * 100)

    # AGENTS.md section 1.0 combined verdict.
    q = solver._check_mesh_quality(solver.u)
    verdict = fold_success_verdict(
        solver.u, coords, nid_to_idx, hinge_node_ids,
        tie_constraints=[tie_left, tie_right],
        driving_dofs=plate_dofs, driven_dofs=display_dofs,
        plate_closure_kwargs=dict(
            left_master_id=left_plate["master_rp_id"], left_ref_id=left_plate["slave_nids"][-1],
            right_master_id=right_plate["master_rp_id"], right_ref_id=right_plate["slave_nids"][-1],
            target_theta_l_rad=theta_L, target_theta_r_rad=theta_R,
        ),
    )
    closure = verdict.get("closure")
    print(f"AGENTS.md 1.0 verdict: u_shape_ok={verdict['u_shape_ok']}  "
          f"n_inverted={q['n_inverted']}  kink_detected={verdict['curvature']['kink_detected']}  "
          f"closure_both_ok={closure['both_ok'] if closure else None}  "
          f"plate_gap_mm={closure['plate_gap_mm']:.2f}" if closure else
          f"AGENTS.md 1.0 verdict: u_shape_ok={verdict['u_shape_ok']}  n_inverted={q['n_inverted']}  "
          f"kink_detected={verdict['curvature']['kink_detected']}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 7. Export Final Shape PNG
    # ------------------------------------------------------------------
    png_path = plot_and_save_deformed_shape_png(
        mesh=mesh,
        u=solver.u,
        save_path="output/ex12_final_folding_shape.png",
        title_prefix="EX12: 90 deg Display Folding Final State (Co-rotational, exact rigid plate BC)",
        tie_constraints=[tie_left, tie_right],
    )
    print(f"Final deformed shape PNG exported successfully to {png_path}")


if __name__ == "__main__":
    run_rigid_plate_display_fold_corotational()
