"""
ex12_abaqus_inp_plate_fold.py
=============================
Runs a 90-degree folding simulation by reading all parameters, mesh,
materials, rigid bodies, tie constraints, and boundary conditions
directly from an Abaqus input deck (.inp) file.
"""

import os
import time
import numpy as np
from dispsolver.io import read_abaqus_input
from dispsolver.solver import DynamicSolver
from dispsolver.solver.dt_controller import AdaptiveDtController
from dispsolver.export.plotter import plot_and_save_deformed_shape_png

def run_abaqus_inp_folding():
    print("=" * 100)
    print(" EX12: 90-Degree Display Folding Directly from Abaqus .inp File")
    print("=" * 100)

    inp_path = os.path.join(os.path.dirname(__file__), "ex12_rigid_plate_display_fold.inp")
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"Input file not found at {inp_path}. Run gen_ex12_inp.py first.")

    print(f"Reading Abaqus input deck: {inp_path}")
    result = read_abaqus_input(inp_path)
    print(f"Parsed result: {result}")

    mesh = result.mesh
    nid_to_idx = mesh.node_id_to_index()

    # Adjust rigid plate material parameters to virtually zero stiffness
    # to avoid spurious stresses from finite rotations of linear Q4 elements.
    for pid, params in result.material_params.items():
        if params.get("E", 0.0) > 10000.0:
            print(f"Adjusting plate material (pid={pid}) Young's modulus from {params['E']} to 1e-6 to eliminate spurious stresses.")
            params["E"] = 1e-6
            
    for tc in result.penalty_constraints:
        print(f"Adjusting tie constraint ({tc.name}) penalty stiffness from {tc.k_tie} to 1e4.")
        tc.k_tie = 1e4

    # ------------------------------------------------------------------
    # 1. Process boundary conditions from .inp
    # ------------------------------------------------------------------
    translation_bc_dofs = []
    translation_bc_vals = []
    prescribed_rotations = {}  # master_node_id -> prescribed_theta_value

    for bc in result.boundaries:
        # Resolve node IDs for the boundary condition
        # (could be a node ID string like "10000" or a set name)
        try:
            node_id = int(bc.nset)
            node_ids = [node_id]
        except ValueError:
            ns = mesh.node_sets.get(bc.nset)
            if ns:
                node_ids = list(ns.node_ids)
            else:
                continue

        # Check DOFs
        for nid in node_ids:
            n_idx = nid_to_idx[nid]
            for dof_num in range(bc.dof1, bc.dof2 + 1):
                if dof_num in (1, 2):
                    # Translation DOFs (1=UX, 2=UY)
                    dof_idx = n_idx * 2 + (dof_num - 1)
                    translation_bc_dofs.append(dof_idx)
                    translation_bc_vals.append(bc.value)
                elif dof_num in (3, 6):
                    # Rotation DOF (3 in 2D, 6 in 3D)
                    prescribed_rotations[nid] = bc.value

    print(f"Found translation boundary conditions on {len(translation_bc_dofs)} DOFs.")
    print(f"Found prescribed rotations on master nodes: {prescribed_rotations}")

    # ------------------------------------------------------------------
    # 2. Setup Solver
    # ------------------------------------------------------------------
    # We use exact kinematic condensation for rigid plate movement
    solver = DynamicSolver(
        mesh=mesh,
        material=result.materials,
        material_params=result.material_params,
        rho=result.solver_config.get("density", 1e-9),
        mode="quasistatic",
        tol=1e-3,
        rtol=1e-4,
        atol=1e-6,
        max_iter=25,
        element_type={1: "Q4_COROTATIONAL", 2: "Q4_VISCO_SIMO"},
        rbe2_constraints=result.rbe2_constraints,
        penalty_constraints=result.penalty_constraints,
        ul_mode=True,
        alpha=-0.15,
        verbose=True,
    )

    solver.theta_penalty_k = 1e8

    # Set translation boundaries
    if translation_bc_dofs:
        solver.set_prescribed_dofs(
            np.array(translation_bc_dofs, dtype=np.int32),
            bc_vals=np.array(translation_bc_vals, dtype=np.float64)
        )

    # ------------------------------------------------------------------
    # 3. Time Stepping Loop
    # ------------------------------------------------------------------
    t_total = result.solver_config.get("t_total", 1.0)
    dt_init = result.solver_config.get("dt_init", 0.005)
    dt_max = result.solver_config.get("dt_max", 0.01)
    dt_min = result.solver_config.get("dt_min", 1e-5)

    dt_ctrl = AdaptiveDtController(dt_init=dt_init, dt_min=dt_min, dt_max=dt_max, target_iters=5)
    t_start_total = time.time()
    step = 0

    while solver.time < t_total - 1e-10:
        dt = dt_ctrl.dt
        if solver.time + dt > t_total:
            dt = t_total - solver.time

        t_next = solver.time + dt
        ratio = t_next / t_total

        # Prescribe current step rotation values proportionally via theta_targets
        targets = {}
        for idx, rbe_con in enumerate(result.rbe2_constraints):
            m_id = rbe_con.master_id
            if m_id in prescribed_rotations:
                theta_target = prescribed_rotations[m_id]
                targets[idx] = theta_target * ratio
        solver.theta_targets = targets

        print("-" * 100)
        print(f"Step {step + 1} | Attempting time increment: dt = {dt:.5f}s (t: {solver.time:.4f}s -> {t_next:.4f}s)")
        print("-" * 100)

        # Solve step
        conv_code = solver.solve_step(dt)

        if conv_code >= 0:
            # Step converged
            solver.time = t_next
            step += 1
            # Log current angles from u_extra
            n_extra_regular = solver.n_extra - len(result.rbe2_constraints)
            angles = [np.degrees(solver.u_extra[n_extra_regular + idx]) for idx in range(len(result.rbe2_constraints))]
            print(f"\n  STEP {step:2d} | t={solver.time:.4f}s | dt={dt:.4f}s | Iters={conv_code:2d} | θ={angles} deg | max|u|={np.max(np.abs(solver.u)):.2f}mm\n")
            dt = dt_ctrl.update(n_iter=conv_code, converged=True)
        else:
            # Cutback
            dt = dt_ctrl.update(n_iter=25, converged=False)
            if dt <= dt_ctrl.dt_min:
                # AdaptiveDtController.update() clips dt to exactly dt_min
                # (np.clip), so it never goes strictly below it -- a `<`
                # check here would never fire and the run would cutback
                # forever at the floor instead of aborting cleanly.
                print("FATAL: Time increment below minimum tolerance. Simulation aborted.")
                break
            print(f"  *** STEP {step+1} FAILED (conv={conv_code}). Cutting back dt to {dt_ctrl.dt:.5f}s ***\n")

    t_elapsed = time.time() - t_start_total
    print("=" * 100)
    if solver.time >= t_total - 1e-10:
        print(f" SUCCESS: 90-degree display folding simulation completed in {t_elapsed:.2f}s!")
    else:
        print(" FAILED: Simulation did not reach target time.")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 4. Save visualization to PNG
    # ------------------------------------------------------------------
    png_path = os.path.join(os.path.dirname(__file__), "ex12_final_folding_shape.png")
    print(f"Saving deformed shape visualization to: {png_path}")
    plot_and_save_deformed_shape_png(
        mesh=mesh,
        u=solver.u,
        tie_constraints=result.penalty_constraints,
        rbe2_elements=result.rbe2_elements,
        save_path=png_path
    )
    print("Done.")

if __name__ == "__main__":
    run_abaqus_inp_folding()
