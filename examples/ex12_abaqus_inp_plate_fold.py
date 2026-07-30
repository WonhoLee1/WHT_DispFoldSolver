"""
ex12_abaqus_inp_plate_fold.py
=============================
Runs a 90-degree folding simulation by reading all parameters, mesh,
materials, rigid bodies, tie constraints, and boundary conditions
directly from an Abaqus input deck (.inp) file.
"""

import os
import sys
import time
import numpy as np
from dispsolver.io import read_abaqus_input
from dispsolver.solver import DynamicSolver
from dispsolver.solver.dt_controller import AdaptiveDtController
from dispsolver.export.plotter import plot_and_save_deformed_shape_png
from dispsolver.postprocess import LayerSlipTracker, ResultWriter
from dispsolver.postprocess.model_review import print_model_review_from_builder_result

sys.path.insert(0, os.path.dirname(__file__))
from dispsolver.fold_model_config import FoldModelConfig, DEFAULT_CONFIG


def _laminate_layer_materials(mesh, material_names, max_node_id=10000):
    """Material name of each display layer, bottom -> top.

    Derived from the mesh rather than hard-coded so the table stays
    correct if gen_ex12_inp.py's layup changes: each element is bucketed
    by its centroid height, and each layer takes the *MATERIAL name of
    the pid its elements carry.
    """
    ys = sorted({round(float(n.y), 6) for nid, n in mesh.nodes.items()
                 if nid < max_node_id})
    if len(ys) < 2:
        return None
    mid_to_layer = {}
    for j in range(len(ys) - 1):
        mid_to_layer[round(0.5 * (ys[j] + ys[j + 1]), 6)] = j

    layers = [None] * (len(ys) - 1)
    for elem in mesh.elements.values():
        if any(nid >= max_node_id for nid in elem.node_ids):
            continue
        yc = round(float(np.mean([mesh.nodes[n].y for n in elem.node_ids])), 6)
        j = mid_to_layer.get(yc)
        if j is not None and layers[j] is None:
            layers[j] = material_names.get(elem.pid, f"pid{elem.pid}")
    return [m or "?" for m in layers]

def run_abaqus_inp_folding(inp_path: str = None, before_png_name: str = "ex12_before_folding_shape.png",
                            after_png_name: str = "ex12_final_folding_shape.png",
                            result_name: str = "ex12_result.pkl",
                            max_steps: int = None,
                            elem_jit: str = "jax",
                            config: FoldModelConfig = DEFAULT_CONFIG):
    """Run the folding solve from a given .inp deck."""
    print("=" * 100)
    print(" EX12: 90-Degree Display Folding Directly from Abaqus .inp File")
    print("=" * 100)

    if inp_path is None:
        inp_path = os.path.join(os.path.dirname(__file__), "ex12_rigid_plate_display_fold.inp")
    if not os.path.exists(inp_path):
        raise FileNotFoundError(f"Input file not found at {inp_path}. Run gen_ex12_inp.py first.")

    t_pre_start = time.time()
    print(f"Reading Abaqus input deck: {inp_path}")
    result = read_abaqus_input(inp_path)
    t_pre = time.time() - t_pre_start
    print(f"Parsed result: {result}")

    res_dict = run_folding_from_result(result, before_png_name, after_png_name,
                                        case_name=os.path.basename(inp_path),
                                        result_name=result_name, max_steps=max_steps,
                                        elem_jit=elem_jit, config=config)
    if isinstance(res_dict, dict) and "solver" in res_dict:
        res_dict["solver"].t_preprocess = t_pre
    return res_dict


def run_folding_from_result(result, before_png_name: str = "ex12_before_folding_shape.png",
                             after_png_name: str = "ex12_final_folding_shape.png",
                             case_name: str = "in-memory",
                             result_name: str = "ex12_result.pkl",
                             max_steps: int = None,
                             elem_jit: str = "jax",
                             config: FoldModelConfig = DEFAULT_CONFIG):
    """Shared solve loop, driven by any object exposing the same fields as
    `dispsolver.io.model_builder.ModelBuilderResult` (`.mesh`, `.materials`,
    `.material_params`, `.solver_config`, `.rbe2_constraints`,
    `.penalty_constraints`, `.rbe2_elements`, `.boundaries`).

    `run_abaqus_inp_folding` builds this via `read_abaqus_input`;
    `ex13_unified_model_io.py`'s `build` mode constructs an equivalent
    result directly as Python objects (no .inp text at all) and calls
    this function directly, so all three model sources (read/roundtrip/
    build) drive the exact same solve loop with no duplication.
    """
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
    # Element type per pid, derived from the *MATERIAL name rather than
    # hard-coded to {1: ..., 2: ...} -- the 14-physical-layer remodel
    # (gen_ex12_inp.py) means the display alone now spans pids 1..14
    # (PET_1..PET_7, PSA_1..PSA_7), not just pid 1/2. A pid missing from
    # this dict silently defaults to plain "Q4" (DynamicSolver._pid_element_type),
    # which for J2Plasticity/ViscoelasticMaterial pids means falling
    # through to a much slower path instead of the fast per-pid-cached
    # Q4_COROTATIONAL/Q4_VISCO_SIMO batch kernels (see dynamic.py's
    # _coro_jax_vmap_by_pid/_visco_simo_jax_vmap_by_pid pre-build) --
    # exactly the bug that made the first Newton iteration take 20+
    # minutes after the remodel, even after those caches were fixed to
    # be shared, because only pid 1/2 were ever tagged to use them.
    st = config.solver
    element_type = {}
    for pid, name in (getattr(result, "material_names", {}) or {}).items():
        if name.startswith("PET"):
            element_type[pid] = st.pet_element_type
        elif name.startswith("PSA"):
            element_type[pid] = st.psa_element_type
        # STEEL (or anything else): leave unset, defaults to plain "Q4"
        # -- NeoHookean already dispatches through the generic
        # pk2_tangent_voigt_batch path regardless of this dict.

    solver = DynamicSolver(
        mesh=mesh,
        material=result.materials,
        material_params=result.material_params,
        rho=result.solver_config.get("density", 1e-9),
        mode="quasistatic",
        tol=st.tol,
        rtol=st.rtol,
        atol=st.atol,
        max_iter=st.max_iter,
        element_type=element_type,
        rbe2_constraints=result.rbe2_constraints,
        penalty_constraints=result.penalty_constraints,
        ul_mode=st.ul_mode,
        alpha=st.alpha,
        elem_jit=elem_jit,
        verbose=True,
    )

    solver.theta_penalty_k = st.theta_penalty_k

    print_model_review_from_builder_result(result, pid_element_type=getattr(solver, 'element_type_by_pid', None))

    # Set translation boundaries
    if translation_bc_dofs:
        solver.set_prescribed_dofs(
            np.array(translation_bc_dofs, dtype=np.int32),
            bc_vals=np.array(translation_bc_vals, dtype=np.float64)
        )

    # ------------------------------------------------------------------
    # 3. Time Stepping Loop
    # ------------------------------------------------------------------
    drv = config.drive
    t_total = drv.t_total
    dt_init = drv.dt_init
    dt_max = drv.dt_max
    dt_min = drv.dt_min

    dt_ctrl = AdaptiveDtController(dt_init=dt_init, dt_min=dt_min, dt_max=dt_max, target_iters=st.target_iters)
    t_start_total = time.time()
    step = 0

    # Record every converged increment so the finished run can be
    # re-analyzed/animated without re-solving (~5 min), and track the
    # per-layer tip slip that the soft PSA rows exist to produce.
    layer_materials = _laminate_layer_materials(
        mesh, getattr(result, "material_names", {}) or {}
    )
    writer = ResultWriter(
        mesh,
        materials=result.material_params,
        constraints={"ties": [getattr(tc, "name", "") for tc in result.penalty_constraints],
                     "rbe2_masters": [c.master_id for c in result.rbe2_constraints]},
        meta={"case": case_name, "t_total": t_total,
              "theta_targets_rad": prescribed_rotations,
              "layer_materials": layer_materials,
              # pid -> *MATERIAL name, so the Qt viewer's Part/Layer
              # checkboxes can show e.g. "Layer 3 (PET_2)" instead of a
              # bare pid number.
              "pid_names": getattr(result, "material_names", {}) or {},
              # pid -> canonical type tag (dispsolver.material.type_tags),
              # so a loaded Result can print representative material
              # properties (model_review.py) without re-inferring the
              # material kind from raw param-dict key presence.
              "pid_types": getattr(result, "material_types", {}) or {},
              # pid -> human display-name override for non-layer parts
              # (e.g. rigid plates: "Plate Left"/"Plate Right"). Additive,
              # backward compatible -- viewer.py/model_review.py .get()
              # this with a {} fallback for older saved results.
              "pid_part_names": getattr(result, "part_names", {}) or {}},
        # Live material objects (not the JSON-safe material_params above),
        # so the Qt viewer can recompute stress from a saved result the
        # same way it does from a live solver. Pickle-only.
        material_objects=result.materials,
    )
    slip_tracker = LayerSlipTracker(
        mesh, probe_x=(-40.0, 40.0), layer_materials=layer_materials,
    )

    while solver.time < t_total - 1e-10:
        if max_steps is not None and step >= max_steps:
            # Early stop for profiling/smoke runs. Unlike shrinking
            # t_total this leaves the angle ramp untouched, so the steps
            # taken are the same ones a full run would take.
            print(f"  [MAX STEPS] stopping after {step} steps as requested.")
            break
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

        # Snapshot before the attempt so a failed step can be rolled back
        # cleanly -- solve_step() only commits self.u/v/a/state/lam on
        # success (see its docstring: "On failure ... return negative
        # iteration count for caller cutback"), so the caller owns cutback
        # rollback. Every other example script in this repo (ex03-ex08,
        # mode_comparison.py, profile_solver.py, ...) already follows this
        # save_state()/restore_state() pattern; this loop was missing it.
        # Without it, self.eas_alpha (Q4_EAS/Q4_COROTATIONAL_EAS's per-
        # element enhanced-strain warm start) stays poisoned by the failed
        # attempt's last (possibly wildly divergent) Newton iterate --
        # eas_alpha is mutated in-place inside _assemble() on every trial
        # iteration, unlike self.state, which only commits via a local
        # state_new on convergence. A poisoned alpha warm start can settle
        # on a different EAS stationary-point branch than the rest of the
        # (correctly rolled-back) structure expects, producing a residual
        # mismatch at Newton iteration 1 that no amount of dt shrinking
        # fixes, since the corruption isn't in the load increment size.
        checkpoint = solver.save_state()

        # Solve step
        conv_code = solver.solve_step(dt)

        if conv_code >= 0:
            # Step converged
            solver.time = t_next
            step += 1
            # Log current angles from u_extra
            n_extra_regular = solver.n_extra - len(result.rbe2_constraints)
            angles = [float(np.degrees(solver.u_extra[n_extra_regular + idx])) for idx in range(len(result.rbe2_constraints))]
            angles_str = "[" + ", ".join(f"{a:.4f}" for a in angles) + "]" if angles else "[]"
            print(f"\n  STEP {step:2d} | t={solver.time:.4f}s | dt={dt:.4f}s | Iters={conv_code:2d} | θ={angles_str} deg | max|u|={np.max(np.abs(solver.u)):.2f}mm\n")

            theta_deg = float(np.mean(np.abs(angles))) if angles else 0.0
            writer.add_step(solver.time, solver.u,
                            scalars={"theta_deg": theta_deg, "dt": dt,
                                     "n_iter": float(conv_code)},
                            state=(solver.state.copy() if solver.state is not None else None))
            slip_tracker.record(solver.u, theta_deg=theta_deg, time=solver.time)

            dt = dt_ctrl.update(n_iter=conv_code, converged=True)
        else:
            # Cutback -- restore the pre-attempt snapshot (see checkpoint
            # comment above) before retrying at a smaller dt, so eas_alpha
            # (and u/v/a/state/lam) start the next attempt from the last
            # known-good state rather than the failed attempt's residue.
            solver.restore_state(checkpoint)
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
    reached_target = solver.time >= t_total - 1e-10
    print("=" * 100)
    if reached_target:
        print(f" SUCCESS: 90-degree display folding simulation completed in {t_elapsed:.2f}s!")
    else:
        print(" FAILED: Simulation did not reach target time.")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 4. Persist the run and report interlayer shear
    #
    # Runs on BOTH exit paths on purpose: a run that stalled at 40 deg is
    # exactly the one whose history you want to inspect, and re-solving to
    # get it back costs ~5 minutes.
    # ------------------------------------------------------------------
    writer.update_meta(reached_target=reached_target, wall_seconds=t_elapsed,
                       n_steps=writer.n_steps, final_time=solver.time)
    for name, arr in slip_tracker.history().items():
        writer.add_array(f"slip__{name}", arr)
    t_post_start = time.time()
    result_path = os.path.join(os.path.dirname(__file__), result_name)
    writer.save(result_path)
    print(f"Saved result ({writer.n_steps} steps) for re-analysis: {result_path}")
    print(f"  reload with: "
          f"from dispsolver.postprocess import load_result; "
          f"r = load_result(r'{result_path}')")

    print(slip_tracker.format_table())
    print()

    # ------------------------------------------------------------------
    # 5. Save before (flat, t=0) / after (final) visualization PNGs
    # ------------------------------------------------------------------
    before_png_path = os.path.join(os.path.dirname(__file__), before_png_name)
    print(f"Saving before (flat) shape visualization to: {before_png_path}")
    plot_and_save_deformed_shape_png(
        mesh=mesh,
        u=np.zeros_like(solver.u),
        tie_constraints=result.penalty_constraints,
        rbe2_elements=result.rbe2_elements,
        save_path=before_png_path
    )

    after_png_path = os.path.join(os.path.dirname(__file__), after_png_name)
    print(f"Saving after (final) shape visualization to: {after_png_path}")
    plot_and_save_deformed_shape_png(
        mesh=mesh,
        u=solver.u,
        tie_constraints=result.penalty_constraints,
        rbe2_elements=result.rbe2_elements,
        save_path=after_png_path
    )
    solver.t_postprocess = time.time() - t_post_start
    print("Done.")
    return {
        "solver": solver,
        "result": result,
        "n_nodes": len(mesh.nodes),
        "n_elements": len(mesh.elements),
        "n_rbe2_constraints": len(result.rbe2_constraints),
        "n_penalty_constraints": len(result.penalty_constraints),
        "reached_target": solver.time >= t_total - 1e-10,
        "result_path": result_path,
    }

if __name__ == "__main__":
    run_abaqus_inp_folding()
