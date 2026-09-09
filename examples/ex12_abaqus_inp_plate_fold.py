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
from dispsolver.solver.diagnostics import check_region_tracking, check_tie_gap, check_smooth_curvature
from dispsolver.export.plotter import plot_and_save_deformed_shape_png
from dispsolver.postprocess import LayerSlipTracker, ResultWriter
from dispsolver.postprocess.model_review import (
    compute_pid_thickness_from_mesh,
    print_model_review_from_builder_result,
)

sys.path.insert(0, os.path.dirname(__file__))
from dispsolver.fold_model_config import FoldModelConfig, DEFAULT_CONFIG, make_teardrop_config


def _laminate_layer_materials(mesh, material_names, max_node_id=100000):
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
                            threads: int = None,
                            config: FoldModelConfig = None):
    """Run the folding solve from a given .inp deck."""
    if config is None:
        config = make_teardrop_config()
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
                                        elem_jit=elem_jit, threads=threads, config=config)
    if isinstance(res_dict, dict) and "solver" in res_dict:
        res_dict["solver"].t_preprocess = t_pre
    return res_dict


def run_folding_from_result(result, before_png_name: str = "ex12_before_folding_shape.png",
                             after_png_name: str = "ex12_final_folding_shape.png",
                             case_name: str = "in-memory",
                             result_name: str = "ex12_result.pkl",
                             max_steps: int = None,
                             elem_jit: str = "jax",
                             threads: int = None,
                             config: FoldModelConfig = None):
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
    if config is None:
        config = make_teardrop_config()
    if threads is not None and threads > 0:
        n_str = str(threads)
        os.environ["OMP_NUM_THREADS"] = n_str
        os.environ["MKL_NUM_THREADS"] = n_str
        os.environ["NUMBA_NUM_THREADS"] = n_str
        os.environ["OPENBLAS_NUM_THREADS"] = n_str
    mesh = result.mesh
    nid_to_idx = mesh.node_id_to_index()

    # ------------------------------------------------------------------
    # 0. Diagnostics setup (AGENTS.md sec 4.9: convergence success alone
    #    does not prove the plate<->display coupling is real -- the
    #    printed "TieNodes: N/N active" count is NOT evidence of a tight
    #    tie, since reproject_deformed() unconditionally re-pairs every
    #    initially-tied slave node to its nearest master segment every
    #    iteration regardless of how large the actual gap has grown.
    #    These checks compute the real geometric gap/tracking instead.
    # ------------------------------------------------------------------
    _master_ids = {c.master_id for c in result.rbe2_constraints}
    _driving_dofs = []
    for c in result.rbe2_constraints:
        mi = nid_to_idx[c.master_id]
        _driving_dofs += [2 * mi, 2 * mi + 1]
    _driven_dofs = []
    for nid in mesh.nodes:
        if nid < 100000 and nid not in _master_ids:
            di = nid_to_idx[nid]
            _driven_dofs += [2 * di, 2 * di + 1]
    _hinge_half_gap = getattr(getattr(config, "geometry", None), "hinge_half_gap", 7.5)
    _min_y = min(n.y for nid, n in mesh.nodes.items() if nid < 100000)
    _hinge_node_ids = [nid for nid, n in mesh.nodes.items()
                       if nid < 100000 and abs(n.y - _min_y) < 1e-6 and abs(n.x) <= _hinge_half_gap + 2.0]

    # ------------------------------------------------------------------
    # 0b. Kinematic (MPC-elimination) tie map -- Abaqus *TIE default behaviour
    # ------------------------------------------------------------------
    # Each tie pair gives u_s = N1 u_m1 + N2 u_m2. Both masters sit on a rigid
    # plate whose motion is an exact closed-form function of the drive angle,
    # so the tied slave displacement is known analytically each step and can be
    # prescribed -- zero gap by construction, no penalty stiffness to tune.
    _tie_method = getattr(config.solver, "tie_method", "penalty")
    _kin_tie = []          # (s_idx, m1_idx, m2_idx, N1, N2)
    if _tie_method == "kinematic":
        for tc in (result.penalty_constraints or []):
            for (s_nid, m1_nid, m2_nid, xi) in getattr(tc, "pairs", []):
                _kin_tie.append((nid_to_idx[s_nid], nid_to_idx[m1_nid], nid_to_idx[m2_nid],
                                 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)))
        print(f"[TIE] kinematic (MPC elimination): {len(_kin_tie)} slave nodes "
              f"prescribed from the rigid plate motion (penalty disabled)")

    # ------------------------------------------------------------------
    # 1. Process boundary conditions from .inp
    # ------------------------------------------------------------------
    translation_bc_dofs = []
    translation_bc_vals = []
    prescribed_rotations = {}  # master_node_id -> prescribed_theta_value

    for bc in result.boundaries:
        # Resolve node IDs for the boundary condition
        # (could be a node ID string like "100000" or a set name)
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
    st = config.solver
    element_type = {}
    for pid, name in (getattr(result, "material_names", {}) or {}).items():
        if name.startswith("PET"):
            element_type[pid] = st.pet_element_type
        elif name.startswith("PSA"):
            element_type[pid] = st.psa_element_type
        elif name.startswith("GLASS"):
            element_type[pid] = getattr(st, "glass_element_type", "Q4_COROTATIONAL_SRI")
        elif "STEEL" in name.upper() or "PLATE" in name.upper():
            element_type[pid] = "Q4_COROTATIONAL"

    solver = DynamicSolver.from_config(
        mesh=mesh,
        material=result.materials,
        material_params=result.material_params,
        config=config,
        rho=result.solver_config.get("density", 1e-9),
        element_type=element_type,
        rbe2_constraints=result.rbe2_constraints,
        # tie_method="kinematic" enforces the tie by master-slave (MPC)
        # elimination instead of a penalty spring -- the tied display DOFs are
        # prescribed exactly from the rigid plate's theta each step (see the
        # per-step block below), so no penalty constraint is registered.
        penalty_constraints=([] if _tie_method == "kinematic"
                             else result.penalty_constraints),
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
              "pid_part_names": getattr(result, "part_names", {}) or {},
              # pid -> layer/part physical thickness in mm (Delta y = max_y - min_y)
              "pid_thickness": compute_pid_thickness_from_mesh(mesh)},
        # Live material objects (not the JSON-safe material_params above),
        # so the Qt viewer can recompute stress from a saved result the
        # same way it does from a live solver. Pickle-only.
        material_objects=result.materials,
    )
    slip_tracker = LayerSlipTracker(
        mesh, probe_x=(-40.0, 40.0), layer_materials=layer_materials,
    )

    # Save initial flat shape PNG at step 0
    before_png_path = os.path.join(os.path.dirname(__file__), before_png_name)
    after_png_path = os.path.join(os.path.dirname(__file__), after_png_name)
    plot_and_save_deformed_shape_png(
        mesh=mesh,
        u=np.zeros_like(solver.u),
        tie_constraints=result.penalty_constraints,
        rbe2_elements=result.rbe2_elements,
        rbe2_constraints=result.rbe2_constraints,
        save_path=before_png_path
    )

    cutback_count = 0
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
        from dispsolver.fold_model_config import smoothstep_amp
        amp_val = smoothstep_amp(t_next, t_total)

        # Prescribe current step rotation values proportionally via C2 smoothstep ramp
        targets = {}
        for idx, rbe_con in enumerate(result.rbe2_constraints):
            m_id = rbe_con.master_id
            if m_id in prescribed_rotations:
                theta_target = prescribed_rotations[m_id]
                targets[idx] = theta_target * amp_val
        solver.theta_targets = targets

        if _kin_tie:
            # rigid-plate displacement of every master node at this theta:
            #   u_m = u_rp + (R(theta) - I) (X_m - X_rp),  u_rp = 0 (pinned)
            _um = {}
            for idx_c, rbe_con in enumerate(result.rbe2_constraints):
                th = float(targets.get(idx_c, 0.0))
                c_, s_ = np.cos(th), np.sin(th)
                m_idx = nid_to_idx[rbe_con.master_id]
                xm, ym = solver.coords[m_idx]
                for sid in rbe_con.slave_ids:
                    k = nid_to_idx[sid]
                    dx, dy = solver.coords[k][0] - xm, solver.coords[k][1] - ym
                    _um[k] = ((c_ - 1.0) * dx - s_ * dy, s_ * dx + (c_ - 1.0) * dy)
            _d, _v = [], []
            for (si, m1, m2, N1, N2) in _kin_tie:
                u1 = _um.get(m1); u2 = _um.get(m2)
                if u1 is None or u2 is None:
                    continue
                _d += [2 * si, 2 * si + 1]
                _v += [N1 * u1[0] + N2 * u2[0], N1 * u1[1] + N2 * u2[1]]
            if _d:
                _all_d = np.concatenate([np.array(translation_bc_dofs, dtype=np.int32),
                                         np.array(_d, dtype=np.int32)])
                _all_v = np.concatenate([np.array(translation_bc_vals, dtype=np.float64),
                                         np.array(_v, dtype=np.float64)])
                solver.set_prescribed_dofs(_all_d, bc_vals=_all_v)

        print("-" * 100)
        print(f"Step {step + 1} | Attempting time increment: dt = {dt:.5f}s (t: {solver.time:.4f}s -> {t_next:.4f}s)")
        print("-" * 100)

        # Snapshot before the attempt so a failed step can be rolled back
        checkpoint = solver.save_state()

        # Solve step
        conv_code = solver.solve_step(dt)

        if conv_code >= 0:
            # Step converged
            solver.time = t_next
            step += 1
            cutback_count = 0

            # --- Outer Augmented Lagrangian correction loop ---
            # A single converged Newton solve triggers exactly ONE
            # update_augmented_lagrange() call (inside dynamic.py, on
            # commit). Under continuously growing bending-moment demand
            # as fold angle increases, one lambda update per step cannot
            # catch up: the tie is a *penalty* spring, gap ~= F/k_tie,
            # and F keeps climbing every step, so the gap grows smoothly
            # and without bound (observed: 0mm through ~8deg, ~34mm by
            # 68deg) even though Newton itself converges cleanly every
            # step. Re-solve at the SAME theta target (tiny dt, does not
            # meaningfully advance solver.time) so the lambda update that
            # already fired gets a chance to actually cancel the residual
            # gap, repeating (a real Uzawa/AL outer iteration) until the
            # gap is small or a cap is hit.
            _al_iters = 0
            _al_max_iters = 8
            _al_gap_tol_mm = 0.02
            if result.penalty_constraints:
                _al_max_gap = max(
                    (check_tie_gap(solver.u, solver.coords, tc, nid_to_idx)["max_gap_mm"]
                     for tc in result.penalty_constraints), default=0.0
                )
                while _al_max_gap > _al_gap_tol_mm and _al_iters < _al_max_iters:
                    _al_conv = solver.solve_step(1e-6)
                    if _al_conv < 0:
                        break  # AL correction itself failed -- keep last good state, move on
                    _al_iters += 1
                    _al_max_gap = max(
                        (check_tie_gap(solver.u, solver.coords, tc, nid_to_idx)["max_gap_mm"]
                         for tc in result.penalty_constraints), default=0.0
                    )
                if _al_iters > 0:
                    print(f"            [AL] tie gap correction: {_al_iters} extra solve(s), "
                          f"final max_gap={_al_max_gap:.4f}mm")

            # Log current angles from u_extra and plate tip Y positions
            n_extra_regular = solver.n_extra - len(result.rbe2_constraints)
            angles = [float(np.degrees(solver.u_extra[n_extra_regular + idx])) for idx in range(len(result.rbe2_constraints))]
            angles_str = "[" + ", ".join(f"{a:+.2f}deg" for a in angles) + "]" if angles else "[]"
            
            # Find plate tip nodes (outermost x near -40mm and +40mm)
            y_left_tip, y_right_tip = 0.0, 0.0
            for nid, n in mesh.nodes.items():
                if abs(n.x - (-40.0)) < 1e-3 and abs(n.y - (-0.5)) < 1e-3:
                    n_idx = nid_to_idx[nid]
                    y_left_tip = n.y + solver.u[2 * n_idx + 1]
                elif abs(n.x - 40.0) < 1e-3 and abs(n.y - (-0.5)) < 1e-3:
                    n_idx = nid_to_idx[nid]
                    y_right_tip = n.y + solver.u[2 * n_idx + 1]

            target_theta_deg = 90.0 * amp_val
            max_u = np.max(np.abs(solver.u))
            print(f"\n  [MONITOR] Step {step:3d} | Time {solver.time:.4f}s / {t_total:.1f}s ({solver.time/t_total*100:5.1f}%)")
            print(f"            Fold Angle: Target +- {target_theta_deg:5.2f}deg | Actual {angles_str}")
            print(f"            Plate Tips Y: Left={y_left_tip:+6.2f}mm | Right={y_right_tip:+6.2f}mm | Max Disp={max_u:6.2f}mm")
            print(f"            Newton Iters: {conv_code:2d} | dt: {dt:.5f}s")

            _track = check_region_tracking(solver.u, _driving_dofs, _driven_dofs, "plate", "display")
            _gap_bits = []
            for tc in (result.penalty_constraints or []):
                g = check_tie_gap(solver.u, solver.coords, tc, nid_to_idx)
                _gap_bits.append(
                    f"{getattr(tc, 'name', 'tie')}: max_gap={g['max_gap_mm']:.4f}mm "
                    f"mean={g['mean_gap_mm']:.4f}mm n={g['n_pairs']}"
                )
            print(
                f"            [DIAGNOSTIC] display/plate max|u| ratio={_track['ratio']:.4f} "
                f"(display={_track['max|u|_display']:.3f}mm, plate={_track['max|u|_plate']:.3f}mm)"
                + ("  ** SUSPICIOUS: display not tracking plate **" if _track["suspicious"] else "")
            )
            if _gap_bits:
                print(f"            [DIAGNOSTIC] Real tie gap (not just active-pair count): " + " | ".join(_gap_bits))
            if step % 10 == 0 or step == 1:
                _curv = check_smooth_curvature(solver.u, solver.coords, nid_to_idx, _hinge_node_ids)
                print(
                    f"            [DIAGNOSTIC] Hinge curvature: max_curv_ratio={_curv['max_curv_ratio']:.2f} "
                    f"kink_detected={_curv['kink_detected']}"
                    + (f" at x={_curv['kink_x']:.2f}mm" if _curv["kink_detected"] else "")
                )
            print()

            theta_deg = float(np.mean(np.abs(angles))) if angles else 0.0
            writer.add_step(solver.time, solver.u,
                            scalars={"theta_deg": theta_deg, "dt": dt,
                                     "n_iter": float(conv_code)},
                            state=(solver.state.copy() if solver.state is not None else None))
            slip_tracker.record(solver.u, theta_deg=theta_deg, time=solver.time)

            # Format monitor summary string for bottom plot label
            n_tie_active = sum(getattr(tc, 'n_active', len(getattr(tc, 'pairs', []))) for tc in (result.penalty_constraints or []))
            monitor_text = (
                f"Step {step:3d} | Time {solver.time:.4f}s/{t_total:.1f}s ({solver.time/t_total*100:5.1f}%) | "
                f"Angle: Target ±{target_theta_deg:.2f}° / Actual {angles_str} | "
                f"Tips Y: L={y_left_tip:+.2f}mm, R={y_right_tip:+.2f}mm | Max|u|={max_u:.2f}mm | "
                f"Iters: {conv_code:2d} | dt: {dt:.5f}s | Tie: {n_tie_active} active"
            )

            # Live update final PNG image on every converged step
            plot_and_save_deformed_shape_png(
                mesh=mesh,
                u=solver.u,
                tie_constraints=result.penalty_constraints,
                rbe2_elements=result.rbe2_elements,
        rbe2_constraints=result.rbe2_constraints,
                save_path=after_png_path,
                monitor_text=monitor_text
            )

            dt = dt_ctrl.update(n_iter=conv_code, converged=True)
        else:
            # Cutback -- restore pre-attempt state and retry
            solver.restore_state(checkpoint)
            cutback_count = getattr(locals(), 'cutback_count', 0) + 1
            dt = dt_ctrl.update(n_iter=25, converged=False)
            max_cb = getattr(st, 'max_cutbacks', 20)
            if cutback_count >= max_cb or dt <= dt_ctrl.dt_min:
                print(f"FATAL: Exceeded maximum cutbacks ({max_cb}) or dt below minimum tolerance. Simulation aborted.")
                break
            print(f"  *** STEP {step+1} FAILED (conv={conv_code}, cutback {cutback_count}/{max_cb}). Cutting back dt to {dt_ctrl.dt:.5f}s ***\n")

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

    # Final plot update
    print(f"Saving final shape visualization to: {after_png_path}")
    plot_and_save_deformed_shape_png(
        mesh=mesh,
        u=solver.u,
        tie_constraints=result.penalty_constraints,
        rbe2_elements=result.rbe2_elements,
        rbe2_constraints=result.rbe2_constraints,
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
    import argparse
    from dispsolver.fold_model_config import add_config_cli_args, apply_cli_overrides, FoldModelConfig, DEFAULT_CONFIG

    parser = argparse.ArgumentParser(description="Run Abaqus .inp folding simulation with external configuration support.")
    parser.add_argument("--inp", type=str, default=None, help="Path to Abaqus .inp file")
    parser.add_argument("--elem_jit", type=str, default="numba", choices=["jax", "numba"], help="Element JIT backend")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum solve steps")
    add_config_cli_args(parser)

    args = parser.parse_args()
    config = apply_cli_overrides(DEFAULT_CONFIG, args)

    run_abaqus_inp_folding(
        inp_path=args.inp,
        elem_jit=args.elem_jit,
        threads=args.threads,
        max_steps=args.max_steps,
        config=config,
    )
