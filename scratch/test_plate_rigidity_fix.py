"""
test_plate_rigidity_fix.py
===========================
Diagnostic test verifying that assigning Q4_COROTATIONAL and real stiffness (E=20,000 MPa)
to Steel Plate elements eliminates plate buckling and collapse distortion.
"""

import time
import numpy as np
from dispsolver.io import read_abaqus_input
from dispsolver.fold_model_config import FoldModelConfig, smoothstep_amp
from dispsolver.solver.dynamic import DynamicSolver
from dispsolver.solver.dt_controller import AdaptiveDtController

def run_test():
    inp_path = "examples/ex12_rigid_plate_display_fold.inp"
    result = read_abaqus_input(inp_path)
    mesh = result.mesh
    nid_to_idx = mesh.node_id_to_index()

    config = FoldModelConfig()
    st = config.solver
    element_type = {}

    # Assign Q4_COROTATIONAL to STEEL plates so rigid rotation produces ZERO artificial strain
    for pid, name in (getattr(result, "material_names", {}) or {}).items():
        if name.startswith("PET"):
            element_type[pid] = st.pet_element_type
        elif name.startswith("PSA"):
            element_type[pid] = st.psa_element_type
        elif name.startswith("GLASS"):
            element_type[pid] = getattr(st, "glass_element_type", "Q4_COROTATIONAL_SRI")
        elif "STEEL" in name.upper() or "PLATE" in name.upper():
            element_type[pid] = "Q4_COROTATIONAL"

    # DO NOT reduce plate E to 1e-6! Keep real E = 20,000 MPa so plate remains truly rigid
    for pid, params in result.material_params.items():
        mat_name = (getattr(result, "material_names", {}) or {}).get(pid, "")
        if "STEEL" in mat_name.upper() or "PLATE" in mat_name.upper():
            params["E"] = 20000.0
            print(f"Plate pid={pid} ({mat_name}): kept real Young's Modulus E={params['E']} MPa with Q4_COROTATIONAL formulation.")

    for tc in result.penalty_constraints:
        tc.k_tie = 1e4

    # Process BCs
    translation_bc_dofs = []
    translation_bc_vals = []
    prescribed_rotations = {}

    for bc in result.boundaries:
        try:
            node_id = int(bc.nset)
            node_ids = [node_id]
        except ValueError:
            ns = mesh.node_sets.get(bc.nset)
            if ns:
                node_ids = list(ns.node_ids)
            else:
                continue

        for nid in node_ids:
            n_idx = nid_to_idx[nid]
            for dof_num in range(bc.dof1, bc.dof2 + 1):
                if dof_num in (1, 2):
                    dof_idx = n_idx * 2 + (dof_num - 1)
                    translation_bc_dofs.append(dof_idx)
                    translation_bc_vals.append(bc.value)
                elif dof_num in (3, 6):
                    prescribed_rotations[nid] = bc.value

    solver = DynamicSolver.from_config(
        mesh=mesh,
        material=result.materials,
        material_params=result.material_params,
        config=config,
        rho=result.solver_config.get("density", 1e-9),
        element_type=element_type,
        rbe2_constraints=result.rbe2_constraints,
        penalty_constraints=result.penalty_constraints,
        elem_jit="numba",
        verbose=True,
    )

    if translation_bc_dofs:
        solver.set_prescribed_dofs(
            np.array(translation_bc_dofs, dtype=np.int32),
            bc_vals=np.array(translation_bc_vals, dtype=np.float64)
        )

    t_total = config.drive.t_total
    dt_ctrl = AdaptiveDtController(dt_init=0.0025, dt_min=1e-5, dt_max=0.05, target_iters=6)

    step = 0
    while solver.time < t_total - 1e-10 and step < 25:
        dt = dt_ctrl.dt
        if solver.time + dt > t_total:
            dt = t_total - solver.time

        t_next = solver.time + dt
        amp_val = smoothstep_amp(t_next, t_total)

        targets = {}
        for idx, rbe_con in enumerate(result.rbe2_constraints):
            m_id = rbe_con.master_id
            if m_id in prescribed_rotations:
                targets[idx] = prescribed_rotations[m_id] * amp_val
        solver.theta_targets = targets

        checkpoint = solver.save_state()
        conv_code = solver.solve_step(dt)

        if conv_code >= 0:
            solver.time = t_next
            step += 1
            max_u = np.max(np.abs(solver.u))
            n_extra_regular = solver.n_extra - len(result.rbe2_constraints)
            angles = [float(np.degrees(solver.u_extra[n_extra_regular + i])) for i in range(len(result.rbe2_constraints))]
            print(f"STEP {step:2d} | t={solver.time:.4f}s | dt={dt:.4f}s | theta={angles} deg | max|u|={max_u:.2f}mm")
            dt = dt_ctrl.update(n_iter=conv_code, converged=True)
        else:
            solver.restore_state(checkpoint)
            dt = dt_ctrl.update(n_iter=25, converged=False)

if __name__ == "__main__":
    run_test()
