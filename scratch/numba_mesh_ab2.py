"""Mesh-level A/B at the ACTUAL first Newton iteration (BCs + theta applied)."""
import numpy as np
from dispsolver.io import read_abaqus_input
from dispsolver.solver import DynamicSolver
from dispsolver.fold_model_config import make_teardrop_config, smoothstep_amp

cfg = make_teardrop_config()
result = read_abaqus_input("examples/ex12_rigid_plate_display_fold.inp")
mesh = result.mesh
st = cfg.solver

element_type = {}
for pid, name in (getattr(result, "material_names", {}) or {}).items():
    if name.startswith("PET"): element_type[pid] = st.pet_element_type
    elif name.startswith("PSA"): element_type[pid] = st.psa_element_type
    elif name.startswith("GLASS"): element_type[pid] = getattr(st, "glass_element_type", "Q4_COROTATIONAL_SRI")
    elif "STEEL" in name.upper() or "PLATE" in name.upper(): element_type[pid] = "Q4_COROTATIONAL"

nid_to_idx = mesh.node_id_to_index()
translation_bc_dofs, translation_bc_vals = [], []
prescribed_rotations = {}
for bc in result.boundaries:
    try:
        node_ids = [int(bc.nset)]
    except ValueError:
        ns = mesh.node_sets.get(bc.nset)
        node_ids = list(ns.node_ids) if ns else []
    for nid in node_ids:
        n_idx = nid_to_idx[nid]
        for dof_num in range(bc.dof1, bc.dof2 + 1):
            if dof_num in (1, 2):
                translation_bc_dofs.append(n_idx*2 + (dof_num-1))
                translation_bc_vals.append(bc.value)
            elif dof_num in (3, 6):
                prescribed_rotations[nid] = bc.value

def build(elem_jit, enable_new):
    s = DynamicSolver.from_config(
        mesh=mesh, material=result.materials, material_params=result.material_params,
        config=cfg, rho=result.solver_config.get("density", 1e-9),
        element_type=element_type, rbe2_constraints=result.rbe2_constraints,
        penalty_constraints=result.penalty_constraints, elem_jit=elem_jit, verbose=True,
    )
    s.enable_new_numba_elements = enable_new
    s.theta_penalty_k = st.theta_penalty_k
    s.sta_status = False
    if translation_bc_dofs:
        s.set_prescribed_dofs(np.array(translation_bc_dofs, dtype=np.int32),
                              bc_vals=np.array(translation_bc_vals, dtype=np.float64))
    return s

dt = cfg.drive.dt_init
t_next = dt
amp = smoothstep_amp(t_next, cfg.drive.t_total)
targets = {}
for idx, rbe in enumerate(result.rbe2_constraints):
    if rbe.master_id in prescribed_rotations:
        targets[idx] = prescribed_rotations[rbe.master_id] * amp
print("theta targets:", targets, " dt=", dt)

for label, ej, en in (("JAX", "jax", False), ("NUMBA-new", "numba", True)):
    s = build(ej, en)
    s.theta_targets = dict(targets)
    print(f"\n=== {label} ===")
    code = s.solve_step(dt)
    print(f"{label}: conv_code={code}")
