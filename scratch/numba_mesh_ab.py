"""Mesh-level A/B: JAX vs new-Numba assembly on the real model, per pid."""
import numpy as np
from dispsolver.io import read_abaqus_input
from dispsolver.solver import DynamicSolver
from dispsolver.fold_model_config import make_teardrop_config

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

def build(elem_jit, enable_new):
    s = DynamicSolver.from_config(
        mesh=mesh, material=result.materials, material_params=result.material_params,
        config=cfg, rho=result.solver_config.get("density", 1e-9),
        element_type=element_type, rbe2_constraints=result.rbe2_constraints,
        penalty_constraints=result.penalty_constraints, elem_jit=elem_jit, verbose=False,
    )
    s.enable_new_numba_elements = enable_new
    return s

s_jax = build("jax", False)
s_nb  = build("numba", True)

u0 = np.zeros(s_jax.n_dofs)
f_jax, K_jax, _ = s_jax._assemble(u0, 1.0)
f_nb,  K_nb,  _ = s_nb._assemble(u0, 1.0)

print("n_dofs:", s_jax.n_dofs, " |f_jax|=", np.linalg.norm(f_jax), " |f_nb|=", np.linalg.norm(f_nb))

# per-pid breakdown using element->pid map
nid_to_idx = mesh.node_id_to_index()
pid_of_elem = {}
for eid, elem in mesh.elements.items():
    pid_of_elem[eid] = elem.pid

elem_ids = s_jax.elem_ids
pids_arr = np.array([pid_of_elem[eid] for eid in elem_ids])

for pid in sorted(set(pids_arr.tolist())):
    mask = pids_arr == pid
    idxs = np.where(mask)[0]
    dof_set = set()
    for ei in idxs:
        eid = elem_ids[ei]
        for nid in mesh.elements[eid].node_ids:
            k = nid_to_idx[nid]
            dof_set.add(2*k); dof_set.add(2*k+1)
    dofs = np.array(sorted(dof_set))
    diff = np.linalg.norm(f_jax[dofs]-f_nb[dofs])
    print(f"pid={pid:3d} n_elem={len(idxs):5d} et={element_type.get(pid,'?'):20s} |f_jax|={np.linalg.norm(f_jax[dofs]):.4e} |f_nb|={np.linalg.norm(f_nb[dofs]):.4e} |diff|={diff:.4e}")
