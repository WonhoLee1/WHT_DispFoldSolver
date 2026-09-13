import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dispsolver.io import read_abaqus_input
from dispsolver.solver import DynamicSolver
from dispsolver.fold_model_config import make_teardrop_config, smoothstep_amp
from dispsolver.element.q4_visco_eas_jax import compute_single_hybrid as JAXH
from dispsolver.element.q4_visco_hybrid_up_numba import compute_visco_hybrid_up_single_numba as NBH

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
    try: node_ids = [int(bc.nset)]
    except ValueError:
        ns = mesh.node_sets.get(bc.nset); node_ids = list(ns.node_ids) if ns else []
    for nid in node_ids:
        n_idx = nid_to_idx[nid]
        for dof_num in range(bc.dof1, bc.dof2 + 1):
            if dof_num in (1,2):
                translation_bc_dofs.append(n_idx*2+(dof_num-1)); translation_bc_vals.append(bc.value)
            elif dof_num in (3,6):
                prescribed_rotations[nid] = bc.value

s = DynamicSolver.from_config(
    mesh=mesh, material=result.materials, material_params=result.material_params,
    config=cfg, rho=result.solver_config.get("density", 1e-9),
    element_type=element_type, rbe2_constraints=result.rbe2_constraints,
    penalty_constraints=result.penalty_constraints, elem_jit="jax", verbose=False,
)
s.theta_penalty_k = st.theta_penalty_k
s.sta_status = False
if translation_bc_dofs:
    s.set_prescribed_dofs(np.array(translation_bc_dofs, dtype=np.int32),
                          bc_vals=np.array(translation_bc_vals, dtype=np.float64))

dt = cfg.drive.dt_init
t_next = dt
amp = smoothstep_amp(t_next, cfg.drive.t_total)
targets = {}
for idx, rbe in enumerate(result.rbe2_constraints):
    if rbe.master_id in prescribed_rotations:
        targets[idx] = prescribed_rotations[rbe.master_id] * amp
s.theta_targets = targets

# Drive the solver to exactly the point right before the first _assemble call
# by replicating _solve_step_impl's predictor + BC + RBE2 pre-projection.
u_k = s.u.copy()  # zeros
if s.bc_dofs.size:
    u_k[s.bc_dofs] = s.bc_vals
n_extra_regular = s.n_extra - len(s.rbe2_constraints)
u_ext_k = s.u_extra.copy()
for idx, th in targets.items():
    u_ext_k[n_extra_regular + idx] = th
# RBE2 slave pre-projection (mirror dynamic.py's pre-Newton block)
if s.rbe2_constraints:
    for c_idx, c in enumerate(s.condensation_mgr.constraints):
        theta = u_ext_k[s.condensation_mgr.constraint_extra_offsets[c_idx]]
        cost, sint = np.cos(theta), np.sin(theta)
        m_idx = s.nid_to_idx[c.master_id]
        x_m, y_m = s.coords[m_idx]
        u_mx, u_my = u_k[2*m_idx], u_k[2*m_idx+1]
        for sid in c.slave_ids:
            s_idx = s.nid_to_idx[sid]
            x_sc, y_sc = s.coords[s_idx]
            dx, dy = x_sc - x_m, y_sc - y_m
            u_k[2*s_idx]   = u_mx + (cost-1.0)*dx - sint*dy
            u_k[2*s_idx+1] = u_my + sint*dx + (cost-1.0)*dy

print("max|u_k| =", np.max(np.abs(u_k)))

# --- probe element 9000 (pid=7, CPE4H) directly ---
eid = 9000
elem = mesh.elements[eid]
nids = elem.node_ids
idxs = [nid_to_idx[n] for n in nids]
coords = np.array([mesh.nodes[n].xy if hasattr(mesh.nodes[n],'xy') else [mesh.nodes[n].x, mesh.nodes[n].y] for n in nids])
u_elem = np.concatenate([[u_k[2*i], u_k[2*i+1]] for i in idxs])
print("elem", eid, "pid", elem.pid, "nodes", nids)
print("coords:\n", coords)
print("u_elem:", u_elem)

vmat = s.materials[elem.pid].material
base_name, bparams, kappa = vmat.simo_fs_args(s.materials[elem.pid].params)
NV = 6*(len(vmat.g_i)+1)
state = np.zeros((4, NV))
eye4 = np.tile(np.eye(2), (4,1,1))

fj, Kj, qj, sj, fnj = JAXH(jnp.asarray(coords), jnp.asarray(u_elem), jnp.zeros(5), jnp.asarray(state),
                          float(kappa), jnp.asarray(bparams), jnp.asarray(vmat.g_i), jnp.asarray(vmat.tau_i),
                          float(vmat.g_inf), 1.0, s._elem_thickness[s.elem_ids.tolist().index(eid) if hasattr(s.elem_ids,'tolist') else 0],
                          base=base_name, use_eas=False)
code = {"neohookean":0,"yeoh":1,"arruda":2}[base_name]
fn, Kn, qn, sn = NBH(code, coords, u_elem, np.zeros(5), state, float(kappa),
                     np.asarray(bparams,dtype=np.float64), np.asarray(vmat.g_i,dtype=np.float64),
                     np.asarray(vmat.tau_i,dtype=np.float64), float(vmat.g_inf), 1.0, 1.0, eye4)
print("JAX  |f|=", float(jnp.linalg.norm(fj)), " p=", float(qj[4]))
print("NUMBA|f|=", np.linalg.norm(fn), " p=", qn[4])
