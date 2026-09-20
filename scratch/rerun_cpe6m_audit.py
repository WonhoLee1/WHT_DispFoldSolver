"""Instrumented CPE6M rollup rerun: per-step residual/M/iters audit (no JSON overwrite)."""
import sys, os
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
os.environ.setdefault("ROLLUP_TOL", "1e-5")

from benchmark_element.benchmark_pure_moment_rollup import (
    run_rollup_2d, make_2d_mesh, L_BEAM, T_PET, T_PSA, PID_PET, PID_PSA, E_PET, NU_PET, E_PSA, NU_PSA,
    THETA_MAX, Z_MID, compute_roll_up_displacements,
)
from dispsolver.mesh2d.conformal_cpe6m_cpe4h_builder import create_5layer_cpe6m_cpe4h_conformal_mesh
from dispsolver.solver2d.dynamic2d import DynamicSolver2D

CASE = os.environ.get("AUDIT_CASE", "CPE6M")
if CASE == "CPE6M":
    mesh, sets = create_5layer_cpe6m_cpe4h_conformal_mesh(
        length=L_BEAM, t_pet=T_PET, t_psa=T_PSA, nx=40, pid_pet=PID_PET, pid_psa=PID_PSA)
    root_nodes, tip_nodes = sets["LEFT_NODES"], sets["RIGHT_NODES"]
else:
    mesh, root_nodes, tip_nodes, _grid = make_2d_mesh("CPE4I", "CPE4H")
print(f"[AUDIT] case={CASE}", flush=True)
materials = {PID_PET: {"E": E_PET, "nu": NU_PET}, PID_PSA: {"E": E_PSA, "nu": NU_PSA}}
solver = DynamicSolver2D(mesh, materials=materials, nlgeom=True)
for nid in root_nodes:
    solver.fix_dof(nid, 0, 0.0)
    solver.fix_dof(nid, 1, 0.0)
nid_map = mesh.node_id_to_index()
n = 25
print(f"{'s':>3} {'theta':>7} {'conv':>5} {'it':>3} {'r_free':>10} {'M_root':>10} {'M_tip':>10} {'U_inc':>10}")
u_prev = np.zeros(solver.num_dofs)
f_prev = np.zeros(solver.num_dofs)
for s in range(1, n + 1):
    frac = s / n
    ramp = frac * frac * (3.0 - 2.0 * frac)
    th = ramp * THETA_MAX
    for nid in tip_nodes:
        node = mesh.nodes[nid]
        ux_t, uy_t = compute_roll_up_displacements(node.y, th)
        solver.fix_dof(nid, 0, ux_t)
        solver.fix_dof(nid, 1, uy_t)
    conv, it = solver.solve_step(dt=1.0, max_iters=35, tol=1e-5)
    K_g, f = solver.assemble_system(solver.u, dt=1.0)
    free = np.ones(solver.num_dofs, bool)
    free[list(solver.fixed_dofs.keys())] = False
    r = float(np.linalg.norm(f[free]))
    rx = solver.compute_section_reactions(root_nodes, center_pt=(0.0, Z_MID))
    txm = float(np.mean([mesh.nodes[x].x + solver.u[2 * nid_map[x]] for x in tip_nodes]))
    tym = float(np.mean([mesh.nodes[x].y + solver.u[2 * nid_map[x] + 1] for x in tip_nodes]))
    tx = solver.compute_section_reactions(tip_nodes, center_pt=(txm, tym))
    dU = 0.5 * float(np.dot(f_prev + f, solver.u - u_prev)) if conv else float("nan")
    u_prev, f_prev = solver.u.copy(), np.asarray(f).copy()
    print(f"{s:3d} {np.degrees(th):7.1f} {str(conv):>5} {it:3d} {r:10.3e} {rx['Mz']:10.5f} {tx['Mz']:10.5f} {dU:10.5f}", flush=True)
