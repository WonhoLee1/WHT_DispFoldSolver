"""PARDISO vs SciPy residual shootout on 8x8 Scordelis-Lo."""
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import dispsolver.solver_shell.dynamic_shell as dshell
from dispsolver.solver_shell.dynamic_shell import DynamicSolverShell
from benchmark_element.benchmark_shell_mitc import generate_scordelis_lo_mesh

nodes, elements, bnds = generate_scordelis_lo_mesh(nx=8, ny=8)
solver = DynamicSolverShell(nodes=nodes, elements=elements, elem_type="MITC4+",
                            E=4.32e8, nu=0.0, thickness=0.25)
for nid in bnds["x_sym"]:
    solver.fix_dof(nid, 0, 0.0); solver.fix_dof(nid, 4, 0.0); solver.fix_dof(nid, 5, 0.0)
for nid in bnds["y_sym"]:
    solver.fix_dof(nid, 1, 0.0); solver.fix_dof(nid, 3, 0.0); solver.fix_dof(nid, 5, 0.0)
for nid in bnds["diaphragm"]:
    solver.fix_dof(nid, 0, 0.0); solver.fix_dof(nid, 2, 0.0)
f_ext = np.zeros(solver.num_dofs)
for elem in elements:
    pts = nodes[elem]
    area = 0.5 * np.linalg.norm(np.cross(pts[2] - pts[0], pts[3] - pts[1]))
    for nid in elem:
        f_ext[nid * 6 + 2] += -90.0 * area / 4.0

K = solver.assemble_stiffness_matrix()
fixed = np.array(sorted(solver.fixed_dofs.keys()))
alld = np.arange(solver.num_dofs)
free = np.ones(solver.num_dofs, bool)
free[fixed] = False
Kf = K[free, :][:, free]
print(f"HAS_PARDISO={dshell.HAS_PARDISO} free={free.sum()}/{solver.num_dofs}")
print(f"Kf diag min={Kf.diagonal().min():.3e} max={Kf.diagonal().max():.3e}")

u_sp = spsolve(Kf, f_ext[free])
r_sp = Kf @ u_sp - f_ext[free]
print(f"scipy: |r|={np.linalg.norm(r_sp):.3e}")
ev0 = bnds["eval_node"] * 6 + 2
# map global dof to free position
pos = np.zeros(solver.num_dofs, dtype=int)
pos[free] = np.arange(free.sum())
print(f"scipy: w_eval={u_sp[pos[ev0]]:.6f}")

if dshell.HAS_PARDISO:
    from pypardiso import spsolve as pp_spsolve
    u_pp = pp_spsolve(Kf, f_ext[free])
    r_pp = Kf @ u_pp - f_ext[free]
    print(f"pardiso: |r|={np.linalg.norm(r_pp):.3e} w_eval={u_pp[pos[ev0]]:.6f}")
