"""Scordelis-Lo triage: reaction balance + flat clamped plate vs analytic."""
import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from benchmark_element.benchmark_shell_mitc import generate_scordelis_lo_mesh
from dispsolver.solver_shell.dynamic_shell import DynamicSolverShell
from dispsolver.element_shell.mitc4_plus_numba import compute_mitc4_plus_stiffness

# 1. Reaction balance on 8x8 MITC4 Scordelis-Lo
nodes, elements, bnds = generate_scordelis_lo_mesh(nx=8, ny=8)
E, nu, t, q = 4.32e8, 0.0, 0.25, 90.0
solver = DynamicSolverShell(nodes=nodes, elements=elements, elem_type="MITC4+",
                            E=E, nu=nu, thickness=t)
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
        f_ext[nid * 6 + 2] += -q * area / 4.0
u = solver.solve_static(f_ext)
K = solver.assemble_stiffness_matrix()
R = K @ u - f_ext
print(f"applied Fz total = {f_ext[2::6].sum():.4e}")
print(f"|R_free| = {np.linalg.norm(R):.3e} (expect ~0)")
rz = sum((K @ u)[nid * 6 + 2] for nid in bnds["diaphragm"])
print(f"diaphragm Rz sum = {rz:.4e} vs applied {f_ext[2::6].sum():.4e}")
print(f"eval w = {u[bnds['eval_node'] * 6 + 2]:.6f} (ref -0.3024)")

# 2. Flat clamped square plate 2x2, uniform q=1, E=1e6, nu=0, t=0.01, 4x4 mesh
print("--- flat plate ---")
n = 4
L = 2.0
xs = np.linspace(-1, 1, n + 1)
nd = {}
nodes2 = []
for j in range(n + 1):
    for i in range(n + 1):
        nd[(i, j)] = len(nodes2)
        nodes2.append([xs[i], xs[j], 0.0])
nodes2 = np.array(nodes2)
els2 = []
for j in range(n):
    for i in range(n):
        els2.append([nd[(i, j)], nd[(i + 1, j)], nd[(i + 1, j + 1)], nd[(i, j + 1)]])
els2 = np.array(els2)
E2, t2, q2 = 1e6, 0.01, 1.0
s2 = DynamicSolverShell(nodes=nodes2, elements=els2, elem_type="MITC4+", E=E2, nu=0.0, thickness=t2)
for (i, j), nid in nd.items():
    if i == 0 or j == 0 or i == n or j == n:
        for d in range(6):
            s2.fix_dof(nid, d, 0.0)
f2 = np.zeros(s2.num_dofs)
a = (L / n) ** 2
f2[2::6] = -q2 * a
u2 = s2.solve_static(f2)
wc = u2[nd[(n // 2, n // 2)] * 6 + 2]
# clamped square uniform load, nu=0: w = 0.00126 q a^4 / D (Timoshenko), a=half-side=1
D2 = E2 * t2 ** 3 / 12.0
w_ref = -0.00126 * q2 * 1.0 ** 4 / D2
print(f"center w = {wc:.6e} ref = {w_ref:.6e} ratio = {wc / w_ref:.4f}")
