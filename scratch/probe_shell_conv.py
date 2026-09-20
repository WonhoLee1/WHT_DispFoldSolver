"""Flat simply-supported square plate convergence + correct w-shear scale."""
import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from dispsolver.solver_shell.dynamic_shell import DynamicSolverShell
from dispsolver.element_shell.mitc4_plus_numba import compute_mitc4_plus_stiffness

E2, t2, q2, L = 1e6, 0.01, 1.0, 2.0
D2 = E2 * t2 ** 3 / 12.0
# simply-supported square side L=2, uniform q, nu=0: Timoshenko alpha~0.00406 (nu=0.3);
# use series value for nu=0 approx 0.00406*1.0 -> compare TREND (must converge flat)
w_ref_ss = -0.00406 * q2 * L ** 4 / D2
print(f"ss ref ~ {w_ref_ss:.6e} (trend check)")

for n in [4, 8, 16]:
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
    s2 = DynamicSolverShell(nodes=nodes2, elements=els2, elem_type="MITC4+", E=E2, nu=0.0, thickness=t2)
    for (i, j), nid in nd.items():
        if i == 0 or j == 0 or i == n or j == n:
            s2.fix_dof(nid, 2, 0.0)  # wz only: simply supported
    f2 = np.zeros(s2.num_dofs)
    f2[2::6] = -q2 * (L / n) ** 2
    u2 = s2.solve_static(f2)
    ci = nd[(n // 2, n // 2)] if n % 2 == 0 else None
    # center node exists for even n
    wc = u2[ci * 6 + 2]
    print(f"n={n:3d}: center w = {wc:.6e} ratio = {wc / w_ref_ss:.4f}", flush=True)

# correct w-shear probe: w = c*x, rest 0 -> eng shear gamma = c
sq = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
K = compute_mitc4_plus_stiffness(sq, 4.32e8, 0.0, 0.25)
c = 0.01
u = np.zeros(24)
u[2::6] = c * sq[:, 0]
Uw = 0.5 * float(u @ K @ u)
Uw_ref = 0.5 * (5.0 / 6.0) * (4.32e8 / 2) * 0.25 * c ** 2 * 4.0
print(f"w-shear: kernel={Uw:.6e} analytic={Uw_ref:.6e} ratio={Uw / Uw_ref:.4f}")
