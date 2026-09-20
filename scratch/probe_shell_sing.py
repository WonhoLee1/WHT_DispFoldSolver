"""Shell solver singularity probe: symmetry + spectrum of assembled K."""
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from dispsolver.solver_shell.dynamic_shell import DynamicSolverShell

sq = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
               [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
els = np.array([[0, 1, 2, 3]])
s = DynamicSolverShell(nodes=sq, elements=els, elem_type="MITC4+", E=1e6, nu=0.0, thickness=0.1)
K = s.assemble_stiffness_matrix()
Kd = K.toarray()
print(f"sym err = {np.linalg.norm(Kd - Kd.T):.3e}")
ev = np.linalg.eigvalsh(Kd)
print("eigvals:", np.array2string(ev, precision=3, suppress_small=True))
print(f"n_zero(<1e-6*max) = {(ev < 1e-6 * ev[-1]).sum()} (expect 6 RBM)")
