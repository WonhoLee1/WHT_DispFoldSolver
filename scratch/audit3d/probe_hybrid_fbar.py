import numpy as np
import sys
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver")
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver\scratch\audit3d")
import _bootstrap

hybrid = _bootstrap.load_element3d_module("c3d8_hybrid_numba")
fbar = _bootstrap.load_element3d_module("c3d8_fbar_tl_numba")
c3d8 = _bootstrap.load_element3d_module("c3d8_numba")
from dispsolver.material3d.numba_materials import MAT_CUSTOM_ELASTIC, MAT_J2_PLASTICITY, MAT_HYPERELASTIC_NEOHOOKEAN

np.random.seed(2)
coords0 = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1],
], dtype=np.float64)
coords0 = coords0 + 0.15 * np.array([
    [0.3,-0.2,0.1],[-0.1,0.25,-0.15],[0.2,-0.1,0.05],[-0.25,0.1,0.2],
    [0.1,0.05,-0.3],[-0.2,-0.15,0.25],[0.05,0.2,-0.1],[-0.15,-0.05,0.3],
])

E, nu = 1000.0, 0.3
mu = E/(2*(1+nu)); K = E/(3*(1-2*nu))
props = np.zeros(36)
props[0] = mu
props[1] = K
sdvs = np.zeros((8,0))
u_elem = 1e-4*np.random.randn(24)

print("=== Attempting compute_c3d8_hybrid_element_umat_numba call (default stress_init) ===")
try:
    f, K_, err = hybrid.compute_c3d8_hybrid_element_umat_numba(
        coords0, u_elem, MAT_CUSTOM_ELASTIC, props, sdvs, 1.0
    )
    print("SUCCESS: err=", err, "max|f|=", np.max(np.abs(f)))
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:2000])
