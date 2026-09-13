"""
C8-style check for the 3D C3D8_EAS (C3D8I) Numba TL kernel:
Is the returned condensed tangent K_elem actually the Jacobian of the
kernel's own f_int(u), by finite difference, on a distorted (non axis-
aligned) reference?
"""
import numpy as np
import sys
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver")
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver\scratch\audit3d")
import _bootstrap

eas_tl = _bootstrap.load_element3d_module("c3d8_eas_tl_numba")
from dispsolver.material3d.numba_materials import MAT_CUSTOM_ELASTIC

np.random.seed(1)
coords0 = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1],
], dtype=np.float64)
coords0 = coords0 + 0.15 * np.array([
    [0.3,-0.2,0.1],[-0.1,0.25,-0.15],[0.2,-0.1,0.05],[-0.25,0.1,0.2],
    [0.1,0.05,-0.3],[-0.2,-0.15,0.25],[0.05,0.2,-0.1],[-0.15,-0.05,0.3],
])

E, nu = 1000.0, 0.3
lam = E*nu/((1+nu)*(1-2*nu)); mu = E/(2*(1+nu))
C_mat = np.zeros((6,6))
C_mat[0,0]=C_mat[1,1]=C_mat[2,2]=lam+2*mu
C_mat[0,1]=C_mat[0,2]=C_mat[1,0]=C_mat[1,2]=C_mat[2,0]=C_mat[2,1]=lam
C_mat[3,3]=C_mat[4,4]=C_mat[5,5]=mu
props = np.zeros(36); props[:] = C_mat.ravel()
sdvs = np.zeros((8,0))

# Moderate displacement so the element is meaningfully deformed (not just
# probing the linear limit), matching the spirit of AGENTS.md C8 (rotated
# / distorted reference, finite deformation, not just small-strain).
u0 = 0.05 * np.random.randn(24)

def f_of_u(u):
    K, f, err = eas_tl._compute_c3d8_eas_tl_element_umat_numba(
        coords0, u, MAT_CUSTOM_ELASTIC, props, sdvs.copy(), 1.0
    )
    return f

K_analytic, f0, err = eas_tl._compute_c3d8_eas_tl_element_umat_numba(
    coords0, u0, MAT_CUSTOM_ELASTIC, props, sdvs.copy(), 1.0
)

n = 24
K_fd = np.zeros((n, n))
h = 1e-6 * max(np.max(np.abs(u0)), 1.0)
for j in range(n):
    up = u0.copy(); up[j] += h
    um = u0.copy(); um[j] -= h
    fp = f_of_u(up)
    fm = f_of_u(um)
    K_fd[:, j] = (fp - fm) / (2*h)

diff = K_analytic - K_fd
rel_err = np.max(np.abs(diff)) / max(np.max(np.abs(K_fd)), 1e-300)
print("=== C8-style FD Jacobian check: C3D8_EAS(TL) condensed K vs FD Jacobian of its own f_int ===")
print("max|K_analytic - K_fd|        =", np.max(np.abs(diff)))
print("max|K_fd|                     =", np.max(np.abs(K_fd)))
print("relative error                =", rel_err)
print("K_analytic symmetric?         =", np.max(np.abs(K_analytic - K_analytic.T)))
print("K_fd symmetric? (expect ~0)   =", np.max(np.abs(K_fd - K_fd.T)))

# Also compare against the UNCONDENSED K_uu (rebuild manually is complex;
# instead compare condensed K vs FD, and separately report how far the
# condensed K is from K_fd as a fraction -- if K_uu (uncondensed) equals
# K_fd (since f_int truly never depends on alpha), that's the smoking gun
# that condensation subtracts a term that does NOT belong in the Jacobian.
print()
print("If K_analytic (condensed) != K_fd but is close in magnitude, the kernel")
print("returns a tangent that is NOT the Jacobian of its own residual --")
print("Newton will use a systematically wrong (softer) stiffness.")
