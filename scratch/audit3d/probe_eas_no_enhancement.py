"""
Probe: does the 3D C3D8_EAS (C3D8I) kernel's internal force / stress actually
depend on the enhanced-strain parameters alpha, or is alpha computed and discarded?

Method: build a single distorted hex element, apply a displacement field that
is NOT a state of pure homogeneous strain (so EAS should matter), and:
  1. Compute f_int, K via the production Numba TL EAS kernel
     (_compute_c3d8_eas_tl_element_umat_numba).
  2. Compute f_int, K via the plain compatible C3D8 kernel
     (compute_c3d8_element_numba) with an equivalent linear-elastic C_mat.
  3. Compare f_int bit-for-bit (should be identical if EAS never enhances strain).
  4. Compare K (should differ, since K_ua/K_aa correction is still subtracted).
  5. Do the same comparison via the JAX Hexa8EASElement class.
"""
import numpy as np
import sys
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver")
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver\scratch\audit3d")
import _bootstrap  # noqa: F401

_eas_tl = _bootstrap.load_element3d_module("c3d8_eas_tl_numba")
_c3d8 = _bootstrap.load_element3d_module("c3d8_numba")
_compute_c3d8_eas_tl_element_umat_numba = _eas_tl._compute_c3d8_eas_tl_element_umat_numba
compute_c3d8_element_numba = _c3d8.compute_c3d8_element_numba

from dispsolver.element3d.c3d8_eas_jax import Hexa8EASElement
from dispsolver.element3d.base3d import QuadraturePointState3D
from dispsolver.material3d.numba_materials import MAT_CUSTOM_ELASTIC

np.random.seed(0)

# Unit cube reference coords (standard C3D8 node order)
coords0 = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1],
], dtype=np.float64)

# Distort it a bit (non-parallelepiped) so bending/shear modes actually matter
coords0 = coords0 + 0.15 * np.array([
    [0.3,-0.2,0.1],[-0.1,0.25,-0.15],[0.2,-0.1,0.05],[-0.25,0.1,0.2],
    [0.1,0.05,-0.3],[-0.2,-0.15,0.25],[0.05,0.2,-0.1],[-0.15,-0.05,0.3],
])

# Isotropic linear elastic C_mat, E=1000, nu=0.3
E, nu = 1000.0, 0.3
lam = E*nu/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))
C_mat = np.zeros((6,6))
C_mat[0,0]=C_mat[1,1]=C_mat[2,2]=lam+2*mu
C_mat[0,1]=C_mat[0,2]=C_mat[1,0]=C_mat[1,2]=C_mat[2,0]=C_mat[2,1]=lam
C_mat[3,3]=C_mat[4,4]=C_mat[5,5]=mu

props = np.zeros(36)
props[:] = C_mat.ravel()

# A generic small, non-homogeneous nodal displacement field (random, small so
# TL-vs-linear differences are negligible -> isolates the EAS-vs-compatible gap)
u_elem = 1e-4 * np.random.randn(24)

sdvs_dummy = np.zeros((8,0))

# --- Numba TL EAS kernel ---
K_eas, f_eas, err_eas = _compute_c3d8_eas_tl_element_umat_numba(
    coords0, u_elem, MAT_CUSTOM_ELASTIC, props, sdvs_dummy, 1.0
)

# --- Plain compatible C3D8 kernel ---
K_c3d8, f_c3d8 = compute_c3d8_element_numba(coords0, u_elem, C_mat)

print("=== Numba: C3D8_EAS(TL) vs plain C3D8 ===")
print("max|f_eas - f_c3d8|          =", np.max(np.abs(f_eas - f_c3d8)))
print("max|f_eas|                   =", np.max(np.abs(f_eas)))
print("relative f_int difference    =", np.max(np.abs(f_eas - f_c3d8)) / max(np.max(np.abs(f_eas)), 1e-300))
print("max|K_eas - K_c3d8|           =", np.max(np.abs(K_eas - K_c3d8)))
print("max|K_eas|                    =", np.max(np.abs(K_eas)))
print("relative K difference        =", np.max(np.abs(K_eas - K_c3d8)) / max(np.max(np.abs(K_eas)), 1e-300))
print("K symmetric?                 =", np.max(np.abs(K_eas - K_eas.T)))

# --- JAX EAS element class ---
elem = Hexa8EASElement(num_eas_modes=9)
states = [QuadraturePointState3D.create_initial() for _ in range(8)]
K_jax, f_jax, alpha_opt = elem.compute_element_stiffness_and_force(coords0, u_elem, C_mat, states)

print()
print("=== JAX: Hexa8EASElement vs plain C3D8 ===")
print("max|f_jax - f_c3d8|           =", np.max(np.abs(f_jax - f_c3d8)))
print("alpha_opt (should be small, unused nonzero est.) =", np.max(np.abs(alpha_opt)))
print("max|K_jax - K_eas(numba)|     =", np.max(np.abs(K_jax - K_eas)))
print("max|f_jax - f_eas(numba)|     =", np.max(np.abs(f_jax - f_eas)))
