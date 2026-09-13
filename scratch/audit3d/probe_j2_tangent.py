import numpy as np
import sys
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver")
sys.path.insert(0, r"D:\PythonCodeStudy\WHT_DispFoldSolver\scratch\audit3d")
import _bootstrap
_bootstrap.load_full_element3d_package()

from dispsolver.material3d.numba_materials import material_dispatch_3d, MAT_J2_PLASTICITY

E, nu, sigma_y0, H = 1000.0, 0.3, 1.0, 100.0
props = np.array([E, nu, sigma_y0, H], dtype=np.float64)

# Push well past yield with a uniaxial-ish strain state
E_voigt0 = np.array([0.05, -0.01, -0.01, 0.0, 0.0, 0.0])
sdv_prev = np.zeros(7)
F_dummy = np.eye(3)

S0, C0, sdv0, err0 = material_dispatch_3d(MAT_J2_PLASTICITY, props, sdv_prev, E_voigt0, F_dummy, 1.0, 1.0)
print("f_trial > 0 (plastic)?  eq_plastic_strain after step =", sdv0[6], " (nonzero means yes, plastic branch taken)")

# FD Jacobian of S_voigt(E_voigt) at this point (holding sdv_prev fixed --
# i.e. this is the CONSISTENT/algorithmic tangent an implicit return-map
# element should report as C_tangent for THIS step, given fixed history)
n = 6
C_fd = np.zeros((6, 6))
h = 1e-6
for j in range(n):
    Ep = E_voigt0.copy(); Ep[j] += h
    Em = E_voigt0.copy(); Em[j] -= h
    Sp, _, _, _ = material_dispatch_3d(MAT_J2_PLASTICITY, props, sdv_prev, Ep, F_dummy, 1.0, 1.0)
    Sm, _, _, _ = material_dispatch_3d(MAT_J2_PLASTICITY, props, sdv_prev, Em, F_dummy, 1.0, 1.0)
    C_fd[:, j] = (Sp - Sm) / (2*h)

print()
print("=== 3D J2 plasticity: returned C_mat vs FD Jacobian of its own S_voigt(E_voigt) ===")
print("Returned C_mat (elastic tangent, diag) =", np.diag(C0))
print("FD Jacobian C_fd (diag)                 =", np.diag(C_fd))
print("max|C0 - C_fd|                          =", np.max(np.abs(C0 - C_fd)))
print("max|C_fd|                               =", np.max(np.abs(C_fd)))
print("relative error                          =", np.max(np.abs(C0-C_fd))/max(np.max(np.abs(C_fd)),1e-300))

# Also check the purely elastic case (f_trial<=0) to confirm C_mat IS correct there
E_voigt_elastic = np.array([0.0001, -0.00003, -0.00003, 0.0, 0.0, 0.0])
S_el, C_el, sdv_el, err_el = material_dispatch_3d(MAT_J2_PLASTICITY, props, sdv_prev, E_voigt_elastic, F_dummy, 1.0, 1.0)
print()
print("Sanity check, still-elastic state: eq_plastic_strain =", sdv_el[6], "(expect 0.0, confirms this state is elastic)")
