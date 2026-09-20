"""MITC4+ shear-scale probe: constant shear states vs analytic Mindlin energy."""
import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from dispsolver.element_shell.mitc4_plus_numba import compute_mitc4_plus_stiffness

E, NU, T = 4.32e8, 0.0, 0.25
G = E / (2 * (1 + NU))
KAP = 5.0 / 6.0

sq = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0],
               [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])
K = compute_mitc4_plus_stiffness(sq, E, NU, T)
A = 4.0


def U(u):
    return 0.5 * float(u @ K @ u)


# (a) constant rotation tilt: th_y = c all nodes, rest 0 -> eng shear gamma = c
c = 0.01
u = np.zeros(24)
u[4::6] = c
Ua = U(u)
Ua_ref = 0.5 * KAP * G * T * c ** 2 * A
print(f"(a) rot-tilt: kernel={Ua:.6e} analytic={Ua_ref:.6e} ratio={Ua / Ua_ref:.4f}")

# (b) w-tilt: w = c*x -> eng shear gamma_xz = c
u2 = np.zeros(24)
u2[0::6] = c * sq[:, 0]
Ub = U(u2)
print(f"(b) w-tilt:   kernel={Ub:.6e} analytic={Ua_ref:.6e} ratio={Ub / Ua_ref:.4f}")

# (c) rigid body: U must be ~0
u3 = np.zeros(24)
u3[0::6] = 1.0
u3[2::6] = 0.5
print(f"(c) rigid: U={U(u3):.3e} (expect ~0)")

# (d) pure bending: th_y = k*x -> kappa_xx = k; U = 1/2*D*k^2*A, D=E t^3/12
k = 1e-3
u4 = np.zeros(24)
u4[4::6] = k * sq[:, 0]
D = E * T ** 3 / 12.0
Ud_ref = 0.5 * D * k ** 2 * A
print(f"(d) bending:  kernel={U(u4):.6e} analytic={Ud_ref:.6e} ratio={U(u4) / Ud_ref:.4f}")
