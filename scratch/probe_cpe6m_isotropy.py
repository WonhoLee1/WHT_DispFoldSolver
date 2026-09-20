"""Probe2: global-axis isotropy (reference + displacement rotated together).
CPE6M_CR energy/moment must be rotation-independent. Controls: CPE4I_CR, CPE4H_CR.
"""
import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from dispsolver.element2d.cr_wrapper_2d import (
    compute_cpe6m_cr_element_numba,
    compute_cpe4i_cr_element_numba,
    compute_cpe4h_cr_element_numba,
)

E, NU = 4000.0, 0.3
props = np.array([E, NU], dtype=np.float64)
sdvs = np.zeros((3, 7), dtype=np.float64)


def quad_block(dx=1.0, h=0.2):
    return np.array([[0, 0], [dx, 0], [dx, h], [0, h]], dtype=np.float64)


def tri_block(dx=1.0, h=0.2):
    V1 = np.array([0.0, 0.0])
    V2 = np.array([dx, 0.0])
    V3 = np.array([dx, h])
    M1 = (V1 + V2) / 2
    M2 = (V2 + V3) / 2
    Md = (V1 + V3) / 2
    return np.array([V1, V2, V3, M1, M2, Md])


def bend_u(coords, kappa=2e-3):
    y0 = coords[:, 1].mean()
    x = coords[:, 0]
    yt = coords[:, 1] - y0
    u = np.zeros_like(coords)
    u[:, 0] = -kappa * x * yt
    u[:, 1] = 0.5 * kappa * x ** 2
    return u


def rot(th):
    c, s = np.cos(np.radians(th)), np.sin(np.radians(th))
    return np.array([[c, -s], [s, c]])


kernels = {
    "CPE6M_CR": (compute_cpe6m_cr_element_numba, tri_block()),
    "CPE4I_CR": (compute_cpe4i_cr_element_numba, quad_block()),
    "CPE4H_CR": (compute_cpe4h_cr_element_numba, quad_block()),
}

for name, (ker, X0) in kernels.items():
    u0 = bend_u(X0)
    f_ref, _, _ = ker(X0, u0.reshape(-1), 0, props, sdvs, 1.0, None)
    U_ref = 0.5 * float(u0.reshape(-1) @ f_ref)
    print(f"{name}: U_ref={U_ref:.6e}")
    for th in [10.0, 30.0, 45.0, 90.0]:
        R = rot(th)
        X = X0 @ R.T
        u = ((X0 + u0) @ R.T - X).reshape(-1)
        f, _, _ = ker(X, u, 0, props, sdvs, 1.0, None)
        U = 0.5 * float(u @ f)
        print(f"  th={th:5.1f}: U/U_ref={U / U_ref:.6f}  |f|/|f_ref|={np.linalg.norm(f) / np.linalg.norm(f_ref):.6f}")
