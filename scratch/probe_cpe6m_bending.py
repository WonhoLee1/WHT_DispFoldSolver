"""CPE6M bending-tax probe (element level, no solve).
P1: pure-bending energy CPE6 vs CPE6M (H1 core tax).
P2: rigid-rotation objectivity of CPE6M_CR Tri1 vs Tri2 orientation (H2 frame leak).
P3: HG alpha sweep on CPE6M core under bending (H3 HG tax).
"""
import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from dispsolver.element2d.cpe6_numba import compute_cpe6_element_numba
from dispsolver.element2d.cpe6m_numba import compute_cpe6m_element_numba
from dispsolver.element2d.cr_wrapper_2d import (
    compute_cpe6m_cr_element_numba,
    _extract_corotational_frame_tri6_2d,
)

E, NU = 4000.0, 0.3  # PET
props = np.array([E, NU], dtype=np.float64)
sdvs = np.zeros((3, 7), dtype=np.float64)
ctrl0 = np.zeros(8, dtype=np.float64)  # elem_controls[2]=0 -> kernel default alpha 0.05


def block_tris(dx=1.0, h=0.05):
    """One quad block split into 2 Tri6, Abaqus ordering [V1,V2,V3,M12,M23,M31]."""
    V1 = np.array([0.0, 0.0])
    V2 = np.array([dx, 0.0])
    V3 = np.array([dx, h])
    V4 = np.array([0.0, h])
    M1 = (V1 + V2) / 2
    M2 = (V2 + V3) / 2
    Md = (V1 + V3) / 2
    M3 = (V3 + V4) / 2
    M4 = (V4 + V1) / 2
    tri1 = np.array([V1, V2, V3, M1, M2, Md])  # bottom edge 0->1
    tri2 = np.array([V1, V3, V4, Md, M3, M4])  # diagonal edge 0->1
    return tri1, tri2


def bending_u(coords, kappa=1e-3):
    y0 = coords[:, 1].mean()
    x = coords[:, 0]
    yt = coords[:, 1] - y0
    u = np.zeros_like(coords)
    u[:, 0] = -kappa * x * yt
    u[:, 1] = 0.5 * kappa * x ** 2
    return u.reshape(-1)


def energy(fe, u):
    return 0.5 * float(u @ fe)


print("=== P1: pure-bending energy, CPE6 vs CPE6M (kappa=1e-3, dx=1, h=0.05) ===")
tri1, tri2 = block_tris()
for name, tri in [("Tri1", tri1), ("Tri2", tri2)]:
    u = bending_u(tri)
    f6, K6, _ = compute_cpe6_element_numba(tri, u, 0, props, sdvs, 1.0, None)
    fM, KM, _ = compute_cpe6m_element_numba(tri, u, 0, props, sdvs, 1.0, None)
    U6, UM = energy(f6, u), energy(fM, u)
    print(f"  {name}: U_CPE6={U6:.6e}  U_CPE6M={UM:.6e}  ratio={UM / U6:.4f}")

print("=== P3: HG alpha sweep on CPE6M core (Tri1, bending) ===")
u = bending_u(tri1)
for a in [0.0, 0.05, 0.2]:
    c = np.zeros(8)
    c[2] = a
    fM, _, _ = compute_cpe6m_element_numba(tri1, u, 0, props, sdvs, 1.0, c)
    print(f"  alpha={a:.2f}: U={energy(fM, u):.6e}")

print("=== P2: rigid rotation objectivity, CPE6M_CR (theta=30deg) ===")
for th in [30.0, 90.0]:
    print(f"  theta={th}:")
    R = np.array([[np.cos(np.radians(th)), -np.sin(np.radians(th))],
                  [np.sin(np.radians(th)), np.cos(np.radians(th))]])
    for name, tri in [("Tri1", tri1), ("Tri2", tri2)]:
        u = (tri @ R.T - tri).reshape(-1)
        fC, _, _ = compute_cpe6m_cr_element_numba(tri, u, 0, props, sdvs, 1.0, None)
        Rc, uloc, _ = _extract_corotational_frame_tri6_2d(tri, u)
        ang = np.degrees(np.arctan2(-Rc[1, 0], Rc[0, 0]))
        print(f"    {name}: |f|={np.linalg.norm(fC):.3e}  frame_ang={ang:.4f}  |u_loc|={np.linalg.norm(uloc):.3e}")
