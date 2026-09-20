"""Probe3: Tri1 vs Tri2 CR-frame split under bending + superposed rotation."""
import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from dispsolver.element2d.cr_wrapper_2d import (
    compute_cpe6m_cr_element_numba,
    _extract_corotational_frame_tri6_2d,
)

E, NU = 4000.0, 0.3
props = np.array([E, NU], dtype=np.float64)
sdvs = np.zeros((3, 7), dtype=np.float64)


def block_tris(dx=1.0, h=0.2):
    V1 = np.array([0.0, 0.0])
    V2 = np.array([dx, 0.0])
    V3 = np.array([dx, h])
    V4 = np.array([0.0, h])
    return (np.array([V1, V2, V3, (V1 + V2) / 2, (V2 + V3) / 2, (V1 + V3) / 2]),
            np.array([V1, V3, V4, (V1 + V3) / 2, (V3 + V4) / 2, (V4 + V1) / 2]))


def bend_u(coords, kappa=2e-3):
    y0 = coords[:, 1].mean()
    u = np.zeros_like(coords)
    u[:, 0] = -kappa * coords[:, 0] * (coords[:, 1] - y0)
    u[:, 1] = 0.5 * kappa * coords[:, 0] ** 2
    return u


tri1, tri2 = block_tris()
u1, u2 = bend_u(tri1), bend_u(tri2)
for rot in [0.0, 45.0, 90.0]:
    c, s = np.cos(np.radians(rot)), np.sin(np.radians(rot))
    R = np.array([[c, -s], [s, c]])
    print(f"superposed rotation {rot}:")
    for name, tri, u in [("Tri1", tri1, u1), ("Tri2", tri2, u2)]:
        X = tri @ R.T
        uu = ((tri + u) @ R.T - X).reshape(-1)
        Rf, uloc, _ = _extract_corotational_frame_tri6_2d(X, uu)
        fr, _, _ = compute_cpe6m_cr_element_numba(X, uu, 0, props, sdvs, 1.0, None)
        U = 0.5 * float(uu @ fr)
        ang = np.degrees(np.arctan2(-Rf[1, 0], Rf[0, 0]))
        print(f"  {name}: frame={ang:9.4f}deg  U={U:.6e}  |u_loc|={np.linalg.norm(uloc):.3e}")
