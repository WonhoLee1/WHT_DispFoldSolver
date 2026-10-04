"""
t2d2_numba.py
=============
Numba JIT parallel kernel for 2-node 2D Truss/Line/Beam element assembly.
Supports large-deformation corotational geometric non-linearity.
"""

import numpy as np
import numba
from numba import njit, prange

@njit(fastmath=True)
def compute_t2d2_element_numba(coords: np.ndarray, u_elem: np.ndarray, E_mod: float, A_area: float = 1.0):
    """Compute internal force vector (4,) and tangent stiffness (4, 4) for 2-node 2D element."""
    x1 = coords[0] + u_elem[0:2]
    x2 = coords[1] + u_elem[2:4]

    dX0 = coords[1] - coords[0]
    L0 = np.sqrt(dX0[0]*dX0[0] + dX0[1]*dX0[1])
    if L0 < 1e-12:
        L0 = 1e-12

    dx = x2 - x1
    L = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
    if L < 1e-12:
        L = 1e-12

    nx = dx[0] / L
    ny = dx[1] / L
    n = np.array([nx, ny], dtype=np.float64)

    strain = (L - L0) / L0
    F = A_area * E_mod * strain

    fe = np.array([-F * nx, -F * ny, F * nx, F * ny], dtype=np.float64)

    nnT = np.outer(n, n)
    I2 = np.eye(2, dtype=np.float64)

    K_axial = (A_area * E_mod / L0) * nnT
    K_geo = (F / L) * (I2 - nnT)
    K_local = K_axial + K_geo

    Ke = np.zeros((4, 4), dtype=np.float64)
    Ke[0:2, 0:2] = K_local
    Ke[0:2, 2:4] = -K_local
    Ke[2:4, 0:2] = -K_local
    Ke[2:4, 2:4] = K_local

    return fe, Ke

@njit(parallel=True, fastmath=True)
def assemble_mesh_t2d2_numba(
    node_coords: np.ndarray,
    elem_conn: np.ndarray,
    u: np.ndarray,
    elem_mat_types: np.ndarray,
    elem_props: np.ndarray,
    elem_sdvs: np.ndarray,
    dt: float,
    elem_controls: np.ndarray = None,
    elem_stress_init: np.ndarray = None,
):
    ne = elem_conn.shape[0]
    f_sub = np.zeros((ne, 4), dtype=np.float64)
    K_sub = np.zeros((ne, 4, 4), dtype=np.float64)
    err_out = 0

    for e in prange(ne):
        conn = elem_conn[e]
        coords = np.zeros((2, 2), dtype=np.float64)
        coords[0] = node_coords[conn[0]]
        coords[1] = node_coords[conn[1]]

        u_elem = np.zeros(4, dtype=np.float64)
        u_elem[0:2] = u[2 * conn[0]: 2 * conn[0] + 2]
        u_elem[2:4] = u[2 * conn[1]: 2 * conn[1] + 2]

        E_mod = elem_props[e, 0]
        if E_mod <= 0.0:
            E_mod = 200000.0
        A_area = elem_props[e, 2] if elem_props[e, 2] > 0.0 else 1.0

        fe, Ke = compute_t2d2_element_numba(coords, u_elem, E_mod, A_area)
        f_sub[e] = fe
        K_sub[e] = Ke

    return f_sub, K_sub, err_out
