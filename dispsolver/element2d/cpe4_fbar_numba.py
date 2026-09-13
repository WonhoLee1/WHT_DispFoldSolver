"""
cpe4_fbar_numba.py
==================
Numba JIT accelerated 4-node Multiplicative F-bar Quadrilateral (CPE4_FBAR).
de Souza Neto, Peric & Owen (1996) F-bar centroid volumetric projection.
Equivalents: Abaqus CPE4 (F-bar option) / Large Strain Solid Quad.
"""

import numpy as np
from numba import njit, prange

_GP1D = 1.0 / np.sqrt(3.0)
_GPS = np.array([
    [-_GP1D, -_GP1D],
    [ _GP1D, -_GP1D],
    [ _GP1D,  _GP1D],
    [-_GP1D,  _GP1D],
], dtype=np.float64)
_WTS = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)


@njit(fastmath=True)
def _shape_derivs_cpe4_fbar(xi: float, eta: float) -> np.ndarray:
    dN = np.zeros((2, 4), dtype=np.float64)
    dN[0, 0] = -0.25 * (1.0 - eta)
    dN[0, 1] =  0.25 * (1.0 - eta)
    dN[0, 2] =  0.25 * (1.0 + eta)
    dN[0, 3] = -0.25 * (1.0 + eta)

    dN[1, 0] = -0.25 * (1.0 - xi)
    dN[1, 1] = -0.25 * (1.0 + xi)
    dN[1, 2] =  0.25 * (1.0 + xi)
    dN[1, 3] =  0.25 * (1.0 - xi)
    return dN


@njit(fastmath=True)
def compute_cpe4_fbar_element_numba(
    coords: np.ndarray,      # (4, 2)
    u_elem: np.ndarray,      # (8,)
    mat_type: int,
    props: np.ndarray,       # [E, nu, ...]
    sdvs: np.ndarray,        # (4, 7)
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    Computes 8x8 Ke and 8-element fe for 1 CPE4_FBAR element.
    """
    E = props[0]
    nu = props[1]
    if nu > 0.499999:
        nu = 0.499999

    c_fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C11 = c_fac * (1.0 - nu)
    C12 = c_fac * nu
    C33 = c_fac * 0.5 * (1.0 - 2.0 * nu)

    # 1. Evaluate centroid Jacobian and B_vol0 (xi=0, eta=0)
    dN0 = _shape_derivs_cpe4_fbar(0.0, 0.0)
    J0 = dN0 @ coords
    detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]
    has_error = 0
    if detJ0 <= 1e-14:
        has_error = 1
        detJ0 = 1e-14

    invJ00 =  J0[1, 1] / detJ0
    invJ01 = -J0[0, 1] / detJ0
    invJ10 = -J0[1, 0] / detJ0
    invJ11 =  J0[0, 0] / detJ0

    dN_dX0 = np.zeros((2, 4), dtype=np.float64)
    for a in range(4):
        dN_dX0[0, a] = invJ00 * dN0[0, a] + invJ01 * dN0[1, a]
        dN_dX0[1, a] = invJ10 * dN0[0, a] + invJ11 * dN0[1, a]

    # Centroid volumetric strain gradient B_vol0 (8,)
    B_vol0 = np.zeros(8, dtype=np.float64)
    for a in range(4):
        B_vol0[2 * a + 0] = dN_dX0[0, a]
        B_vol0[2 * a + 1] = dN_dX0[1, a]

    Ke = np.zeros((8, 8), dtype=np.float64)
    fe = np.zeros(8, dtype=np.float64)

    B = np.zeros((3, 8), dtype=np.float64)
    B_bar = np.zeros((3, 8), dtype=np.float64)

    for q in range(4):
        xi = _GPS[q, 0]
        eta = _GPS[q, 1]
        wt = _WTS[q]

        dN_dxi = _shape_derivs_cpe4_fbar(xi, eta)
        J = dN_dxi @ coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 1e-14:
            has_error = 1
            detJ = 1e-14

        invJ00 =  J[1, 1] / detJ
        invJ01 = -J[0, 1] / detJ
        invJ10 = -J[1, 0] / detJ
        invJ11 =  J[0, 0] / detJ

        dN_dX = np.zeros((2, 4), dtype=np.float64)
        for a in range(4):
            dN_dX[0, a] = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
            dN_dX[1, a] = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]

        B.fill(0.0)
        for a in range(4):
            B[0, 2 * a + 0] = dN_dX[0, a]
            B[1, 2 * a + 1] = dN_dX[1, a]
            B[2, 2 * a + 0] = dN_dX[1, a]
            B[2, 2 * a + 1] = dN_dX[0, a]

        # B_bar = B_dev + 0.5 * m * B_vol0
        for j in range(8):
            trB = (B[0, j] + B[1, j]) * 0.5
            trB0 = B_vol0[j] * 0.5
            B_bar[0, j] = B[0, j] - trB + trB0
            B_bar[1, j] = B[1, j] - trB + trB0
            B_bar[2, j] = B[2, j]

        eps = B_bar @ u_elem
        s_xx = C11 * eps[0] + C12 * eps[1]
        s_yy = C12 * eps[0] + C11 * eps[1]
        s_xy = C33 * eps[2]
        sigma = np.array([s_xx, s_yy, s_xy], dtype=np.float64)

        dV = detJ * wt

        for i in range(8):
            fe[i] += (B_bar[0, i] * sigma[0] + B_bar[1, i] * sigma[1] + B_bar[2, i] * sigma[2]) * dV

        CB = np.zeros((3, 8), dtype=np.float64)
        for j in range(8):
            CB[0, j] = C11 * B_bar[0, j] + C12 * B_bar[1, j]
            CB[1, j] = C12 * B_bar[0, j] + C11 * B_bar[1, j]
            CB[2, j] = C33 * B_bar[2, j]

        for i in range(8):
            for j in range(8):
                Ke[i, j] += (B_bar[0, i] * CB[0, j] + B_bar[1, i] * CB[1, j] + B_bar[2, i] * CB[2, j]) * dV

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe4_fbar_numba(
    node_coords_all: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    elem_mat_types: np.ndarray,
    elem_props: np.ndarray,
    elem_sdvs: np.ndarray,
    dt: float,
    elem_controls: np.ndarray = None,
    elem_stress_init: np.ndarray = None
):
    n_elems = elem_conn.shape[0]
    f_elems = np.zeros((n_elems, 8), dtype=np.float64)
    k_elems = np.zeros((n_elems, 8, 8), dtype=np.float64)
    errors = np.zeros(n_elems, dtype=np.int32)

    for e in prange(n_elems):
        conn_e = elem_conn[e]
        coords_e = np.zeros((4, 2), dtype=np.float64)
        u_e = np.zeros(8, dtype=np.float64)
        for i in range(4):
            nid = conn_e[i]
            coords_e[i, 0] = node_coords_all[nid, 0]
            coords_e[i, 1] = node_coords_all[nid, 1]
            u_e[2 * i + 0] = u_global[2 * nid + 0]
            u_e[2 * i + 1] = u_global[2 * nid + 1]

        m_type = elem_mat_types[e]
        props_e = elem_props[e]
        sdvs_e = elem_sdvs[e]
        ctrl_e = elem_controls[e] if elem_controls is not None else None

        fe, Ke, err = compute_cpe4_fbar_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
