"""
cpe6_numba.py
=============
Numba JIT accelerated 6-node Quadratic Triangle Element (CPE6).
3-point symmetric Gauss integration.
Equivalents: Abaqus CPE6 / Ansys PLANE183 (Triangle).
"""

import numpy as np
from numba import njit, prange

# 3-point Dunavant/Hammer Gauss rule for triangle
_GPS_TRI = np.array([
    [1.0 / 6.0, 1.0 / 6.0],
    [2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0]
], dtype=np.float64)
_WTS_TRI = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)


@njit(fastmath=True)
def _shape_derivs_cpe6(xi: float, eta: float) -> np.ndarray:
    """Natural derivatives dN_i/d(xi, eta) for 6-node triangle: shape (2, 6)."""
    lam1 = 1.0 - xi - eta
    lam2 = xi
    lam3 = eta

    dN = np.zeros((2, 6), dtype=np.float64)

    # d/dxi: d(lam1)/dxi = -1, d(lam2)/dxi = 1, d(lam3)/dxi = 0
    # N1 = lam1*(2*lam1 - 1) -> dN1/dxi = -(4*lam1 - 1)
    # N2 = lam2*(2*lam2 - 1) -> dN2/dxi = 4*lam2 - 1
    # N3 = lam3*(2*lam3 - 1) -> dN3/dxi = 0
    # N4 = 4*lam1*lam2 -> dN4/dxi = 4*(lam1 - lam2)
    # N5 = 4*lam2*lam3 -> dN5/dxi = 4*lam3
    # N6 = 4*lam3*lam1 -> dN6/dxi = -4*lam3
    dN[0, 0] = -(4.0 * lam1 - 1.0)
    dN[0, 1] =   4.0 * lam2 - 1.0
    dN[0, 2] =   0.0
    dN[0, 3] =   4.0 * (lam1 - lam2)
    dN[0, 4] =   4.0 * lam3
    dN[0, 5] =  -4.0 * lam3

    # d/deta: d(lam1)/deta = -1, d(lam2)/deta = 0, d(lam3)/deta = 1
    dN[1, 0] = -(4.0 * lam1 - 1.0)
    dN[1, 1] =   0.0
    dN[1, 2] =   4.0 * lam3 - 1.0
    dN[1, 3] =  -4.0 * lam2
    dN[1, 4] =   4.0 * lam2
    dN[1, 5] =   4.0 * (lam1 - lam3)

    return dN


@njit(fastmath=True)
def compute_cpe6_element_numba(
    coords: np.ndarray,      # (6, 2)
    u_elem: np.ndarray,      # (12,)
    mat_type: int,
    props: np.ndarray,       # [E, nu, ...]
    sdvs: np.ndarray,        # (3, 7)
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    Computes 12x12 Ke and 12-element fe for 1 CPE6 element.
    """
    E = props[0]
    nu = props[1]

    c_fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C11 = c_fac * (1.0 - nu)
    C12 = c_fac * nu
    C33 = c_fac * 0.5 * (1.0 - 2.0 * nu)

    Ke = np.zeros((12, 12), dtype=np.float64)
    fe = np.zeros(12, dtype=np.float64)
    has_error = 0

    B = np.zeros((3, 12), dtype=np.float64)

    for q in range(3):
        xi = _GPS_TRI[q, 0]
        eta = _GPS_TRI[q, 1]
        wt = _WTS_TRI[q]

        dN_dxi = _shape_derivs_cpe6(xi, eta)
        J = dN_dxi @ coords  # (2, 2)
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

        if detJ <= 1e-14:
            has_error = 1
            detJ = 1e-14

        invJ00 =  J[1, 1] / detJ
        invJ01 = -J[0, 1] / detJ
        invJ10 = -J[1, 0] / detJ
        invJ11 =  J[0, 0] / detJ

        dN_dX = np.zeros((2, 6), dtype=np.float64)
        for a in range(6):
            dN_dX[0, a] = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
            dN_dX[1, a] = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]

        B.fill(0.0)
        for a in range(6):
            B[0, 2 * a + 0] = dN_dX[0, a]
            B[1, 2 * a + 1] = dN_dX[1, a]
            B[2, 2 * a + 0] = dN_dX[1, a]
            B[2, 2 * a + 1] = dN_dX[0, a]

        eps = B @ u_elem
        s_xx = C11 * eps[0] + C12 * eps[1]
        s_yy = C12 * eps[0] + C11 * eps[1]
        s_xy = C33 * eps[2]
        sigma = np.array([s_xx, s_yy, s_xy], dtype=np.float64)

        dV = detJ * wt

        for i in range(12):
            fe[i] += (B[0, i] * sigma[0] + B[1, i] * sigma[1] + B[2, i] * sigma[2]) * dV

        CB = np.zeros((3, 12), dtype=np.float64)
        for j in range(12):
            CB[0, j] = C11 * B[0, j] + C12 * B[1, j]
            CB[1, j] = C12 * B[0, j] + C11 * B[1, j]
            CB[2, j] = C33 * B[2, j]

        for i in range(12):
            for j in range(12):
                Ke[i, j] += (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j]) * dV

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe6_numba(
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
    f_elems = np.zeros((n_elems, 12), dtype=np.float64)
    k_elems = np.zeros((n_elems, 12, 12), dtype=np.float64)
    errors = np.zeros(n_elems, dtype=np.int32)

    for e in prange(n_elems):
        conn_e = elem_conn[e]
        coords_e = np.zeros((6, 2), dtype=np.float64)
        u_e = np.zeros(12, dtype=np.float64)
        for i in range(6):
            nid = conn_e[i]
            coords_e[i, 0] = node_coords_all[nid, 0]
            coords_e[i, 1] = node_coords_all[nid, 1]
            u_e[2 * i + 0] = u_global[2 * nid + 0]
            u_e[2 * i + 1] = u_global[2 * nid + 1]

        m_type = elem_mat_types[e]
        props_e = elem_props[e]
        sdvs_e = elem_sdvs[e]
        ctrl_e = elem_controls[e] if elem_controls is not None else None

        fe, Ke, err = compute_cpe6_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
