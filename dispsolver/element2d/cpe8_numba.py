"""
cpe8_numba.py
=============
Numba JIT accelerated 8-node Quadratic Serendipity Quadrilateral (CPE8 / CPE8R).
Supports full 3x3 (9 GPs) and reduced 2x2 (4 GPs) integration.
Equivalents: Abaqus CPE8, CPE8R / Ansys PLANE183 (Quad).
"""

import numpy as np
from numba import njit, prange

# 3x3 Gauss rule for CPE8 full integration
_G3 = np.sqrt(0.6)
_GPS_3X3 = np.array([
    [-_G3, -_G3], [ 0.0, -_G3], [ _G3, -_G3],
    [-_G3,  0.0], [ 0.0,  0.0], [ _G3,  0.0],
    [-_G3,  _G3], [ 0.0,  _G3], [ _G3,  _G3]
], dtype=np.float64)
_W1 = 5.0 / 9.0
_W2 = 8.0 / 9.0
_WTS_3X3 = np.array([
    _W1*_W1, _W2*_W1, _W1*_W1,
    _W1*_W2, _W2*_W2, _W1*_W2,
    _W1*_W1, _W2*_W1, _W1*_W1
], dtype=np.float64)

# 2x2 Gauss rule for CPE8R reduced integration
_G2 = 1.0 / np.sqrt(3.0)
_GPS_2X2 = np.array([
    [-_G2, -_G2],
    [ _G2, -_G2],
    [ _G2,  _G2],
    [-_G2,  _G2]
], dtype=np.float64)
_WTS_2X2 = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)


@njit(fastmath=True)
def _shape_derivs_cpe8(xi: float, eta: float) -> np.ndarray:
    """Natural derivatives dN_i/d(xi, eta) for 8-node serendipity quad: shape (2, 8)."""
    dN = np.zeros((2, 8), dtype=np.float64)

    # d/dxi:
    # N1 = -0.25*(1-xi)*(1-eta)*(1+xi+eta)
    dN[0, 0] = 0.25 * (1.0 - eta) * (2.0 * xi + eta)
    dN[0, 1] = 0.25 * (1.0 - eta) * (2.0 * xi - eta)
    dN[0, 2] = 0.25 * (1.0 + eta) * (2.0 * xi + eta)
    dN[0, 3] = 0.25 * (1.0 + eta) * (2.0 * xi - eta)
    dN[0, 4] = -xi * (1.0 - eta)
    dN[0, 5] = 0.5 * (1.0 - eta * eta)
    dN[0, 6] = -xi * (1.0 + eta)
    dN[0, 7] = -0.5 * (1.0 - eta * eta)

    # d/deta:
    dN[1, 0] = 0.25 * (1.0 - xi) * (xi + 2.0 * eta)
    dN[1, 1] = 0.25 * (1.0 + xi) * (-xi + 2.0 * eta)
    dN[1, 2] = 0.25 * (1.0 + xi) * (xi + 2.0 * eta)
    dN[1, 3] = 0.25 * (1.0 - xi) * (-xi + 2.0 * eta)
    dN[1, 4] = -0.5 * (1.0 - xi * xi)
    dN[1, 5] = -eta * (1.0 + xi)
    dN[1, 6] = 0.5 * (1.0 - xi * xi)
    dN[1, 7] = -eta * (1.0 - xi)

    return dN


@njit(fastmath=True)
def compute_cpe8_element_numba(
    coords: np.ndarray,      # (8, 2)
    u_elem: np.ndarray,      # (16,)
    mat_type: int,
    props: np.ndarray,       # [E, nu, ...]
    sdvs: np.ndarray,        # (9, 7)
    dt: float,
    elem_controls: np.ndarray = None,
    reduced: bool = False
):
    """
    Computes 16x16 Ke and 16-element fe for 1 CPE8/CPE8R element.
    """
    E = props[0]
    nu = props[1]

    c_fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C11 = c_fac * (1.0 - nu)
    C12 = c_fac * nu
    C33 = c_fac * 0.5 * (1.0 - 2.0 * nu)

    Ke = np.zeros((16, 16), dtype=np.float64)
    fe = np.zeros(16, dtype=np.float64)
    has_error = 0

    gps = _GPS_2X2 if reduced else _GPS_3X3
    wts = _WTS_2X2 if reduced else _WTS_3X3
    n_gps = 4 if reduced else 9

    B = np.zeros((3, 16), dtype=np.float64)

    for q in range(n_gps):
        xi = gps[q, 0]
        eta = gps[q, 1]
        wt = wts[q]

        dN_dxi = _shape_derivs_cpe8(xi, eta)
        J = dN_dxi @ coords  # (2, 2)
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

        if detJ <= 1e-14:
            has_error = 1
            detJ = 1e-14

        invJ00 =  J[1, 1] / detJ
        invJ01 = -J[0, 1] / detJ
        invJ10 = -J[1, 0] / detJ
        invJ11 =  J[0, 0] / detJ

        dN_dX = np.zeros((2, 8), dtype=np.float64)
        for a in range(8):
            dN_dX[0, a] = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
            dN_dX[1, a] = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]

        B.fill(0.0)
        for a in range(8):
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

        for i in range(16):
            fe[i] += (B[0, i] * sigma[0] + B[1, i] * sigma[1] + B[2, i] * sigma[2]) * dV

        CB = np.zeros((3, 16), dtype=np.float64)
        for j in range(16):
            CB[0, j] = C11 * B[0, j] + C12 * B[1, j]
            CB[1, j] = C12 * B[0, j] + C11 * B[1, j]
            CB[2, j] = C33 * B[2, j]

        for i in range(16):
            for j in range(16):
                Ke[i, j] += (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j]) * dV

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe8_numba(
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
    f_elems = np.zeros((n_elems, 16), dtype=np.float64)
    k_elems = np.zeros((n_elems, 16, 16), dtype=np.float64)
    errors = np.zeros(n_elems, dtype=np.int32)

    for e in prange(n_elems):
        conn_e = elem_conn[e]
        coords_e = np.zeros((8, 2), dtype=np.float64)
        u_e = np.zeros(16, dtype=np.float64)
        for i in range(8):
            nid = conn_e[i]
            coords_e[i, 0] = node_coords_all[nid, 0]
            coords_e[i, 1] = node_coords_all[nid, 1]
            u_e[2 * i + 0] = u_global[2 * nid + 0]
            u_e[2 * i + 1] = u_global[2 * nid + 1]

        m_type = elem_mat_types[e]
        props_e = elem_props[e]
        sdvs_e = elem_sdvs[e]
        ctrl_e = elem_controls[e] if elem_controls is not None else None

        fe, Ke, err = compute_cpe8_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e, reduced=False)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe8r_numba(
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
    f_elems = np.zeros((n_elems, 16), dtype=np.float64)
    k_elems = np.zeros((n_elems, 16, 16), dtype=np.float64)
    errors = np.zeros(n_elems, dtype=np.int32)

    for e in prange(n_elems):
        conn_e = elem_conn[e]
        coords_e = np.zeros((8, 2), dtype=np.float64)
        u_e = np.zeros(16, dtype=np.float64)
        for i in range(8):
            nid = conn_e[i]
            coords_e[i, 0] = node_coords_all[nid, 0]
            coords_e[i, 1] = node_coords_all[nid, 1]
            u_e[2 * i + 0] = u_global[2 * nid + 0]
            u_e[2 * i + 1] = u_global[2 * nid + 1]

        m_type = elem_mat_types[e]
        props_e = elem_props[e]
        sdvs_e = elem_sdvs[e]
        ctrl_e = elem_controls[e] if elem_controls is not None else None

        fe, Ke, err = compute_cpe8_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e, reduced=True)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
