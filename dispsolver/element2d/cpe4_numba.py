"""
cpe4_numba.py
=============
Numba JIT accelerated 4-node Bilinear Plane Strain Quadrilateral Element (CPE4).
Standard 2x2 Gauss full numerical integration (4 GPs).
Equivalents: Abaqus CPE4 / Ansys SOLID182 (Full Integration).
"""

import numpy as np
from numba import njit, prange

from dispsolver.material3d.numba_materials import material_dispatch_2d

# 2x2 Gauss points and weights in [-1, 1]
_GP1D = 1.0 / np.sqrt(3.0)
_GPS = np.array([
    [-_GP1D, -_GP1D],
    [ _GP1D, -_GP1D],
    [ _GP1D,  _GP1D],
    [-_GP1D,  _GP1D],
], dtype=np.float64)
_WTS = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)


@njit(fastmath=True)
def _shape_derivs_cpe4(xi: float, eta: float) -> np.ndarray:
    """Natural derivatives dN_i/d(xi, eta) for 4-node quad: shape (2, 4)."""
    dN = np.zeros((2, 4), dtype=np.float64)
    # N1 = 0.25*(1-xi)*(1-eta)
    # N2 = 0.25*(1+xi)*(1-eta)
    # N3 = 0.25*(1+xi)*(1+eta)
    # N4 = 0.25*(1-xi)*(1+eta)
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
def _shape_funcs_cpe4(xi: float, eta: float) -> np.ndarray:
    """Shape functions N_i(xi, eta) for 4-node quad: shape (4,)."""
    N = np.zeros(4, dtype=np.float64)
    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta)
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta)
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta)
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta)
    return N


@njit(fastmath=True)
def compute_cpe4_element_numba(
    coords: np.ndarray,      # (4, 2)
    u_elem: np.ndarray,      # (8,)
    mat_type: int,
    props: np.ndarray,       # [E, nu, ...]
    sdvs: np.ndarray,        # (4, 7)
    dt: float,
    elem_controls: np.ndarray = None,
    axisymmetric: bool = False
):
    """
    Computes 8x8 tangent stiffness Ke and 8-element internal force vector fe for 1 CPE4/CAX4 element with B-bar volumetric projection.
    """
    E = props[0]
    nu = props[1]
    if nu > 0.499999:
        nu = 0.499999

    use_bbar = (nu > 0.45)
    Ke = np.zeros((8, 8), dtype=np.float64)
    fe = np.zeros(8, dtype=np.float64)
    has_error = 0

    if axisymmetric:
        c_fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        C11 = c_fac * (1.0 - nu)
        C12 = c_fac * nu
        C44 = c_fac * 0.5 * (1.0 - 2.0 * nu)

        B_all = np.zeros((4, 4, 8), dtype=np.float64)
        dV_all = np.zeros(4, dtype=np.float64)
        B_vol_bar = np.zeros(8, dtype=np.float64)
        vol_elem = 0.0

        for q in range(4):
            xi = _GPS[q, 0]
            eta = _GPS[q, 1]
            wt = _WTS[q]

            dN_dxi = _shape_derivs_cpe4(xi, eta)
            J = dN_dxi @ coords
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

            if abs(detJ) <= 1e-14:
                has_error = 1
                detJ = 1e-14 if detJ >= 0.0 else -1e-14

            invJ00 =  J[1, 1] / detJ
            invJ01 = -J[0, 1] / detJ
            invJ10 = -J[1, 0] / detJ
            invJ11 =  J[0, 0] / detJ

            dN_dX = np.zeros((2, 4), dtype=np.float64)
            for a in range(4):
                dN_dX[0, a] = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
                dN_dX[1, a] = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]

            N_vec = _shape_funcs_cpe4(xi, eta)
            r_gp = 0.0
            for a in range(4):
                r_gp += N_vec[a] * coords[a, 0]
            if r_gp < 1e-9:
                r_gp = 1e-9

            for a in range(4):
                B_all[q, 0, 2 * a + 0] = dN_dX[0, a]
                B_all[q, 1, 2 * a + 1] = dN_dX[1, a]
                B_all[q, 2, 2 * a + 0] = N_vec[a] / r_gp
                B_all[q, 3, 2 * a + 0] = dN_dX[1, a]
                B_all[q, 3, 2 * a + 1] = dN_dX[0, a]

            dV = 2.0 * np.pi * r_gp * abs(detJ) * wt
            dV_all[q] = dV
            vol_elem += dV

            if use_bbar:
                for m in range(8):
                    B_vol_bar[m] += (B_all[q, 0, m] + B_all[q, 1, m] + B_all[q, 2, m]) * dV

        if use_bbar and vol_elem > 0.0:
            for m in range(8):
                B_vol_bar[m] /= vol_elem

        for q in range(4):
            B = B_all[q].copy()
            if use_bbar:
                for m in range(8):
                    b_vol_q = B_all[q, 0, m] + B_all[q, 1, m] + B_all[q, 2, m]
                    delta = (B_vol_bar[m] - b_vol_q) / 3.0
                    B[0, m] += delta
                    B[1, m] += delta
                    B[2, m] += delta

            eps = B @ u_elem

            s_rr = C11 * eps[0] + C12 * (eps[1] + eps[2])
            s_zz = C11 * eps[1] + C12 * (eps[0] + eps[2])
            s_tt = C11 * eps[2] + C12 * (eps[0] + eps[1])
            s_rz = C44 * eps[3]
            sigma = np.array([s_rr, s_zz, s_tt, s_rz], dtype=np.float64)

            dV = dV_all[q]

            for i in range(8):
                fe[i] += (B[0, i] * sigma[0] + B[1, i] * sigma[1] + B[2, i] * sigma[2] + B[3, i] * sigma[3]) * dV

            CB = np.zeros((4, 8), dtype=np.float64)
            for j in range(8):
                CB[0, j] = C11 * B[0, j] + C12 * (B[1, j] + B[2, j])
                CB[1, j] = C11 * B[1, j] + C12 * (B[0, j] + B[2, j])
                CB[2, j] = C11 * B[2, j] + C12 * (B[0, j] + B[1, j])
                CB[3, j] = C44 * B[3, j]

            for i in range(8):
                for j in range(8):
                    Ke[i, j] += (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j] + B[3, i] * CB[3, j]) * dV

        return fe, Ke, has_error

    B_all_2d = np.zeros((4, 3, 8), dtype=np.float64)
    dV_all_2d = np.zeros(4, dtype=np.float64)
    B_vol_bar_2d = np.zeros(8, dtype=np.float64)
    vol_elem_2d = 0.0

    for q in range(4):
        xi = _GPS[q, 0]
        eta = _GPS[q, 1]
        wt = _WTS[q]

        dN_dxi = _shape_derivs_cpe4(xi, eta)
        J = dN_dxi @ coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

        if abs(detJ) <= 1e-14:
            has_error = 1
            detJ = 1e-14 if detJ >= 0.0 else -1e-14

        invJ00 =  J[1, 1] / detJ
        invJ01 = -J[0, 1] / detJ
        invJ10 = -J[1, 0] / detJ
        invJ11 =  J[0, 0] / detJ

        dN_dX = np.zeros((2, 4), dtype=np.float64)
        for a in range(4):
            dN_dX[0, a] = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
            dN_dX[1, a] = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]

        for a in range(4):
            B_all_2d[q, 0, 2 * a + 0] = dN_dX[0, a]
            B_all_2d[q, 1, 2 * a + 1] = dN_dX[1, a]
            B_all_2d[q, 2, 2 * a + 0] = dN_dX[1, a]
            B_all_2d[q, 2, 2 * a + 1] = dN_dX[0, a]

        dV = abs(detJ) * wt
        dV_all_2d[q] = dV
        vol_elem_2d += dV

        if use_bbar:
            for m in range(8):
                B_vol_bar_2d[m] += (B_all_2d[q, 0, m] + B_all_2d[q, 1, m]) * dV

    if use_bbar and vol_elem_2d > 0.0:
        for m in range(8):
            B_vol_bar_2d[m] /= vol_elem_2d

    for q in range(4):
        B = B_all_2d[q].copy()
        if use_bbar:
            for m in range(8):
                b_vol_q = B_all_2d[q, 0, m] + B_all_2d[q, 1, m]
                delta = (B_vol_bar_2d[m] - b_vol_q) / 2.0
                B[0, m] += delta
                B[1, m] += delta

        eps = B @ u_elem
        sdv_pt = sdvs[q] if sdvs is not None and len(sdvs) > q else np.zeros(7, dtype=np.float64)
        sigma, C_mat_2d, sdv_new, mat_err = material_dispatch_2d(mat_type, props, sdv_pt, eps, dt)
        if mat_err != 0:
            has_error = 1
        if sdvs is not None and len(sdvs) > q:
            for s_i in range(min(len(sdv_new), len(sdvs[q]))):
                sdvs[q, s_i] = sdv_new[s_i]

        dV = dV_all_2d[q]

        for i in range(8):
            fe[i] += (B[0, i] * sigma[0] + B[1, i] * sigma[1] + B[2, i] * sigma[2]) * dV

        CB = C_mat_2d @ B

        for i in range(8):
            for j in range(8):
                Ke[i, j] += (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j]) * dV

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe4_numba(
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
    """Parallel assembly of all CPE4 elements in the mesh."""
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

        fe, Ke, err = compute_cpe4_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e, axisymmetric=False)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cax4_numba(
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
    """Parallel assembly of all CAX4 / CAX4I / CAX4H / CAX4IH elements in the mesh."""
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

        fe, Ke, err = compute_cpe4_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e, axisymmetric=True)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
