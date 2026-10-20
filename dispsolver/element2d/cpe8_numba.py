"""
cpe8_numba.py
=============
Numba JIT accelerated 8-node Quadratic Serendipity Quadrilateral (CPE8 / CPE8R).
Supports full 3x3 (9 GPs) and reduced 2x2 (4 GPs) integration.
Equivalents: Abaqus CPE8, CPE8R / Ansys PLANE183 (Quad).
"""

import numpy as np
from numba import njit, prange

from dispsolver.material3d.numba_materials import material_dispatch_2d

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
def _shape_funcs_cpe8(xi: float, eta: float) -> np.ndarray:
    """Shape functions N_i(xi, eta) for 8-node serendipity quad: shape (8,)."""
    N = np.zeros(8, dtype=np.float64)
    N[0] = -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta)
    N[1] = -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta)
    N[2] = -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta)
    N[3] = -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta)
    N[4] = 0.5 * (1.0 - xi * xi) * (1.0 - eta)
    N[5] = 0.5 * (1.0 + xi) * (1.0 - eta * eta)
    N[6] = 0.5 * (1.0 - xi * xi) * (1.0 + eta)
    N[7] = 0.5 * (1.0 - xi) * (1.0 - eta * eta)
    return N


@njit(fastmath=True)
def compute_cpe8_element_numba(
    coords: np.ndarray,      # (8, 2)
    u_elem: np.ndarray,      # (16,)
    mat_type: int,
    props: np.ndarray,       # [E, nu, ...]
    sdvs: np.ndarray,        # (9, 7)
    dt: float,
    elem_controls: np.ndarray = None,
    reduced: bool = False,
    axisymmetric: bool = False
):
    """
    Computes 16x16 Ke and 16-element fe for 1 CPE8/CPE8R/CAX8/CAX8R element.
    """
    E = props[0]
    nu = props[1]

    Ke = np.zeros((16, 16), dtype=np.float64)
    fe = np.zeros(16, dtype=np.float64)
    has_error = 0

    gps = _GPS_2X2 if reduced else _GPS_3X3
    wts = _WTS_2X2 if reduced else _WTS_3X3
    n_gps = 4 if reduced else 9

    if axisymmetric:
        c_fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        C11 = c_fac * (1.0 - nu)
        C12 = c_fac * nu
        C44 = c_fac * 0.5 * (1.0 - 2.0 * nu)
        G_shear = C44
        K_bulk = E / (3.0 * (1.0 - 2.0 * nu))

        B = np.zeros((4, 16), dtype=np.float64)

        for q in range(n_gps):
            xi = gps[q, 0]
            eta = gps[q, 1]
            wt = wts[q]

            dN_dxi = _shape_derivs_cpe8(xi, eta)
            J = dN_dxi @ coords  # (2, 2)
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

            if abs(detJ) <= 1e-14:
                has_error = 1
                detJ = 1e-14 if detJ >= 0 else -1e-14

            invJ00 =  J[1, 1] / detJ
            invJ01 = -J[0, 1] / detJ
            invJ10 = -J[1, 0] / detJ
            invJ11 =  J[0, 0] / detJ

            dN_dX = np.zeros((2, 8), dtype=np.float64)
            for a in range(8):
                dN_dX[0, a] = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
                dN_dX[1, a] = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]

            N_vec = _shape_funcs_cpe8(xi, eta)
            r_gp = 0.0
            for a in range(8):
                r_gp += N_vec[a] * coords[a, 0]
            if r_gp < 1e-9:
                r_gp = 1e-9

            B.fill(0.0)
            for a in range(8):
                B[0, 2 * a + 0] = dN_dX[0, a]
                B[1, 2 * a + 1] = dN_dX[1, a]
                B[2, 2 * a + 0] = N_vec[a] / r_gp
                B[3, 2 * a + 0] = dN_dX[1, a]
                B[3, 2 * a + 1] = dN_dX[0, a]

            eps = B @ u_elem

            sigma_y0 = props[2] if len(props) > 2 else 0.0
            H_hard = props[3] if len(props) > 3 else 0.0

            s_rr_tr = C11 * eps[0] + C12 * (eps[1] + eps[2])
            s_zz_tr = C11 * eps[1] + C12 * (eps[0] + eps[2])
            s_tt_tr = C11 * eps[2] + C12 * (eps[0] + eps[1])
            s_rz_tr = C44 * eps[3]

            Cep = np.zeros((4, 4), dtype=np.float64)

            if sigma_y0 > 0.0:
                p_mean = (s_rr_tr + s_zz_tr + s_tt_tr) / 3.0
                S_rr = s_rr_tr - p_mean
                S_zz = s_zz_tr - p_mean
                S_tt = s_tt_tr - p_mean
                S_rz = s_rz_tr

                q_tr = np.sqrt(1.5 * (S_rr*S_rr + S_zz*S_zz + S_tt*S_tt + 2.0*S_rz*S_rz))
                phi_tr = q_tr - sigma_y0
                if phi_tr > 0.0:
                    d_gamma = phi_tr / (3.0 * G_shear + H_hard)
                    q_new = q_tr - 3.0 * G_shear * d_gamma
                    beta = q_new / q_tr
                    s_rr = p_mean + beta * S_rr
                    s_zz = p_mean + beta * S_zz
                    s_tt = p_mean + beta * S_tt
                    s_rz = beta * S_rz
                    sigma = np.array([s_rr, s_zz, s_tt, s_rz], dtype=np.float64)

                    mu_h = 3.0 * G_shear / (3.0 * G_shear + H_hard)
                    theta_h = mu_h - (1.0 - beta)
                    f_dir = 3.0 * G_shear * theta_h / (q_tr * q_tr)

                    S_vec = np.array([S_rr, S_zz, S_tt], dtype=np.float64)
                    for i in range(3):
                        for j in range(3):
                            kronecker = 1.0 if i == j else 0.0
                            Cep[i, j] = K_bulk + 2.0 * G_shear * beta * (kronecker - 1.0/3.0) - f_dir * S_vec[i] * S_vec[j]
                            
                    for i in range(3):
                        Cep[i, 3] = - f_dir * S_vec[i] * S_rz
                        Cep[3, i] = Cep[i, 3]
                    Cep[3, 3] = G_shear * beta - f_dir * S_rz * S_rz
                else:
                    sigma = np.array([s_rr_tr, s_zz_tr, s_tt_tr, s_rz_tr], dtype=np.float64)
                    Cep[0, 0], Cep[0, 1], Cep[0, 2] = C11, C12, C12
                    Cep[1, 0], Cep[1, 1], Cep[1, 2] = C12, C11, C12
                    Cep[2, 0], Cep[2, 1], Cep[2, 2] = C12, C12, C11
                    Cep[3, 3] = C44
            else:
                sigma = np.array([s_rr_tr, s_zz_tr, s_tt_tr, s_rz_tr], dtype=np.float64)
                Cep[0, 0], Cep[0, 1], Cep[0, 2] = C11, C12, C12
                Cep[1, 0], Cep[1, 1], Cep[1, 2] = C12, C11, C12
                Cep[2, 0], Cep[2, 1], Cep[2, 2] = C12, C12, C11
                Cep[3, 3] = C44

            dV = 2.0 * np.pi * r_gp * abs(detJ) * wt

            for i in range(16):
                fe[i] += (B[0, i] * sigma[0] + B[1, i] * sigma[1] + B[2, i] * sigma[2] + B[3, i] * sigma[3]) * dV

            CB = Cep @ B  # (4, 16)

            for i in range(16):
                for j in range(16):
                    Ke[i, j] += (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j] + B[3, i] * CB[3, j]) * dV

        return fe, Ke, has_error

    B = np.zeros((3, 16), dtype=np.float64)

    for q in range(n_gps):
        xi = gps[q, 0]
        eta = gps[q, 1]
        wt = wts[q]

        dN_dxi = _shape_derivs_cpe8(xi, eta)
        J = dN_dxi @ coords  # (2, 2)
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

        if abs(detJ) <= 1e-14:
            has_error = 1
            detJ = 1e-14 if detJ >= 0 else -1e-14

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

        # Stress and tangent from the material model at this Gauss point (this kernel used to carry a stateless
        # perfectly-plastic J2 of its own, read from props[2], props[3], and ignored the material).
        sdv_pt = sdvs[q] if (sdvs is not None and len(sdvs) > q) else np.zeros(8, dtype=np.float64)
        sigma, Cep, sdv_new, mat_err = material_dispatch_2d(mat_type, props, sdv_pt, eps, dt)
        if mat_err != 0:
            has_error = 1
        if sdvs is not None and len(sdvs) > q:
            for s_i in range(min(len(sdv_new), len(sdvs[q]))):
                sdvs[q, s_i] = sdv_new[s_i]

        dV = abs(detJ) * wt

        for i in range(16):
            fe[i] += (B[0, i] * sigma[0] + B[1, i] * sigma[1] + B[2, i] * sigma[2]) * dV

        CB = Cep @ B  # (3, 16)

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

        fe, Ke, err = compute_cpe8_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e, reduced=True, axisymmetric=False)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cax8_numba(
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

        fe, Ke, err = compute_cpe8_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e, reduced=False, axisymmetric=True)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cax8r_numba(
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

        fe, Ke, err = compute_cpe8_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e, reduced=True, axisymmetric=True)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
