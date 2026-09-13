"""
cpe4h_numba.py
==============
Numba JIT accelerated 4-node Mixed u-P Hybrid Quadrilateral (CPE4H).
Herrmann (1965) / Simo-Taylor-Pister (1985) mixed variational formulation.
Completely cures volumetric locking for nearly incompressible solids (nu -> 0.5).
Equivalents: Abaqus CPE4H / Ansys Mixed u-P SOLID182.
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
def _shape_derivs_cpe4h(xi: float, eta: float) -> np.ndarray:
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
def compute_cpe4h_element_numba(
    coords: np.ndarray,      # (4, 2)
    u_elem: np.ndarray,      # (8,)
    mat_type: int,
    props: np.ndarray,       # [E, nu, ...]
    sdvs: np.ndarray,        # (4, 7)
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    Computes 8x8 tangent Ke and 8-element fe for 1 CPE4H mixed hybrid element.
    """
    E = props[0]
    nu = props[1]
    if nu > 0.499999:
        nu = 0.499999

    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))

    # Deviatoric C matrix in plane strain:
    # 2G * [4/3, -2/3, 0; -2/3, 4/3, 0; 0, 0, 1] ...
    # Wait, in plane strain eps_zz = 0, so eps_vol = eps_xx + eps_yy.
    # Deviatoric strain: e_xx = eps_xx - 1/3(eps_xx+eps_yy), e_yy = eps_yy - 1/3(eps_xx+eps_yy), e_zz = -1/3(eps_xx+eps_yy).
    # Deviatoric stress: s_xx = 2G e_xx, s_yy = 2G e_yy.
    # Total stress: sigma_xx = s_xx + p, sigma_yy = s_yy + p, sigma_xy = 2G * eps_xy.

    Ke_dev = np.zeros((8, 8), dtype=np.float64)
    fe = np.zeros(8, dtype=np.float64)
    vol_vec = np.zeros(8, dtype=np.float64)  # int B_vol dV
    tot_vol = 0.0

    has_error = 0
    B = np.zeros((3, 8), dtype=np.float64)

    for q in range(4):
        xi = _GPS[q, 0]
        eta = _GPS[q, 1]
        wt = _WTS[q]

        dN_dxi = _shape_derivs_cpe4h(xi, eta)
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

        dV = detJ * wt
        tot_vol += dV

        # B_vol = [dN_1/dx, dN_1/dy, dN_2/dx, dN_2/dy, ...]
        for a in range(4):
            vol_vec[2 * a + 0] += dN_dX[0, a] * dV
            vol_vec[2 * a + 1] += dN_dX[1, a] * dV

        # Deviatoric B: B_dev = B - 1/3 m B_vol where m = [1, 1, 0]^T
        B_dev = np.zeros((3, 8), dtype=np.float64)
        for j in range(8):
            trB = (B[0, j] + B[1, j]) / 3.0
            B_dev[0, j] = B[0, j] - trB
            B_dev[1, j] = B[1, j] - trB
            B_dev[2, j] = B[2, j]

        # 2G * B_dev
        for i in range(8):
            for j in range(8):
                # deviatoric energy: 2G * (e_xx*e_xx + e_yy*e_yy + e_zz*e_zz + 2*e_xy*e_xy)
                # Note: e_zz = -1/3*(eps_xx+eps_yy)
                tr_j = (B[0, j] + B[1, j]) / 3.0
                e_zz_j = -tr_j
                tr_i = (B[0, i] + B[1, i]) / 3.0
                e_zz_i = -tr_i

                term = (
                    B_dev[0, i] * B_dev[0, j] +
                    B_dev[1, i] * B_dev[1, j] +
                    e_zz_i * e_zz_j +
                    0.5 * B_dev[2, i] * B_dev[2, j]
                )
                Ke_dev[i, j] += 2.0 * G * term * dV

        # Deviatoric internal forces
        eps = B @ u_elem
        tr_eps = (eps[0] + eps[1]) / 3.0
        e_xx = eps[0] - tr_eps
        e_yy = eps[1] - tr_eps
        e_zz = -tr_eps
        gamma_xy = eps[2]

        s_xx = 2.0 * G * e_xx
        s_yy = 2.0 * G * e_yy
        s_xy = G * gamma_xy

        for i in range(8):
            e_zz_i = -(B[0, i] + B[1, i]) / 3.0
            fe[i] += (B_dev[0, i] * s_xx + B_dev[1, i] * s_yy + e_zz_i * (2.0 * G * e_zz) + B_dev[2, i] * s_xy) * dV

    # Average volumetric strain over the element: eps_vol_avg = vol_vec^T * u_elem / tot_vol
    eps_vol_avg = np.sum(vol_vec * u_elem) / tot_vol
    p_avg = K * eps_vol_avg

    # Add volumetric force: fe += p_avg * vol_vec
    for i in range(8):
        fe[i] += p_avg * vol_vec[i]

    # Add volumetric stiffness: Ke = Ke_dev + (K / tot_vol) * (vol_vec (x) vol_vec)
    Ke = Ke_dev.copy()
    k_vol_fac = K / tot_vol
    for i in range(8):
        for j in range(8):
            Ke[i, j] += k_vol_fac * vol_vec[i] * vol_vec[j]

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe4h_numba(
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

        fe, Ke, err = compute_cpe4h_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
