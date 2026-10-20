"""
cpe6m_numba.py
==============
Numba JIT accelerated Modified 6-node Quadratic Triangle Element (CPE6M).
Volumetric B-bar + positive contact tractions compatibility.
Equivalents: Abaqus CPE6M / Modified 6-node Tri.
"""

import numpy as np
from numba import njit, prange

from dispsolver.material3d.numba_materials import material_dispatch_2d

_GPS_TRI = np.array([
    [1.0 / 6.0, 1.0 / 6.0],
    [2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0]
], dtype=np.float64)
_WTS_TRI = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=np.float64)


@njit(fastmath=True)
def _shape_derivs_cpe6m(xi: float, eta: float) -> np.ndarray:
    lam1 = 1.0 - xi - eta
    lam2 = xi
    lam3 = eta

    dN = np.zeros((2, 6), dtype=np.float64)

    dN[0, 0] = -(4.0 * lam1 - 1.0)
    dN[0, 1] =   4.0 * lam2 - 1.0
    dN[0, 2] =   0.0
    dN[0, 3] =   4.0 * (lam1 - lam2)
    dN[0, 4] =   4.0 * lam3
    dN[0, 5] =  -4.0 * lam3

    dN[1, 0] = -(4.0 * lam1 - 1.0)
    dN[1, 1] =   0.0
    dN[1, 2] =   4.0 * lam3 - 1.0
    dN[1, 3] =  -4.0 * lam2
    dN[1, 4] =   4.0 * lam2
    dN[1, 5] =   4.0 * (lam1 - lam3)

    return dN


@njit(fastmath=True)
def compute_cpe6m_element_numba(
    coords: np.ndarray,      # (6, 2)
    u_elem: np.ndarray,      # (12,)
    mat_type: int,
    props: np.ndarray,       # [E, nu, ...]
    sdvs: np.ndarray,        # (3, 7)
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    Computes 12x12 Ke and 12-element fe for 1 CPE6M modified element with volumetric B-bar.
    """
    # The stabilisation below needs the elastic shear modulus; it comes from the material's INITIAL tangent
    # (a plastic tangent would soften it and release the volumetric hourglass mode).
    _, C0, _, _ = material_dispatch_2d(mat_type, props, np.zeros(8, dtype=np.float64),
                                       np.zeros(3, dtype=np.float64), 0.0)
    C33 = C0[2, 2]

    Ke = np.zeros((12, 12), dtype=np.float64)
    fe = np.zeros(12, dtype=np.float64)
    has_error = 0

    # 1. Compute element area-averaged volumetric B-vector B_vol_avg
    B_vol_sum = np.zeros(12, dtype=np.float64)
    tot_area = 0.0

    for q in range(3):
        xi = _GPS_TRI[q, 0]
        eta = _GPS_TRI[q, 1]
        wt = _WTS_TRI[q]

        dN_dxi = _shape_derivs_cpe6m(xi, eta)
        J = dN_dxi @ coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 1e-14:
            has_error = 1
            detJ = 1e-14

        invJ00 =  J[1, 1] / detJ
        invJ01 = -J[0, 1] / detJ
        invJ10 = -J[1, 0] / detJ
        invJ11 =  J[0, 0] / detJ

        dA = detJ * wt
        tot_area += dA

        for a in range(6):
            bx = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
            by = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]
            B_vol_sum[2 * a + 0] += bx * dA
            B_vol_sum[2 * a + 1] += by * dA

    B_vol_avg = B_vol_sum / tot_area

    # 2. Assemble with B_bar = B_dev + 0.5 * m * B_vol_avg
    B = np.zeros((3, 12), dtype=np.float64)
    B_bar = np.zeros((3, 12), dtype=np.float64)

    for q in range(3):
        xi = _GPS_TRI[q, 0]
        eta = _GPS_TRI[q, 1]
        wt = _WTS_TRI[q]

        dN_dxi = _shape_derivs_cpe6m(xi, eta)
        J = dN_dxi @ coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 1e-14:
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

        # B_bar formulation
        for j in range(12):
            trB = (B[0, j] + B[1, j]) * 0.5
            trB_avg = B_vol_avg[j] * 0.5
            B_bar[0, j] = B[0, j] - trB + trB_avg
            B_bar[1, j] = B[1, j] - trB + trB_avg
            B_bar[2, j] = B[2, j]

        eps = B_bar @ u_elem
        sdv_pt = sdvs[q] if (sdvs is not None and len(sdvs) > q) else np.zeros(8, dtype=np.float64)
        sigma, Dm, sdv_new, mat_err = material_dispatch_2d(mat_type, props, sdv_pt, eps, dt)
        if mat_err != 0:
            has_error = 1
        if sdvs is not None and len(sdvs) > q:
            for s_i in range(min(len(sdv_new), len(sdvs[q]))):
                sdvs[q, s_i] = sdv_new[s_i]

        dV = detJ * wt

        for i in range(12):
            fe[i] += (B_bar[0, i] * sigma[0] + B_bar[1, i] * sigma[1] + B_bar[2, i] * sigma[2]) * dV

        CB = Dm @ B_bar

        for i in range(12):
            for j in range(12):
                Ke[i, j] += (B_bar[0, i] * CB[0, j] + B_bar[1, i] * CB[1, j] + B_bar[2, i] * CB[2, j]) * dV

        # Volumetric hourglass stabilization (orthogonal to mean dilatation)
        alpha_hg = 0.05
        if elem_controls is not None and elem_controls.shape[0] >= 3 and elem_controls[2] > 0.0:
            alpha_hg = elem_controls[2]

        Delta_B = np.zeros(12, dtype=np.float64)
        for j in range(12):
            Delta_B[j] = (B[0, j] + B[1, j]) - B_vol_avg[j]

        eps_hg = float(Delta_B @ u_elem)
        k_hg_factor = alpha_hg * (2.0 * C33) * dV

        for i in range(12):
            fe[i] += Delta_B[i] * (k_hg_factor * eps_hg)
            for j in range(12):
                Ke[i, j] += k_hg_factor * Delta_B[i] * Delta_B[j]

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe6m_numba(
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

        fe, Ke, err = compute_cpe6m_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
