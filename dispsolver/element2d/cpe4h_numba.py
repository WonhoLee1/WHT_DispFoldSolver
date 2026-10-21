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

from dispsolver.material3d.numba_materials import (
    material_dispatch_3d,
    _eig3x3_jacobi_numba,
    is_hyperelastic_material,
)

@njit(fastmath=True)
def _hencky_to_green_gl_numba(eps_voigt: np.ndarray):
    """Convert local small/Hencky strain eps_voigt into genuine Green-Lagrange strain E_voigt
    and deformation gradient F_loc to ensure det(C) > 0 and prevent material explosion under large compression/shear.
    """
    eps_mat = np.zeros((3, 3), dtype=np.float64)
    eps_mat[0, 0] = eps_voigt[0]
    eps_mat[1, 1] = eps_voigt[1]
    eps_mat[2, 2] = eps_voigt[2]
    eps_mat[0, 1] = eps_mat[1, 0] = 0.5 * eps_voigt[3]
    eps_mat[1, 2] = eps_mat[2, 1] = 0.5 * eps_voigt[4]
    eps_mat[2, 0] = eps_mat[0, 2] = 0.5 * eps_voigt[5]

    vals, V = _eig3x3_jacobi_numba(eps_mat)

    exp_vals = np.zeros(3, dtype=np.float64)
    detF = 1.0
    for i in range(3):
        exp_v = np.exp(vals[i])
        exp_vals[i] = exp_v
        detF *= exp_v

    F_loc = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                F_loc[i, j] += V[i, k] * exp_vals[k] * V[j, k]

    C_loc = F_loc.T @ F_loc
    E_gl = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            E_gl[i, j] = 0.5 * (C_loc[i, j] - (1.0 if i == j else 0.0))

    E_voigt = np.zeros(6, dtype=np.float64)
    E_voigt[0] = E_gl[0, 0]
    E_voigt[1] = E_gl[1, 1]
    E_voigt[2] = E_gl[2, 2]
    E_voigt[3] = 2.0 * E_gl[0, 1]
    E_voigt[4] = 2.0 * E_gl[1, 2]
    E_voigt[5] = 2.0 * E_gl[2, 0]

    return E_voigt, F_loc, detF

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
    props: np.ndarray,       # material properties (layout depends on mat_type)
    sdvs: np.ndarray,        # (4, n_sdv), updated in place
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    8x8 tangent Ke and 8-element fe of one CPE4H element: constant pressure (Q1P0), any material.

    The strain every Gauss point hands to the material is  eps* = eps + (D/3) (1, 1, 1, 0)  with
    D = eps_vol_avg - eps_vol(point): the local DEVIATORIC strain of the plane-strain state (so eps_zz picks up
    the out-of-plane part, -eps_vol/3) plus the element-AVERAGE volumetric strain. For a linear elastic material
    this is exactly the classical Q1P0 element (deviatoric energy at the Gauss points, K eps_vol_avg^2 V / 2 for
    the volume); for any other material the stress and tangent come from the same 3D material dispatch as every
    other element, so plasticity, hyperelasticity and their state variables work.

    eps* = B* u with B* = B + (Bvol_avg - Bvol)/3 on the xx, yy rows and (Bvol_avg - Bvol)/3 on the zz row, so
    f = int B*^T sigma* dV and K = int B*^T C* B* dV are the exact derivative pair.
    """
    has_error = 0
    B = np.zeros((4, 3, 8), dtype=np.float64)      # standard rows xx, yy, xy per Gauss point
    dV = np.zeros(4, dtype=np.float64)
    bvol_avg = np.zeros(8, dtype=np.float64)
    tot_vol = 0.0

    for q in range(4):
        dN_dxi = _shape_derivs_cpe4h(_GPS[q, 0], _GPS[q, 1])
        J = dN_dxi @ coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 1e-14:
            has_error = 1
            detJ = 1e-14

        invJ00 = J[1, 1] / detJ
        invJ01 = -J[0, 1] / detJ
        invJ10 = -J[1, 0] / detJ
        invJ11 = J[0, 0] / detJ
        for a in range(4):
            dx = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
            dy = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]
            B[q, 0, 2 * a + 0] = dx
            B[q, 1, 2 * a + 1] = dy
            B[q, 2, 2 * a + 0] = dy
            B[q, 2, 2 * a + 1] = dx
            bvol_avg[2 * a + 0] += dx * detJ * _WTS[q]
            bvol_avg[2 * a + 1] += dy * detJ * _WTS[q]
        dV[q] = detJ * _WTS[q]
        tot_vol += dV[q]

    for m in range(8):
        bvol_avg[m] /= tot_vol

    fe = np.zeros(8, dtype=np.float64)
    Ke = np.zeros((8, 8), dtype=np.float64)
    I3 = np.eye(3, dtype=np.float64)
    Bs = np.zeros((4, 8), dtype=np.float64)        # rows xx, yy, zz, xy of eps*
    hyper = is_hyperelastic_material(mat_type)

    # Hyperelastic laws: the Gauss points carry the DEVIATORIC stress and the element carries the pressure
    # K_bulk * eps_vol_avg. K_bulk is the material's own reference bulk modulus (1^T C 1 / 9 at zero strain),
    # not a props slot: Neo-Hookean accepts [C10, D1], [E, nu] and [mu, K], J2 has nu in slot 1. It is a
    # constant, so f and K stay an exact derivative pair.
    K_bulk = 0.0
    if hyper:
        sdv0 = np.zeros(sdvs.shape[1] if sdvs is not None and sdvs.ndim == 2 else 8, dtype=np.float64)
        _S0, C0, _sd0, _e0 = material_dispatch_3d(mat_type, props, sdv0, np.zeros(6, dtype=np.float64), I3, 1.0, dt)
        K_bulk = (C0[0, 0] + C0[1, 1] + C0[2, 2] + 2.0 * (C0[0, 1] + C0[0, 2] + C0[1, 2])) / 9.0
    for q in range(4):
        for m in range(8):
            d = (bvol_avg[m] - B[q, 0, m] - B[q, 1, m]) / 3.0
            Bs[0, m] = B[q, 0, m] + d
            Bs[1, m] = B[q, 1, m] + d
            Bs[2, m] = d
            Bs[3, m] = B[q, 2, m]
        eps = Bs @ u_elem
        E_voigt = np.zeros(6, dtype=np.float64)
        E_voigt[0] = eps[0]
        E_voigt[1] = eps[1]
        E_voigt[2] = eps[2]
        E_voigt[3] = eps[3]
        if hyper:
            E_gl_voigt, F_loc, detF = _hencky_to_green_gl_numba(E_voigt)
        else:
            # small-strain laws (linear elastic, J2 family) take the strain as it is
            E_gl_voigt = E_voigt
            F_loc = I3
            detF = 1.0
        sdv_pt = sdvs[q] if sdvs is not None and len(sdvs) > q else np.zeros(8, dtype=np.float64)
        S, C6, sdv_new, mat_err = material_dispatch_3d(mat_type, props, sdv_pt, E_gl_voigt, F_loc, detF, dt)
        if mat_err != 0:
            has_error = 1
        if sdvs is not None and len(sdvs) > q:
            for s_i in range(min(len(sdv_new), len(sdvs[q]))):
                sdvs[q, s_i] = sdv_new[s_i]

        if hyper:
            tr_S = (S[0] + S[1] + S[2]) / 3.0
            S_use0 = S[0] - tr_S
            S_use1 = S[1] - tr_S
            S_use2 = S[2] - tr_S
            S_use3 = S[3]

            # dS_dev/d eps = P_dev C6 dE/d eps : the material tangent is dS/dE, and the strain it sees is
            # E(eps), so the chain factor dE/d eps (central difference of the conversion, 4 columns) is part
            # of the tangent. Without it K is off by O(eps) (3-7% at 5% strain) and Newton loses its rate.
            dEde = np.zeros((6, 4), dtype=np.float64)
            for c in range(4):
                h_c = 1.0e-7
                e_p = E_voigt.copy()
                e_m = E_voigt.copy()
                e_p[c] += h_c
                e_m[c] -= h_c
                E_p, _F1, _d1 = _hencky_to_green_gl_numba(e_p)
                E_m, _F2, _d2 = _hencky_to_green_gl_numba(e_m)
                for r in range(6):
                    dEde[r, c] = (E_p[r] - E_m[r]) / (2.0 * h_c)
            C4 = np.zeros((4, 4), dtype=np.float64)
            for r in range(4):
                for c in range(4):
                    acc = 0.0
                    for k in range(6):
                        acc += C6[r, k] * dEde[k, c]
                    C4[r, c] = acc
            P_dev = np.eye(4, dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    P_dev[r, c] -= 1.0 / 3.0
            C4 = P_dev @ C4
        else:
            S_use0 = S[0]
            S_use1 = S[1]
            S_use2 = S[2]
            S_use3 = S[3]
            C4 = np.zeros((4, 4), dtype=np.float64)
            for r in range(4):
                for c in range(4):
                    C4[r, c] = C6[r, c]

        for i in range(8):
            fe[i] += (Bs[0, i] * S_use0 + Bs[1, i] * S_use1 + Bs[2, i] * S_use2 + Bs[3, i] * S_use3) * dV[q]

        CB = C4 @ Bs
        for i in range(8):
            for j in range(8):
                Ke[i, j] += (Bs[0, i] * CB[0, j] + Bs[1, i] * CB[1, j] + Bs[2, i] * CB[2, j] + Bs[3, i] * CB[3, j]) * dV[q]

    if hyper:
        eps_vol_mean = 0.0
        for m in range(8):
            eps_vol_mean += bvol_avg[m] * u_elem[m]
        p_mean = K_bulk * eps_vol_mean
        for i in range(8):
            fe[i] += bvol_avg[i] * p_mean * tot_vol
            for j in range(8):
                Ke[i, j] += bvol_avg[i] * bvol_avg[j] * (K_bulk * tot_vol)

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
