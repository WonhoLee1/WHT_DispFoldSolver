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

from dispsolver.material3d.numba_materials import material_dispatch_3d

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
        sdv_pt = sdvs[q] if sdvs is not None and len(sdvs) > q else np.zeros(8, dtype=np.float64)
        S, C6, sdv_new, mat_err = material_dispatch_3d(mat_type, props, sdv_pt, E_voigt, I3, 1.0, dt)
        if mat_err != 0:
            has_error = 1
        if sdvs is not None and len(sdvs) > q:
            for s_i in range(min(len(sdv_new), len(sdvs[q]))):
                sdvs[q, s_i] = sdv_new[s_i]

        for i in range(8):
            fe[i] += (Bs[0, i] * S[0] + Bs[1, i] * S[1] + Bs[2, i] * S[2] + Bs[3, i] * S[3]) * dV[q]
        C4 = np.zeros((4, 4), dtype=np.float64)
        for r in range(4):
            for c in range(4):
                C4[r, c] = C6[r, c]
        CB = C4 @ Bs
        for i in range(8):
            for j in range(8):
                Ke[i, j] += (Bs[0, i] * CB[0, j] + Bs[1, i] * CB[1, j] + Bs[2, i] * CB[2, j] + Bs[3, i] * CB[3, j]) * dV[q]

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
