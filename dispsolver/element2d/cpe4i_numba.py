"""
cpe4i_numba.py
==============
Numba JIT accelerated 4-node Incompatible Modes / Enhanced Assumed Strain Quadrilateral (CPE4I).
Simo & Rifai (1990) 4-mode EAS formulation with element-level static condensation.
Completely eliminates shear locking in bending at ANY aspect ratio and arbitrary orientation.
Equivalents: Abaqus CPE4I / Ansys SOLID182 (Enhanced Strain) / Wilson Incompatible Quad.
"""

import numpy as np
from numba import njit, prange

from dispsolver.material3d.numba_materials import material_dispatch_2d

_GP1D = 1.0 / np.sqrt(3.0)
_GPS = np.array([
    [-_GP1D, -_GP1D],
    [ _GP1D, -_GP1D],
    [ _GP1D,  _GP1D],
    [-_GP1D,  _GP1D],
], dtype=np.float64)
_WTS = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)


@njit(fastmath=True)
def _shape_derivs_cpe4i(xi: float, eta: float) -> np.ndarray:
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
def _voigt_transform_2d(J0: np.ndarray) -> np.ndarray:
    """Computes exact T0_inv for transformation of enhanced strain tensor: shape (3, 3).

    Maps natural Voigt strain [eps_xi, eps_eta, gamma_xi_eta]^T to physical [eps_x, eps_y, gamma_xy]^T
    via exact inversion of Simo-Rifai (1990) T0 strain transformation matrix.
    """
    j11 = J0[0, 0]
    j12 = J0[0, 1]
    j21 = J0[1, 0]
    j22 = J0[1, 1]
    T0 = np.array([
        [j11 * j11, j21 * j21, j11 * j21],
        [j12 * j12, j22 * j22, j12 * j22],
        [2.0 * j11 * j12, 2.0 * j21 * j22, j11 * j22 + j12 * j21],
    ], dtype=np.float64)
    return np.linalg.inv(T0)
@njit(fastmath=True)
def compute_cpe4i_element_numba(
    coords: np.ndarray,      # (4, 2)
    u_elem: np.ndarray,      # (8,)
    mat_type: int,
    props: np.ndarray,       # material properties (layout depends on mat_type)
    sdvs: np.ndarray,        # (4, n_sdv), updated in place with the converged state
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    Statically condensed 8x8 tangent Ke and 8-element fe of one CPE4I element.

    The enhanced strain parameters alpha are the solution of  fa(alpha) = sum M^T sigma(B u + M alpha) dV = 0
    for the CURRENT displacements. For a linear material one Newton step from alpha = 0 is exact
    (alpha = -Kaa^-1 fa0, fe = fu0 - Kua Kaa^-1 fa0); for a nonlinear material (plasticity, hyperelasticity)
    alpha is iterated to equilibrium, otherwise the condensed force is not the derivative of the element energy and
    the condensed tangent is not its Jacobian (measured: 1.9e-3 off, and the Newton loop of plastic bending
    stalled). The material is evaluated from the state variables at the START of the increment on every iteration;
    only the converged evaluation is written back.
    """
    # Centroid Jacobian J0
    dN0 = _shape_derivs_cpe4i(0.0, 0.0)
    J0 = dN0 @ coords
    detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]
    if detJ0 <= 1e-14:
        detJ0 = 1e-14

    T0_inv_T = _voigt_transform_2d(J0)
    has_error = 0

    # geometry of the four Gauss points: B (3x8), M (3x4), volume weight
    Bq = np.zeros((4, 3, 8), dtype=np.float64)
    Mq = np.zeros((4, 3, 4), dtype=np.float64)
    dVq = np.zeros(4, dtype=np.float64)
    M0 = np.zeros((3, 4), dtype=np.float64)
    for q in range(4):
        xi = _GPS[q, 0]
        eta = _GPS[q, 1]
        dN_dxi = _shape_derivs_cpe4i(xi, eta)
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
            Bq[q, 0, 2 * a + 0] = dx
            Bq[q, 1, 2 * a + 1] = dy
            Bq[q, 2, 2 * a + 0] = dy
            Bq[q, 2, 2 * a + 1] = dx
        # Natural EAS modes M0 = [[xi, 0, 0, 0], [0, eta, 0, 0], [0, 0, xi, eta]]; physical M = (detJ0/detJ) T M0
        M0.fill(0.0)
        M0[0, 0] = xi
        M0[1, 1] = eta
        M0[2, 2] = xi
        M0[2, 3] = eta
        Mq[q] = (detJ0 / detJ) * (T0_inv_T @ M0)
        dVq[q] = detJ * _WTS[q]

    # state variables at the start of the increment (the material is always asked from these)
    n_sdv = sdvs.shape[1] if (sdvs is not None and sdvs.ndim == 2) else 0
    sdv0 = np.zeros((4, max(n_sdv, 1)), dtype=np.float64)
    for q in range(4):
        for s_i in range(n_sdv):
            sdv0[q, s_i] = sdvs[q, s_i]
    sdv_last = np.zeros((4, max(n_sdv, 1)), dtype=np.float64)

    alpha = np.zeros(4, dtype=np.float64)
    Kuu = np.zeros((8, 8), dtype=np.float64)
    Kua = np.zeros((8, 4), dtype=np.float64)
    Kaa = np.zeros((4, 4), dtype=np.float64)
    fu = np.zeros(8, dtype=np.float64)
    fa = np.zeros(4, dtype=np.float64)

    for it in range(30):
        Kuu.fill(0.0)
        Kua.fill(0.0)
        Kaa.fill(0.0)
        fu.fill(0.0)
        fa.fill(0.0)
        for q in range(4):
            B = Bq[q]
            M = Mq[q]
            dV = dVq[q]
            eps = B @ u_elem + M @ alpha
            sdv_pt = sdv0[q, :n_sdv].copy() if n_sdv > 0 else np.zeros(7, dtype=np.float64)
            sigma, C_mat_2d, sdv_new, mat_err = material_dispatch_2d(mat_type, props, sdv_pt, eps, dt)
            if mat_err != 0:
                has_error = 1
            for s_i in range(min(len(sdv_new), n_sdv)):
                sdv_last[q, s_i] = sdv_new[s_i]

            CB = C_mat_2d @ B
            CM = C_mat_2d @ M
            for i in range(8):
                for j in range(8):
                    Kuu[i, j] += (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j]) * dV
                for j in range(4):
                    Kua[i, j] += (B[0, i] * CM[0, j] + B[1, i] * CM[1, j] + B[2, i] * CM[2, j]) * dV
                fu[i] += (B[0, i] * sigma[0] + B[1, i] * sigma[1] + B[2, i] * sigma[2]) * dV
            for i in range(4):
                for j in range(4):
                    Kaa[i, j] += (M[0, i] * CM[0, j] + M[1, i] * CM[1, j] + M[2, i] * CM[2, j]) * dV
                fa[i] += (M[0, i] * sigma[0] + M[1, i] * sigma[1] + M[2, i] * sigma[2]) * dV

        fa_norm = np.sqrt(np.sum(fa * fa))
        fu_norm = np.sqrt(np.sum(fu * fu))
        if fa_norm <= 1e-12 * max(fu_norm, 1.0e-3):
            break
        alpha = alpha - np.linalg.solve(Kaa, fa)
    else:
        fa_norm = np.sqrt(np.sum(fa * fa))
        if fa_norm > 1e-6 * max(np.sqrt(np.sum(fu * fu)), 1.0e-3):
            has_error = 1               # alpha did not reach equilibrium: let the increment be cut back

    # write back the converged material state
    for q in range(4):
        for s_i in range(n_sdv):
            sdvs[q, s_i] = sdv_last[q, s_i]

    # Static condensation at the converged alpha:  Ke = Kuu - Kua Kaa^-1 Kua^T ,  fe = fu - Kua Kaa^-1 fa (fa ~ 0)
    invKaa = np.linalg.inv(Kaa)
    Ke = Kuu - Kua @ invKaa @ Kua.T
    fe = fu - Kua @ (invKaa @ fa)

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe4i_numba(
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

        fe, Ke, err = compute_cpe4i_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
