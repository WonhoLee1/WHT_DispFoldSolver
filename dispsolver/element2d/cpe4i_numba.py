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
    """Computes T0_inv_T for transformation of enhanced strain tensor: shape (3, 3)."""
    # Inverse of centroid Jacobian J0
    detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]
    invJ00 =  J0[1, 1] / detJ0
    invJ01 = -J0[0, 1] / detJ0
    invJ10 = -J0[1, 0] / detJ0
    invJ11 =  J0[0, 0] / detJ0

    # T0_inv_T maps natural Voigt strain [eps_xi, eps_eta, gamma_xi_eta]^T to physical [eps_x, eps_y, gamma_xy]^T
    T = np.zeros((3, 3), dtype=np.float64)
    T[0, 0] = invJ00 * invJ00
    T[0, 1] = invJ10 * invJ10
    T[0, 2] = invJ00 * invJ10

    T[1, 0] = invJ01 * invJ01
    T[1, 1] = invJ11 * invJ11
    T[1, 2] = invJ01 * invJ11

    T[2, 0] = 2.0 * invJ00 * invJ01
    T[2, 1] = 2.0 * invJ10 * invJ11
    T[2, 2] = invJ00 * invJ11 + invJ01 * invJ10
    return T


@njit(fastmath=True)
def compute_cpe4i_element_numba(
    coords: np.ndarray,      # (4, 2)
    u_elem: np.ndarray,      # (8,)
    mat_type: int,
    props: np.ndarray,       # [E, nu, ...]
    sdvs: np.ndarray,        # (4, 7)
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    Computes statically condensed 8x8 tangent Ke and 8-element fe for 1 CPE4I element.
    """
    E = props[0]
    nu = props[1]

    # Plane strain C matrix (3x3)
    c_fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C11 = c_fac * (1.0 - nu)
    C12 = c_fac * nu
    C33 = c_fac * 0.5 * (1.0 - 2.0 * nu)

    # Centroid Jacobian J0
    dN0 = _shape_derivs_cpe4i(0.0, 0.0)
    J0 = dN0 @ coords
    detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]
    if detJ0 <= 1e-14:
        detJ0 = 1e-14

    T0_inv_T = _voigt_transform_2d(J0)

    Kuu = np.zeros((8, 8), dtype=np.float64)
    Kua = np.zeros((8, 4), dtype=np.float64)
    Kaa = np.zeros((4, 4), dtype=np.float64)
    fu = np.zeros(8, dtype=np.float64)
    fa = np.zeros(4, dtype=np.float64)

    has_error = 0
    B = np.zeros((3, 8), dtype=np.float64)
    M0 = np.zeros((3, 4), dtype=np.float64)

    for q in range(4):
        xi = _GPS[q, 0]
        eta = _GPS[q, 1]
        wt = _WTS[q]

        dN_dxi = _shape_derivs_cpe4i(xi, eta)
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

        # Natural EAS modes: M0 = [[xi, 0, 0, 0], [0, eta, 0, 0], [0, 0, xi, eta]]
        M0.fill(0.0)
        M0[0, 0] = xi
        M0[1, 1] = eta
        M0[2, 2] = xi
        M0[2, 3] = eta

        # Physical EAS modes: M = (detJ0 / detJ) * T0_inv_T * M0
        scale = detJ0 / detJ
        M = scale * (T0_inv_T @ M0)  # (3, 4)

        dV = detJ * wt

        # CB = C * B (3, 8)
        CB = np.zeros((3, 8), dtype=np.float64)
        for j in range(8):
            CB[0, j] = C11 * B[0, j] + C12 * B[1, j]
            CB[1, j] = C12 * B[0, j] + C11 * B[1, j]
            CB[2, j] = C33 * B[2, j]

        # CM = C * M (3, 4)
        CM = np.zeros((3, 4), dtype=np.float64)
        for j in range(4):
            CM[0, j] = C11 * M[0, j] + C12 * M[1, j]
            CM[1, j] = C12 * M[0, j] + C11 * M[1, j]
            CM[2, j] = C33 * M[2, j]

        # Kuu += B^T * CB * dV
        for i in range(8):
            for j in range(8):
                Kuu[i, j] += (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j]) * dV

        # Kua += B^T * CM * dV
        for i in range(8):
            for j in range(4):
                Kua[i, j] += (B[0, i] * CM[0, j] + B[1, i] * CM[1, j] + B[2, i] * CM[2, j]) * dV

        # Kaa += M^T * CM * dV
        for i in range(4):
            for j in range(4):
                Kaa[i, j] += (M[0, i] * CM[0, j] + M[1, i] * CM[1, j] + M[2, i] * CM[2, j]) * dV

        # Initial compatible strain and stress
        eps = B @ u_elem
        s_xx = C11 * eps[0] + C12 * eps[1]
        s_yy = C12 * eps[0] + C11 * eps[1]
        s_xy = C33 * eps[2]
        sigma = np.array([s_xx, s_yy, s_xy], dtype=np.float64)

        for i in range(8):
            fu[i] += (B[0, i] * sigma[0] + B[1, i] * sigma[1] + B[2, i] * sigma[2]) * dV

        for i in range(4):
            fa[i] += (M[0, i] * sigma[0] + M[1, i] * sigma[1] + M[2, i] * sigma[2]) * dV

    # Invert 4x4 Kaa for static condensation
    invKaa = np.linalg.inv(Kaa)

    # Condensation: Ke = Kuu - Kua * invKaa * Kua^T
    Ke = Kuu - Kua @ invKaa @ Kua.T

    # Condensed internal force: fe = fu - Kua * invKaa * fa
    alpha = invKaa @ fa
    fe = fu - Kua @ alpha

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
