"""
q4_hybrid_numba.py
==================
Numba C-JIT accelerated implementation for Abaqus-grade Hybrid Q4 Elements
(CPE4H / Q4_HYBRID, CPE4RH / Q4_HYBRID_RH, CPE4IH / Q4_COROTATIONAL_HYBRID_EAS).

Provides multi-threaded parallel element force vector and tangent stiffness matrix assembly.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(fastmath=True)
def _compute_q4_hybrid_single_element(
    coords: np.ndarray,
    u_elem: np.ndarray,
    E: float,
    nu: float,
    thickness: float = 1.0,
):
    """Compute element internal force vector (8,) and stiffness matrix (8,8) for Q1P0 Hybrid Q4 element."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    K_bulk = lam + (2.0 / 3.0) * mu

    # 2x2 Gauss integration points & weights
    gp_pos = 1.0 / np.sqrt(3.0)
    gp_coords = np.array([
        [-gp_pos, -gp_pos],
        [ gp_pos, -gp_pos],
        [ gp_pos,  gp_pos],
        [-gp_pos,  gp_pos]
    ])
    gp_weights = np.ones(4)

    f_dev = np.zeros(8)
    K_dev = np.zeros((8, 8))
    V_elem = 0.0
    int_J_minus_1 = 0.0
    B_vol_int = np.zeros(8)

    for gp in range(4):
        xi = gp_coords[gp, 0]
        eta = gp_coords[gp, 1]

        # Shape function derivatives w.r.t xi, eta
        dN_dxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
        dN_deta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

        J_mat = np.zeros((2, 2))
        for i in range(4):
            J_mat[0, 0] += dN_dxi[i] * coords[i, 0]
            J_mat[0, 1] += dN_dxi[i] * coords[i, 1]
            J_mat[1, 0] += dN_deta[i] * coords[i, 0]
            J_mat[1, 1] += dN_deta[i] * coords[i, 1]

        detJ = J_mat[0, 0] * J_mat[1, 1] - J_mat[0, 1] * J_mat[1, 0]
        invJ = np.array([[J_mat[1, 1], -J_mat[0, 1]], [-J_mat[1, 0], J_mat[0, 0]]]) / detJ

        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

        w = detJ * gp_weights[gp] * thickness

        # Displacement gradients
        ux = u_elem[0::2]
        uy = u_elem[1::2]

        grad_u_00 = np.sum(ux * dN_dx)
        grad_u_01 = np.sum(ux * dN_dy)
        grad_u_10 = np.sum(uy * dN_dx)
        grad_u_11 = np.sum(uy * dN_dy)

        F00 = 1.0 + grad_u_00
        F01 = grad_u_01
        F10 = grad_u_10
        F11 = 1.0 + grad_u_11

        J_gp = F00 * F11 - F01 * F10

        # B-matrix for strain
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2 * i] = dN_dx[i] * F00
            B[0, 2 * i + 1] = dN_dx[i] * F10
            B[1, 2 * i] = dN_dy[i] * F01
            B[1, 2 * i + 1] = dN_dy[i] * F11
            B[2, 2 * i] = dN_dx[i] * F01 + dN_dy[i] * F00
            B[2, 2 * i + 1] = dN_dx[i] * F11 + dN_dy[i] * F10

        # Volumetric vector
        b_vol = np.zeros(8)
        for i in range(4):
            b_vol[2 * i] = dN_dx[i] * F11 - dN_dy[i] * F10
            b_vol[2 * i + 1] = -dN_dx[i] * F01 + dN_dy[i] * F00

        # Linear elasticity constitutive matrix (deviatoric + shear)
        D_dev = mu * np.array([
            [4.0 / 3.0, -2.0 / 3.0, 0.0],
            [-2.0 / 3.0, 4.0 / 3.0, 0.0],
            [0.0, 0.0, 1.0]
        ])

        # Deviatoric contributions
        strain = np.array([grad_u_00, grad_u_11, grad_u_01 + grad_u_10])
        sigma_dev = D_dev @ strain

        f_dev += (B.T @ sigma_dev) * w
        K_dev += (B.T @ D_dev @ B) * w

        V_elem += w
        int_J_minus_1 += (J_gp - 1.0) * w
        B_vol_int += b_vol * w

    # Statically condensed pressure p_condensed = (K_bulk / V_elem) * int_J_minus_1
    p_condensed = (K_bulk / V_elem) * int_J_minus_1

    # Volumetric force & stiffness
    f_vol = p_condensed * B_vol_int
    K_vol = (K_bulk / V_elem) * np.outer(B_vol_int, B_vol_int)

    f_int = f_dev + f_vol
    K_e = K_dev + K_vol

    return f_int, K_e, p_condensed


@numba.njit(parallel=True, fastmath=True)
def assemble_q4_hybrid_elements_numba(
    coords: np.ndarray,       # (N_nodes, 2)
    conn: np.ndarray,         # (N_elem, 4)
    u_global: np.ndarray,     # (N_dofs,)
    E_arr: np.ndarray,        # (N_elem,)
    nu_arr: np.ndarray,       # (N_elem,)
    thickness: float = 1.0,
):
    """Assemble global internal force vector and element stiffness matrices for all Q4_HYBRID elements in parallel."""
    n_elem = conn.shape[0]
    f_elem_all = np.zeros((n_elem, 8))
    K_elem_all = np.zeros((n_elem, 8, 8))
    p_elem_all = np.zeros(n_elem)

    for e in numba.prange(n_elem):
        nodes = conn[e]
        elem_coords = np.zeros((4, 2))
        elem_u = np.zeros(8)

        for i in range(4):
            nid = nodes[i]
            elem_coords[i, 0] = coords[nid, 0]
            elem_coords[i, 1] = coords[nid, 1]
            elem_u[2 * i] = u_global[2 * nid]
            elem_u[2 * i + 1] = u_global[2 * nid + 1]

        f_e, K_e, p_e = _compute_q4_hybrid_single_element(
            elem_coords, elem_u, E_arr[e], nu_arr[e], thickness
        )

        f_elem_all[e] = f_e
        K_elem_all[e] = K_e
        p_elem_all[e] = p_e

    return f_elem_all, K_elem_all, p_elem_all
