"""
q4_sri_hybrid_numba.py
======================
Numba C-JIT accelerated implementation for Selective Reduced Integration (SRI) + Q1P0 Hydrostatic Pressure Hybrid Elements
(Q4_HYBRID_SRI, Q4_COROTATIONAL_HYBRID_SRI).

Provides multi-threaded parallel element force vector and tangent stiffness matrix assembly.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(fastmath=True)
def _compute_q4_sri_hybrid_single_element(
    coords: np.ndarray,
    u_elem: np.ndarray,
    E: float,
    nu: float,
    thickness: float = 1.0,
):
    """Compute element internal force vector (8,) and stiffness matrix (8,8) for SRI Hybrid Q4 element."""
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    K_bulk = lam + (2.0 / 3.0) * mu

    # Plane strain deviatoric constitutive matrix (3,3)
    C_dev = np.array([
        [4.0/3.0 * mu, -2.0/3.0 * mu, 0.0],
        [-2.0/3.0 * mu, 4.0/3.0 * mu, 0.0],
        [0.0, 0.0, mu]
    ])

    gp_pos = 1.0 / np.sqrt(3.0)
    gp_coords = np.array([
        [-gp_pos, -gp_pos],
        [ gp_pos, -gp_pos],
        [ gp_pos,  gp_pos],
        [-gp_pos,  gp_pos]
    ])
    gp_weights = np.ones(4)

    # Compute centroid B-matrix row 2 (shear) at (0, 0)
    dN_dxi_0 = 0.25 * np.array([-1.0, 1.0, 1.0, -1.0])
    dN_deta_0 = 0.25 * np.array([-1.0, -1.0, 1.0, 1.0])

    J00 = np.sum(dN_dxi_0 * coords[:, 0])
    J01 = np.sum(dN_dxi_0 * coords[:, 1])
    J10 = np.sum(dN_deta_0 * coords[:, 0])
    J11 = np.sum(dN_deta_0 * coords[:, 1])
    detJ0 = J00 * J11 - J01 * J10
    invJ0 = np.array([[J11, -J01], [-J10, J00]]) / detJ0

    B_shear_0 = np.zeros(8)
    for i in range(4):
        dN_dx_0 = invJ0[0, 0] * dN_dxi_0[i] + invJ0[0, 1] * dN_deta_0[i]
        dN_dy_0 = invJ0[1, 0] * dN_dxi_0[i] + invJ0[1, 1] * dN_deta_0[i]
        B_shear_0[2 * i] = dN_dy_0
        B_shear_0[2 * i + 1] = dN_dx_0

    f_dev = np.zeros(8)
    K_dev = np.zeros((8, 8))
    V_elem = 0.0
    int_eps_vol = 0.0
    B_vol_int = np.zeros(8)

    for gp in range(4):
        xi = gp_coords[gp, 0]
        eta = gp_coords[gp, 1]

        dN_dxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
        dN_deta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

        J0 = np.sum(dN_dxi * coords[:, 0])
        J1 = np.sum(dN_dxi * coords[:, 1])
        J2 = np.sum(dN_deta * coords[:, 0])
        J3 = np.sum(dN_deta * coords[:, 1])
        detJ = J0 * J3 - J1 * J2
        invJ = np.array([[J3, -J1], [-J2, J0]]) / detJ

        B_sri = np.zeros((3, 8))
        B_vol_gp = np.zeros(8)
        for i in range(4):
            dN_dx = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i]
            dN_dy = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i]

            B_sri[0, 2 * i] = dN_dx
            B_sri[1, 2 * i + 1] = dN_dy
            B_sri[2, 2 * i] = B_shear_0[2 * i]
            B_sri[2, 2 * i + 1] = B_shear_0[2 * i + 1]

            B_vol_gp[2 * i] = dN_dx
            B_vol_gp[2 * i + 1] = dN_dy

        eps = B_sri @ u_elem
        sigma_dev = C_dev @ eps
        w = detJ * gp_weights[gp] * thickness

        f_dev += B_sri.T @ sigma_dev * w
        K_dev += B_sri.T @ C_dev @ B_sri * w

        V_elem += w
        eps_vol = B_vol_gp @ u_elem
        int_eps_vol += eps_vol * w
        B_vol_int += B_vol_gp * w

    # Statically condensed hydrostatic pressure force & stiffness
    p_condensed = K_bulk * (int_eps_vol / V_elem)
    f_vol = B_vol_int * p_condensed
    K_vol = (K_bulk / V_elem) * np.outer(B_vol_int, B_vol_int)

    f_int = f_dev + f_vol
    K_e = K_dev + K_vol
    return f_int, K_e


@numba.njit(parallel=True, fastmath=True)
def assemble_q4_sri_hybrid_elements_numba(
    coords_all: np.ndarray,
    u_all: np.ndarray,
    elems: np.ndarray,
    E: float,
    nu: float,
    thickness: float = 1.0,
):
    """Assemble all SRI Hybrid Q4 elements in parallel."""
    n_elems = elems.shape[0]
    f_elems = np.zeros((n_elems, 8))
    K_elems = np.zeros((n_elems, 8, 8))

    for e in numba.prange(n_elems):
        elem_nodes = elems[e]
        e_coords = coords_all[elem_nodes]
        e_u = u_all[elem_nodes].reshape(8)

        f_e, K_e = _compute_q4_sri_hybrid_single_element(e_coords, e_u, E, nu, thickness)
        f_elems[e] = f_e
        K_elems[e] = K_e

    return f_elems, K_elems


@numba.njit(fastmath=True)
def _compute_element_rotation_numba(coords_ref: np.ndarray, coords_curr: np.ndarray) -> np.ndarray:
    """Compute rigid element rotation matrix R (2,2) from reference to current coordinates."""
    dx_ref = coords_ref[1] - coords_ref[0] + coords_ref[2] - coords_ref[3]
    dy_ref = coords_ref[3] - coords_ref[0] + coords_ref[2] - coords_ref[1]

    dx_curr = coords_curr[1] - coords_curr[0] + coords_curr[2] - coords_curr[3]
    dy_curr = coords_curr[3] - coords_curr[0] + coords_curr[2] - coords_curr[1]

    theta_ref = np.arctan2(dx_ref[1], dx_ref[0])
    theta_curr = np.arctan2(dx_curr[1], dx_curr[0])
    dtheta = theta_curr - theta_ref

    cos_t = np.cos(dtheta)
    sin_t = np.sin(dtheta)

    return np.array([[cos_t, -sin_t], [sin_t, cos_t]])


@numba.njit(parallel=True, fastmath=True)
def assemble_q4_corotational_sri_hybrid_elements_numba(
    coords_all: np.ndarray,
    u_all: np.ndarray,
    elems: np.ndarray,
    E: float,
    nu: float,
    thickness: float = 1.0,
):
    """Assemble all Co-rotational SRI Hybrid Q4 elements in parallel."""
    n_elems = elems.shape[0]
    f_elems = np.zeros((n_elems, 8))
    K_elems = np.zeros((n_elems, 8, 8))

    for e in numba.prange(n_elems):
        elem_nodes = elems[e]
        e_coords_ref = coords_all[elem_nodes]
        e_u_g = u_all[elem_nodes].reshape(8)
        e_coords_curr = e_coords_ref + e_u_g.reshape((4, 2))

        # Compute rigid element rotation R
        R_elem = _compute_element_rotation_numba(e_coords_ref, e_coords_curr)

        # Build block rotation T (8,8)
        T8 = np.zeros((8, 8))
        for i in range(4):
            T8[2*i:2*i+2, 2*i:2*i+2] = R_elem

        u_l = (e_coords_curr @ R_elem - e_coords_ref).reshape(8)

        f_l, K_l = _compute_q4_sri_hybrid_single_element(e_coords_ref, u_l, E, nu, thickness)

        f_elems[e] = T8 @ f_l
        K_elems[e] = T8 @ K_l @ T8.T

    return f_elems, K_elems
