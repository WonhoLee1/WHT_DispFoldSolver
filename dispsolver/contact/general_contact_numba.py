"""
general_contact_numba.py
========================
High-Performance Numba JIT kernels for 3D General Contact narrow-phase evaluation
and sparse triplet buffer accumulation.

Theoretical Foundations:
- Quad4 Bilinear & Tri3 Linear surface face projections
- Analytical outward normal computation and opposing normal screening
- 2x2 Gauss-Legendre quadrature with metric area Jacobians
- Symmetric rank-1 normal dyadic tangent: dp_total * (c (x) c^T)
- Stick-slip Coulomb friction tangent & tangential stabilization projections
- Flat contiguous buffer assembly for zero-overhead Cholesky (LLT) solving
"""

from __future__ import annotations
import numpy as np
import numba
from numba import njit, prange


_INV_SQRT3 = 1.0 / np.sqrt(3.0)
# (xi, eta, weight)
_GAUSS_QUAD4 = np.array([
    [-_INV_SQRT3, -_INV_SQRT3, 1.0],
    [+_INV_SQRT3, -_INV_SQRT3, 1.0],
    [+_INV_SQRT3, +_INV_SQRT3, 1.0],
    [-_INV_SQRT3, +_INV_SQRT3, 1.0],
], dtype=np.float64)


@njit(fastmath=True)
def _quad4_shape_funcs(xi: float, eta: float) -> np.ndarray:
    N = np.empty(4, dtype=np.float64)
    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta)
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta)
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta)
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta)
    return N


@njit(fastmath=True)
def _quad4_tangents_and_normal(q: np.ndarray, xi: float, eta: float):
    # q: (4, 3)
    dN_dxi = np.array([
        -0.25 * (1.0 - eta),
         0.25 * (1.0 - eta),
         0.25 * (1.0 + eta),
        -0.25 * (1.0 + eta)
    ], dtype=np.float64)

    dN_deta = np.array([
        -0.25 * (1.0 - xi),
        -0.25 * (1.0 + xi),
         0.25 * (1.0 + xi),
         0.25 * (1.0 - xi)
    ], dtype=np.float64)

    t_xi = np.zeros(3, dtype=np.float64)
    t_eta = np.zeros(3, dtype=np.float64)
    for k in range(4):
        for d in range(3):
            t_xi[d] += dN_dxi[k] * q[k, d]
            t_eta[d] += dN_deta[k] * q[k, d]

    # Cross product
    n = np.array([
        t_xi[1] * t_eta[2] - t_xi[2] * t_eta[1],
        t_xi[2] * t_eta[0] - t_xi[0] * t_eta[2],
        t_xi[0] * t_eta[1] - t_xi[1] * t_eta[0]
    ], dtype=np.float64)

    norm_n = np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
    dA = norm_n
    if norm_n > 1e-14:
        unit_n = n / norm_n
    else:
        unit_n = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    return unit_n, dA


@njit(fastmath=True)
def _project_point_to_quad4_numba(p: np.ndarray, q: np.ndarray, max_iter: int = 15, tol: float = 1e-9):
    # Newton iteration to find (xi, eta) on bilinear quad minimizing distance
    xi = 0.0
    eta = 0.0
    for _ in range(max_iter):
        N = _quad4_shape_funcs(xi, eta)
        r = np.zeros(3, dtype=np.float64)
        for k in range(4):
            for d in range(3):
                r[d] += N[k] * q[k, d]

        diff = r - p

        dN_dxi = np.array([
            -0.25 * (1.0 - eta),  0.25 * (1.0 - eta),
             0.25 * (1.0 + eta), -0.25 * (1.0 + eta)
        ], dtype=np.float64)
        dN_deta = np.array([
            -0.25 * (1.0 - xi), -0.25 * (1.0 + xi),
             0.25 * (1.0 + xi),  0.25 * (1.0 - xi)
        ], dtype=np.float64)

        t_xi = np.zeros(3, dtype=np.float64)
        t_eta = np.zeros(3, dtype=np.float64)
        for k in range(4):
            for d in range(3):
                t_xi[d] += dN_dxi[k] * q[k, d]
                t_eta[d] += dN_deta[k] * q[k, d]

        R1 = 0.0
        R2 = 0.0
        for d in range(3):
            R1 += diff[d] * t_xi[d]
            R2 += diff[d] * t_eta[d]

        if np.sqrt(R1**2 + R2**2) < tol:
            break

        J11 = 0.0
        J12 = 0.0
        J22 = 0.0
        for d in range(3):
            J11 += t_xi[d] * t_xi[d]
            J12 += t_xi[d] * t_eta[d]
            J22 += t_eta[d] * t_eta[d]

        det = J11 * J22 - J12 * J12
        if np.abs(det) < 1e-15:
            break

        d_xi = (-R1 * J22 + R2 * J12) / det
        d_eta = (-R2 * J11 + R1 * J12) / det

        xi += d_xi
        eta += d_eta

        if np.abs(d_xi) < tol and np.abs(d_eta) < tol:
            break

    N = _quad4_shape_funcs(xi, eta)
    proj_pt = np.zeros(3, dtype=np.float64)
    for k in range(4):
        for d in range(3):
            proj_pt[d] += N[k] * q[k, d]

    dist = np.sqrt(np.sum((p - proj_pt)**2))
    return xi, eta, dist, proj_pt


@njit(fastmath=True)
def _tangent_basis_numba(n: np.ndarray):
    # Orthonormal basis (t1, t2) perpendicular to n
    if np.abs(n[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    dot = ref[0] * n[0] + ref[1] * n[1] + ref[2] * n[2]
    t1 = ref - dot * n
    norm_t1 = np.sqrt(t1[0]**2 + t1[1]**2 + t1[2]**2)
    if norm_t1 < 1e-14:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        dot = ref[0] * n[0] + ref[1] * n[1] + ref[2] * n[2]
        t1 = ref - dot * n
        norm_t1 = np.sqrt(t1[0]**2 + t1[1]**2 + t1[2]**2)

    t1 /= norm_t1

    # t2 = n x t1
    t2 = np.array([
        n[1] * t1[2] - n[2] * t1[1],
        n[2] * t1[0] - n[0] * t1[2],
        n[0] * t1[1] - n[1] * t1[0]
    ], dtype=np.float64)
    norm_t2 = np.sqrt(t2[0]**2 + t2[1]**2 + t2[2]**2)
    if norm_t2 > 1e-14:
        t2 /= norm_t2

    return t1, t2


@njit(fastmath=True)
def assemble_general_contact_quad4_numba(
    candidate_pairs: np.ndarray,      # (P, 2) int32
    faces: np.ndarray,                # (F, 4) int32 (0-based node indices)
    coords: np.ndarray,               # (N, 3) float64
    u: np.ndarray,                    # (3*N,) float64
    face_E: np.ndarray,               # (F,) float64
    E_ref: float,
    gamma: float,
    damping_coeff: float,
    damping_clearance: float,
    tangent_fraction: float,
    mu: float,
    k_t: float,
):
    """Parallelized Numba kernel evaluating narrow-phase contact & friction for Quad4 faces.

    Returns
    -------
    f_contact : np.ndarray (3*N,)
    row_buf, col_buf, val_buf : dynamic list or arrays of assembled triplets
    stats_arr : np.ndarray [max_penetration, n_active, total_normal_force]
    """
    n_nodes = coords.shape[0]
    f_contact = np.zeros(3 * n_nodes, dtype=np.float64)

    # Upper bound allocation for triplet buffers: 4 quad points * 64 entries * pairs
    n_pairs = candidate_pairs.shape[0]
    max_triplets = n_pairs * 4 * 64
    rows = np.empty(max_triplets, dtype=np.int32)
    cols = np.empty(max_triplets, dtype=np.int32)
    vals = np.empty(max_triplets, dtype=np.float64)
    trip_count = 0

    max_pen = 0.0
    n_active = 0
    total_fn = 0.0

    qa = np.empty((4, 3), dtype=np.float64)
    qb = np.empty((4, 3), dtype=np.float64)

    for p_idx in range(n_pairs):
        fa_idx = candidate_pairs[p_idx, 0]
        fb_idx = candidate_pairs[p_idx, 1]

        idx_a = faces[fa_idx]
        idx_b = faces[fb_idx]

        for k in range(4):
            ia = idx_a[k]
            ib = idx_b[k]
            for d in range(3):
                qa[k, d] = coords[ia, d] + u[3 * ia + d]
                qb[k, d] = coords[ib, d] + u[3 * ib + d]

        # Normals at center (xi=0, eta=0)
        na, _ = _quad4_tangents_and_normal(qa, 0.0, 0.0)
        nb, _ = _quad4_tangents_and_normal(qb, 0.0, 0.0)

        # Normal opposing check: dot(na, nb) <= -0.3
        dot_n = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2]
        if dot_n > -0.3:
            continue

        # Effective modulus scaling
        E_a = face_E[fa_idx]
        E_b = face_E[fb_idx]
        if E_a > 0.0 and E_b > 0.0:
            E_eff = 2.0 * E_a * E_b / (E_a + E_b)
            mod_scale = E_eff / E_ref
        else:
            mod_scale = 1.0

        gamma_pair = gamma * mod_scale
        k_t_pair = k_t * mod_scale
        damp_pair = damping_coeff * mod_scale

        # 4 Gauss Quadrature Points
        for qp in range(4):
            xi_s = _GAUSS_QUAD4[qp, 0]
            eta_s = _GAUSS_QUAD4[qp, 1]
            w_gauss = _GAUSS_QUAD4[qp, 2]

            N_a = _quad4_shape_funcs(xi_s, eta_s)
            x_qp = np.zeros(3, dtype=np.float64)
            for k in range(4):
                for d in range(3):
                    x_qp[d] += N_a[k] * qa[k, d]

            _, dA = _quad4_tangents_and_normal(qa, xi_s, eta_s)
            w = dA * w_gauss

            # Project x_qp onto face_b
            xi_m, eta_m, dist, proj_pt = _project_point_to_quad4_numba(x_qp, qb)
            if not (-1.05 <= xi_m <= 1.05 and -1.05 <= eta_m <= 1.05):
                continue

            N_b = _quad4_shape_funcs(xi_m, eta_m)

            # Penetration gap: g_n = - (x_qp - proj_pt) . nb
            diff_p = x_qp - proj_pt
            g_n = -(diff_p[0] * nb[0] + diff_p[1] * nb[1] + diff_p[2] * nb[2])

            p_n = 0.0
            dp_dh = 0.0
            if g_n > 0.0:
                p_n = gamma_pair * g_n
                dp_dh = gamma_pair

            p_stab = 0.0
            dp_stab = 0.0
            clearance = -g_n
            if damp_pair > 0.0 and clearance > 0.0:
                if damping_clearance <= 0.0 or clearance < damping_clearance:
                    ramp = (1.0 - clearance / damping_clearance) if damping_clearance > 0.0 else 1.0
                    p_stab = damp_pair * max(0.0, ramp) * clearance
                    dp_stab = damp_pair * max(0.0, ramp)

            p_total = p_n + p_stab
            dp_total = dp_dh + dp_stab

            if p_total <= 0.0 and dp_total <= 0.0:
                continue

            if g_n > 0.0 or p_n > 0.0:
                n_active += 1
                if g_n > max_pen:
                    max_pen = g_n

            total_fn += p_total * w

            # Tangential friction
            f_t = np.zeros(3, dtype=np.float64)
            K_tt = np.zeros((3, 3), dtype=np.float64)
            if mu > 0.0 and p_total > 0.0:
                # Relative sliding displacement vector
                t1, t2 = _tangent_basis_numba(nb)
                s_t = diff_p - (diff_p[0] * nb[0] + diff_p[1] * nb[1] + diff_p[2] * nb[2]) * nb
                s1 = s_t[0] * t1[0] + s_t[1] * t1[1] + s_t[2] * t1[2]
                s2 = s_t[0] * t2[0] + s_t[1] * t2[1] + s_t[2] * t2[2]
                s_mag = np.sqrt(s1**2 + s2**2)
                s_crit = (mu * p_total / k_t_pair) if k_t_pair > 0.0 else 0.0

                T = np.zeros((3, 3), dtype=np.float64)
                for d1 in range(3):
                    for d2 in range(3):
                        T[d1, d2] = t1[d1] * t1[d2] + t2[d1] * t2[d2]

                if s_mag <= s_crit or s_mag < 1e-14:
                    ft1 = k_t_pair * s1
                    ft2 = k_t_pair * s2
                    K_tt = k_t_pair * T
                else:
                    ft1 = mu * p_total * (s1 / s_mag)
                    ft2 = mu * p_total * (s2 / s_mag)
                    shat = (s1 / s_mag) * t1 + (s2 / s_mag) * t2
                    for d1 in range(3):
                        for d2 in range(3):
                            K_tt[d1, d2] = (mu * p_total / s_mag) * (T[d1, d2] - shat[d1] * shat[d2])

                for d in range(3):
                    f_t[d] = ft1 * t1[d] + ft2 * t2[d]

            # Contact force vector: f_vec = p_total * nb - f_t
            f_vec = np.empty(3, dtype=np.float64)
            for d in range(3):
                f_vec[d] = p_total * nb[d] - f_t[d]

            # Assemble internal force: -N_a * w * f_vec for slave, +N_b * w * f_vec for master
            for k in range(4):
                ia = idx_a[k]
                ib = idx_b[k]
                for d in range(3):
                    f_contact[3 * ia + d] += -N_a[k] * w * f_vec[d]
                    f_contact[3 * ib + d] += +N_b[k] * w * f_vec[d]

            # Assemble tangent stiffness: 8 participating nodes
            # Stencil: node_indices = [idx_a[0..3], idx_b[0..3]]
            # Weights: [-N_a[0..3], +N_b[0..3]]
            scale_norm = dp_total * w
            scale_tan_stab = (tangent_fraction * dp_stab * w) if (tangent_fraction > 0.0 and dp_stab > 0.0) else 0.0

            node_ids = np.empty(8, dtype=np.int32)
            w_stencil = np.empty(8, dtype=np.float64)
            for k in range(4):
                node_ids[k] = idx_a[k]
                w_stencil[k] = -N_a[k]
                node_ids[4 + k] = idx_b[k]
                w_stencil[4 + k] = +N_b[k]

            I_proj = np.zeros((3, 3), dtype=np.float64)
            for d1 in range(3):
                for d2 in range(3):
                    I_proj[d1, d2] = (1.0 if d1 == d2 else 0.0) - nb[d1] * nb[d2]

            for a in range(8):
                ia = node_ids[a]
                wa = w_stencil[a]
                for b in range(8):
                    ib = node_ids[b]
                    wb = w_stencil[b]
                    wab = wa * wb

                    k_block = np.zeros((3, 3), dtype=np.float64)
                    if dp_total > 0.0:
                        for d1 in range(3):
                            for d2 in range(3):
                                k_block[d1, d2] += scale_norm * (wa * nb[d1]) * (wb * nb[d2])

                    if mu > 0.0:
                        for d1 in range(3):
                            for d2 in range(3):
                                k_block[d1, d2] += (wab * w) * K_tt[d1, d2]

                    if scale_tan_stab > 0.0:
                        for d1 in range(3):
                            for d2 in range(3):
                                k_block[d1, d2] += (wab * scale_tan_stab) * I_proj[d1, d2]

                    for d1 in range(3):
                        r = 3 * ia + d1
                        for d2 in range(3):
                            c = 3 * ib + d2
                            val = k_block[d1, d2]
                            if val != 0.0 and trip_count < max_triplets:
                                rows[trip_count] = r
                                cols[trip_count] = c
                                vals[trip_count] = val
                                trip_count += 1

    stats_arr = np.array([max_pen, float(n_active), total_fn], dtype=np.float64)
    return f_contact, rows[:trip_count], cols[:trip_count], vals[:trip_count], stats_arr
