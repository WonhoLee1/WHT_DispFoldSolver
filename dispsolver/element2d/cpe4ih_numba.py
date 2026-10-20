"""
cpe4ih_numba.py
===============
Numba JIT accelerated 4-node Incompatible Modes + Hybrid Linear Pressure Quadrilateral (CPE4IH).

Theory (Abaqus CPE4IH — "4-node bilinear, incompatible modes, hybrid with linear pressure"):
- Combines Simo & Rifai (1990) 4-mode Enhanced Assumed Strain (EAS) kinematics with
  Simo-Taylor-Pister / Herrmann 3-DOF linear hybrid hydrostatic pressure (p0, p_xi, p_eta).
- Piola-scaled linear pressure basis n_p_lin(xi, eta) = (detJ0 / detJ) * [xi, eta]^T ensures:
  1. Exact L2-orthogonality between constant mean pressure p0 and zero-mean fluctuating modes
     (alpha, p_lin) on arbitrary distorted quadrilateral meshes.
  2. Exact satisfaction of Irons constant-strain patch test on distorted meshes (alpha = 0, p_lin = 0).
  3. Two-stage Schur complement condensation (condensing alpha via K_aa_dev^-1 first, then p_lin
     via S_pp^-1) completely avoids O(K) - O(K) floating-point cancellation as nu -> 0.5,
     while preserving a strictly symmetric positive-definite (SPD) 8x8 element stiffness matrix.
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange

from dispsolver.element2d.cpe4i_numba import _shape_derivs_cpe4i, _voigt_transform_2d, _GPS, _WTS, solve_small
from dispsolver.material3d.numba_materials import (
    material_dispatch_3d,
    MAT_LINEAR_ELASTIC,
    MAT_CUSTOM_ELASTIC,
    MAT_J2_PLASTICITY,
    MAT_J2_PLASTICITY_TABULAR,
    MAT_HYPERELASTIC_NEOHOOKEAN,
    MAT_HYPERELASTIC_ARRUDA_BOYCE,
    MAT_HYPERELASTIC_YEOH,
    MAT_HYPERELASTIC_MOONEY_RIVLIN,
    MAT_HYPERELASTIC_OGDEN,
    MAT_HYPERELASTIC_POLYNOMIAL,
    MAT_HYPERELASTIC_MARLOW,
    MAT_HYPER_VISCOELASTIC,
)


@njit(fastmath=True)
def _decode_shear_bulk_2d(mat_type: int, props: np.ndarray):
    """Decode baseline shear modulus G and bulk modulus K from material props."""
    if mat_type == MAT_CUSTOM_ELASTIC:
        lam = props[1]
        mu = props[21]
        if mu <= 0.0:
            E_ = props[0] if props[0] > 0.0 else 4000.0
            nu_ = props[1] if (props[1] >= 0.0 and props[1] < 0.5) else 0.3
            nu_eff = min(nu_, 0.4999999)
            mu = E_ / (2.0 * (1.0 + nu_eff))
            K = E_ / (3.0 * max(1.0 - 2.0 * nu_eff, 1e-9))
        else:
            K = lam + (2.0 / 3.0) * mu
    elif mat_type == MAT_LINEAR_ELASTIC or mat_type == MAT_J2_PLASTICITY or mat_type == MAT_J2_PLASTICITY_TABULAR:
        E_ = props[0]
        nu_ = min(props[1], 0.4999999)
        mu = E_ / (2.0 * (1.0 + nu_))
        K = E_ / (3.0 * max(1.0 - 2.0 * nu_, 1e-9))
    elif mat_type == MAT_HYPERELASTIC_ARRUDA_BOYCE:
        mu = props[0]
        p2 = props[2]
        if p2 <= 0.05 and p2 > 0.0:
            K = 2.0 / max(p2, 1e-12)
        else:
            K = max(p2, 1e-8)
    elif mat_type == MAT_HYPERELASTIC_YEOH or mat_type == MAT_HYPERELASTIC_MARLOW:
        c10 = props[0]
        d1 = props[3]
        mu = 2.0 * c10
        K = 2.0 / max(d1, 1e-12)
    elif mat_type == MAT_HYPERELASTIC_MOONEY_RIVLIN:
        c10 = props[0]
        c01 = props[1]
        d1 = props[2]
        mu = 2.0 * (c10 + c01)
        K = 2.0 / max(d1, 1e-12)
    elif mat_type == MAT_HYPERELASTIC_OGDEN:
        n_terms = int(props[0])
        d1 = props[1]
        mu = 0.0
        for p in range(min(n_terms, 6)):
            mu += props[2 + 2 * p]
        K = 2.0 / max(d1, 1e-12)
    elif mat_type == MAT_HYPERELASTIC_POLYNOMIAL:
        c10 = props[0]
        c01 = props[1]
        d1 = props[5]
        mu = 2.0 * (c10 + c01)
        K = 2.0 / max(d1, 1e-12)
    elif mat_type == MAT_HYPER_VISCOELASTIC:
        base_model = int(props[0])
        if base_model == MAT_HYPERELASTIC_ARRUDA_BOYCE:
            mu = props[1]
            K = props[3]
        elif base_model == MAT_HYPERELASTIC_YEOH:
            mu = 2.0 * props[1]
            d1 = props[5] if len(props) > 5 else 0.01
            K = 2.0 / max(d1, 1e-12)
        else:
            mu = props[1]
            K = props[2]
    else:
        p0 = props[0]
        p1 = props[1]
        if p1 <= 0.05 and p0 > 0.0:
            mu = 2.0 * p0
            K = 2.0 / max(p1, 1e-12)
        elif p1 < 0.5 and p0 > 1.0:
            mu = p0 / (2.0 * (1.0 + p1))
            K = p0 / (3.0 * max(1.0 - 2.0 * p1, 1e-9))
        else:
            mu = max(p0, 1e-8)
            K = max(p1, 1e-8)
    return mu, max(K, 1e-8)



@njit(fastmath=True)
def compute_cpe4ih_nonlinear_numba(
    coords: np.ndarray,      # (4, 2)
    u_elem: np.ndarray,      # (8,)
    mat_type: int,
    props: np.ndarray,       # (64,)
    sdvs: np.ndarray,        # (4, n_sdv), updated in place with the converged state
    dt: float,
):
    """CPE4IH for a path-dependent / nonlinear material (plasticity, hyperelasticity).

    The unknowns (alpha: 4 incompatible-mode amplitudes, p_lin: 2 linear-pressure amplitudes) are iterated to
    element equilibrium for the CURRENT displacements:

        R_alpha = int M^T s_dev(B u + M alpha) dV + Kap p_lin = 0
        R_p     = Kup^T u + Kap^T alpha - Hpp p_lin / K        = 0

    The material sees the strain B u + M alpha, not B u: the enhanced strain carries the bending, so a
    law evaluated on B u alone has the wrong plastic strain and the wrong tangent (measured: plastic bending
    of a strip, kappa = 0.04, did not converge and was 2.3 % off the fibre oracle). For a linear material one
    Newton step from zero is exact and this routine reproduces compute_cpe4ih_element_numba. The material is
    evaluated from the state at the START of the increment on every iteration and only the converged
    evaluation is written back (same rule as CPE4I).
    """
    mu, K = _decode_shear_bulk_2d(mat_type, props)

    dN0 = _shape_derivs_cpe4i(0.0, 0.0)
    J0 = dN0 @ coords
    detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]
    if detJ0 <= 1e-14:
        return np.zeros(8, dtype=np.float64), np.zeros((8, 8), dtype=np.float64), 1
    T0_inv_T = _voigt_transform_2d(J0)

    # geometry of the four Gauss points
    Bq = np.zeros((4, 3, 8), dtype=np.float64)
    Mq = np.zeros((4, 3, 4), dtype=np.float64)
    npq = np.zeros((4, 2), dtype=np.float64)
    dVq = np.zeros(4, dtype=np.float64)
    vol_vec = np.zeros(8, dtype=np.float64)
    tot_vol = 0.0
    M0 = np.zeros((3, 4), dtype=np.float64)
    for q in range(4):
        xi = _GPS[q, 0]
        eta = _GPS[q, 1]
        dN_dxi = _shape_derivs_cpe4i(xi, eta)
        J = dN_dxi @ coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 1e-14:
            return np.zeros(8, dtype=np.float64), np.zeros((8, 8), dtype=np.float64), 1
        invJ00 = J[1, 1] / detJ
        invJ01 = -J[0, 1] / detJ
        invJ10 = -J[1, 0] / detJ
        invJ11 = J[0, 0] / detJ
        dV = detJ * _WTS[q]
        dVq[q] = dV
        tot_vol += dV
        for a in range(4):
            dN_dx = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
            dN_dy = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]
            Bq[q, 0, 2 * a + 0] = dN_dx
            Bq[q, 1, 2 * a + 1] = dN_dy
            Bq[q, 2, 2 * a + 0] = dN_dy
            Bq[q, 2, 2 * a + 1] = dN_dx
            vol_vec[2 * a + 0] += dN_dx * dV
            vol_vec[2 * a + 1] += dN_dy * dV
        M0.fill(0.0)
        M0[0, 0] = xi
        M0[1, 1] = eta
        M0[2, 2] = xi
        M0[2, 3] = eta
        scale = detJ0 / detJ
        Mq[q] = scale * (T0_inv_T @ M0)
        npq[q, 0] = scale * xi
        npq[q, 1] = scale * eta

    eps_vol_mean = np.sum(vol_vec * u_elem) / max(tot_vol, 1e-14)
    p0 = K * eps_vol_mean

    Kup = np.zeros((8, 2), dtype=np.float64)
    Kap = np.zeros((4, 2), dtype=np.float64)
    Hpp = np.zeros((2, 2), dtype=np.float64)
    for q in range(4):
        dV = dVq[q]
        for i in range(8):
            bv_i = Bq[q, 0, i] + Bq[q, 1, i]
            Kup[i, 0] += bv_i * npq[q, 0] * dV
            Kup[i, 1] += bv_i * npq[q, 1] * dV
        for j in range(4):
            mv_j = Mq[q, 0, j] + Mq[q, 1, j]
            Kap[j, 0] += mv_j * npq[q, 0] * dV
            Kap[j, 1] += mv_j * npq[q, 1] * dV
        Hpp[0, 0] += npq[q, 0] * npq[q, 0] * dV
        Hpp[0, 1] += npq[q, 0] * npq[q, 1] * dV
        Hpp[1, 0] += npq[q, 1] * npq[q, 0] * dV
        Hpp[1, 1] += npq[q, 1] * npq[q, 1] * dV

    P_dev_3d = np.eye(6, dtype=np.float64)
    for r in range(3):
        for c in range(3):
            P_dev_3d[r, c] -= 1.0 / 3.0

    n_sdv = sdvs.shape[1] if (sdvs is not None and sdvs.ndim == 2) else 0
    sdv0 = np.zeros((4, max(n_sdv, 1)), dtype=np.float64)
    for q in range(4):
        for s_i in range(n_sdv):
            sdv0[q, s_i] = sdvs[q, s_i]
    sdv_last = sdv0.copy()

    alpha = np.zeros(4, dtype=np.float64)
    p_lin = np.zeros(2, dtype=np.float64)
    Kuu = np.zeros((8, 8), dtype=np.float64)
    Kua = np.zeros((8, 4), dtype=np.float64)
    Kaa = np.zeros((4, 4), dtype=np.float64)
    fu = np.zeros(8, dtype=np.float64)
    fa = np.zeros(4, dtype=np.float64)
    has_error = 0
    F_dummy = np.eye(3, dtype=np.float64)
    converged = False

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
            tr_eps = eps[0] + eps[1]
            E_voigt = np.zeros(6, dtype=np.float64)
            E_voigt[0] = eps[0] - tr_eps / 3.0 + eps_vol_mean / 3.0
            E_voigt[1] = eps[1] - tr_eps / 3.0 + eps_vol_mean / 3.0
            E_voigt[2] = -tr_eps / 3.0 + eps_vol_mean / 3.0
            E_voigt[3] = eps[2]
            sdv_pt = sdv0[q, :n_sdv].copy() if n_sdv > 0 else np.zeros(7, dtype=np.float64)
            S_voigt, C_6x6, sdv_new, mat_err = material_dispatch_3d(
                mat_type, props, sdv_pt, E_voigt, F_dummy, 1.0, dt
            )
            if mat_err != 0:
                has_error = 1
            for s_i in range(min(len(sdv_new), n_sdv)):
                sdv_last[q, s_i] = sdv_new[s_i]

            tr_s = (S_voigt[0] + S_voigt[1] + S_voigt[2]) / 3.0
            s_dev = np.zeros(3, dtype=np.float64)
            s_dev[0] = S_voigt[0] - tr_s
            s_dev[1] = S_voigt[1] - tr_s
            s_dev[2] = S_voigt[3]
            C_dev_6x6 = P_dev_3d @ C_6x6 @ P_dev_3d
            C_dev = np.zeros((3, 3), dtype=np.float64)
            C_dev[0, 0] = C_dev_6x6[0, 0]
            C_dev[0, 1] = C_dev_6x6[0, 1]
            C_dev[0, 2] = C_dev_6x6[0, 3]
            C_dev[1, 0] = C_dev_6x6[1, 0]
            C_dev[1, 1] = C_dev_6x6[1, 1]
            C_dev[1, 2] = C_dev_6x6[1, 3]
            C_dev[2, 0] = C_dev_6x6[3, 0]
            C_dev[2, 1] = C_dev_6x6[3, 1]
            C_dev[2, 2] = C_dev_6x6[3, 3]
            CB = C_dev @ B
            CM = C_dev @ M
            for i in range(8):
                fu[i] += (B[0, i] * s_dev[0] + B[1, i] * s_dev[1] + B[2, i] * s_dev[2]) * dV
                for j in range(8):
                    Kuu[i, j] += (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j]) * dV
                for j in range(4):
                    Kua[i, j] += (B[0, i] * CM[0, j] + B[1, i] * CM[1, j] + B[2, i] * CM[2, j]) * dV
            for i in range(4):
                fa[i] += (M[0, i] * s_dev[0] + M[1, i] * s_dev[1] + M[2, i] * s_dev[2]) * dV
                for j in range(4):
                    Kaa[i, j] += (M[0, i] * CM[0, j] + M[1, i] * CM[1, j] + M[2, i] * CM[2, j]) * dV

        R_alpha = fa + Kap @ p_lin
        R_p = Kup.T @ u_elem + Kap.T @ alpha - (Hpp @ p_lin) / K
        r_norm = np.sqrt(np.sum(R_alpha * R_alpha) + np.sum(R_p * R_p))
        f_scale = max(np.sqrt(np.sum(fu * fu)), 1.0e-3)
        if r_norm <= 1e-12 * f_scale:
            converged = True
            break
        Jm = np.zeros((6, 6), dtype=np.float64)
        Jm[:4, :4] = Kaa
        Jm[:4, 4:] = Kap
        Jm[4:, :4] = Kap.T
        Jm[4:, 4:] = -Hpp / K
        rhs = np.zeros((6, 1), dtype=np.float64)
        rhs[:4, 0] = R_alpha
        rhs[4:, 0] = R_p
        dx, solved = solve_small(Jm, rhs)
        if not solved:
            has_error = 1               # singular enhanced block (collapse): let the solver cut back
            break
        alpha = alpha - dx[:4, 0]
        p_lin = p_lin - dx[4:, 0]
    if (not converged) and has_error == 0:
        R_alpha = fa + Kap @ p_lin
        R_p = Kup.T @ u_elem + Kap.T @ alpha - (Hpp @ p_lin) / K
        r_norm = np.sqrt(np.sum(R_alpha * R_alpha) + np.sum(R_p * R_p))
        if r_norm > 1e-6 * max(np.sqrt(np.sum(fu * fu)), 1.0e-3):
            has_error = 1               # (alpha, p_lin) did not reach equilibrium

    for q in range(4):
        for s_i in range(n_sdv):
            sdvs[q, s_i] = sdv_last[q, s_i]

    invKaa, ok1 = solve_small(Kaa, np.eye(4))
    if not ok1:
        return fu + p0 * vol_vec, Kuu, 1
    Kuu_tilde = Kuu - Kua @ invKaa @ Kua.T
    Kup_tilde = Kup - Kua @ invKaa @ Kap
    Spp = Kap.T @ invKaa @ Kap + (1.0 / K) * Hpp
    invSpp, ok2 = solve_small(Spp, np.eye(2))
    if not ok2:
        return fu + p0 * vol_vec, Kuu, 1
    Ke = Kuu_tilde + Kup_tilde @ invSpp @ Kup_tilde.T
    k_vol_fac = K / max(tot_vol, 1e-14)
    for i in range(8):
        for j in range(8):
            Ke[i, j] += k_vol_fac * vol_vec[i] * vol_vec[j]

    fe = fu + Kup @ p_lin + p0 * vol_vec
    return fe, Ke, has_error


@njit(fastmath=True)
def compute_cpe4ih_element_numba(
    coords: np.ndarray,      # (4, 2)
    u_elem: np.ndarray,      # (8,)
    mat_type: int,
    props: np.ndarray,       # (64,)
    sdvs: np.ndarray,        # (4, n_sdv)
    dt: float,
    elem_controls: np.ndarray = None
):
    """
    Computes statically condensed 8x8 SPD tangent Ke and 8-element internal force fe
    for 1 CPE4IH (4-mode EAS + 3-DOF linear hybrid pressure) element.
    """
    if mat_type != MAT_LINEAR_ELASTIC:
        return compute_cpe4ih_nonlinear_numba(coords, u_elem, mat_type, props, sdvs, dt)

    mu, K = _decode_shear_bulk_2d(mat_type, props)

    # Centroid Jacobian J0 and Simo-Rifai Voigt transformation T0^-T
    dN0 = _shape_derivs_cpe4i(0.0, 0.0)
    J0 = dN0 @ coords
    detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]
    if detJ0 <= 1e-14:
        return np.zeros(8, dtype=np.float64), np.zeros((8, 8), dtype=np.float64), 1

    T0_inv_T = _voigt_transform_2d(J0)

    # Pass 1: Compute total volume V0 and mean volumetric strain-displacement vector vol_vec
    vol_vec = np.zeros(8, dtype=np.float64)
    tot_vol = 0.0
    has_error = 0

    for q in range(4):
        xi = _GPS[q, 0]
        eta = _GPS[q, 1]
        wt = _WTS[q]

        dN_dxi = _shape_derivs_cpe4i(xi, eta)
        J = dN_dxi @ coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 1e-14:
            return np.zeros(8, dtype=np.float64), np.zeros((8, 8), dtype=np.float64), 1

        invJ00 =  J[1, 1] / detJ
        invJ01 = -J[0, 1] / detJ
        invJ10 = -J[1, 0] / detJ
        invJ11 =  J[0, 0] / detJ

        dV = detJ * wt
        tot_vol += dV
        for a in range(4):
            dN_dx = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
            dN_dy = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]
            vol_vec[2 * a + 0] += dN_dx * dV
            vol_vec[2 * a + 1] += dN_dy * dV

    eps_vol_mean = np.sum(vol_vec * u_elem) / max(tot_vol, 1e-14)
    p0 = K * eps_vol_mean

    # Pass 2: Accumulate deviatoric stiffnesses and linear pressure couplings
    Kuu_dev = np.zeros((8, 8), dtype=np.float64)
    Kua_dev = np.zeros((8, 4), dtype=np.float64)
    Kaa_dev = np.zeros((4, 4), dtype=np.float64)

    Kup = np.zeros((8, 2), dtype=np.float64)
    Kap = np.zeros((4, 2), dtype=np.float64)
    Hpp = np.zeros((2, 2), dtype=np.float64)

    fu_dev = np.zeros(8, dtype=np.float64)
    fa_dev = np.zeros(4, dtype=np.float64)

    B = np.zeros((3, 8), dtype=np.float64)
    M0 = np.zeros((3, 4), dtype=np.float64)

    P_dev_3d = np.eye(6, dtype=np.float64)
    for r in range(3):
        for c in range(3):
            P_dev_3d[r, c] -= 1.0 / 3.0

    for q in range(4):
        xi = _GPS[q, 0]
        eta = _GPS[q, 1]
        wt = _WTS[q]

        dN_dxi = _shape_derivs_cpe4i(xi, eta)
        J = dN_dxi @ coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

        invJ00 =  J[1, 1] / detJ
        invJ01 = -J[0, 1] / detJ
        invJ10 = -J[1, 0] / detJ
        invJ11 =  J[0, 0] / detJ

        B.fill(0.0)
        for a in range(4):
            dN_dx = invJ00 * dN_dxi[0, a] + invJ01 * dN_dxi[1, a]
            dN_dy = invJ10 * dN_dxi[0, a] + invJ11 * dN_dxi[1, a]
            B[0, 2 * a + 0] = dN_dx
            B[1, 2 * a + 1] = dN_dy
            B[2, 2 * a + 0] = dN_dy
            B[2, 2 * a + 1] = dN_dx

        # 4-mode Simo-Rifai EAS matrix M = (detJ0 / detJ) * T0_inv_T @ M0
        M0.fill(0.0)
        M0[0, 0] = xi
        M0[1, 1] = eta
        M0[2, 2] = xi
        M0[2, 3] = eta

        scale = detJ0 / detJ
        M = scale * (T0_inv_T @ M0)  # (3, 4)

        # Piola-scaled linear pressure basis n_p = (detJ0 / detJ) * [xi, eta]
        np0 = scale * xi
        np1 = scale * eta

        dV = detJ * wt

        # Volumetric trace vectors b_v = B^T m and m_v = M^T m (m = [1, 1, 0]^T)
        for i in range(8):
            bv_i = B[0, i] + B[1, i]
            Kup[i, 0] += bv_i * np0 * dV
            Kup[i, 1] += bv_i * np1 * dV

        for j in range(4):
            mv_j = M[0, j] + M[1, j]
            Kap[j, 0] += mv_j * np0 * dV
            Kap[j, 1] += mv_j * np1 * dV

        Hpp[0, 0] += np0 * np0 * dV
        Hpp[0, 1] += np0 * np1 * dV
        Hpp[1, 0] += np1 * np0 * dV
        Hpp[1, 1] += np1 * np1 * dV

        # Constitutive deviatoric evaluation
        eps = B @ u_elem
        if mat_type == MAT_LINEAR_ELASTIC:
            C_dev = np.zeros((3, 3), dtype=np.float64)
            C_dev[0, 0] = (4.0 / 3.0) * mu
            C_dev[0, 1] = -(2.0 / 3.0) * mu
            C_dev[1, 0] = -(2.0 / 3.0) * mu
            C_dev[1, 1] = (4.0 / 3.0) * mu
            C_dev[2, 2] = mu
            s_dev = C_dev @ eps
        else:
            # Evaluate 3D material law on volume-stabilized strain_bar (avoiding K-explosion)
            tr_eps = eps[0] + eps[1]
            E_voigt = np.zeros(6, dtype=np.float64)
            E_voigt[0] = eps[0] - tr_eps / 3.0 + eps_vol_mean / 3.0
            E_voigt[1] = eps[1] - tr_eps / 3.0 + eps_vol_mean / 3.0
            E_voigt[2] = -tr_eps / 3.0 + eps_vol_mean / 3.0
            E_voigt[3] = eps[2]

            F_dummy = np.eye(3, dtype=np.float64)
            sdv_pt = sdvs[q] if (sdvs is not None and len(sdvs) > q) else np.zeros(7, dtype=np.float64)
            S_voigt, C_6x6, sdv_new, mat_err = material_dispatch_3d(
                mat_type, props, sdv_pt, E_voigt, F_dummy, 1.0, dt
            )
            if mat_err != 0:
                has_error = 1
            if sdvs is not None and len(sdvs) > q:
                for s_i in range(min(len(sdv_new), len(sdvs[q]))):
                    sdvs[q, s_i] = sdv_new[s_i]

            tr_s = (S_voigt[0] + S_voigt[1] + S_voigt[2]) / 3.0
            s_dev = np.zeros(3, dtype=np.float64)
            s_dev[0] = S_voigt[0] - tr_s
            s_dev[1] = S_voigt[1] - tr_s
            s_dev[2] = S_voigt[3]

            C_dev_6x6 = P_dev_3d @ C_6x6 @ P_dev_3d
            C_dev = np.zeros((3, 3), dtype=np.float64)
            C_dev[0, 0] = C_dev_6x6[0, 0]
            C_dev[0, 1] = C_dev_6x6[0, 1]
            C_dev[0, 2] = C_dev_6x6[0, 3]
            C_dev[1, 0] = C_dev_6x6[1, 0]
            C_dev[1, 1] = C_dev_6x6[1, 1]
            C_dev[1, 2] = C_dev_6x6[1, 3]
            C_dev[2, 0] = C_dev_6x6[3, 0]
            C_dev[2, 1] = C_dev_6x6[3, 1]
            C_dev[2, 2] = C_dev_6x6[3, 3]

        CB = C_dev @ B
        CM = C_dev @ M

        for i in range(8):
            fu_dev[i] += (B[0, i] * s_dev[0] + B[1, i] * s_dev[1] + B[2, i] * s_dev[2]) * dV
            for j in range(8):
                Kuu_dev[i, j] += (B[0, i] * CB[0, j] + B[1, i] * CB[1, j] + B[2, i] * CB[2, j]) * dV
            for j in range(4):
                Kua_dev[i, j] += (B[0, i] * CM[0, j] + B[1, i] * CM[1, j] + B[2, i] * CM[2, j]) * dV

        for i in range(4):
            fa_dev[i] += (M[0, i] * s_dev[0] + M[1, i] * s_dev[1] + M[2, i] * s_dev[2]) * dV
            for j in range(4):
                Kaa_dev[i, j] += (M[0, i] * CM[0, j] + M[1, i] * CM[1, j] + M[2, i] * CM[2, j]) * dV

    # Two-stage Schur complement condensation of (alpha, p_lin):
    # Stage 1: Condense 4 EAS modes alpha using strictly SPD Kaa_dev
    invKaa, ok_a = solve_small(Kaa_dev, np.eye(4))
    if not ok_a:
        return np.zeros(8, dtype=np.float64), np.zeros((8, 8), dtype=np.float64), 1
    Kuu_tilde = Kuu_dev - Kua_dev @ invKaa @ Kua_dev.T
    Kup_tilde = Kup - Kua_dev @ invKaa @ Kap

    # Stage 2: Condense 2 linear pressure modes p_lin using SPD Spp
    Spp = Kap.T @ invKaa @ Kap + (1.0 / K) * Hpp
    invSpp, ok_p = solve_small(Spp, np.eye(2))
    if not ok_p:
        return np.zeros(8, dtype=np.float64), np.zeros((8, 8), dtype=np.float64), 1

    Ke = Kuu_tilde + Kup_tilde @ invSpp @ Kup_tilde.T

    # Add constant pressure p0 contribution: (K / V0) * vol_vec (x) vol_vec
    k_vol_fac = K / max(tot_vol, 1e-14)
    for i in range(8):
        for j in range(8):
            Ke[i, j] += k_vol_fac * vol_vec[i] * vol_vec[j]

    # Consistent internal force condensation
    gp_rhs = Kup.T @ u_elem - Kap.T @ (invKaa @ fa_dev)
    p_lin = invSpp @ gp_rhs
    alpha_opt = -invKaa @ (fa_dev + Kap @ p_lin)

    fe = fu_dev + Kua_dev @ alpha_opt + Kup @ p_lin + p0 * vol_vec

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_cpe4ih_numba(
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
    """Parallel Numba mesh assembly for small-strain / generic-CR-wrapped CPE4IH elements."""
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

        fe, Ke, err = compute_cpe4ih_element_numba(coords_e, u_e, m_type, props_e, sdvs_e, dt, ctrl_e)
        f_elems[e] = fe
        k_elems[e] = Ke
        errors[e] = err

    has_error = int(np.sum(errors) > 0)
    return f_elems, k_elems, has_error
