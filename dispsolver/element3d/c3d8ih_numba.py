"""
c3d8ih_numba.py
===============
Numba JIT accelerated 3D 8-Node Incompatible Modes + Hybrid Linear Pressure Hexahedron (C3D8IH).

Theory (Abaqus C3D8IH — "8-node linear brick, incompatible modes, hybrid with linear pressure"):
- Combines Simo & Armero (1992) 9-mode Enhanced Assumed Strain (EAS) kinematics with
  Simo-Taylor-Pister / Herrmann 4-DOF linear hybrid hydrostatic pressure (p0, p_xi, p_eta, p_zeta).
- Piola-scaled linear pressure basis n_p_lin(xi, eta, zeta) = (detJ0 / detJ) * [xi, eta, zeta]^T ensures:
  1. Exact L2-orthogonality between constant mean pressure p0 and zero-mean fluctuating modes
     (alpha, p_lin) on arbitrary distorted hexahedral meshes.
  2. Exact satisfaction of Irons constant-strain patch test on distorted meshes (alpha = 0, p_lin = 0).
  3. Two-stage Schur complement condensation (condensing 9-mode alpha via K_aa_dev^-1 first, then
     3-mode p_lin via S_pp^-1) completely avoids O(K) - O(K) floating-point cancellation as nu -> 0.5,
     while preserving a strictly symmetric positive-definite (SPD) 24x24 element stiffness matrix.
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange

from dispsolver.element3d.c3d8_numba import _sd3d, _jacobian3d, _GP_GAUSS
from dispsolver.element3d.c3d8_eas_numba import _compute_invT0_T_numba, _enhanced_strain_matrix_m_numba
from dispsolver.element3d.c3d8_hybrid_numba import _hencky_to_green_gl_numba
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

from dispsolver.element2d.cpe4i_numba import solve_small

_EMPTY_CONTROLS = np.empty(0, dtype=np.float64)
_EMPTY_STRESS_INIT = np.empty((0, 6), dtype=np.float64)


@njit(fastmath=True)
def _decode_shear_bulk_3d(mat_type: int, props: np.ndarray):
    """Decode baseline shear modulus G and bulk modulus K from 3D material props."""
    if mat_type == MAT_CUSTOM_ELASTIC:
        lam = props[1]
        mu = props[21]
        if mu <= 0.0:
            E_ = props[0] if props[0] > 0.0 else 200000.0
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
def compute_c3d8ih_nonlinear_numba(
    coords: np.ndarray,                              # (8, 3)
    u_elem: np.ndarray,                              # (24,)
    mat_type: int,
    props: np.ndarray,                               # (64,)
    sdvs: np.ndarray,                                # (8, n_sdv), updated in place with the converged state
    dt: float,
    K_eff: float,                                    # bulk modulus incl. the distortion-control stiffness
    p0: float,                                       # constant pressure (incl. distortion-control pressure)
    vol_vec: np.ndarray,                             # (24,) int B_v dV
    V0_total: float,
    eps_vol_mean: float,
    detJ0: float,
    invT0_T: np.ndarray,                             # (6, 6)
    stress_init: np.ndarray,                         # (8, 6) or empty
):
    """C3D8IH for a path-dependent / nonlinear material (plasticity, hyperelasticity).

    Same repair as CPE4IH: the unknowns (alpha: 9 incompatible-mode amplitudes, p_lin: 3 linear-pressure
    amplitudes) are iterated to element equilibrium for the CURRENT displacements,

        R_alpha = int B~^T s_dev(B u + B~ alpha) dV + Kap p_lin = 0
        R_p     = Kup^T u + Kap^T alpha - Hpp p_lin / K_eff     = 0 ,

    and the material sees the strain B u + B~ alpha, not B u (the enhanced strain carries the bending). One
    Newton step from zero is exact for a linear material, so for those the closed form in
    compute_c3d8ih_element_numba is kept. The material is evaluated from the state at the START of the
    increment on every iteration; only the converged evaluation is written back.
    """
    # geometry of the eight Gauss points
    Bq = np.zeros((8, 6, 24), dtype=np.float64)
    Btq = np.zeros((8, 6, 9), dtype=np.float64)
    npq = np.zeros((8, 3), dtype=np.float64)
    dVq = np.zeros(8, dtype=np.float64)
    for gi in range(2):
        xi = _GP_GAUSS[gi]
        for gj in range(2):
            eta = _GP_GAUSS[gj]
            for gk in range(2):
                zeta = _GP_GAUSS[gk]
                g = 4 * gi + 2 * gj + gk
                dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
                _, detJ, invJ = _jacobian3d(xi, eta, zeta, coords)
                if detJ <= 1e-14:
                    return np.zeros(24, dtype=np.float64), np.zeros((24, 24), dtype=np.float64), 1
                dVq[g] = detJ
                for i in range(8):
                    dN_dX = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                    dN_dY = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                    dN_dZ = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]
                    Bq[g, 0, 3 * i + 0] = dN_dX
                    Bq[g, 1, 3 * i + 1] = dN_dY
                    Bq[g, 2, 3 * i + 2] = dN_dZ
                    Bq[g, 3, 3 * i + 0] = dN_dY
                    Bq[g, 3, 3 * i + 1] = dN_dX
                    Bq[g, 4, 3 * i + 1] = dN_dZ
                    Bq[g, 4, 3 * i + 2] = dN_dY
                    Bq[g, 5, 3 * i + 0] = dN_dZ
                    Bq[g, 5, 3 * i + 2] = dN_dX
                scale = detJ0 / detJ
                M0 = _enhanced_strain_matrix_m_numba(xi, eta, zeta)
                Btq[g] = scale * (invT0_T @ M0)
                npq[g, 0] = scale * xi
                npq[g, 1] = scale * eta
                npq[g, 2] = scale * zeta

    Kup = np.zeros((24, 3), dtype=np.float64)
    Kap = np.zeros((9, 3), dtype=np.float64)
    Hpp = np.zeros((3, 3), dtype=np.float64)
    f_init = np.zeros(24, dtype=np.float64)
    has_stress_init = (stress_init is not None and stress_init.shape[0] == 8)
    for g in range(8):
        dV = dVq[g]
        for i in range(24):
            bv_i = Bq[g, 0, i] + Bq[g, 1, i] + Bq[g, 2, i]
            for k in range(3):
                Kup[i, k] += bv_i * npq[g, k] * dV
        for j in range(9):
            mv_j = Btq[g, 0, j] + Btq[g, 1, j] + Btq[g, 2, j]
            for k in range(3):
                Kap[j, k] += mv_j * npq[g, k] * dV
        for a in range(3):
            for b in range(3):
                Hpp[a, b] += npq[g, a] * npq[g, b] * dV
        if has_stress_init:
            f_init += (Bq[g].T @ stress_init[g]) * dV

    P_dev = np.eye(6, dtype=np.float64)
    for r in range(3):
        for c in range(3):
            P_dev[r, c] -= 1.0 / 3.0

    n_sdv = sdvs.shape[1] if (sdvs is not None and sdvs.ndim == 2) else 0
    sdv0 = np.zeros((8, max(n_sdv, 1)), dtype=np.float64)
    for g in range(8):
        for s_i in range(n_sdv):
            sdv0[g, s_i] = sdvs[g, s_i]
    sdv_last = sdv0.copy()

    alpha = np.zeros(9, dtype=np.float64)
    p_lin = np.zeros(3, dtype=np.float64)
    Kuu = np.zeros((24, 24), dtype=np.float64)
    Kua = np.zeros((24, 9), dtype=np.float64)
    Kaa = np.zeros((9, 9), dtype=np.float64)
    fu = np.zeros(24, dtype=np.float64)
    fa = np.zeros(9, dtype=np.float64)
    has_error = 0
    converged = False

    for it in range(30):
        Kuu.fill(0.0)
        Kua.fill(0.0)
        Kaa.fill(0.0)
        fu.fill(0.0)
        fa.fill(0.0)
        for g in range(8):
            B = Bq[g]
            Bt = Btq[g]
            dV = dVq[g]
            eps = B @ u_elem + Bt @ alpha
            tr_eps = (eps[0] + eps[1] + eps[2]) / 3.0
            strain_bar = eps.copy()
            strain_bar[0] = eps[0] - tr_eps + eps_vol_mean / 3.0
            strain_bar[1] = eps[1] - tr_eps + eps_vol_mean / 3.0
            strain_bar[2] = eps[2] - tr_eps + eps_vol_mean / 3.0

            sdv_gp = sdv0[g, :n_sdv].copy() if n_sdv > 0 else np.zeros(0, dtype=np.float64)
            E_gl_gp, F_loc_gp, detF_loc_gp = _hencky_to_green_gl_numba(strain_bar)
            stress_bar, C_tangent, sdv_gp_new, mat_err = material_dispatch_3d(
                mat_type, props, sdv_gp, E_gl_gp, F_loc_gp, detF_loc_gp, dt
            )
            if mat_err != 0:
                has_error = 1
            for s_i in range(min(len(sdv_gp_new), n_sdv)):
                sdv_last[g, s_i] = sdv_gp_new[s_i]

            tr_s = (stress_bar[0] + stress_bar[1] + stress_bar[2]) / 3.0
            s_dev = stress_bar.copy()
            s_dev[0] -= tr_s
            s_dev[1] -= tr_s
            s_dev[2] -= tr_s
            C_dev = P_dev @ C_tangent @ P_dev
            CB = C_dev @ B
            CM = C_dev @ Bt
            fu += (B.T @ s_dev) * dV
            fa += (Bt.T @ s_dev) * dV
            Kuu += (B.T @ CB) * dV
            Kua += (B.T @ CM) * dV
            Kaa += (Bt.T @ CM) * dV

        R_alpha = fa + Kap @ p_lin
        R_p = Kup.T @ u_elem + Kap.T @ alpha - (Hpp @ p_lin) / K_eff
        r_norm = np.sqrt(np.sum(R_alpha * R_alpha) + np.sum(R_p * R_p))
        f_scale = max(np.sqrt(np.sum(fu * fu)), 1.0e-3)
        if r_norm <= 1e-12 * f_scale:
            converged = True
            break
        Jm = np.zeros((12, 12), dtype=np.float64)
        Jm[:9, :9] = Kaa
        Jm[:9, 9:] = Kap
        Jm[9:, :9] = Kap.T
        Jm[9:, 9:] = -Hpp / K_eff
        rhs = np.zeros((12, 1), dtype=np.float64)
        rhs[:9, 0] = R_alpha
        rhs[9:, 0] = R_p
        dx, solved = solve_small(Jm, rhs)
        if not solved:
            has_error = 1               # singular enhanced block (collapse): let the solver cut back
            break
        alpha = alpha - dx[:9, 0]
        p_lin = p_lin - dx[9:, 0]
    if (not converged) and has_error == 0:
        R_alpha = fa + Kap @ p_lin
        R_p = Kup.T @ u_elem + Kap.T @ alpha - (Hpp @ p_lin) / K_eff
        r_norm = np.sqrt(np.sum(R_alpha * R_alpha) + np.sum(R_p * R_p))
        if r_norm > 1e-6 * max(np.sqrt(np.sum(fu * fu)), 1.0e-3):
            has_error = 1               # (alpha, p_lin) did not reach equilibrium

    for g in range(8):
        for s_i in range(n_sdv):
            sdvs[g, s_i] = sdv_last[g, s_i]

    invKaa, ok1 = solve_small(Kaa, np.eye(9))
    if not ok1:
        return fu + p0 * vol_vec + f_init, Kuu, 1
    Kuu_tilde = Kuu - Kua @ invKaa @ Kua.T
    Kup_tilde = Kup - Kua @ invKaa @ Kap
    Spp = Kap.T @ invKaa @ Kap + (1.0 / K_eff) * Hpp
    invSpp, ok2 = solve_small(Spp, np.eye(3))
    if not ok2:
        return fu + p0 * vol_vec + f_init, Kuu, 1
    Ke = Kuu_tilde + Kup_tilde @ invSpp @ Kup_tilde.T
    k_vol_fac = K_eff / max(V0_total, 1e-14)
    for i in range(24):
        for j in range(24):
            Ke[i, j] += k_vol_fac * vol_vec[i] * vol_vec[j]

    fe = fu + Kup @ p_lin + p0 * vol_vec + f_init
    return fe, Ke, has_error


@njit(fastmath=True)
def compute_c3d8ih_element_numba(
    coords: np.ndarray,                              # (8, 3)
    u_elem: np.ndarray,                              # (24,)
    mat_type: int,
    props: np.ndarray,                               # (64,)
    sdvs: np.ndarray,                                # (8, n_sdv)
    dt: float = 1.0,
    controls: np.ndarray = _EMPTY_CONTROLS,
    stress_init: np.ndarray = _EMPTY_STRESS_INIT,
):
    """
    Computes statically condensed 24x24 SPD tangent Ke and 24-element internal force fe
    for 1 C3D8IH (9-mode EAS + 4-DOF linear hybrid pressure) element in its local/reference frame.
    """
    mu, K = _decode_shear_bulk_3d(mat_type, props)

    # Central Jacobian J0 and Simo-Armero Voigt transformation inv(T0).T (6, 6)
    J0, detJ0, _ = _jacobian3d(0.0, 0.0, 0.0, coords)
    if detJ0 <= 1e-14:
        return np.zeros(24, dtype=np.float64), np.zeros((24, 24), dtype=np.float64), 1

    invT0_T = _compute_invT0_T_numba(J0)

    # Pass 1: Total reference volume V0 and mean volumetric strain-displacement vector vol_vec
    vol_vec = np.zeros(24, dtype=np.float64)
    V0_total = 0.0
    for gi in range(2):
        xi = _GP_GAUSS[gi]
        for gj in range(2):
            eta = _GP_GAUSS[gj]
            for gk in range(2):
                zeta = _GP_GAUSS[gk]
                dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
                _, detJ, invJ = _jacobian3d(xi, eta, zeta, coords)
                if detJ <= 1e-14:
                    return np.zeros(24, dtype=np.float64), np.zeros((24, 24), dtype=np.float64), 1

                dV = detJ
                V0_total += dV
                for i in range(8):
                    dN_dX = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                    dN_dY = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                    dN_dZ = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]
                    vol_vec[3 * i + 0] += dN_dX * dV
                    vol_vec[3 * i + 1] += dN_dY * dV
                    vol_vec[3 * i + 2] += dN_dZ * dV

    eps_vol_mean = np.sum(vol_vec * u_elem) / max(V0_total, 1e-14)

    # Check distortion / anti-inversion controls on mean volume ratio J_vol
    has_controls = (controls is not None and controls.shape[0] >= 5)
    distortion_control_on = has_controls and (controls[0] > 0.5)
    j_crit = controls[1] if (has_controls and controls[1] > 0.0) else 0.1
    anti_inversion_on = has_controls and (controls[3] > 0.5)
    min_det_f = controls[4] if (has_controls and controls[4] > 0.0) else 0.02

    J_vol = 1.0 + eps_vol_mean
    if anti_inversion_on and J_vol < min_det_f:
        return np.zeros(24, dtype=np.float64), np.zeros((24, 24), dtype=np.float64), 1

    p_dist = 0.0
    c_dist = 0.0
    if distortion_control_on and J_vol < j_crit:
        if J_vol <= 0.05:
            return np.zeros(24, dtype=np.float64), np.zeros((24, 24), dtype=np.float64), 1
        j_eff = max(J_vol, 0.05)
        k_dist = mu
        p_dist = k_dist * j_crit * (j_crit - j_eff) / (j_eff * j_eff * j_eff)
        c_dist = k_dist * j_crit * (3.0 * j_crit - 2.0 * j_eff) / (j_eff * j_eff * j_eff * j_eff)

    p0 = K * eps_vol_mean - p_dist
    K_eff = K + c_dist

    if mat_type != MAT_LINEAR_ELASTIC and mat_type != MAT_CUSTOM_ELASTIC:
        fe_n, Ke_n, err_n = compute_c3d8ih_nonlinear_numba(
            coords, u_elem, mat_type, props, sdvs, dt, K_eff, p0, vol_vec, V0_total, eps_vol_mean,
            detJ0, invT0_T, stress_init
        )
        damping_n = controls[2] if has_controls else 0.0
        if damping_n > 0.0:
            for i in range(24):
                k_ii = abs(Ke_n[i, i])
                fe_n[i] += damping_n * u_elem[i] * k_ii
                Ke_n[i, i] += damping_n * k_ii
        return fe_n, Ke_n, err_n

    # Pass 2: Accumulate deviatoric stiffnesses and linear pressure couplings
    Kuu_dev = np.zeros((24, 24), dtype=np.float64)
    Kua_dev = np.zeros((24, 9), dtype=np.float64)
    Kaa_dev = np.zeros((9, 9), dtype=np.float64)

    Kup = np.zeros((24, 3), dtype=np.float64)
    Kap = np.zeros((9, 3), dtype=np.float64)
    Hpp = np.zeros((3, 3), dtype=np.float64)

    fu_dev = np.zeros(24, dtype=np.float64)
    fa_dev = np.zeros(9, dtype=np.float64)
    f_init = np.zeros(24, dtype=np.float64)

    P_dev = np.eye(6, dtype=np.float64)
    for r in range(3):
        for c in range(3):
            P_dev[r, c] -= 1.0 / 3.0

    has_error = 0
    has_stress_init = (stress_init is not None and stress_init.shape[0] == 8)

    for gi in range(2):
        xi = _GP_GAUSS[gi]
        for gj in range(2):
            eta = _GP_GAUSS[gj]
            for gk in range(2):
                zeta = _GP_GAUSS[gk]
                gp_idx = 4 * gi + 2 * gj + gk

                dN_dxi, dN_deta, dN_dzeta = _sd3d(xi, eta, zeta)
                _, detJ, invJ = _jacobian3d(xi, eta, zeta, coords)
                dV = detJ

                B = np.zeros((6, 24), dtype=np.float64)
                for i in range(8):
                    dN_dX = invJ[0, 0] * dN_dxi[i] + invJ[0, 1] * dN_deta[i] + invJ[0, 2] * dN_dzeta[i]
                    dN_dY = invJ[1, 0] * dN_dxi[i] + invJ[1, 1] * dN_deta[i] + invJ[1, 2] * dN_dzeta[i]
                    dN_dZ = invJ[2, 0] * dN_dxi[i] + invJ[2, 1] * dN_deta[i] + invJ[2, 2] * dN_dzeta[i]

                    B[0, 3 * i + 0] = dN_dX
                    B[1, 3 * i + 1] = dN_dY
                    B[2, 3 * i + 2] = dN_dZ
                    B[3, 3 * i + 0] = dN_dY
                    B[3, 3 * i + 1] = dN_dX
                    B[4, 3 * i + 1] = dN_dZ
                    B[4, 3 * i + 2] = dN_dY
                    B[5, 3 * i + 0] = dN_dZ
                    B[5, 3 * i + 2] = dN_dX

                # 9-mode Simo-Armero EAS matrix B_tilde = (detJ0 / detJ) * invT0_T @ M
                scale = detJ0 / detJ
                M0 = _enhanced_strain_matrix_m_numba(xi, eta, zeta)
                B_tilde = scale * (invT0_T @ M0)  # (6, 9)

                # Piola-scaled linear pressure basis n_p = (detJ0 / detJ) * [xi, eta, zeta]
                np_vec = np.array([scale * xi, scale * eta, scale * zeta], dtype=np.float64)

                # Volumetric trace vectors b_v = B^T m and m_v = B_tilde^T m
                for i in range(24):
                    bv_i = B[0, i] + B[1, i] + B[2, i]
                    Kup[i, 0] += bv_i * np_vec[0] * dV
                    Kup[i, 1] += bv_i * np_vec[1] * dV
                    Kup[i, 2] += bv_i * np_vec[2] * dV

                for j in range(9):
                    mv_j = B_tilde[0, j] + B_tilde[1, j] + B_tilde[2, j]
                    Kap[j, 0] += mv_j * np_vec[0] * dV
                    Kap[j, 1] += mv_j * np_vec[1] * dV
                    Kap[j, 2] += mv_j * np_vec[2] * dV

                for p_i in range(3):
                    for p_j in range(3):
                        Hpp[p_i, p_j] += np_vec[p_i] * np_vec[p_j] * dV

                # Deviatoric constitutive evaluation
                eps = B @ u_elem
                if mat_type == MAT_LINEAR_ELASTIC or mat_type == MAT_CUSTOM_ELASTIC:
                    C_dev = np.zeros((6, 6), dtype=np.float64)
                    for r in range(3):
                        for c in range(3):
                            C_dev[r, c] = -(2.0 / 3.0) * mu
                        C_dev[r, r] += 2.0 * mu
                    C_dev[3, 3] = mu
                    C_dev[4, 4] = mu
                    C_dev[5, 5] = mu
                    s_dev = C_dev @ eps
                else:
                    tr_eps = (eps[0] + eps[1] + eps[2]) / 3.0
                    strain_bar = eps.copy()
                    strain_bar[0] = eps[0] - tr_eps + eps_vol_mean / 3.0
                    strain_bar[1] = eps[1] - tr_eps + eps_vol_mean / 3.0
                    strain_bar[2] = eps[2] - tr_eps + eps_vol_mean / 3.0

                    sdv_gp = sdvs[gp_idx] if (sdvs is not None and sdvs.shape[0] > gp_idx) else np.zeros(0, dtype=np.float64)
                    E_gl_gp, F_loc_gp, detF_loc_gp = _hencky_to_green_gl_numba(strain_bar)
                    stress_bar, C_tangent, sdv_gp_new, mat_err = material_dispatch_3d(
                        mat_type, props, sdv_gp, E_gl_gp, F_loc_gp, detF_loc_gp, dt
                    )
                    if mat_err != 0:
                        has_error = 1
                    if sdvs is not None and sdvs.shape[0] > gp_idx:
                        for s_i in range(min(len(sdv_gp_new), len(sdvs[gp_idx]))):
                            sdvs[gp_idx, s_i] = sdv_gp_new[s_i]

                    tr_s = (stress_bar[0] + stress_bar[1] + stress_bar[2]) / 3.0
                    s_dev = stress_bar.copy()
                    s_dev[0] -= tr_s
                    s_dev[1] -= tr_s
                    s_dev[2] -= tr_s
                    C_dev = P_dev @ C_tangent @ P_dev

                if has_stress_init:
                    f_init += (B.T @ stress_init[gp_idx]) * dV

                CB = C_dev @ B
                CM = C_dev @ B_tilde

                fu_dev += (B.T @ s_dev) * dV
                fa_dev += (B_tilde.T @ s_dev) * dV

                Kuu_dev += (B.T @ CB) * dV
                Kua_dev += (B.T @ CM) * dV
                Kaa_dev += (B_tilde.T @ CM) * dV

    # Two-stage Schur complement condensation of (alpha(9), p_lin(3)):
    # Stage 1: Condense 9 EAS modes alpha using strictly SPD Kaa_dev
    invKaa, ok_a = solve_small(Kaa_dev, np.eye(9))
    if not ok_a:
        return np.zeros(24, dtype=np.float64), np.zeros((24, 24), dtype=np.float64), 1
    Kuu_tilde = Kuu_dev - Kua_dev @ invKaa @ Kua_dev.T
    Kup_tilde = Kup - Kua_dev @ invKaa @ Kap

    # Stage 2: Condense 3 linear pressure modes p_lin using SPD Spp
    Spp = Kap.T @ invKaa @ Kap + (1.0 / K_eff) * Hpp
    invSpp, ok_p = solve_small(Spp, np.eye(3))
    if not ok_p:
        return np.zeros(24, dtype=np.float64), np.zeros((24, 24), dtype=np.float64), 1

    Ke = Kuu_tilde + Kup_tilde @ invSpp @ Kup_tilde.T

    # Add constant pressure p0 contribution: (K_eff / V0) * vol_vec (x) vol_vec
    k_vol_fac = K_eff / max(V0_total, 1e-14)
    for i in range(24):
        for j in range(24):
            Ke[i, j] += k_vol_fac * vol_vec[i] * vol_vec[j]

    # Consistent internal force condensation
    gp_rhs = Kup.T @ u_elem - Kap.T @ (invKaa @ fa_dev)
    p_lin = invSpp @ gp_rhs
    alpha_opt = -invKaa @ (fa_dev + Kap @ p_lin)

    fe = fu_dev + Kua_dev @ alpha_opt + Kup @ p_lin + p0 * vol_vec + f_init

    # Optional viscous damping control
    damping_coeff = controls[2] if has_controls else 0.0
    if damping_coeff > 0.0:
        for i in range(24):
            k_ii = abs(Ke[i, i])
            fe[i] += damping_coeff * u_elem[i] * k_ii
            Ke[i, i] += damping_coeff * k_ii

    return fe, Ke, has_error


@njit(parallel=True, fastmath=True)
def assemble_mesh_c3d8ih_numba(
    node_coords: np.ndarray,
    elem_conn: np.ndarray,
    u_global: np.ndarray,
    elem_mat_types: np.ndarray,
    elem_props: np.ndarray,
    elem_sdvs: np.ndarray,
    dt: float = 1.0,
    elem_controls: np.ndarray = None,
    elem_stress_init: np.ndarray = None,
):
    """Parallel Numba mesh assembly for small-strain / generic-CR-wrapped C3D8IH elements."""
    n_elems = elem_conn.shape[0]
    f_elems = np.zeros((n_elems, 24), dtype=np.float64)
    K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)
    err_flags = np.zeros(n_elems, dtype=np.int32)
    use_controls = (elem_controls is not None and elem_controls.shape[0] == n_elems)
    has_stress_init = (elem_stress_init is not None and elem_stress_init.shape[0] == n_elems)

    for e in prange(n_elems):
        conn = elem_conn[e]
        elem_coords = np.zeros((8, 3), dtype=np.float64)
        u_elem = np.zeros(24, dtype=np.float64)
        for i in range(8):
            nid = conn[i]
            elem_coords[i, :] = node_coords[nid, :]
            u_elem[3 * i : 3 * i + 3] = u_global[3 * nid : 3 * nid + 3]

        ctrl = elem_controls[e] if use_controls else _EMPTY_CONTROLS
        s_init = elem_stress_init[e] if has_stress_init else _EMPTY_STRESS_INIT

        fe, Ke, err = compute_c3d8ih_element_numba(
            elem_coords, u_elem, elem_mat_types[e], elem_props[e], elem_sdvs[e], dt, ctrl, s_init
        )
        f_elems[e, :] = fe
        K_elems[e, :, :] = Ke
        err_flags[e] = err

    return f_elems, K_elems, int(np.sum(err_flags) > 0)
