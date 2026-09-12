"""
numba_materials.py
==================
Data-Oriented Material Dispatcher for 3D Non-Linear Numba Assembly.
Provides a unified interface decoupling kinematics from constitutive laws.
"""

import numpy as np
from numba import njit

# Material Type Constants
MAT_LINEAR_ELASTIC = 0
MAT_HYPERELASTIC_NEOHOOKEAN = 1
MAT_NEO_HOOKEAN = MAT_HYPERELASTIC_NEOHOOKEAN
MAT_J2_PLASTICITY = 2
MAT_VISCOELASTIC_PRONY = 3
MAT_CUSTOM_ELASTIC = 99

@njit(fastmath=True)
def get_default_sdv_count(mat_type: int) -> int:
    """Return the number of state-dependent variables (SDVs) needed per integration point."""
    if mat_type == MAT_LINEAR_ELASTIC:
        return 0
    elif mat_type == MAT_HYPERELASTIC_NEOHOOKEAN:
        return 0
    elif mat_type == MAT_J2_PLASTICITY:
        return 7  # eq_plastic_strain (1), plastic_strain_tensor (6)
    elif mat_type == MAT_VISCOELASTIC_PRONY:
        return 12  # viscous history tensor h_1 (6) + prev elastic deviatoric stress (6)
    elif mat_type == MAT_CUSTOM_ELASTIC:
        return 0
    return 0


@njit(fastmath=True)
def _neohookean_pk2_voigt(E_voigt: np.ndarray, mu: float, K: float) -> np.ndarray:
    """PK2 stress for genuine compressible Neo-Hookean:
        W(C) = (mu/2)*(J^(-2/3)*tr(C) - 3) + (K/2)*(J-1)^2
    (mu = 2*C10, K = 2/D1, Abaqus's own C10/D1 convention -- see
    dispsolver/model/material.py's to_numba_props()).

    S = 2*dW/dC = mu*J^(-2/3)*(I - (tr(C)/3)*Cinv) + K*J*(J-1)*Cinv

    Reference: Simo & Hughes (1998), Computational Inelasticity, Sec 9.3;
    equivalently Bonet & Wood (2008), Nonlinear Continuum Mechanics for
    Finite Element Analysis, 2nd ed., eq. 6.28 (their C10-only deviatoric
    term plus a (J-1)^2 volumetric term, the same combination this
    project's own `to_numba_props()` C10/D1 pair already assumes).

    Built directly from the Green-Lagrange strain (E_voigt, engineering-
    shear Voigt convention [E11,E22,E33,2E12,2E23,2E31]) via C = 2E + I,
    NOT from a separately-passed F -- this makes the Neo-Hookean branch's
    own finite-difference tangent (below) self-consistent by
    construction: perturbing E_voigt and re-evaluating this same function
    is guaranteed to differentiate the exact stress law that was coded,
    with no risk of a "which configuration" F/E mismatch (the class of
    bug this project hit repeatedly this session -- F4/F6/C3D8_FBAR's
    tangent bug/C3D8H's double-counted tangent).
    """
    C = np.empty((3, 3), dtype=np.float64)
    C[0, 0] = 1.0 + 2.0 * E_voigt[0]
    C[1, 1] = 1.0 + 2.0 * E_voigt[1]
    C[2, 2] = 1.0 + 2.0 * E_voigt[2]
    C[0, 1] = C[1, 0] = E_voigt[3]
    C[1, 2] = C[2, 1] = E_voigt[4]
    C[2, 0] = C[0, 2] = E_voigt[5]

    c00 = C[0, 0]; c01 = C[0, 1]; c02 = C[0, 2]
    c10 = C[1, 0]; c11 = C[1, 1]; c12 = C[1, 2]
    c20 = C[2, 0]; c21 = C[2, 1]; c22 = C[2, 2]

    cof00 = c11 * c22 - c12 * c21
    cof01 = -(c10 * c22 - c12 * c20)
    cof02 = c10 * c21 - c11 * c20
    detC = c00 * cof00 + c01 * cof01 + c02 * cof02
    if detC < 1e-12:
        detC = 1e-12
    J = np.sqrt(detC)

    inv_detC = 1.0 / detC
    Cinv = np.empty((3, 3), dtype=np.float64)
    Cinv[0, 0] = cof00 * inv_detC
    Cinv[0, 1] = -(c01 * c22 - c02 * c21) * inv_detC
    Cinv[0, 2] = (c01 * c12 - c02 * c11) * inv_detC
    Cinv[1, 0] = cof01 * inv_detC
    Cinv[1, 1] = (c00 * c22 - c02 * c20) * inv_detC
    Cinv[1, 2] = -(c00 * c12 - c02 * c10) * inv_detC
    Cinv[2, 0] = cof02 * inv_detC
    Cinv[2, 1] = -(c00 * c21 - c01 * c20) * inv_detC
    Cinv[2, 2] = (c00 * c11 - c01 * c10) * inv_detC

    trC = c00 + c11 + c22
    Jm23 = J ** (-2.0 / 3.0)
    vol_scale = K * J * (J - 1.0)

    S = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            ident = 1.0 if i == j else 0.0
            S[i, j] = mu * Jm23 * (ident - (trC / 3.0) * Cinv[i, j]) + vol_scale * Cinv[i, j]

    S_voigt = np.empty(6, dtype=np.float64)
    S_voigt[0] = S[0, 0]
    S_voigt[1] = S[1, 1]
    S_voigt[2] = S[2, 2]
    S_voigt[3] = S[0, 1]
    S_voigt[4] = S[1, 2]
    S_voigt[5] = S[2, 0]
    return S_voigt


@njit(fastmath=True)
def material_dispatch_3d(
    mat_type: int,
    props: np.ndarray,
    sdv_prev: np.ndarray,
    E_voigt: np.ndarray,
    F: np.ndarray,
    detF: float,
    dt: float
):
    """
    Unified Material Routine (UMAT style).
    Takes kinematic inputs (Strain/Deformation Gradient) and returns
    2nd Piola-Kirchhoff Stress (S_voigt) and Tangent Modulus (C_mat).
    
    Args:
        mat_type (int): Material model identifier.
        props (np.ndarray): Flattened array of material properties [E, nu, yield, ...].
        sdv_prev (np.ndarray): State dependent variables from previous converged step.
        E_voigt (np.ndarray): Green-Lagrange strain tensor (6,).
        F (np.ndarray): Deformation gradient (3x3).
        detF (float): Determinant of F (J).
        dt (float): Time increment.
        
    Returns:
        S_voigt (np.ndarray): 2nd Piola-Kirchhoff stress in Voigt notation (6,).
        C_mat (np.ndarray): Material tangent stiffness matrix (6x6).
        sdv_new (np.ndarray): Updated state dependent variables.
        error_flag (int): 0 for success, 1 for failure (e.g. invalid deformation).
    """
    S_voigt = np.zeros(6, dtype=np.float64)
    C_mat = np.zeros((6, 6), dtype=np.float64)
    sdv_new = sdv_prev.copy()
    error_flag = 0
    
    if detF <= 0.0:
        error_flag = 1  # Element Inverted (Negative Volume)
        return S_voigt, C_mat, sdv_new, error_flag

    if mat_type == MAT_LINEAR_ELASTIC:
        # props: [E, nu]
        E = props[0]
        nu = props[1]
        lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        
        C_mat[0, 0] = lam + 2*mu; C_mat[0, 1] = lam;        C_mat[0, 2] = lam
        C_mat[1, 0] = lam;        C_mat[1, 1] = lam + 2*mu; C_mat[1, 2] = lam
        C_mat[2, 0] = lam;        C_mat[2, 1] = lam;        C_mat[2, 2] = lam + 2*mu
        C_mat[3, 3] = mu
        C_mat[4, 4] = mu
        C_mat[5, 5] = mu
        
        S_voigt = C_mat @ E_voigt

    elif mat_type == MAT_HYPERELASTIC_NEOHOOKEAN:
        # Robust decoding for Neo-Hookean parameters:
        # Case 1: [C10, D1] (e.g., C10=0.1, D1=0.01) -> mu = 2*C10, K = 2/D1
        # Case 2: [E, nu] (e.g., E=200000, nu=0.4999) -> mu = E/(2*(1+nu)), K = E/(3*(1-2*nu))
        # Case 3: [mu, K] -> direct
        p0 = props[0]
        p1 = props[1]
        if p1 <= 0.05 and p0 > 0.0:
            mu = 2.0 * p0
            K = 2.0 / max(p1, 1e-12)
        elif p1 < 0.5 and p0 > 1.0:
            mu = p0 / (2.0 * (1.0 + p1))
            K = p0 / (3.0 * max(1.0 - 2.0 * p1, 1e-8))
        else:
            mu = p0
            K = max(p1, 1e-8)

        # Genuine finite-strain compressible Neo-Hookean (see
        # _neohookean_pk2_voigt's docstring for the W(C)/S(C) formula and
        # citation) -- this REPLACES the previous placeholder, which
        # computed `S = C_mat @ E_voigt` with a CONSTANT small-strain C_mat,
        # bit-identical to MAT_CUSTOM_ELASTIC/MAT_LINEAR_ELASTIC at the same
        # (E,nu): no J-dependent volumetric term, no deviatoric-invariant
        # stored-energy function -- verified fake this session
        # (dev_log/3d_capability_buildout... T3 / .omc/plans/
        # 3d_capability_buildout_20260913.md).
        S_voigt = _neohookean_pk2_voigt(E_voigt, mu, K)

        # Consistent tangent via a self-contained central finite
        # difference of _neohookean_pk2_voigt itself, perturbing E_voigt
        # only (C = 2E+I is reconstructed from E_voigt inside the helper,
        # not from the separately-passed F -- see that function's
        # docstring for why this keeps the tangent exact-by-construction
        # against the coded stress law, the same principle this project's
        # C3D8_FBAR fix already established, dev_log/
        # solve_step_false_convergence_20260913.md). Per-column adaptive
        # step (Dennis & Schnabel 1983 Sec 5.4, AGENTS.md's B4 convention)
        # -- NOT a fixed absolute step, which was a latent mesh-refinement
        # bug the last time this project used one.
        sqrt_eps = 1.4901161193847656e-08
        for j in range(6):
            h = sqrt_eps * max(abs(E_voigt[j]), 1.0)
            E_p = E_voigt.copy()
            E_p[j] += h
            E_m = E_voigt.copy()
            E_m[j] -= h
            S_p = _neohookean_pk2_voigt(E_p, mu, K)
            S_m = _neohookean_pk2_voigt(E_m, mu, K)
            for i in range(6):
                C_mat[i, j] = (S_p[i] - S_m[i]) / (2.0 * h)

    elif mat_type == MAT_J2_PLASTICITY:
        # props: [E, nu, sigma_y0, H]
        E = props[0]
        nu = props[1]
        sigma_y0 = props[2]
        H = props[3]

        mu = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        lam = K - (2.0 / 3.0) * mu

        # Elastic tangent matrix
        C_mat[0, 0] = lam + 2*mu; C_mat[0, 1] = lam;        C_mat[0, 2] = lam
        C_mat[1, 0] = lam;        C_mat[1, 1] = lam + 2*mu; C_mat[1, 2] = lam
        C_mat[2, 0] = lam;        C_mat[2, 1] = lam;        C_mat[2, 2] = lam + 2*mu
        C_mat[3, 3] = mu
        C_mat[4, 4] = mu
        C_mat[5, 5] = mu

        ep_old = np.zeros(6, dtype=np.float64)
        eq_p_old = 0.0
        if len(sdv_prev) >= 7:
            for i in range(6):
                ep_old[i] = sdv_prev[i]
            eq_p_old = sdv_prev[6]

        e_trial = E_voigt - ep_old
        trace_e = e_trial[0] + e_trial[1] + e_trial[2]
        e_dev = np.zeros(6, dtype=np.float64)
        e_dev[0] = e_trial[0] - trace_e / 3.0
        e_dev[1] = e_trial[1] - trace_e / 3.0
        e_dev[2] = e_trial[2] - trace_e / 3.0
        e_dev[3] = e_trial[3]
        e_dev[4] = e_trial[4]
        e_dev[5] = e_trial[5]

        s_trial = np.zeros(6, dtype=np.float64)
        s_trial[0] = 2.0 * mu * e_dev[0]
        s_trial[1] = 2.0 * mu * e_dev[1]
        s_trial[2] = 2.0 * mu * e_dev[2]
        s_trial[3] = mu * e_dev[3]
        s_trial[4] = mu * e_dev[4]
        s_trial[5] = mu * e_dev[5]

        norm_s = np.sqrt(
            s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2 +
            2.0 * (s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2)
        )
        q_trial = np.sqrt(1.5) * norm_s
        sigma_y_cur = sigma_y0 + H * eq_p_old
        f_trial = q_trial - sigma_y_cur

        hydro_p = K * trace_e
        if f_trial <= 0.0:
            S_voigt = s_trial.copy()
            S_voigt[0] += hydro_p
            S_voigt[1] += hydro_p
            S_voigt[2] += hydro_p
        else:
            d_gamma = f_trial / (3.0 * mu + H)
            eq_p_new = eq_p_old + d_gamma
            scale_s = 1.0 - (3.0 * mu * d_gamma) / max(q_trial, 1e-12)
            s_new = s_trial * scale_s

            S_voigt = s_new.copy()
            S_voigt[0] += hydro_p
            S_voigt[1] += hydro_p
            S_voigt[2] += hydro_p

            if len(sdv_new) >= 7:
                n_flow = s_trial / max(norm_s, 1e-12)
                flow_fac = d_gamma * np.sqrt(1.5)
                for i in range(6):
                    sdv_new[i] = ep_old[i] + flow_fac * n_flow[i]
                sdv_new[6] = eq_p_new

            # Algorithmic consistent elastoplastic tangent C_ep (Simo & Hughes 1998)
            beta0 = (3.0 * mu * d_gamma) / max(q_trial, 1e-12)
            beta1 = (3.0 * mu) / (3.0 * mu + H)
            n_vec = s_trial / max(norm_s, 1e-12)

            C_vol = np.zeros((6, 6), dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    C_vol[r, c] = K

            C_dev = np.zeros((6, 6), dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    C_dev[r, c] = - (2.0 / 3.0) * mu
                C_dev[r, r] += 2.0 * mu
            C_dev[3, 3] = mu
            C_dev[4, 4] = mu
            C_dev[5, 5] = mu

            C_mat = C_vol + (1.0 - beta0) * C_dev - 2.0 * mu * (beta1 - beta0) * np.outer(n_vec, n_vec)

    elif mat_type == MAT_VISCOELASTIC_PRONY:
        # props: [E/mu, nu/K, g1, tau1]
        p0 = props[0]
        p1 = props[1]
        if p1 < 0.5 and p0 > 0.0:
            mu = p0 / (2.0 * (1.0 + p1))
            K = p0 / (3.0 * max(1.0 - 2.0 * p1, 1e-8))
        else:
            mu = p0
            K = p1

        g1 = props[2]
        tau1 = max(props[3], 1e-12)

        # Deviatoric and volumetric strain
        eps_vol = E_voigt[0] + E_voigt[1] + E_voigt[2]
        e_dev = np.zeros(6, dtype=np.float64)
        e_dev[0] = E_voigt[0] - eps_vol / 3.0
        e_dev[1] = E_voigt[1] - eps_vol / 3.0
        e_dev[2] = E_voigt[2] - eps_vol / 3.0
        e_dev[3] = E_voigt[3]
        e_dev[4] = E_voigt[4]
        e_dev[5] = E_voigt[5]

        # Instantaneous elastic deviatoric stress (shear: 2*mu*(gamma/2) = mu*gamma)
        s_dev_el = np.zeros(6, dtype=np.float64)
        s_dev_el[0] = 2.0 * mu * e_dev[0]
        s_dev_el[1] = 2.0 * mu * e_dev[1]
        s_dev_el[2] = 2.0 * mu * e_dev[2]
        s_dev_el[3] = mu * e_dev[3]
        s_dev_el[4] = mu * e_dev[4]
        s_dev_el[5] = mu * e_dev[5]

        # Time integration parameters (linearized midpoint rule, Simo & Hughes 1998)
        dt_safe = max(dt, 1e-12)
        xi = dt_safe / tau1
        beta1 = np.exp(-xi)
        gamma1 = (1.0 - beta1) / xi if xi > 1e-6 else (1.0 - 0.5 * xi)

        h_old = np.zeros(6, dtype=np.float64)
        s_dev_prev = np.zeros(6, dtype=np.float64)
        if sdv_prev.shape[0] >= 12:
            for i in range(6):
                h_old[i] = sdv_prev[i]
                s_dev_prev[i] = sdv_prev[6 + i]

        # Update overstress h1(t + dt) = beta1 * h1(t) + g1 * gamma1 * (s_dev_el - s_dev_prev)
        h_new = np.zeros(6, dtype=np.float64)
        for i in range(6):
            delta_s = s_dev_el[i] - s_dev_prev[i]
            h_new[i] = beta1 * h_old[i] + g1 * gamma1 * delta_s
            if sdv_new.shape[0] >= 12:
                sdv_new[i] = h_new[i]
                sdv_new[6 + i] = s_dev_el[i]

        # Total stress: S = S_vol + S_dev, where S_dev = (1 - g1) * s_dev_el + h_new
        g_inf = 1.0 - g1
        S_voigt[0] = K * eps_vol + g_inf * s_dev_el[0] + h_new[0]
        S_voigt[1] = K * eps_vol + g_inf * s_dev_el[1] + h_new[1]
        S_voigt[2] = K * eps_vol + g_inf * s_dev_el[2] + h_new[2]
        S_voigt[3] = g_inf * s_dev_el[3] + h_new[3]
        S_voigt[4] = g_inf * s_dev_el[4] + h_new[4]
        S_voigt[5] = g_inf * s_dev_el[5] + h_new[5]

        # Algorithmic consistent tangent C_visco = C_vol + (g_inf + g1 * gamma1) * C_dev
        factor_dev = g_inf + g1 * gamma1
        mu_eff = mu * factor_dev

        for r in range(3):
            for c in range(3):
                C_mat[r, c] = K - (2.0 / 3.0) * mu_eff
            C_mat[r, r] += 2.0 * mu_eff
        C_mat[3, 3] = mu_eff
        C_mat[4, 4] = mu_eff
        C_mat[5, 5] = mu_eff

    elif mat_type == MAT_CUSTOM_ELASTIC:
        # props: flattened 6x6 matrix (36 values)
        for i in range(6):
            for j in range(6):
                C_mat[i, j] = props[6 * i + j]
        S_voigt = C_mat @ E_voigt

    else:
        # Unsupported material type fallback
        error_flag = 2

    return S_voigt, C_mat, sdv_new, error_flag
