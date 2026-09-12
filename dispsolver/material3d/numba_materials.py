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
        return 6  # viscous history tensor (6) per term
    elif mat_type == MAT_CUSTOM_ELASTIC:
        return 0
    return 0


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

        lam = K - (2.0 / 3.0) * mu
        
        C_mat[0, 0] = lam + 2*mu; C_mat[0, 1] = lam;        C_mat[0, 2] = lam
        C_mat[1, 0] = lam;        C_mat[1, 1] = lam + 2*mu; C_mat[1, 2] = lam
        C_mat[2, 0] = lam;        C_mat[2, 1] = lam;        C_mat[2, 2] = lam + 2*mu
        C_mat[3, 3] = mu
        C_mat[4, 4] = mu
        C_mat[5, 5] = mu
        
        S_voigt = C_mat @ E_voigt

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
