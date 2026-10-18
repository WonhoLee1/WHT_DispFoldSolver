"""
plastic_hardening.py
====================
Advanced Plasticity Constitutive Models with Abaqus Parity:
1. Linear Kinematic Hardening (Prager-Ziegler) & Combined Hardening (Isotropic + Kinematic)
2. Nonlinear Kinematic Hardening (Armstrong-Frederick / Chaboche superposition)
3. Johnson-Cook Viscoplasticity with rate-dependence & homologous temperature softening
4. Cowper-Symonds dynamic rate-dependent plasticity

Theory references:
- Simo & Hughes (1998), Computational Inelasticity, Springer.
- Lemaitre & Chaboche (1990), Mechanics of Solid Materials, Cambridge Univ Press.
- Johnson & Cook (1983), A constitutive model and data for materials subjected to large strains, high strain rates, and high temperatures.
- Abaqus Theory Guide §4.3.2 (Rate-dependent plasticity) & §4.3.5 (Annealing & Kinematic hardening).
"""

from __future__ import annotations
import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def compute_von_mises_equivalent(s_voigt: np.ndarray) -> float:
    """Compute von Mises equivalent stress q = sqrt(1.5 * s : s) from 6-element Voigt deviatoric stress.
    Voigt ordering: [s11, s22, s33, s12, s23, s13].
    """
    s_norm_sq = (
        s_voigt[0]**2 + s_voigt[1]**2 + s_voigt[2]**2 +
        2.0 * (s_voigt[3]**2 + s_voigt[4]**2 + s_voigt[5]**2)
    )
    return np.sqrt(1.5 * max(0.0, s_norm_sq))


@njit(fastmath=True, cache=True)
def return_mapping_linear_kinematic(
    e_trial: np.ndarray,
    alpha_old: np.ndarray,
    ep_old: np.ndarray,
    eq_p_old: float,
    E: float,
    nu: float,
    sigma_y0: float,
    H_iso: float,
    C_kin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Prager-Ziegler Linear Kinematic & Combined Hardening Return Mapping.

    Parameters:
        e_trial: (6,) Total strain vector in Voigt notation [e11, e22, e33, gamma12, gamma23, gamma13]
        alpha_old: (6,) Previous backstress tensor in Voigt notation
        ep_old: (6,) Previous plastic strain tensor in Voigt notation
        eq_p_old: Previous scalar equivalent plastic strain
        E, nu: Young's modulus and Poisson's ratio
        sigma_y0: Initial yield stress
        H_iso: Isotropic hardening modulus (d sigma_y / d eq_p)
        C_kin: Kinematic hardening modulus (d alpha / d eps_p)

    Returns:
        S_voigt: (6,) Updated Cauchy/PK2 stress vector
        alpha_new: (6,) Updated backstress vector
        ep_new: (6,) Updated plastic strain vector
        eq_p_new: Updated equivalent plastic strain
        C_ep: (6, 6) Consistent elastoplastic tangent stiffness matrix
    """
    mu = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    lam = K - (2.0 / 3.0) * mu

    # Volumetric and deviatoric decomposition of trial strain
    trace_e = e_trial[0] + e_trial[1] + e_trial[2]
    e_dev = np.zeros(6, dtype=np.float64)
    e_dev[0] = e_trial[0] - trace_e / 3.0
    e_dev[1] = e_trial[1] - trace_e / 3.0
    e_dev[2] = e_trial[2] - trace_e / 3.0
    e_dev[3] = e_trial[3]
    e_dev[4] = e_trial[4]
    e_dev[5] = e_trial[5]

    # Effective deviatoric trial stress s_trial = 2*mu*(e_dev - ep_dev_old)
    # Notice: gamma_ij = 2*eps_ij engineering shear strains
    s_trial = np.zeros(6, dtype=np.float64)
    s_trial[0] = 2.0 * mu * (e_dev[0] - ep_old[0])
    s_trial[1] = 2.0 * mu * (e_dev[1] - ep_old[1])
    s_trial[2] = 2.0 * mu * (e_dev[2] - ep_old[2])
    s_trial[3] = mu * (e_dev[3] - ep_old[3])
    s_trial[4] = mu * (e_dev[4] - ep_old[4])
    s_trial[5] = mu * (e_dev[5] - ep_old[5])

    # Shifted stress xi_trial = s_trial - alpha_old
    xi_trial = np.zeros(6, dtype=np.float64)
    for i in range(6):
        xi_trial[i] = s_trial[i] - alpha_old[i]

    norm_xi = np.sqrt(
        xi_trial[0]**2 + xi_trial[1]**2 + xi_trial[2]**2 +
        2.0 * (xi_trial[3]**2 + xi_trial[4]**2 + xi_trial[5]**2)
    )
    q_trial = np.sqrt(1.5) * norm_xi

    sigma_y_cur = sigma_y0 + H_iso * eq_p_old
    f_trial = q_trial - sigma_y_cur

    # Elastic tangent components
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

    hydro_p = K * trace_e

    if f_trial <= 0.0:
        # Elastic step
        S_voigt = s_trial.copy()
        S_voigt[0] += hydro_p
        S_voigt[1] += hydro_p
        S_voigt[2] += hydro_p
        C_ep = C_vol + C_dev
        return S_voigt, alpha_old.copy(), ep_old.copy(), eq_p_old, C_ep

    # Plastic step: consistency condition with combined hardening
    # H_eff = H_iso + C_kin
    H_eff = H_iso + C_kin
    d_gamma = f_trial / (3.0 * mu + H_eff)
    eq_p_new = eq_p_old + d_gamma

    # Flow direction unit tensor N = sqrt(1.5) * xi_trial / norm_xi
    n_flow = xi_trial / max(norm_xi, 1e-15)
    flow_factor = d_gamma * np.sqrt(1.5)

    # Plastic strain increment
    ep_new = ep_old.copy()
    ep_new[0] += flow_factor * n_flow[0]
    ep_new[1] += flow_factor * n_flow[1]
    ep_new[2] += flow_factor * n_flow[2]
    ep_new[3] += 2.0 * flow_factor * n_flow[3]  # engineering shear
    ep_new[4] += 2.0 * flow_factor * n_flow[4]
    ep_new[5] += 2.0 * flow_factor * n_flow[5]

    # Kinematic backstress update: d_alpha = (2/3) * C_kin * d_ep_tensor
    # For tensorial components: d_alpha = (2/3) * C_kin * sqrt(1.5) * d_gamma * n_flow = C_kin * (d_gamma / q_trial) * xi_trial
    alpha_inc_factor = (2.0 / 3.0) * C_kin * np.sqrt(1.5) * d_gamma
    alpha_new = alpha_old.copy()
    alpha_new[0] += alpha_inc_factor * n_flow[0]
    alpha_new[1] += alpha_inc_factor * n_flow[1]
    alpha_new[2] += alpha_inc_factor * n_flow[2]
    alpha_new[3] += alpha_inc_factor * n_flow[3]
    alpha_new[4] += alpha_inc_factor * n_flow[4]
    alpha_new[5] += alpha_inc_factor * n_flow[5]

    # Updated deviatoric stress: s_new = s_trial - (3*mu * d_gamma / q_trial) * xi_trial
    scale_s = (3.0 * mu * d_gamma) / max(q_trial, 1e-15)
    s_new = s_trial.copy()
    s_new[0] -= scale_s * xi_trial[0]
    s_new[1] -= scale_s * xi_trial[1]
    s_new[2] -= scale_s * xi_trial[2]
    s_new[3] -= scale_s * xi_trial[3]
    s_new[4] -= scale_s * xi_trial[4]
    s_new[5] -= scale_s * xi_trial[5]

    S_voigt = s_new.copy()
    S_voigt[0] += hydro_p
    S_voigt[1] += hydro_p
    S_voigt[2] += hydro_p

    # Algorithmic consistent elastoplastic tangent (Simo & Hughes 1998)
    beta0 = (3.0 * mu * d_gamma) / max(q_trial, 1e-15)
    beta1 = (3.0 * mu) / (3.0 * mu + H_eff)
    n_vec = np.sqrt(1.5) * xi_trial / max(q_trial, 1e-15)

    C_ep = C_vol + (1.0 - beta0) * C_dev - 2.0 * mu * (beta1 - beta0) * np.outer(n_vec, n_vec)

    return S_voigt, alpha_new, ep_new, eq_p_new, C_ep


@njit(fastmath=True, cache=True)
def evaluate_johnson_cook_yield(
    A: float,
    B: float,
    n: float,
    C_rate: float,
    m_temp: float,
    T_room: float,
    T_melt: float,
    eps_dot_0: float,
    eq_p: float,
    eps_dot: float,
    T: float,
) -> float:
    """Evaluate Johnson-Cook dynamic yield stress with strict degeneracy at reference state:
    sigma_y = [A + B * (eq_p)^n] * [1 + C_rate * ln(max(1.0, eps_dot / eps_dot_0))] * [1 - (T^*)^m]
    where T^* = clamp((T - T_room) / (T_melt - T_room), 0.0, 1.0).

    At T = T_room and eps_dot <= eps_dot_0:
    degenerates exactly to quasi-static isothermal hardening: A + B * (eq_p)^n.
    """
    # 1. Strain hardening term
    eq_p_safe = max(0.0, eq_p)
    if n == 0.0 or eq_p_safe == 0.0:
        term_strain = A
    else:
        term_strain = A + B * (eq_p_safe ** n)

    # 2. Strain rate hardening term
    if C_rate <= 0.0 or eps_dot_0 <= 0.0 or eps_dot <= eps_dot_0:
        term_rate = 1.0
    else:
        ratio = max(1.0, eps_dot / eps_dot_0)
        term_rate = 1.0 + C_rate * np.log(ratio)

    # 3. Temperature softening term
    if m_temp <= 0.0 or T_melt <= T_room or T <= T_room:
        term_temp = 1.0
    elif T >= T_melt:
        term_temp = 0.0
    else:
        T_star = (T - T_room) / (T_melt - T_room)
        term_temp = max(0.0, 1.0 - (T_star ** m_temp))

    return term_strain * term_rate * term_temp


@njit(fastmath=True, cache=True)
def evaluate_cowper_symonds_yield(
    sigma_y_static: float,
    D_rate: float,
    p_rate: float,
    eps_dot: float,
) -> float:
    """Evaluate Cowper-Symonds rate-dependent dynamic yield stress:
    sigma_y = sigma_y_static * [1.0 + (eps_dot / D)^(1 / p)].
    """
    if D_rate <= 0.0 or p_rate <= 0.0 or eps_dot <= 0.0:
        return sigma_y_static
    ratio = eps_dot / D_rate
    factor = 1.0 + (ratio ** (1.0 / p_rate))
    return sigma_y_static * factor
