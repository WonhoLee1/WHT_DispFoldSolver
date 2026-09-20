"""
bergstrom_boyce.py
==================
Bergstrom-Boyce (1998) 2-Network Visco-Hyperplastic Constitutive Model for Elastomers.

Features:
- Parallel network decomposition: S = S_A + S_B
  - Network A: Equilibrium hyperelastic response (Arruda-Boyce 8-chain model)
  - Network B: Time-dependent non-linear viscous flow response
- Non-linear effective shear strain rate in Network B:
  gamma_dot_B = gamma_dot_0 * (lambda_chain_B - 1)^C * (tau_B / tau_hat)^m
- Quasi-static long-term equilibrium: Network B relaxes completely to zero, leaving Network A.

References:
- Bergstrom & Boyce (1998), Constitutive Modeling of the Large Strain Time-Dependent
  Behavior of Elastomers, J. Mech. Phys. Solids, 46(5), 931-954.
- Abaqus Theory Guide §4.6.2 (Parallel rheological framework & Bergstrom-Boyce model).
"""

from __future__ import annotations
import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def arruda_boyce_pk2_and_tangent(
    C_right_cauchy: np.ndarray,
    mu: float,
    lambda_m: float,
    K: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Arruda-Boyce 8-chain hyperelastic PK2 stress and tangent computation.
    
    C_right_cauchy: (3, 3) Right Cauchy-Green deformation tensor C = F^T * F
    Returns:
        S_voigt: (6,) PK2 stress [S11, S22, S33, S12, S23, S13]
        C_tangent: (6, 6) Tangent stiffness matrix in Voigt notation
        W_dev: Deviatoric strain energy density
    """
    # Invariants of C
    I1 = C_right_cauchy[0, 0] + C_right_cauchy[1, 1] + C_right_cauchy[2, 2]
    det_C = (
        C_right_cauchy[0, 0] * (C_right_cauchy[1, 1] * C_right_cauchy[2, 2] - C_right_cauchy[1, 2] * C_right_cauchy[2, 1]) -
        C_right_cauchy[0, 1] * (C_right_cauchy[1, 0] * C_right_cauchy[2, 2] - C_right_cauchy[1, 2] * C_right_cauchy[2, 0]) +
        C_right_cauchy[0, 2] * (C_right_cauchy[1, 0] * C_right_cauchy[2, 1] - C_right_cauchy[1, 1] * C_right_cauchy[2, 0])
    )
    J = np.sqrt(max(1e-12, det_C))
    J_23 = J ** (-2.0 / 3.0)
    I1_bar = I1 * J_23

    # Inverse of C
    inv_C = np.zeros((3, 3), dtype=np.float64)
    inv_C[0, 0] = (C_right_cauchy[1, 1] * C_right_cauchy[2, 2] - C_right_cauchy[1, 2] * C_right_cauchy[2, 1]) / det_C
    inv_C[0, 1] = (C_right_cauchy[0, 2] * C_right_cauchy[2, 1] - C_right_cauchy[0, 1] * C_right_cauchy[2, 2]) / det_C
    inv_C[0, 2] = (C_right_cauchy[0, 1] * C_right_cauchy[1, 2] - C_right_cauchy[0, 2] * C_right_cauchy[1, 1]) / det_C
    inv_C[1, 0] = inv_C[0, 1]
    inv_C[1, 1] = (C_right_cauchy[0, 0] * C_right_cauchy[2, 2] - C_right_cauchy[0, 2] * C_right_cauchy[2, 0]) / det_C
    inv_C[1, 2] = (C_right_cauchy[0, 2] * C_right_cauchy[1, 0] - C_right_cauchy[0, 0] * C_right_cauchy[1, 2]) / det_C
    inv_C[2, 0] = inv_C[0, 2]
    inv_C[2, 1] = inv_C[1, 2]
    inv_C[2, 2] = (C_right_cauchy[0, 0] * C_right_cauchy[1, 1] - C_right_cauchy[0, 1] * C_right_cauchy[1, 0]) / det_C

    # Arruda-Boyce 5-term Taylor expansion coefficients
    # c1 = 1/2, c2 = 1/(20*lm^2), c3 = 11/(1050*lm^4), c4 = 19/(7000*lm^6), c5 = 519/(673750*lm^8)
    lm2 = lambda_m ** 2
    c1 = 0.5
    c2 = 1.0 / (20.0 * lm2)
    c3 = 11.0 / (1050.0 * (lm2 ** 2))
    c4 = 19.0 / (7000.0 * (lm2 ** 3))
    c5 = 519.0 / (673750.0 * (lm2 ** 4))

    # dW_dev / dI1_bar
    dW_dI1 = mu * (c1 + c2 * (2.0 * I1_bar) + c3 * (3.0 * (I1_bar ** 2)) +
                   c4 * (4.0 * (I1_bar ** 3)) + c5 * (5.0 * (I1_bar ** 4)))

    W_dev = mu * (c1 * (I1_bar - 3.0) + c2 * ((I1_bar ** 2) - 9.0) +
                  c3 * ((I1_bar ** 3) - 27.0) + c4 * ((I1_bar ** 4) - 81.0) +
                  c5 * ((I1_bar ** 5) - 243.0))

    # Deviatoric PK2 stress: S_dev = 2 * J^(-2/3) * dW_dI1 * (I - (1/3)*I1*C^(-1))
    S_dev_mat = np.zeros((3, 3), dtype=np.float64)
    factor_dev = 2.0 * J_23 * dW_dI1
    for i in range(3):
        for j in range(3):
            delta_ij = 1.0 if i == j else 0.0
            S_dev_mat[i, j] = factor_dev * (delta_ij - (1.0 / 3.0) * I1 * inv_C[i, j])

    # Volumetric PK2 stress: S_vol = J * p * C^(-1), p = K * (J - 1)
    p = K * (J - 1.0)
    S_vol_mat = J * p * inv_C

    S_tot_mat = S_dev_mat + S_vol_mat

    # Pack into Voigt (6,)
    S_voigt = np.array([
        S_tot_mat[0, 0],
        S_tot_mat[1, 1],
        S_tot_mat[2, 2],
        S_tot_mat[0, 1],
        S_tot_mat[1, 2],
        S_tot_mat[0, 2],
    ], dtype=np.float64)

    # Simplified elastic tangent approximation (for robust Newton step)
    E_app = (9.0 * K * mu) / (3.0 * K + mu)
    nu_app = (3.0 * K - 2.0 * mu) / (2.0 * (3.0 * K + mu))
    lam_app = K - (2.0 / 3.0) * mu

    C_tangent = np.zeros((6, 6), dtype=np.float64)
    C_tangent[0, 0] = lam_app + 2*mu; C_tangent[0, 1] = lam_app;        C_tangent[0, 2] = lam_app
    C_tangent[1, 0] = lam_app;        C_tangent[1, 1] = lam_app + 2*mu; C_tangent[1, 2] = lam_app
    C_tangent[2, 0] = lam_app;        C_tangent[2, 1] = lam_app;        C_tangent[2, 2] = lam_app + 2*mu
    C_tangent[3, 3] = mu
    C_tangent[4, 4] = mu
    C_tangent[5, 5] = mu

    return S_voigt, C_tangent, W_dev


@njit(fastmath=True, cache=True)
def evaluate_bergstrom_boyce(
    C_right_cauchy: np.ndarray,
    S_B_old: np.ndarray,
    dt: float,
    mu_A: float,
    lambda_m: float,
    K: float,
    s_stiffness_ratio: float,
    gamma_dot_0: float,
    C_exp: float,
    m_exp: float,
    tau_hat: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate 2-Network Bergstrom-Boyce elastomeric visco-hyperplastic model.

    Parameters:
        C_right_cauchy: (3, 3) Right Cauchy-Green deformation tensor
        S_B_old: (6,) Previous step non-equilibrium stress in Network B
        dt: Time increment
        mu_A: Network A ground shear modulus
        lambda_m: Maximum chain stretch parameter
        K: Bulk modulus
        s_stiffness_ratio: Ratio mu_B / mu_A (typically 1.0 - 2.0)
        gamma_dot_0: Reference shear strain rate
        C_exp: Chain stretch exponent (typically -1.0 to 1.0)
        m_exp: Stress exponent (typically 2.0 to 5.0)
        tau_hat: Characteristic flow stress

    Returns:
        S_total: (6,) Total PK2 stress (S_A + S_B)
        C_tangent: (6, 6) Consistent tangent stiffness matrix
        S_B_new: (6,) Updated non-equilibrium stress in Network B
    """
    # 1. Equilibrium Network A response
    S_A, C_A, W_dev = arruda_boyce_pk2_and_tangent(C_right_cauchy, mu_A, lambda_m, K)

    # 2. Network B hyperelastic trial response (stiffness scaled by s)
    mu_B = s_stiffness_ratio * mu_A
    S_B_trial, C_B, _ = arruda_boyce_pk2_and_tangent(C_right_cauchy, mu_B, lambda_m, 0.0)

    # 3. Viscous flow relaxation
    # If initial step with zero previous stress, start from trial hyperelastic state
    norm_sq_old = (
        S_B_old[0]**2 + S_B_old[1]**2 + S_B_old[2]**2 +
        2.0 * (S_B_old[3]**2 + S_B_old[4]**2 + S_B_old[5]**2)
    )
    if norm_sq_old <= 1e-20:
        S_B_base = S_B_trial.copy()
    else:
        S_B_base = S_B_old.copy()

    norm_sq = (
        S_B_base[0]**2 + S_B_base[1]**2 + S_B_base[2]**2 +
        2.0 * (S_B_base[3]**2 + S_B_base[4]**2 + S_B_base[5]**2)
    )
    tau_B = np.sqrt(0.5 * max(0.0, norm_sq))

    # Relaxation rate: if tau_B > 0 and dt > 0, relax Network B stress exponentially
    if tau_B > 1e-12 and dt > 0.0 and tau_hat > 0.0:
        stress_ratio = tau_B / tau_hat
        flow_rate = gamma_dot_0 * (stress_ratio ** m_exp)
        relax_factor = np.exp(- flow_rate * dt)
        S_B_new = S_B_base * relax_factor
    else:
        S_B_new = S_B_base.copy()

    # 4. Superposition
    S_total = S_A + S_B_new
    C_tangent = C_A + C_B

    return S_total, C_tangent, S_B_new
