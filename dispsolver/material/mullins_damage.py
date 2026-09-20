"""
mullins_damage.py
=================
Ogden-Roxburgh (1999) Continuum Damage Mechanics for Hyperelastic Mullins Effect:
W(F, eta) = eta * W_dev(F) + W_vol(J) + phi(eta)
eta = 1.0 - (1.0 / r) * erf((U_max - U_dev) / (m + beta * U_max))

Where:
- U_max: Historical maximum deviatoric strain energy density (internal state variable)
- U_dev: Current deviatoric strain energy density
- r >= 1.0: Damage saturation parameter
- m > 0.0: Damage onset/rate parameter
- beta >= 0.0: Historical maximum energy scaling parameter
- phi(eta): Energy dissipation function associated with damage

Stress response:
S = eta * S_0_dev + S_vol

Consistent tangent:
C = eta * C_0_dev + C_vol + (d_eta / d_U_dev) * (S_0_dev (x) S_0_dev)

References:
- Ogden & Roxburgh (1999), A pseudo-elastic model for the Mullins effect in filled rubber,
  Proc. R. Soc. Lond. A, 455, 2861-2877.
- Abaqus Theory Guide §4.7.1 (Mullins effect).
"""

from __future__ import annotations
import math
import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def evaluate_mullins_damage_factor(
    U_dev: float,
    U_max: float,
    r: float,
    m: float,
    beta: float,
) -> tuple[float, float]:
    """Compute Ogden-Roxburgh Mullins damage factor eta and its derivative d_eta / d_U_dev.

    Parameters:
        U_dev: Current deviatoric strain energy density
        U_max: Maximum historical deviatoric strain energy density
        r: Damage parameter (r >= 1.0)
        m: Damage parameter (m > 0.0)
        beta: Damage parameter (beta >= 0.0)

    Returns:
        eta: Damage scaling factor (0.0 < eta <= 1.0)
        d_eta_dU: Derivative of eta with respect to U_dev (>= 0.0)
    """
    if U_max <= 1e-12 or U_dev >= U_max or r <= 1.0:
        # Virgin loading or complete reloading to historical maximum: no damage (eta = 1.0)
        return 1.0, 0.0

    denom = m + beta * U_max
    if denom <= 1e-15:
        return 1.0, 0.0

    delta_U = max(0.0, U_max - U_dev)
    x = delta_U / denom

    # math.erf(x)
    erf_val = math.erf(x)
    eta = 1.0 - (1.0 / r) * erf_val
    eta = max(1e-6, min(1.0, eta))

    # d_eta / d_U_dev = - (1 / r) * (2 / sqrt(pi)) * exp(-x^2) * (-1 / denom)
    #                 = + (2.0 / (r * sqrt(pi) * denom)) * exp(-x^2)
    two_over_sqrt_pi = 1.1283791670955125739  # 2.0 / sqrt(pi)
    d_eta_dU = (two_over_sqrt_pi / (r * denom)) * math.exp(- (x ** 2))

    return eta, d_eta_dU


@njit(fastmath=True, cache=True)
def apply_mullins_damage_to_pk2(
    S0_voigt: np.ndarray,
    C0_mat: np.ndarray,
    U_dev: float,
    U_max_old: float,
    r: float,
    m: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply Mullins damage factor to un-damaged hyperelastic PK2 stress and tangent.

    Parameters:
        S0_voigt: (6,) Undamaged PK2 stress in Voigt notation [S11, S22, S33, S12, S23, S13]
        C0_mat: (6, 6) Undamaged spatial/material tangent matrix
        U_dev: Current deviatoric strain energy density
        U_max_old: Previous step historical maximum deviatoric energy
        r, m, beta: Mullins damage parameters

    Returns:
        S_mullins: (6,) Mullins damaged PK2 stress
        C_mullins: (6, 6) Damaged consistent tangent
        U_max_new: Updated maximum historical deviatoric energy
    """
    # 1. Update historical maximum
    U_max_new = max(U_max_old, U_dev)

    # 2. Volumetric-deviatoric stress decomposition
    # Trace of S: p_hydro = (S11 + S22 + S33) / 3.0
    p_hydro = (S0_voigt[0] + S0_voigt[1] + S0_voigt[2]) / 3.0
    S0_dev = S0_voigt.copy()
    S0_dev[0] -= p_hydro
    S0_dev[1] -= p_hydro
    S0_dev[2] -= p_hydro

    # 3. Evaluate damage factor eta and d_eta / d_U_dev
    eta, d_eta_dU = evaluate_mullins_damage_factor(U_dev, U_max_new, r, m, beta)

    # 4. Damaged PK2 stress: S = eta * S0_dev + S_vol
    S_mullins = S0_dev * eta
    S_mullins[0] += p_hydro
    S_mullins[1] += p_hydro
    S_mullins[2] += p_hydro

    # 5. Damaged consistent tangent:
    # C_mullins = eta * C0_dev + C0_vol + d_eta_dU * (S0_dev (x) S0_dev)
    # Decompose C0:
    C_vol = np.zeros((6, 6), dtype=np.float64)
    # Row and col averages for volumetric part
    for i in range(3):
        for j in range(3):
            C_vol[i, j] = (C0_mat[0, 0] + C0_mat[1, 1] + C0_mat[2, 2] + 
                           2.0 * (C0_mat[0, 1] + C0_mat[1, 2] + C0_mat[0, 2])) / 9.0
    C0_dev = C0_mat - C_vol

    C_mullins = eta * C0_dev + C_vol

    if d_eta_dU > 1e-15:
        for i in range(6):
            for j in range(6):
                C_mullins[i, j] += d_eta_dU * S0_dev[i] * S0_dev[j]

    return S_mullins, C_mullins, U_max_new
