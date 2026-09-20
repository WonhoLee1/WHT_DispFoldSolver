"""
Thermal Expansion Adapter Module (Abaqus Formulation Parity).

Computes thermal strain and decomposes total deformation into mechanical and thermal parts:
  - Small strain: eps_me = eps_tot - eps_th
  - Large strain multiplicative: F = F_me @ F_th, where F_th = theta * I, theta = 1 + alpha*(T - T0)
    => F_me = (1 / theta) * F

Fully compatible with Numba njit (fastmath=True, cache=True).
Complies with Abaqus Theory Manual §4.4.1.
"""

import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def compute_thermal_strain_3d(
    alpha: float,
    current_temp: float,
    ref_temp: float,
    strain_th_out: np.ndarray
) -> None:
    """
    Compute 3D thermal strain vector in Voigt notation [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy].

    Parameters
    ----------
    alpha : float
        Coefficient of thermal expansion
    current_temp : float
        Current temperature T
    ref_temp : float
        Reference temperature T0 where thermal strain is zero
    strain_th_out : np.ndarray (float64, shape (6,))
        Output array for thermal strain vector
    """
    delta_t = current_temp - ref_temp
    eps_th = alpha * delta_t

    strain_th_out[0] = eps_th
    strain_th_out[1] = eps_th
    strain_th_out[2] = eps_th
    strain_th_out[3] = 0.0
    strain_th_out[4] = 0.0
    strain_th_out[5] = 0.0


@njit(fastmath=True, cache=True)
def compute_thermal_strain_2d(
    alpha: float,
    current_temp: float,
    ref_temp: float,
    strain_th_out: np.ndarray
) -> None:
    """
    Compute 2D plane strain thermal strain vector in Voigt notation [eps_xx, eps_yy, gamma_xy].

    Parameters
    ----------
    alpha : float
        Coefficient of thermal expansion
    current_temp : float
        Current temperature T
    ref_temp : float
        Reference temperature T0 where thermal strain is zero
    strain_th_out : np.ndarray (float64, shape (3,))
        Output array for thermal strain vector
    """
    delta_t = current_temp - ref_temp
    eps_th = alpha * delta_t

    strain_th_out[0] = eps_th
    strain_th_out[1] = eps_th
    strain_th_out[2] = 0.0


@njit(fastmath=True, cache=True)
def decompose_f_thermal_3d(
    F: np.ndarray,
    alpha: float,
    current_temp: float,
    ref_temp: float,
    F_me_out: np.ndarray
) -> float:
    """
    Multiplicative split of deformation gradient: F = F_me @ F_th.
    For isotropic thermal expansion: F_th = theta * I, where theta = 1 + alpha * delta_T.
    Thus, F_me = (1 / theta) * F.

    Parameters
    ----------
    F : np.ndarray (float64, shape (3, 3))
        Total deformation gradient
    alpha : float
        Thermal expansion coefficient
    current_temp : float
        Current temperature
    ref_temp : float
        Reference temperature
    F_me_out : np.ndarray (float64, shape (3, 3))
        Output mechanical deformation gradient

    Returns
    -------
    float
        Thermal stretch ratio theta = 1 + alpha * delta_T
    """
    delta_t = current_temp - ref_temp
    theta = 1.0 + alpha * delta_t
    inv_theta = 1.0 / theta if abs(theta) > 1e-12 else 1.0

    for i in range(3):
        for j in range(3):
            F_me_out[i, j] = F[i, j] * inv_theta

    return theta


@njit(fastmath=True, cache=True)
def decompose_f_thermal_2d(
    F: np.ndarray,
    alpha: float,
    current_temp: float,
    ref_temp: float,
    F_me_out: np.ndarray
) -> float:
    """
    2D plane strain multiplicative split: F_me = (1 / theta) * F.

    Parameters
    ----------
    F : np.ndarray (float64, shape (2, 2))
        Total 2D deformation gradient
    alpha : float
        Thermal expansion coefficient
    current_temp : float
        Current temperature
    ref_temp : float
        Reference temperature
    F_me_out : np.ndarray (float64, shape (2, 2))
        Output mechanical 2D deformation gradient

    Returns
    -------
    float
        Thermal stretch ratio theta = 1 + alpha * delta_T
    """
    delta_t = current_temp - ref_temp
    theta = 1.0 + alpha * delta_t
    inv_theta = 1.0 / theta if abs(theta) > 1e-12 else 1.0

    for i in range(2):
        for j in range(2):
            F_me_out[i, j] = F[i, j] * inv_theta

    return theta
