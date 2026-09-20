"""
creep_numba.py
==============
Abaqus Power-Law Secondary Creep Formulation (Norton / Bailey-Norton).
Law: STRAIN (time hardening / strain hardening)

Rate equation:
    \\dot{\\varepsilon}^{cr} = A \\cdot q^n \\cdot t^m

where:
    q   = Mises equivalent deviatoric stress \\sqrt{3/2 s : s}
    t   = total time
    A   = creep multiplier (>= 0)
    n   = stress exponent (typically 1.0 to 10.0)
    m   = time exponent (-1.0 < m <= 0.0, default 0 for steady-state Norton creep)

Time integration:
    Backward Euler (implicit) radial return mapping:
        q_{n+1} + 3G \\Delta \\varepsilon^{cr}_{n+1} = q^{tr}
        \\Delta \\varepsilon^{cr}_{n+1} = A (q_{n+1})^n (t_{n+1})^m \\Delta t

Algorithmic consistent tangent:
    \\mathbf{C}^{cr} = K \\mathbf{I} \\otimes \\mathbf{I} + 2G \\beta \\mathbb{I}_{dev} + 2G (\\gamma - \\beta) \\mathbf{n} \\otimes \\mathbf{n}
"""

from __future__ import annotations

import numpy as np
import numba


@numba.njit(fastmath=True, cache=True)
def integrate_power_law_creep(
    eps_tot: np.ndarray,
    eps_cr_old: np.ndarray,
    q_old: float,
    dt: float,
    t_total: float,
    E: float,
    nu: float,
    A: float,
    n: float,
    m: float,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Radial return integration for power-law creep in 3D Voigt notation.

    Voigt layout: [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy]
    Engineering shear strains are converted to tensor shear strains for deviatoric calculations.

    Returns:
        sigma: (6,) Cauchy stress Voigt
        eps_cr_new: (6,) updated creep strain Voigt
        q_new: updated Mises stress
        delta_p: incremental equivalent creep strain
        C_tangent: (6, 6) consistent tangent operator
    """
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))

    # Volumetric strain
    eps_v = eps_tot[0] + eps_tot[1] + eps_tot[2]
    mean_pressure = K * eps_v

    # Deviatoric total strain (tensor shear)
    dev_tot = np.zeros(6, dtype=np.float64)
    dev_tot[0] = eps_tot[0] - eps_v / 3.0
    dev_tot[1] = eps_tot[1] - eps_v / 3.0
    dev_tot[2] = eps_tot[2] - eps_v / 3.0
    dev_tot[3] = 0.5 * eps_tot[3]
    dev_tot[4] = 0.5 * eps_tot[4]
    dev_tot[5] = 0.5 * eps_tot[5]

    # Deviatoric trial elastic strain: e_tr = dev_tot - eps_cr_old_dev
    dev_cr_old = np.zeros(6, dtype=np.float64)
    dev_cr_old[0] = eps_cr_old[0]
    dev_cr_old[1] = eps_cr_old[1]
    dev_cr_old[2] = eps_cr_old[2]
    dev_cr_old[3] = 0.5 * eps_cr_old[3]
    dev_cr_old[4] = 0.5 * eps_cr_old[4]
    dev_cr_old[5] = 0.5 * eps_cr_old[5]

    e_tr = dev_tot - dev_cr_old
    s_tr = 2.0 * G * e_tr

    # Mises equivalent trial stress
    # s : s = s_xx^2 + s_yy^2 + s_zz^2 + 2*(s_yz^2 + s_xz^2 + s_xy^2)
    s_norm_sq = (
        s_tr[0] ** 2 + s_tr[1] ** 2 + s_tr[2] ** 2 +
        2.0 * (s_tr[3] ** 2 + s_tr[4] ** 2 + s_tr[5] ** 2)
    )
    q_tr = np.sqrt(1.5 * s_norm_sq)

    if q_tr < 1e-12 or A <= 0.0 or dt <= 0.0:
        # Pure elastic step (no creep evolution)
        sigma = np.zeros(6, dtype=np.float64)
        sigma[0] = s_tr[0] + mean_pressure
        sigma[1] = s_tr[1] + mean_pressure
        sigma[2] = s_tr[2] + mean_pressure
        sigma[3] = s_tr[3]
        sigma[4] = s_tr[4]
        sigma[5] = s_tr[5]

        # Standard elastic isotropic tangent
        C = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                C[i, j] = K - (2.0 / 3.0) * G
            C[i, i] += 2.0 * G
        C[3, 3] = G
        C[4, 4] = G
        C[5, 5] = G
        return sigma, eps_cr_old.copy(), q_tr, 0.0, C

    # Time factor for Bailey-Norton: t_eff = max(t_total, dt)
    t_eff = t_total if t_total > 1e-12 else dt
    time_factor = (t_eff ** m) * dt

    # Local 1D Newton-Raphson solve for q_new:
    # Residual: R(q) = q + 3G * A * q^n * time_factor - q_tr = 0
    q = q_tr
    for _ in range(max_iter):
        delta_p_val = A * (q ** n) * time_factor
        R = q + 3.0 * G * delta_p_val - q_tr
        if abs(R) < tol * q_tr:
            break
        # dR/dq = 1 + 3G * A * n * q^(n-1) * time_factor
        dR = 1.0 + 3.0 * G * A * n * (q ** (n - 1.0)) * time_factor
        dq = -R / dR
        q += dq
        if q < 0.0:
            q = 1e-12

    delta_p = A * (q ** n) * time_factor
    ratio = q / q_tr

    # Updated deviatoric stress and Cauchy stress
    s_new = ratio * s_tr
    sigma = np.zeros(6, dtype=np.float64)
    sigma[0] = s_new[0] + mean_pressure
    sigma[1] = s_new[1] + mean_pressure
    sigma[2] = s_new[2] + mean_pressure
    sigma[3] = s_new[3]
    sigma[4] = s_new[4]
    sigma[5] = s_new[5]

    # Flow direction unit tensor: n_dir = (3 / (2*q)) * s_new
    # In tensor components
    n_dir = (1.5 / q) * s_new
    # Delta creep strain increment in Voigt engineering shear
    eps_cr_new = eps_cr_old.copy()
    eps_cr_new[0] += delta_p * n_dir[0]
    eps_cr_new[1] += delta_p * n_dir[1]
    eps_cr_new[2] += delta_p * n_dir[2]
    eps_cr_new[3] += 2.0 * delta_p * n_dir[3]
    eps_cr_new[4] += 2.0 * delta_p * n_dir[4]
    eps_cr_new[5] += 2.0 * delta_p * n_dir[5]

    # Algorithmic consistent tangent:
    # beta = q / q_tr
    # gamma = 1 / (1 + 3G * d(delta_p)/dq)
    # d(delta_p)/dq = A * n * q^(n-1) * time_factor
    d_delta_p_dq = A * n * (q ** (n - 1.0)) * time_factor
    gamma = 1.0 / (1.0 + 3.0 * G * d_delta_p_dq)
    beta = ratio

    # Construct 6x6 C_tangent
    C = np.zeros((6, 6), dtype=np.float64)
    # Volumetric part
    for i in range(3):
        for j in range(3):
            C[i, j] += K - (2.0 / 3.0) * G * beta
        C[i, i] += 2.0 * G * beta
    C[3, 3] = G * beta
    C[4, 4] = G * beta
    C[5, 5] = G * beta

    # Deviatoric contraction (gamma - beta) * n_dir (x) n_dir
    factor = 2.0 * G * (gamma - beta)
    # Voigt scaling for shear DOFs in outer product
    n_v = np.zeros(6, dtype=np.float64)
    n_v[0] = n_dir[0]
    n_v[1] = n_dir[1]
    n_v[2] = n_dir[2]
    n_v[3] = n_dir[3]
    n_v[4] = n_dir[4]
    n_v[5] = n_dir[5]

    for i in range(6):
        for j in range(6):
            C[i, j] += factor * n_v[i] * n_v[j]

    return sigma, eps_cr_new, q, delta_p, C
