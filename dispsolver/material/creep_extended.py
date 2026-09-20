# -*- coding: utf-8 -*-
"""creep_extended.py: hyperbolic-sine creep + Anand viscoplasticity (Numba).

1. Hyperbolic-sine (Garofalo, Abaqus *CREEP LAW=HYPERB, time hardening):
     deps_cr = A * sinh(B*q)^n * t^m * dt
   Small-Bq limit recovers Norton power law with A' = A*B^n (cross-checked
   in tests). sinh/cosh arguments are capped at 50 (float64-safe; B*q > 50
   is non-physical for calibrated constants).

2. Anand viscoplasticity (Abaqus *CREEP LAW=ANAND; Brown-Anand 1989):
     eps_dot = A * exp(-QR/T) * sinh(xi*q/s)^(1/m)
     s_dot   = h0 * |1 - s/s*|^a * sign(1 - s/s*) * eps_dot
     s*      = shat * (eps_dot / (A*exp(-QR/T)))^nn
   Implicit backward-Euler radial return with a coupled 2x2 Newton solve
   for (q, s) and the fully linearized consistent tangent (no fixed-s
   approximation). All Voigt layouts match creep_numba.py:
   [eps_xx, eps_yy, eps_zz, gamma_yz, gamma_xz, gamma_xy].
"""

from __future__ import annotations

import numpy as np
import numba


@numba.njit(fastmath=True, cache=True)
def _sinh_capped(x: float) -> float:
    """sinh with argument cap (float64 overflow guard)."""
    if x > 50.0:
        x = 50.0
    elif x < -50.0:
        x = -50.0
    return np.sinh(x)


@numba.njit(fastmath=True, cache=True)
def _cosh_capped(x: float) -> float:
    """cosh with argument cap (float64 overflow guard)."""
    if x > 50.0:
        x = 50.0
    elif x < -50.0:
        x = -50.0
    return np.cosh(x)


@numba.njit(fastmath=True, cache=True)
def integrate_sinh_creep(
    eps_tot: np.ndarray,
    eps_cr_old: np.ndarray,
    q_old: float,
    dt: float,
    t_total: float,
    E: float,
    nu: float,
    A: float,
    B: float,
    n: float,
    m: float,
    tol: float = 1e-10,
    max_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    """Radial return integration for hyperbolic-sine creep, 3D Voigt.

    Returns (sigma, eps_cr_new, q_new, delta_p, C_tangent) -- same layout
    as creep_numba.integrate_power_law_creep.
    """
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))

    eps_v = eps_tot[0] + eps_tot[1] + eps_tot[2]
    mean_pressure = K * eps_v

    dev_tot = np.zeros(6, dtype=np.float64)
    dev_tot[0] = eps_tot[0] - eps_v / 3.0
    dev_tot[1] = eps_tot[1] - eps_v / 3.0
    dev_tot[2] = eps_tot[2] - eps_v / 3.0
    dev_tot[3] = 0.5 * eps_tot[3]
    dev_tot[4] = 0.5 * eps_tot[4]
    dev_tot[5] = 0.5 * eps_tot[5]

    dev_cr_old = np.zeros(6, dtype=np.float64)
    dev_cr_old[0] = eps_cr_old[0]
    dev_cr_old[1] = eps_cr_old[1]
    dev_cr_old[2] = eps_cr_old[2]
    dev_cr_old[3] = 0.5 * eps_cr_old[3]
    dev_cr_old[4] = 0.5 * eps_cr_old[4]
    dev_cr_old[5] = 0.5 * eps_cr_old[5]

    e_tr = dev_tot - dev_cr_old
    s_tr = 2.0 * G * e_tr

    s_norm_sq = (
        s_tr[0] ** 2 + s_tr[1] ** 2 + s_tr[2] ** 2 +
        2.0 * (s_tr[3] ** 2 + s_tr[4] ** 2 + s_tr[5] ** 2)
    )
    q_tr = np.sqrt(1.5 * s_norm_sq)

    if q_tr < 1e-12 or A <= 0.0 or dt <= 0.0 or B <= 0.0:
        sigma = np.zeros(6, dtype=np.float64)
        sigma[0] = s_tr[0] + mean_pressure
        sigma[1] = s_tr[1] + mean_pressure
        sigma[2] = s_tr[2] + mean_pressure
        sigma[3] = s_tr[3]
        sigma[4] = s_tr[4]
        sigma[5] = s_tr[5]
        C = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                C[i, j] = K - (2.0 / 3.0) * G
            C[i, i] += 2.0 * G
        C[3, 3] = G
        C[4, 4] = G
        C[5, 5] = G
        return sigma, eps_cr_old.copy(), q_tr, 0.0, C

    t_eff = t_total if t_total > 1e-12 else dt
    time_factor = (t_eff ** m) * dt

    q = q_tr
    for _ in range(max_iter):
        sh = _sinh_capped(B * q)
        delta_p_val = A * (sh ** n) * time_factor
        R = q + 3.0 * G * delta_p_val - q_tr
        if abs(R) < tol * q_tr:
            break
        ch = _cosh_capped(B * q)
        dR = 1.0 + 3.0 * G * A * n * (sh ** (n - 1.0)) * ch * B * time_factor
        q += -R / dR
        if q < 0.0:
            q = 1e-12

    sh = _sinh_capped(B * q)
    delta_p = A * (sh ** n) * time_factor
    ratio = q / q_tr

    s_new = ratio * s_tr
    sigma = np.zeros(6, dtype=np.float64)
    sigma[0] = s_new[0] + mean_pressure
    sigma[1] = s_new[1] + mean_pressure
    sigma[2] = s_new[2] + mean_pressure
    sigma[3] = s_new[3]
    sigma[4] = s_new[4]
    sigma[5] = s_new[5]

    n_dir = (1.5 / q) * s_new
    eps_cr_new = eps_cr_old.copy()
    eps_cr_new[0] += delta_p * n_dir[0]
    eps_cr_new[1] += delta_p * n_dir[1]
    eps_cr_new[2] += delta_p * n_dir[2]
    eps_cr_new[3] += 2.0 * delta_p * n_dir[3]
    eps_cr_new[4] += 2.0 * delta_p * n_dir[4]
    eps_cr_new[5] += 2.0 * delta_p * n_dir[5]

    ch = _cosh_capped(B * q)
    d_delta_p_dq = A * n * (sh ** (n - 1.0)) * ch * B * time_factor
    gamma = 1.0 / (1.0 + 3.0 * G * d_delta_p_dq)
    beta = ratio

    C = np.zeros((6, 6), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            C[i, j] += K - (2.0 / 3.0) * G * beta
        C[i, i] += 2.0 * G * beta
    C[3, 3] = G * beta
    C[4, 4] = G * beta
    C[5, 5] = G * beta

    # Rank-one update coefficient 4G/3 (n_dir norm^2 = 3/2); see creep_numba.py.
    factor = (4.0 / 3.0) * G * (gamma - beta)
    for i in range(6):
        for j in range(6):
            C[i, j] += factor * n_dir[i] * n_dir[j]

    return sigma, eps_cr_new, q, delta_p, C


@numba.njit(fastmath=True, cache=True)
def integrate_anand(
    eps_tot: np.ndarray,
    eps_cr_old: np.ndarray,
    s_old: float,
    dt: float,
    T: float,
    E: float,
    nu: float,
    s0: float,
    QR: float,
    A: float,
    xi: float,
    m: float,
    h0: float,
    shat: float,
    nn: float,
    a_exp: float,
    tol: float = 1e-10,
    max_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray, float, float, float, np.ndarray]:
    """Implicit Anand viscoplasticity, radial return + coupled (q, s) Newton.

    Params: s0 initial deformation resistance, QR = Q/R activation term (K),
    A pre-exponential (/s), xi stress multiplier, m rate sensitivity,
    h0 hardening constant, shat saturation coefficient, nn saturation rate
    exponent, a_exp hardening exponent. T in Kelvin.

    Returns (sigma, eps_cr_new, q_new, delta_p, s_new, C_tangent, converged).
    converged is False when the Newton loop exits without meeting tol
    (NaN guard, singular Jacobian, or max_iter) -- never trust (q, s)
    when it is False.
    """
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))

    eps_v = eps_tot[0] + eps_tot[1] + eps_tot[2]
    mean_pressure = K * eps_v

    dev_tot = np.zeros(6, dtype=np.float64)
    dev_tot[0] = eps_tot[0] - eps_v / 3.0
    dev_tot[1] = eps_tot[1] - eps_v / 3.0
    dev_tot[2] = eps_tot[2] - eps_v / 3.0
    dev_tot[3] = 0.5 * eps_tot[3]
    dev_tot[4] = 0.5 * eps_tot[4]
    dev_tot[5] = 0.5 * eps_tot[5]

    dev_cr_old = np.zeros(6, dtype=np.float64)
    dev_cr_old[0] = eps_cr_old[0]
    dev_cr_old[1] = eps_cr_old[1]
    dev_cr_old[2] = eps_cr_old[2]
    dev_cr_old[3] = 0.5 * eps_cr_old[3]
    dev_cr_old[4] = 0.5 * eps_cr_old[4]
    dev_cr_old[5] = 0.5 * eps_cr_old[5]

    e_tr = dev_tot - dev_cr_old
    s_tr = 2.0 * G * e_tr

    s_norm_sq = (
        s_tr[0] ** 2 + s_tr[1] ** 2 + s_tr[2] ** 2 +
        2.0 * (s_tr[3] ** 2 + s_tr[4] ** 2 + s_tr[5] ** 2)
    )
    q_tr = np.sqrt(1.5 * s_norm_sq)

    if q_tr < 1e-12 or A <= 0.0 or dt <= 0.0 or s_old <= 0.0:
        sigma = np.zeros(6, dtype=np.float64)
        sigma[0] = s_tr[0] + mean_pressure
        sigma[1] = s_tr[1] + mean_pressure
        sigma[2] = s_tr[2] + mean_pressure
        sigma[3] = s_tr[3]
        sigma[4] = s_tr[4]
        sigma[5] = s_tr[5]
        C = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                C[i, j] = K - (2.0 / 3.0) * G
            C[i, i] += 2.0 * G
        C[3, 3] = G
        C[4, 4] = G
        C[5, 5] = G
        return sigma, eps_cr_old.copy(), q_tr, 0.0, s_old, C, True

    # Arrhenius factor, guarded
    qrt = QR / max(T, 1.0)
    if qrt > 700.0:
        qrt = 700.0
    elif qrt < -50.0:
        qrt = -50.0
    arr = A * np.exp(-qrt)
    pwr = 1.0 / m
    inv_dt_arr = 1.0 / max(dt * arr, 1e-300)

    q = q_tr
    s = s_old
    converged = False
    for _ in range(max_iter):
        sq = xi * q / s
        u = _sinh_capped(sq)
        rate = arr * (u ** pwr)
        r = rate * dt
        e_rate = r * inv_dt_arr
        s_star = shat * (e_rate ** nn) if e_rate > 0.0 else 0.0
        w = 1.0 - s / max(s_star, 1e-300)
        if abs(w) < 1e-12:
            g = 0.0
            dg_dw = 0.0
        else:
            sgn = 1.0 if w > 0.0 else -1.0
            g = h0 * ((abs(w) ** a_exp) * sgn)
            dg_dw = h0 * a_exp * (abs(w) ** (a_exp - 1.0))
        R1 = q + 3.0 * G * r - q_tr
        R2 = s - s_old - g * r
        if abs(R1) < tol * q_tr and abs(R2) < tol * max(s_old, 1.0):
            converged = True
            break
        # Analytic Jacobian
        ch = _cosh_capped(sq)
        dr_dq = dt * arr * pwr * (u ** (pwr - 1.0)) * ch * (xi / s)
        dr_ds = dt * arr * pwr * (u ** (pwr - 1.0)) * ch * (-xi * q / (s * s))
        de_dq = dr_dq * inv_dt_arr
        de_ds = dr_ds * inv_dt_arr
        dsst_dq = shat * nn * (e_rate ** (nn - 1.0)) * de_dq if e_rate > 0.0 else 0.0
        dsst_ds = shat * nn * (e_rate ** (nn - 1.0)) * de_ds if e_rate > 0.0 else 0.0
        ss2 = max(s_star, 1e-300) ** 2
        dw_dq = (s / ss2) * dsst_dq
        dw_ds = -1.0 / max(s_star, 1e-300) + (s / ss2) * dsst_ds
        dg_dq = dg_dw * dw_dq
        dg_ds = dg_dw * dw_ds
        J11 = 1.0 + 3.0 * G * dr_dq
        J12 = 3.0 * G * dr_ds
        J21 = -(dg_dq * r + g * dr_dq)
        J22 = 1.0 - (dg_ds * r + g * dr_ds)
        det = J11 * J22 - J12 * J21
        if abs(det) < 1e-300:
            break
        dq = (-R1 * J22 + R2 * J12) / det
        ds = (-J11 * R2 + J21 * R1) / det
        if dq != dq or ds != ds:
            break
        q += dq
        s += ds
        if q < 0.0:
            q = 1e-12
        if s < 1e-12:
            s = 1e-12

    sq = xi * q / s
    u = _sinh_capped(sq)
    rate = arr * (u ** pwr)
    delta_p = rate * dt
    ratio = q / q_tr

    s_new_v = ratio * s_tr
    sigma = np.zeros(6, dtype=np.float64)
    sigma[0] = s_new_v[0] + mean_pressure
    sigma[1] = s_new_v[1] + mean_pressure
    sigma[2] = s_new_v[2] + mean_pressure
    sigma[3] = s_new_v[3]
    sigma[4] = s_new_v[4]
    sigma[5] = s_new_v[5]

    n_dir = (1.5 / q) * s_new_v
    eps_cr_new = eps_cr_old.copy()
    eps_cr_new[0] += delta_p * n_dir[0]
    eps_cr_new[1] += delta_p * n_dir[1]
    eps_cr_new[2] += delta_p * n_dir[2]
    eps_cr_new[3] += 2.0 * delta_p * n_dir[3]
    eps_cr_new[4] += 2.0 * delta_p * n_dir[4]
    eps_cr_new[5] += 2.0 * delta_p * n_dir[5]

    # Fully linearized consistent tangent via 2x2 sensitivity.
    ch = _cosh_capped(sq)
    r_q = dt * arr * pwr * (u ** (pwr - 1.0)) * ch * (xi / s)
    r_s = dt * arr * pwr * (u ** (pwr - 1.0)) * ch * (-xi * q / (s * s))
    e_rate = delta_p * inv_dt_arr
    dsst_q = shat * nn * (e_rate ** (nn - 1.0)) * (r_q * inv_dt_arr) if e_rate > 0.0 else 0.0
    dsst_s = shat * nn * (e_rate ** (nn - 1.0)) * (r_s * inv_dt_arr) if e_rate > 0.0 else 0.0
    ss2 = max(shat * (e_rate ** nn) if e_rate > 0.0 else 0.0, 1e-300) ** 2
    s_star = np.sqrt(ss2)
    w = 1.0 - s / max(s_star, 1e-300)
    if abs(w) < 1e-12:
        dg_dw = 0.0
        gg = 0.0
    else:
        sgn = 1.0 if w > 0.0 else -1.0
        gg = h0 * ((abs(w) ** a_exp) * sgn)
        dg_dw = h0 * a_exp * (abs(w) ** (a_exp - 1.0))
    dw_q = (s / ss2) * dsst_q
    dw_s = -1.0 / max(s_star, 1e-300) + (s / ss2) * dsst_s
    g_q = dg_dw * dw_q
    g_s = dg_dw * dw_s
    den = 1.0 - g_s * delta_p - gg * r_s
    if abs(den) < 1e-14:
        big_gamma = 1.0 / (1.0 + 3.0 * G * r_q)
    else:
        ell = (g_q * delta_p + gg * r_q) / den
        big_gamma = 1.0 / (1.0 + 3.0 * G * (r_q + r_s * ell))
    beta = ratio

    C = np.zeros((6, 6), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            C[i, j] += K - (2.0 / 3.0) * G * beta
        C[i, i] += 2.0 * G * beta
    C[3, 3] = G * beta
    C[4, 4] = G * beta
    C[5, 5] = G * beta

    # Rank-one update coefficient 4G/3 (n_dir norm^2 = 3/2); see creep_numba.py.
    # big_gamma is the fully linearized Anand sensitivity (2x2), reducing to
    # the power-law gamma when the s-coupling vanishes.
    factor = (4.0 / 3.0) * G * (big_gamma - beta)
    for i in range(6):
        for j in range(6):
            C[i, j] += factor * n_dir[i] * n_dir[j]

    return sigma, eps_cr_new, q, delta_p, s, C, converged
