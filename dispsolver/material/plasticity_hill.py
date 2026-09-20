# -*- coding: utf-8 -*-
"""plasticity_hill.py: Hill48 anisotropic yield, associated flow, linear hardening.

Voigt ordering [11,22,33,12,23,13] with engineering shear strains throughout,
same convention as plastic_hardening.py. Flow vector in engineering form
m_eng = [m11,m22,m33,2*m12,2*m23,2*m13] so that m_eng^T C_eng m_eng equals
the tensor contraction m:C:m (the shear doubling compensates C's mu diagonal).
"""

from __future__ import annotations
import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def hill48_phi(s, F, G, H, L, M, N):
    """Hill48 equivalent stress from tensor-component Voigt stress."""
    q2 = (F * (s[1] - s[2]) ** 2 + G * (s[2] - s[0]) ** 2
          + H * (s[0] - s[1]) ** 2
          + 2.0 * N * s[3] ** 2 + 2.0 * L * s[4] ** 2 + 2.0 * M * s[5] ** 2)
    return np.sqrt(max(0.0, q2))


@njit(fastmath=True, cache=True)
def _hill_m_eng(s, phi, F, G, H, L, M, N):
    m = np.zeros(6, dtype=np.float64)
    if phi <= 1e-300:
        return m
    m[0] = ((G + H) * s[0] - H * s[1] - G * s[2]) / phi
    m[1] = (-H * s[0] + (F + H) * s[1] - F * s[2]) / phi
    m[2] = (-G * s[0] - F * s[1] + (F + G) * s[2]) / phi
    m[3] = 2.0 * (2.0 * N * s[3]) / phi
    m[4] = 2.0 * (2.0 * L * s[4]) / phi
    m[5] = 2.0 * (2.0 * M * s[5]) / phi
    return m


@njit(fastmath=True, cache=True)
def _solve6(A, B):
    """Solve A X = B (6x6, B 6x6) by Gauss-Jordan with partial pivot. A kept."""
    M = np.zeros((6, 12), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            M[i, j] = A[i, j]
            M[i, 6 + j] = B[i, j]
    for c in range(6):
        piv = c
        best = abs(M[c, c])
        for r in range(c + 1, 6):
            if abs(M[r, c]) > best:
                best = abs(M[r, c])
                piv = r
        if best < 1e-300:
            Z = np.zeros((6, 6), dtype=np.float64)
            return Z
        if piv != c:
            for j in range(12):
                tmp = M[c, j]
                M[c, j] = M[piv, j]
                M[piv, j] = tmp
        inv = 1.0 / M[c, c]
        for j in range(12):
            M[c, j] *= inv
        for r in range(6):
            if r != c:
                f = M[r, c]
                if f != 0.0:
                    for j in range(12):
                        M[r, j] -= f * M[c, j]
    X = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            X[i, j] = M[i, 6 + j]
    return X


@njit(fastmath=True, cache=True)
def _solve6v(A, b):
    """Solve A x = b (6x6, 6-vector) by Gauss-Jordan with partial pivot."""
    M = np.zeros((6, 7), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            M[i, j] = A[i, j]
        M[i, 6] = b[i]
    for c in range(6):
        piv = c
        best = abs(M[c, c])
        for r in range(c + 1, 6):
            if abs(M[r, c]) > best:
                best = abs(M[r, c])
                piv = r
        if best < 1e-300:
            return np.zeros(6, dtype=np.float64)
        if piv != c:
            for j in range(7):
                tmp = M[c, j]
                M[c, j] = M[piv, j]
                M[piv, j] = tmp
        inv = 1.0 / M[c, c]
        for j in range(7):
            M[c, j] *= inv
        for r in range(6):
            if r != c:
                f = M[r, c]
                if f != 0.0:
                    for j in range(7):
                        M[r, j] -= f * M[c, j]
    x = np.zeros(6, dtype=np.float64)
    for i in range(6):
        x[i] = M[i, 6]
    return x


@njit(fastmath=True, cache=True)
def _c_eng(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    C = np.zeros((6, 6), dtype=np.float64)
    for r in range(3):
        for c in range(3):
            C[r, c] = K
            C[r, c] += (-(2.0 / 3.0) * mu) if True else 0.0
        C[r, r] += 2.0 * mu
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C, K


@njit(fastmath=True, cache=True)
def return_mapping_hill48(e_trial, ep_old, eq_p_old, E, nu,
                          F, G, H, L, M, N, sigma_y0, H_iso):
    """Elastic predictor + iterated scalar return map.

    Returns S_voigt, ep_new, eq_p_new, C_ep. Plastic multiplier iterated to
    |f| < 1e-13*sigma_y0 (max 50); linear hardening keeps it to a few passes.
    Tangent is the standard associated-flow linearization at the converged m.
    """
    C, K = _c_eng(E, nu)
    trace_e = e_trial[0] + e_trial[1] + e_trial[2]
    hydro = K * trace_e
    s_tr = np.zeros(6, dtype=np.float64)
    for i in range(6):
        v = 0.0
        for j in range(6):
            v += C[i, j] * (e_trial[j] - ep_old[j])
        s_tr[i] = v
    phi_tr = hill48_phi(s_tr, F, G, H, L, M, N)
    if phi_tr <= sigma_y0 + H_iso * eq_p_old:
        S = s_tr.copy()
        return S, ep_old.copy(), eq_p_old, C.copy()
    lam = 0.0
    eq_p = eq_p_old
    s = s_tr.copy()
    for _ in range(50):
        phi = hill48_phi(s, F, G, H, L, M, N)
        m = _hill_m_eng(s, phi, F, G, H, L, M, N)
        mt = m.copy()
        mt[3] *= 0.5
        mt[4] *= 0.5
        mt[5] *= 0.5
        Cm = np.zeros(6, dtype=np.float64)
        for i in range(6):
            v = 0.0
            for j in range(6):
                v += C[i, j] * m[j]
            Cm[i] = v
        Rs = np.zeros(6, dtype=np.float64)
        for i in range(6):
            Rs[i] = s[i] - s_tr[i] + lam * Cm[i]
        Rf = phi - (sigma_y0 + H_iso * (eq_p_old + lam))
        if np.sqrt(Rs[0]**2 + Rs[1]**2 + Rs[2]**2 + Rs[3]**2 + Rs[4]**2 + Rs[5]**2) < 1e-12 * max(1.0, phi_tr) and abs(Rf) < 1e-13 * max(1.0, sigma_y0):
            break
        Mt = np.zeros((6, 6), dtype=np.float64)
        Mt[0, 0] = G + H
        Mt[0, 1] = -H
        Mt[0, 2] = -G
        Mt[1, 0] = -H
        Mt[1, 1] = F + H
        Mt[1, 2] = -F
        Mt[2, 0] = -G
        Mt[2, 1] = -F
        Mt[2, 2] = F + G
        Mt[3, 3] = 2.0 * N
        Mt[4, 4] = 2.0 * L
        Mt[5, 5] = 2.0 * M
        W = np.zeros((6, 6), dtype=np.float64)
        fac = lam / max(phi, 1e-300)
        for i in range(6):
            for j in range(6):
                v = 0.0
                for k in range(6):
                    pk = Mt[k, j]
                    if k >= 3:
                        pk *= 2.0
                    v += C[i, k] * pk
                W[i, j] = fac * (v - Cm[i] * mt[j])
            W[i, i] += 1.0
        a = _solve6v(W, Rs)
        b = _solve6v(W, Cm)
        mta = 0.0
        mtb = 0.0
        for i in range(6):
            mta += mt[i] * a[i]
            mtb += mt[i] * b[i]
        dlam = (Rf - mta) / max(mtb + H_iso, 1e-300)
        ds = np.zeros(6, dtype=np.float64)
        for i in range(6):
            ds[i] = -a[i] - b[i] * dlam
        Rn = np.sqrt(Rs[0]**2 + Rs[1]**2 + Rs[2]**2 + Rs[3]**2 + Rs[4]**2 + Rs[5]**2) + abs(Rf)
        step = 1.0
        for _bis in range(20):
            s_try = np.zeros(6, dtype=np.float64)
            for i in range(6):
                s_try[i] = s[i] + step * ds[i]
            lam_try = lam + step * dlam
            m_try = _hill_m_eng(s_try, hill48_phi(s_try, F, G, H, L, M, N), F, G, H, L, M, N)
            Cm_try = np.zeros(6, dtype=np.float64)
            for i in range(6):
                v = 0.0
                for j in range(6):
                    v += C[i, j] * m_try[j]
                Cm_try[i] = v
            Rs_try = np.zeros(6, dtype=np.float64)
            for i in range(6):
                Rs_try[i] = s_try[i] - s_tr[i] + lam_try * Cm_try[i]
            Rf_try = hill48_phi(s_try, F, G, H, L, M, N) - (sigma_y0 + H_iso * (eq_p_old + lam_try))
            Rn_try = np.sqrt(Rs_try[0]**2 + Rs_try[1]**2 + Rs_try[2]**2 + Rs_try[3]**2 + Rs_try[4]**2 + Rs_try[5]**2) + abs(Rf_try)
            if Rn_try < Rn:
                break
            step *= 0.5
        for i in range(6):
            s[i] = s[i] + step * ds[i]
        lam += step * dlam
    eq_p = eq_p_old + lam
    # Exact consistent tangent. With s = s_tr - lam*C*m_eng, m_eng = P*mt,
    # dmt = (M_t*ds - mt*(mt^T*ds))/phi, consistency mt^T*ds = H*dlam:
    # W*ds = A*de - Cm*dlam, W = I + (lam/phi)*(A*P*M_t - Cm*mt^T),
    # dS/de = W^-1*A - W^-1*Cm*(mt^T*W^-1*A)/(mt^T*W^-1*Cm + H).
    phi = hill48_phi(s, F, G, H, L, M, N)
    m = _hill_m_eng(s, phi, F, G, H, L, M, N)
    mt = m.copy()
    mt[3] *= 0.5
    mt[4] *= 0.5
    mt[5] *= 0.5
    Mt = np.zeros((6, 6), dtype=np.float64)
    Mt[0, 0] = G + H
    Mt[0, 1] = -H
    Mt[0, 2] = -G
    Mt[1, 0] = -H
    Mt[1, 1] = F + H
    Mt[1, 2] = -F
    Mt[2, 0] = -G
    Mt[2, 1] = -F
    Mt[2, 2] = F + G
    Mt[3, 3] = 2.0 * N
    Mt[4, 4] = 2.0 * L
    Mt[5, 5] = 2.0 * M
    Pm = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            Pm[i, j] = Mt[i, j]
    Pm[3, 3] *= 2.0
    Pm[4, 4] *= 2.0
    Pm[5, 5] *= 2.0
    APMt = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            v = 0.0
            for k in range(6):
                v += C[i, k] * Pm[k, j]
            APMt[i, j] = v
    Cm = np.zeros(6, dtype=np.float64)
    for i in range(6):
        v = 0.0
        for j in range(6):
            v += C[i, j] * m[j]
        Cm[i] = v
    W = np.zeros((6, 6), dtype=np.float64)
    fac = lam / max(phi, 1e-300)
    for i in range(6):
        for j in range(6):
            W[i, j] = APMt[i, j] - Cm[i] * mt[j]
            W[i, j] *= fac
        W[i, i] += 1.0
    u1 = _solve6(W, C)
    u2 = _solve6v(W, Cm)
    r = np.zeros(6, dtype=np.float64)
    for j in range(6):
        v = 0.0
        for i in range(6):
            v += mt[i] * u1[i, j]
        r[j] = v
    d = H_iso
    for i in range(6):
        d += mt[i] * u2[i]
    C_ep = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        for j in range(6):
            C_ep[i, j] = u1[i, j] - u2[i] * r[j] / max(d, 1e-300)
    ep_new = ep_old.copy()
    for i in range(6):
        ep_new[i] = ep_old[i] + lam * m[i]
    S = s.copy()
    return S, ep_new, eq_p_old + lam, C_ep
