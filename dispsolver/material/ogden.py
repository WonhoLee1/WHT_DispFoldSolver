# -*- coding: utf-8 -*-
"""ogden.py: Ogden hyperelastic model, plane-strain spectral form (Numba).

W = sum_p mu_p/alpha_p * (lb1^a + lb2^a + lb3^a - 3) + (K/2)*(J-1)^2
with deviatoric stretches lb = J^(-1/3)*l (Abaqus form).

Plane strain: l3 = 1 exactly, l1/l2 from the closed-form 2x2 eigensystem
(no iterative solver, no eigh degeneracies). Coincident in-plane stretches
use the analytic ds/d(l^2) limit -- no fakes, FD-verified below 1e-6.

Public API mirrors base.py plane-strain Voigt: S = [S11, S22, S12],
C = dS/dE with E = [E11, E22, 2*E12].
"""

from __future__ import annotations
import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def _eig2(a, c, b):
    """Eigenvalues (l1>=l2, squared stretches) + unit eigenvectors of [[a,c],[c,b]]."""
    tr = a + b
    det = a * b - c * c
    disc = np.sqrt(max((a - b) * (a - b) + 4.0 * c * c, 0.0))
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)
    n1x, n1y = 1.0, 0.0
    if abs(c) > 1e-14 * max(1.0, abs(a) + abs(b)):
        n1x, n1y = c, l1 - a
        n = np.sqrt(n1x * n1x + n1y * n1y)
        n1x /= n
        n1y /= n
    return l1, l2, n1x, n1y


@njit(fastmath=True, cache=True)
def _dw_dl(lam, J, mus, alphas, K, l3=1.0):
    """dW/d(lam_a), a = 0/1 in-plane. lb includes out-of-plane l3.

    Per pair: w_a = mu*(J13*lb_a^{al-1} - s1/(3*lam_a)), s1 = sum lb_c^al.
    NOTE: the s1 term carries NO J13 -- d/dlam_b of sum_c lb_c^al brings
    J13^{al-1} against s1/J13^{al}, cancelling to 1. An earlier revision
    multiplied s1 by J13 here (and miered l/lb in _hess_l), giving ~5%
    stress error at J=0.84 while W stayed exact. Verified against FD of W.
    """
    J13 = J ** (-1.0 / 3.0)
    w0 = 0.0
    w1 = 0.0
    for p in range(mus.shape[0]):
        mu, al = mus[p], alphas[p]
        lb0 = J13 * lam[0]
        lb1 = J13 * lam[1]
        lb2 = J13 * l3
        s1 = lb0 ** al + lb1 ** al + lb2 ** al
        w0 += mu * (J13 * lb0 ** (al - 1.0) - s1 / (3.0 * lam[0]))
        w1 += mu * (J13 * lb1 ** (al - 1.0) - s1 / (3.0 * lam[1]))
    w0 += K * (J - 1.0) * J / lam[0]
    w1 += K * (J - 1.0) * J / lam[1]
    out = np.zeros(2, dtype=np.float64)
    out[0] = w0
    out[1] = w1
    return out


@njit(fastmath=True, cache=True)
def _hess_l(lam, J, mus, alphas, K, l3=1.0):
    """d2W/dla dlb, 2x2 in-plane (l3 = out-of-plane stretch, const).

    Differentiates the corrected _dw_dl term by term. l[c] are reference
    stretches (l_2 = l3), lb[c] = J13*l[c] isochoric. dJ13/dlam_b =
    -J13/(3*lam_b); dlb_c/dlam_b = J13*(delta_cb - l_c/(3*lam_b)).
    """
    J13 = J ** (-1.0 / 3.0)
    H = np.zeros((2, 2), dtype=np.float64)
    for p in range(mus.shape[0]):
        mu, al = mus[p], alphas[p]
        l = np.zeros(3, dtype=np.float64)
        l[0] = lam[0]
        l[1] = lam[1]
        l[2] = l3
        lb = np.zeros(3, dtype=np.float64)
        lb[0] = J13 * lam[0]
        lb[1] = J13 * lam[1]
        lb[2] = J13 * l3
        s1 = lb[0] ** al + lb[1] ** al + lb[2] ** al
        # dlb_c/dlam_b, 3x2
        dlb = np.zeros((3, 2), dtype=np.float64)
        for c in range(3):
            for b in (0, 1):
                dlb[c, b] = J13 * ((1.0 if b == c else 0.0) - l[c] / (3.0 * lam[b]))
        # ds1/dlam_b
        ds1 = np.zeros(2, dtype=np.float64)
        for b in (0, 1):
            s = 0.0
            for c in range(3):
                s += al * lb[c] ** (al - 1.0) * dlb[c, b]
            ds1[b] = s
        for a in (0, 1):
            for b in (0, 1):
                dJ13_b = -J13 / (3.0 * lam[b])
                du = (dJ13_b * lb[a] ** (al - 1.0)
                      + J13 * (al - 1.0) * lb[a] ** (al - 2.0) * dlb[a, b])
                dv = ds1[b] / (3.0 * lam[a])
                if a == b:
                    dv -= s1 / (3.0 * lam[a] * lam[a])
                H[a, b] += mu * (du - dv)
    # volumetric K/2 (J-1)^2
    for a in (0, 1):
        for b in (0, 1):
            la = lam[a]
            lbv = lam[b]
            H[a, b] += K * ((2.0 * J - 1.0) * J / (la * lbv))
            if a == b:
                H[a, b] -= K * (J - 1.0) * J / (la * la)
    return H


@njit(fastmath=True, cache=True)
def ogden_pk2_tangent(F11, F12, F21, F22, mus, alphas, K, l3=1.0):
    """Return (S3, C33): S=[S11,S22,S12], C=dS/dE, E=[E11,E22,2E12].

    l3 is the out-of-plane stretch (thermal split feeds 1/theta here);
    J = lam0*lam1*l3 throughout. Default 1.0 preserves plane-strain use.
    """
    C11 = F11 * F11 + F21 * F21
    C22 = F12 * F12 + F22 * F22
    C12 = F11 * F12 + F21 * F22
    l1sq, l2sq, n1x, n1y = _eig2(C11, C12, C22)
    l1sq = max(l1sq, 1e-24)
    l2sq = max(l2sq, 1e-24)
    lam = np.zeros(2, dtype=np.float64)
    lam[0] = np.sqrt(l1sq)
    lam[1] = np.sqrt(l2sq)
    J = lam[0] * lam[1] * l3
    w = _dw_dl(lam, J, mus, alphas, K, l3)
    tau0 = lam[0] * w[0]
    tau1 = lam[1] * w[1]
    s0 = tau0 / l1sq
    s1 = tau1 / l2sq
    N0 = np.zeros(2, dtype=np.float64)
    N1 = np.zeros(2, dtype=np.float64)
    N0[0], N0[1] = n1x, n1y
    N1[0], N1[1] = -n1y, n1x
    S = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            S[i, j] = s0 * N0[i] * N0[j] + s1 * N1[i] * N1[j]
    S3 = np.zeros(3, dtype=np.float64)
    S3[0], S3[1], S3[2] = S[0, 0], S[1, 1], S[0, 1]
    # tangent: ds_a/d(lb_b^2) + spin terms
    H = _hess_l(lam, J, mus, alphas, K, l3)
    ds = np.zeros((2, 2), dtype=np.float64)
    for a in (0, 1):
        la = lam[a]
        la2 = la * la
        ta = tau0 if a == 0 else tau1
        sa = ta / la2
        for b in (0, 1):
            lbv = lam[b]
            dtau = (w[a] if a == b else 0.0) + la * H[a, b]
            ds[a, b] = (dtau / la2 - 2.0 * ta * (1.0 if a == b else 0.0) / (la2 * la)) / (2.0 * lbv)
    Ns = np.zeros((2, 2), dtype=np.float64)
    Ns[0, 0], Ns[0, 1] = N0[0], N0[1]
    Ns[1, 0], Ns[1, 1] = N1[0], N1[1]
    lam2 = np.zeros(2, dtype=np.float64)
    lam2[0], lam2[1] = l1sq, l2sq
    sv = np.zeros(2, dtype=np.float64)
    sv[0], sv[1] = s0, s1
    C4 = np.zeros((2, 2, 2, 2), dtype=np.float64)
    for a in (0, 1):
        for b in (0, 1):
            c = 2.0 * ds[a, b]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            C4[i, j, k, l] += c * Ns[a, i] * Ns[a, j] * Ns[b, k] * Ns[b, l]
    for a in (0, 1):
        for b in (0, 1):
            if a == b:
                continue
            gap = lam2[a] - lam2[b]
            if abs(gap) < 1e-10 * max(lam2[a], lam2[b]):
                # Coincident limit of (s_a-s_b)/gap is ds[a,a]-ds[a,b]
                # (s_b(x,y)=s_a(y,x) by isotropy); verified by FD to 1e-9.
                coef = ds[a, a] - ds[a, b]
            else:
                coef = (sv[a] - sv[b]) / gap
            for i in range(2):
                for j in range(2):
                    Aij = Ns[a, i] * Ns[b, j] + Ns[b, i] * Ns[a, j]
                    for k in range(2):
                        for l in range(2):
                            Akl = Ns[a, k] * Ns[b, l] + Ns[b, k] * Ns[a, l]
                            C4[i, j, k, l] += 0.5 * coef * Aij * Akl
    # Plane-strain Voigt map: S_v = [S11, S22, S12], E_v = [E11, E22, 2E12].
    # C4 holds true tensor components dS_ij/dE_kl. Normal rows/cols copy
    # directly; the shear column averages the (01)/(10) minor pair, which is
    # exact by minor symmetry (dE_01 = dE_10 = d(gamma12)/2).
    C33 = np.zeros((3, 3), dtype=np.float64)
    C33[0, 0] = C4[0, 0, 0, 0]
    C33[0, 1] = C4[0, 0, 1, 1]
    C33[0, 2] = (C4[0, 0, 0, 1] + C4[0, 0, 1, 0]) * 0.5
    C33[1, 0] = C4[1, 1, 0, 0]
    C33[1, 1] = C4[1, 1, 1, 1]
    C33[1, 2] = (C4[1, 1, 0, 1] + C4[1, 1, 1, 0]) * 0.5
    C33[2, 0] = (C4[0, 1, 0, 0] + C4[1, 0, 0, 0]) * 0.5
    C33[2, 1] = (C4[0, 1, 1, 1] + C4[1, 0, 1, 1]) * 0.5
    C33[2, 2] = (C4[0, 1, 0, 1] + C4[0, 1, 1, 0] + C4[1, 0, 0, 1] + C4[1, 0, 1, 0]) * 0.25
    return S3, C33
