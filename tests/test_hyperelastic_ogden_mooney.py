# -*- coding: utf-8 -*-
"""Ogden + Mooney-Rivlin tests: limits, degeneracy, monotonicity, FD tangent."""
import numpy as np
import jax.numpy as jnp
from dispsolver.material.mooney_rivlin import MooneyRivlin
from dispsolver.material.ogden import ogden_pk2_tangent

MU, K = 0.5, 10.0
MR_PARAMS = {"C10": 0.15, "C01": 0.10, "K": K}  # mu = 0.5
OG_MUS = np.array([1.0])
OG_ALPHAS = np.array([2.0])  # mu0 = 1.0*2/2 = 1.0


def _iso_C(mu, K):
    lam = K - 2.0 * mu / 3.0
    return np.array([[lam + 2 * mu, lam, 0.0],
                     [lam, lam + 2 * mu, 0.0],
                     [0.0, 0.0, mu]])


def _F(eps=1e-6):
    return np.array([[1.0 + eps, 0.3 * eps], [0.0, 1.0 - 0.5 * eps]])


def test_mr_small_strain_limit():
    mr = MooneyRivlin()
    F = jnp.array(_F())
    S = np.asarray(mr.pk2_voigt(F, MR_PARAMS))
    E_v = np.array([0.5 * (F[0, 0]**2 - 1.0), 0.5 * (F[1, 1]**2 - 1.0),
                    F[0, 0] * F[0, 1]])
    Cref = _iso_C(0.5, K)
    assert np.allclose(S, Cref @ E_v, atol=1e-9 * max(1.0, abs(S).max()))
    C0 = np.asarray(mr.tangent_voigt(jnp.eye(2), MR_PARAMS))
    assert np.allclose(C0, Cref, atol=1e-9 * abs(Cref).max())


def test_ogden_small_strain_limit():
    F = _F()
    S, _ = ogden_pk2_tangent(F[0, 0], F[0, 1], F[1, 0], F[1, 1],
                             OG_MUS, OG_ALPHAS, K)
    E_v = np.array([0.5 * (F[0, 0]**2 + F[1, 0]**2 - 1.0),
                    0.5 * (F[0, 1]**2 + F[1, 1]**2 - 1.0),
                    F[0, 0] * F[0, 1] + F[1, 0] * F[1, 1]])
    Cref = _iso_C(1.0, K)
    assert np.allclose(S, Cref @ E_v, atol=1e-9 * max(1.0, abs(S).max()))
    _, C0 = ogden_pk2_tangent(1.0, 0.0, 0.0, 1.0, OG_MUS, OG_ALPHAS, K)
    assert np.allclose(C0, Cref, atol=1e-9 * abs(Cref).max())


def test_ogden_n1_a2_matches_mooney_c01_zero():
    mr = MooneyRivlin()
    mrp = {"C10": 0.5, "C01": 0.0, "K": K}
    mus = np.array([1.0])
    als = np.array([2.0])
    rng = np.random.default_rng(1)
    for _ in range(4):
        F = np.eye(2) + rng.standard_normal((2, 2)) * 0.15
        S_o, C_o = ogden_pk2_tangent(F[0, 0], F[0, 1], F[1, 0], F[1, 1],
                                     mus, als, K)
        S_m = np.asarray(mr.pk2_voigt(jnp.array(F), mrp))
        C_m = np.asarray(mr.tangent_voigt(jnp.array(F), mrp))
        assert np.allclose(S_o, S_m, atol=1e-9 * max(1.0, abs(S_m).max()))
        assert np.allclose(C_o, C_m, atol=1e-9 * abs(C_m).max())


def test_identity_stress_free():
    mr = MooneyRivlin()
    S = np.asarray(mr.pk2_voigt(jnp.eye(2), MR_PARAMS))
    assert np.allclose(S, 0.0, atol=1e-15)
    S_o, _ = ogden_pk2_tangent(1.0, 0.0, 0.0, 1.0, OG_MUS, OG_ALPHAS, K)
    assert np.allclose(S_o, 0.0, atol=1e-12)


def test_uniaxial_monotonic():
    mr = MooneyRivlin()
    prev_m, prev_o = -1e300, -1e300
    for s in (1.0, 1.1, 1.2, 1.3, 1.5):
        F = jnp.array([[s, 0.0], [0.0, 1.0]])
        sm = float(np.asarray(mr.pk2_voigt(F, MR_PARAMS))[0])
        Fn = np.array([[s, 0.0], [0.0, 1.0]])
        so, _ = ogden_pk2_tangent(Fn[0, 0], Fn[0, 1], Fn[1, 0], Fn[1, 1],
                                  OG_MUS, OG_ALPHAS, K)
        assert sm > prev_m and float(so[0]) > prev_o
        prev_m, prev_o = sm, float(so[0])


def test_volumetric_compression_negative():
    mr = MooneyRivlin()
    F = jnp.array([[0.9, 0.0], [0.0, 0.9]])
    S = np.asarray(mr.pk2_voigt(F, MR_PARAMS))
    assert S[0] < 0.0 and S[1] < 0.0
    Fn = np.array([[0.9, 0.0], [0.0, 0.9]])
    So, _ = ogden_pk2_tangent(Fn[0, 0], Fn[0, 1], Fn[1, 0], Fn[1, 1],
                              OG_MUS, OG_ALPHAS, K)
    assert So[0] < 0.0 and So[1] < 0.0


def _dir_check(S_fn, F, h=1e-6):
    rng = np.random.default_rng(7)
    errs = []
    for _ in range(3):
        dH = rng.standard_normal((2, 2)) * h
        Fp = F + dH
        Fm = F - dH
        Ep = 0.5 * (Fp.T @ Fp - np.eye(2))
        Em = 0.5 * (Fm.T @ Fm - np.eye(2))
        dE = np.array([Ep[0, 0] - Em[0, 0], Ep[1, 1] - Em[1, 1],
                       2.0 * (Ep[0, 1] - Em[0, 1])])
        dS = S_fn(Fp) - S_fn(Fm)
        errs.append(np.linalg.norm(dS - C_AT[F_KEY] @ dE) / max(np.linalg.norm(dS), 1e-30))
    return max(errs)


C_AT = {}
F_KEY = None


def test_tangent_directional_fd():
    mr = MooneyRivlin()
    states = [np.array([[1.1, 0.05], [0.0, 0.95]]),
              np.array([[1.1, 0.0], [0.0, 1.1]]),
              np.array([[1.2, 0.1], [-0.05, 0.9]])]
    global F_KEY
    for F in states:
        Cm = np.asarray(mr.tangent_voigt(jnp.array(F), MR_PARAMS))
        F_KEY = id(F)
        C_AT[F_KEY] = Cm
        assert _dir_check(lambda X: np.asarray(mr.pk2_voigt(jnp.array(X), MR_PARAMS)), F) < 1e-6
        So_fn = lambda X: ogden_pk2_tangent(X[0, 0], X[0, 1], X[1, 0], X[1, 1],
                                            OG_MUS, OG_ALPHAS, K)[0]
        Co = ogden_pk2_tangent(F[0, 0], F[0, 1], F[1, 0], F[1, 1],
                               OG_MUS, OG_ALPHAS, K)[1]
        F_KEY = id(F) + 1
        C_AT[F_KEY] = Co
        assert _dir_check(So_fn, F) < 1e-6
