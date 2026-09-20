# -*- coding: utf-8 -*-
"""Tests for Kilian lock-up on Mooney-Rivlin base (JAX)."""
import numpy as np
import jax.numpy as jnp

from dispsolver.material.van_der_waals import VanDerWaals
from dispsolver.material.mooney_rivlin import MooneyRivlin

C10, C01, K = 0.15, 0.10, 10.0
MU = 2.0 * (C10 + C01)
VDW = {"C10": C10, "C01": C01, "K": K, "k_lock": 0.5, "lam_m": 2.0}
MRP = {"C10": C10, "C01": C01, "K": K}


def _rng_F(seed=31):
    rng = np.random.default_rng(seed)
    return np.eye(2) + rng.standard_normal((2, 2)) * 0.10


def test_identity_stress_free():
    v = VanDerWaals()
    S = np.asarray(v.pk2_voigt(jnp.eye(2), VDW))
    np.testing.assert_allclose(S, 0.0, atol=1e-12)


def test_k_zero_reduces_to_mooney():
    v = VanDerWaals()
    mr = MooneyRivlin()
    p0 = dict(VDW, k_lock=0.0)
    for seed in (31, 32):
        F = _rng_F(seed)
        S = np.asarray(v.pk2_voigt(jnp.array(F), p0))
        Sm = np.asarray(mr.pk2_voigt(jnp.array(F), MRP))
        np.testing.assert_allclose(S, Sm, atol=1e-12)
        C = np.asarray(v.tangent_voigt(jnp.array(F), p0))
        Cm = np.asarray(mr.tangent_voigt(jnp.array(F), MRP))
        np.testing.assert_allclose(C, Cm, atol=1e-12)


def test_small_strain_limit():
    v = VanDerWaals()
    F = np.array([[1.000001, 0.0000003], [0.0, 0.9999995]])
    S = np.asarray(v.pk2_voigt(jnp.array(F), VDW))
    E_v = np.array([0.5 * (1.000001 ** 2 - 1.0), 0.5 * (0.9999995 ** 2 - 1.0),
                    1.000001 * 0.0000003])
    lam = K - 2.0 * MU / 3.0
    Cref = np.array([[lam + 2 * MU, lam, 0.0],
                     [lam, lam + 2 * MU, 0.0],
                     [0.0, 0.0, MU]])
    np.testing.assert_allclose(S, Cref @ E_v, rtol=1e-4)
    C0 = np.asarray(v.tangent_voigt(jnp.eye(2), VDW))
    np.testing.assert_allclose(C0, Cref, rtol=1e-3)


def test_uniaxial_monotonic():
    v = VanDerWaals()
    prev = -1e300
    for s in (1.0, 1.05, 1.1, 1.2, 1.3):
        S = np.asarray(v.pk2_voigt(jnp.array([[s, 0.0], [0.0, 1.0]]), VDW))
        assert float(S[0]) > prev
        prev = float(S[0])


def test_locking_steeper_than_mooney():
    v = VanDerWaals()
    mr = MooneyRivlin()
    plock = dict(VDW, lam_m=0.12)
    F = np.array([[1.3, 0.0], [0.0, 1.0]])
    Sv = float(np.asarray(v.pk2_voigt(jnp.array(F), plock))[0])
    Sm = float(np.asarray(mr.pk2_voigt(jnp.array(F), MRP))[0])
    assert Sv > Sm * 2.0


def _central_fd(S_fn, F, C, h=1e-7):
    rng = np.random.default_rng(9)
    errs = []
    for _ in range(3):
        dH = rng.standard_normal((2, 2)) * h
        Fp, Fm = F + dH, F - dH
        Ep = 0.5 * (Fp.T @ Fp - np.eye(2))
        Em = 0.5 * (Fm.T @ Fm - np.eye(2))
        dE = np.array([Ep[0, 0] - Em[0, 0], Ep[1, 1] - Em[1, 1],
                       2.0 * (Ep[0, 1] - Em[0, 1])])
        dS = S_fn(Fp) - S_fn(Fm)
        errs.append(float(np.linalg.norm(dS - C @ dE) / max(np.linalg.norm(dS), 1e-30)))
    return max(errs)


def test_tangent_fd():
    v = VanDerWaals()
    F = _rng_F(31)
    C = np.asarray(v.tangent_voigt(jnp.array(F), VDW))
    assert _central_fd(lambda X: np.asarray(v.pk2_voigt(jnp.array(X), VDW)), F, C) < 1e-6
