# -*- coding: utf-8 -*-
"""Tests for HGO/GOH anisotropic hyperelasticity (JAX path)."""
import numpy as np
import jax.numpy as jnp

from dispsolver.material.anisotropic_hyperelastic import HolzapfelGasser
from dispsolver.material.mooney_rivlin import MooneyRivlin

MU, K = 0.5, 10.0
A4 = np.array([1.0, 0.0, 0.0])
A6 = np.array([0.0, 1.0, 0.0])
FB = {"mu": MU, "K": K, "k1": 1.0, "k2": 5.0, "kappa": 0.0, "a4": A4, "a6": A6}


def _rng_F(seed=21):
    rng = np.random.default_rng(seed)
    return np.eye(2) + rng.standard_normal((2, 2)) * 0.12


def test_degeneracy_to_isotropic_base():
    hgo = HolzapfelGasser()
    mr = MooneyRivlin()
    mrp = {"C10": MU / 2.0, "C01": 0.0, "K": K}
    p0 = dict(FB, k1=0.0)
    for seed in (21, 22, 23):
        F = _rng_F(seed)
        Sh = np.asarray(hgo.pk2_voigt(jnp.array(F), p0))
        Sm = np.asarray(mr.pk2_voigt(jnp.array(F), mrp))
        np.testing.assert_allclose(Sh, Sm, atol=1e-12)
        Ch = np.asarray(hgo.tangent_voigt(jnp.array(F), p0))
        Cm = np.asarray(mr.tangent_voigt(jnp.array(F), mrp))
        np.testing.assert_allclose(Ch, Cm, atol=1e-12)


def test_fiber_off_in_compression():
    hgo = HolzapfelGasser()
    F = jnp.array([[0.9, 0.0], [0.0, 0.9]])
    S = np.asarray(hgo.pk2_voigt(F, FB))
    mr = MooneyRivlin()
    Sm = np.asarray(mr.pk2_voigt(F, {"C10": MU / 2.0, "C01": 0.0, "K": K}))
    np.testing.assert_allclose(S, Sm, atol=1e-12)


def test_kappa_one_third_isotropic():
    hgo = HolzapfelGasser()
    p = dict(FB, kappa=1.0 / 3.0)
    F = _rng_F(24)
    S = np.asarray(hgo.pk2_voigt(jnp.array(F), p))
    # At kappa=1/3 the fiber directions must drop out: rotated fibers,
    # same F, identical stress.
    th = 0.7
    c, s = np.cos(th), np.sin(th)
    Q3 = np.eye(3)
    Q3[:2, :2] = np.array([[c, -s], [s, c]])
    p2 = dict(p, a4=Q3 @ A4, a6=Q3 @ A6)
    S2 = np.asarray(hgo.pk2_voigt(jnp.array(F), p2))
    np.testing.assert_allclose(S2, S, atol=1e-9 * max(1.0, abs(S).max()))
    # PK2 is invariant under superposed current rotation: S(QF) = S(F).
    # (Q S Q^T is the Cauchy rule, not PK2.)
    Q = Q3[:2, :2]
    Sr = np.asarray(hgo.pk2_voigt(jnp.array(Q @ F), p))
    np.testing.assert_allclose(Sr, S, atol=1e-9 * max(1.0, abs(S).max()))


def test_objectivity_reference_plus_fiber():
    hgo = HolzapfelGasser()
    F = _rng_F(25)
    th = 0.4
    c, s = np.cos(th), np.sin(th)
    Q = np.array([[c, -s], [s, c]])
    Q3 = np.eye(3)
    Q3[:2, :2] = Q
    p_rot = dict(FB, a4=Q3 @ A4, a6=Q3 @ A6)
    S = np.asarray(hgo.pk2_voigt(jnp.array(F), FB))
    # Reference rotation: S_{Qa}(F Q^T) = Q S_a(F) Q^T.
    F2 = F @ Q.T
    Sr = np.asarray(hgo.pk2_voigt(jnp.array(F2), p_rot))
    Sm = np.array([[S[0], S[2]], [S[2], S[1]]])
    Sr_m = Q @ Sm @ Q.T
    np.testing.assert_allclose(Sr, [Sr_m[0, 0], Sr_m[1, 1], Sr_m[0, 1]],
                               atol=1e-9 * max(1.0, abs(S).max()))


def _central_fd(S_fn, F, C, h=1e-7):
    rng = np.random.default_rng(7)
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
    hgo = HolzapfelGasser()
    F = _rng_F(26)
    C = np.asarray(hgo.tangent_voigt(jnp.array(F), FB))
    assert _central_fd(lambda X: np.asarray(hgo.pk2_voigt(jnp.array(X), FB)), F, C) < 1e-6
