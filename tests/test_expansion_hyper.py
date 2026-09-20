# -*- coding: utf-8 -*-
"""Tests for the thermal-split hyperelastic adapter (Ogden/Mooney/AB)."""
import numpy as np
import jax.numpy as jnp

from dispsolver.material.expansion_hyper import (
    thermal_stretch,
    ogden_thermal,
    mooney_thermal,
    arruda_boyce_thermal,
)
from dispsolver.material.ogden import ogden_pk2_tangent
from dispsolver.material.mooney_rivlin import MooneyRivlin

K = 10.0
OG_MUS = np.array([1.0])
OG_ALPHAS = np.array([2.0])
MR_PARAMS = {"C10": 0.5, "C01": 0.0, "K": K}
AB_PARAMS = {"mu": 1.0, "lambda_m": 7.0, "K": K}
THETA = thermal_stretch(1e-4, 100.0)


def _rng_F(seed=11):
    rng = np.random.default_rng(seed)
    return np.eye(2) + rng.standard_normal((2, 2)) * 0.12


def test_theta_limit_recovers_core():
    F = _rng_F()
    So, Co = ogden_thermal(F, 1.0, OG_MUS, OG_ALPHAS, K)
    So0, Co0 = ogden_pk2_tangent(F[0, 0], F[0, 1], F[1, 0], F[1, 1],
                                 OG_MUS, OG_ALPHAS, K)
    assert np.allclose(So, So0, atol=1e-12)
    assert np.allclose(Co, Co0, atol=1e-12)
    mr = MooneyRivlin()
    Sm, Cm = mooney_thermal(F, 1.0, MR_PARAMS)
    Sm0 = np.asarray(mr.pk2_voigt(jnp.array(F), MR_PARAMS))
    assert np.allclose(Sm, Sm0, atol=1e-12)


def test_ogden_mooney_cross_model_with_split():
    for seed in (11, 12, 13):
        F = _rng_F(seed)
        So, _ = ogden_thermal(F, THETA, OG_MUS, OG_ALPHAS, K)
        Sm, _ = mooney_thermal(F, THETA, MR_PARAMS)
        assert np.allclose(So, Sm, atol=1e-9 * max(1.0, abs(Sm).max()))


def test_spherical_symmetry():
    F = THETA * np.eye(2)
    for S, _ in (ogden_thermal(F, THETA, OG_MUS, OG_ALPHAS, K),
                 mooney_thermal(F, THETA, MR_PARAMS),
                 arruda_boyce_thermal(F, THETA, AB_PARAMS)):
        assert abs(S[0] - S[1]) < 1e-12 * max(1.0, abs(S).max())
        assert abs(S[2]) < 1e-12 * max(1.0, abs(S).max())


def _central_fd(S_fn, F, C, h=1e-7):
    rng = np.random.default_rng(5)
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


def test_adapter_tangent_fd():
    F = _rng_F(14)
    So, Co = ogden_thermal(F, THETA, OG_MUS, OG_ALPHAS, K)
    assert _central_fd(lambda X: ogden_thermal(X, THETA, OG_MUS, OG_ALPHAS, K)[0], F, Co) < 1e-6
    Sm, Cm = mooney_thermal(F, THETA, MR_PARAMS)
    assert _central_fd(lambda X: mooney_thermal(X, THETA, MR_PARAMS)[0], F, Cm) < 1e-6
    Sa, Ca = arruda_boyce_thermal(F, THETA, AB_PARAMS)
    assert _central_fd(lambda X: arruda_boyce_thermal(X, THETA, AB_PARAMS)[0], F, Ca) < 1e-6
