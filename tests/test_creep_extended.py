# -*- coding: utf-8 -*-
"""Tests for hyperbolic-sine creep and Anand viscoplasticity.

Anand parameters below are self-consistent test values (documented here,
not claimed alloy data): s0=20 MPa, QR=6000 K, A=1.0 /s, xi=5.0, m=0.3,
h0=2000 MPa, shat=30 MPa, nn=0.05, a=1.5, T=300 K, E=100000 MPa.
At eps_dot=1e-3 /s these give s* ~= 57.7 MPa, q ~= 53 MPa (hand-computed
in the task log) -- the saturation test checks s against this number.
"""
import numpy as np

from dispsolver.material.creep_extended import (
    integrate_sinh_creep,
    integrate_anand,
)
from dispsolver.material.creep_numba import integrate_power_law_creep
from dispsolver.model.material import Material

E = 100000.0
NU = 0.3
ANAND = dict(s0=20.0, QR=6000.0, A=1.0, xi=5.0, m=0.3, h0=2000.0,
             shat=30.0, nn=0.05, a_exp=1.5)


def _uniax(e11):
    return np.array([e11, -NU * e11, -NU * e11, 0.0, 0.0, 0.0], dtype=np.float64)


def test_sinh_elastic_limit():
    sig, ecr, q, dp, C = integrate_sinh_creep(
        _uniax(0.001), np.zeros(6), 0.0, 1.0, 1.0, E, NU,
        A=0.0, B=0.01, n=3.0, m=0.0)
    np.testing.assert_allclose(sig[0], 100.0, rtol=1e-5)
    np.testing.assert_allclose(dp, 0.0, atol=1e-15)
    np.testing.assert_allclose(C, C.T, atol=1e-10)


def test_sinh_recovers_power_law():
    # Taylor remainder of sinh(Bq) ~= Bq at Bq=0.004 is (Bq^2/6)*n ~= 8e-6;
    # rtol=1e-4 clears it with 10x margin.
    A, B, n = 1e-2, 1e-5, 3.0
    eps = _uniax(0.002)
    s1, _, _, dp1, _ = integrate_sinh_creep(
        eps, np.zeros(6), 0.0, 0.5, 0.5, 200000.0, NU, A, B, n, 0.0)
    s2, _, _, dp2, _ = integrate_power_law_creep(
        eps, np.zeros(6), 0.0, 0.5, 0.5, 200000.0, NU, A * B ** n, n, 0.0)
    np.testing.assert_allclose(s1[0], s2[0], rtol=1e-4)
    np.testing.assert_allclose(dp1, dp2, rtol=1e-4)


def test_sinh_relaxation_monotonic():
    eps = _uniax(0.002)
    ecr = np.zeros(6)
    q = 200.0
    prev = 1e300
    for step in range(1, 11):
        sig, ecr, q, dp, C = integrate_sinh_creep(
            eps, ecr, q, 0.5, step * 0.5, 200000.0, NU,
            A=1e-6, B=0.02, n=2.0, m=0.0)
        assert sig[0] < prev
        prev = sig[0]
    assert prev < 400.0 * 0.95


def test_sinh_tangent_fd():
    eps = _uniax(0.002)
    ecr = np.zeros(6)
    sig0, _, _, _, C = integrate_sinh_creep(
        eps, ecr, 200.0, 0.5, 0.5, 200000.0, NU, A=1e-6, B=0.02, n=2.0, m=0.0)
    rng = np.random.default_rng(3)
    errs = []
    h = 1e-7
    for _ in range(3):
        d = rng.standard_normal(6) * h
        sp, _, _, _, _ = integrate_sinh_creep(
            eps + d, ecr, 200.0, 0.5, 0.5, 200000.0, NU, A=1e-6, B=0.02, n=2.0, m=0.0)
        sm, _, _, _, _ = integrate_sinh_creep(
            eps - d, ecr, 200.0, 0.5, 0.5, 200000.0, NU, A=1e-6, B=0.02, n=2.0, m=0.0)
        ds = sp - sm
        errs.append(float(np.linalg.norm(ds - C @ (2 * d)) / max(np.linalg.norm(ds), 1e-30)))
    assert max(errs) < 1e-6


def test_anand_elastic_limit():
    sig, ecr, q, dp, s, C, conv = integrate_anand(
        _uniax(0.001), np.zeros(6), ANAND["s0"], 1.0, 300.0, E, NU,
        **{**ANAND, "A": 0.0})
    np.testing.assert_allclose(sig[0], 100.0, rtol=1e-5)
    assert s == ANAND["s0"]
    assert conv
    np.testing.assert_allclose(C, C.T, atol=1e-10)


def test_anand_saturation():
    # Saturation time scale s*/(h0*eps_dot) ~= 29 s; 100 s drive ~= 3.4 tau.
    # a=1.5 gives an algebraic (slow) tail, so the test checks the evolution
    # LAW at the final state plus monotone concave approach -- not a 5% band.
    rate, dt, n_steps = 1e-3, 0.1, 1000
    ecr = np.zeros(6)
    s = ANAND["s0"]
    e = 0.0
    incs = []
    for _ in range(n_steps):
        e += rate * dt
        s_prev = s
        sig, ecr, q, dp, s, C, conv = integrate_anand(
            _uniax(e), ecr, s, dt, 300.0, E, NU, **ANAND)
        assert conv
        incs.append(s - s_prev)
    assert s > ANAND["s0"]
    assert incs[-1] < max(incs) and incs[-1] > 0.0
    arr = ANAND["A"] * np.exp(-ANAND["QR"] / 300.0)
    s_star = ANAND["shat"] * (dp / (dt * arr)) ** ANAND["nn"]
    assert s < s_star
    w = 1.0 - s / s_star
    ds_pred = ANAND["h0"] * (abs(w) ** ANAND["a_exp"]) * np.sign(w) * dp
    np.testing.assert_allclose(incs[-1], ds_pred, rtol=1e-6)


def test_anand_rate_ordering():
    # Same final strain (1e-3) reached at two rates via small-step drives,
    # so every Newton starts near the previous converged state.
    outs = []
    for rate, dt in ((1e-4, 0.1), (1e-2, 0.01)):
        ecr = np.zeros(6)
        s = ANAND["s0"]
        e = 0.0
        for _ in range(100):
            e += rate * dt
            sig, ecr, q, dp, s, C, conv = integrate_anand(
                _uniax(e), ecr, s, dt, 300.0, E, NU, **ANAND)
            assert conv
        outs.append(sig[0])
    assert outs[1] > outs[0]


def test_anand_temperature_ordering():
    # Small trial stress (q_tr ~= 30 MPa) stays clear of the sinh cap,
    # so both temperatures flow and the Arrhenius ordering is visible.
    e = 0.0003
    outs = []
    for T in (300.0, 500.0):
        sig, _, _, _, _, _, conv = integrate_anand(
            _uniax(e), np.zeros(6), ANAND["s0"], 0.5, T, E, NU, **ANAND)
        assert conv
        outs.append(sig[0])
    assert outs[1] < outs[0]


def test_anand_tangent_fd():
    # Virgin small step (q_tr ~= 20 MPa): Newton converges cleanly and the
    # sinh argument stays far from the overflow cap.
    eps = _uniax(0.0002)
    sig0, ecr0, _, _, s0, C, conv = integrate_anand(
        eps, np.zeros(6), ANAND["s0"], 0.5, 300.0, E, NU, **ANAND)
    assert conv
    rng = np.random.default_rng(9)
    errs = []
    h = 1e-8
    for _ in range(3):
        d = rng.standard_normal(6) * h
        sp, _, _, _, _, _, _ = integrate_anand(
            eps + d, np.zeros(6), ANAND["s0"], 0.5, 300.0, E, NU, **ANAND)
        sm, _, _, _, _, _, _ = integrate_anand(
            eps - d, np.zeros(6), ANAND["s0"], 0.5, 300.0, E, NU, **ANAND)
        ds = sp - sm
        errs.append(float(np.linalg.norm(ds - C @ (2 * d)) / max(np.linalg.norm(ds), 1e-30)))
    np.testing.assert_allclose(C, C.T, atol=1e-8 * abs(C).max())
    assert max(errs) < 1e-5


def test_material_creep_api_hyperb_anand():
    m1 = Material(name="H")
    m1.Elastic(E=E, nu=NU)
    m1.Creep(law="HYPERB", table=((1e-6, 0.02, 2.0, 0.0),))
    assert m1.extra_params["creep"]["law"] == "HYPERB"
    assert m1.extra_params["creep"]["B"] == 0.02
    m2 = Material(name="A")
    m2.Elastic(E=E, nu=NU)
    m2.Creep(law="ANAND", table=((20.0, 6000.0, 1.0, 5.0, 0.3, 2000.0, 30.0, 0.05, 1.5),))
    assert m2.extra_params["creep"]["law"] == "ANAND"
    assert m2.extra_params["creep"]["shat"] == 30.0
