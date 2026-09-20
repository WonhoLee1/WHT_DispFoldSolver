# -*- coding: utf-8 -*-
"""Tests for frequency-domain Prony evaluation (exact interconversion)."""
import numpy as np

from dispsolver.material.frequency_viscoelastic import (
    prony_storage_loss_numba,
    prony_sweep_numba,
    shear_complex,
    bulk_complex,
    loss_tangent,
    young_complex,
    reduced_omega,
)

G0 = 100.0
G = np.array([0.3, 0.2])
TAU = np.array([0.5, 2.0])
GINF = 1.0 - G.sum()


def test_zero_high_frequency_limits():
    s0, l0 = prony_storage_loss_numba(1e-12, G0, G, TAU)
    np.testing.assert_allclose(s0, G0 * GINF, rtol=1e-9)
    np.testing.assert_allclose(l0, 0.0, atol=1e-9)
    s8, l8 = prony_storage_loss_numba(1e12, G0, G, TAU)
    np.testing.assert_allclose(s8, G0, rtol=1e-9)
    np.testing.assert_allclose(l8, 0.0, atol=1e-6)


def test_single_term_debye_peak():
    g1 = np.array([0.4])
    t1 = np.array([0.5])
    s, l = prony_storage_loss_numba(1.0 / 0.5, G0, g1, t1)
    np.testing.assert_allclose(l, G0 * 0.4 / 2.0, rtol=1e-12)
    _, lm = prony_storage_loss_numba(0.5 / 0.5, G0, g1, t1)
    _, lp = prony_storage_loss_numba(2.0 / 0.5, G0, g1, t1)
    assert lm < l and lp < l


def test_additivity():
    w = 3.0
    s, l = prony_storage_loss_numba(w, G0, G, TAU)
    s1, l1 = prony_storage_loss_numba(w, G0, G[:1], TAU[:1])
    s2, l2 = prony_storage_loss_numba(w, G0, G[1:], TAU[1:])
    np.testing.assert_allclose(s, G0 * GINF + (s1 - G0 * (1 - G[0])) + (s2 - G0 * (1 - G[1])), rtol=1e-12)
    np.testing.assert_allclose(l, l1 + l2, rtol=1e-12)


def test_wlf_peak_shift():
    wlf = {"C1": 17.44, "C2": 51.6, "T_ref": 20.0}
    aT = reduced_omega(1.0, 0.0, **wlf)
    assert aT > 1.0
    g1 = np.array([0.4])
    t1 = np.array([0.5])
    w_peak = 1.0 / (t1[0] * aT)
    _, lpk = shear_complex(w_peak, G0, g1, t1, T=0.0, wlf=wlf)
    np.testing.assert_allclose(lpk, G0 * 0.4 / 2.0, rtol=1e-9)


def test_bulk_and_young_consistency():
    K0 = 500.0
    k = np.array([0.25])
    t = np.array([1.0])
    ks, kl = bulk_complex(1e-12, K0, k, t)
    np.testing.assert_allclose(ks, K0 * 0.75, rtol=1e-9)
    gs, gl = shear_complex(1e-12, G0, G, TAU)
    E0, E0i = young_complex(gs, gl, ks, kl)
    Eref = 9.0 * (K0 * 0.75) * (G0 * GINF) / (3 * K0 * 0.75 + G0 * GINF)
    np.testing.assert_allclose(E0, Eref, rtol=1e-12)
    np.testing.assert_allclose(E0i, 0.0, atol=1e-9)
    assert loss_tangent(gs, gl) >= 0.0


def test_sweep_shape():
    om = np.logspace(-3, 3, 61)
    out = np.zeros((61, 2))
    prony_sweep_numba(om, G0, G, TAU, out)
    assert np.all(np.diff(out[:, 0]) >= -1e-9)
    assert out[:, 1].max() > 0.0
    i = int(np.argmax(out[:, 1]))
    assert 0 < i < 60
