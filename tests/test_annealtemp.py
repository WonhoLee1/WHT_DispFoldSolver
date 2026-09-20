# -*- coding: utf-8 -*-
"""Anneal-temperature reset tests (Abaqus simaver-c-annealtemp semantics)."""
import numpy as np
from dispsolver.material.anneal import anneal_reset
from dispsolver.material.plastic_hardening import return_mapping_linear_kinematic


def _state():
    alpha = np.array([10.0, -5.0, 3.0, 1.0, -2.0, 0.5])
    ep = np.array([0.02, -0.01, 0.005, 0.004, -0.002, 0.001])
    return alpha, ep, 0.05


def test_below_anneal_noop_exact():
    a, e, q = _state()
    an, en, qn, flag = anneal_reset(a, e, q, 500.0, 600.0)
    assert flag == 0.0
    assert np.allclose(an, a, atol=0.0, rtol=0.0)
    assert np.allclose(en, e, atol=0.0, rtol=0.0)
    assert qn == q


def test_above_anneal_full_reset():
    a, e, q = _state()
    an, en, qn, flag = anneal_reset(a, e, q, 600.0, 600.0)
    assert flag == 1.0
    assert np.allclose(an, 0.0, atol=0.0, rtol=0.0)
    assert np.allclose(en, 0.0, atol=0.0, rtol=0.0)
    assert qn == 0.0


def test_disabled_anneal_never_resets():
    a, e, q = _state()
    an, en, qn, flag = anneal_reset(a, e, q, 1e9, 0.0)
    assert flag == 0.0 and qn == q


def test_reyield_at_virgin_yield_after_reset():
    E, nu, y0, Hi, Ck = 200000.0, 0.3, 250.0, 1000.0, 5000.0
    e_uni = np.zeros(6)
    e_uni[0] = 0.002
    _, _, _, q1, _ = return_mapping_linear_kinematic(
        e_uni, np.zeros(6), np.zeros(6), 0.0, E, nu, y0, Hi, Ck)
    assert q1 > 0.0
    a, e, q = _state()
    an, en, qn, _ = anneal_reset(a, e, q, 700.0, 600.0)
    S, _, _, q2, _ = return_mapping_linear_kinematic(
        e_uni, an, en, qn, E, nu, y0, Hi, Ck)
    Sv, _, _, _, _ = return_mapping_linear_kinematic(
        e_uni, np.zeros(6), np.zeros(6), 0.0, E, nu, y0, Hi, Ck)
    assert np.allclose(S, Sv, atol=1e-12, rtol=0.0)
