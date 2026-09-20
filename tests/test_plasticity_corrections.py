# -*- coding: utf-8 -*-
"""Neuber/Glinka correction tests (closed-form roots computed in-test)."""
import numpy as np
from dispsolver.material.neuber_glinka import (
    neuber_correct, glinka_correct, glinka_sed)


E, Y0, K, N = 200000.0, 250.0, 800.0, 0.2


def test_elastic_identity():
    assert neuber_correct(100.0, E, Y0, K, N) == 100.0
    assert glinka_correct(100.0, E, Y0, K, N) == 100.0


def test_neuber_hyperbola_holds():
    s_el = 600.0
    s = neuber_correct(s_el, E, Y0, K, N)
    assert Y0 < s < s_el
    e = s / E + (s / K) ** (1.0 / N)
    assert abs(s * e - s_el * s_el / E) < 1e-9 * (s_el * s_el / E)


def test_glinka_energy_equivalence():
    s_el = 600.0
    s = glinka_correct(s_el, E, Y0, K, N)
    assert Y0 < s < s_el
    assert abs(glinka_sed(s, E, K, N) - s_el * s_el / (2 * E)) \
        < 1e-9 * (s_el * s_el / (2 * E))


def test_monotonic_response():
    prev_n, prev_g = Y0, Y0
    for s_el in (300.0, 500.0, 700.0, 900.0):
        sn = neuber_correct(s_el, E, Y0, K, N)
        sg = glinka_correct(s_el, E, Y0, K, N)
        assert sn >= prev_n and sg >= prev_g
        prev_n, prev_g = sn, sg
