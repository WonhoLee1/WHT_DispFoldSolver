# -*- coding: utf-8 -*-
"""neuber_glinka.py: Neuber/Glinka notch corrections as pure postprocessing.

No solver coupling: given a linear-elastic notch stress, return the
elastoplastic corrected stress/energy on the material hardening curve.
"""

from __future__ import annotations
import numpy as np
from numba import njit


@njit(fastmath=True, cache=True)
def _ramberg_osgood_eps(sig, E, K_pl, n_pl):
    e = sig / E
    if sig > 0.0 and K_pl > 0.0 and n_pl > 0.0:
        e += (sig / K_pl) ** (1.0 / n_pl)
    return e


@njit(fastmath=True, cache=True)
def neuber_correct(sig_el, E, sigma_y0, K_pl, n_pl):
    """Neuber hyperbola sig*eps = sig_el^2/E intersected with hardening curve.

    Returns corrected stress. Below yield (sig_el <= sigma_y0) returns sig_el.
    Solved by bisection to <1e-12 on the residual.
    """
    if sig_el <= sigma_y0:
        return sig_el
    target = sig_el * sig_el / E
    lo, hi = sigma_y0, sig_el
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * _ramberg_osgood_eps(mid, E, K_pl, n_pl) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


@njit(fastmath=True, cache=True)
def glinka_sed(sig, E, K_pl, n_pl):
    """Strain energy density up to sig on the hardening curve (trapezoid, fine)."""
    N = 2000
    h = sig / N
    s = 0.5 * (_ramberg_osgood_eps(0.0, E, K_pl, n_pl)
               + _ramberg_osgood_eps(sig, E, K_pl, n_pl))
    for i in range(1, N):
        s += _ramberg_osgood_eps(i * h, E, K_pl, n_pl)
    return s * h


@njit(fastmath=True, cache=True)
def glinka_correct(sig_el, E, sigma_y0, K_pl, n_pl):
    """Glinka: corrected stress with SED equal to elastic SED sig_el^2/2E."""
    if sig_el <= sigma_y0:
        return sig_el
    target = sig_el * sig_el / (2.0 * E)
    lo, hi = sigma_y0, sig_el
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if glinka_sed(mid, E, K_pl, n_pl) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    return 0.5 * (lo + hi)
