# -*- coding: utf-8 -*-
"""damage_initiation.py: ductile/JC initiation + energy evolution kernels (Numba).

All functions are pure @njit (fastmath, cache) and independently unit
tested. The 3D dispatcher (MAT_J2_DUCTILE_DAMAGE) calls the same formulas
inline; these helpers are the specification both must match.

- Initiation: omega_D += d_eps_p / eD(eta). eD from constant / 2-point
  triaxiality table (linear + clamp) / Johnson-Cook triaxiality part
  D1 + D2*exp(D3*eta). JC rate/temperature multipliers are NOT here:
  the dispatcher carries neither strain rate nor temperature, so D4/D5
  must be 0 (dispatcher raises error_flag otherwise).
- Evolution (energy-based, mesh-objective via characteristic length L):
  linear: D = u_bar/u_f; exponential: D = 1 - exp(-3*u_bar/u_f),
  with u_f = 2*G_f/sigma_f and u_bar = L * eps_p_post_init.
"""

from __future__ import annotations

import numpy as np
import numba


@numba.njit(fastmath=True, cache=True)
def fracture_strain_const(eD: float) -> float:
    """Constant fracture strain (triaxiality-independent)."""
    return eD


@numba.njit(fastmath=True, cache=True)
def fracture_strain_triax_table(eta: float, eta0: float, e0: float,
                                eta1: float, e1: float) -> float:
    """2-point triaxiality table, linear interpolation with clamp."""
    if eta <= eta0:
        return e0
    if eta >= eta1:
        return e1
    t = (eta - eta0) / max(eta1 - eta0, 1e-300)
    return e0 + t * (e1 - e0)


@numba.njit(fastmath=True, cache=True)
def fracture_strain_jc(eta: float, D1: float, D2: float, D3: float) -> float:
    """Johnson-Cook quasi-static isothermal fracture strain."""
    return D1 + D2 * np.exp(D3 * eta)


@numba.njit(fastmath=True, cache=True)
def failure_displacement(Gf: float, sig_f: float) -> float:
    """Characteristic failure displacement u_f = 2*G_f/sigma_f."""
    return 2.0 * Gf / max(sig_f, 1e-300)


@numba.njit(fastmath=True, cache=True)
def evolution_linear(u_bar: float, u_f: float) -> float:
    """Linear damage evolution D = u_bar/u_f (caller clamps to [0, 1])."""
    return u_bar / max(u_f, 1e-300)


@numba.njit(fastmath=True, cache=True)
def evolution_exponential(u_bar: float, u_f: float) -> float:
    """Exponential damage evolution D = 1 - exp(-3*u_bar/u_f)."""
    return 1.0 - np.exp(-3.0 * u_bar / max(u_f, 1e-300))
