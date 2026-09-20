# -*- coding: utf-8 -*-
"""frequency_viscoelastic.py: frequency-domain evaluation of Prony viscoelasticity.

Exact interconversion (no new material parameters). For G(t) = G0*(g_inf +
sum g_i*exp(-t/tau_i)) with g_inf = 1 - sum(g_i):

    G'(w)  = G0*(g_inf + sum g_i*(w*t)^2/(1+(w*t)^2))
    G''(w) = G0*sum g_i*(w*t)/(1+(w*t)^2)

Bulk part identical with (k_i, K0). WLF time-temperature superposition enters
as reduced frequency w_r = w*aT(T); aT reuses viscoelastic.wlf_shift
(aT = 10^(-C1*dT/(C2+dT)), guarded).

Conventions match viscoelastic.py (g_i/tau_i arrays, sum(g_i) < 1) and the
((g, k, tau), ...) prony table in model/material.py.
"""

from __future__ import annotations

import numpy as np
import numba

from dispsolver.material.viscoelastic import wlf_shift


@numba.njit(fastmath=True, cache=True)
def prony_storage_loss_numba(
    omega: float,
    G0: float,
    g: np.ndarray,
    tau: np.ndarray,
) -> tuple[float, float]:
    """Storage/loss moduli at angular frequency omega (Debye sum)."""
    gs = 1.0
    gl = 0.0
    for i in range(g.shape[0]):
        gs -= g[i]
        wt = omega * tau[i]
        den = 1.0 + wt * wt
        gs += g[i] * wt * wt / den
        gl += g[i] * wt / den
    return G0 * gs, G0 * gl


@numba.njit(fastmath=True, cache=True)
def prony_sweep_numba(
    omegas: np.ndarray,
    G0: float,
    g: np.ndarray,
    tau: np.ndarray,
    out: np.ndarray,
) -> None:
    """Fill out[:, 0] = G', out[:, 1] = G'' over a frequency sweep."""
    for k in range(omegas.shape[0]):
        s, l = prony_storage_loss_numba(omegas[k], G0, g, tau)
        out[k, 0] = s
        out[k, 1] = l


def reduced_omega(omega: float, T: float, C1: float, C2: float, T_ref: float) -> float:
    """WLF-shifted angular frequency w*aT(T)."""
    return float(omega) * float(wlf_shift(float(T), float(C1), float(C2), float(T_ref)))


def shear_complex(omega: float, G0: float, g, tau,
                  T: float | None = None, wlf: dict | None = None):
    """(G', G'') with optional WLF shift at temperature T."""
    w = float(omega)
    if T is not None and wlf is not None:
        w = reduced_omega(w, T, wlf["C1"], wlf["C2"], wlf["T_ref"])
    ga = np.asarray(g, dtype=np.float64)
    ta = np.asarray(tau, dtype=np.float64)
    return prony_storage_loss_numba(w, float(G0), ga, ta)


def bulk_complex(omega: float, K0: float, k, tau,
                 T: float | None = None, wlf: dict | None = None):
    """(K', K'') with optional WLF shift at temperature T."""
    w = float(omega)
    if T is not None and wlf is not None:
        w = reduced_omega(w, T, wlf["C1"], wlf["C2"], wlf["T_ref"])
    ka = np.asarray(k, dtype=np.float64)
    ta = np.asarray(tau, dtype=np.float64)
    return prony_storage_loss_numba(w, float(K0), ka, ta)


def loss_tangent(storage: float, loss: float) -> float:
    """tan(delta) = G''/G'."""
    return float(loss) / max(float(storage), 1e-300)


def young_complex(Gs: float, Gl: float, Ks: float, Kl: float):
    """Complex Young's modulus (E', E'') from complex G, K.

    E* = 9*K*G*/(3*K*+G*), complex algebra in full.
    """
    Gc = complex(Gs, Gl)
    Kc = complex(Ks, Kl)
    Ec = 9.0 * Kc * Gc / (3.0 * Kc + Gc)
    return Ec.real, Ec.imag
