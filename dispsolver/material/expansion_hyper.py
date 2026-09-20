# -*- coding: utf-8 -*-
"""expansion_hyper.py: thermal-split adapter for hyperelastic cores (2D plane strain).

Coupling: F = F_me @ F_th with isotropic F_th = theta*I, theta = 1 + alpha*dT.
In-plane:  F_me = F/theta. Out-of-plane (plane strain constrains total F33 = 1,
so the mechanical out-of-plane stretch is 1/theta):
  Ogden (Numba spectral): l3 = 1/theta, J = lam0*lam1*l3.
  JAX cores (Mooney-Rivlin, Arruda-Boyce): f33 = 1/theta in the C embedding.

Covered cores: Ogden, Mooney-Rivlin, Arruda-Boyce. Van der Waals has no core
in this repo (docs-only), so it is out of scope here -- no model invented.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from dispsolver.material.ogden import ogden_pk2_tangent
from dispsolver.material.mooney_rivlin import MooneyRivlin
from dispsolver.material.arruda_boyce import ArrudaBoyce


def thermal_stretch(alpha: float, delta_T: float) -> float:
    """Thermal stretch ratio theta = 1 + alpha*delta_T."""
    return 1.0 + float(alpha) * float(delta_T)


def ogden_thermal(F: np.ndarray, theta: float, mus: np.ndarray,
                  alphas: np.ndarray, K: float):
    """Ogden stress/tangent at total in-plane F with thermal split theta."""
    F_me = np.asarray(F, dtype=np.float64) / theta
    l3_me = 1.0 / theta
    S, C = ogden_pk2_tangent(F_me[0, 0], F_me[0, 1], F_me[1, 0], F_me[1, 1],
                             np.asarray(mus, dtype=np.float64),
                             np.asarray(alphas, dtype=np.float64),
                             float(K), l3_me)
    return S, C / theta ** 2


def mooney_thermal(F: np.ndarray, theta: float, params: dict):
    """Mooney-Rivlin stress/tangent at total in-plane F with split theta."""
    mr = MooneyRivlin()
    F_me = jnp.array(np.asarray(F, dtype=np.float64) / theta)
    f33 = 1.0 / theta
    S = np.asarray(mr.pk2_voigt(F_me, params, f33=f33))
    C = np.asarray(mr.tangent_voigt(F_me, params, f33=f33))
    return S, C / theta ** 2


def arruda_boyce_thermal(F: np.ndarray, theta: float, params: dict):
    """Arruda-Boyce stress/tangent at total in-plane F with split theta."""
    ab = ArrudaBoyce()
    F_me = jnp.array(np.asarray(F, dtype=np.float64) / theta)
    f33 = 1.0 / theta
    S = np.asarray(ab.pk2_voigt(F_me, params, f33=f33))
    C = np.asarray(ab.tangent_voigt(F_me, params, f33=f33))
    return S, C / theta ** 2
