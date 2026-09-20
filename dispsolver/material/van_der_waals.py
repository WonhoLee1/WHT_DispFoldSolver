# -*- coding: utf-8 -*-
"""van_der_waals.py: Kilian-type lock-up on a Mooney-Rivlin base (JAX).

W = C10*(I1b-3) + C01*(I2b-3) - k*ln(1-u/lam_m) - k*u/lam_m
    + (K/2)*(J-1)^2,  u = I1b - 3.

The lock term has zero value AND zero slope at rest, so small-strain
response is exactly Mooney-Rivlin (mu = 2*(C10+C01)) for any (k, lam_m),
and diverges logarithmically as I1b -> 3 + lam_m. k = 0 reduces to
Mooney-Rivlin (cross-model gate). lam_m is the excess-invariant
ceiling (NOT a stretch; must be > 0).

params: {'C10', 'C01', 'K', 'k_lock', 'lam_m'}.
Reference: Kilian & Vilgis 1984 (lock-up idea); Mooney-Rivlin base.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from .base import MaterialModel, _invariants_from_C_flat


class VanDerWaals(MaterialModel):
    """Mooney-Rivlin base with Kilian logarithmic chain-locking."""

    def _strain_energy_from_C_flat(self, C_flat: jnp.ndarray, params: Dict) -> jnp.ndarray:
        C10 = params["C10"]
        C01 = params["C01"]
        K = params["K"]
        k = params["k_lock"]
        lam_m = params["lam_m"]
        C11, C22, C33, C12, C13, C23 = C_flat
        trC = C11 + C22 + C33
        trC2 = C11**2 + C22**2 + C33**2 + 2.0 * (C12**2 + C13**2 + C23**2)
        I2 = 0.5 * (trC**2 - trC2)
        detC = (C11 * (C22 * C33 - C23 * C23)
                - C12 * (C12 * C33 - C13 * C23)
                + C13 * (C12 * C23 - C13 * C22))
        J = jnp.sqrt(jnp.maximum(detC, 1e-30))
        I1b = J ** (-2.0 / 3.0) * trC
        I2b = J ** (-4.0 / 3.0) * I2
        u = I1b - 3.0
        uu = jnp.maximum(u / lam_m, 1e-12)
        uu = jnp.minimum(uu, 1.0 - 1e-12)
        lock = -k * jnp.log(1.0 - uu) - k * uu
        W = C10 * (I1b - 3.0) + C01 * (I2b - 3.0) + lock
        return W + 0.5 * K * (J - 1.0) ** 2

    def linear_elastic_moduli(self, params: Dict) -> tuple:
        G0 = 2.0 * (params["C10"] + params["C01"])
        K = params["K"]
        nu = (3.0 * K - 2.0 * G0) / (2.0 * (3.0 * K + G0))
        E = 2.0 * G0 * (1.0 + nu)
        return E, nu
