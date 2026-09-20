# -*- coding: utf-8 -*-
"""anisotropic_hyperelastic.py: Holzapfel-Gasser-Ogden fiber dispersion model.

W = W_base(I1b) + sum_{i=4,6} (k1/2k2)*[exp(k2*E_i^2) - 1],
E_i = kappa*I1b + (1-3*kappa)*I_i - 1, I_i = a_i . C . a_i.

kappa = 0 reduces to HGO; kappa = 1/3 is isotropic. E_i < 0 switches the
fiber term off (buckling, jnp.where -- subgradient at the kink, monitored
by the FD gate). JAX path only (autodiff tangent); Numba is out of scope
per the design doc (non-display-core material).

params: {'mu', 'K', 'k1', 'k2', 'kappa', 'a4', 'a6'} with a4/a6 3-vectors.
Reference: Holzapfel-Gasser-Ogden (2000); Gasser-Ogden-Holzapfel (2006).
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from .base import MaterialModel


class HolzapfelGasser(MaterialModel):
    """HGO/GOH anisotropic hyperelastic material (2D plane strain + 3D C)."""

    def _strain_energy_from_C_flat(self, C_flat: jnp.ndarray, params: Dict) -> jnp.ndarray:
        mu = params["mu"]
        K = params["K"]
        k1 = params["k1"]
        k2 = params["k2"]
        kappa = params["kappa"]
        a4 = jnp.asarray(params["a4"], dtype=C_flat.dtype)
        a6 = jnp.asarray(params["a6"], dtype=C_flat.dtype)

        C11, C22, C33, C12, C13, C23 = C_flat
        trC = C11 + C22 + C33
        detC = (C11 * (C22 * C33 - C23 * C23)
                - C12 * (C12 * C33 - C13 * C23)
                + C13 * (C12 * C23 - C13 * C22))
        J = jnp.sqrt(jnp.maximum(detC, 1e-30))
        I1b = J ** (-2.0 / 3.0) * trC
        W = 0.5 * mu * (I1b - 3.0) + 0.5 * K * (J - 1.0) ** 2

        Cmat = jnp.array([[C11, C12, C13],
                          [C12, C22, C23],
                          [C13, C23, C33]])
        for a in (a4, a6):
            Ca = Cmat @ a
            Ii = jnp.dot(a, Ca)
            Ei = kappa * I1b + (1.0 - 3.0 * kappa) * Ii - 1.0
            W = W + jnp.where(Ei > 0.0,
                              (k1 / (2.0 * k2)) * (jnp.exp(k2 * Ei * Ei) - 1.0),
                              0.0)
        return W

    def linear_elastic_moduli(self, params: Dict) -> tuple:
        G0 = params["mu"]
        K = params["K"]
        nu = (3.0 * K - 2.0 * G0) / (2.0 * (3.0 * K + G0))
        E = 2.0 * G0 * (1.0 + nu)
        return E, nu
