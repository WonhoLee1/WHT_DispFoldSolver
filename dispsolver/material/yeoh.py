"""
yeoh.py
=======
YEOH hyperelastic material model (3-term polynomial in deviatoric Ī1).

Strain energy density:
    W = C10·(Ī1 − 3) + C20·(Ī1 − 3)² + C30·(Ī1 − 3)³
      + (1/D1)·(J − 1)² + (1/D2)·(J − 1)⁴ + (1/D3)·(J − 1)⁶

where:
    Ī1 = J^{-2/3}·I1  — deviatoric first invariant
    J = det(F)          — volume ratio

Small-strain limit (C20=C30=0, D1>0):
    G₀ = 2·C10
    K₀ = 2/D1
    E = 9·K₀·G₀ / (3·K₀ + G₀)
    ν = (3·K₀ − 2·G₀) / (2·(3·K₀ + G₀))

Parameters dict:
    {'C10': float, 'C20': float, 'C30': float,
     'D1': float, 'D2': float, 'D3': float}
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from .base import MaterialModel, _invariants_from_C_flat


class Yeoh(MaterialModel):
    """YEOH (3-term polynomial) hyperelastic material.

    Uses deviatoric invariant Ī1 = J^{-2/3}·I1 for the polynomial terms
    to ensure zero stress at the undeformed state.
    """

    def _strain_energy_from_C_flat(self, C_flat: jnp.ndarray, params: Dict) -> jnp.ndarray:
        _, I1_bar, J = _invariants_from_C_flat(C_flat)

        Ibar = I1_bar - 3.0
        W_dev = (params['C10'] * Ibar
                 + params.get('C20', 0.0) * Ibar ** 2
                 + params.get('C30', 0.0) * Ibar ** 3)

        Jm1 = J - 1.0
        D1 = params.get('D1', None)
        W_vol = 0.0
        if D1 is not None and D1 != 0.0:
            W_vol = Jm1 ** 2 / D1
            D2 = params.get('D2', 0.0)
            D3 = params.get('D3', 0.0)
            if D2 != 0.0:
                W_vol += Jm1 ** 4 / D2
            if D3 != 0.0:
                W_vol += Jm1 ** 6 / D3

        return W_dev + W_vol

    def linear_elastic_moduli(self, params: Dict) -> tuple:
        C10 = params['C10']
        D1 = params.get('D1', None)
        G0 = 2.0 * C10
        if D1 is not None and D1 != 0.0:
            K0 = 2.0 / D1
            E = 9.0 * K0 * G0 / (3.0 * K0 + G0)
            nu = (3.0 * K0 - 2.0 * G0) / (2.0 * (3.0 * K0 + G0))
        else:
            E = 6.0 * C10
            nu = 0.5 - 1e-6
        return E, nu
