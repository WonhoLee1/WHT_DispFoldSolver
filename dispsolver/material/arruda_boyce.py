"""
arruda_boyce.py
===============
Arruda-Boyce (8-chain) hyperelastic model with deviatoric/volumetric split.

Strain energy density (5-term expansion in Ī1):
    W = μ · Σₖ₌₁⁵ Cₖ / λₘ^{2(k−1)} · (Ī₁ᵏ − 3ᵏ)
      + (K/2) · (J − 1)²

where:
    C₁ = 1/2,  C₂ = 1/20,  C₃ = 11/1050,
    C₄ = 19/7000,  C₅ = 519/673750
    Ī₁ = J^{-2/3}·I₁ — deviatoric invariant
    λₘ — locking stretch

Parameters dict:
    {'mu': float, 'lambda_m': float, 'K': float}

Reference
---------
Arruda, E.M. & Boyce, M.C. (1993). J. Mech. Phys. Solids, 41(2), 389-412.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from .base import MaterialModel, _invariants_from_C_flat


_AB_COEFFS = [
    (0.5, 0),                # k=1: C1=1/2,   λ_m^{0}
    (1.0 / 20.0, 2),         # k=2: C2=1/20,  λ_m^{2}
    (11.0 / 1050.0, 4),      # k=3: C3=11/1050,  λ_m^{4}
    (19.0 / 7000.0, 6),      # k=4: C4=19/7000,  λ_m^{6}
    (519.0 / 673750.0, 8),   # k=5: C5=519/673750, λ_m^{8}
]


class ArrudaBoyce(MaterialModel):
    """Arruda-Boyce (8-chain) hyperelastic material.

    Uses deviatoric invariant Ī1 for the chain stretch terms,
    bulk modulus K for the volumetric term.
    """

    def _strain_energy_from_C_flat(self, C_flat: jnp.ndarray, params: Dict) -> jnp.ndarray:
        _, I1_bar, J = _invariants_from_C_flat(C_flat)

        mu = params['mu']
        lambda_m = params.get('lambda_m', 3.0)
        K = params.get('K', 0.0)

        # Deviatoric part: 5-term series in I1_bar
        W_dev = 0.0
        for Ck, exp in _AB_COEFFS:
            if exp == 0:
                term = Ck * (I1_bar - 3.0)
            else:
                term = Ck * (I1_bar ** (exp // 2 + 1) - 3.0 ** (exp // 2 + 1))
                denom = lambda_m ** exp
                if denom > 0:
                    term = term / denom
            W_dev = W_dev + term
        W_dev = mu * W_dev

        # Volumetric part
        Jm1 = J - 1.0
        W_vol = 0.5 * K * Jm1 ** 2

        return W_dev + W_vol

    def linear_elastic_moduli(self, params: Dict) -> tuple:
        mu = params['mu']
        K = params.get('K', 0.0)
        G0 = mu
        if K > 0:
            nu = (3.0 * K - 2.0 * G0) / (2.0 * (3.0 * K + G0))
            E = 2.0 * G0 * (1.0 + nu)
        else:
            nu = 0.5 - 1e-6
            E = 3.0 * G0
        return E, nu
