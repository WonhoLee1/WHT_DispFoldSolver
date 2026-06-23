"""
neohookean.py
=============
Compressible Neo-Hookean hyperelastic material model.

Strain energy density (Simo & Hughes 1998):
    W = μ/2 · (I1 − 3) − μ · ln J + λ/2 · (ln J)²

where I1 = tr(C), J = det(F), and μ, λ are Lamé parameters.

Parameters dict:
    {'mu': float, 'lambda': float}  or  {'E': float, 'nu': float}
"""

from __future__ import annotations

from typing import Dict
import numpy as np
import jax.numpy as jnp

from .base import MaterialModel, _invariants_from_C_flat


class NeoHookean(MaterialModel):
    """Compressible Neo-Hookean material."""

    def _strain_energy_from_C_flat(self, C_flat: jnp.ndarray, params: Dict) -> jnp.ndarray:
        mu = params.get('mu', None)
        lam = params.get('lambda', None)
        if mu is None or lam is None:
            E = params['E']
            nu = params['nu']
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        I1_raw, _, J = _invariants_from_C_flat(C_flat)
        lnJ = jnp.log(jnp.maximum(J, 1e-30))

        W = 0.5 * mu * (I1_raw - 3.0) - mu * lnJ + 0.5 * lam * lnJ ** 2
        return W

    def linear_elastic_moduli(self, params: Dict) -> tuple:
        mu = params.get('mu')
        lam = params.get('lambda')
        if mu is None or lam is None:
            E = params['E']
            nu = params['nu']
            return E, nu
        E = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
        nu = lam / (2.0 * (lam + mu))
        return E, nu

    def pk2_voigt(self, F_2d: np.ndarray | jax.Array, params: Dict) -> np.ndarray | jax.Array:
        if not isinstance(F_2d, np.ndarray):
            return super().pk2_voigt(F_2d, params)
        F = F_2d
        mu = params.get('mu', None)
        lam = params.get('lambda', None)
        if mu is None or lam is None:
            E = params['E']
            nu = params['nu']
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            
        C_2d = F.T @ F
        detC = C_2d[0, 0] * C_2d[1, 1] - C_2d[0, 1] * C_2d[1, 0]
        J = np.sqrt(max(detC, 1e-30))
        
        inv_det = 1.0 / max(detC, 1e-30)
        C_inv_11 = C_2d[1, 1] * inv_det
        C_inv_22 = C_2d[0, 0] * inv_det
        C_inv_12 = -C_2d[0, 1] * inv_det
        
        lnJ = np.log(max(J, 1e-30))
        
        S11 = mu * (1.0 - C_inv_11) + lam * lnJ * C_inv_11
        S22 = mu * (1.0 - C_inv_22) + lam * lnJ * C_inv_22
        S12 = -mu * C_inv_12 + lam * lnJ * C_inv_12
        
        return np.array([S11, S22, S12], dtype=np.float64)

    def pk2_tensor(self, F_2d: np.ndarray | jax.Array, params: Dict) -> np.ndarray | jax.Array:
        if not isinstance(F_2d, np.ndarray):
            return super().pk2_tensor(F_2d, params)
        F = F_2d
        mu = params.get('mu', None)
        lam = params.get('lambda', None)
        if mu is None or lam is None:
            E = params['E']
            nu = params['nu']
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            
        C_2d = F.T @ F
        detC = C_2d[0, 0] * C_2d[1, 1] - C_2d[0, 1] * C_2d[1, 0]
        J = np.sqrt(max(detC, 1e-30))
        
        inv_det = 1.0 / max(detC, 1e-30)
        C_inv_11 = C_2d[1, 1] * inv_det
        C_inv_22 = C_2d[0, 0] * inv_det
        C_inv_12 = -C_2d[0, 1] * inv_det
        
        lnJ = np.log(max(J, 1e-30))
        
        S = np.zeros((3, 3), dtype=np.float64)
        S[0, 0] = mu * (1.0 - C_inv_11) + lam * lnJ * C_inv_11
        S[1, 1] = mu * (1.0 - C_inv_22) + lam * lnJ * C_inv_22
        S[0, 1] = S[1, 0] = -mu * C_inv_12 + lam * lnJ * C_inv_12
        S[2, 2] = lam * lnJ
        return S

    def tangent_voigt(self, F_2d: np.ndarray | jax.Array, params: Dict) -> np.ndarray | jax.Array:
        if not isinstance(F_2d, np.ndarray):
            return super().tangent_voigt(F_2d, params)
        F = F_2d
        mu = params.get('mu', None)
        lam = params.get('lambda', None)
        if mu is None or lam is None:
            E = params['E']
            nu = params['nu']
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            
        C_2d = F.T @ F
        detC = C_2d[0, 0] * C_2d[1, 1] - C_2d[0, 1] * C_2d[1, 0]
        J = np.sqrt(max(detC, 1e-30))
        
        inv_det = 1.0 / max(detC, 1e-30)
        C_inv_11 = C_2d[1, 1] * inv_det
        C_inv_22 = C_2d[0, 0] * inv_det
        C_inv_12 = -C_2d[0, 1] * inv_det
        
        lnJ = np.log(max(J, 1e-30))
        fac = mu - lam * lnJ
        
        C_voigt = np.zeros((3, 3), dtype=np.float64)
        
        C_voigt[0, 0] = lam * (C_inv_11 ** 2) + 2.0 * fac * (C_inv_11 ** 2)
        C_voigt[1, 1] = lam * (C_inv_22 ** 2) + 2.0 * fac * (C_inv_22 ** 2)
        C_voigt[0, 1] = C_voigt[1, 0] = lam * C_inv_11 * C_inv_22 + 2.0 * fac * (C_inv_12 ** 2)
        
        C_voigt[0, 2] = C_voigt[2, 0] = lam * C_inv_11 * C_inv_12 + 2.0 * fac * C_inv_11 * C_inv_12
        C_voigt[1, 2] = C_voigt[2, 1] = lam * C_inv_22 * C_inv_12 + 2.0 * fac * C_inv_22 * C_inv_12
        
        C_voigt[2, 2] = lam * (C_inv_12 ** 2) + fac * (C_inv_11 * C_inv_22 + C_inv_12 ** 2)

        return C_voigt

    # ------------------------------------------------------------------
    # Batched numpy interface (N elements/GPs simultaneously)
    # These vectorize the analytical numpy formulas above *exactly*, so
    # batch assembly matches the sequential per-element path bit-for-bit.
    # ------------------------------------------------------------------

    @staticmethod
    def _lame(params: Dict) -> tuple:
        mu = params.get('mu', None)
        lam = params.get('lambda', None)
        if mu is None or lam is None:
            E = params['E']
            nu = params['nu']
            mu = E / (2.0 * (1.0 + nu))
            lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return float(mu), float(lam)

    @staticmethod
    def _cinv_batch(F_batch: np.ndarray):
        """Return C^{-1} components + J, lnJ for a batch of (N,2,2) F."""
        C = np.einsum('nki,nkj->nij', F_batch, F_batch)   # F^T F (symmetric)
        C11 = C[:, 0, 0]; C22 = C[:, 1, 1]; C12 = C[:, 0, 1]
        detC = np.maximum(C11 * C22 - C12 * C12, 1e-30)
        J = np.sqrt(detC)
        inv = 1.0 / detC
        Ci11 = C22 * inv
        Ci22 = C11 * inv
        Ci12 = -C12 * inv
        lnJ = np.log(np.maximum(J, 1e-30))
        return Ci11, Ci22, Ci12, lnJ

    def pk2_tensor_batch(self, F_batch: np.ndarray, params: Dict) -> np.ndarray:
        """Batched PK2 (N,2,2) -> (N,3,3). Matches pk2_tensor exactly."""
        mu, lam = self._lame(params)
        N = F_batch.shape[0]
        Ci11, Ci22, Ci12, lnJ = self._cinv_batch(F_batch)
        S = np.zeros((N, 3, 3), dtype=np.float64)
        S[:, 0, 0] = mu * (1.0 - Ci11) + lam * lnJ * Ci11
        S[:, 1, 1] = mu * (1.0 - Ci22) + lam * lnJ * Ci22
        S[:, 0, 1] = S[:, 1, 0] = -mu * Ci12 + lam * lnJ * Ci12
        S[:, 2, 2] = lam * lnJ
        return S

    def tangent_voigt_batch(self, F_batch: np.ndarray, params: Dict) -> np.ndarray:
        """Batched plane-strain Voigt tangent (N,2,2) -> (N,3,3). Matches tangent_voigt."""
        mu, lam = self._lame(params)
        N = F_batch.shape[0]
        Ci11, Ci22, Ci12, lnJ = self._cinv_batch(F_batch)
        fac = mu - lam * lnJ
        Cv = np.zeros((N, 3, 3), dtype=np.float64)
        Cv[:, 0, 0] = lam * Ci11 ** 2 + 2.0 * fac * Ci11 ** 2
        Cv[:, 1, 1] = lam * Ci22 ** 2 + 2.0 * fac * Ci22 ** 2
        Cv[:, 0, 1] = Cv[:, 1, 0] = lam * Ci11 * Ci22 + 2.0 * fac * Ci12 ** 2
        Cv[:, 0, 2] = Cv[:, 2, 0] = lam * Ci11 * Ci12 + 2.0 * fac * Ci11 * Ci12
        Cv[:, 1, 2] = Cv[:, 2, 1] = lam * Ci22 * Ci12 + 2.0 * fac * Ci22 * Ci12
        Cv[:, 2, 2] = lam * Ci12 ** 2 + fac * (Ci11 * Ci22 + Ci12 ** 2)
        return Cv
