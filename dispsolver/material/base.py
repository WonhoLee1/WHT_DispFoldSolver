"""
base.py
=======
Abstract base class for hyperelastic material models.

JAX autodiff pipeline
---------------------
Each material defines W(C_flat) — strain energy density as a function
of the independent components of C (upper triangle of 3×3).

Pipeline:
1. Embed F_2d → C_3d (plane strain: F_33=1)
2. Extract 6 independent components of C: [C11, C22, C33, C12, C13, C23]
3. W(C_flat) → S_6 = 2 * dW/dC_flat (6-component 2nd P-K stress) via jax.grad
4. C_6x6 = 4 * d²W/dC_flat² (tangent moduli) via jax.hessian
5. For plane strain: extract first 3 components → [S_11, S_22, S_12] and 3×3 tangent

Usage
-----
    model = NeoHookean()
    F = jnp.array([[1.01, 0.0], [0.0, 1.0]])
    S = model.pk2_voigt(F, params)  # [S_11, S_22, S_12]
    C = model.tangent_voigt(F, params)  # 3×3
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import jax
import jax.numpy as jnp


# Type alias
Params = Dict[str, jnp.ndarray]


def _F_to_C_flat(F_2d: jnp.ndarray) -> jnp.ndarray:
    """Embed 2D F into 3D and extract 6 independent components of C.

    Plane strain: F_33 = 1, F_13 = F_23 = F_31 = F_32 = 0

    Returns
    -------
    C_flat : (6,) — [C_11, C_22, C_33, C_12, C_13, C_23]
    """
    F = jnp.eye(3, dtype=F_2d.dtype)
    F = F.at[:2, :2].set(F_2d)
    C = F.T @ F  # (3, 3) symmetric
    return jnp.array([
        C[0, 0], C[1, 1], C[2, 2],
        C[0, 1], C[0, 2], C[1, 2],
    ])


def _invariants_from_C_flat(C_flat: jnp.ndarray) -> Tuple:
    """Compute invariants from 6-component flattened C.

    Returns
    -------
    I1 : scalar — tr(C)
    I1_bar : scalar — J^{-2/3} * I1 (deviatoric invariant)
    J : scalar — det(F) = sqrt(det(C))
    """
    C11, C22, C33, C12, C13, C23 = C_flat
    I1 = C11 + C22 + C33
    # det(C) for symmetric 3×3 from 6 components
    detC = (C11 * (C22 * C33 - C23 * C23)
            - C12 * (C12 * C33 - C13 * C23)
            + C13 * (C12 * C23 - C13 * C22))
    J = jnp.sqrt(jnp.maximum(detC, 1e-30))
    I1_bar = J ** (-2.0 / 3.0) * I1
    return I1, I1_bar, J


def _symmetrize_grad(dW_dC_flat: jnp.ndarray) -> jnp.ndarray:
    """Convert gradient w.r.t. 6 C-components to symmetric 3×3 tensor.

    The 2nd P-K stress S is symmetric.  The gradient w.r.t. the 6 independent
    components already accounts for the symmetry via the factor:
        ∂W/∂C_ij (i≠j) contributes to both S_ij and S_ji.

    For the conversion to full tensor, off-diagonal components
    need a factor of 1/2 because δC_12 varies both C_12 and C_21.

    S_12 = 2 * ∂W/∂C_12 (where C_12 is the stored off-diagonal)
    """
    S_full = jnp.zeros((3, 3), dtype=dW_dC_flat.dtype)
    # Diagonal
    S_full = S_full.at[0, 0].set(dW_dC_flat[0])
    S_full = S_full.at[1, 1].set(dW_dC_flat[1])
    S_full = S_full.at[2, 2].set(dW_dC_flat[2])
    # Off-diagonal: C_12 stored at index 3, ∂/∂C_12 already accounts for both C_12 and C_21
    S_full = S_full.at[0, 1].set(dW_dC_flat[3])
    S_full = S_full.at[1, 0].set(dW_dC_flat[3])
    # C_13 at index 4, C_23 at index 5
    S_full = S_full.at[0, 2].set(dW_dC_flat[4])
    S_full = S_full.at[2, 0].set(dW_dC_flat[4])
    S_full = S_full.at[1, 2].set(dW_dC_flat[5])
    S_full = S_full.at[2, 1].set(dW_dC_flat[5])
    return S_full


class MaterialModel(ABC):
    """Abstract hyperelastic material model.

    Subclasses must implement:
        _strain_energy_from_C_flat(C_flat, params) → scalar
    """

    @abstractmethod
    def _strain_energy_from_C_flat(self, C_flat: jnp.ndarray, params: Params) -> jnp.ndarray:
        """Strain energy density W from 6 components of C.

        Parameters
        ----------
        C_flat : (6,) — [C_11, C_22, C_33, C_12, C_13, C_23]
        params : dict — material parameters

        Returns
        -------
        scalar — strain energy density W
        """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def strain_energy(self, F_2d: jnp.ndarray, params: Params) -> jnp.ndarray:
        """W(F) at a given 2D deformation gradient."""
        C_flat = _F_to_C_flat(F_2d)
        return self._strain_energy_from_C_flat(C_flat, params)

    def pk2_voigt(self, F_2d: jnp.ndarray, params: Params) -> jnp.ndarray:
        """2nd P-K stress in plane strain Voigt: [S_11, S_22, S_12].

        S = 2 * ∂W/∂C (stress conjugate to Green-Lagrange strain E)
        """
        C_flat = _F_to_C_flat(F_2d)
        dW_dC = jax.grad(self._strain_energy_from_C_flat)(C_flat, params)
        S_6 = 2.0 * dW_dC
        # Plane strain Voigt: [S_11, S_22, S_12]
        return jnp.array([S_6[0], S_6[1], S_6[3]])

    def pk2_tensor(self, F_2d: jnp.ndarray, params: Params) -> jnp.ndarray:
        """Full symmetric 3×3 2nd P-K stress tensor."""
        C_flat = _F_to_C_flat(F_2d)
        dW_dC = jax.grad(self._strain_energy_from_C_flat)(C_flat, params)
        return _symmetrize_grad(2.0 * dW_dC)

    def pk1_tensor(self, F_2d: jnp.ndarray, params: Params) -> jnp.ndarray:
        """1st P-K stress tensor P = F @ S."""
        S = self.pk2_tensor(F_2d, params)
        F = jnp.eye(3)
        F = F.at[:2, :2].set(F_2d)
        return F @ S

    def tangent_voigt(self, F_2d: jnp.ndarray, params: Params) -> jnp.ndarray:
        """Material tangent in plane strain Voigt (3×3).

        C_voigt[i, j] = ∂S_voigt[i] / ∂E_voigt[j]

        Computed via 4 * hessian(W) w.r.t. C_flat (6 components),
        then contracted to 3×3 plane strain Voigt.

        E_voigt = [E_11, E_22, 2*E_12] = [C_11/2 - 1/2, C_22/2 - 1/2, C_12]

        So:
            C_voigt[0,j] = ∂S_11 / ∂E_voigt[j]
                         = ∂(2*∂W/∂C_11) / ∂E_voigt[j]
                         = 4 * ∂²W/∂C_11² * ∂C_11/∂E_11 + ... [chain rule]

        Since E_11 = C_11/2 - 1/2:  dC_11/dE_11 = 2
        Since E_22 = C_22/2 - 1/2:  dC_22/dE_22 = 2
        Since 2*E_12 = C_12:        dC_12/d(2*E_12) = 1

        C_voigt[i, j] = 4 * H[unique_i, unique_j] * scale_j
        where H = ∂²W/∂C² (6×6 Hessian), and scale_j = dC_j/dE_voigt[j]
        """
        C_flat = _F_to_C_flat(F_2d)
        # Hessian: ∂²W/∂Cₐ ∂C_b  for a,b in [0..5]
        H = jax.hessian(self._strain_energy_from_C_flat)(C_flat, params)  # (6, 6)

        # Map from 6-component C indices to Voigt [C_11 → 0, C_22 → 1, C_12 → 3]
        # E_voigt = [E_11, E_22, 2*E_12]
        # dC_11/dE_11 = 2, dC_22/dE_22 = 2, dC_12/d(2*E_12) = 1
        C_voigt = jnp.zeros((3, 3), dtype=H.dtype)

        # C_voigt[0,0] = 4 * H[0,0] * 2  (E_11 ← C_11)
        # C_voigt[0,1] = 4 * H[0,1] * 2  (E_22 ← C_22)
        # C_voigt[0,2] = 4 * H[0,3] * 1  (2*E_12 ← C_12)
        # C_voigt[1,0] = 4 * H[1,0] * 2
        # C_voigt[1,1] = 4 * H[1,1] * 2
        # C_voigt[1,2] = 4 * H[1,3] * 1
        # C_voigt[2,0] = 4 * H[3,0] * 2
        # C_voigt[2,1] = 4 * H[3,1] * 2
        # C_voigt[2,2] = 4 * H[3,3] * 1

        C_voigt = C_voigt.at[0, 0].set(4.0 * H[0, 0])
        C_voigt = C_voigt.at[0, 1].set(4.0 * H[0, 1])
        C_voigt = C_voigt.at[0, 2].set(2.0 * H[0, 3])
        C_voigt = C_voigt.at[1, 0].set(4.0 * H[1, 0])
        C_voigt = C_voigt.at[1, 1].set(4.0 * H[1, 1])
        C_voigt = C_voigt.at[1, 2].set(2.0 * H[1, 3])
        C_voigt = C_voigt.at[2, 0].set(2.0 * H[3, 0])
        C_voigt = C_voigt.at[2, 1].set(2.0 * H[3, 1])
        C_voigt = C_voigt.at[2, 2].set(2.0 * H[3, 3])

        return C_voigt

    def linear_elastic_moduli(self, params: Params) -> Tuple[float, float]:
        """Return (E_young, nu_poisson) at small-strain limit."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
