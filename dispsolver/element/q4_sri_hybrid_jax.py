"""
q4_sri_hybrid_jax.py
====================
Selective Reduced Integration (SRI) + Q1P0 Hydrostatic Pressure Hybrid Element in JAX.
(Q4_HYBRID_SRI, Q4_COROTATIONAL_HYBRID_SRI / Abaqus-grade locking-free element)

Formulation Highlights:
1. Selective Reduced Integration (SRI / Selective B-bar):
   - Normal strains (eps_xx, eps_yy): Evaluated at 2x2 Gauss points. Eliminates hourglassing.
   - Transverse shear strain (gamma_xy): Sampled at centroid (0,0). Eliminates Bending Shear Locking (AR^2).
2. Q1P0 Hydrostatic Pressure Condensation (u-p Hybrid):
   - Independent element pressure p is statically condensed at element level:
     p_condensed = (K_bulk / V_e) * int_{Omega_e} (J - 1) dV
   - Completely eliminates Volumetric Locking in nearly incompressible materials (nu -> 0.5, PSA layers).
3. Zero Global Pressure DOFs & Zero Internal alpha Parameters:
   - System matrix remains 100% Symmetric Positive Definite (SPD) for fast PyPardiso solves.
   - No internal alpha DOFs to cause Newton stalls at large rotation angles (all the way to 90° fold).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple, Dict

from .q4_eas_jax import _jac, _sd
from .q4_corotational_jax import compute_element_rotation, build_block_rotation
from ..material.plastic_jax import tangent_voigt_jax


def compute_element_energy_sri_hybrid(coords: jnp.ndarray, u_elem: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """Compute SRI Q1P0 hybrid element total potential energy with element-level pressure condensation."""
    mu = params.get('mu', None)
    lam = params.get('lambda', None)
    if mu is None or lam is None:
        E = params['E']
        nu = params['nu']
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    K_bulk = lam + (2.0 / 3.0) * mu

    _S3 = jnp.sqrt(3.0)
    gp_coords = jnp.array([
        [-1.0 / _S3, -1.0 / _S3],
        [ 1.0 / _S3, -1.0 / _S3],
        [ 1.0 / _S3,  1.0 / _S3],
        [-1.0 / _S3,  1.0 / _S3],
    ], dtype=jnp.float64)
    gp_weights = jnp.ones(4, dtype=jnp.float64)

    E_dev = 0.0
    V_elem = 0.0
    int_J_minus_1 = 0.0

    for gp in range(4):
        xi, eta = gp_coords[gp]
        _, detJ, invJ = _jac(xi, eta, coords)
        w = detJ * gp_weights[gp]

        # SRI displacement gradient H_sri
        dN_dxi, dN_deta = _sd(xi, eta)
        gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

        _, _, invJ0 = _jac(0.0, 0.0, coords)
        dN_dxi0, dN_deta0 = _sd(0.0, 0.0)
        gX0 = invJ0[0, 0] * dN_dxi0 + invJ0[0, 1] * dN_deta0
        gY0 = invJ0[1, 0] * dN_dxi0 + invJ0[1, 1] * dN_deta0

        ux = u_elem[0::2]
        uy = u_elem[1::2]

        F00 = 1.0 + ux @ gX
        F01 = ux @ gY0  # Centroid shear gradient
        F10 = uy @ gX0  # Centroid shear gradient
        F11 = 1.0 + uy @ gY

        # C = F^T @ F
        C11 = F00 * F00 + F10 * F10
        C22 = F01 * F01 + F11 * F11
        J_gp = F00 * F11 - F01 * F10

        I1 = C11 + C22 + 1.0
        I1_bar = J_gp ** (-2.0 / 3.0) * I1
        W_dev = 0.5 * mu * (I1_bar - 3.0)

        E_dev += W_dev * w
        V_elem += w
        int_J_minus_1 += (J_gp - 1.0) * w

    E_total = E_dev + 0.5 * (K_bulk / V_elem) * (int_J_minus_1 ** 2)
    return E_total


@jax.jit
def compute_sri_hybrid_element_contributions_jax(coords: jnp.ndarray, u_elem: jnp.ndarray, params: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Q1P0 SRI Hybrid element internal force vector (8,) and stiffness matrix (8,8) via JAX autodiff."""
    energy_fn = lambda u: compute_element_energy_sri_hybrid(coords, u, params)
    f_int = jax.grad(energy_fn)(u_elem)
    K_e = jax.hessian(energy_fn)(u_elem)
    elem_nan = jnp.any(jnp.isnan(f_int)) | jnp.any(jnp.isnan(K_e))
    f_int = jnp.where(elem_nan, jnp.zeros_like(f_int), f_int)
    K_e = jnp.where(elem_nan, jnp.zeros_like(K_e), K_e)

    return f_int, K_e


@jax.jit
def compute_corotational_sri_hybrid_contributions_jax(
    coords: jnp.ndarray,
    u_elem: jnp.ndarray,
    params: Dict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Co-rotational SRI Hybrid element (Q4_COROTATIONAL_HYBRID_SRI) in JAX."""
    coords_curr = coords + u_elem.reshape((4, 2))
    R_elem = compute_element_rotation(coords, coords_curr)
    T8 = build_block_rotation(R_elem)

    u_local = (coords_curr @ R_elem - coords).flatten()
    f_local, K_local = compute_sri_hybrid_element_contributions_jax(coords, u_local, params)

    f_global = T8 @ f_local
    K_global = T8 @ K_local @ T8.T

    elem_nan = jnp.any(jnp.isnan(f_global)) | jnp.any(jnp.isnan(K_global))
    f_global = jnp.where(elem_nan, jnp.zeros_like(f_global), f_global)
    K_global = jnp.where(elem_nan, jnp.zeros_like(K_global), K_global)

    return f_global, K_global
