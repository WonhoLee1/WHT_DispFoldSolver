"""
q4_sri_jax.py
=============
Selective Reduced Integration (SRI / Selective B-bar) Q4 Plane Strain Element in JAX.

Formulation:
- Normal / Volumetric Strains (eps_xx, eps_yy): Evaluated at 2x2 Gauss points.
  No hourglass modes form because 2x2 normal integration fully constrains h=[1,-1,1,-1].
- Transverse Shear Strain (gamma_xy): Sampled at element centroid (xi=0, eta=0).
  Completely eliminates Bending Shear Locking (AR^2 locking) in thin multi-layer stacks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Tuple

from .q4_eas_jax import _jac, _sd, _BL_columns
from .q4_corotational_jax import compute_element_rotation, build_block_rotation
from ..material.plastic_jax import tangent_voigt_jax

_S3 = jnp.sqrt(3.0)
_SRI_GP2 = jnp.array([
    [-1.0 / _S3, -1.0 / _S3],
    [ 1.0 / _S3, -1.0 / _S3],
    [ 1.0 / _S3,  1.0 / _S3],
    [-1.0 / _S3,  1.0 / _S3],
], dtype=jnp.float64)
_SRI_W2 = jnp.ones(4, dtype=jnp.float64)


@jax.jit
def compute_sri_j2_contributions_jax(
    coords: jnp.ndarray,
    u_elem: jnp.ndarray,
    state_elem: jnp.ndarray,
    lam: float,
    mu: float,
    sigma_y0: float,
    H: float,
    thickness: float = 1.0,
    F_n_gps: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute SRI Q4 internal force vector (8,) and tangent stiffness matrix (8, 8) with J2 plasticity."""
    # Compute centroid derivatives gX0, gY0
    _, _, invJ0 = _jac(0.0, 0.0, coords)
    dN_dxi0, dN_deta0 = _sd(0.0, 0.0)
    gX0 = invJ0[0, 0] * dN_dxi0 + invJ0[0, 1] * dN_deta0
    gY0 = invJ0[1, 0] * dN_dxi0 + invJ0[1, 1] * dN_deta0

    f_local = jnp.zeros(8, dtype=jnp.float64)
    K_local = jnp.zeros((8, 8), dtype=jnp.float64)
    state_new = jnp.zeros_like(state_elem)

    for k in range(4):
        xi, eta = _SRI_GP2[k]
        _, detJ, invJ = _jac(xi, eta, coords)
        dN_dxi, dN_deta = _sd(xi, eta)
        gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

        ux = u_elem[0::2]
        uy = u_elem[1::2]
        Hc = jnp.array([[ux @ gX, ux @ gY], [uy @ gX, uy @ gY]])
        F_local = jnp.eye(2, dtype=jnp.float64) + Hc

        S_v, C_v, sn = tangent_voigt_jax(F_local, state_elem[k], lam, mu, sigma_y0, H)
        state_new = state_new.at[k].set(sn)

        BL_full = _BL_columns(F_local, gX, gY)
        BL_centroid = _BL_columns(F_local, gX0, gY0)

        # SRI B-matrix: rows 0,1 from full GP, row 2 (shear) from centroid
        BL_sri = jnp.zeros((3, 8), dtype=jnp.float64)
        BL_sri = BL_sri.at[0, :].set(BL_full[0, :])
        BL_sri = BL_sri.at[1, :].set(BL_full[1, :])
        BL_sri = BL_sri.at[2, :].set(BL_centroid[2, :])

        w = detJ * _SRI_W2[k] * thickness
        f_local = f_local + BL_sri.T @ S_v * w
        K_local = K_local + (BL_sri.T @ C_v @ BL_sri) * w

    elem_nan = jnp.any(jnp.isnan(f_local)) | jnp.any(jnp.isnan(K_local))
    f_local = jnp.where(elem_nan, jnp.zeros_like(f_local), f_local)
    K_local = jnp.where(elem_nan, jnp.zeros_like(K_local), K_local)

    return f_local, K_local, state_new, None


@jax.jit
def compute_corotational_sri_j2_contributions_jax(
    coords: jnp.ndarray,
    u_elem: jnp.ndarray,
    state_elem: jnp.ndarray,
    lam: float,
    mu: float,
    sigma_y0: float,
    H: float,
    thickness: float = 1.0,
    F_n_gps: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Co-rotational SRI Q4 element (Q4_COROTATIONAL_SRI) in JAX."""
    coords_curr = coords + u_elem.reshape((4, 2))
    R_elem = compute_element_rotation(coords, coords_curr)
    T8 = build_block_rotation(R_elem)

    # Local deformational displacement
    u_local = (coords_curr @ R_elem - coords).flatten()

    # Compute SRI local contributions
    f_local, K_local, state_new, _ = compute_sri_j2_contributions_jax(
        coords=coords,
        u_elem=u_local,
        state_elem=state_elem,
        lam=lam,
        mu=mu,
        sigma_y0=sigma_y0,
        H=H,
        thickness=thickness,
        F_n_gps=F_n_gps,
    )

    f_global = T8 @ f_local
    K_global = T8 @ K_local @ T8.T

    elem_nan = jnp.any(jnp.isnan(f_global)) | jnp.any(jnp.isnan(K_global))
    f_global = jnp.where(elem_nan, jnp.zeros_like(f_global), f_global)
    K_global = jnp.where(elem_nan, jnp.zeros_like(K_global), K_global)

    return f_global, K_global, state_new, None
