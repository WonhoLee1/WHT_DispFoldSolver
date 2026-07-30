"""
q4_corotational_eas_jax.py
==========================
Co-rotational Enhanced Assumed Strain (CR-EAS) Q4 Plane Strain Element in JAX.

Combines:
1. Co-rotational frame extraction (R_elem) to isolate element rigid rotation.
2. Simo & Rifai (1990) Enhanced Assumed Strain (EAS-4) in the local frame with JAX AutoDiff & vmap.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .q4_eas_jax import compute_eas_j2_contributions_jax
from .q4_corotational_jax import compute_element_rotation, build_block_rotation

def _f_global_kinematics(u_g, coords, f_l_fixed, K_l_fixed):
    """
    Computes global internal force assuming a fixed local force-displacement
    relationship (f_l_fixed, K_l_fixed). Differentiating this with respect to
    u_g yields the exact consistent corotational tangent including spin terms.
    """
    c_curr = coords + u_g.reshape((4, 2))
    R = compute_element_rotation(coords, c_curr)
    T = build_block_rotation(R)
    u_l = (c_curr @ R - coords).flatten()
    
    # First-order Taylor expansion around the current local displacement
    u_l_current = jax.lax.stop_gradient(u_l)
    f_l_approx = f_l_fixed + K_l_fixed @ (u_l - u_l_current)
    
    return T @ f_l_approx

_K_global_fn = jax.jacobian(_f_global_kinematics, argnums=0)


@jax.jit
def compute_corotational_eas_j2_contributions_jax(
    coords,          # (4, 2) reference coords
    u_elem,          # (8,)   global displacement
    alpha,           # (4,)   warm-start EAS parameters
    state_elem,      # (4, 5) per-GP J2 state
    lam, mu, sigma_y0, H,
    thickness=1.0,
    F_n_gps=None,    # (4, 2, 2) total F at last converged step
):
    """Finite-strain Co-rotational EAS Q4 element with J2 plasticity — pure JAX.

    Returns
    -------
    f_global  : (8,)   global condensed internal force
    K_global  : (8, 8) global condensed tangent
    alpha_new : (4,)   converged EAS parameters
    state_new : (4, 5) updated material state
    F_n_new   : (4, 2, 2) updated total F per GP
    """
    coords_curr = coords + u_elem.reshape((4, 2))
    R_elem = compute_element_rotation(coords, coords_curr)
    T8 = build_block_rotation(R_elem)

    # Local deformational displacement (rigid rotation subtracted)
    u_local = (coords_curr @ R_elem - coords).flatten()

    # Compute EAS contributions in local co-rotational frame
    f_local, K_local, alpha_new, state_new, F_n_new = compute_eas_j2_contributions_jax(
        coords=coords,
        u_elem=u_local,
        alpha=alpha,
        state_elem=state_elem,
        lam=lam,
        mu=mu,
        sigma_y0=sigma_y0,
        H=H,
        thickness=thickness,
        F_n_gps=F_n_gps,
    )

    f_global = T8 @ f_local
    K_global = _K_global_fn(u_elem, coords, f_local, K_local)

    elem_nan = jnp.any(jnp.isnan(f_global)) | jnp.any(jnp.isnan(K_global))
    f_global = jnp.where(elem_nan, jnp.zeros_like(f_global), f_global)
    K_global = jnp.where(elem_nan, jnp.zeros_like(K_global), K_global)

    return f_global, K_global, alpha_new, state_new, F_n_new
