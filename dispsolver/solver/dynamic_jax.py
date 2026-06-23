"""
dynamic_jax.py
==============
JAX-vectorized routines for element contributions.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Any
from ..element import q4_jax

def _compute_F_jax(coords: jnp.ndarray, u_elem: jnp.ndarray, xi: float, eta: float) -> jnp.ndarray:
    _, _, invJ = q4_jax.jacobian(xi, eta, coords)
    dN_dxi, dN_deta = q4_jax.shape_derivatives(xi, eta)

    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    
    # u_elem is [ux1, uy1, ux2, uy2, ux3, uy3, ux4, uy4]
    ux = u_elem[0::2]
    uy = u_elem[1::2]
    
    grad_u_00 = jnp.sum(ux * dN_dx)
    grad_u_01 = jnp.sum(ux * dN_dy)
    grad_u_10 = jnp.sum(uy * dN_dx)
    grad_u_11 = jnp.sum(uy * dN_dy)
    
    grad_u = jnp.array([[grad_u_00, grad_u_01], [grad_u_10, grad_u_11]])
    return jnp.eye(2) + grad_u

def build_element_contributions_jax(material_model: Any, params: dict):
    """
    Builds a JIT-compilable JAX function for a specific hyperelastic material.
    Returns a function: (coords, u_elem) -> (f_int, K_e)
    """
    
    def _element_contributions_jax(coords: jnp.ndarray, u_elem: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        _, _, invJ0 = q4_jax.jacobian(0.0, 0.0, coords)
        B0 = q4_jax.B_matrix(0.0, 0.0, invJ0)

        f_int = jnp.zeros(8)
        K_e = jnp.zeros((8, 8))

        # We can use lax.fori_loop or just unroll since n_gp=4 is small
        for gp in range(4):
            xi, eta = q4_jax._GP2[gp]
            _, detJ, invJ = q4_jax.jacobian(xi, eta, coords)

            Bb = q4_jax.B_bar_matrix(xi, eta, invJ, B0, invJ0)
            F = _compute_F_jax(coords, u_elem, xi, eta)
            F0 = _compute_F_jax(coords, u_elem, 0.0, 0.0)
            J = jnp.linalg.det(F)
            J0 = jnp.linalg.det(F0)
            F_bar = F * jnp.sqrt(J0 / J)

            # Evaluate material (assuming NeoHookean or similar pure JAX material)
            # pk2_voigt and tangent_voigt must be JAX compatible
            S_v = material_model.pk2_voigt(F_bar, params)
            C_v = material_model.tangent_voigt(F_bar, params)

            w = detJ * q4_jax._W2[gp]
            f_int += Bb.T @ S_v * w
            
            # Geometric stiffness
            dN_dxi, dN_deta = q4_jax.shape_derivatives(xi, eta)
            dN_dX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
            dN_dY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
            grad_N = jnp.stack([dN_dX, dN_dY], axis=1)
            
            S_tensor = jnp.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
            gamma = grad_N @ S_tensor @ grad_N.T
            
            K_geo = jnp.zeros((8, 8))
            K_geo = K_geo.at[0::2, 0::2].set(gamma)
            K_geo = K_geo.at[1::2, 1::2].set(gamma)
            
            K_e += (Bb.T @ C_v @ Bb + K_geo) * w

        return f_int, K_e
        
    return _element_contributions_jax


def build_hybrid_element_contributions_jax(params: dict):
    """
    Builds a JIT-compilable JAX function for the Q1P0 Hybrid (u-p) element.
    Returns a function: (coords, u_elem) -> (f_int, K_e)
    """
    from ..element.q4_up_jax import compute_hybrid_element_contributions
    
    def _hybrid_element_contributions_jax(coords: jnp.ndarray, u_elem: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Directly evaluate using static-condensation energy-based formulation
        f_int, K_e = compute_hybrid_element_contributions(coords, u_elem, params)
        return f_int, K_e
        
    return _hybrid_element_contributions_jax

