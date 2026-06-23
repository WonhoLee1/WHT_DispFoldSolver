"""
q4_up_jax.py
============
JAX-compatible Q1P0 Hybrid (u-p) Element formulation via Static Condensation.
Provides energy-based evaluation of element forces and stiffness matrix.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict
from . import q4_jax

def _compute_C_components(coords: jnp.ndarray, u_elem: jnp.ndarray, xi: float, eta: float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute green-lagrange strain tensor C components for plane strain."""
    _, _, invJ = q4_jax.jacobian(xi, eta, coords)
    dN_dxi, dN_deta = q4_jax.shape_derivatives(xi, eta)

    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    
    ux = u_elem[0::2]
    uy = u_elem[1::2]
    
    grad_u_00 = jnp.sum(ux * dN_dx)
    grad_u_01 = jnp.sum(ux * dN_dy)
    grad_u_10 = jnp.sum(uy * dN_dx)
    grad_u_11 = jnp.sum(uy * dN_dy)
    
    # F = I + grad_u
    F00 = 1.0 + grad_u_00; F01 = grad_u_01
    F10 = grad_u_10;       F11 = 1.0 + grad_u_11
    
    # C = F^T @ F
    C11 = F00*F00 + F10*F10
    C22 = F01*F01 + F11*F11
    C12 = F00*F01 + F10*F11
    
    J = F00*F11 - F01*F10
    return C11, C22, C12, J

def compute_element_energy(coords: jnp.ndarray, u_elem: jnp.ndarray, params: Dict) -> jnp.ndarray:
    """Compute the hybrid (u-p) element total potential energy.
    
    p is statically condensed by inserting its analytical extremum:
      p(u) = K/V * sum_{gp} (J_gp - 1) * dV_gp
    into the Legendre-transformed energy.
    """
    mu = params.get('mu', None)
    lam = params.get('lambda', None)
    if mu is None or lam is None:
        E = params['E']
        nu = params['nu']
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        
    K_bulk = lam + (2.0 / 3.0) * mu

    # 1. Integrate deviatoric energy & volume change over Gauss points
    gp_coords = q4_jax._GP2
    gp_weights = q4_jax._W2
    
    E_dev = 0.0
    V_elem = 0.0
    int_J_minus_1 = 0.0
    
    for gp in range(4):
        xi, eta = gp_coords[gp]
        _, detJ, _ = q4_jax.jacobian(xi, eta, coords)
        w = detJ * gp_weights[gp]
        
        C11, C22, C12, J_gp = _compute_C_components(coords, u_elem, xi, eta)
        
        # 3D invariants for Plane Strain (C33 = 1)
        I1 = C11 + C22 + 1.0
        I1_bar = J_gp ** (-2.0 / 3.0) * I1
        
        # Deviatoric Neo-Hookean energy density
        W_dev = 0.5 * mu * (I1_bar - 3.0)
        
        E_dev += W_dev * w
        V_elem += w
        int_J_minus_1 += (J_gp - 1.0) * w

    # 2. Analytical value of condensed pressure p(u)
    p_condensed = (K_bulk / V_elem) * int_J_minus_1
    
    # 3. Total hybrid potential energy
    # E_total = E_dev + p * int_J_minus_1 - 0.5 * (p^2 * V) / K
    # Substituting p = (K/V) * int_J_minus_1:
    # E_total = E_dev + 0.5 * (K/V) * (int_J_minus_1)^2
    E_total = E_dev + 0.5 * (K_bulk / V_elem) * (int_J_minus_1 ** 2)
    return E_total

def compute_hybrid_element_contributions(coords: jnp.ndarray, u_elem: jnp.ndarray, params: Dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the statically condensed element internal force vector (8,) and stiffness matrix (8,8) using JAX autodiff."""
    energy_fn = lambda u: compute_element_energy(coords, u, params)
    
    f_int = jax.grad(energy_fn)(u_elem)
    K_e = jax.hessian(energy_fn)(u_elem)
    
    return f_int, K_e
