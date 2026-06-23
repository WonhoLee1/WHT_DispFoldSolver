"""
q4_jax.py
=========
JAX-compatible Q4 Plane Strain Element kinematics.
"""

import jax.numpy as jnp
from typing import Tuple

_GP2 = jnp.array([
    [-1.0 / jnp.sqrt(3), -1.0 / jnp.sqrt(3)],
    [ 1.0 / jnp.sqrt(3), -1.0 / jnp.sqrt(3)],
    [ 1.0 / jnp.sqrt(3),  1.0 / jnp.sqrt(3)],
    [-1.0 / jnp.sqrt(3),  1.0 / jnp.sqrt(3)],
], dtype=jnp.float64)
_W2 = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)

_GP0 = jnp.array([[0.0, 0.0]], dtype=jnp.float64)
_W0 = jnp.array([4.0], dtype=jnp.float64)

def shape_functions(xi: float, eta: float) -> jnp.ndarray:
    return 0.25 * jnp.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ])

def shape_derivatives(xi: float, eta: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dN_dxi = 0.25 * jnp.array([
        -(1.0 - eta),
         (1.0 - eta),
         (1.0 + eta),
        -(1.0 + eta),
    ])
    dN_deta = 0.25 * jnp.array([
        -(1.0 - xi),
        -(1.0 + xi),
         (1.0 + xi),
         (1.0 - xi),
    ])
    return dN_dxi, dN_deta

def jacobian(xi: float, eta: float, coords: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dN_dxi, dN_deta = shape_derivatives(xi, eta)
    
    J00 = jnp.sum(dN_dxi * coords[:, 0])
    J01 = jnp.sum(dN_dxi * coords[:, 1])
    J10 = jnp.sum(dN_deta * coords[:, 0])
    J11 = jnp.sum(dN_deta * coords[:, 1])
    
    J = jnp.array([[J00, J01], [J10, J11]])
    detJ = J00 * J11 - J01 * J10
    
    invJ = jnp.array([[J11, -J01], [-J10, J00]]) / detJ
    return J, detJ, invJ

def B_matrix(xi: float, eta: float, invJ: jnp.ndarray) -> jnp.ndarray:
    dN_dxi, dN_deta = shape_derivatives(xi, eta)
    
    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    
    B00 = dN_dx[0]; B01 = 0.0; B02 = dN_dx[1]; B03 = 0.0; B04 = dN_dx[2]; B05 = 0.0; B06 = dN_dx[3]; B07 = 0.0
    B10 = 0.0; B11 = dN_dy[0]; B12 = 0.0; B13 = dN_dy[1]; B14 = 0.0; B15 = dN_dy[2]; B16 = 0.0; B17 = dN_dy[3]
    B20 = dN_dy[0]; B21 = dN_dx[0]; B22 = dN_dy[1]; B23 = dN_dx[1]; B24 = dN_dy[2]; B25 = dN_dx[2]; B26 = dN_dy[3]; B27 = dN_dx[3]
    
    B = jnp.array([
        [B00, B01, B02, B03, B04, B05, B06, B07],
        [B10, B11, B12, B13, B14, B15, B16, B17],
        [B20, B21, B22, B23, B24, B25, B26, B27]
    ])
    return B

def B_bar_matrix(xi: float, eta: float, invJ: jnp.ndarray, B0: jnp.ndarray, invJ0: jnp.ndarray) -> jnp.ndarray:
    B_std = B_matrix(xi, eta, invJ)
    P_vol = (1.0 / 2.0) * jnp.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    B_dev = B_std - P_vol @ B_std
    B0_vol = P_vol @ B0
    return B_dev + B0_vol
