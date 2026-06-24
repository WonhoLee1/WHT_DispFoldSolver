"""Finite-strain (Total-Lagrangian) F-bar viscoelastic Q4 hybrid — pure JAX.

Upgrade of q4_visco_hybrid_jax (small-strain) for the PSA layers, which undergo
large rotations in the 90-degree fold. The small-strain element mistakes rigid
rotation for strain (spurious stress) and carries no geometric stiffness, giving
an inconsistent global tangent -> only linear Newton convergence.

This element fixes both:
  * strain measure = Green-Lagrange  E = 1/2 (Fbar^T Fbar - I)  -> objective
    under large rotation (rigid rotation gives E = 0, hence S = 0);
  * tangent = autodiff of the internal force (jax.jacobian). The linear
    viscoelastic constitutive has NO eigendecomposition, so autodiff is
    NaN-safe everywhere (unlike the J2 spectral path) and yields the exact,
    consistent algorithmic tangent including geometric stiffness.

Volumetric locking (PSA nu=0.49) is controlled with the plane-strain F-bar
mean-dilatation  Fbar = F * sqrt(J0/J),  J0 = det F at the element centre —
the same device already used in the hyperelastic path of the solver.

State layout per GP (unchanged from the small-strain hybrid):
    [ e_dev(4) ,  q_i(4) for each Prony term i ]
with e_dev = [exx, eyy, ezz, exy(tensor)] the previous deviatoric strain.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_GP2 = jnp.array([
    [-1.0 / jnp.sqrt(3), -1.0 / jnp.sqrt(3)],
    [ 1.0 / jnp.sqrt(3), -1.0 / jnp.sqrt(3)],
    [ 1.0 / jnp.sqrt(3),  1.0 / jnp.sqrt(3)],
    [-1.0 / jnp.sqrt(3),  1.0 / jnp.sqrt(3)],
])
_W2 = jnp.ones(4)


def _sd(xi, eta):
    dN_dxi  = 0.25 * jnp.array([-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * jnp.array([-(1 - xi),  -(1 + xi),   (1 + xi),   (1 - xi)])
    return dN_dxi, dN_deta


def _jac(xi, eta, coords):
    dN_dxi, dN_deta = _sd(xi, eta)
    J = jnp.array([
        [jnp.dot(dN_dxi, coords[:, 0]), jnp.dot(dN_dxi, coords[:, 1])],
        [jnp.dot(dN_deta, coords[:, 0]), jnp.dot(dN_deta, coords[:, 1])],
    ])
    return J, jnp.linalg.det(J), jnp.linalg.inv(J)


def _grads(xi, eta, coords):
    """Physical reference shape-function gradients gX, gY (each (4,))."""
    _, detJ, invJ = _jac(xi, eta, coords)
    dN_dxi, dN_deta = _sd(xi, eta)
    gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    return gX, gY, detJ


def _F_at(gX, gY, u_elem):
    ux = u_elem[0::2]
    uy = u_elem[1::2]
    H = jnp.array([[ux @ gX, ux @ gY], [uy @ gX, uy @ gY]])
    return jnp.eye(2) + H


def _BL_columns(Ft, gX, gY):
    """Total-Lagrangian B_L (3x8):  delta E = sym(Ft^T delta F)."""
    F11, F12 = Ft[0, 0], Ft[0, 1]
    F21, F22 = Ft[1, 0], Ft[1, 1]
    B = jnp.zeros((3, 8))
    for a in range(4):
        gx, gy = gX[a], gY[a]
        B = B.at[0, 2 * a].set(F11 * gx)
        B = B.at[1, 2 * a].set(F12 * gy)
        B = B.at[2, 2 * a].set(F11 * gy + F12 * gx)
        B = B.at[0, 2 * a + 1].set(F21 * gx)
        B = B.at[1, 2 * a + 1].set(F22 * gy)
        B = B.at[2, 2 * a + 1].set(F21 * gy + F22 * gx)
    return B


def _prony_coeffs(dt, g_i, tau_i, G0):
    ratio = dt / jnp.maximum(tau_i, 1e-30)
    a = jnp.where(ratio < 1e-12, 1.0 - ratio, jnp.exp(-ratio))
    gamma = jnp.where(ratio < 1e-12, 1.0, (1.0 - a) / ratio)
    G_inf = G0 * (1.0 - jnp.sum(g_i))
    G_terms = G0 * g_i
    return a, gamma, G_inf, G_terms


def _internal_force(u_elem, coords, state_elem, K_bulk, g_i, tau_i, G0, dt, thickness):
    """Element internal force (8,) and updated state — finite-strain F-bar visco."""
    M = g_i.shape[0]
    a, gamma, G_inf, G_terms = _prony_coeffs(dt, g_i, tau_i, G0)

    # F-bar reference dilatation: J0 = det(F) at element centre
    gX0, gY0, _ = _grads(0.0, 0.0, coords)
    F0 = _F_at(gX0, gY0, u_elem)
    J0 = jnp.linalg.det(F0)

    f_int = jnp.zeros(8)
    state_new = jnp.empty_like(state_elem)

    for gp in range(4):
        xi, eta = _GP2[gp]
        gX, gY, detJ = _grads(xi, eta, coords)
        w = detJ * _W2[gp] * thickness

        F = _F_at(gX, gY, u_elem)
        J = jnp.linalg.det(F)
        # plane-strain F-bar mean dilatation
        Fbar = F * jnp.sqrt(jnp.maximum(J0 / J, 1e-12))

        # Green-Lagrange strain (Voigt: [Exx, Eyy, 2*Exy])
        E = 0.5 * (Fbar.T @ Fbar - jnp.eye(2))
        eps = jnp.array([E[0, 0], E[1, 1], 2.0 * E[0, 1]])
        theta = eps[0] + eps[1]

        # deviatoric tensorial strain (plane strain Ezz = 0)
        e_dev = jnp.array([eps[0] - theta / 3.0, eps[1] - theta / 3.0,
                           -theta / 3.0, 0.5 * eps[2]])
        e_prev = state_elem[gp, :4]
        q_prev_all = state_elem[gp, 4:].reshape(M, 4)
        de = e_dev - e_prev

        s_dev = 2.0 * G_inf * e_dev
        q_new_all = jnp.empty((M, 4), dtype=jnp.float64)
        for i in range(M):
            q_new = a[i] * q_prev_all[i] + 2.0 * G_terms[i] * gamma[i] * de
            s_dev = s_dev + q_new
            q_new_all = q_new_all.at[i].set(q_new)

        p = K_bulk * theta
        S_v = jnp.array([s_dev[0] + p, s_dev[1] + p, s_dev[3]])  # PK2 Voigt

        BL = _BL_columns(Fbar, gX, gY)
        f_int = f_int + BL.T @ S_v * w

        state_new = state_new.at[gp, :4].set(e_dev)
        for i in range(M):
            state_new = state_new.at[gp, 4 + 4 * i:4 + 4 * (i + 1)].set(q_new_all[i])

    return f_int, state_new


@jax.jit
def compute_single(coords, u_elem, state_elem, K_bulk, g_i, tau_i, G0, dt, thickness):
    """Finite-strain F-bar viscoelastic hybrid element.

    Returns (f_int(8,), K_e(8,8), state_new). The tangent is the exact
    consistent algorithmic tangent via autodiff (no eigendecomposition ->
    NaN-safe at any deformation, including F = I).
    """
    f_int, state_new = _internal_force(
        u_elem, coords, state_elem, K_bulk, g_i, tau_i, G0, dt, thickness)

    K_e = jax.jacobian(
        lambda u: _internal_force(
            u, coords, state_elem, K_bulk, g_i, tau_i, G0, dt, thickness)[0]
    )(u_elem)

    return f_int, K_e, state_new
