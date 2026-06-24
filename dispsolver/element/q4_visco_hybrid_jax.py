"""Pure JAX Q1P0 hybrid element for linear viscoelasticity — no NumPy in-place ops."""

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
_P_VOL = 0.5 * jnp.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=jnp.float64)


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


def _B_mat(xi, eta, invJ):
    dN_dxi, dN_deta = _sd(xi, eta)
    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    B = jnp.zeros((3, 8))
    for i in range(4):
        col = 2 * i
        B = B.at[0, col].set(dN_dx[i])
        B = B.at[1, col + 1].set(dN_dy[i])
        B = B.at[2, col].set(dN_dy[i])
        B = B.at[2, col + 1].set(dN_dx[i])
    return B


def _B_bar(xi, eta, invJ, B0):
    B_std = _B_mat(xi, eta, invJ)
    return B_std - _P_VOL @ B_std + _P_VOL @ B0


def _prony_coeffs(dt, g_i, tau_i, G0):
    tau_eff = tau_i
    ratio = dt / jnp.maximum(tau_eff, 1e-30)
    a = jnp.where(ratio < 1e-12, 1.0 - ratio, jnp.exp(-ratio))
    gamma = jnp.where(ratio < 1e-12, 1.0, (1.0 - a) / ratio)
    G_inf = G0 * (1.0 - jnp.sum(g_i))
    G_terms = G0 * g_i
    G_alg = G_inf + jnp.sum(G_terms * gamma)
    return a, gamma, G_inf, G_terms, G_alg


def _dev_update(e_dev_new, e_prev, q_prev_all, a, G_inf, G_terms):
    de = e_dev_new - e_prev
    s_dev = 2.0 * G_inf * e_dev_new
    q_new_all = jnp.empty_like(q_prev_all)
    for i in range(len(a)):
        q_new = a[i] * q_prev_all[i] + 2.0 * G_terms[i] * (1.0 - a[i]) / jnp.maximum(dt_ratio(a[i]), 1e-30) * de
        s_dev = s_dev + q_new
        q_new_all = q_new_all.at[i].set(q_new)
    return s_dev, q_new_all


def dt_ratio(a):
    return -jnp.log(jnp.maximum(a, 1e-30))


@jax.jit
def compute_single(coords, u_elem, state_elem, K_bulk, g_i, tau_i, G0, dt, thickness):
    M = g_i.shape[0]

    a, gamma, G_inf, G_terms, G_alg = _prony_coeffs(dt, g_i, tau_i, G0)

    lam_eff = K_bulk - (2.0 / 3.0) * G_alg
    D = jnp.array([
        [lam_eff + 2.0 * G_alg, lam_eff, 0.0],
        [lam_eff, lam_eff + 2.0 * G_alg, 0.0],
        [0.0, 0.0, G_alg],
    ])

    _, _, invJ0 = _jac(0.0, 0.0, coords)
    B0 = _B_mat(0.0, 0.0, invJ0)

    f_int = jnp.zeros(8)
    K_e = jnp.zeros((8, 8))
    state_new = jnp.empty_like(state_elem)

    for gp in range(4):
        xi, eta = _GP2[gp]
        _, detJ, invJ = _jac(xi, eta, coords)
        w = detJ * _W2[gp] * thickness

        Bb = _B_bar(xi, eta, invJ, B0)
        eps = Bb @ u_elem
        theta = eps[0] + eps[1]

        e_dev = jnp.array([eps[0] - theta / 3.0, eps[1] - theta / 3.0, -theta / 3.0, 0.5 * eps[2]])
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
        sig = jnp.array([s_dev[0] + p, s_dev[1] + p, s_dev[3]])
        f_int = f_int + Bb.T @ sig * w
        K_e = K_e + (Bb.T @ D @ Bb) * w

        state_new = state_new.at[gp, :4].set(e_dev)
        for i in range(M):
            state_new = state_new.at[gp, 4 + 4 * i:4 + 4 * (i + 1)].set(q_new_all[i])

    return f_int, K_e, state_new
