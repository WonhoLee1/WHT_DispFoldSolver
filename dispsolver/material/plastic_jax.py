"""Pure JAX J2 plasticity — differentiable return mapping with autodiff tangent."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _embed_3d(F_2d):
    F3 = jnp.eye(3, dtype=jnp.float64)
    return F3.at[:2, :2].set(F_2d)


@jax.jit
def pk2_voigt_jax(F_2d, state, lam, mu, sigma_y0, H):
    F_3d = _embed_3d(F_2d)
    J_F = jnp.linalg.det(F_3d)

    F_p_inv_2d = state[:4].reshape(2, 2)
    det_Fp_inv = F_p_inv_2d[0, 0] * F_p_inv_2d[1, 1] - F_p_inv_2d[0, 1] * F_p_inv_2d[1, 0]
    F_p_inv_3d = jnp.eye(3, dtype=jnp.float64).at[:2, :2].set(F_p_inv_2d).at[2, 2].set(1.0 / jnp.maximum(jnp.abs(det_Fp_inv), 1e-30))

    F_e_tr = F_3d @ F_p_inv_3d
    b_e_tr = F_e_tr @ F_e_tr.T

    bad = (jnp.linalg.det(F_3d) < 1e-6) | ~jnp.all(jnp.isfinite(b_e_tr))
    b_e_tr_safe = jnp.where(bad, jnp.eye(3), b_e_tr)

    w, v = jnp.linalg.eigh(b_e_tr_safe)
    lambda_sq = jnp.maximum(w, 1e-30)
    eps_log = 0.5 * jnp.log(lambda_sq)

    tr_eps = jnp.sum(eps_log)
    tau_a = lam * tr_eps + 2.0 * mu * eps_log
    p = jnp.mean(tau_a)
    s_a = tau_a - p
    s_norm = jnp.sqrt(jnp.sum(s_a ** 2))
    q_tr = jnp.sqrt(1.5) * s_norm

    eqps = state[4]
    sigma_y = sigma_y0 + H * eqps
    is_plastic = q_tr > sigma_y + 1e-12

    safe_q = jnp.where(q_tr > 1e-30, q_tr, 1.0)
    n_a = 1.5 * s_a / safe_q
    dgamma = jnp.where(is_plastic, (q_tr - sigma_y) / (3.0 * mu + H), 0.0)

    s_a_new = s_a - 2.0 * mu * dgamma * n_a
    tau_a_plastic = s_a_new + p
    tau_a_final = jnp.where(is_plastic, tau_a_plastic, tau_a)

    eps_e_new = eps_log - dgamma * n_a
    lambda_new = jnp.exp(eps_e_new)
    lambda_tr = jnp.sqrt(lambda_sq)
    lambda_final = jnp.where(is_plastic, lambda_new, lambda_tr)

    tau_tensor = jnp.einsum('a,ia,ja->ij', tau_a_final, v, v)
    F3_safe = jnp.where(bad, jnp.eye(3), F_3d)
    F3_inv = jnp.linalg.inv(F3_safe)
    S3 = F3_inv @ tau_tensor @ F3_inv.T
    S_voigt = jnp.array([S3[0, 0], S3[1, 1], S3[0, 1]])

    lambda_tr_safe = jnp.maximum(lambda_tr, 1e-30)
    V_e_tr_inv = jnp.einsum('a,ia,ja->ij', 1.0 / lambda_tr_safe, v, v)
    R_e_tr = V_e_tr_inv @ F_e_tr
    V_e_new = jnp.einsum('a,ia,ja->ij', lambda_final, v, v)
    F_e_new = V_e_new @ R_e_tr
    F_p_inv_new = jnp.linalg.solve(F3_safe, F_e_new)
    eqps_new = eqps + dgamma

    state_new = state.at[:4].set(F_p_inv_new[:2, :2].ravel())
    state_new = state_new.at[4].set(eqps_new)

    return S_voigt, state_new


@jax.jit
def tangent_voigt_jax(F_2d, state, lam, mu, sigma_y0, H, h=1e-6):
    """PK2 stress + consistent material tangent C = dS/dE.

    The tangent is built by *central finite differences* of the (finite,
    well-defined) stress along the three Green-Lagrange Voigt directions:

        dF = F^{-T} dE_tensor   =>   sym(F^T dF) = dE_tensor.

    Autodiff through the spectral decomposition (jnp.linalg.eigh) cannot be
    used here: at repeated eigenvalues — which occur at the undeformed state
    (b_e = I, the ex03 initial condition) and under any isotropic stretch —
    the eigenvector derivative carries 1/(lambda_i - lambda_j) terms and the
    gradient becomes NaN. The stress itself stays finite at those states, so
    FD of the stress yields a finite, consistent tangent everywhere.
    """
    S0, state_new = pk2_voigt_jax(F_2d, state, lam, mu, sigma_y0, H)

    # Inversion guard: at a degenerate / inverted trial F (det <= 0, which the
    # global line search probes), inv(F) blows up and the FD tangent becomes
    # NaN. Replace it with a finite isotropic tangent (matching pk2_voigt_jax,
    # whose `bad` branch already returns ~zero stress there) so the line search
    # can back off instead of stalling on NaN.
    bad = jnp.linalg.det(F_2d) <= 1e-8
    F_safe = jnp.where(bad, jnp.eye(2), F_2d)

    FinvT = jnp.linalg.inv(F_safe).T
    dE_tensors = jnp.array([
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.5], [0.5, 0.0]],
    ])

    C = jnp.zeros((3, 3), dtype=jnp.float64)
    for j in range(3):
        dF = FinvT @ dE_tensors[j]
        Sp, _ = pk2_voigt_jax(F_safe + h * dF, state, lam, mu, sigma_y0, H)
        Sm, _ = pk2_voigt_jax(F_safe - h * dF, state, lam, mu, sigma_y0, H)
        C = C.at[:, j].set((Sp - Sm) / (2.0 * h))

    C = 0.5 * (C + C.T)  # symmetrise (minor symmetry)
    C_iso = jnp.array([[lam + 2 * mu, lam, 0.0],
                       [lam, lam + 2 * mu, 0.0],
                       [0.0, 0.0, mu]], dtype=jnp.float64)
    C = jnp.where(bad, C_iso, C)
    return S0, C, state_new


@jax.jit
def pk2_with_tangent_jax(F_2d, state, lam, mu, sigma_y0, H):
    return tangent_voigt_jax(F_2d, state, lam, mu, sigma_y0, H)
