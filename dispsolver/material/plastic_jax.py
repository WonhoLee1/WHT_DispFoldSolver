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

    bad = (jnp.abs(jnp.linalg.det(F_3d)) < 1e-6) | ~jnp.all(jnp.isfinite(b_e_tr))
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

    # Smooth yield surface: tanh transition avoids the kink at q_tr == sigma_y
    # that makes the FD tangent inaccurate.  epsilon ~ 0.1% sigma_y keeps
    # behaviour indistinguishable from a hard cutoff.
    epsilon_s = jnp.maximum(1e-3 * sigma_y, 1e-30)
    xi = (q_tr - sigma_y) / epsilon_s
    alpha_s = 0.5 * (1.0 + jnp.tanh(xi))

    safe_q = jnp.where(q_tr > 1e-30, q_tr, 1.0)
    n_a = 1.5 * s_a / safe_q
    dgamma = alpha_s * (q_tr - sigma_y) / (3.0 * mu + H)

    s_a_new = s_a - 2.0 * mu * dgamma * n_a
    tau_a_plastic = s_a_new + p
    tau_a_final = alpha_s * tau_a_plastic + (1.0 - alpha_s) * tau_a

    eps_e_new = eps_log - dgamma * n_a
    lambda_new = jnp.exp(eps_e_new)
    lambda_tr = jnp.sqrt(lambda_sq)
    lambda_final = alpha_s * lambda_new + (1.0 - alpha_s) * lambda_tr

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
def tangent_voigt_jax(F_2d, state, lam, mu, sigma_y0, H):
    """PK2 stress + consistent material tangent C = dS/dE.

    Uses forward-mode autodiff (jacfwd) to compute the exact derivative of
    the algorithmic stress w.r.t. the deformation gradient, then projects
    to dS/dE via the kinematic relation dF = F^{-T} dE.

    A tiny randomness (1e-10) is added to F before jacfwd to break the
    eigenvalue degeneracy at b_e_tr = I (repeated eigenvalues) that would
    otherwise produce NaN eigenvector gradients.  The noise is ~1e-10
    relative to unit stretch, well below any material or tolerance scale.

    Compared with the prior central-FD implementation this is:
      - ~2× faster  (1 jacfwd pass vs 6 pk2 evaluations),
      - exact (no O(h²) truncation error),
      - equally robust (the noise is negligible).
    """
    S0, state_new = pk2_voigt_jax(F_2d, state, lam, mu, sigma_y0, H)

    # Inversion guard: singular F → isotropic tangent
    bad = jnp.abs(jnp.linalg.det(F_2d)) < 1e-8
    F_safe = jnp.where(bad, jnp.eye(2), F_2d)

    # Tiny noise to break repeated eigenvalues at b_e_tr = I
    key = jax.random.PRNGKey(0)
    noise = 1e-10 * jax.random.normal(key, F_safe.shape, dtype=jnp.float64)
    F_noisy = F_safe + noise

    # dS/dF via forward-mode autodiff (shape: (3,) w.r.t. (2,2) → (3, 2, 2))
    dS_dF = jax.jacfwd(lambda F: pk2_voigt_jax(F, state, lam, mu, sigma_y0, H)[0])(F_noisy)

    # Convert dS/dF to C = dS/dE in Voigt using the kinematic relation
    #   dF = F^{-T} dE   (holds for each Voigt direction of E)
    FinvT = jnp.linalg.inv(F_noisy).T
    dE_tensors = jnp.array([
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.5], [0.5, 0.0]],
    ])

    C = jnp.zeros((3, 3), dtype=jnp.float64)
    for j in range(3):
        dF_j = FinvT @ dE_tensors[j]                     # (2, 2)
        C = C.at[:, j].set(jnp.einsum('imn,mn->i', dS_dF, dF_j))

    C = 0.5 * (C + C.T)  # enforce minor symmetry
    C_iso = jnp.array([[lam + 2 * mu, lam, 0.0],
                       [lam, lam + 2 * mu, 0.0],
                       [0.0, 0.0, mu]], dtype=jnp.float64)
    C = jnp.where(bad, C_iso, C)
    return S0, C, state_new


@jax.jit
def pk2_with_tangent_jax(F_2d, state, lam, mu, sigma_y0, H):
    return tangent_voigt_jax(F_2d, state, lam, mu, sigma_y0, H)
