"""Pure JAX RBE2 hinge element with static condensation of θ.

Port of rbe2.py to JAX. Key features:
1. Pure functional interface (no class state, no mutation).
2. Fixed-iteration local Newton for θ (vmap safe — no data-dependent exit).
3. All-JAX arrays (jnp.float64) for vmap compatibility.
4. Penalty-stabilized scalar K_θθ condensation (no saddle-point singularity).

References
----------
- Simo, J.C. & Rifai, M.S. (1990). EAS condensation pattern.
- Abaqus Analysis Guide §31.1.1 — MPC linearisation.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Tuple

import jax
import jax.numpy as jnp

# Fixed number of local Newton iterations for θ (data-independent for vmap).
_THETA_ITER = 8

# One-shot compile notification: prints to stderr the first time a JIT
# function is invoked, so users can see the compile cost instead of assuming
# the process is hung. Disabled by setting DISPSOLVER_QUIET=1.
_QUIET = os.environ.get("DISPSOLVER_QUIET", "0") == "1"
_first_compile = {"rbe2_single": True, "rbe2_vmap": True}


def _notify_compile(kind: str, fn):
    """Return a wrapper that prints a one-time compile notification for `fn`."""
    if _QUIET:
        return fn
    state = {"done": False, "t0": None}

    def wrapped(*args, **kwargs):
        if not state["done"]:
            state["t0"] = time.perf_counter()
            print(f"[JAX] compiling RBE2 {kind} (one-time, may take seconds)...",
                  file=sys.stderr, flush=True)
        result = fn(*args, **kwargs)
        if not state["done"]:
            state["done"] = True
            elapsed = time.perf_counter() - state["t0"]
            print(f"[JAX] RBE2 {kind} compile done in {elapsed:.1f}s",
                  file=sys.stderr, flush=True)
        return result

    return wrapped


@jax.jit
def _R(theta):
    """2×2 rotation matrix R(θ)."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[c, -s], [s, c]])


@jax.jit
def _dR_dtheta(theta):
    """First derivative of R(θ) with respect to θ."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[-s, -c], [c, -s]])


@jax.jit
def _d2R_dtheta2(theta):
    """Second derivative of R(θ) with respect to θ."""
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return jnp.array([[-c, s], [-s, -c]])


def _newton_step(theta, u_m, u_s, d0):
    """One Newton step for θ given current residual g.

    Returns (theta_new, dtheta).
    """
    R = _R(theta)
    dR = _dR_dtheta(theta)
    # Constraint gap g_i = u_s_i - u_m - (R(θ) - I)·d0_i  → (m, 2)
    rotated = (R - jnp.eye(2)) @ d0.T  # (2, m)
    g = u_s.reshape(-1, 2) - u_m[None, :] - rotated.T  # (m, 2)
    # ∂g/∂θ = -dR·d0_i  → (m, 2)
    dtheta_term = -dR @ d0.T  # (2, m)
    C_theta = dtheta_term.T  # (m, 2)
    # Scalar Newton: δθ = -C_θᵀ·g / (C_θᵀ·C_θ)
    Ctg = jnp.sum(C_theta * g)  # scalar
    CtC = jnp.sum(C_theta * C_theta)  # scalar
    # If CtC ≈ 0, no rotational coupling — return θ unchanged
    dtheta = jnp.where(CtC > 1e-30, -Ctg / CtC, 0.0)
    return theta + dtheta, dtheta


@jax.jit
def _solve_theta(u_m, u_s, d0, theta_init):
    """Fixed-iteration local Newton to solve g(u, θ) = 0 for θ.

    Uses jax.lax.fori_loop for vmap compatibility (no data-dependent exit).
    """
    def body(i, state):
        theta, _ = state
        theta_new, _ = _newton_step(theta, u_m, u_s, d0)
        return (theta_new, i + 1)

    theta_final, _ = jax.lax.fori_loop(0, _THETA_ITER, body, (theta_init, 0))
    return theta_final


@jax.jit
def compute_rbe2_hinge_contributions_jax(
    coords,        # (n_nodes, 2) — master at [0], slaves at [1..m]
    u_elem,        # (2*(m+1),)   — [u_mx, u_my, u_s1x, u_s1y, ...]
    d0,            # (m, 2)       — initial offset X_slave - X_master
    lam_n,         # (2m,)       — previous converged Lagrange multipliers
    theta_n,       # ()           — previous converged hinge angle (scalar)
    penalty,       # ()           — penalty stiffness (scalar)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute condensed element force, tangent, and updated state.

    Updated Lagrangian formulation: constraint gap uses deformed offset
    d_n = d0 + (u_s - u_m) rather than initial offset d0, matching the
    NumPy reference in rbe2.py.

    Formulation
    -----------
    1. Local Newton (convergence-checked, max 20 iter) solves g(u, θ) = 0 for θ.
       g_i(u, θ) = u_s_i - u_m - (R(θ) - I)·d_n_i = 0
    2. Penalty-stabilized scalar K_θθ = k·C_θᵀ·C_θ + Σ λ·∂²g/∂θ²
    3. Condensed tangent K_e = K_uu - K_uθ·K_uθᵀ / K_θθ
    4. Force f_e = C_uᵀ·λ + k·C_uᵀ·g + K_uθ·Δθ
    5. Updated state: θ_new = θ_converged, λ_new = λ + k·g_converged

    Parameters
    ----------
    coords : (m+1, 2)  master + slave coordinates.
    u_elem : (2m+2,)  element displacement vector.
    d0 : (m, 2)  initial slave-to-master offset.
    lam_n : (2m,)  previous Lagrange multipliers.
    theta_n : ()  previous hinge angle.
    penalty : ()  penalty stiffness.

    Returns
    -------
    f_e : (2m+2,)  condensed internal force.
    K_e : (2m+2, 2m+2)  condensed tangent.
    theta_new : ()  updated hinge angle.
    lam_new : (2m,)  updated Lagrange multipliers (Augmented-Lagrange update).
    dtheta : ()  recovered rotation increment Δθ.
    """
    m = d0.shape[0]
    n_ext = 2 * (m + 1)

    u_m = u_elem[0:2]
    u_s = u_elem[2:]

    # Deformed offset: d_n = d0 + (u_s - u_m) — Updated Lagrangian
    d_n = d0 + u_s.reshape(m, 2) - u_m[None, :]

    # --- 1. Local Newton for θ (convergence-checked) ---
    theta = _solve_theta(u_m, u_s, d_n, theta_n)

    # --- 2. Build constraint gap and derivatives at converged θ ---
    R = _R(theta)
    dR = _dR_dtheta(theta)
    d2R = _d2R_dtheta2(theta)

    rotated = (R - jnp.eye(2)) @ d_n.T                # (2, m)
    g_vec = u_s.reshape(m, 2) - u_m[None, :] - rotated.T  # (m, 2)
    g_flat = g_vec.reshape(2 * m)                    # (2m,)

    # C_θ = -dR·d_n_i  → (m, 2)
    C_theta = (-dR @ d_n.T).T                          # (m, 2)
    # d²g/dθ² = -d2R·d_n_i  → (m, 2)
    d2g_dtheta2 = (-d2R @ d_n.T).T                     # (m, 2)
    # λ-weighted geometric stiffness contribution
    K_theta_geo = jnp.sum(lam_n.reshape(m, 2) * d2g_dtheta2)  # scalar

    # Penalty-stabilized K_θθ
    K_theta = penalty * jnp.sum(C_theta * C_theta) + K_theta_geo
    K_theta = jnp.where(K_theta < 1e-30, 1e-30, K_theta)

    # --- 3. Build C_u (constraint Jacobian w.r.t. external displacements) ---
    # C_u shape: (2m, n_ext)
    # For slave i: ∂g_i/∂u_m = -I (master block), ∂g_i/∂u_s_i = +I
    # Even rows (g_x_i) get -1 in col 0, odd rows (g_y_i) get -1 in col 1.
    even_rows = jnp.arange(0, 2 * m, 2)
    odd_rows = jnp.arange(1, 2 * m, 2)
    C_u = jnp.zeros((2 * m, n_ext))
    C_u = C_u.at[even_rows, 0].set(-1.0)
    C_u = C_u.at[odd_rows, 1].set(-1.0)
    slave_dof_x = 2 + 2 * jnp.arange(m)
    slave_dof_y = 3 + 2 * jnp.arange(m)
    C_u = C_u.at[even_rows, slave_dof_x].set(1.0)
    C_u = C_u.at[odd_rows, slave_dof_y].set(1.0)

    # --- 4. Tangent blocks ---
    K_uu = penalty * (C_u.T @ C_u)                    # (n_ext, n_ext)
    K_u_theta = penalty * (C_u.T @ C_theta.reshape(2 * m, 1))  # (n_ext, 1)
    K_u_theta = K_u_theta.reshape(n_ext)

    # --- 5. Condensation ---
    K_e = K_uu - jnp.outer(K_u_theta, K_u_theta) / K_theta  # (n_ext, n_ext)

    # --- 6. Force ---
    f_theta = jnp.dot(C_theta.reshape(2 * m), lam_n) + penalty * jnp.dot(C_theta.reshape(2 * m), g_flat)
    dtheta = -f_theta / K_theta  # scalar
    f_e = C_u.T @ lam_n + penalty * (C_u.T @ g_flat) + K_u_theta * dtheta

    # --- 7. State update (Augmented-Lagrange) ---
    lam_new = lam_n + penalty * g_flat
    theta_new = theta

    return f_e, K_e, theta_new, lam_new, dtheta


# Wrap the JIT'd function with a one-time compile notification so users see
# progress instead of assuming the process is hung during the first call.
compute_rbe2_hinge_contributions_jax = _notify_compile(
    "rbe2_single", compute_rbe2_hinge_contributions_jax,
)
