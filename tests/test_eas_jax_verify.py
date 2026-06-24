"""Verify JAX EAS element matches the NumPy implementation."""

import numpy as np
import jax
import jax.numpy as jnp

from dispsolver.material.plastic import J2Plasticity
from dispsolver.element.q4_eas import compute_eas_j2_contributions
from dispsolver.element.q4_eas_jax import compute_eas_j2_contributions_jax


def _make_state(E, nu, sigma_y0, H):
    mat = J2Plasticity(E=E, nu=nu, sigma_y0=sigma_y0, H=H)
    return mat


def test_elastic_force_and_tangent():
    """Single element, small elastic displacement — force & tangent match."""
    E, nu, sigma_y0, H = 200e3, 0.3, 400.0, 0.0
    mat = _make_state(E, nu, sigma_y0, H)
    lam, mu = mat.lam, mat.mu

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    u_elem = np.zeros(8, dtype=np.float64)
    u_elem[0] = 0.005
    u_elem[3] = 0.003

    state = np.zeros((4, 5), dtype=np.float64)
    state[:, 0] = 1.0
    state[:, 3] = 1.0
    alpha0 = np.zeros(4, dtype=np.float64)

    f_np, K_np, alpha_np, sn_np = compute_eas_j2_contributions(
        coords, u_elem, alpha0, state, mat, {},
    )
    f_jax, K_jax, alpha_jax, sn_jax = compute_eas_j2_contributions_jax(
        jnp.asarray(coords), jnp.asarray(u_elem), jnp.asarray(alpha0),
        jnp.asarray(state), lam, mu, sigma_y0, H,
    )
    f_jax = np.asarray(f_jax)
    K_jax = np.asarray(K_jax)
    alpha_jax = np.asarray(alpha_jax)
    sn_jax = np.asarray(sn_jax)

    f_err = np.max(np.abs(f_np - f_jax))
    K_err = np.max(np.abs(K_np - K_jax))
    a_err = np.max(np.abs(alpha_np - alpha_jax))
    s_err = np.max(np.abs(sn_np - sn_jax))

    print(f"Elastic force  max diff: {f_err:.2e}")
    print(f"Elastic K      max diff: {K_err:.2e}")
    print(f"Elastic alpha  max diff: {a_err:.2e}")
    print(f"Elastic state  max diff: {s_err:.2e}")

    K_rel = K_err / np.max(np.abs(K_np))
    print(f"Elastic K  rel diff:  {K_rel:.2e}  (NumPy=FD tangent, JAX=autodiff)")

    assert f_err < 1e-6, f"Force mismatch: {f_err}"
    # NumPy EAS uses FD tangent (h=1e-6), JAX uses exact autodiff — 40 ppm diff is expected
    assert K_rel < 1e-3, f"K relative mismatch: {K_rel}"
    assert s_err < 1e-10, f"State mismatch: {s_err}"
    print("PASSED")


def test_plastic_force():
    """Single element, plastic deformation — force matches."""
    E, nu, sigma_y0, H = 200e3, 0.3, 400.0, 1000.0
    mat = _make_state(E, nu, sigma_y0, H)
    lam, mu = mat.lam, mat.mu

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    u_elem = np.zeros(8, dtype=np.float64)
    u_elem[0] = 0.05
    u_elem[3] = -0.02
    u_elem[2] = 0.01
    u_elem[5] = 0.01

    state = np.zeros((4, 5), dtype=np.float64)
    state[:, 0] = 1.0
    state[:, 3] = 1.0
    alpha0 = np.zeros(4, dtype=np.float64)

    f_np, K_np, alpha_np, sn_np = compute_eas_j2_contributions(
        coords, u_elem, alpha0, state, mat, {},
    )
    f_jax, K_jax, alpha_jax, sn_jax = compute_eas_j2_contributions_jax(
        jnp.asarray(coords), jnp.asarray(u_elem), jnp.asarray(alpha0),
        jnp.asarray(state), lam, mu, sigma_y0, H,
    )
    f_jax = np.asarray(f_jax)
    K_jax = np.asarray(K_jax)
    sn_jax = np.asarray(sn_jax)

    f_err = np.max(np.abs(f_np - f_jax))
    K_err = np.max(np.abs(K_np - K_jax))
    s_err = np.max(np.abs(sn_np - sn_jax))

    print(f"Plastic force  max diff: {f_err:.2e}")
    print(f"Plastic K      max diff: {K_err:.2e}")
    print(f"Plastic state  max diff: {s_err:.2e}")

    # Force should match closely (both use same Newton path)
    assert f_err < 1.0, f"Force mismatch: {f_err}"
    assert s_err < 1e-4, f"State mismatch: {s_err}"
    print("PASSED")


def test_vmap_batch():
    """Batch of elements via vmap — no errors, finite output."""
    E, nu, sigma_y0, H = 200e3, 0.3, 400.0, 1000.0
    lam, mu = 115384.62, 76923.08

    np.random.seed(42)
    N = 4
    coords_batch = np.array([
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        [[0, 0], [2, 0], [2, 1], [0, 1]],
        [[0, 0], [1, 0], [1, 2], [0, 2]],
        [[0, 0], [1.5, 0], [1.5, 1.5], [0, 1.5]],
    ], dtype=np.float64)
    u_batch = np.random.randn(N, 8) * 0.01
    alpha_batch = np.zeros((N, 4), dtype=np.float64)
    state_batch = np.zeros((N, 4, 5), dtype=np.float64)
    state_batch[:, :, 0] = 1.0
    state_batch[:, :, 3] = 1.0

    _vmap = jax.vmap(
        lambda co, u, a, s: compute_eas_j2_contributions_jax(
            co, u, a, s, lam, mu, sigma_y0, H,
        ),
    )
    f_batch, K_batch, alpha_batch_out, sn_batch = _vmap(
        jnp.asarray(coords_batch),
        jnp.asarray(u_batch),
        jnp.asarray(alpha_batch),
        jnp.asarray(state_batch),
    )

    f_batch = np.asarray(f_batch)
    K_batch = np.asarray(K_batch)

    assert f_batch.shape == (N, 8), f"Force shape: {f_batch.shape}"
    assert K_batch.shape == (N, 8, 8), f"Tangent shape: {K_batch.shape}"
    assert np.all(np.isfinite(f_batch)), "Non-finite forces"
    assert np.all(np.isfinite(K_batch)), "Non-finite tangent"
    print(f"vmap batch: forces finite, shapes correct")
    print("PASSED")


if __name__ == "__main__":
    test_elastic_force_and_tangent()
    test_plastic_force()
    test_vmap_batch()
