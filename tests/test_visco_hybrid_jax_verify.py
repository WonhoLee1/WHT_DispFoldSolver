"""Verify q4_visco_hybrid_jax matches the original NumPy implementation."""

import numpy as np
import jax
import jax.numpy as jnp

from dispsolver.element.q4_visco_hybrid import compute_visco_hybrid_contributions
from dispsolver.element.q4_visco_hybrid_jax import compute_single as jax_compute_single
from dispsolver.material.linear_viscoelastic import LinearViscoelastic


def test_single_element_match():
    np.random.seed(42)
    mat = LinearViscoelastic(E=3e6, nu=0.45, g_i=[0.3, 0.2], tau_i=[1.0, 10.0])

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    u_elem = np.array([0, 0, 0.01, 0, 0.01, 0.01, 0, 0.01], dtype=np.float64)
    n_vars = mat.n_internal_vars
    state_elem = np.random.randn(4, n_vars) * 0.001

    dt = 0.1
    temp = 20.0
    thickness = 1.0

    f_np, K_np, state_np = compute_visco_hybrid_contributions(
        coords, u_elem, state_elem, mat, dt, temp, thickness
    )

    g_i = jnp.asarray(mat.g_i)
    tau_i = jnp.asarray(mat.tau_i)
    f_jax, K_jax, state_jax = jax_compute_single(
        jnp.asarray(coords), jnp.asarray(u_elem), jnp.asarray(state_elem),
        mat.K, g_i, tau_i, mat.G0, dt, thickness,
    )
    f_jax = np.asarray(f_jax)
    K_jax = np.asarray(K_jax)
    state_jax = np.asarray(state_jax)

    print(f"f_int  max diff: {np.max(np.abs(f_np - f_jax)):.2e}")
    print(f"K_e    max diff: {np.max(np.abs(K_np - K_jax)):.2e}")
    print(f"state  max diff: {np.max(np.abs(state_np - state_jax)):.2e}")

    assert np.allclose(f_np, f_jax, atol=1e-10), f"f_int mismatch"
    assert np.allclose(K_np, K_jax, atol=1e-10), f"K_e mismatch"
    assert np.allclose(state_np, state_jax, atol=1e-10), f"state mismatch"
    print("PASSED")


def test_vmap_multiple_elements():
    np.random.seed(123)
    mat = LinearViscoelastic(E=3e6, nu=0.45, g_i=[0.3, 0.2], tau_i=[1.0, 10.0])
    n_elem = 8
    n_vars = mat.n_internal_vars

    all_coords = np.random.randn(n_elem, 4, 2) * 0.5 + 0.5
    all_u = np.random.randn(n_elem, 8) * 0.01
    all_state = np.random.randn(n_elem, 4, n_vars) * 0.001

    dt = 0.1
    temp = 20.0
    thickness = 1.0

    results_np = []
    for e in range(n_elem):
        f, K, s = compute_visco_hybrid_contributions(
            all_coords[e], all_u[e], all_state[e], mat, dt, temp, thickness
        )
        results_np.append((f, K, s))

    g_i = jnp.asarray(mat.g_i)
    tau_i = jnp.asarray(mat.tau_i)
    vmap_fn = jax.vmap(
        jax_compute_single,
        in_axes=(0, 0, 0, None, None, None, None, None, None),
    )
    f_batch, K_batch, state_batch = vmap_fn(
        jnp.asarray(all_coords), jnp.asarray(all_u), jnp.asarray(all_state),
        mat.K, g_i, tau_i, mat.G0, dt, thickness,
    )
    f_batch = np.asarray(f_batch)
    K_batch = np.asarray(K_batch)
    state_batch = np.asarray(state_batch)

    f_err = max(np.max(np.abs(f_batch[e] - results_np[e][0])) for e in range(n_elem))
    K_err = max(np.max(np.abs(K_batch[e] - results_np[e][1])) for e in range(n_elem))
    s_err = max(np.max(np.abs(state_batch[e] - results_np[e][2])) for e in range(n_elem))

    print(f"vmap f_int  max diff: {f_err:.2e}")
    print(f"vmap K_e    max diff: {K_err:.2e}")
    print(f"vmap state  max diff: {s_err:.2e}")

    assert f_err < 1e-8, f"vmap f_int mismatch"
    assert K_err < 1e-8, f"vmap K_e mismatch"
    assert s_err < 1e-8, f"vmap state mismatch"
    print("PASSED")


if __name__ == "__main__":
    test_single_element_match()
    test_vmap_multiple_elements()
