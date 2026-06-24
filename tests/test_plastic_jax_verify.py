"""Verify JAX J2 plasticity matches the original NumPy implementation."""

import numpy as np
import jax
import jax.numpy as jnp

from dispsolver.material.plastic import J2Plasticity
from dispsolver.material.plastic_jax import pk2_voigt_jax, tangent_voigt_jax


def _numerical_tangent(F, state, lam, mu, sigma_y0, H):
    """Compute dS/dE_voigt via central finite differences."""
    from dispsolver.material.plastic_jax import pk2_voigt_jax as _pk2

    eps = 1e-7
    dE_tensors = np.array([
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.5], [0.5, 0.0]],
    ])
    Finv = np.linalg.inv(F)
    C_fd = np.zeros((3, 3))
    for j in range(3):
        dF = Finv.T @ dE_tensors[j]
        S_p = np.asarray(_pk2(jnp.asarray(F + eps * dF), jnp.asarray(state), lam, mu, sigma_y0, H)[0])
        S_m = np.asarray(_pk2(jnp.asarray(F - eps * dF), jnp.asarray(state), lam, mu, sigma_y0, H)[0])
        C_fd[:, j] = (S_p - S_m) / (2 * eps)
    return C_fd


def test_elastic_step():
    mat = J2Plasticity(E=200e3, nu=0.3, sigma_y0=400.0)
    F = np.array([[1.01, 0.002], [-0.001, 0.998]])
    state = np.zeros(5, dtype=np.float64)
    state[0] = 1.0
    state[3] = 1.0

    S_np, _C_np, sn_np = mat.pk2_voigt(F, {}, state)
    S_jax, C_jax, sn_jax = tangent_voigt_jax(
        jnp.asarray(F), jnp.asarray(state),
        mat.lam, mat.mu, mat.sigma_y0, mat.H,
    )
    S_jax = np.asarray(S_jax)
    C_jax = np.asarray(C_jax)
    sn_jax = np.asarray(sn_jax)

    C_fd = _numerical_tangent(F, state, mat.lam, mat.mu, mat.sigma_y0, mat.H)

    print(f"Elastic S  max diff: {np.max(np.abs(S_np - S_jax)):.2e}")
    print(f"Elastic state max diff: {np.max(np.abs(sn_np - sn_jax)):.2e}")
    print(f"Elastic C (autodiff vs FD) max diff: {np.max(np.abs(C_jax - C_fd)):.2e}")

    assert np.allclose(S_np, S_jax, atol=1e-8), f"S mismatch"
    assert np.allclose(sn_np, sn_jax, atol=1e-8), f"state mismatch"
    assert np.allclose(C_jax, C_fd, atol=1.0), f"C mismatch (autodiff vs FD)"
    print("PASSED")


def test_plastic_step():
    mat = J2Plasticity(E=200e3, nu=0.3, sigma_y0=400.0, H=1000.0)
    F = np.array([[1.05, 0.02], [-0.01, 0.97]])
    state = np.zeros(5, dtype=np.float64)
    state[0] = 1.0
    state[3] = 1.0

    S_np, C_np, sn_np = mat.pk2_voigt(F, {}, state)
    S_jax, C_jax, sn_jax = tangent_voigt_jax(
        jnp.asarray(F), jnp.asarray(state),
        mat.lam, mat.mu, mat.sigma_y0, mat.H,
    )
    S_jax = np.asarray(S_jax)
    C_jax = np.asarray(C_jax)
    sn_jax = np.asarray(sn_jax)

    print(f"Plastic S  max diff: {np.max(np.abs(S_np - S_jax)):.2e}")
    print(f"Plastic state max diff: {np.max(np.abs(sn_np - sn_jax)):.2e}")

    assert np.allclose(S_np, S_jax, atol=1e-6), f"S mismatch"
    assert np.allclose(sn_np, sn_jax, atol=1e-6), f"state mismatch"
    print("PASSED")


def test_plastic_incremental():
    mat = J2Plasticity(E=200e3, nu=0.3, sigma_y0=400.0, H=1000.0)
    state = np.zeros(5, dtype=np.float64)
    state[0] = 1.0
    state[3] = 1.0

    dF_steps = [
        np.array([[0.02, 0.005], [0.005, -0.01]]),
        np.array([[0.01, 0.003], [-0.002, -0.005]]),
        np.array([[0.015, 0.001], [0.003, -0.008]]),
    ]
    F = np.eye(2)

    for i, dF in enumerate(dF_steps):
        F = F + dF
        S_np, C_np, sn_np = mat.pk2_voigt(F, {}, state)
        S_jax, C_jax, sn_jax = tangent_voigt_jax(
            jnp.asarray(F), jnp.asarray(state),
            mat.lam, mat.mu, mat.sigma_y0, mat.H,
        )
        s_err = np.max(np.abs(S_np - np.asarray(S_jax)))
        st_err = np.max(np.abs(sn_np - np.asarray(sn_jax)))
        print(f"  Step {i}: S err={s_err:.2e}, state err={st_err:.2e}")
        assert s_err < 1e-5, f"S mismatch at step {i}"
        assert st_err < 1e-5, f"state mismatch at step {i}"
        state = sn_np
    print("PASSED")


def test_vmap_batch():
    mat = J2Plasticity(E=200e3, nu=0.3, sigma_y0=400.0, H=1000.0)
    np.random.seed(42)
    N = 16
    F_batch = np.eye(2)[None] + np.random.randn(N, 2, 2) * 0.03
    state_batch = np.zeros((N, 5), dtype=np.float64)
    state_batch[:, 0] = 1.0
    state_batch[:, 3] = 1.0

    S_np = np.zeros((N, 3))
    for e in range(N):
        S_np[e], _, _ = mat.pk2_voigt(F_batch[e], {}, state_batch[e])

    _vmap_pk2 = jax.vmap(
        lambda F, s: pk2_voigt_jax(F, s, mat.lam, mat.mu, mat.sigma_y0, mat.H),
    )
    S_jax, _ = _vmap_pk2(jnp.asarray(F_batch), jnp.asarray(state_batch))
    S_jax = np.asarray(S_jax)

    err = np.max(np.abs(S_np - S_jax))
    print(f"vmap batch S max diff: {err:.2e}")
    assert err < 1e-5
    print("PASSED")


if __name__ == "__main__":
    test_elastic_step()
    test_plastic_step()
    test_plastic_incremental()
    test_vmap_batch()
