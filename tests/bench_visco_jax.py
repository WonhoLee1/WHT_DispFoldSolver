"""Benchmark: NumPy sequential vs JAX vmap for Q4_UP viscoelastic assembly."""

import time
import numpy as np
import jax
import jax.numpy as jnp

from dispsolver.element.q4_visco_hybrid import compute_visco_hybrid_contributions
from dispsolver.element.q4_visco_hybrid_jax import compute_single as jax_compute_single
from dispsolver.material.linear_viscoelastic import LinearViscoelastic


def bench_numpy_sequential(coords_all, u_all, state_all, mat, dt, thickness_all, n_repeat=10):
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        for e in range(len(coords_all)):
            f, K, s = compute_visco_hybrid_contributions(
                coords_all[e], u_all[e], state_all[e], mat, dt, 20.0, thickness_all[e],
            )
        times.append(time.perf_counter() - t0)
    return np.median(times)


def bench_jax_vmap(coords_all, u_all, state_all, mat, dt, thickness_all, n_repeat=10):
    g_i = jnp.asarray(mat.g_i)
    tau_i = jnp.asarray(mat.tau_i)
    thickness_j = jnp.asarray(thickness_all)
    coords_j = jnp.asarray(coords_all)
    u_j = jnp.asarray(u_all)
    state_j = jnp.asarray(state_all)

    _vmap_fn = jax.vmap(
        jax_compute_single,
        in_axes=(0, 0, 0, None, None, None, None, None, 0),
    )

    for _ in range(3):
        _vmap_fn(coords_j, u_j, state_j, mat.K, g_i, tau_i, mat.G0, dt, thickness_j)

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        f_es, K_es, se_all = _vmap_fn(coords_j, u_j, state_j, mat.K, g_i, tau_i, mat.G0, dt, thickness_j)
        _ = np.asarray(f_es)
        times.append(time.perf_counter() - t0)
    return np.median(times)


def run_benchmark(n_elem):
    np.random.seed(42)
    mat = LinearViscoelastic(E=3e6, nu=0.45, g_i=[0.3, 0.2], tau_i=[1.0, 10.0])
    n_vars = mat.n_internal_vars

    coords_all = np.random.randn(n_elem, 4, 2) * 0.5 + 1.0
    u_all = np.random.randn(n_elem, 8) * 0.005
    state_all = np.random.randn(n_elem, 4, n_vars) * 0.001
    thickness_all = np.ones(n_elem)
    dt = 0.1

    t_np = bench_numpy_sequential(coords_all, u_all, state_all, mat, dt, thickness_all)
    t_jax = bench_jax_vmap(coords_all, u_all, state_all, mat, dt, thickness_all)

    speedup = t_np / t_jax
    print(f"  {n_elem:>6d} elements | NumPy: {t_np*1000:>8.2f} ms | JAX vmap: {t_jax*1000:>8.2f} ms | speedup: {speedup:.2f}x")
    return t_np, t_jax


if __name__ == "__main__":
    print("=" * 72)
    print("Q4_UP Viscoelastic Assembly: NumPy sequential vs JAX vmap")
    print("=" * 72)
    for n in [8, 32, 128, 512, 2048]:
        run_benchmark(n)
