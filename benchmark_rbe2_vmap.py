"""JAX RBE2 vmap vs serial benchmark — run on CPU or GPU.

Usage:
    python benchmark_rbe2_vmap.py                  # CPU (default)
    CUDA_VISIBLE_DEVICES=0 python benchmark_rbe2_vmap.py  # GPU

Reports per-iteration time (ms) and speedup for batch sizes N in {4,8,16,32,64}.
"""

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import time
import numpy as np
import jax
import jax.numpy as jnp


def main():
    print(f"JAX version: {jax.__version__}")
    print(f"Default backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    from dispsolver.solver.dynamic_jax import (
        build_rbe2_vmap_contributions_jax,
        build_rbe2_element_contributions_jax,
    )

    rng = np.random.RandomState(42)
    fn_single = build_rbe2_element_contributions_jax()
    fn_vmap = build_rbe2_vmap_contributions_jax()

    NITER = 100
    print(f"{'N':>4} | {'serial (ms/iter)':>18} | {'vmap (ms/iter)':>16} | {'speedup':>8}")
    print("-" * 60)
    for N in (4, 8, 16, 32, 64):
        coords_b = rng.rand(N, 3, 2) * 5
        u_b = rng.randn(N, 6) * 0.01
        d0_b = rng.randn(N, 2, 2)
        lam_b = np.zeros((N, 4))
        theta_b = np.zeros(N)
        pen_b = np.full(N, 4e5)

        # Warmup (JIT compile)
        for i in range(N):
            fn_single(coords_b[i], u_b[i], d0_b[i], lam_b[i], theta_b[i], pen_b[i])
        fn_vmap(
            jnp.asarray(coords_b), jnp.asarray(u_b), jnp.asarray(d0_b),
            jnp.asarray(lam_b), jnp.asarray(theta_b), jnp.asarray(pen_b),
        )

        # Benchmark serial
        t0 = time.perf_counter()
        for _ in range(NITER):
            for i in range(N):
                fn_single(coords_b[i], u_b[i], d0_b[i], lam_b[i], theta_b[i], pen_b[i])
        t_serial = time.perf_counter() - t0

        # Benchmark vmap
        t0 = time.perf_counter()
        for _ in range(NITER):
            fn_vmap(
                jnp.asarray(coords_b), jnp.asarray(u_b), jnp.asarray(d0_b),
                jnp.asarray(lam_b), jnp.asarray(theta_b), jnp.asarray(pen_b),
            )
        t_vmap = time.perf_counter() - t0

        speedup = t_serial / t_vmap
        marker = " <-- vmap wins" if speedup > 1.0 else ""
        print(f"{N:>4} | {t_serial*1000/NITER:>18.3f} | {t_vmap*1000/NITER:>16.3f} | {speedup:>7.2f}x{marker}")


if __name__ == "__main__":
    main()
