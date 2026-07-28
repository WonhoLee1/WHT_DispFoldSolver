"""
speed_bench.py
==============
Performance/benchmarking utilities: times ``DynamicSolver.solve_step()``
for different mesh sizes and solver backends.

All results are **informational only** — no PASS/FAIL gating.
Invoked via ``python -m verification.run_all --include-speed``.
"""

from __future__ import annotations

import time
import numpy as np
from typing import Dict, List, Tuple, Callable

from .mesh_utils import build_beam_mesh
from .element_backends import make_solver


SOLVER_KWARGS = dict(
    max_iter=50, tol=1e-4, atol=1e-7, rtol=5e-3,
    force_atol=1e-4, max_du_atol=1e-6,
)
N_LOAD_STEPS = 10


def _time_solver_repeated(
    mesh,
    E: float,
    nu: float,
    backend: str,
    element_type: str,
    bc_dofs: List[int],
    bc_vals: List[float],
    load_dofs: List[int],
    load_vals: List[float],
    n_repeats: int = 5,
    n_warmup: int = 2,
) -> dict:
    """Time a single ``solve_step()`` call across repeated runs.

    Builds ONE solver so JAX's shape-keyed JIT cache is populated once.
    Per repeat: resets ``u``/``v``/``a``/``time`` to zero, re-applies
    BCs and loads, times ``solve_step(dt=1.0)`` with ``time.perf_counter()``.

    Only valid for path-independent materials (NeoHookean via Q4 B-bar).
    """
    solver = make_solver(mesh, E, nu, backend=backend,
                         element_type=element_type, **SOLVER_KWARGS)
    solver.sta_status = False  # suppress .sta line noise

    times = []
    n_iter_last = 0

    for i in range(n_warmup + n_repeats):
        solver.u[:] = 0.0
        solver.v[:] = 0.0
        solver.a[:] = 0.0
        solver.time = 0.0
        solver.set_prescribed_dofs(bc_dofs, bc_vals)
        solver.apply_load(load_dofs, load_vals)

        t0 = time.perf_counter()
        n_iter = solver.solve_step(dt=1.0)
        elapsed = time.perf_counter() - t0

        if i >= n_warmup:
            times.append(elapsed)
            n_iter_last = n_iter if n_iter > 0 else n_iter_last

    times_arr = np.array(times)
    return {
        'mean_s': float(np.mean(times_arr)),
        'median_s': float(np.median(times_arr)),
        'std_s': float(np.std(times_arr, ddof=1)),
        'min_s': float(np.min(times_arr)),
        'max_s': float(np.max(times_arr)),
        'n_iter': int(n_iter_last),
        'n_repeats': int(n_repeats),
        'n_warmup': int(n_warmup),
    }


def _cantilever_BCs(mesh, info: dict, L: float, P: float):
    """Build BC/load arrays for the cantilever benchmark.

    Returns (bc_dofs, bc_vals, load_dofs, load_vals).
    """
    nid_to_idx = info['nid_to_idx']
    nx = info['nx']
    ny = info['ny']

    bc_dofs: List[int] = []
    bc_vals: List[float] = []
    for nid in info['left']:
        idx = nid_to_idx[nid]
        bc_dofs.extend([idx * 2, idx * 2 + 1])
        bc_vals.extend([0.0, 0.0])

    top_j = ny - 1
    tip_nid = top_j * nx + (nx - 1)
    load_dofs = [nid_to_idx[tip_nid] * 2 + 1]
    load_vals = [-P]

    return bc_dofs, bc_vals, load_dofs, load_vals


def speed_benchmark_cantilever(
    mesh_sizes: Tuple[Tuple[int, int], ...] = ((10, 4), (40, 4), (80, 8)),
    backends: Tuple[str, ...] = ('jax', 'numpy_sequential'),
    n_repeats: int = 3,
    n_warmup: int = 2,
    E: float = 1000.0,
    nu: float = 0.3,
    L: float = 10.0,
    H: float = 1.0,
    P: float = 0.001,
) -> dict:
    """Run the speed benchmark over multiple mesh sizes and backends.

    Reuses the existing cantilever geometry/BCs (same as the
    ``cantilever`` benchmark in ``benchmarks.py``).  This is purely a
    timing measurement, not a physical test — no theory comparison.

    Returns a nested dict with speed data per mesh size and backend.
    """
    results = {
        'name': 'Speed Benchmark (Cantilever)',
        'description': (
            'Times solve_step() for cantilever beam (L=10, H=1, P=0.001) '
            'across mesh densities and solver backends. Informational only.'
        ),
        'mesh_sizes': {},
    }

    for nx, ny in mesh_sizes:
        mesh, info = build_beam_mesh(nx, ny, length=L, height=H)
        n_elem = (nx - 1) * (ny - 1)
        bc_dofs, bc_vals, load_dofs, load_vals = _cantilever_BCs(
            mesh, info, L, P
        )

        backend_data = {}
        for bk in backends:
            timing = _time_solver_repeated(
                mesh, E, nu, backend=bk, element_type='Q4',
                bc_dofs=bc_dofs, bc_vals=bc_vals,
                load_dofs=load_dofs, load_vals=load_vals,
                n_repeats=n_repeats, n_warmup=n_warmup,
            )
            backend_data[bk] = timing

        speedup = float('nan')
        if 'jax' in backend_data and 'numpy_sequential' in backend_data:
            t_jax = backend_data['jax']['mean_s']
            t_numpy = backend_data['numpy_sequential']['mean_s']
            if t_jax > 0 and t_numpy > 0:
                speedup = t_numpy / t_jax

        results['mesh_sizes'][f'({nx}×{ny})'] = {
            'n_elements': n_elem,
            'backends': backend_data,
            'speedup_jax_over_numpy': float(speedup),
        }

    return results


def generate_speed_report(results: dict) -> str:
    """Generate a markdown performance report (no PASS/FAIL column)."""
    lines = []
    lines.append("## Performance Report (Opt-In)")
    lines.append("")
    lines.append("**Benchmark**: " + results.get('name', '?'))
    lines.append("")
    lines.append(
        "**Note**: This data is informational only - no PASS/FAIL gating."
    )
    lines.append("")

    for ms_key, ms_data in results.get('mesh_sizes', {}).items():
        n_elem = ms_data.get('n_elements', '?')
        lines.append(f"### Mesh: {ms_key} - {n_elem} elements")
        lines.append("")
        lines.append(
            "| Backend | Mean (s) | Median (s) | Std (s) | "
            "Min (s) | Max (s) | N Iter |"
        )
        lines.append(
            "|---------|----------|-------------|---------|---------|---------|--------|"
        )

        for bk, timing in ms_data.get('backends', {}).items():
            lines.append(
                f"| {bk} | {timing['mean_s']:.6e} | {timing['median_s']:.6e} | "
                f"{timing['std_s']:.6e} | {timing['min_s']:.6e} | "
                f"{timing['max_s']:.6e} | {timing['n_iter']} |"
            )

        speedup = ms_data.get('speedup_jax_over_numpy', float('nan'))
        if not np.isnan(speedup):
            lines.append("")
            lines.append(
                f"**Speedup (JAX over NumPy sequential)**: {speedup:.2f}×"
            )
        lines.append("")

    return "\n".join(lines)


def element_assembly_microbenchmark(
    n_elements: int = 1000,
    n_repeats: int = 10,
    n_warmup: int = 3,
) -> dict:
    """Microbenchmark raw element assembly time: NumPy vs Numba vs JAX.

    Assembles N elements repeatedly to isolate element stiffness & force
    evaluation time from the linear solver time.
    """
    from .mesh_utils import build_beam_mesh
    mesh, info = build_beam_mesh(nx=100, ny=10, length=20.0, height=2.0)
    coords = np.array([[node.x, node.y] for node in mesh.nodes.values()])

    # 1. NumPy Q4 B-bar
    from dispsolver.element.q4 import compute_K_elem
    single_quad = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    elem_coords = np.array([single_quad for _ in range(n_elements)])
    u_elems = np.zeros((n_elements, 8), dtype=np.float64)

    # Warmup + timing for NumPy
    for _ in range(n_warmup):
        for e in range(n_elements):
            _K = compute_K_elem(elem_coords[e], 1000.0, 0.3)

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        for e in range(n_elements):
            _K = compute_K_elem(elem_coords[e], 1000.0, 0.3)
    t_numpy = (time.perf_counter() - t0) / n_repeats

    # 2. Numba Q4 B-bar
    t_numba = float('nan')
    try:
        from dispsolver.element.q4_numba import assemble_q4_bbar_batch_numba
        thicknesses = np.ones(n_elements, dtype=np.float64)
        for _ in range(n_warmup):
            assemble_q4_bbar_batch_numba(elem_coords, u_elems, 1000.0, 0.3, thicknesses)
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            assemble_q4_bbar_batch_numba(elem_coords, u_elems, 1000.0, 0.3, thicknesses)
        t_numba = (time.perf_counter() - t0) / n_repeats
    except Exception:
        pass

    # 3. JAX Q4 B-bar
    t_jax = float('nan')
    try:
        import jax
        import jax.numpy as jnp
        from dispsolver.solver.dynamic_jax import build_element_contributions_jax
        from dispsolver.material import NeoHookean
        mat = NeoHookean()
        params = {'E': 1000.0, 'nu': 0.3}
        fn = jax.jit(jax.vmap(build_element_contributions_jax(mat, params)))
        elem_coords_j = jnp.asarray(elem_coords)
        u_elems_j = jnp.asarray(u_elems)

        for _ in range(n_warmup):
            f_j, K_j = fn(elem_coords_j, u_elems_j)
            f_j.block_until_ready()

        t0 = time.perf_counter()
        for _ in range(n_repeats):
            f_j, K_j = fn(elem_coords_j, u_elems_j)
            f_j.block_until_ready()
        t_jax = (time.perf_counter() - t0) / n_repeats
    except Exception:
        pass

    return {
        'n_elements': n_elements,
        'numpy_mean_s': t_numpy,
        'numba_mean_s': t_numba,
        'jax_mean_s': t_jax,
        'speedup_numba_over_numpy': t_numpy / t_numba if not np.isnan(t_numba) else float('nan'),
        'speedup_jax_over_numpy': t_numpy / t_jax if not np.isnan(t_jax) else float('nan'),
        'speedup_jax_over_numba': t_numba / t_jax if not np.isnan(t_numba) and not np.isnan(t_jax) else float('nan'),
    }


if __name__ == "__main__":
    # Quick smoke test
    res = speed_benchmark_cantilever(
        mesh_sizes=((10, 4),), backends=('jax',), n_repeats=2, n_warmup=1
    )
    print(generate_speed_report(res))
    micro = element_assembly_microbenchmark(n_elements=500, n_repeats=5, n_warmup=2)
    print("Microbenchmark:", micro)

