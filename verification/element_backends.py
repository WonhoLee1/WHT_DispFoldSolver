"""
element_backends.py
===================
Unified element API that wraps the NumPy and JAX element implementations
behind a common interface, so benchmarks can run the same problem on
multiple backends and compare results.

Backends
--------
- ``numpy_q4_bbar``  — Q4 B-bar (SRI) via dispsolver.element.q4 (pure NumPy)
- ``numpy_q4_eas``   — Q4 EAS-4 linear elastic via dispsolver.element.q4_eas
- ``jax_q4_bbar``    — Q4 B-bar via dispsolver.solver.dynamic_jax (JAX autodiff)
- ``jax_q4_up``      — Q1P0 hybrid via dispsolver.element.q4_up_jax (JAX autodiff)

Note: Numba is not used anywhere in the dispsolver codebase.  If a Numba
backend is added in the future, register it here with the same interface.

Each backend function has the signature:

    compute_K(coords: (4,2), E: float, nu: float) -> (8, 8) ndarray

and (optionally):

    compute_f_int(coords, u_elem, E, nu) -> (8,) ndarray
    compute_strain(coords, u_elem, E, nu) -> (3,) ndarray
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Tuple

# JAX must be in float64 mode
import jax
jax.config.update("jax_enable_x64", True)


# ------------------------------------------------------------------
# NumPy Q4 B-bar
# ------------------------------------------------------------------

def numpy_q4_bbar_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Q4 B-bar stiffness via pure NumPy 2×2 Gauss quadrature."""
    from dispsolver.element.q4 import compute_K_elem
    return compute_K_elem(coords, E, nu)


def numpy_q4_bbar_strain(coords: np.ndarray, u_elem: np.ndarray,
                         E: float, nu: float) -> np.ndarray:
    from dispsolver.element.q4 import compute_strains
    return compute_strains(coords, u_elem, xi=0.0, eta=0.0)


def numpy_q4_bbar_stress(coords: np.ndarray, u_elem: np.ndarray,
                         E: float, nu: float) -> np.ndarray:
    from dispsolver.element.q4 import compute_stress
    return compute_stress(coords, u_elem, E, nu, xi=0.0, eta=0.0)


# ------------------------------------------------------------------
# NumPy Q4 EAS (linear elastic, statically condensed)
# ------------------------------------------------------------------

def numpy_q4_eas_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Q4 EAS-4 condensed stiffness via pure NumPy."""
    from dispsolver.element.q4_eas import compute_eas_linear_K
    from .theory import plane_strain_D
    D = plane_strain_D(E, nu)
    return compute_eas_linear_K(coords, D)


# ------------------------------------------------------------------
# JAX Q4 B-bar (via dynamic_jax builder + NeoHookean)
# ------------------------------------------------------------------

_jax_q4_fn_cache: Dict[tuple, Callable] = {}


def _get_jax_q4_fn(E: float, nu: float):
    """Build (and cache) the JAX element contribution function for given E, ν."""
    key = (E, nu)
    if key not in _jax_q4_fn_cache:
        from dispsolver.material import NeoHookean
        from dispsolver.solver.dynamic_jax import build_element_contributions_jax
        mat = NeoHookean()
        params = {'E': float(E), 'nu': float(nu)}
        fn = build_element_contributions_jax(mat, params)
        _jax_q4_fn_cache[key] = jax.jit(fn)
    return _jax_q4_fn_cache[key]


def jax_q4_bbar_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Q4 B-bar stiffness via JAX (NeoHookean at small strain = linear elastic)."""
    import jax.numpy as jnp
    fn = _get_jax_q4_fn(E, nu)
    u_zero = jnp.zeros(8)
    _, K = fn(jnp.asarray(coords), u_zero)
    return np.asarray(K)


def jax_q4_bbar_f_int(coords: np.ndarray, u_elem: np.ndarray,
                      E: float, nu: float) -> np.ndarray:
    import jax.numpy as jnp
    fn = _get_jax_q4_fn(E, nu)
    f_int, _ = fn(jnp.asarray(coords), jnp.asarray(u_elem))
    return np.asarray(f_int)


# ------------------------------------------------------------------
# JAX Q4 UP (Q1P0 hybrid, NeoHookean)
# ------------------------------------------------------------------

_jax_up_fn_cache: Dict[tuple, Callable] = {}


def _get_jax_up_fn(E: float, nu: float):
    key = (E, nu)
    if key not in _jax_up_fn_cache:
        from dispsolver.element.q4_up_jax import compute_hybrid_element_contributions
        params = {'E': float(E), 'nu': float(nu)}
        _jax_up_fn_cache[key] = compute_hybrid_element_contributions
    return _jax_up_fn_cache[key]


def jax_q4_up_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Q1P0 hybrid stiffness via JAX (energy-based autodiff)."""
    import jax.numpy as jnp
    fn = _get_jax_up_fn(E, nu)
    params = {'E': float(E), 'nu': float(nu)}
    u_zero = jnp.zeros(8)
    _, K = fn(jnp.asarray(coords), u_zero, params)
    return np.asarray(K)


def jax_q4_up_f_int(coords: np.ndarray, u_elem: np.ndarray,
                    E: float, nu: float) -> np.ndarray:
    import jax.numpy as jnp
    fn = _get_jax_up_fn(E, nu)
    params = {'E': float(E), 'nu': float(nu)}
    f_int, _ = fn(jnp.asarray(coords), jnp.asarray(u_elem), params)
    return np.asarray(f_int)


# ------------------------------------------------------------------
# JAX Q4 EAS (J2 high-yield → linear elastic)
# ------------------------------------------------------------------

_jax_eas_fn_cache: Dict[tuple, Callable] = {}


def _get_jax_eas_fn(E: float, nu: float):
    """Build and cache the JAX EAS+J2 element function for given E, ν."""
    key = (E, nu)
    if key not in _jax_eas_fn_cache:
        from dispsolver.element.q4_eas_jax import compute_eas_j2_contributions_jax
        from dispsolver.material.plastic import J2Plasticity
        mat = J2Plasticity(E=E, nu=nu, sigma_y0=1e12, H=0.0)
        lam = float(mat.lam)
        mu = float(mat.mu)
        sigma_y0 = 1e12
        H_hard = 0.0
        _single = lambda coords, u_e, a, s: compute_eas_j2_contributions_jax(
            coords, u_e, a, s, lam, mu, sigma_y0, H_hard, 1.0,
        )
        _jax_eas_fn_cache[key] = jax.jit(_single)
    return _jax_eas_fn_cache[key]


def jax_q4_eas_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """JAX EAS-4 condensed stiffness (high-yield J2 = linear elastic)."""
    import jax.numpy as jnp
    fn = _get_jax_eas_fn(E, nu)
    u_zero = jnp.zeros(8)
    alpha_zero = jnp.zeros(4)
    state_init = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0])
    state_batch = jnp.broadcast_to(state_init, (4, 5))
    _, K, _, _, _ = fn(jnp.asarray(coords), u_zero, alpha_zero, state_batch)
    return np.asarray(K)


def jax_q4_eas_f_int(coords: np.ndarray, u_elem: np.ndarray,
                     E: float, nu: float) -> np.ndarray:
    import jax.numpy as jnp
    fn = _get_jax_eas_fn(E, nu)
    alpha_zero = jnp.zeros(4)
    state_init = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0])
    state_batch = jnp.broadcast_to(state_init, (4, 5))
    f_int, _, _, _, _ = fn(jnp.asarray(coords), jnp.asarray(u_elem), alpha_zero, state_batch)
    return np.asarray(f_int)


# ------------------------------------------------------------------
# JAX Q4 visco hybrid fs (ex03 PSA path, g_i=[0] → pure elastic)
# ------------------------------------------------------------------

_jax_visco_fs_fn_cache: Dict[tuple, Callable] = {}


def _get_jax_visco_fs_fn(E: float, nu: float):
    """Build and cache the JAX finite-strain visco hybrid element function."""
    import jax.numpy as jnp
    key = (E, nu)
    if key not in _jax_visco_fs_fn_cache:
        from dispsolver.element.q4_visco_hybrid_fs_jax import compute_single
        from dispsolver.material.linear_viscoelastic import LinearViscoelastic
        mat = LinearViscoelastic(E=E, nu=nu, g_i=[0.0], tau_i=[1.0])
        K_bulk = float(mat.K)
        g_i = jnp.asarray(mat.g_i)
        tau_i = jnp.asarray(mat.tau_i)
        G0 = float(mat.G0)
        _single = lambda coords, u_e, s, dt: compute_single(
            coords, u_e, s, K_bulk, g_i, tau_i, G0, dt, 1.0,
        )
        _jax_visco_fs_fn_cache[key] = jax.jit(_single)
    return _jax_visco_fs_fn_cache[key]


def jax_q4_visco_fs_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """JAX finite-strain visco hybrid F-bar stiffness (g_i=[0] → pure elastic)."""
    import jax.numpy as jnp
    fn = _get_jax_visco_fs_fn(E, nu)
    u_zero = jnp.zeros(8)
    M = 1
    n_state = 4 * (M + 1)
    state_batch = jnp.zeros((4, n_state))
    _, K, _ = fn(jnp.asarray(coords), u_zero, state_batch, 1.0)
    return np.asarray(K)


def jax_q4_visco_fs_f_int(coords: np.ndarray, u_elem: np.ndarray,
                          E: float, nu: float) -> np.ndarray:
    import jax.numpy as jnp
    fn = _get_jax_visco_fs_fn(E, nu)
    M = 1
    n_state = 4 * (M + 1)
    state_batch = jnp.zeros((4, n_state))
    f_int, _, _ = fn(jnp.asarray(coords), jnp.asarray(u_elem), state_batch, 1.0)
    return np.asarray(f_int)


# ------------------------------------------------------------------
# JAX Q4 Simo visco fs (pluggable base, g_i=[0] → pure NeoHookean elastic)
# ------------------------------------------------------------------

_jax_simo_fs_fn_cache: Dict[tuple, Callable] = {}


def _get_jax_simo_fs_fn(E: float, nu: float):
    """Build and cache the JAX Simo finite-strain visco element function."""
    import jax.numpy as jnp
    key = (E, nu)
    if key not in _jax_simo_fs_fn_cache:
        from functools import partial
        from dispsolver.element.q4_visco_simo_fs_jax import compute_single
        from dispsolver.material import NeoHookean
        from dispsolver.material.viscoelastic import ViscoelasticMaterial
        base_mat = NeoHookean()
        visco_mat = ViscoelasticMaterial(base_mat, g_i=[0.0], tau_i=[1.0])
        params = {'E': float(E), 'nu': float(nu)}
        base_name, bparams, kappa = visco_mat.simo_fs_args(params)
        g_i = jnp.asarray(visco_mat.g_i)
        tau_i = jnp.asarray(visco_mat.tau_i)
        g_inf = float(visco_mat.g_inf)
        _single = partial(compute_single, base=base_name)
        _jax_simo_fs_fn_cache[key] = (_single, float(kappa), jnp.asarray(bparams),
                                      g_i, tau_i, g_inf)
    return _jax_simo_fs_fn_cache[key]


def jax_q4_simo_fs_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """JAX Simo finite-strain visco F-bar stiffness (g_i=[0] → pure NeoHookean)."""
    import jax.numpy as jnp
    fn, kappa, bparams, g_i, tau_i, g_inf = _get_jax_simo_fs_fn(E, nu)
    u_zero = jnp.zeros(8)
    M = 1
    n_state = 6 * (M + 1)
    state_batch = jnp.zeros((4, n_state))
    _, K, _ = fn(jnp.asarray(coords), u_zero, state_batch,
                 kappa, bparams, g_i, tau_i, g_inf, 1.0, 1.0)
    return np.asarray(K)


def jax_q4_simo_fs_f_int(coords: np.ndarray, u_elem: np.ndarray,
                         E: float, nu: float) -> np.ndarray:
    import jax.numpy as jnp
    fn, kappa, bparams, g_i, tau_i, g_inf = _get_jax_simo_fs_fn(E, nu)
    M = 1
    n_state = 6 * (M + 1)
    state_batch = jnp.zeros((4, n_state))
    f_int, _, _ = fn(jnp.asarray(coords), jnp.asarray(u_elem), state_batch,
                     kappa, bparams, g_i, tau_i, g_inf, 1.0, 1.0)
    return np.asarray(f_int)


# ------------------------------------------------------------------
# Backend registry
# ------------------------------------------------------------------

BACKENDS: Dict[str, Dict[str, Callable]] = {
    'numpy_q4_bbar': {
        'name': 'NumPy Q4 B-bar',
        'compute_K': numpy_q4_bbar_K,
        'compute_strain': numpy_q4_bbar_strain,
        'compute_stress': numpy_q4_bbar_stress,
        'element_type': 'Q4',
    },
    'numpy_q4_eas': {
        'name': 'NumPy Q4 EAS-4',
        'compute_K': numpy_q4_eas_K,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4_EAS',
    },
    'jax_q4_bbar': {
        'name': 'JAX Q4 B-bar (NeoHookean)',
        'compute_K': jax_q4_bbar_K,
        'compute_f_int': jax_q4_bbar_f_int,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4',
    },
    'jax_q4_eas': {
        'name': 'JAX Q4 EAS-4 (J2 high-yield)',
        'compute_K': jax_q4_eas_K,
        'compute_f_int': jax_q4_eas_f_int,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4_EAS',
    },
    'jax_q4_up': {
        'name': 'JAX Q1P0 Hybrid (NeoHookean)',
        'compute_K': jax_q4_up_K,
        'compute_f_int': jax_q4_up_f_int,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4_UP',
    },
    'jax_q4_visco_fs': {
        'name': 'JAX Visco Hybrid F-bar (g_i=0, linear visco)',
        'compute_K': jax_q4_visco_fs_K,
        'compute_f_int': jax_q4_visco_fs_f_int,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4_UP',
    },
    'jax_q4_simo_fs': {
        'name': 'JAX Simo Visco F-bar (g_i=0, NeoHookean base)',
        'compute_K': jax_q4_simo_fs_K,
        'compute_f_int': jax_q4_simo_fs_f_int,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4_VISCO_SIMO',
    },
}


def list_backends() -> list:
    """Return available backend keys."""
    return list(BACKENDS.keys())


def get_backend(name: str) -> dict:
    if name not in BACKENDS:
        raise KeyError(f"Unknown backend {name!r}. Available: {list(BACKENDS)}")
    return BACKENDS[name]


# ------------------------------------------------------------------
# Solver-level backend configuration
# ------------------------------------------------------------------

def make_solver(mesh, E: float, nu: float, backend: str = 'jax',
                element_type: str = 'Q4', rho: float = 1e-6,
                **kwargs) -> object:
    """Create a DynamicSolver configured for the specified backend.

    Parameters
    ----------
    mesh : Mesh
    E, nu : float — material properties (NeoHookean small-strain ≡ linear elastic)
    backend : 'jax' | 'numpy' | 'numpy_sequential'
        - 'jax'            → fast_assembly=True (JAX vmap if available)
        - 'numpy'          → fast_assembly=False (J2 batch or sequential)
        - 'numpy_sequential' → force sequential fallback (monkey-patch)
    element_type : 'Q4' | 'Q4_EAS' | 'Q4_UP'
    rho : float — density (small to approximate static)
    """
    from dispsolver.material import NeoHookean, J2Plasticity
    from dispsolver.solver import DynamicSolver

    # Use NeoHookean for Q4/Q4_UP (pure JAX path), J2Plasticity (high yield)
    # for Q4_EAS (requires J2 for the JAX vmap path).
    if element_type == 'Q4_EAS':
        # J2Plasticity with very high yield stays in the elastic regime
        mat = J2Plasticity(E=E, nu=nu, sigma_y0=1e12, H=0.0)
        material_params = {}
    else:
        mat = NeoHookean()
        material_params = {'E': E, 'nu': nu}

    fast_assembly = (backend == 'jax')

    solver = DynamicSolver(
        mesh, mat, rho=rho, material_params=material_params,
        max_iter=kwargs.get('max_iter', 50),
        tol=kwargs.get('tol', 1e-8),
        atol=kwargs.get('atol', 1e-9),
        rtol=kwargs.get('rtol', 1e-6),
        verbose=kwargs.get('verbose', False),
        element_type=element_type,
        fast_assembly=fast_assembly,
        mode='quasistatic',
    )
    # The .sta status line (dynamic.py::_print_sta_line) is meant for
    # long folding runs, not this suite's many small per-benchmark
    # solves -- it just adds noise to run_all's output.
    solver.sta_status = False

    # Force the sequential NumPy path if requested
    if backend == 'numpy_sequential':
        solver.use_jax_vmap = False
        solver.use_j2_batch = False
        solver.use_multi_material_batch = False
        solver.use_jax_grouped_vmap = False

    return solver
