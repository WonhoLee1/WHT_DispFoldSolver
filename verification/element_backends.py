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


def numpy_t3_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """T3 Constant Strain Triangle stiffness via pure NumPy."""
    from dispsolver.element.t3 import compute_K_elem
    return compute_K_elem(coords[:3], E, nu)


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
# Numba Q4 B-bar (LLVM JIT)
# ------------------------------------------------------------------

def numba_q4_bbar_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Q4 B-bar stiffness via Numba LLVM JIT 2x2 Gauss quadrature."""
    from dispsolver.element.q4_numba import compute_q4_bbar_element
    u_zero = np.zeros(8, dtype=np.float64)
    _, K = compute_q4_bbar_element(coords, u_zero, E, nu, thickness=1.0)
    return K


def numba_q4_bbar_f_int(coords: np.ndarray, u_elem: np.ndarray,
                        E: float, nu: float) -> np.ndarray:
    from dispsolver.element.q4_numba import compute_q4_bbar_element
    f_int, _ = compute_q4_bbar_element(coords, u_elem, E, nu, thickness=1.0)
    return f_int


def numba_q4_eas_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Q4 EAS-4 stiffness via Numba LLVM JIT 4-mode static condensation."""
    from dispsolver.element.q4_numba import compute_q4_eas_element
    u_zero = np.zeros(8, dtype=np.float64)
    _, K = compute_q4_eas_element(coords, u_zero, E, nu, thickness=1.0)
    return K


def numba_q4_up_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Q1P0 hybrid stiffness via Numba LLVM JIT."""
    from dispsolver.element.q4_numba import compute_q4_up_element
    u_zero = np.zeros(8, dtype=np.float64)
    _, K = compute_q4_up_element(coords, u_zero, E, nu, thickness=1.0)
    return K


def numba_q4_up_f_int(coords: np.ndarray, u_elem: np.ndarray,
                      E: float, nu: float) -> np.ndarray:
    from dispsolver.element.q4_numba import compute_q4_up_element
    f_int, _ = compute_q4_up_element(coords, u_elem, E, nu, thickness=1.0)
    return f_int


def numba_q4_corotational_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Co-rotational Q4 stiffness via Numba LLVM JIT."""
    from dispsolver.element.q4_numba import compute_q4_corotational_element
    u_zero = np.zeros(8, dtype=np.float64)
    _, K = compute_q4_corotational_element(coords, u_zero, E, nu, thickness=1.0)
    return K


def numba_t3_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """T3 Constant Strain Triangle stiffness via Numba LLVM JIT."""
    from dispsolver.element.q4_numba import compute_t3_element
    u_zero = np.zeros(6, dtype=np.float64)
    coords_3 = coords[:3]
    _, K = compute_t3_element(coords_3, u_zero, E, nu, thickness=1.0)
    return K


def jax_t3_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """T3 Constant Strain Triangle stiffness via JAX energy-based autodiff."""
    import jax
    import jax.numpy as jnp
    coords_3 = jnp.asarray(coords[:3], dtype=jnp.float64)

    # Linear plane strain D matrix
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = c * jnp.array([
        [1.0 - nu, nu, 0.0],
        [nu, 1.0 - nu, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]
    ], dtype=jnp.float64)

    def t3_energy(u_e):
        dN_dxi = jnp.array([-1.0, 1.0, 0.0], dtype=jnp.float64)
        dN_deta = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float64)
        J00 = jnp.sum(dN_dxi * coords_3[:, 0])
        J01 = jnp.sum(dN_dxi * coords_3[:, 1])
        J10 = jnp.sum(dN_deta * coords_3[:, 0])
        J11 = jnp.sum(dN_deta * coords_3[:, 1])
        detJ = J00 * J11 - J01 * J10
        area = 0.5 * jnp.abs(detJ)
        invJ = jnp.array([[J11, -J01], [-J10, J00]], dtype=jnp.float64) / detJ

        gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

        ux = u_e[0::2]
        uy = u_e[1::2]
        exx = jnp.sum(ux * gX)
        eyy = jnp.sum(uy * gY)
        gxy = jnp.sum(ux * gY) + jnp.sum(uy * gX)
        strain = jnp.array([exx, eyy, gxy], dtype=jnp.float64)

        W = 0.5 * jnp.dot(strain, jnp.dot(D, strain))
        return W * area

    u_zero = jnp.zeros(6, dtype=jnp.float64)
    K = jax.hessian(t3_energy)(u_zero)
    return np.asarray(K)


def numpy_q4_corotational_eas_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """CR-EAS condensed stiffness via pure NumPy."""
    from dispsolver.element.q4_corotational_eas import compute_corotational_eas_j2_contributions
    from dispsolver.material import J2Plasticity
    mat = J2Plasticity(E=E, nu=nu, sigma_y0=1e9, H=0.0)
    params = {'E': float(E), 'nu': float(nu), 'sigma_y0': 1e9, 'H': 0.0}
    u_zero = np.zeros(8, dtype=np.float64)
    alpha_zero = np.zeros(4, dtype=np.float64)
    state_zero = np.tile(mat.initial_internal_vars(), (4, 1))
    _, K, _, _ = compute_corotational_eas_j2_contributions(coords, u_zero, alpha_zero, state_zero, mat, params)
    return K


def jax_q4_corotational_eas_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """CR-EAS condensed stiffness via JAX."""
    from dispsolver.element.q4_corotational_eas_jax import compute_corotational_eas_j2_contributions_jax
    import jax.numpy as jnp
    u_zero = jnp.zeros(8, dtype=jnp.float64)
    alpha_zero = jnp.zeros(4, dtype=jnp.float64)
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    state_zero = jnp.tile(jnp.array([1.0, 0.0, 0.0, 1.0, 0.0]), (4, 1))
    _, K, _, _, _ = compute_corotational_eas_j2_contributions_jax(
        jnp.asarray(coords), u_zero, alpha_zero, state_zero, lam, mu, 1e9, 0.0
    )
    return np.asarray(K)


def numba_q4_corotational_eas_K(coords: np.ndarray, E: float, nu: float) -> np.ndarray:
    """CR-EAS condensed stiffness via Numba."""
    from dispsolver.element.q4_numba import compute_q4_corotational_eas_element
    u_zero = np.zeros(8, dtype=np.float64)
    _, K = compute_q4_corotational_eas_element(coords, u_zero, E, nu)
    return K


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
    'numpy_q4_corotational_eas': {
        'name': 'NumPy Co-rotational EAS (CR-EAS)',
        'compute_K': numpy_q4_corotational_eas_K,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4_COROTATIONAL_EAS',
    },
    'numpy_t3': {
        'name': 'NumPy T3 Triangle',
        'compute_K': numpy_t3_K,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'T3',
    },
    'numba_q4_bbar': {
        'name': 'Numba Q4 B-bar (LLVM JIT)',
        'compute_K': numba_q4_bbar_K,
        'compute_f_int': numba_q4_bbar_f_int,
        'compute_strain': numpy_q4_bbar_strain,
        'compute_stress': numpy_q4_bbar_stress,
        'element_type': 'Q4',
    },
    'numba_q4_eas': {
        'name': 'Numba Q4 EAS-4 (LLVM JIT)',
        'compute_K': numba_q4_eas_K,
        'compute_f_int': None,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4_EAS',
    },
    'numba_q4_corotational_eas': {
        'name': 'Numba Co-rotational EAS (LLVM JIT)',
        'compute_K': numba_q4_corotational_eas_K,
        'compute_f_int': None,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4_COROTATIONAL_EAS',
    },
    'numba_q4_up': {
        'name': 'Numba Q1P0 Hybrid (LLVM JIT)',
        'compute_K': numba_q4_up_K,
        'compute_f_int': numba_q4_up_f_int,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4_UP',
    },
    'numba_q4_corotational': {
        'name': 'Numba Co-rotational Q4 (LLVM JIT)',
        'compute_K': numba_q4_corotational_K,
        'compute_f_int': None,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4',
    },
    'numba_t3': {
        'name': 'Numba T3 Triangle (LLVM JIT)',
        'compute_K': numba_t3_K,
        'compute_f_int': None,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'T3',
    },
    'jax_t3': {
        'name': 'JAX T3 Triangle (NeoHookean)',
        'compute_K': jax_t3_K,
        'compute_f_int': None,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'T3',
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
    'jax_q4_corotational_eas': {
        'name': 'JAX Co-rotational EAS (CR-EAS)',
        'compute_K': jax_q4_corotational_eas_K,
        'compute_f_int': None,
        'compute_strain': None,
        'compute_stress': None,
        'element_type': 'Q4_COROTATIONAL_EAS',
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
    # Q4_COROTATIONAL requires BOTH material and element_type as dicts
    # (see Finding 1 in AGENTS.md — plain string routes to Q4 B-bar silently).
    if element_type in ('Q4_COROTATIONAL', 'Q4_COROTATIONAL_EAS'):
        if backend == 'numpy_sequential':
            raise ValueError(
                f"{element_type} has no working numpy_sequential path. "
                "Use backend='jax' or 'numba'."
            )
        mat = {0: J2Plasticity(E=E, nu=nu, sigma_y0=1e12, H=0.0)}
        element_type_arg = {0: element_type}
        material_params = {}
    elif element_type == 'Q4_EAS':
        # J2Plasticity with very high yield stays in the elastic regime
        mat = J2Plasticity(E=E, nu=nu, sigma_y0=1e12, H=0.0)
        material_params = {}
    else:
        mat = NeoHookean()
        material_params = {'E': E, 'nu': nu}

    fast_assembly = (backend == 'jax')

    # For dict-form element_type, override element_type to the dict version
    if isinstance(element_type_arg, dict) if 'element_type_arg' in locals() else False:
        element_type = element_type_arg

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
