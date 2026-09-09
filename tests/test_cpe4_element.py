"""
tests/test_cpe4_element.py
===========================
Verification for the new `dispsolver/element/cpe4_jax.py` -- the real
Abaqus CPE4 (full 2x2 integration + centroid-sampled volumetric/F-bar
treatment, finite strain, UL-capable, material-agnostic).

Held to the same standard as this session's CPE4RH work, plus the two
gates the opus reviews added:

1. Rigid-rotation canary (necessary, not sufficient -- kept as a
   standing guard).
2. **T1 global-axis isotropy**: rotate the WHOLE problem (reference
   geometry AND displacement) into a rotated frame; an isotropic
   implementation must give `f(rotated) == T8(theta) @ f(original)`.
   Measured this session: plain B-bar Q4 passes at 1e-16 while the
   home-grown SRI device (1e-2..5e-2) and the EAS mode set (5e-2..9e-2)
   do NOT -- so this gate has real discriminating power and CPE4 must
   pass it.
3. **T2 UL == TL consistency**: one total deformation reached directly
   (TL) vs incrementally (UL) must give the same internal force -- the
   gate that catches a missing PK2 push-forward (finding F4).
4. Constant-strain patch test.
5. **Material-agnosticism**: the identical element code must run with
   both `J2Plasticity` and `ViscoelasticMaterial` responses through the
   Phase A interface, with no element-side branching.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from dispsolver.element.cpe4_jax import compute_cpe4, compute_cpe4_material_tangent
from dispsolver.material.plastic import J2Plasticity
from dispsolver.material.response_interface import (
    J2Params, ViscoParamsTraced, j2_response_jax, j2_tangent_response_jax,
    make_visco_response,
)

_UNIT = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.15], [0.0, 0.15]])
_EYE2 = jnp.eye(2, dtype=jnp.float64)
_F_N4_ID = jnp.stack([_EYE2] * 4)

# Elastic-range J2 (huge yield) so these measure the ELEMENT.
_MAT = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=1e12, H=0.0)
_J2P = J2Params(lam=_MAT.lam, mu=_MAT.mu, sigma_y0=1e12, H=0.0)
_J2_STATE = jnp.tile(jnp.array([1.0, 0.0, 0.0, 1.0, 0.0]), (4, 1))  # F_p_inv = I

# Material-agnostic entry point: `base` is bound into the callable, so
# the element only ever traces arrays/floats (see make_visco_response).
_VISCO_RESPONSE = make_visco_response("arruda")
_VISCO_P = ViscoParamsTraced(kappa=8.3333,
                             bparams=jnp.array([0.015614, 3.0]),
                             g_i=jnp.array([0.20]), tau_i=jnp.array([3.33]),
                             g_inf=0.80, distortion_j_crit=0.0)
_VISCO_STATE = jnp.zeros((4, 12))   # M=1 -> 6*(M+1)


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _T8(theta):
    R = _rot(theta)
    T = np.zeros((8, 8))
    for i in range(4):
        T[2 * i:2 * i + 2, 2 * i:2 * i + 2] = R
    return T


def _affine_u(coords, F0):
    H = np.asarray(F0) - np.eye(2)
    u = np.zeros(8)
    for a in range(4):
        d = H @ np.asarray(coords)[a]
        u[2 * a], u[2 * a + 1] = d[0], d[1]
    return jnp.asarray(u)


def _f_j2(coords, u, F_n=_F_N4_ID):
    f, K, s, fn = compute_cpe4(j2_response_jax, coords, u, _J2_STATE, _J2P,
                               0.0, 1.0, F_n)
    return np.asarray(f), np.asarray(K)


def _f_visco(coords, u, F_n=_F_N4_ID):
    f, K, s, fn = compute_cpe4(_VISCO_RESPONSE, coords, u, _VISCO_STATE,
                               _VISCO_P, 0.5, 1.0, F_n)
    return np.asarray(f), np.asarray(K)


# ---------------------------------------------------------------------
# 1. Rigid-rotation canary (both materials)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("theta_deg", [0.001, 1.0, 30.0, 90.0, 135.0])
@pytest.mark.parametrize("kernel", [_f_j2, _f_visco], ids=["j2", "visco"])
def test_rigid_rotation_zero_force(kernel, theta_deg):
    R = jnp.asarray(_rot(np.deg2rad(theta_deg)))
    u = _affine_u(_UNIT, R)
    f, _ = kernel(_UNIT, u)
    assert np.all(np.isfinite(f))
    assert np.max(np.abs(f)) < 1e-9, (
        f"rigid rotation {theta_deg} deg gave |f|={np.max(np.abs(f)):.3e}")


# ---------------------------------------------------------------------
# 2. T1 global-axis isotropy (the gate SRI and EAS both fail)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("theta", [0.1, 0.5, 1.0, np.pi / 2])
@pytest.mark.parametrize("kernel", [_f_j2, _f_visco], ids=["j2", "visco"])
def test_global_axis_isotropy(kernel, theta):
    rng = np.random.default_rng(0)
    d = rng.uniform(-1.0, 1.0, size=8) * 4e-4
    f0, _ = kernel(_UNIT, jnp.asarray(d))

    R = _rot(theta)
    coords_rot = jnp.asarray(np.asarray(_UNIT) @ R.T)
    d_rot = jnp.asarray((d.reshape(4, 2) @ R.T).reshape(8))
    f_rot, _ = kernel(coords_rot, d_rot)

    err = np.linalg.norm(f_rot - _T8(theta) @ f0) / max(np.linalg.norm(f0), 1e-30)
    assert err < 1e-10, (
        f"CPE4 is not global-axis isotropic at theta={theta:.3f}: "
        f"rel err {err:.3e} (plain B-bar Q4 achieves ~1e-16 here)")


# ---------------------------------------------------------------------
# 3. T2 UL == TL consistency (catches a missing PK2 push-forward)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("kernel", [_f_j2, _f_visco], ids=["j2", "visco"])
def test_ul_equals_tl(kernel):
    F_n_case = np.array([[1.20, 0.0], [0.0, 0.90]])
    F_extra = np.array([[1.0, 0.03], [0.0, 1.0]])
    F_total = F_extra @ F_n_case

    u_tl = _affine_u(_UNIT, F_total)
    f_tl, _ = kernel(_UNIT, u_tl, _F_N4_ID)

    coords_n = jnp.asarray(np.asarray(_UNIT) @ F_n_case.T)
    u_inc = _affine_u(coords_n, F_extra)
    f_ul, _ = kernel(coords_n, u_inc, jnp.stack([jnp.asarray(F_n_case)] * 4))

    err = np.max(np.abs(f_tl - f_ul)) / max(np.max(np.abs(f_tl)), 1e-30)
    assert err < 1e-10, f"CPE4 UL != TL: rel err {err:.3e}"


# ---------------------------------------------------------------------
# 4. Constant-strain patch test (affine field is an exact equilibrium)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("F0", [
    np.array([[1.002, 0.0], [0.0, 0.999]]),
    np.array([[1.0, 0.002], [0.0, 1.0]]),
    np.array([[1.001, 0.0], [0.0, 1.001]]),
])
def test_constant_strain_patch(F0):
    """4 irregular quads sharing one interior node, all nodes driven by
    the same affine field -- the assembled interior residual must vanish."""
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.3, 0.9], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
    ])
    quads = [(0, 1, 4, 3), (1, 2, 5, 4), (4, 5, 8, 7), (3, 4, 7, 6)]
    interior = 4
    H = F0 - np.eye(2)
    u_glob = np.array([H @ nodes[n] for n in range(len(nodes))])

    resid = np.zeros(2)
    for q in quads:
        coords = jnp.asarray(nodes[list(q)])
        u_e = jnp.asarray(np.concatenate([u_glob[n] for n in q]))
        f, _ = _f_j2(coords, u_e)
        if interior in q:
            loc = q.index(interior)
            resid += f[2 * loc:2 * loc + 2]

    assert np.max(np.abs(resid)) < 1e-8, (
        f"patch test residual at the interior node: {resid}")


# ---------------------------------------------------------------------
# 5. Material-agnosticism: same element, material-supplied tangent path
# ---------------------------------------------------------------------
def test_material_tangent_variant_matches_autodiff_force():
    """`compute_cpe4_material_tangent` must produce the SAME internal
    force as the autodiff variant (only the tangent differs -- it drops
    the geometric term, the established modified-Newton precedent)."""
    F0 = np.array([[1.01, 0.004], [0.002, 0.995]])
    u = _affine_u(_UNIT, F0)
    f_ad, _ = _f_j2(_UNIT, u)
    f_mt, K_mt, _, _ = compute_cpe4_material_tangent(
        j2_tangent_response_jax, _UNIT, u, _J2_STATE, _J2P, 0.0, 1.0, _F_N4_ID)
    f_mt = np.asarray(f_mt)
    err = np.max(np.abs(f_ad - f_mt)) / max(np.max(np.abs(f_ad)), 1e-30)
    assert err < 1e-10, f"force mismatch between the two CPE4 entry points: {err:.3e}"
    assert np.all(np.isfinite(np.asarray(K_mt)))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
