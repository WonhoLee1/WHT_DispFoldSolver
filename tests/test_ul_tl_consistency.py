"""
tests/test_ul_tl_consistency.py
=================================
T2 (dev_log/plan_abaqus_element_consolidation_20260908.md's opus review,
finding F4 / fix): every Updated-Lagrangian-capable viscoelastic element
kernel must reach the SAME internal force whether a total deformation is
computed directly (TL: `F_n = I`, `u = u_total`) or incrementally (UL:
pre-deform the reference by `F_n`, then apply the remaining increment
`u_inc` such that `F_inc @ F_n` equals the same total F).

This is the single most important test this session's opus review
identified: it is the one that actually catches the missing PK2
push-forward bug (`S_n = F_n @ S0 @ F_n.T / det(F_n)`) that was silently
shipping in every viscoelastic UL kernel (`Q4_VISCO_SIMO`/`Q4_UP`,
`CPE4I`, `CPE4H`, `CPE4RH`) -- measured up to 24.9% relative force error
before the fix, on a homogeneous deformation chosen specifically so
F-bar and quadrature differences are identically zero and only the
conjugacy term survives. The existing rigid-rotation canary used
throughout the CPE4RH/CPE4I work this session CANNOT catch this bug (a
homogeneous zero-strain rotation makes the missing push-forward an exact
no-op too) -- this test is the one that actually gates it.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from dispsolver.element.q4_visco_hybrid_reduced_jax import compute_single_reduced_hybrid_jax
from dispsolver.element.q4_visco_eas_jax import compute_single_eas, compute_single_hybrid
from dispsolver.element.q4_visco_simo_fs_jax import compute_single as visco_simo_compute_single

_KAPPA = 8.3333
_BPARAMS = jnp.array([0.015614, 3.0])
_G_I = jnp.array([0.20])
_TAU_I = jnp.array([3.33])
_G_INF = 0.80
_DT = 0.5
_EYE2 = jnp.eye(2, dtype=jnp.float64)
_UNIT = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_F_N4_ID = jnp.stack([_EYE2] * 4)
_N_STATE = 6 * (1 + 1)  # M=1 Prony term -> 12


def _affine_u(coords, F0):
    H = F0 - jnp.eye(2, dtype=jnp.float64)
    u = jnp.zeros(8, dtype=jnp.float64)
    for a in range(4):
        d = H @ coords[a]
        u = u.at[2 * a].set(d[0])
        u = u.at[2 * a + 1].set(d[1])
    return u


# Two distinct homogeneous-deformation cases (anisotropic stretch and a
# rotation+stretch, matching the two magnitudes the opus review measured:
# 24.9% and 7.2% respectively, before the fix).
_CASES = [
    ("stretch_then_shear", jnp.array([[1.20, 0.0], [0.0, 0.90]]),
     jnp.array([[1.0, 0.03], [0.0, 1.0]])),
    ("rotation_then_stretch", jnp.array([[np.cos(1.2), -np.sin(1.2)],
                                         [np.sin(1.2), np.cos(1.2)]]),
     jnp.array([[1.05, 0.0], [0.0, 1.0]])),
]


def _tl_ul_states(F_n_case, F_extra):
    F_total = F_extra @ F_n_case
    u_tl = _affine_u(_UNIT, F_total)
    coords_n = jnp.stack([F_n_case @ _UNIT[a] for a in range(4)])
    u_inc = _affine_u(coords_n, F_extra)
    F_n4 = jnp.stack([F_n_case] * 4)
    return u_tl, coords_n, u_inc, F_n4


def _rel_err(f_tl, f_ul):
    f_tl = np.asarray(f_tl); f_ul = np.asarray(f_ul)
    return np.max(np.abs(f_tl - f_ul)) / max(np.max(np.abs(f_tl)), 1e-12)


@pytest.mark.parametrize("name,F_n_case,F_extra", _CASES)
def test_cpe4rh_ul_equals_tl(name, F_n_case, F_extra):
    u_tl, coords_n, u_inc, F_n4 = _tl_ul_states(F_n_case, F_extra)
    state = jnp.zeros((4, _N_STATE))
    f_tl, _, _, _ = compute_single_reduced_hybrid_jax(
        "arruda", _UNIT, u_tl, state, _KAPPA, _BPARAMS, _G_I, _TAU_I, _G_INF,
        _DT, 1.0, _F_N4_ID, 0.0)
    f_ul, _, _, _ = compute_single_reduced_hybrid_jax(
        "arruda", coords_n, u_inc, state, _KAPPA, _BPARAMS, _G_I, _TAU_I, _G_INF,
        _DT, 1.0, F_n4, 0.0)
    err = _rel_err(f_tl, f_ul)
    assert err < 1e-10, f"CPE4RH UL!=TL ({name}): rel err {err:.3e}"


@pytest.mark.parametrize("name,F_n_case,F_extra", _CASES)
def test_cpe4i_ul_equals_tl(name, F_n_case, F_extra):
    u_tl, coords_n, u_inc, F_n4 = _tl_ul_states(F_n_case, F_extra)
    state = jnp.zeros((4, _N_STATE))
    alpha0 = jnp.zeros(4)
    f_tl, _, _, _, _ = compute_single_eas(
        _UNIT, u_tl, alpha0, state, _KAPPA, _BPARAMS, _G_I, _TAU_I, _G_INF,
        _DT, 1.0, distortion_j_crit=0.0, F_n_gps=_F_N4_ID, base="arruda")
    f_ul, _, _, _, _ = compute_single_eas(
        coords_n, u_inc, alpha0, state, _KAPPA, _BPARAMS, _G_I, _TAU_I, _G_INF,
        _DT, 1.0, distortion_j_crit=0.0, F_n_gps=F_n4, base="arruda")
    err = _rel_err(f_tl, f_ul)
    assert err < 1e-10, f"CPE4I UL!=TL ({name}): rel err {err:.3e}"


@pytest.mark.parametrize("name,F_n_case,F_extra", _CASES)
def test_cpe4h_ul_equals_tl(name, F_n_case, F_extra):
    u_tl, coords_n, u_inc, F_n4 = _tl_ul_states(F_n_case, F_extra)
    state = jnp.zeros((4, _N_STATE))
    q0 = jnp.zeros(5)
    f_tl, _, _, _, _ = compute_single_hybrid(
        _UNIT, u_tl, q0, state, _KAPPA, _BPARAMS, _G_I, _TAU_I, _G_INF,
        _DT, 1.0, distortion_j_crit=0.0, F_n_gps=_F_N4_ID, base="arruda",
        use_eas=False)
    f_ul, _, _, _, _ = compute_single_hybrid(
        coords_n, u_inc, q0, state, _KAPPA, _BPARAMS, _G_I, _TAU_I, _G_INF,
        _DT, 1.0, distortion_j_crit=0.0, F_n_gps=F_n4, base="arruda",
        use_eas=False)
    err = _rel_err(f_tl, f_ul)
    assert err < 1e-10, f"CPE4H UL!=TL ({name}): rel err {err:.3e}"


@pytest.mark.parametrize("name,F_n_case,F_extra", _CASES)
def test_q4_visco_simo_ul_equals_tl(name, F_n_case, F_extra):
    u_tl, coords_n, u_inc, F_n4 = _tl_ul_states(F_n_case, F_extra)
    state = jnp.zeros((4, _N_STATE))
    f_tl, _, _, _ = visco_simo_compute_single(
        _UNIT, u_tl, state, _KAPPA, _BPARAMS, _G_I, _TAU_I, _G_INF, _DT, 1.0,
        distortion_j_crit=0.0, F_n_gps=_F_N4_ID, base="arruda")
    f_ul, _, _, _ = visco_simo_compute_single(
        coords_n, u_inc, state, _KAPPA, _BPARAMS, _G_I, _TAU_I, _G_INF, _DT, 1.0,
        distortion_j_crit=0.0, F_n_gps=F_n4, base="arruda")
    err = _rel_err(f_tl, f_ul)
    assert err < 1e-10, f"Q4_VISCO_SIMO UL!=TL ({name}): rel err {err:.3e}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
