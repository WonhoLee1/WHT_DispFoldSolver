"""
tests/test_response_interface_phase_a.py
==========================================
Phase A verification (dev_log/plan_abaqus_element_consolidation_20260908.md):
`dispsolver/material/response_interface.py`'s wrappers must reproduce
their underlying existing functions BIT-FOR-BIT (this phase is pure
additive infrastructure, zero behavior change) at several representative
states: identity, rigid rotation, elastic stretch, and (for J2) a state
that actually yields plastically.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from dispsolver.material.plastic_jax import pk2_voigt_jax, tangent_voigt_jax
from dispsolver.material.response_interface import (
    J2Params, ViscoParams, j2_response_jax, j2_tangent_response_jax,
    visco_response_jax,
)
from dispsolver.element.q4_visco_simo_fs_jax import _simo_pk2

_EYE2 = jnp.eye(2, dtype=jnp.float64)


def _rot(theta_deg):
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.float64)


# J2 internal-variable state layout is [F_p_inv(4, flattened 2x2), eqps(1)]
# -- the INITIAL F_p_inv must be the identity matrix [1,0,0,1] (no prior
# plastic flow), not zeros (a zero F_p_inv is singular, which degenerates
# tangent_voigt_jax's forward-mode autodiff into NaN -- this is the exact
# eigenvalue-collision-under-differentiation class AGENTS.md 4.3 documents,
# triggered here by a wrong test fixture, not a bug in the function itself).
_J2_STATE0 = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0])

J2_STATES = [
    ("identity", _EYE2, _J2_STATE0),
    ("rotation_37deg", _rot(37.0), _J2_STATE0),
    ("elastic_stretch", jnp.array([[1.02, 0.0], [0.0, 0.99]]), _J2_STATE0),
    ("plastic_yield", jnp.array([[1.15, 0.05], [0.02, 0.9]]),
     jnp.array([1.0, 0.0, 0.0, 1.0, 0.01])),  # nonzero eqps -> exercises hardening
]


@pytest.mark.parametrize("name,F,state", J2_STATES)
def test_j2_response_matches_raw(name, F, state):
    lam, mu, sigma_y0, H = 5000.0, 2000.0, 80.0, 400.0
    params = J2Params(lam=lam, mu=mu, sigma_y0=sigma_y0, H=H)

    S_ref, state_ref = pk2_voigt_jax(F, state, lam, mu, sigma_y0, H)
    S_wrap, state_wrap = j2_response_jax(F, state, 0.0, params)

    assert np.array_equal(np.asarray(S_ref), np.asarray(S_wrap)), name
    assert np.array_equal(np.asarray(state_ref), np.asarray(state_wrap)), name


@pytest.mark.parametrize("name,F,state", J2_STATES)
def test_j2_tangent_response_matches_raw(name, F, state):
    lam, mu, sigma_y0, H = 5000.0, 2000.0, 80.0, 400.0
    params = J2Params(lam=lam, mu=mu, sigma_y0=sigma_y0, H=H)

    S_ref, C_ref, state_ref = tangent_voigt_jax(F, state, lam, mu, sigma_y0, H)
    S_wrap, C_wrap, state_wrap = j2_tangent_response_jax(F, state, 0.0, params)

    assert np.array_equal(np.asarray(S_ref), np.asarray(S_wrap)), name
    assert np.array_equal(np.asarray(C_ref), np.asarray(C_wrap)), name
    assert np.array_equal(np.asarray(state_ref), np.asarray(state_wrap)), name


VISCO_STATES = [
    ("identity", _EYE2),
    ("rotation_52deg", _rot(52.0)),
    ("uniaxial", jnp.array([[1.2, 0.0], [0.0, 0.9]])),
    ("shear", jnp.array([[1.0, 0.15], [0.0, 1.0]])),
]


@pytest.mark.parametrize("name,F", VISCO_STATES)
def test_visco_response_matches_raw(name, F):
    base = "arruda"
    kappa = 8.3333
    bparams = jnp.array([0.015614, 3.0])
    g_i = jnp.array([0.20])
    tau_i = jnp.array([3.33])
    g_inf = 0.80
    dt = 0.5
    n_state = 6 * (len(g_i) + 1)
    state = jnp.zeros(n_state)
    params = ViscoParams(base=base, kappa=kappa, bparams=bparams, g_i=g_i,
                         tau_i=tau_i, g_inf=g_inf, distortion_j_crit=0.0)

    S_ref, state_ref = _simo_pk2(base, F, state, kappa, bparams, g_i, tau_i,
                                 g_inf, dt, distortion_j_crit=0.0)
    S_wrap, state_wrap = visco_response_jax(F, state, dt, params)

    assert np.array_equal(np.asarray(S_ref), np.asarray(S_wrap)), name
    assert np.array_equal(np.asarray(state_ref), np.asarray(state_wrap)), name


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
