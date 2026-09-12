"""
tests/test_rate_form_adapter.py
================================
Contracts for `dispsolver/material/rate_form_adapter.py`.

The adapter is **dormant** -- no material in this repository is rate-form,
so nothing exercises it in production. That is exactly why it needs its own
tests: a dormant objectivity layer with no tests is a layer nobody can
trust the day it is first used.

Two things are checked, and they are the two that matter for an objective
rate:

1. **Rigid-rotation objectivity.** Superposing a rigid body rotation on a
   deformed state must leave the PK2 stress (referred to the original
   configuration) unchanged. This is the property the whole
   `DeltaF = DeltaR DeltaU` construction exists to deliver, and it is
   checked incrementally -- rotating in many steps, which is where a
   naive stress update drifts.
2. **A hand-integrated reference.** Simple shear of a hypoelastic
   isotropic law under the Jaumann rate has the closed-form oscillatory
   solution of Dienes (1979):

       sigma_12 = mu * sin(gamma),   sigma_11 = -sigma_22 = mu * (1 - cos gamma)

   Reproducing an oscillation the law itself never mentions is a real
   check on the rate integration -- a wrong or missing spin treatment
   gives the monotone `mu*gamma` instead, which is off by 57% at
   `gamma = 2` and diverges after.

Plus the two closed-form kernels (`polar_rotation_2d`, `log_spd_2d`),
which are written in closed form specifically to avoid `jnp.linalg.eigh`'s
NaN-gradient-at-coincident-eigenvalues class (AGENTS.md 4.3) -- and
coincident eigenvalues are the NORMAL case here, since `DeltaU ~= I` for
any small increment.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dispsolver.material.rate_form_adapter import (
    adapter_state_size, hypoelastic_response, initial_state, log_spd_2d,
    make_rate_form_response, polar_rotation_2d,
)

E, NU = 200.0, 0.3
MU = E / (2.0 * (1.0 + NU))
LAM = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
PARAMS = (LAM, MU)

_RESPONSE_RAW = make_rate_form_response(hypoelastic_response, n_law_state=0)
# jit-wrapped: these tests integrate thousands of increments, and eager JAX
# dispatch of the polar decomposition + matrix log per step dominates
# everything else. Also proves the adapter is traceable, which is the only
# way an element would ever call it.
RESPONSE = jax.jit(lambda F, state, dt, params: _RESPONSE_RAW(F, state, dt, params),
                   static_argnums=(3,))
S0 = initial_state(0)


def _rot(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]])


# ----------------------------------------------------------------------
# Closed-form kernels
# ----------------------------------------------------------------------
@pytest.mark.parametrize("th", [0.0, 0.05, 0.7, 1.9, -2.6, np.pi - 1e-3])
def test_polar_rotation_recovers_the_rotation(th):
    """`A = R U` with `U` SPD must give back exactly `R`."""
    U = np.array([[1.3, 0.2], [0.2, 0.8]])          # SPD
    R = _rot(th)
    got = np.asarray(polar_rotation_2d(jnp.asarray(R @ U)))
    assert np.max(np.abs(got - R)) < 1e-13, f"polar rotation off at {th}"


def test_polar_rotation_of_identity_is_identity():
    got = np.asarray(polar_rotation_2d(jnp.eye(2)))
    assert np.max(np.abs(got - np.eye(2))) == 0.0


@pytest.mark.parametrize("eps", [0.0, 1e-12, 1e-8, 1e-5, 1e-2, 0.4])
def test_log_spd_matches_eigendecomposition_including_coincident(eps):
    """`log_spd_2d` vs an explicit eigendecomposition, across the removable
    singularity at coincident eigenvalues that it exists to survive."""
    U = np.array([[1.2 + eps, 0.0], [0.0, 1.2]])
    w, V = np.linalg.eigh(U)
    ref = V @ np.diag(np.log(w)) @ V.T
    got = np.asarray(log_spd_2d(jnp.asarray(U)))
    assert np.max(np.abs(got - ref)) < 1e-12, f"log_spd_2d off at eps={eps}"


def test_log_spd_gradient_is_finite_at_coincident_eigenvalues():
    """The whole reason for the closed form: `eigh` NaNs here (AGENTS.md 4.3)."""
    g = jax.grad(lambda a: log_spd_2d(jnp.array([[a, 0.0], [0.0, a]]))[0, 0])(1.0)
    assert np.isfinite(g), "gradient is not finite at U = a*I"
    assert abs(float(g) - 1.0) < 1e-9, "d/da ln(a) should be 1/a = 1 at a=1"


def test_log_spd_of_identity_is_zero():
    assert np.max(np.abs(np.asarray(log_spd_2d(jnp.eye(2))))) < 1e-15


# ----------------------------------------------------------------------
# Objectivity
# ----------------------------------------------------------------------
def test_superposed_rigid_rotation_leaves_pk2_unchanged():
    """PK2 on REF0 is invariant under a rotation superposed on the motion.

    Deformed first, then rotated in 60 increments -- incrementally, because
    a single large rotation step is the case a broken stress update most
    easily survives.
    """
    F0 = np.array([[1.15, 0.07], [-0.03, 0.92]])
    state = S0
    n_def = 20
    for k in range(1, n_def + 1):
        Fk = np.eye(2) + (k / n_def) * (F0 - np.eye(2))
        S_v, state = RESPONSE(jnp.asarray(Fk), state, 1.0, PARAMS)
    S_ref = np.asarray(S_v).copy()

    worst = 0.0
    n_rot = 60
    for k in range(1, n_rot + 1):
        S_v, state = RESPONSE(jnp.asarray(_rot(2.0 * np.pi * k / n_rot) @ F0),
                              state, 1.0, PARAMS)
        worst = max(worst, float(np.max(np.abs(np.asarray(S_v) - S_ref)))
                    / float(np.max(np.abs(S_ref))))
    assert worst < 1e-9, (
        f"PK2 drifts by {worst:.3e} under a superposed rigid rotation -- "
        f"the objective-rate treatment is not objective")


# ----------------------------------------------------------------------
# Hand-integrated reference: Dienes (1979) simple-shear oscillation
# ----------------------------------------------------------------------
def _simple_shear_cauchy(gamma_max: float, n_steps: int):
    """Integrate simple shear through the adapter; return Cauchy stress."""
    state = S0
    for k in range(1, n_steps + 1):
        g = gamma_max * k / n_steps
        F = jnp.array([[1.0, g], [0.0, 1.0]])
        _, state = RESPONSE(F, state, 1.0, PARAMS)
    return np.asarray(state[4:7])          # [s11, s22, s12], Cauchy


@pytest.mark.parametrize("gamma", [0.5, 2.0, 4.0])
def test_simple_shear_reproduces_the_jaumann_oscillation(gamma):
    """Dienes (1979): `s12 = mu sin(gamma)`, `s11 = -s22 = mu (1 - cos gamma)`.

    The signature result of the Jaumann rate -- the shear stress turns
    over and comes back down, and normal stresses appear in a pure shear
    that the constitutive law never mentions. A missing spin treatment
    gives the monotone `mu*gamma` instead (off by 57% at gamma=2).
    """
    s11, s22, s12 = _simple_shear_cauchy(gamma, 4000)
    ref12 = MU * np.sin(gamma)
    ref11 = MU * (1.0 - np.cos(gamma))
    scale = MU
    assert abs(s12 - ref12) / scale < 2e-3, f"s12 {s12:.5f} vs {ref12:.5f}"
    assert abs(s11 - ref11) / scale < 2e-3, f"s11 {s11:.5f} vs {ref11:.5f}"
    assert abs(s22 + ref11) / scale < 2e-3, f"s22 {s22:.5f} vs {-ref11:.5f}"


def test_simple_shear_converges_with_step_refinement():
    """Error must fall with the increment size -- a genuine integration, not
    a coincidence at one step count."""
    gamma = 3.0
    ref = MU * np.sin(gamma)
    errs = [abs(_simple_shear_cauchy(gamma, n)[2] - ref) / MU
            for n in (50, 200, 800)]
    assert errs[0] > errs[1] > errs[2], f"not converging: {errs}"
    assert errs[2] < 1e-2, f"finest step still off by {errs[2]:.3e}"


def test_shear_is_not_the_naive_monotone_answer():
    """Guard the guard: at gamma=2 the oscillation and the naive `mu*gamma`
    answer differ by 57%, so the test above has real discriminating power."""
    s12 = _simple_shear_cauchy(2.0, 4000)[2]
    naive = MU * 2.0
    assert abs(s12 - naive) / naive > 0.5


# ----------------------------------------------------------------------
# Interface plumbing
# ----------------------------------------------------------------------
def test_state_size_and_identity_start():
    assert adapter_state_size(5) == 12
    st = np.asarray(initial_state(5))
    assert st.shape == (12,)
    assert np.allclose(st[0:4], [1.0, 0.0, 0.0, 1.0]), "F_prev must start at I"
    assert np.allclose(st[4:], 0.0)


def test_undeformed_state_is_stress_free():
    S_v, _ = RESPONSE(jnp.eye(2), S0, 1.0, PARAMS)
    assert np.max(np.abs(np.asarray(S_v))) < 1e-14


def test_response_is_jittable():
    """It must survive `jax.jit` -- an element would only ever call it traced."""
    f = jax.jit(lambda F, s: RESPONSE(F, s, 1.0, PARAMS))
    S_v, _ = f(jnp.array([[1.01, 0.0], [0.0, 1.0]]), S0)
    assert np.all(np.isfinite(np.asarray(S_v)))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
