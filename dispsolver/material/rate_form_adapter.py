"""
dispsolver/material/rate_form_adapter.py
=========================================
Wraps a **rate-form** constitutive law so it satisfies the total-form
`response_fn` contract of `response_interface.py`. Dormant: no material in
this repository is rate-form today, and none of the element work depends on
it. It ships now, with its own tests, so that the day one arrives the
objectivity machinery is already written, verified, and -- crucially -- on
the **material** side of the interface where it belongs.

Why the material side and not the element side
----------------------------------------------
Abaqus's per-integration-point `DeltaF = DeltaR DeltaU` polar machinery
exists for exactly one purpose: to make rate-form constitutive equations
objective. Abaqus says so in as many words:

    "Objective rates are relevant only for rate form constitutive
     equations (e.g., elastoplasticity). **For hyperelastic materials a
     total formulation is used; hence, the concept of an objective rate is
     not relevant for the constitutive law.**"
                                        -- Abaqus 2016 Theory Guide 1.5.3

Every material here is total-form: `plastic_jax.pk2_voigt_jax` is
finite-strain *multiplicative* J2 taking the total `F` through a spectral
return map; `_simo_pk2` likewise takes total `F` and returns PK2. They are
the DFGRD1-consuming branch of the UMAT contract and Abaqus would not hand
them an objective rate either. So putting a polar-decomposition
"frame handler" between the element and the material would add a device
that neither our materials nor Abaqus's treatment of them requires -- and
it would put material-specific machinery in the one layer
(`element/kinematics/frame.py`) whose entire value is being material-blind.

**Jaumann, not Green-Naghdi.** This project's prior notes assumed
Green-Naghdi. For Abaqus/Standard *solid (continuum)* elements the objective
rate is **Jaumann** for all materials (Theory Guide 1.5.3, Table 1.5.3-1);
Green-Naghdi is used for *structural* elements and for Abaqus/Explicit
VUMAT / viscoelastic / brittle-cracking. The difference is real: Jaumann's
spin is `skew(L)`, which for an increment is exactly the polar rotation of
`DeltaF`; Green-Naghdi's is defined against `R` from the **total** `F`.
The algorithm below is the Jaumann one, realised the way Abaqus realises
it -- by rotating the stored stress with `DeltaR` rather than by adding a
spin term (Hughes & Winget 1980).

The algorithm (Abaqus Theory Guide 1.4.3 / 1.5.4, at the material side)
-----------------------------------------------------------------------
Given the TOTAL `F` (what the total-form interface supplies) and the
previous step's `F` (carried in this adapter's own state):

  1. `DeltaF = F @ inv(F_prev)`
  2. polar decompose `DeltaF = DeltaR @ DeltaU`   (closed form, 2x2)
  3. `Deps = ln(DeltaU)`                          -- the logarithmic strain
     increment, which is what Abaqus hands constitutive routines
     ("STRAN ... are approximations to logarithmic strain", SUB 1.1.44)
  4. rotate the stored Cauchy stress: `sigma <- DeltaR sigma DeltaR^T`
  5. call the rate law with `(Deps, sigma_rotated, state, dt, DeltaR)`
  6. pull the returned Cauchy stress back to PK2 on REF0:
     `S = J F^-1 sigma F^-T`, which is what the response interface promises
     its callers and what `kinematics/frame.py` then pushes forward.

Step 4 is the objective rate. Step 5 passes `DeltaR` through **because
Abaqus does**: built-in material state is rotated by Abaqus, but user state
is not -- "any vector-valued or tensor-valued state variables must be
rotated ... The rotation increment matrix, DROT, is provided for this
purpose" (SUB 1.1.44, STATEV). The adapter owns the in-plane stress and
rotates it; anything else tensor-valued that the law keeps in `state` --
plane strain's `sigma_33`, a back stress, a material director -- is the
law's to rotate, with the `DeltaR` it is given.

No eigendecomposition anywhere
------------------------------
Both the polar decomposition and the matrix logarithm are closed-form for
2x2. That is deliberate, not an optimisation: `jnp.linalg.eigh` produces
**NaN reverse-mode gradients whenever two eigenvalues coincide**
(AGENTS.md 4.3), and `DeltaU ~= I` -- coincident eigenvalues -- is the
*normal* case for a small increment, not an edge case. An eigendecomposed
version of this file would NaN on almost every step it was asked to
differentiate through.

References
----------
* Hughes, T.J.R. & Winget, J. (1980). Finite rotation effects in numerical
  integration of rate constitutive equations arising in large-deformation
  analysis. Int. J. Numer. Meth. Engng 15(12), 1862-1867.
* Dienes, J.K. (1979). On the analysis of rotation and stress rate in
  deforming bodies. Acta Mechanica 32, 217-232. -- the simple-shear
  oscillation used as this module's hand-integrated reference.
* Abaqus 2016 Theory Guide 1.4.3, 1.5.3 (Table 1.5.3-1), 1.5.4;
  Abaqus 2016 User Subroutines 1.1.44 (UMAT: STRESS, STRAN, DROT, STATEV).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

# State layout this adapter owns, ahead of the wrapped law's own state:
#   [0:4]  F_prev, row-major (2x2)
#   [4:7]  Cauchy stress in the CURRENT configuration, [s11, s22, s12]
_N_ADAPTER_STATE = 7


def adapter_state_size(n_law_state: int) -> int:
    """Total per-Gauss-point state length for a wrapped rate-form law."""
    return _N_ADAPTER_STATE + int(n_law_state)


def initial_state(n_law_state: int, law_initial=None):
    """Initial state vector: `F_prev = I`, zero stress, then the law's own."""
    head = jnp.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    tail = (jnp.zeros(int(n_law_state)) if law_initial is None
            else jnp.asarray(law_initial))
    return jnp.concatenate([head, tail])


def polar_rotation_2d(A):
    """`R` from `A = R U`, closed form, no eigendecomposition.

    Requiring `U = R^T A` to be symmetric gives, for `A = [[a,b],[c,d]]`,
    `cos(theta)(b - c) = -sin(theta)(a + d)`, i.e.

        theta = atan2(c - b, a + d)

    Exact for any `A` with `det A > 0`, and smooth there -- `a + d` and
    `c - b` cannot both vanish for a positive-determinant `A` (that would
    make `A` a negative multiple of a rotation).
    """
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    th = jnp.arctan2(c - b, a + d)
    ct, st = jnp.cos(th), jnp.sin(th)
    return jnp.array([[ct, -st], [st, ct]])


def log_spd_2d(U):
    """`ln(U)` for a symmetric positive-definite 2x2, closed form.

    With eigenvalues `m +- D` (`m = tr(U)/2`, `D = sqrt(m^2 - det U)`),
    the isotropic-tensor-function representation is

        ln(U) = alpha*I + beta*(U - m*I),
        alpha = ln(det U)/2,   beta = ln((m+D)/(m-D)) / (2D) = atanh(D/m)/D

    `beta` is removable-singular at `D = 0` (the coincident-eigenvalue case,
    i.e. `U` spherical -- which for an increment is the COMMON case, not an
    edge case) with limit `1/m`. Evaluated there by its Taylor series
    `atanh(x)/x = 1 + x^2/3 + x^4/5 + ...`, guarded with the double-`where`
    idiom so the unused branch cannot poison the reverse-mode gradient.
    """
    m = 0.5 * (U[0, 0] + U[1, 1])
    detU = U[0, 0] * U[1, 1] - U[0, 1] * U[1, 0]
    D2 = jnp.maximum(m * m - detU, 0.0)

    # The double-`where` must be taken on D2, NOT on x = sqrt(D2)/m: the
    # branch predicate keeps the *value* finite, but `d/dD2 sqrt(D2)` is
    # infinite at D2 = 0 and reverse mode multiplies that into the unused
    # branch's cotangent, producing NaN even though the taken branch is
    # perfectly smooth. So `sqrt` must never see the singular argument at
    # all. `0.25*m*m` is the stand-in (x = 1/2, comfortably inside atanh's
    # domain -- m*m itself would give atanh(1) = inf and reintroduce the
    # problem from the other side).
    small = D2 < 1e-8 * m * m
    D2_safe = jnp.where(small, 0.25 * m * m, D2)
    x_safe = jnp.sqrt(D2_safe) / m
    beta = jnp.where(
        small,
        (1.0 + D2 / (3.0 * m * m) + D2 * D2 / (5.0 * m ** 4)) / m,
        jnp.arctanh(x_safe) / (m * x_safe),
    )
    alpha = 0.5 * jnp.log(jnp.maximum(detU, 1e-30))
    return alpha * jnp.eye(2) + beta * (U - m * jnp.eye(2))


def _sym_to_voigt(T):
    """Symmetric 2x2 -> `[T11, T22, 2*T12]`.

    ENGINEERING shear, matching Abaqus's `DSTRAN` and this codebase's own
    Voigt strain convention. The stress direction uses tensor shear
    (`[s11, s22, s12]`), which is the pairing that makes `S_v . E_v = S : E`
    -- the same convention `kinematics/frame.py` documents.
    """
    return jnp.array([T[0, 0], T[1, 1], 2.0 * T[0, 1]])


def _stress_voigt_to_tensor(s):
    return jnp.array([[s[0], s[2]], [s[2], s[1]]])


def make_rate_form_response(rate_response_fn, n_law_state: int):
    """Wrap a rate-form law as a total-form `response_fn`.

    Parameters
    ----------
    rate_response_fn : callable, STATIC under jit
        `(deps_voigt(3,), sigma_voigt(3,), law_state(n,), dt, dR(2,2), params)
         -> (sigma_new_voigt(3,), law_state_new(n,))`

        `deps_voigt` is `ln(DeltaU)` in engineering-shear Voigt -- Abaqus's
        `DSTRAN`. `sigma_voigt` is the Cauchy stress **already rotated by
        `dR`**, exactly as Abaqus states: "the stress tensor has already
        been rotated to account for rigid body motion in the increment
        before UMAT is called, so that only the corotational part of the
        stress integration should be done in UMAT" (SUB 1.1.44). `dR` is
        `DROT`, passed so the law can rotate any tensor-valued quantity it
        keeps in `law_state` -- the adapter does not touch those.
    n_law_state : int
        Length of the law's own state block.

    Returns
    -------
    callable
        `(F(2,2), state(7+n,), dt, params) -> (S_voigt_ref0(3,), state_new)`
        -- the Phase A total-form contract. The returned stress is **PK2
        referred to REF0**, which is what `kinematics/frame.py` assumes and
        then pushes forward; the adapter does the `S = J F^-1 sigma F^-T`
        pull-back so the element never learns that a rate law is involved.
    """

    def response(F, state, dt, params):
        F = jnp.asarray(F, dtype=jnp.float64)
        F_prev = state[0:4].reshape(2, 2)
        sigma_prev = state[4:7]
        law_state = state[_N_ADAPTER_STATE:]

        dF = F @ jnp.linalg.inv(F_prev)
        dR = polar_rotation_2d(dF)
        dU = dR.T @ dF
        dU = 0.5 * (dU + dU.T)            # symmetrise against roundoff
        deps = _sym_to_voigt(log_spd_2d(dU))

        # Objective rate, Abaqus-style: rotate the stored stress with the
        # increment's rigid rotation, then integrate the CO-ROTATIONAL part
        # only. No spin term is added -- this rotation IS the Jaumann rate
        # under the Hughes-Winget midincrement construction.
        sig_rot = dR @ _stress_voigt_to_tensor(sigma_prev) @ dR.T
        sig_rot_v = jnp.array([sig_rot[0, 0], sig_rot[1, 1], sig_rot[0, 1]])

        sigma_new, law_state_new = rate_response_fn(
            deps, sig_rot_v, law_state, dt, dR, params)

        # Cauchy (current config) -> PK2 (REF0), the response interface's
        # promised frame.
        J = jnp.linalg.det(F)
        Finv = jnp.linalg.inv(F)
        S = J * (Finv @ _stress_voigt_to_tensor(sigma_new) @ Finv.T)
        S_voigt = jnp.array([S[0, 0], S[1, 1], S[0, 1]])

        state_new = jnp.concatenate(
            [F.reshape(4), jnp.asarray(sigma_new), jnp.asarray(law_state_new)])
        return S_voigt, state_new

    return response


# ----------------------------------------------------------------------
# A reference rate-form law, for the adapter's own tests
# ----------------------------------------------------------------------
def hypoelastic_response(deps_voigt, sigma_voigt, law_state, dt, dR, params):
    """`sigma_dot = lambda tr(d) I + 2 mu d` -- the simplest rate-form law.

    `params = (lam, mu)`. Exists to exercise the adapter: under simple
    shear with a Jaumann rate this law has a closed-form oscillatory
    solution (Dienes 1979), which is the hand-integrated reference the
    adapter is tested against. Not intended for production use -- it is a
    hypoelastic law, i.e. not exactly path-independent, which is precisely
    why total-form laws are preferred and why this repo has none.
    """
    lam, mu = params
    e11, e22, g12 = deps_voigt[0], deps_voigt[1], deps_voigt[2]
    tr = e11 + e22
    return (jnp.array([
        sigma_voigt[0] + lam * tr + 2.0 * mu * e11,
        sigma_voigt[1] + lam * tr + 2.0 * mu * e22,
        sigma_voigt[2] + mu * g12,          # engineering shear -> mu*gamma
    ]), law_state)
