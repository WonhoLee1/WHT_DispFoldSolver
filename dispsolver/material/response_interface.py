"""
response_interface.py
======================
Common material-response interface -- Phase A of the Abaqus/OptiStruct-
standard element/material unification
(`dev_log/plan_abaqus_element_consolidation_20260908.md`).

Real Abaqus (and OptiStruct) separate element kinematics (integration
scheme, incompatible modes, hybrid pressure, reduced+hourglass -- CPE4,
CPE4I, CPE4H, CPE4IH, CPE4RH) from material constitutive response
(elastic/plastic/hyperelastic/viscoelastic, or a user UMAT) through one
generic interface: given a strain/deformation-gradient state and the
material's own history variables, return stress, (optionally) a
consistent tangent, and updated history variables. The element never
inspects WHICH material is attached; the material never inspects WHICH
element it is embedded in. This codebase currently violates that
separation -- material choice is baked into the element NAME (`CPE4I`
only exists paired with `ViscoelasticMaterial`; a near-identical
`Q4_COROTATIONAL_EAS` is separately coded for `J2Plasticity`) as a
historical build-order artifact, not a physical necessity (confirmed:
real Abaqus does not distinguish CPE4I by attached material at all).

Canonical shape
----------------
    S_voigt, state_new = response_jax(F_2d, state, dt, params)
    S_voigt, C_voigt, state_new = tangent_response_jax(F_2d, state, dt, params)

`params` is an OPAQUE, material-defined blob (tuple of scalars/arrays) --
the element layer forwards it without ever inspecting its contents. This
mirrors Abaqus's own UMAT contract (`PROPS`/`STATEV` in, `STATEV` out,
`DDSDDE` required for an implicit solver) -- it's exactly why one CPE4I
kinematics implementation can accept any material in real Abaqus.

`tangent_response_jax` is OPTIONAL per material, not part of the
mandatory contract: an element is always free to get its tangent by
differentiating (`jax.jacobian`/finite-difference) straight through
`response_jax` at the element level instead, which is what every
viscoelastic element kernel in this codebase already does (`_simo_pk2`
has no eigendecomposition anywhere, so it is safe to autodiff directly --
see `q4_visco_simo_fs_jax.py`'s own docstring). `tangent_response_jax` is
provided for materials where that direct approach is NOT safe or NOT
cheap -- J2 plasticity's spectral return map hits the eigenvalue-collision
NaN-under-`jax.jacobian` class documented in AGENTS.md §4.3 whenever it
is embedded inside a larger autodiff'd expression (e.g. a co-rotational
transform composed with it), so its elements instead call a
material-ONLY tangent (computed once, safely, via `jacfwd` restricted to
just the material response) and assemble `K = Sum BL^T C_v BL` by hand.
This asymmetry is a REAL numerical-safety difference between the two
material classes, not something this interface should or does hide --
each material's own functions remain responsible for their own safety;
this module only gives them a uniform calling shape.

Phase A scope (this file, this commit)
----------------------------------------
Define the interface and verify -- via
`tests/test_response_interface_phase_a.py` -- that every wrapper here
reproduces its underlying existing function BIT-FOR-BIT at several
states (identity, rotation, stretch, plastic yielding for J2; several
homogeneous F for viscoelastic). Zero element kernel calls through this
module yet (that is Phase C, per the plan doc) -- this is pure additive
infrastructure with no behavior change to any currently-live code path.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .plastic_jax import pk2_voigt_jax, tangent_voigt_jax


class J2Params(NamedTuple):
    """Opaque params blob for J2Plasticity's response functions.

    J2 plasticity is RATE-INDEPENDENT -- `dt` is accepted by
    `j2_response_jax`/`j2_tangent_response_jax` for interface uniformity
    (every material's `response_jax` takes the same 4 positional
    arguments) but is genuinely unused inside, unlike the viscoelastic
    family where `dt` drives the Prony overstress recurrence.
    """
    lam: float
    mu: float
    sigma_y0: float
    H: float


def j2_response_jax(F_2d, state, dt, params: J2Params):
    """(S_voigt(3,), state_new) -- thin pass-through to `pk2_voigt_jax`."""
    lam, mu, sigma_y0, H = params
    return pk2_voigt_jax(F_2d, state, lam, mu, sigma_y0, H)


def j2_tangent_response_jax(F_2d, state, dt, params: J2Params):
    """(S_voigt(3,), C_voigt(3,3), state_new) -- thin pass-through to
    `tangent_voigt_jax` (the material-only tangent J2 elements use to
    avoid autodiffing the spectral return map inside a larger composed
    expression -- see module docstring)."""
    lam, mu, sigma_y0, H = params
    return tangent_voigt_jax(F_2d, state, lam, mu, sigma_y0, H)


class ViscoParams(NamedTuple):
    """Opaque params blob for ViscoelasticMaterial's response function.

    `base` (str: "neohookean"|"yeoh"|"arruda") must be STATIC under
    `jax.jit` -- it selects a code path, it is not a number. Two ways to
    satisfy that:

    * Element-facing (preferred, and what a material-agnostic element
      like `cpe4_jax.py` needs): use `make_visco_response(base)` below,
      which binds `base` into the returned callable so the `params` blob
      the element traces contains ONLY arrays and floats. This keeps the
      element's `response_fn` static-argname contract clean and stops the
      string from ever reaching a traced pytree.
    * Direct callers already inside a `static_argnames=("base",)` jit can
      keep passing `ViscoParams` with `base` populated to
      `visco_response_jax`.
    """
    base: str
    kappa: float
    bparams: jnp.ndarray
    g_i: jnp.ndarray
    tau_i: jnp.ndarray
    g_inf: float
    distortion_j_crit: float = 0.0


class ViscoParamsTraced(NamedTuple):
    """`ViscoParams` minus the static `base` -- fully traceable."""
    kappa: float
    bparams: jnp.ndarray
    g_i: jnp.ndarray
    tau_i: jnp.ndarray
    g_inf: float
    distortion_j_crit: float = 0.0


def make_visco_response(base: str):
    """Bind the static `base` model selector into a response callable.

    Returns `fn(F, state, dt, params: ViscoParamsTraced) -> (S_voigt,
    state_new)` -- the exact shape an element's `response_fn` expects,
    with nothing unhashable or untraceable left in `params`. The returned
    function is cached per `base` so repeated calls give the SAME object,
    which matters because elements take `response_fn` as a
    `static_argnames` argument: a fresh closure per call would retrigger
    JAX tracing every time.
    """
    return _VISCO_RESPONSE_CACHE.setdefault(base, _build_visco_response(base))


def _build_visco_response(base: str):
    def _fn(F_2d, state, dt, params):
        from ..element.q4_visco_simo_fs_jax import _simo_pk2
        return _simo_pk2(base, F_2d, state, params.kappa, params.bparams,
                         params.g_i, params.tau_i, params.g_inf, dt,
                         distortion_j_crit=params.distortion_j_crit)
    _fn.__name__ = f"visco_response_{base}"
    return _fn


_VISCO_RESPONSE_CACHE: dict = {}


def visco_response_jax(F_2d, state, dt, params: ViscoParams):
    """(S_voigt(3,), state_new) -- thin pass-through to `_simo_pk2`.

    No `visco_tangent_response_jax` counterpart: every existing
    viscoelastic element kernel gets its tangent by autodiffing straight
    through this function (or the element force built from it) at the
    element level, which is safe here (no eigendecomposition in
    `_simo_pk2` at all) -- see module docstring.
    """
    # Local import: avoids a hard import-order dependency between
    # dispsolver.material and dispsolver.element (q4_visco_simo_fs_jax
    # imports from dispsolver.material.viscoelastic's WLF helper, so a
    # module-level import here would be circular).
    from ..element.q4_visco_simo_fs_jax import _simo_pk2

    base, kappa, bparams, g_i, tau_i, g_inf, distortion_j_crit = params
    return _simo_pk2(base, F_2d, state, kappa, bparams, g_i, tau_i, g_inf,
                     dt, distortion_j_crit=distortion_j_crit)
