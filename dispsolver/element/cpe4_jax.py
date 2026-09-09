"""
cpe4_jax.py
============
**CPE4** -- the real Abaqus element: 4-node bilinear plane-strain quad,
full 2x2 Gauss integration with Abaqus's own **selectively reduced
(volumetric) integration** -- the dilatational part evaluated at the
element centroid, i.e. the B-bar / mean-dilatation treatment Abaqus
builds into its fully-integrated first-order continuum elements. Finite
strain, Updated-Lagrangian capable, and **material-agnostic**.

Why this file exists
--------------------
Not because the existing `Q4` (`q4.py`/`q4_jax.py`/`q4_numba.py`) has the
wrong formulation -- it does not. An earlier note in
`dev_log/plan_abaqus_element_consolidation_20260908.md` claimed the B-bar
in `q4.py` made it "not really CPE4"; that claim was **wrong and has been
retracted**: Abaqus's fully-integrated first-order solids (CPE4, C3D8)
apply exactly this volumetric selective reduced integration by default.
It is why CPE4 resists volumetric locking but still SHEAR-locks in
bending -- and shear locking is precisely what CPE4I (incompatible modes)
exists to cure. A "pure full integration, no volumetric treatment"
element would be the invention, not this.

This file exists because the legacy `q4.py` family is small-strain,
linear-elastic-only, has no Updated-Lagrangian support and is hard-wired
to one material. This is the same element formulation rebuilt to the
architecture the Abaqus/OptiStruct consolidation is standardising on:
finite strain, UL-capable, and material-agnostic (element = kinematics,
material = constitutive, joined by one generic interface).

Formulation
-----------
* Kinematics: standard bilinear shape functions, 2x2 Gauss quadrature.
  The DEVIATORIC response is sampled at each Gauss point; the
  VOLUMETRIC (dilatational) part is taken from the element centroid --
  Abaqus's selectively-reduced treatment for this element class. No
  incompatible modes, no hybrid pressure field, no hourglass control:
  those belong to CPE4I / CPE4H / CPE4R(H), which are separate elements
  in this library exactly as they are in Abaqus.
* Finite strain: `F_inc` from the reference-config gradients,
  Updated-Lagrangian total `F = F_inc @ F_n[gp]` (pass a tile of
  identity for pure Total-Lagrangian). Green-Lagrange `B_L` from
  `F_inc`, with the material's PK2 pushed forward to configuration n
  (`S_n = F_n S F_n^T / det F_n`) before contraction -- the work-conjugacy
  requirement established in
  `dev_log/plan_abaqus_element_consolidation_20260908.md` finding F4;
  omitting it is a first-order error in the stretch of `F_n` (measured
  up to 24.9% on the sibling kernels before that fix).
* **Material-agnostic**: the constitutive response arrives as a callable
  bound at trace time (`response_fn(F, state, dt, params) -> (S_voigt,
  state_new)`, the Phase A interface in
  `dispsolver/material/response_interface.py`). The element never
  inspects which material it is; the material never inspects which
  element it is in. This is the Abaqus/OptiStruct separation the whole
  consolidation effort is built around, and CPE4 is the first element in
  this codebase written to it from the start rather than retrofitted.
* Tangent: `jax.jacobian` of the internal force. Exact/consistent for
  any material whose response is itself differentiable (all the
  viscoelastic ones are -- no eigendecomposition anywhere). For J2
  plasticity, whose spectral return map carries the AGENTS.md 4.3
  eigenvalue-collision NaN risk under composed autodiff, pass a material
  that supplies its own material-level tangent and use
  `compute_cpe4_material_tangent` instead (assembles
  `K = sum B_L^T C B_L w`, the established modified-Newton precedent for
  the J2 elements in this codebase).

What this element is NOT good at (by design, matching real CPE4)
-----------------------------------------------------------------
Shear locking in bending at high aspect ratio, and volumetric locking as
nu -> 0.5. Both are genuine properties of CPE4, not defects of this
implementation -- if a model needs them cured, that is what CPE4I
(incompatible modes), CPE4H (hybrid pressure) and CPE4RH exist for, and
those are separate elements in this library, exactly as in Abaqus.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .q4_visco_simo_fs_jax import _GP2, _W2, _grads, _F_at, _BL_columns


def _push_forward(S_voigt, F_n_gp):
    """PK2 referred to the ORIGINAL config -> PK2 on configuration n.

    `S_n = F_n S F_n^T / det(F_n)`; identity (exact no-op) when
    `F_n = I`, i.e. Total-Lagrangian mode is untouched by this. See the
    module docstring's F4 note for why this is mandatory whenever the
    B-operator and integration weight are built on config n while the
    material response is computed from the TOTAL deformation gradient.
    """
    S = jnp.array([[S_voigt[0], S_voigt[2]], [S_voigt[2], S_voigt[1]]])
    detFn = jnp.maximum(jnp.abs(F_n_gp[0, 0] * F_n_gp[1, 1]
                                - F_n_gp[0, 1] * F_n_gp[1, 0]), 1e-30)
    Sn = (F_n_gp @ S @ F_n_gp.T) / detFn
    return jnp.array([Sn[0, 0], Sn[1, 1], Sn[0, 1]])


def _cpe4_internal_force(response_fn, coords, u_elem, state_elem, params,
                         dt, thickness, F_n):
    """(f_int(8,), state_new, F_n_new) -- full 2x2 integration with the
    centroid-sampled volumetric (F-bar) treatment.

    F-bar (de Souza Neto et al. 1996) is the finite-strain form of the
    same selectively-reduced volumetric integration Abaqus applies to its
    fully-integrated first-order continuum elements: the dilatation is
    taken from the element centroid,
        `F_bar = F * sqrt(J0 / J)`  (2D plane strain; cube root in 3D)
    with `J0 = det F` at the centroid. The deviatoric response still
    comes from each Gauss point -- only the volumetric part is
    homogenised, which is exactly what makes CPE4 resist volumetric
    locking while still shear-locking in bending.
    """
    f_int = jnp.zeros(8)
    state_new = jnp.empty_like(state_elem)
    F_n_new = jnp.zeros((4, 2, 2))

    # Centroid dilatation for the F-bar volumetric treatment.
    gX0, gY0, _ = _grads(0.0, 0.0, coords)
    F0 = _F_at(gX0, gY0, u_elem) @ (0.25 * (F_n[0] + F_n[1] + F_n[2] + F_n[3]))
    J0 = F0[0, 0] * F0[1, 1] - F0[0, 1] * F0[1, 0]

    for gp in range(4):
        xi, eta = _GP2[gp]
        gX, gY, detJ = _grads(xi, eta, coords)
        w = detJ * _W2[gp] * thickness

        F_inc = _F_at(gX, gY, u_elem)
        F = F_inc @ F_n[gp]                       # total deformation gradient
        J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
        # Clamps mirror q4_visco_simo_fs_jax.py's own F-bar guards: a
        # line-search trial state can drive J through zero before the
        # outer inversion check rejects it.
        J_ratio = jnp.maximum(J0, 0.05) / jnp.maximum(J, 0.05)
        F_bar = F * jnp.sqrt(jnp.clip(J_ratio, 0.1, 10.0))

        S_voigt, st_new = response_fn(F_bar, state_elem[gp], dt, params)
        S_n = _push_forward(S_voigt, F_n[gp])

        BL = _BL_columns(F_inc, gX, gY)
        f_int = f_int + BL.T @ S_n * w

        state_new = state_new.at[gp].set(st_new)
        F_n_new = F_n_new.at[gp].set(F)

    return f_int, state_new, F_n_new


@partial(jax.jit, static_argnames=("response_fn",))
def compute_cpe4(response_fn, coords, u_elem, state_elem, params,
                 dt=0.0, thickness=1.0, F_n=None):
    """CPE4 with an autodiff consistent tangent.

    Parameters
    ----------
    response_fn : callable (STATIC under jit)
        `(F(2,2), state(n,), dt, params) -> (S_voigt(3,), state_new(n,))`
        -- the Phase A material interface. The element does not inspect
        it beyond calling it.
    coords : (4,2) reference coords (UL: the last converged configuration)
    u_elem : (8,) displacement (UL: the increment since `coords`)
    state_elem : (4, n_state) per-Gauss-point material history
    params : material-defined opaque blob, forwarded untouched
    F_n : (4,2,2) UL total F at the last converged step; `None` -> TL.

    Returns
    -------
    (f_int(8,), K_e(8,8), state_new(4,n_state), F_n_new(4,2,2))
    """
    _eye = jnp.eye(2)
    F_n = jnp.stack([_eye, _eye, _eye, _eye]) if F_n is None else F_n

    f_int, state_new, F_n_new = _cpe4_internal_force(
        response_fn, coords, u_elem, state_elem, params, dt, thickness, F_n)

    K_e = jax.jacobian(
        lambda u: _cpe4_internal_force(
            response_fn, coords, u, state_elem, params, dt, thickness, F_n)[0]
    )(u_elem)

    bad = jnp.any(jnp.isnan(f_int)) | jnp.any(jnp.isnan(K_e)) \
        | jnp.any(jnp.isinf(f_int)) | jnp.any(jnp.isinf(K_e))
    f_int = jnp.where(bad, jnp.zeros_like(f_int), f_int)
    K_e = jnp.where(bad, jnp.zeros_like(K_e), K_e)
    return f_int, K_e, state_new, F_n_new


@partial(jax.jit, static_argnames=("tangent_response_fn",))
def compute_cpe4_material_tangent(tangent_response_fn, coords, u_elem,
                                  state_elem, params, dt=0.0, thickness=1.0,
                                  F_n=None):
    """CPE4 assembled with a MATERIAL-supplied tangent instead of autodiff.

    For materials whose response cannot safely be differentiated through
    as part of a larger expression -- J2 plasticity's spectral return map
    hits the eigenvalue-collision NaN class (AGENTS.md 4.3) when composed
    inside `jax.jacobian` -- the material supplies `C = dS/dE` itself and
    the element assembles `K = sum B_L^T C B_L w`. This drops the
    geometric/initial-stress term, the same established modified-Newton
    precedent every J2 element in this codebase already uses
    (AGENTS.md 4.4): a couple of extra Newton iterations near large
    rotation increments, in exchange for a tangent that is finite and
    compiles in seconds.

    `tangent_response_fn`: `(F, state, dt, params) -> (S_voigt, C(3,3), state_new)`.
    """
    _eye = jnp.eye(2)
    F_n = jnp.stack([_eye, _eye, _eye, _eye]) if F_n is None else F_n

    f_int = jnp.zeros(8)
    K_e = jnp.zeros((8, 8))
    state_new = jnp.empty_like(state_elem)
    F_n_new = jnp.zeros((4, 2, 2))

    # Same centroid-sampled volumetric (F-bar) treatment as the autodiff
    # variant above -- see `_cpe4_internal_force`'s docstring.
    gX0, gY0, _ = _grads(0.0, 0.0, coords)
    F0 = _F_at(gX0, gY0, u_elem) @ (0.25 * (F_n[0] + F_n[1] + F_n[2] + F_n[3]))
    J0 = F0[0, 0] * F0[1, 1] - F0[0, 1] * F0[1, 0]

    for gp in range(4):
        xi, eta = _GP2[gp]
        gX, gY, detJ = _grads(xi, eta, coords)
        w = detJ * _W2[gp] * thickness

        F_inc = _F_at(gX, gY, u_elem)
        F = F_inc @ F_n[gp]
        J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
        J_ratio = jnp.maximum(J0, 0.05) / jnp.maximum(J, 0.05)
        F_bar = F * jnp.sqrt(jnp.clip(J_ratio, 0.1, 10.0))

        S_voigt, C_v, st_new = tangent_response_fn(F_bar, state_elem[gp], dt, params)
        S_n = _push_forward(S_voigt, F_n[gp])

        BL = _BL_columns(F_inc, gX, gY)
        f_int = f_int + BL.T @ S_n * w
        K_e = K_e + (BL.T @ C_v @ BL) * w

        state_new = state_new.at[gp].set(st_new)
        F_n_new = F_n_new.at[gp].set(F)

    bad = jnp.any(jnp.isnan(f_int)) | jnp.any(jnp.isnan(K_e)) \
        | jnp.any(jnp.isinf(f_int)) | jnp.any(jnp.isinf(K_e))
    f_int = jnp.where(bad, jnp.zeros_like(f_int), f_int)
    K_e = jnp.where(bad, jnp.zeros_like(K_e), K_e)
    return f_int, K_e, state_new, F_n_new
