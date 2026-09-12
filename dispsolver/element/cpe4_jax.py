"""
cpe4_jax.py
============
**CPE4** -- the real Abaqus element: 4-node bilinear plane-strain quad,
full 2x2 Gauss integration with Abaqus's own **selectively reduced
(volumetric) integration** -- the dilatational part taken from the
**element-average** Jacobian, i.e. the B-bar / mean-dilatation treatment
Abaqus builds into its fully-integrated first-order continuum elements.
Finite strain, Updated-Lagrangian capable, and **material-agnostic**.

.. note:: Corrected 2026-09-11 -- this element previously sampled the
   dilatation at the element **centroid**. That is a different device:
   Abaqus defines `F_bar = (J_bar/J)^(1/n) F` with "`J_bar` the AVERAGE
   Jacobian over the element" (Theory Guide 3.2.4), while the centroid
   sample is an optional Abaqus/**Explicit** C3D8R "centroidal strain
   formulation" that the same section calls "less accurate when the
   elements are skewed". The two coincide **exactly for parallelogram
   elements** -- which is every element in this file's own unit tests
   except the irregular-quad patch test. AGENTS.md 4.14's lesson,
   recurring verbatim: a device that is exact on the test geometry and
   wrong on the production geometry (`gen_ex12_inp.py`'s graded display
   mesh is not parallelogram at the transition columns).

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
  VOLUMETRIC (dilatational) part is homogenised over the element --
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
  up to 24.9% on the sibling kernels before that fix). **That
  push-forward is not written here**: it lives in
  `kinematics/frame.py::gp_internal_force`, the only function in the
  library permitted to contract a stress with a B-operator, so an
  element cannot forget it or apply it to only half a mixed residual.
  See that module for why (F4/F6/B1/B2/B3 are all one defect class).
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

from .kinematics.frame import GPKinematics, gp_internal_force
from .q4_visco_simo_fs_jax import _GP2, _W2, _grads, _F_at, _BL_columns


def _det2(A):
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


def _gp_kinematics(coords, u_elem, thickness, F_n):
    """Build all four Gauss points' `GPKinematics`, with Abaqus's F-bar.

    F-bar (de Souza Neto et al. 1996) is the finite-strain form of the
    selectively-reduced volumetric integration Abaqus applies to its
    fully-integrated first-order continuum elements:

        F_bar = (J_bar / J)^(1/n) * F          (n = 2 in plane strain)

    with `J` the Jacobian at the Gauss point and **`J_bar` the AVERAGE
    Jacobian over the element** (Abaqus 2016 Theory Guide 3.2.4, verbatim;
    for 2D "`J_bar` and `J` are the change in area", hence the square
    root). The deviatoric response still comes from each Gauss point --
    only the volumetric part is homogenised, which is what makes CPE4
    resist volumetric locking while still shear-locking in bending.

    The average is taken over the **REFERENCE** volume, `J_bar =
    integral J dV0 / V0` -- so the quadrature weight used for the average
    is `dV0 = dV_n / det(F_n)`, not the config-n weight the element
    integrates the internal force with. The two differ only in UL mode
    and coincide identically when `F_n = I`.
    """
    w_n = []       # integration weight on REF_N (what f_int integrates with)
    w_0 = []       # reference-volume measure (what J_bar averages with)
    Js = []
    F_incs, F_tots, grads = [], [], []

    for gp in range(4):
        xi, eta = _GP2[gp]
        gX, gY, detJ = _grads(xi, eta, coords)
        F_inc = _F_at(gX, gY, u_elem)
        F = F_inc @ F_n[gp]                       # total, referred to REF0
        wn = detJ * _W2[gp] * thickness
        w_n.append(wn)
        w_0.append(wn / jnp.maximum(jnp.abs(_det2(F_n[gp])), 1e-30))
        Js.append(_det2(F))
        F_incs.append(F_inc)
        F_tots.append(F)
        grads.append((gX, gY))

    Js = jnp.stack(Js)
    w_0 = jnp.stack(w_0)
    J_bar = jnp.sum(Js * w_0) / jnp.sum(w_0)

    kins = []
    for gp in range(4):
        # Clamps mirror q4_visco_simo_fs_jax.py's own F-bar guards: a
        # line-search trial state can drive J through zero before the
        # outer inversion check rejects it.
        J_ratio = jnp.maximum(J_bar, 0.05) / jnp.maximum(Js[gp], 0.05)
        F_bar = F_tots[gp] * jnp.sqrt(jnp.clip(J_ratio, 0.1, 10.0))
        gX, gY = grads[gp]
        kins.append(GPKinematics(
            F_total=F_bar,                # REF0 -- the material's frame
            F_inc=F_incs[gp],             # CURRENT relative to REF_N
            F_n=F_n[gp],                  # REF_N relative to REF0
            B_L=_BL_columns(F_incs[gp], gX, gY),   # conjugate on REF_N
            weight=w_n[gp],                        # measure on REF_N
        ))
    # `F_tots` (unbarred) is what gets committed as the next step's `F_n`:
    # F-bar is a locking device applied to what the MATERIAL sees, not a
    # redefinition of the element's deformation history.
    return kins, jnp.stack(F_tots)


def _cpe4_internal_force(response_fn, coords, u_elem, state_elem, params,
                         dt, thickness, F_n):
    """(f_int(8,), state_new, F_n_new) -- full 2x2 integration, F-bar volumetric."""
    kins, F_tots = _gp_kinematics(coords, u_elem, thickness, F_n)

    f_int = jnp.zeros(8)
    state_new = jnp.empty_like(state_elem)
    for gp in range(4):
        S_voigt, st_new = response_fn(kins[gp].F_total, state_elem[gp], dt, params)
        f_gp, _ = gp_internal_force(kins[gp], S_voigt)
        f_int = f_int + f_gp
        state_new = state_new.at[gp].set(st_new)

    return f_int, state_new, F_tots


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

    kins, F_n_new = _gp_kinematics(coords, u_elem, thickness, F_n)

    f_int = jnp.zeros(8)
    K_e = jnp.zeros((8, 8))
    state_new = jnp.empty_like(state_elem)

    for gp in range(4):
        S_voigt, C_v, st_new = tangent_response_fn(
            kins[gp].F_total, state_elem[gp], dt, params)
        # Both the stress AND the tangent go through the same push-forward,
        # in the same call. Pushing one without the other is defect B1
        # (`K_e` stops being the Jacobian of `f_int`); doing it by hand at
        # each site is how B2/B3 ended up pushing `f_u` but not `f_a`.
        f_gp, K_gp = gp_internal_force(kins[gp], S_voigt, C_v)
        f_int = f_int + f_gp
        K_e = K_e + K_gp
        state_new = state_new.at[gp].set(st_new)

    bad = jnp.any(jnp.isnan(f_int)) | jnp.any(jnp.isnan(K_e)) \
        | jnp.any(jnp.isinf(f_int)) | jnp.any(jnp.isinf(K_e))
    f_int = jnp.where(bad, jnp.zeros_like(f_int), f_int)
    K_e = jnp.where(bad, jnp.zeros_like(K_e), K_e)
    return f_int, K_e, state_new, F_n_new
