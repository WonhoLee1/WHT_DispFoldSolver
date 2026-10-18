"""
q4_visco_hybrid_reduced_jax.py
================================
CPE4RH-equivalent element: 1-point reduced integration + Flanagan-
Belytschko hourglass stabilization + element-constant hybrid pressure, on
the finite-strain Arruda-Boyce/Neo-Hookean/Yeoh + Prony viscoelastic (PSA)
material. Pure JAX (R&D backend, per AGENTS.md's JIT strategy).

Why this exists
----------------
Abaqus element-name completeness, requested explicitly: CPE4RH (reduced
integration, hybrid) is the element the user reports actually using for
PSA-type near-incompressible adhesive layers in real Abaqus work. This
codebase had NO honest implementation of it -- `Q4_HYBRID_RH` was only a
name in `model_builder.py`'s Abaqus-name mapping table with no kernel
behind it anywhere (`dynamic.py` never recognised the string in any
dispatch list), so selecting it would have silently fallen back to a
plain small-strain elastic Q4 (AGENTS.md 4.2's "element type quietly
ignored" failure class) -- found and confirmed by direct grep, 2026-09-08.
The one existing "reduced integration" kernel (`q4_reduced_jax.py`) is for
J2 plasticity with no pressure field at all, a different element family.

Scope of this file: THEORETICAL correctness of the element formulation
itself, verified by patch test (see below) -- independent of whether this
element is the right *choice* for any particular mesh in this repo.
`dev_log/session_20260730_reduced_integration_failure.md` already found
that 1-point reduced integration + hourglass control is a poor fit for a
bending-dominated, one-element-through-thickness PSA free span (the pure-
bending nodal pattern is mathematically identical to the hourglass pattern
a 1-point rule cannot see, so hourglass control damps genuine bending --
measured 10.19x mesh dependence, worse than plain corotational). That is a
finding about mesh/loading *suitability*, not about whether an element
named CPE4RH can be built correctly; this file answers the latter question
only, on its own terms, per explicit instruction to implement it
correctly regardless of whether production ever selects it for the PSA
free span.

Formulation
-----------
* Kinematics: ONE centroid (xi=eta=0) Gauss point. `F_inc` from the
  centroid B-operator, Updated-Lagrangian total `F = F_inc @ F_n[0]` (only
  one physical GP -- `F_n`/`state_elem` are still carried as (4,2,2)/
  (4,n_state) arrays for shape-compatibility with the rest of the solver,
  broadcasting the single physical result into all 4 slots, the same
  convention `q4_reduced_jax.py` already established for its `state_new`).
* Hybrid pressure: the perturbed-Lagrangian pressure residual
  `r_p = int(J-1)dV - (p/K)*V` is a volume AVERAGE of `(J-1)`. With exactly
  one sample point, that average is just the sample itself, so the usual
  CPE4H closed form `p = K * int(J-1)dV / V` (see `q4_visco_hybrid_up_numba.py`)
  degenerates to the even simpler `p = K*(J-1)` here -- exact, no
  integration, no iteration.
* Deviatoric/isochoric + Prony overstress response: `_simo_pk2` reused
  verbatim (kappa=0 switches off its own volumetric term, exactly the
  convention `q4_visco_eas_jax.py`'s CPE4H branch already uses), plus the
  hybrid volumetric PK2 `p*J*C^-1` added on top -- the identical
  deviatoric/volumetric split CPE4H uses, evaluated at one point instead
  of four.
* Hourglass control (2026-10-10; replaces the in-house
  `0.05*(kappa+2*bparams[0])` Flanagan-Belytschko coefficient): Abaqus
  *SECTION CONTROLS, HOURGLASS=ENHANCED | STIFFNESS, shared with CPE4R --
  see `cpe4r_jax.py`'s module docstring for the spec quotes and the
  derivation. ENHANCED (the Keywords Guide default for hyperelastic
  materials, i.e. this element) adds the exact hourglass part of the
  linear-elastic CPE4I (Q1E4) stiffness built from the INSTANTANEOUS
  small-strain moduli of this material: G0 = 2 W1(I1=3) (g_inf + sum g_i)
  (the Simo overstress at dt -> 0 recovers the full base response) and
  lam0 = kappa - 2 G0 / 3 (the hybrid p = kappa (J-1) supplies kappa).
  STIFFNESS uses 0.005 * G0 * ss. Both are evaluated with the hourglass
  coordinate pulled back by the centroid polar rotation; force and tangent
  are the exact gradient/Hessian.

  The hourglass energy is a function of the TOTAL hourglass displacement
  on the ORIGINAL configuration, so this element must be called Total
  Lagrangian (`coords` = original, `u_elem` = total, `F_n` = I) --
  `dynamic.py` dispatches it that way (`_ELEMENT_LARGE_DEF["CPE4RH"]`
  `supports_ul=False`). Under the former UL dispatch the hourglass force
  was rebuilt from the INCREMENTAL displacement on config n every step, so
  the accumulated hourglass resistance reset to zero at every converged
  increment; with ENHANCED, where the hourglass term carries the element's
  bending stiffness, that would make bending history-free. The material
  part still accepts `F_n` and stays UL-consistent (tests/test_ul_tl_consistency.py).
* Tangent: because the pressure here has a closed form (no inner Newton
  to condense, unlike CPE4H/CPE4IH's iterative `q`), the internal force is
  already an explicit, differentiable function of `u_elem` alone --
  `K_mat = jax.jacobian(f_mat)(u_elem)` is the EXACT consistent tangent
  (not an approximation, not FD), with `K_hg` the exact Hessian of the
  hourglass energy (cpe4r_jax._hg_tangent).

Verification
------------
See `tests/test_cpe4rh_patch.py`: constant-strain patch test (uniaxial,
equibiaxial, pure shear -- exact to floating-point tolerance against the
analytic PK2 for a homogeneous deformation), rigid-rotation canary (zero
force/energy at several angles including 90 deg), and a 4-irregular-quad
assembled patch (MacNeal & Harder 1985 style) recovering the same
constant-stress state as a single element under an affine boundary
displacement -- the two properties the orthogonalized hourglass term is
specifically constructed to preserve exactly.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .q4_visco_simo_fs_jax import _grads, _F_at, _BL_columns, _simo_pk2, _W1
from .cpe4r_jax import (HOURGLASS_MODES, _ABAQUS_HG_FACTOR, enhanced_hourglass_matrix,
                        _hg_force, _hg_tangent)

# Natural-coordinate hourglass shape vector: nodal values of xi*eta at
# node natural coordinates (-1,-1), (1,-1), (1,1), (-1,1) -- the one
# bilinear pattern a centroid-only sample point cannot see. Identical to
# q4_reduced_jax.py's `_HG_SHAPE`.
_HG_SHAPE = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)

def _centroid_kinematics(coords, u_elem):
    """Centroid gradients, natural-domain "volume" w0, and F_inc."""
    gX, gY, detJ0 = _grads(0.0, 0.0, coords)
    F_inc = _F_at(gX, gY, u_elem)
    return gX, gY, detJ0, F_inc


def _reduced_hybrid_force(base, coords, u_elem, state_elem, kappa, bparams,
                          g_i, tau_i, g_inf, dt, thickness, F_n,
                          distortion_j_crit):
    """f_mat(8,) only -- the piece `jax.jacobian` differentiates for K_mat.

    Returns (f_mat, S_tot, h_new, F, w0, gX, gY) so the caller can reuse
    the state/force/kinematics already computed here without a second
    (and possibly numerically different, since jacobian retraces this
    function) evaluation.
    """
    gX, gY, detJ0, F_inc = _centroid_kinematics(coords, u_elem)
    w0 = detJ0 * 4.0 * thickness   # natural-domain area (2x2 -> 4) x thickness

    F = F_inc @ F_n[0]
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]

    # Closed-form hybrid pressure -- exact at a single sample point (see
    # module docstring): the usual CPE4H volume-average degenerates to the
    # sample itself when there is only one sample.
    p = kappa * (J - 1.0)

    # Deviatoric/isochoric + viscoelastic overstress only (kappa=0 here
    # switches off _simo_pk2's own volumetric law; the hybrid p below
    # supplies it instead) -- identical convention to q4_visco_eas_jax.py's
    # CPE4H branch.
    S_v, h_new = _simo_pk2(base, F, state_elem[0], 0.0, bparams,
                           g_i, tau_i, g_inf, dt,
                           distortion_j_crit=distortion_j_crit)

    C = F.T @ F
    Cinv = jnp.linalg.inv(C + 1e-15 * jnp.eye(2, dtype=jnp.float64))
    S_vol = p * J * jnp.array([Cinv[0, 0], Cinv[1, 1], Cinv[0, 1]])
    S_tot = S_v + S_vol

    # Work-conjugacy push-forward (dev_log/plan_abaqus_element_consolidation_20260908.md
    # finding F4, discovered in q4_visco_simo_fs_jax.py and ported here --
    # this element inherited the same bug at construction time since it
    # copied the established UL pattern without independently re-deriving
    # it, and the patch tests in tests/test_cpe4rh_patch.py all used
    # F_n=identity (TL), which makes this bug a no-op and invisible to
    # them; a true UL state needs it). S_tot is PK2 referred to the
    # ORIGINAL config (built from the TOTAL F); BL/w0 below are step-n
    # quantities. Push forward: S_n = F_n @ S_tot @ F_n.T / det(F_n).
    Fn0 = F_n[0]
    S0_tensor = jnp.array([[S_tot[0], S_tot[2]], [S_tot[2], S_tot[1]]])
    detFn = jnp.maximum(jnp.abs(Fn0[0, 0] * Fn0[1, 1] - Fn0[0, 1] * Fn0[1, 0]), 1e-30)
    Sn_tensor = (Fn0 @ S0_tensor @ Fn0.T) / detFn
    S_n_voigt = jnp.array([Sn_tensor[0, 0], Sn_tensor[1, 1], Sn_tensor[0, 1]])

    BL = _BL_columns(F_inc, gX, gY)
    f_mat = BL.T @ S_n_voigt * w0
    return f_mat, h_new, F, w0, gX, gY


@partial(jax.jit, static_argnames=("base", "hourglass"))
def compute_single_reduced_hybrid_jax(
    base, coords, u_elem, state_elem, kappa, bparams,
    g_i, tau_i, g_inf, dt, thickness=1.0, F_n=None,
    distortion_j_crit=0.0, hourglass="ENHANCED", hg_scale=1.0,
):
    """CPE4RH: 1-point reduced integration + hourglass + hybrid pressure.

    `F_n` (4,2,2): Updated-Lagrangian total F at the last converged step.
    Only slot 0 is physically meaningful (single centroid GP); pass a tile
    of identity for Total-Lagrangian behaviour, matching every other
    UL-capable element in this codebase.

    Returns (f_e(8,), K_e(8,8), state_new(4,n_state), F_n_new(4,2,2)) --
    `state_new`/`F_n_new` broadcast the single physical result into all 4
    slots for array-shape compatibility with the solver's uniform
    (n_elem, 4, ...) storage, the same convention `q4_reduced_jax.py`
    already uses for `state_new`.
    """
    _eye = jnp.eye(2, dtype=jnp.float64)
    F_n = jnp.stack([_eye, _eye, _eye, _eye]) if F_n is None else F_n

    x_nodes = coords[:, 0]
    y_nodes = coords[:, 1]

    def _f_only(u):
        return _reduced_hybrid_force(base, coords, u, state_elem, kappa,
                                     bparams, g_i, tau_i, g_inf, dt,
                                     thickness, F_n, distortion_j_crit)[0]

    f_mat, h_new, F, w0, gX, gY = _reduced_hybrid_force(
        base, coords, u_elem, state_elem, kappa, bparams, g_i, tau_i,
        g_inf, dt, thickness, F_n, distortion_j_crit)

    # Exact consistent tangent: the pressure is already substituted in
    # closed form inside `_f_only`, so there is no separate unknown to
    # condense -- a plain jacobian of the whole force IS `K_uu - K_up
    # K_pp^-1 K_pu` already, the same way CPE4H's Numba port gets the
    # condensed tangent "for free" by recomputing its closed-form p inside
    # every FD perturbation (`q4_visco_hybrid_up_numba.py` docstring) --
    # here it's exact autodiff instead of FD.
    K_mat = jax.jacobian(_f_only)(u_elem)

    state_new = jnp.broadcast_to(h_new, state_elem.shape)
    F_n_new = jnp.broadcast_to(F, F_n.shape)

    # ---- Hourglass control (see module docstring; TL call required) ----
    if hourglass not in HOURGLASS_MODES:
        raise ValueError(f"hourglass={hourglass!r}; expected one of {HOURGLASS_MODES}")
    gamma = _HG_SHAPE - (_HG_SHAPE @ x_nodes) * gX - (_HG_SHAPE @ y_nodes) * gY
    # Instantaneous small-strain moduli of this material (dt -> 0).
    G0 = 2.0 * _W1(base, 3.0, bparams) * (g_inf + jnp.sum(g_i))
    if hourglass == "ENHANCED":
        Hq = enhanced_hourglass_matrix(coords, kappa - 2.0 * G0 / 3.0, G0) * thickness
    else:
        Hq = (hg_scale * _ABAQUS_HG_FACTOR * G0 * w0 * (gX @ gX + gY @ gY)
              * jnp.eye(2, dtype=jnp.float64))
    f_hg = _hg_force(u_elem, gX, gY, gamma, Hq)
    K_hg = _hg_tangent(u_elem, gX, gY, gamma, Hq)

    f_e = f_mat + f_hg
    K_e = K_mat + K_hg

    # Same NaN/Inf-zero guard every element kernel in this codebase has --
    # a line-search TRIAL state (rejected afterwards, evaluated first) can
    # be locally degenerate before the outer inversion check rejects it.
    bad = jnp.any(jnp.isnan(f_e)) | jnp.any(jnp.isnan(K_e)) \
        | jnp.any(jnp.isinf(f_e)) | jnp.any(jnp.isinf(K_e))
    f_e = jnp.where(bad, jnp.zeros_like(f_e), f_e)
    K_e = jnp.where(bad, jnp.zeros_like(K_e), K_e)

    return f_e, K_e, state_new, F_n_new
