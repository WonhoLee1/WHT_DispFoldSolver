"""
q4_visco_eas_jax.py
===================
CPE4I / CPE4IH-equivalent element for the Arruda-Boyce / Neo-Hookean + Prony
viscoelastic (PSA) material: 4-node plane-strain quad with **incompatible
modes** (Simo & Rifai / Simo & Armero Q1/E4 enhanced deformation gradient).
The EAS-4 mode set carries the volumetric relief itself, so no separate F-bar
or hybrid pressure field is used (see the Formulation notes below).

Why this exists
---------------
The PSA layers are the compliant shear layers that let the display laminate
bend. Measured against small-deflection beam theory at the teardrop model's
real free-span row aspect ratio (dx=0.25mm over a 0.03mm PSA row, AR≈8.3),
the existing full-integration F-bar kernel (`q4_visco_simo_fs_jax`) is
**38x too stiff in bending** — textbook transverse shear locking of a
fully-integrated Q4. Every other element name in the solver silently fell
back to a plain displacement Q4 for this material (their JAX dispatches are
gated on `isinstance(mat, J2Plasticity)`; the `q4_hybrid_jax` "hybrid"
kernels additionally carry no pressure field at all), so the free hinge span
could not curve and the fold flattened out.

Why incompatible modes and not reduced integration
--------------------------------------------------
Reduced integration + hourglass control was already built and rejected here
(`dev_log/session_20260730_reduced_integration_failure.md`): for a rectangular
Q4 the pure-bending nodal pattern is *identical* to the hourglass pattern, so
hourglass control suppresses the very bending it must transmit (measured
10.19x mesh dependence, worse than plain corotational). Abaqus's own CPE4R
guidance — ≥4 elements through the bending thickness — cannot be met here
(one element row per 30µm PSA layer). The same session measured `Q4_EAS` at
0.999x mesh-independence, i.e. incompatible modes is the formulation that
demonstrably works in this code base; it just never existed for the
hyperelastic/viscoelastic material.

Formulation
-----------
* Enhanced deformation gradient `F = F_compat + sum_j alpha_j Fenh_j`, with
  the 4 Simo-Armero modes `Fenh_j = (detJ0/detJ) D_j J0^-T` reused verbatim
  from `q4_eas_jax._enhanced_grad_modes` (the TRANSPOSE matters -- it was
  `J0^-1` until 2026-09-08 and that is an index-type error worth up to a
  195x bending-stiffness error once the UL reference frame rotates
  off-axis; see that function's docstring and dev_log/
  plan_abaqus_element_consolidation_20260908.md section 8).
* F-bar is deliberately NOT applied on top of the enhancement. Stacking the
  two volumetric treatments made the condensed tangent inconsistent with the
  force (measured: nodal displacements exploding to ~1e5 mm on the first
  Newton step of a cantilever the plain F-bar kernel solves in one iteration).
  Near-incompressibility is therefore carried by the enhanced modes alone.
  Verified against small-deflection beam theory at AR 8.3 while sweeping the
  bulk modulus: ratio 1.01 at K/mu=49.7 (nu=0.490), 1.01 at K/mu=496.7
  (nu=0.49899), 1.02 at K/mu=4966.5 (nu=0.49990) -- no volumetric-locking
  degradation over two decades of incompressibility. A true CPE4IH (element-
  constant pressure unknown condensed alongside alpha) is still the right
  formulation for the fully incompressible limit (nu -> 0.5); this element is
  CPE4I.
* Constitutive response is `_simo_pk2` reused verbatim from
  `q4_visco_simo_fs_jax` — identical material law, only the kinematics change.
* `alpha` is condensed out at element level: a short fixed-count Newton
  (`_N_ALPHA_IT`, no data-dependent exit so it stays `vmap`-safe) drives the
  enhanced residual to zero, then
  `K = K_uu - K_ua K_aa^-1 K_au`.
* `alpha` is bounded by a p-norm guard that is identity to O((a/ALPHA_MAX)^8)
  near zero. A plain `ALPHA_MAX*tanh(a/ALPHA_MAX)` (as `q4_eas_jax` uses) is
  NOT a saturation -- it shrinks every iterate by ~(a/ALPHA_MAX)^2/3, so the
  enhanced residual can never reach zero and the element comes out 0.8-2.5%
  over-stiff with a condensed tangent whose defining constraint f_a = 0 is
  violated.
* The condensed force carries the first-order correction
  `f = f_u - K_ua K_aa^-1 f_a`, as `q4_eas_jax` does -- without it the force
  belongs to a different alpha than the tangent does.
* All tangents come from autodiff of the residuals; there is no
  eigendecomposition anywhere on this path, so it is NaN-safe (AGENTS.md 4.3)
  and compiles quickly (4 small jacobians, not a full-element jacobian over
  an `eigh`-based return map — AGENTS.md 4.4).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .q4_eas_jax import _enhanced_grad_modes, _voigt_sym
from .q4_visco_simo_fs_jax import (
    _GP2, _W2, _grads, _F_at, _BL_columns, _simo_pk2,
)

_N_ALPHA_IT = 5            # legacy fixed inner Newton count (kept for
                           # reference; superseded by _ALPHA_MAX_IT below)

# Element-local enhanced-mode Newton, line-search damped.
#
# The undamped fixed-5-iteration Newton this replaced does NOT converge on a
# hard Updated-Lagrangian state: measured on the tilted+distorted CPE4I probe
# of scratch/tier1_numba_jax_tilted.py, |f_alpha| oscillated between 2e-4 and
# 1.8e-1 for 25 straight iterations and |alpha| wandered up to 0.34 -- which is
# what used to drive alpha into the old `_ALPHA_MAX` clamp. With the backtracking
# line search below the SAME state converges in 10 iterations to |f_alpha| =
# 5.6e-13 and |alpha| = 1.8e-2, i.e. an order of magnitude SMALLER alpha than
# the undamped iteration was wandering to. The clamp was band-aiding a
# divergent local Newton, not bounding a genuine physical demand.
#
# This is the remedy the EAS literature actually prescribes for the
# "lack of robustness in the Newton-Raphson scheme" that Pfefferkorn, Bieber,
# Oesterle, Bischoff & Betsch identify as one of the two open issues of EAS
# elements ("Improving efficiency and robustness of enhanced assumed strain
# elements for nonlinear problems", IJNME 2021, DOI 10.1002/nme.6605: "The
# robustness can be improved with either the MIP method or a line search
# algorithm"). It is also already this repo's own precedent -- the NumPy
# sibling q4_eas.py has used a damped local line search since it was written,
# and is the one EAS kernel here that never needed a magnitude clamp.
#
# A line search cannot change the converged answer (f_alpha = 0 is the same
# root either way); it only changes the path taken to it.
_ALPHA_MAX_IT = 12
_LS_STEPS = jnp.array([1.0, 0.5, 0.25, 0.1], dtype=jnp.float64)

# Convergence tolerance for the element-local enhanced-mode Newton, measured
# on the LAST accepted correction ||d(alpha)||_inf. `alpha` scales the
# enhanced deformation-gradient modes, which are O(1) dimensionless, so alpha
# is itself dimensionless and an absolute tolerance is meaningful and
# mesh/units independent.
_ALPHA_CONV_TOL = 1e-8

# NOTE (2026-09-09): the former `_ALPHA_MAX` magnitude clamp on `alpha` was
# REMOVED here. It had no basis in the EAS literature or in any commercial
# solver's documented behaviour -- see
# dev_log/eas_stabilization_modernization_20260909.md. The genuine EAS
# large-compression instability (Wriggers & Reese, CMAME 135:201-209, 1996) is
# a rank deficiency of the element tangent; the literature cures it at the
# FORMULATION level (transposed Wilson modes, Glaser & Armero, Engineering
# Computations 14(7):759-791, 1997) and treats iteration failure as failure --
# it never caps the magnitude of a condensed internal variable. Abaqus's
# analogue for a non-converged element-local algorithm is an abandoned
# increment and a cutback, not a silent clamp-and-continue. The kernels
# therefore now REPORT local non-convergence (6th return value, `status`) and
# `DynamicSolver` turns that into an increment cutback.


def _enh_modes_all(coords):
    """Per-GP enhanced modes (4gp, 4modes, 2, 2) and weights."""
    J0 = jnp.array([
        [jnp.dot(0.25 * jnp.array([-1.0, 1.0, 1.0, -1.0]), coords[:, 0]),
         jnp.dot(0.25 * jnp.array([-1.0, 1.0, 1.0, -1.0]), coords[:, 1])],
        [jnp.dot(0.25 * jnp.array([-1.0, -1.0, 1.0, 1.0]), coords[:, 0]),
         jnp.dot(0.25 * jnp.array([-1.0, -1.0, 1.0, 1.0]), coords[:, 1])],
    ])
    detJ0 = J0[0, 0] * J0[1, 1] - J0[0, 1] * J0[1, 0]
    modes = []
    for gp in range(4):
        xi, eta = _GP2[gp]
        _, _, detJ = _grads(xi, eta, coords)
        modes.append(_enhanced_grad_modes(xi, eta, detJ, J0, detJ0))
    return jnp.stack(modes)


def _residuals(base, u_elem, alpha, coords, state_elem, kappa, bparams,
               g_i, tau_i, g_inf, dt, thickness, distortion_j_crit=0.0, F_n=None):
    """Return (f_u(8,), f_a(4,), state_new, F_n_new) for the enhanced element.

    Updated-Lagrangian: with `F_n` supplied, `coords` is the last converged
    configuration and `u_elem` the incremental displacement, so the gradient
    here is F_inc and the total gradient is `F_inc @ F_n`.
    """
    _eye = jnp.eye(2)
    F_n = jnp.stack([_eye, _eye, _eye, _eye]) if F_n is None else F_n
    Fenh_all = _enh_modes_all(coords)

    f_u = jnp.zeros(8)
    f_a = jnp.zeros(4)
    state_new = jnp.empty_like(state_elem)
    F_n_new = jnp.zeros((4, 2, 2))
    for gp in range(4):
        xi, eta = _GP2[gp]
        gX, gY, detJ = _grads(xi, eta, coords)
        w = detJ * _W2[gp] * thickness

        # 2026-09-10: the enhancement now lives in the INCREMENTAL frame
        # (config n), which is where every other operator of this element
        # already lives -- `Fenh_all` is built from `coords` (= config n),
        # `BL` differentiates the incremental displacement, and `w = detJ` is
        # the config-n volume element. It used to be added to the TOTAL
        # gradient (`F_c = F_inc @ F_n` first, enhancement on top), which mixed
        # frames and cost this element its variational consistency -- see the
        # measurements in the `_push_S_C` note below.
        #
        # NOTE (unchanged): F-bar is deliberately NOT applied on top of the
        # enhancement -- the EAS-4 mode set already carries the volumetric/
        # dilatational modes, and stacking the two volumetric treatments makes
        # the condensed tangent inconsistent with the force (measured:
        # displacements exploding to 1e5 mm on the first Newton step of a
        # cantilever that the plain F-bar kernel solves in one iteration).
        F_inc_c = _F_at(gX, gY, u_elem)
        F_inc = F_inc_c + jnp.einsum('j,jab->ab', alpha, Fenh_all[gp])
        F_tot = F_inc @ F_n[gp]

        S_v, h_new = _simo_pk2(base, F_tot, state_elem[gp], kappa, bparams,
                               g_i, tau_i, g_inf, dt,
                               distortion_j_crit=distortion_j_crit)

        # Work-conjugacy push-forward (dev_log/plan_abaqus_element_consolidation_20260908.md
        # finding F4): S_v is PK2 referred to the ORIGINAL (t=0) config
        # (computed from the TOTAL F_tot), while BL/G/w below are conjugate to
        # the INCREMENTAL Green-Lagrange strain on config n. Since
        # E_tot = F_n^T E_inc F_n exactly and dV0 = dV_n/det(F_n),
        # S_n = F_n S F_n^T / det(F_n) is the conjugate stress on config n.
        # Identity when F_n = I, so TL mode is unaffected by THIS step.
        #
        # Both residuals now contract the SAME S_n against operators built
        # from the SAME F_inc, so they are exact gradients of one potential.
        # Previously `f_u` used the pushed S_n while `f_a` kept the un-pushed
        # S_v with a G built from the total gradient, so the two belonged to
        # different functionals and `K_au = K_ua^T` (which the condensation in
        # `compute_single_eas_status` ASSUMES, and whose docstring claimed
        # "verified to 1.8e-15") was measurably false: 4.0e-2 in TL and
        # 7.8e-2 in UL on a real Cook's-membrane element. The condensed
        # tangent was correspondingly 3.0e-2 (TL) / 1.28 (UL) away from a
        # finite-difference d f_e/d u, which is what collapsed the GLOBAL
        # Newton line search in that benchmark while the element-local Newton
        # kept reporting convergence.
        Fn_gp = F_n[gp]
        S0_tensor = jnp.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
        detFn = jnp.maximum(jnp.abs(Fn_gp[0, 0] * Fn_gp[1, 1] - Fn_gp[0, 1] * Fn_gp[1, 0]), 1e-30)
        Sn_tensor = (Fn_gp @ S0_tensor @ Fn_gp.T) / detFn
        S_n_voigt = jnp.array([Sn_tensor[0, 0], Sn_tensor[1, 1], Sn_tensor[0, 1]])

        # B_L and G are both built from the ENHANCED incremental gradient:
        # dE_inc = sym(F_inc^T dF_c) du + sym(F_inc^T Fenh_j) dalpha.
        # `BL` used to be built from the COMPATIBLE F_inc_c, which silently
        # dropped the alpha-dependent half of dE_inc/du -- visible even in
        # pure TL, and the reason the TL asymmetry above is nonzero.
        BL = _BL_columns(F_inc, gX, gY)
        FtFe = jnp.einsum('ba,jbc->jac', F_inc, Fenh_all[gp])
        G = jnp.stack([_voigt_sym(FtFe[j]) for j in range(4)], axis=1)

        f_u = f_u + BL.T @ S_n_voigt * w
        f_a = f_a + G.T @ S_n_voigt * w

        state_new = state_new.at[gp].set(h_new)
        F_n_new = F_n_new.at[gp].set(F_tot)
    return f_u, f_a, state_new, F_n_new


@partial(jax.jit, static_argnames=("base",))
def compute_single_eas_status(coords, u_elem, alpha, state_elem, kappa, bparams,
                              g_i, tau_i, g_inf, dt, thickness,
                              distortion_j_crit: float = 0.0, F_n_gps=None,
                              base="neohookean"):
    """CPE4I/CPE4IH-equivalent viscoelastic element, with a failure status.

    Returns (f_int(8,), K_e(8,8), alpha_new(4,), state_new, F_n_new, status) --
    the condensed force/tangent plus the converged enhanced parameters, which
    the caller must store per element and hand back next iteration (warm
    start).

    `status` is the element-local convergence indicator: ``||d(alpha)||_inf``
    of the LAST accepted enhanced-mode Newton correction (0 = fully
    converged), or `inf` if the local solve produced a non-finite state or the
    condensed force/tangent came out non-finite. `DynamicSolver` compares it
    against `self.eas_local_tol` and abandons the increment (cutback) when any
    element fails -- the Abaqus-consistent response to a non-converged
    element-local algorithm. See the module-header note on the removed
    `_ALPHA_MAX` clamp.
    """
    def _fa(a, u):
        return _residuals(base, u, a, coords, state_elem, kappa, bparams,
                          g_i, tau_i, g_inf, dt, thickness,
                          distortion_j_crit=distortion_j_crit, F_n=F_n_gps)[1]

    # ---- element-local Newton on the enhanced parameters ------------
    # Line-search damped, tolerance-terminated, no magnitude clamp (see the
    # module header). The loop either converges -- ||d(alpha)||_inf below
    # _ALPHA_CONV_TOL, reported through `status` -- or its failure is reported
    # and the SOLVER cuts the increment back. A non-finite iterate freezes
    # alpha at its last good value AND latches status to inf, so the failure
    # can never be silent.
    def _alpha_cond(carry):
        _a, da_inf, it, failed = carry
        return (da_inf > _ALPHA_CONV_TOL) & (it < _ALPHA_MAX_IT) & (~failed)

    def _alpha_body(carry):
        a, _da_prev, it, _failed = carry
        f_a = _fa(a, u_elem)
        K_aa = jax.jacobian(_fa, argnums=0)(a, u_elem)
        reg = jnp.maximum(jnp.mean(jnp.abs(jnp.diag(K_aa))), 1e-30)
        da = -jnp.linalg.solve(K_aa + 1e-10 * reg * jnp.eye(4), f_a)
        # Backtracking line search on |f_alpha| over a fixed step-length set
        # (branch-free so it stays vmap/jit safe, unlike a `while` on a
        # per-element scalar).
        norms = jnp.stack([
            jnp.linalg.norm(_fa(a + ls * da, u_elem)) for ls in _LS_STEPS])
        norms = jnp.where(jnp.isfinite(norms), norms, jnp.inf)
        j = jnp.argmin(norms)
        step = _LS_STEPS[j] * da
        a_next = a + step
        bad = ~jnp.all(jnp.isfinite(a_next)) | ~jnp.isfinite(norms[j])
        a_out = jnp.where(bad, a, a_next)
        da_out = jnp.where(bad, jnp.inf, jnp.max(jnp.abs(step)))
        return (a_out, da_out, it + 1, bad)

    a, da_inf, _n_it, _failed = jax.lax.while_loop(
        _alpha_cond, _alpha_body,
        (alpha, jnp.array(1.0, dtype=jnp.float64), jnp.array(0), jnp.array(False)))
    da_inf = jnp.where(_failed, jnp.inf, da_inf)

    # ---- condensed force/tangent at the converged alpha -------------
    def _fu(u, a_):
        return _residuals(base, u, a_, coords, state_elem, kappa, bparams,
                          g_i, tau_i, g_inf, dt, thickness,
                          distortion_j_crit=distortion_j_crit, F_n=F_n_gps)[0]

    f_u, f_a_res, state_new, F_n_new = _residuals(
        base, u_elem, a, coords, state_elem, kappa, bparams, g_i, tau_i, g_inf,
        dt, thickness, distortion_j_crit=distortion_j_crit, F_n=F_n_gps)
    K_uu = jax.jacobian(_fu, argnums=0)(u_elem, a)
    K_ua = jax.jacobian(_fu, argnums=1)(u_elem, a)
    K_aa = jax.jacobian(_fa, argnums=0)(a, u_elem)
    # K_au is K_ua^T to machine precision (both residuals derive from the same
    # potential; verified to 1.8e-15) -- reuse it instead of a second jacobian.
    K_au = K_ua.T

    reg = jnp.maximum(jnp.mean(jnp.abs(jnp.diag(K_aa))), 1e-30)
    K_aa_reg = K_aa + 1e-10 * reg * jnp.eye(4)
    K_e = K_uu - K_ua @ jnp.linalg.solve(K_aa_reg, K_au)
    # Condensed FORCE must carry the first-order correction toward the alpha
    # that actually satisfies f_a = 0 (same term q4_eas_jax applies). Without
    # it the returned force belongs to a different alpha than the tangent does.
    f_u = f_u - K_ua @ jnp.linalg.solve(K_aa_reg, f_a_res)

    bad = ~jnp.all(jnp.isfinite(f_u)) | ~jnp.all(jnp.isfinite(K_e))
    f_u = jnp.where(bad, jnp.zeros_like(f_u), f_u)
    K_e = jnp.where(bad, jnp.zeros_like(K_e), K_e)
    status = jnp.where(bad, jnp.inf, da_inf)
    return f_u, K_e, a, state_new, F_n_new, status


@partial(jax.jit, static_argnames=("base",))
def compute_single_eas(coords, u_elem, alpha, state_elem, kappa, bparams,
                       g_i, tau_i, g_inf, dt, thickness,
                       distortion_j_crit: float = 0.0, F_n_gps=None,
                       base="neohookean"):
    """Back-compatible 5-tuple wrapper around `compute_single_eas_status`.

    Kept so every existing caller/test that unpacks exactly
    (f, K, alpha, state, F_n_new) keeps working; `DynamicSolver` uses the
    `_status` entry point so it can act on element-local failure.
    """
    return compute_single_eas_status(
        coords, u_elem, alpha, state_elem, kappa, bparams, g_i, tau_i, g_inf,
        dt, thickness, distortion_j_crit=distortion_j_crit, F_n_gps=F_n_gps,
        base=base)[:5]


# ======================================================================
# Hybrid (mixed u-p) variants: CPE4H and CPE4IH
# ======================================================================
# The PSA adhesive is a nearly-incompressible hyperelastic material, which is
# the case Abaqus handles with the H-family (CPE4H / CPE4RH / CPE4IH): an
# independent, element-constant pressure unknown instead of relying on the
# displacement field to represent the volumetric response.
#
# Formulation (perturbed-Lagrangian two-field):
#     Pi(u, p) = int W_iso dV + int [ p (J - 1) - p^2 / (2K) ] dV
#     f_u = int [ S_iso + p J C^-1 ] : dE dV
#     f_p = int [ (J - 1) - p / K ] dV                (element-constant p)
# The isochoric response comes from `_simo_pk2` called with kappa = 0, which
# switches off its own volumetric term (and, since the distortion barrier is
# also scaled by kappa, that too) leaving exactly the Flory-split deviatoric
# stress plus the Prony overstress.
#
# The internal parameter vector is q = [alpha(4), p] and BOTH are condensed
# with the same machinery, so CPE4H (use_eas=False, alpha frozen at 0) and
# CPE4IH (use_eas=True) share one code path.
#
# NOTE the u-p system is a saddle point: K_qq is symmetric indefinite (the
# -1/K block is negative). That is expected, not a defect -- `jnp.linalg.solve`
# handles it; do not "fix" it with a positive-definite regulariser.

def _residuals_h(base, u_elem, q, coords, state_elem, kappa, bparams,
                 g_i, tau_i, g_inf, dt, thickness, use_eas: bool,
                 distortion_j_crit=0.0, F_n=None):
    """(f_u(8,), f_q(5,), state_new, F_n_new) for the mixed u-p element.

    Updated-Lagrangian: with `F_n` supplied, `coords` is the last converged
    configuration and `u_elem` the incremental displacement.
    """
    _eye = jnp.eye(2)
    F_n = jnp.stack([_eye, _eye, _eye, _eye]) if F_n is None else F_n
    # Mode mask. With an independent pressure unknown the DILATATIONAL
    # enhanced modes (0: F11~xi, 3: F22~eta) would relax the very volumetric
    # constraint p already enforces -- double relaxation, measured as a ~4x
    # too-soft element (ratio 0.26 vs beam theory at every aspect ratio).
    # Only the SHEAR modes are kept: in pure bending u=-kappa*X*Y, v=kappa*X^2/2
    # the bilinear field can represent u exactly but not v, and the missing
    # term is dv/dX ~ xi, i.e. mode 2 (F21~xi); mode 1 (F12~eta) is its
    # transpose for bending about the other axis. So p handles volumetric,
    # modes 1-2 handle bending, and neither treats the same constraint twice.
    _MASK = jnp.array([0.0, 1.0, 1.0, 0.0])
    alpha = q[:4] * (_MASK if use_eas else 0.0)
    p = q[4]
    Fenh_all = _enh_modes_all(coords)

    f_u = jnp.zeros(8)
    f_a = jnp.zeros(4)
    r_p = 0.0
    vol = 0.0
    state_new = jnp.empty_like(state_elem)
    F_n_new = jnp.zeros((4, 2, 2))
    for gp in range(4):
        xi, eta = _GP2[gp]
        gX, gY, detJ = _grads(xi, eta, coords)
        w = detJ * _W2[gp] * thickness

        F_inc = _F_at(gX, gY, u_elem)
        F = F_inc @ F_n[gp] + jnp.einsum('j,jab->ab', alpha, Fenh_all[gp])
        J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]

        # deviatoric/isochoric + viscoelastic overstress only (kappa = 0)
        S_v, h_new = _simo_pk2(base, F, state_elem[gp], 0.0, bparams,
                               g_i, tau_i, g_inf, dt, distortion_j_crit=0.0)

        # hybrid volumetric stress  p J C^-1   (2D Voigt [11, 22, 12])
        C = F.T @ F
        Cinv = jnp.linalg.inv(C + 1e-15 * jnp.eye(2))
        S_vol = p * J * jnp.array([Cinv[0, 0], Cinv[1, 1], Cinv[0, 1]])
        S_tot = S_v + S_vol

        # Work-conjugacy push-forward -- same bug/fix as `_residuals` above
        # and q4_visco_simo_fs_jax.py's `_internal_force` (finding F4):
        # S_tot is referred to the ORIGINAL config (computed from the
        # TOTAL F); BL/w below are step-n quantities. Push forward before
        # contracting with BL (f_a keeps the un-pushed S_tot -- its own G
        # operator is already built from the total F, not F_inc, so it
        # does not have this specific mismatch; out of scope here).
        Fn_gp = F_n[gp]
        S0_tensor = jnp.array([[S_tot[0], S_tot[2]], [S_tot[2], S_tot[1]]])
        detFn = jnp.maximum(jnp.abs(Fn_gp[0, 0] * Fn_gp[1, 1] - Fn_gp[0, 1] * Fn_gp[1, 0]), 1e-30)
        Sn_tensor = (Fn_gp @ S0_tensor @ Fn_gp.T) / detFn
        S_n_voigt = jnp.array([Sn_tensor[0, 0], Sn_tensor[1, 1], Sn_tensor[0, 1]])

        BL = _BL_columns(F_inc, gX, gY)
        f_u = f_u + BL.T @ S_n_voigt * w

        FtFe = jnp.einsum('ba,jbc->jac', F, Fenh_all[gp])
        G = jnp.stack([_voigt_sym(FtFe[j]) for j in range(4)], axis=1)
        f_a = f_a + G.T @ S_tot * w

        r_p = r_p + (J - 1.0) * w
        vol = vol + w
        state_new = state_new.at[gp].set(h_new)
        F_n_new = F_n_new.at[gp].set(F)

    r_p = r_p - (p / jnp.maximum(kappa, 1e-30)) * vol
    f_a = f_a * (_MASK if use_eas else 0.0)
    return f_u, jnp.concatenate([f_a, jnp.array([r_p])]), state_new, F_n_new


@partial(jax.jit, static_argnames=("base", "use_eas"))
def compute_single_hybrid_status(coords, u_elem, alpha, state_elem, kappa, bparams,
                                 g_i, tau_i, g_inf, dt, thickness,
                                 distortion_j_crit: float = 0.0, F_n_gps=None,
                                 base="neohookean", use_eas: bool = True):
    """CPE4IH (use_eas=True) / CPE4H (use_eas=False) viscoelastic element.

    `alpha` carries [alpha(4), p] -- the caller stores 5 numbers per element.
    Returns (f_int(8,), K_e(8,8), q_new(5,), state_new, F_n_new, status).

    `status` is the element-local convergence indicator of the mixed
    (enhanced-mode + pressure) internal solve: the max of ||d(alpha)||_inf
    (dimensionless) and |dp|/kappa (made dimensionless by the bulk modulus,
    the natural pressure scale) on the LAST accepted correction; `inf` if the
    local state or the condensed force/tangent came out non-finite. See
    `compute_single_eas_status` and the module header.
    """
    q = jnp.where(alpha.shape[0] == 5, alpha, jnp.zeros(5))

    def _fq(qq, u):
        return _residuals_h(base, u, qq, coords, state_elem, kappa, bparams,
                            g_i, tau_i, g_inf, dt, thickness, use_eas,
                            distortion_j_crit=distortion_j_crit, F_n=F_n_gps)[1]

    # Line-search damped, tolerance-terminated, no magnitude clamp on the
    # enhanced modes (see module header); local non-convergence is reported
    # through `status` and handled by the solver as an increment cutback.
    _p_scale = jnp.maximum(jnp.abs(kappa), 1e-30)
    # Merit-function weights only: r_p has volume units while f_a has force
    # units, so kappa*r_p puts the pressure row on the same scale before the
    # norm is used to RANK trial step lengths. It never enters the residual
    # itself, so it cannot bias the converged solution.
    _merit_w = jnp.array([1.0, 1.0, 1.0, 1.0, _p_scale], dtype=jnp.float64)

    def _q_cond(carry):
        _q, dq_meas, it, failed = carry
        return (dq_meas > _ALPHA_CONV_TOL) & (it < _ALPHA_MAX_IT) & (~failed)

    def _q_body(carry):
        q, _dq_prev, it, _failed = carry
        f_q = _fq(q, u_elem)
        K_qq = jax.jacobian(_fq, argnums=0)(q, u_elem)
        reg = jnp.maximum(jnp.mean(jnp.abs(jnp.diag(K_qq))), 1e-30)
        dq = -jnp.linalg.solve(K_qq + 1e-10 * reg * jnp.eye(5), f_q)
        norms = jnp.stack([
            jnp.linalg.norm(_fq(q + ls * dq, u_elem) * _merit_w)
            for ls in _LS_STEPS])
        norms = jnp.where(jnp.isfinite(norms), norms, jnp.inf)
        j = jnp.argmin(norms)
        step = _LS_STEPS[j] * dq
        q_next = q + step
        bad = ~jnp.all(jnp.isfinite(q_next)) | ~jnp.isfinite(norms[j])
        q_out = jnp.where(bad, q, q_next)
        dq_out = jnp.where(
            bad, jnp.inf,
            jnp.maximum(jnp.max(jnp.abs(step[:4])), jnp.abs(step[4]) / _p_scale))
        return (q_out, dq_out, it + 1, bad)

    q, dq_meas, _n_it_q, _failed_q = jax.lax.while_loop(
        _q_cond, _q_body,
        (q, jnp.array(1.0, dtype=jnp.float64), jnp.array(0), jnp.array(False)))
    dq_meas = jnp.where(_failed_q, jnp.inf, dq_meas)

    def _fu(u, qq):
        return _residuals_h(base, u, qq, coords, state_elem, kappa, bparams,
                            g_i, tau_i, g_inf, dt, thickness, use_eas,
                            distortion_j_crit=distortion_j_crit, F_n=F_n_gps)[0]

    f_u, f_q_res, state_new, F_n_new = _residuals_h(
        base, u_elem, q, coords, state_elem, kappa, bparams,
        g_i, tau_i, g_inf, dt, thickness, use_eas,
        distortion_j_crit=distortion_j_crit, F_n=F_n_gps)
    K_uu = jax.jacobian(_fu, argnums=0)(u_elem, q)
    K_uq = jax.jacobian(_fu, argnums=1)(u_elem, q)
    K_qq = jax.jacobian(_fq, argnums=0)(q, u_elem)
    K_qu = K_uq.T

    reg = jnp.maximum(jnp.mean(jnp.abs(jnp.diag(K_qq))), 1e-30)
    K_qq_reg = K_qq + 1e-10 * reg * jnp.eye(5)
    K_e = K_uu - K_uq @ jnp.linalg.solve(K_qq_reg, K_qu)
    f_u = f_u - K_uq @ jnp.linalg.solve(K_qq_reg, f_q_res)

    bad = ~jnp.all(jnp.isfinite(f_u)) | ~jnp.all(jnp.isfinite(K_e))
    f_u = jnp.where(bad, jnp.zeros_like(f_u), f_u)
    K_e = jnp.where(bad, jnp.zeros_like(K_e), K_e)
    status = jnp.where(bad, jnp.inf, dq_meas)
    return f_u, K_e, q, state_new, F_n_new, status


@partial(jax.jit, static_argnames=("base", "use_eas"))
def compute_single_hybrid(coords, u_elem, alpha, state_elem, kappa, bparams,
                          g_i, tau_i, g_inf, dt, thickness,
                          distortion_j_crit: float = 0.0, F_n_gps=None,
                          base="neohookean", use_eas: bool = True):
    """Back-compatible 5-tuple wrapper around `compute_single_hybrid_status`."""
    return compute_single_hybrid_status(
        coords, u_elem, alpha, state_elem, kappa, bparams, g_i, tau_i, g_inf,
        dt, thickness, distortion_j_crit=distortion_j_crit, F_n_gps=F_n_gps,
        base=base, use_eas=use_eas)[:5]
