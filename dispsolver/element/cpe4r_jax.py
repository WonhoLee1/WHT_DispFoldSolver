"""
cpe4r_jax.py
============
Abaqus CPE4R -- 4-node bilinear plane-strain quadrilateral, reduced
(1-point, uniform-strain) integration, hourglass control -- with finite-
strain J2 plasticity. Pure JAX.

Spec vs. derivation
-------------------
VERBATIM SPEC (Abaqus documentation; quoted, not paraphrased, where marked):

* Element: "CPE4R" = 4-node bilinear plane strain quadrilateral, reduced
  integration, hourglass control (Abaqus Analysis User's Guide, "Solid
  (continuum) elements").
* *SECTION CONTROLS, HOURGLASS= (Keywords Guide):
  "Set HOURGLASS=ENHANCED (default for elements with hyperelastic and
  hyperfoam materials in Abaqus/Standard ...)";
  "Set HOURGLASS=STIFFNESS (default for Abaqus/Standard, except for
  elements with hyperelastic and hyperfoam materials ...)".
  So for the J2Plasticity material this kernel serves, the Abaqus/Standard
  default is STIFFNESS -- `DynamicSolver` resolves an unset `hourglass` to
  that. ENHANCED is selected explicitly (the ex12 fold sets it in
  `SolverTuningConfig.hourglass`, the equivalent of writing
  HOURGLASS=ENHANCED in the deck).
* Data line item 1, `ss`: "If this value is left blank, Abaqus will use the
  default value of 1.0" -- the displacement hourglass scale factor.
* Total stiffness approach (Analysis Guide "Section controls"): for
  membrane or solid elements the hourglass stiffness factor is
  "0.005 G", "G is the shear modulus", scaled by `ss`.
* Enhanced hourglass control (Analysis Guide "Section controls" + Theory
  Guide 3.2.4): "represents a refinement of the pure stiffness method";
  "based on the enhanced assumed strain method"; "gives more accurate
  displacement solutions for coarse meshes with linear elastic materials";
  "There are no hourglass stiffness factors or scale factors for the
  enhanced formulation"; per the Theory Guide it follows Engelmann &
  Whirley (1990), Belytschko & Bindeman (1992) and Puso (2000), and for
  linear elastic materials its displacements match the incompatible-mode
  (CPE4I) element "since both use the same assumed strain formulation".

DERIVATION (this file; the Abaqus pages give the equations only as images,
so the exact Abaqus coefficients could not be transcribed):

1. Uniform strain. In 2D the uniform-strain (element-average) gradient of
   a bilinear quad IS the centroid gradient, exactly, on any geometry
   (AGENTS.md 4.18), so the single integration point sits at the centroid
   with weight = element area.

2. ENHANCED = the exact hourglass part of the linear-elastic CPE4I (Q1E4)
   stiffness. Let b = centroid gradients, h = [1,-1,1,-1] (nodal values of
   xi*eta) and gamma = h - (h.x) b_x - (h.y) b_y the Flanagan-Belytschko
   hourglass SHAPE vector, so gamma . {1, x, y} = 0 and gamma . h = 4.
   Any nodal field splits uniquely as u = u_lin + h (x) c with u_lin
   linear, and q = (gamma.u_x, gamma.u_y) = 4c. For the elastic Q1E4
   stiffness K_E:
     * u_lin K_E u_lin = V eps:D:eps  (Q1E4 passes the patch test);
     * u_lin K_E (h (x) c) = sigma_0 : integral sym(c (x) grad(xi*eta)) dV
       = 0, because integral grad(xi*eta) dA = closed-contour integral of
       xi*eta n ds, and xi*eta is linear and antisymmetric along every
       straight quad edge -- true for ANY quad, not only parallelograms;
     * the centroid strain of u equals that of u_lin (b . h = 0).
   Hence, exactly and for any geometry,
       K_E = K_1pt + Gamma^T (H/16) Gamma,
       H_ij = (h (x) e_i)^T K_E (h (x) e_j)        (2x2),
   which is what this kernel adds as its hourglass stiffness: at small
   strain CPE4R/ENHANCED and CPE4I have the SAME stiffness matrix (checked
   in tests/test_cpe4r_element.py), so bending is as accurate as CPE4I at
   any aspect ratio and the 07-30 mesh dependence cannot occur. H uses the
   ELASTIC moduli (lam, mu), never the plastic tangent -- the "physical
   stabilization" convention of Belytschko & Bindeman / Puso, and what
   keeps the stabilization well-conditioned under plastic flow.

3. Large rotation. q = Gamma u = Gamma x (gamma . X = 0), so a rigid motion
   gives q = 0 and no hourglass force at all. H is however anisotropic
   (aspect ratio), so the energy is evaluated with q pulled back to the
   reference frame by the rotation R of the centroid polar decomposition
   F_c = R U:  Pi_hg = 1/2 (R^T q)^T (H/16) (R^T q). In 2D R is closed
   form, cos = (F11+F22)/n, sin = (F21-F12)/n -- no atan2 branch cut. The
   hourglass force and tangent are the exact gradient and Hessian of Pi_hg
   (jax.grad / jax.hessian of an 8-DOF closed-form scalar; no plasticity
   inside, so none of the AGENTS.md 4.4 compile-time cliff), INCLUDING the
   dR/du terms -- not a modified-Newton tangent.

4. STIFFNESS = Flanagan-Belytschko (1981) form with the Abaqus coefficient:
   Pi_hg = 1/2 k |q|^2,  k = ss * 0.005 * G * V * (b_x.b_x + b_y.b_y).
   The V*sum(b^2) normalisation is FB's; the 0.005 G is the Abaqus default
   factor. Isotropic in q, so the frame does not enter. This is the
   arbitrary-stiffness form whose bending stiffness is mesh dependent --
   kept for Abaqus fidelity and as the comparison baseline, see
   dev_log/session_20260730_reduced_integration_failure.md (that file's
   q4_reduced_jax.py used 0.05*(lam+2mu) instead of 0.005*G).

5. The 1-point material part is Total Lagrangian (F = I + grad u at the
   centroid, PK2/Green-Lagrange), hence objective without any frame: its
   consistent tangent is B_L^T C B_L + the initial-stress term
   grad_N S grad_N^T (both included). The element therefore runs TL under
   NLGEOM (`_ELEMENT_LARGE_DEF["CPE4R"].supports_ul = False`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .q4_eas_jax import _jac, _sd, _BL_columns, _enhanced_grad_modes, _GP2
from ..material.plastic_jax import tangent_voigt_jax

# Nodal values of xi*eta at (-1,-1), (1,-1), (1,1), (-1,1).
_HG = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)

HOURGLASS_MODES = ("ENHANCED", "STIFFNESS")

# Abaqus total-stiffness default factor for membrane/solid elements (x G).
_ABAQUS_HG_FACTOR = 0.005


def _centroid_grads(coords):
    _, detJ0, invJ0 = _jac(0.0, 0.0, coords)
    dN_dxi, dN_deta = _sd(0.0, 0.0)
    gX = invJ0[0, 0] * dN_dxi + invJ0[0, 1] * dN_deta
    gY = invJ0[1, 0] * dN_dxi + invJ0[1, 1] * dN_deta
    return gX, gY, detJ0


def _hg_shape_vector(coords, gX, gY):
    """Flanagan-Belytschko hourglass shape vector (orthogonal to {1, x, y})."""
    return _HG - (_HG @ coords[:, 0]) * gX - (_HG @ coords[:, 1]) * gY


def q1e4_elastic_stiffness(coords, lam, mu):
    """Small-strain linear-elastic Q1E4 (CPE4I) stiffness, unit thickness.

    Same enhanced modes as q4_eas_jax (`_enhanced_grad_modes`, J0^-T form),
    plane-strain isotropic D.
    """
    D = jnp.array([[lam + 2 * mu, lam, 0.0],
                   [lam, lam + 2 * mu, 0.0],
                   [0.0, 0.0, mu]], dtype=jnp.float64)
    J0, detJ0, _ = _jac(0.0, 0.0, coords)
    K_uu = jnp.zeros((8, 8), dtype=jnp.float64)
    K_ua = jnp.zeros((8, 4), dtype=jnp.float64)
    K_aa = jnp.zeros((4, 4), dtype=jnp.float64)
    for k in range(4):
        xi, eta = _GP2[k]
        _, detJ, invJ = _jac(xi, eta, coords)
        dN_dxi, dN_deta = _sd(xi, eta)
        gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        B = jnp.zeros((3, 8), dtype=jnp.float64)
        B = B.at[0, 0::2].set(gX).at[1, 1::2].set(gY)
        B = B.at[2, 0::2].set(gY).at[2, 1::2].set(gX)
        Fe = _enhanced_grad_modes(xi, eta, detJ, J0, detJ0)          # (4,2,2)
        G = jnp.stack([Fe[:, 0, 0], Fe[:, 1, 1], Fe[:, 0, 1] + Fe[:, 1, 0]])  # (3,4)
        K_uu = K_uu + B.T @ D @ B * detJ
        K_ua = K_ua + B.T @ D @ G * detJ
        K_aa = K_aa + G.T @ D @ G * detJ
    return K_uu - K_ua @ jnp.linalg.solve(K_aa, K_ua.T)


def enhanced_hourglass_matrix(coords, lam, mu):
    """H/16 of the module docstring: Pi_hg = 1/2 q_ref^T (H/16) q_ref."""
    K_E = q1e4_elastic_stiffness(coords, lam, mu)
    hx = jnp.zeros(8, dtype=jnp.float64).at[0::2].set(_HG)
    hy = jnp.zeros(8, dtype=jnp.float64).at[1::2].set(_HG)
    Hv = jnp.stack([hx, hy])                                         # (2,8)
    return (Hv @ K_E @ Hv.T) / 16.0


def _hg_energy(u_elem, gX, gY, gamma, Hq):
    ux = u_elem[0::2]
    uy = u_elem[1::2]
    a = 2.0 + ux @ gX + uy @ gY          # F11 + F22
    b = uy @ gX - ux @ gY                # F21 - F12
    n = jnp.sqrt(a * a + b * b)
    c, s = a / n, b / n
    q = jnp.array([gamma @ ux, gamma @ uy])
    q_ref = jnp.array([c * q[0] + s * q[1], -s * q[0] + c * q[1]])   # R^T q
    return 0.5 * q_ref @ Hq @ q_ref


_hg_force = jax.grad(_hg_energy)
_hg_tangent = jax.hessian(_hg_energy)


def compute_cpe4r_j2_contributions_jax(
    coords,        # (4, 2) reference coords (Total Lagrangian)
    u_elem,        # (8,)   total displacement
    state_elem,    # (4, n) J2 state; slot 0 is the integration point
    lam, mu, sigma_y0, H,
    hourglass="ENHANCED",
    hg_scale=1.0,
):
    """CPE4R + finite-strain J2, unit thickness (the caller scales).

    `hourglass` is a Python string fixed at trace time. `hg_scale` is
    Abaqus' `ss`; it applies to STIFFNESS only ("no ... scale factors for
    the enhanced formulation").

    Returns (f_e (8,), K_e (8,8), state_new (4,n)); the single integration
    point's state is broadcast into all 4 slots because the solver's state
    array is uniformly (n_elem, 4, n_vars).
    """
    if hourglass not in HOURGLASS_MODES:
        raise ValueError(f"hourglass={hourglass!r}; expected one of {HOURGLASS_MODES}")

    gX, gY, detJ0 = _centroid_grads(coords)
    V = 4.0 * detJ0

    # ---- 1-point (uniform-strain) material part ----
    ux = u_elem[0::2]
    uy = u_elem[1::2]
    F = jnp.eye(2, dtype=jnp.float64) + jnp.array([[ux @ gX, ux @ gY],
                                                   [uy @ gX, uy @ gY]])
    S_v, C_v, sn = tangent_voigt_jax(F, state_elem[0], lam, mu, sigma_y0, H)
    state_new = jnp.broadcast_to(sn, state_elem.shape)

    BL = _BL_columns(F, gX, gY)
    St = jnp.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
    grad_N = jnp.stack([gX, gY], axis=1)                             # (4,2)
    k_sig = grad_N @ St @ grad_N.T
    K_geo = jnp.zeros((8, 8), dtype=jnp.float64)
    K_geo = K_geo.at[0::2, 0::2].set(k_sig).at[1::2, 1::2].set(k_sig)

    f_e = BL.T @ S_v * V
    K_e = (BL.T @ C_v @ BL + K_geo) * V

    # ---- hourglass control ----
    gamma = _hg_shape_vector(coords, gX, gY)
    if hourglass == "ENHANCED":
        Hq = enhanced_hourglass_matrix(coords, lam, mu)
    else:
        k = hg_scale * _ABAQUS_HG_FACTOR * mu * V * (gX @ gX + gY @ gY)
        Hq = k * jnp.eye(2, dtype=jnp.float64)
    f_e = f_e + _hg_force(u_elem, gX, gY, gamma, Hq)
    K_e = K_e + _hg_tangent(u_elem, gX, gY, gamma, Hq)

    # NaN guard, same pattern as every other element in this codebase.
    elem_nan = ~jnp.all(jnp.isfinite(f_e)) | ~jnp.all(jnp.isfinite(K_e))
    f_e = jnp.where(elem_nan, jnp.zeros_like(f_e), f_e)
    K_e = jnp.where(elem_nan, jnp.zeros_like(K_e), K_e)
    return f_e, K_e, state_new
