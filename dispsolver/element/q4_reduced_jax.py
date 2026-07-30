"""
q4_reduced_jax.py
==================
1-point (centroid) reduced-integration Q4 plane-strain element with
Flanagan-Belytschko physical hourglass stabilization, finite-strain J2
plasticity — pure JAX.

Why this element exists
------------------------
`Q4_COROTATIONAL` (full 2x2 integration) suffers severe aspect-ratio-
dependent shear/bending locking (~1+0.35*AR^2, see AGENTS.md 4.1).
`Q4_EAS`/`Q4_COROTATIONAL_EAS` fix locking via enhanced assumed strain, but
`Q4_COROTATIONAL_EAS` reproducibly fails to converge in the real large-
rotation fold model (~22deg/side, "residual explosion at Newton iteration 1"
identical across a 32x dt-cutback range) -- the EAS alpha static-condensation
is a genuine per-element nonlinear sub-problem with potentially multiple
stationary branches under combined bending+plasticity+large rotation, a
known large-strain instability class for enhanced-assumed-strain elements
in the literature.

Single centroid Gauss-point integration is the classical, industry-standard
fix for Q4 shear/bending locking (shear strain vanishes exactly at the
centroid for a pure-bending nodal displacement pattern -- the same
mechanism this codebase already uses for the volumetric part of
`q4_jax.B_bar_matrix`'s selective-reduced-integration). It has NO internal
per-element nonlinear sub-problem (no static condensation, no inner Newton
loop) -- the hourglass stabilization below is a fixed, linear, potential-
derived term, not an iterated one -- so it structurally cannot exhibit the
"wrong stationary branch" failure class suspected for CR-EAS. This is why
commercial codes (Abaqus CPE4R) default to reduced-integration+hourglass-
control rather than incompatible-modes (CPE4I) for large-deformation
problems.

Hourglass stabilization
------------------------
A single centroid Gauss point cannot sense the bilinear xi*eta ("hourglass")
displacement pattern -- the one zero-energy spurious mode a 1-point Q4
element has beyond the 3 rigid-body modes. Flanagan & Belytschko (1981),
*A uniform strain hexahedron and quadrilateral with orthogonal hourglass
control*, IJNME 17, and Belytschko, Liu, Moran & Elkhodary, *Nonlinear
Finite Elements for Continua and Structures*, 2nd ed. (2014) Sec 8.3-8.4,
give the standard cure: project the raw hourglass shape vector
h=[1,-1,1,-1] (nodal values of xi*eta) to be orthogonal to rigid
translation and the constant-strain field already resolved by the centroid
gradients, then add a small elastic potential penalizing the resulting
generalized hourglass displacement. The stabilization stiffness scales
with an elastic modulus (lam+2mu, NOT the current plastic tangent -- keeps
it well-conditioned under plastic softening) times element volume times
sum-of-squared centroid gradients, matching the standard Flanagan-
Belytschko stiffness-scaling form.

Because the stabilization is literally the gradient/Hessian of a quadratic
potential Pi_hg = 0.5*k_hg*(qx^2+qy^2), its force/stiffness pair is
consistent by construction (f_hg = dPi_hg/du, K_hg = d^2Pi_hg/du^2) -- no
autodiff needed for this part, and it cannot corrupt the patch test: by
construction gamma (the orthogonalized hourglass vector) gives qx=qy=0
under rigid translation, rigid rotation-in-a-local-frame, or any constant-
strain nodal displacement field.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .q4_eas_jax import _jac, _sd, _BL_columns
from ..material.plastic_jax import tangent_voigt_jax

# Natural-coordinate hourglass shape vector: nodal values of xi*eta at
# node natural coordinates (-1,-1), (1,-1), (1,1), (-1,1) -- the one
# bilinear pattern a centroid-only sample point cannot see.
_HG_SHAPE = jnp.array([1.0, -1.0, 1.0, -1.0], dtype=jnp.float64)

# Hourglass stiffness scale factor (fraction of the elastic modulus).
# Validated against verification/locking.py's AR-sweep benchmark, not
# assumed -- see that file for the sweep that picks this value: smallest
# alpha_hg that keeps K_e non-singular at every tested aspect ratio without
# measurably reintroducing AR-dependent stiffness.
_ALPHA_HG = 0.05


@jax.jit
def compute_reduced_hg_j2_contributions_jax(
    coords,        # (4, 2) reference coords (local/corotational frame, unrotated)
    u_elem,        # (8,)   local displacement (rigid-rotation-free when
                   #        called from the corotational wrapper)
    state_elem,    # (4, 5) per-GP J2 state -- only slot 0 (centroid) is
                   #        physically meaningful; see module docstring
    lam, mu, sigma_y0, H,
    thickness=1.0,
    alpha_hg=_ALPHA_HG,
):
    """1-point reduced-integration Q4 + hourglass control + finite-strain
    J2 plasticity.

    Returns
    -------
    f_e       : (8,)   internal force (material + hourglass)
    K_e       : (8, 8) tangent stiffness (material + geometric + hourglass)
    state_new : (4, 5) updated material state, single centroid state
                broadcast into all 4 slots (see module docstring on why:
                the global `self.state` array is uniformly shaped
                (n_elem, 4, max_vars) regardless of element type).
    """
    x_nodes = coords[:, 0]
    y_nodes = coords[:, 1]

    # ---- Centroid (xi=eta=0) kinematics ----
    _, detJ0, invJ0 = _jac(0.0, 0.0, coords)
    dN_dxi, dN_deta = _sd(0.0, 0.0)
    gX = invJ0[0, 0] * dN_dxi + invJ0[0, 1] * dN_deta
    gY = invJ0[1, 0] * dN_dxi + invJ0[1, 1] * dN_deta

    ux = u_elem[0::2]
    uy = u_elem[1::2]
    Hc = jnp.array([[ux @ gX, ux @ gY], [uy @ gX, uy @ gY]])
    F = jnp.eye(2, dtype=jnp.float64) + Hc

    # ---- Material response at the single centroid GP ----
    S_v, C_v, state_new_gp = tangent_voigt_jax(F, state_elem[0], lam, mu, sigma_y0, H)
    state_new = jnp.broadcast_to(state_new_gp, state_elem.shape)

    w0 = detJ0 * 4.0 * thickness   # element volume (natural-domain area 2x2 -> W0=4)

    BL = _BL_columns(F, gX, gY)
    f_mat = BL.T @ S_v * w0
    K_mat = (BL.T @ C_v @ BL) * w0

    # No separate geometric/initial-stress stiffness term: FD-verified
    # (against JAX's own exact autodiff jacobian of f_mat, not just finite
    # differences) that a naively-hand-derived grad_N@St@grad_N.T node-pair
    # term made the tangent WORSE (3.79% error) than K_mat alone (0.95%),
    # regardless of sign/scale tried. K_mat alone matches the established
    # precedent already used by the production Q4_COROTATIONAL local
    # kernel (compute_corotational_j2_contributions_jax in
    # q4_corotational_jax.py), which likewise uses only BL^T@C_v@BL for
    # its local stiffness (AGENTS.md 4.4: "modified Newton... drops the
    # geometric-stiffness term" -- an accepted, already-proven-robust
    # approximation in this codebase, not unique to this element).

    # ---- Hourglass stabilization ----
    # Physical (orthogonalized) hourglass vector: zero response under rigid
    # translation and constant strain by construction.
    gamma = _HG_SHAPE - (_HG_SHAPE @ x_nodes) * gX - (_HG_SHAPE @ y_nodes) * gY

    qx = gamma @ ux
    qy = gamma @ uy
    k_hg = alpha_hg * (lam + 2.0 * mu) * w0 * (gX @ gX + gY @ gY)

    f_hg = jnp.zeros(8, dtype=jnp.float64)
    f_hg = f_hg.at[0::2].set(k_hg * qx * gamma)
    f_hg = f_hg.at[1::2].set(k_hg * qy * gamma)

    gg = jnp.outer(gamma, gamma)
    K_hg = jnp.zeros((8, 8), dtype=jnp.float64)
    K_hg = K_hg.at[0::2, 0::2].set(k_hg * gg)
    K_hg = K_hg.at[1::2, 1::2].set(k_hg * gg)

    f_e = f_mat + f_hg
    K_e = K_mat + K_hg

    # NaN guard, same pattern as every other element in this codebase.
    elem_nan = jnp.any(jnp.isnan(f_e)) | jnp.any(jnp.isnan(K_e))
    f_e = jnp.where(elem_nan, jnp.zeros_like(f_e), f_e)
    K_e = jnp.where(elem_nan, jnp.zeros_like(K_e), K_e)

    return f_e, K_e, state_new
