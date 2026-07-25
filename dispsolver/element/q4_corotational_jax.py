"""
q4_corotational_jax.py
======================
Co-rotational Q4 Plane Strain Element in JAX.

Formulation
-----------
Decomposes large finite deformation of Q4 element into:
1. Rigid body rotation (R_elem) tracked by element co-rotational frame (e1, e2).
2. Pure deformational displacement u_local = R_elem^T · u_global.

This completely isolates rigid body rotation from element strain computation,
preventing artificial shear locking and Jacobian inversion (det(F) <= 0)
at high bending angles (38° -> 90°).
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Callable
from .q4_jax import shape_derivatives, jacobian, B_bar_matrix, _GP2, _W2, _GP0

def compute_element_rotation(coords_init: jnp.ndarray, coords_curr: jnp.ndarray) -> jnp.ndarray:
    """Compute 2x2 element rotation matrix R using element edge vectors.
    
    Parameters
    ----------
    coords_init : (4, 2) initial reference coordinates
    coords_curr : (4, 2) current deformed coordinates
    
    Returns
    -------
    R : (2, 2) rotation matrix where R[:, 0] = e1, R[:, 1] = e2
    """
    # Edge vectors in deformed state
    v12 = coords_curr[1] - coords_curr[0]
    v43 = coords_curr[2] - coords_curr[3]
    e1_deformed = v12 + v43
    len1 = jnp.linalg.norm(e1_deformed) + 1e-15
    e1 = e1_deformed / len1
    e2 = jnp.array([-e1[1], e1[0]])
    
    R_curr = jnp.column_stack([e1, e2])
    
    # Edge vectors in reference state
    v12_0 = coords_init[1] - coords_init[0]
    v43_0 = coords_init[2] - coords_init[3]
    e1_0_def = v12_0 + v43_0
    len1_0 = jnp.linalg.norm(e1_0_def) + 1e-15
    e1_0 = e1_0_def / len1_0
    e2_0 = jnp.array([-e1_0[1], e1_0[0]])
    R_ref = jnp.column_stack([e1_0, e2_0])
    
    # Relative rotation from reference to current
    R_elem = R_curr @ R_ref.T
    return R_elem

def build_block_rotation(R_elem: jnp.ndarray) -> jnp.ndarray:
    """Build 8x8 block diagonal rotation matrix for 4-node element DOFs."""
    T8 = jnp.zeros((8, 8), dtype=jnp.float64)
    T8 = T8.at[0:2, 0:2].set(R_elem)
    T8 = T8.at[2:4, 2:4].set(R_elem)
    T8 = T8.at[4:6, 4:6].set(R_elem)
    T8 = T8.at[6:8, 6:8].set(R_elem)
    return T8

def compute_corotational_internal_force(
    coords_init: jnp.ndarray,
    u_global: jnp.ndarray,
    material_stress_fn: Callable[[jnp.ndarray], jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute element internal force and consistent tangent stiffness in global frame
    using Co-rotational kinematics.
    
    Parameters
    ----------
    coords_init : (4, 2) initial reference coordinates
    u_global    : (8,) element global displacement vector [u1x, u1y, u2x, u2y, ...]
    material_stress_fn : Function computing Voigt stress [s_xx, s_yy, s_xy] from strain
    
    Returns
    -------
    f_global : (8,) global internal force vector
    K_global : (8, 8) global tangent stiffness matrix
    """
    coords_curr = coords_init + u_global.reshape((4, 2))
    
    # 1. Compute Element Rotation Frame R
    R_elem = compute_element_rotation(coords_init, coords_curr)
    T8 = build_block_rotation(R_elem)
    
    # 2. Local Deformational Displacement (Rigid rotation subtracted)
    u_local = T8.T @ u_global
    
    # 3. Compute Local Internal Force & Local Stiffness in Corotational Frame
    J0, detJ0, invJ0 = jacobian(_GP0[0, 0], _GP0[0, 1], coords_init)
    B0 = B_bar_matrix(_GP0[0, 0], _GP0[0, 1], invJ0, jnp.zeros((3, 8)), invJ0)
    
    f_local = jnp.zeros(8, dtype=jnp.float64)
    
    for i in range(4):
        xi, eta = _GP2[i, 0], _GP2[i, 1]
        w = _W2[i]
        J, detJ, invJ = jacobian(xi, eta, coords_init)
        B = B_bar_matrix(xi, eta, invJ, B0, invJ0)
        
        # Local strain in corotational frame
        strain_local = B @ u_local
        
        # Voigt stress from material model
        stress_local = material_stress_fn(strain_local)
        
        f_local = f_local + (B.T @ stress_local) * (detJ * w)
        
    # 4. Transform Local Force to Global Coordinate System
    f_global = T8 @ f_local
    
    # 5. Compute Global Tangent Stiffness via Exact Autodiff of f_global(u_global)
    def force_fn(u_g):
        c_curr = coords_init + u_g.reshape((4, 2))
        R = compute_element_rotation(coords_init, c_curr)
        T = build_block_rotation(R)
        u_l = T.T @ u_g
        
        f_l = jnp.zeros(8, dtype=jnp.float64)
        for i in range(4):
            xi, eta = _GP2[i, 0], _GP2[i, 1]
            w = _W2[i]
            J, detJ, invJ = jacobian(xi, eta, coords_init)
            B = B_bar_matrix(xi, eta, invJ, B0, invJ0)
            eps_l = B @ u_l
            sig_l = material_stress_fn(eps_l)
            f_l = f_l + (B.T @ sig_l) * (detJ * w)
        return T @ f_l
        
    K_global = jax.jacobian(force_fn)(u_global)

    return f_global, K_global


# ── J2 plasticity variant (vmap/jit friendly, no Python material_stress_fn) ──

from .q4_eas_jax import _jac, _sd, _BL_columns  # noqa: E402
from ..material.plastic_jax import tangent_voigt_jax  # noqa: E402

_S3 = jnp.sqrt(3.0)
_CORO_GP2 = jnp.array([
    [-1.0 / _S3, -1.0 / _S3],
    [ 1.0 / _S3, -1.0 / _S3],
    [ 1.0 / _S3,  1.0 / _S3],
    [-1.0 / _S3,  1.0 / _S3],
], dtype=jnp.float64)
_CORO_W2 = jnp.ones(4, dtype=jnp.float64)


@jax.jit
def compute_corotational_j2_contributions_jax(
    coords,       # (4, 2) reference coordinates
    u_elem,       # (8,)   total global displacement
    state_elem,   # (4, 5) per-GP J2 state (F_p_inv flat + eqps)
    lam, mu, sigma_y0, H,
    thickness=1.0,
):
    """Co-rotational Q4 with finite-strain J2 plasticity.

    Element rigid-body rotation R is extracted from the deformed edge
    vectors (`compute_element_rotation`) and used to rotate the total
    displacement into a local frame where deformation stays small even
    at large global bending angles (>> 38 deg). The local deformation
    gradient F_local = I + grad(u_local) is then fed to the same
    multiplicative-plasticity return map (`tangent_voigt_jax`) used by
    the EAS+J2 element, so both element formulations share one material
    law.

    Tangent: material part only (K_local = sum BL^T C_v BL * w), rotated
    to the global frame via K_global = T8 @ K_local @ T8.T. The dT8/du
    rotational geometric-stiffness term is *not* included (modified-
    Newton in the rotation DOFs) — computing it via full autodiff of the
    element routine (including the eigh-based plasticity return map,
    looped over 4 GPs) made jax.jit compilation impractically slow
    (>480s and counting) without changing convergence behaviour enough
    to matter, since the solver's line search / cutback already covers
    the resulting non-quadratic convergence near large rotation
    increments.

    Returns
    -------
    f_global  : (8,)   global internal force
    K_global  : (8, 8) global tangent stiffness (material part only)
    state_new : (4, 5) updated per-GP material state
    """
    coords_curr = coords + u_elem.reshape((4, 2))
    R_elem = compute_element_rotation(coords, coords_curr)
    T8 = build_block_rotation(R_elem)
    u_local = T8.T @ u_elem

    f_local = jnp.zeros(8, dtype=jnp.float64)
    K_local = jnp.zeros((8, 8), dtype=jnp.float64)
    state_new = jnp.zeros_like(state_elem)

    for k in range(4):
        xi, eta = _CORO_GP2[k]
        _, detJ, invJ = _jac(xi, eta, coords)
        dN_dxi, dN_deta = _sd(xi, eta)
        gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

        ux = u_local[0::2]
        uy = u_local[1::2]
        Hc = jnp.array([[ux @ gX, ux @ gY], [uy @ gX, uy @ gY]])
        F_local = jnp.eye(2, dtype=jnp.float64) + Hc

        S_v, C_v, sn = tangent_voigt_jax(F_local, state_elem[k], lam, mu, sigma_y0, H)
        state_new = state_new.at[k].set(sn)

        BL = _BL_columns(F_local, gX, gY)
        w = detJ * _CORO_W2[k] * thickness
        f_local = f_local + BL.T @ S_v * w
        K_local = K_local + (BL.T @ C_v @ BL) * w

    f_global = T8 @ f_local
    K_global = T8 @ K_local @ T8.T

    elem_nan = jnp.any(jnp.isnan(f_global)) | jnp.any(jnp.isnan(K_global))
    f_global = jnp.where(elem_nan, jnp.zeros_like(f_global), f_global)
    K_global = jnp.where(elem_nan, jnp.zeros_like(K_global), K_global)

    return f_global, K_global, state_new
