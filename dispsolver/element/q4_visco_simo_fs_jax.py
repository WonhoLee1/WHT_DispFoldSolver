"""
q4_visco_simo_fs_jax.py
=======================
Finite-strain (large-deformation) viscoelastic Q4 hybrid element — pure JAX,
with a *pluggable* hyperelastic ground state and a Flory volumetric/isochoric
split so the relaxation acts on a genuinely deviatoric (isochoric) stress that
vanishes at the undeformed state.

Technical sources
-----------------
* Flory volumetric/isochoric kinematic split:
    Flory, P.J. (1961). "Thermodynamic relations for high elastic materials."
        Trans. Faraday Soc. 57, 829-838.
        F = (J^{1/3} I) Fbar ,  Fbar = J^{-1/3} F ,  det Fbar = 1 ,  J = det F.
  Isochoric PK2 for an Ibar1-based potential (Holzapfel 2000, "Nonlinear Solid
  Mechanics", Eq. 6.88-6.91):
        S_iso = 2 W1 J^{-2/3} ( I - (1/3) I1 C^{-1} ) ,  W1 = dW_iso/dIbar1 ,
        Ibar1 = J^{-2/3} I1 ,  I1 = tr(C).
  -> S_iso = 0 at F = I (since I - (I1/3) C^{-1} = 0 there), which removes the
     spurious undeformed-state stress of a pressure-only split.

* Pluggable isochoric ground states (functions of Ibar1):
    Neo-Hookean : W = (mu/2)(Ibar1 - 3)                       (Holzapfel 2000)
    Yeoh        : W = c1 x + c2 x^2 + c3 x^3 , x = Ibar1 - 3
        Yeoh, O.H. (1993). Rubber Chem. Technol. 66, 754-771.
    Arruda-Boyce: W = mu * sum_{i=1..5} (c_i / lm^{2i-2}) (Ibar1^i - 3^i)
        Arruda, E.M. & Boyce, M.C. (1993). J. Mech. Phys. Solids 41, 389-412.
        c = [1/2, 1/20, 11/1050, 19/7000, 519/673750].
  Only W1 = dW/dIbar1 changes between models; everything else is shared.

* Volumetric response (purely elastic — no relaxation):
        U(J) = (kappa/2) (ln J)^2  ->  S_vol = kappa ln(J) C^{-1}   (Simo & Hughes 1998).

* Finite-strain viscoelastic relaxation (overstress / internal-variable form):
    Simo, J.C. (1987). CMAME 60(2), 153-173 ; Simo & Hughes (1998) Sec. 10.2-10.3.
        h_i(t+dt) = beta_i h_i(t) + g_i gamma_i ( S_iso(t+dt) - S_iso(t) )
        beta_i = exp(-dt/tau_i) ,  gamma_i = (1 - beta_i)/(dt/tau_i)
        S_eff = S_vol + g_inf S_iso + sum_i h_i ,  g_inf + sum g_i = 1.

* Volumetric-locking control (near-incompressible, nu->0.5), F-bar method:
    de Souza Neto, Peric, Dutko & Owen (1996). Int. J. Solids Struct. 33, 3277.
        Fbar_lock = F sqrt(J0/J) (plane strain) ,  J0 = det F at centroid.

* Total-Lagrangian B_L / geometric tangent:  Belytschko, Liu & Moran (2000), Ch.6.

* Consistent tangent: forward-mode AD of the internal force at frozen history
  (jax.jacobian). No spectral decomposition anywhere -> finite at F = I and
  isotropic stretch, and exact (recovers the gamma_i algorithmic weight, not the
  approximate beta_i of the closed-form tangent in viscoelastic.py).

State layout per GP (Voigt-6 symmetric order [11,22,33,12,13,23]):
    [ h_i (6) for i = 0..M-1 ,  S_iso_prev (6) ]  ->  length 6*(M+1).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

_GP2 = jnp.array([
    [-1.0 / jnp.sqrt(3), -1.0 / jnp.sqrt(3)],
    [ 1.0 / jnp.sqrt(3), -1.0 / jnp.sqrt(3)],
    [ 1.0 / jnp.sqrt(3),  1.0 / jnp.sqrt(3)],
    [-1.0 / jnp.sqrt(3),  1.0 / jnp.sqrt(3)],
])
_W2 = jnp.ones(4)
_SYM = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))

# Arruda-Boyce series coefficients c_1..c_5
_AB_C = jnp.array([0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0, 519.0 / 673750.0])


# ------------------------------------------------------------------
# Pluggable isochoric ground state:  W1 = dW_iso / dIbar1
# ------------------------------------------------------------------
def _W1(base: str, I1b, bparams):
    """First invariant derivative of the isochoric strain-energy density.

    `base` is a *static* string (resolved at trace time). `bparams` packs the
    model constants: neohookean=[mu]; yeoh=[c1,c2,c3]; arruda=[mu, lambda_m].
    """
    if base == "neohookean":
        return 0.5 * bparams[0]
    if base == "yeoh":
        x = I1b - 3.0
        return bparams[0] + 2.0 * bparams[1] * x + 3.0 * bparams[2] * x * x
    if base == "arruda":
        mu, lm = bparams[0], bparams[1]
        s = 0.0
        for i in range(5):  # term i (1-based i+1): (i+1) c_{i+1} / lm^{2i} Ibar1^i
            s = s + (i + 1) * _AB_C[i] / lm ** (2 * i) * I1b ** i
        return mu * s
    raise ValueError(f"unknown base hyperelastic model {base!r}")


# ------------------------------------------------------------------
# Kinematics helpers
# ------------------------------------------------------------------
def _sd(xi, eta):
    dN_dxi  = 0.25 * jnp.array([-(1 - eta),  (1 - eta),  (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * jnp.array([-(1 - xi),  -(1 + xi),   (1 + xi),   (1 - xi)])
    return dN_dxi, dN_deta


def _grads(xi, eta, coords):
    dN_dxi, dN_deta = _sd(xi, eta)
    J = jnp.array([
        [jnp.dot(dN_dxi, coords[:, 0]), jnp.dot(dN_dxi, coords[:, 1])],
        [jnp.dot(dN_deta, coords[:, 0]), jnp.dot(dN_deta, coords[:, 1])],
    ])
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    invJ = jnp.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
    gX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    gY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    return gX, gY, detJ


def _F_at(gX, gY, u_elem):
    ux = u_elem[0::2]
    uy = u_elem[1::2]
    H = jnp.array([[ux @ gX, ux @ gY], [uy @ gX, uy @ gY]])
    return jnp.eye(2) + H


def _BL_columns(Ft, gX, gY):
    """Total-Lagrangian B_L (3x8): delta E = sym(Ft^T delta F)."""
    F11, F12 = Ft[0, 0], Ft[0, 1]
    F21, F22 = Ft[1, 0], Ft[1, 1]
    B = jnp.zeros((3, 8))
    for a in range(4):
        gx, gy = gX[a], gY[a]
        B = B.at[0, 2 * a].set(F11 * gx)
        B = B.at[1, 2 * a].set(F12 * gy)
        B = B.at[2, 2 * a].set(F11 * gy + F12 * gx)
        B = B.at[0, 2 * a + 1].set(F21 * gx)
        B = B.at[1, 2 * a + 1].set(F22 * gy)
        B = B.at[2, 2 * a + 1].set(F21 * gy + F22 * gx)
    return B


def _voigt6_to_tensor(v6):
    T = jnp.zeros((3, 3))
    for k, (r, c) in enumerate(_SYM):
        T = T.at[r, c].set(v6[k]).at[c, r].set(v6[k])
    return T


def _tensor_to_voigt6(T):
    return jnp.array([T[r, c] for (r, c) in _SYM])


# ------------------------------------------------------------------
# Constitutive: Flory split + base ground state + Simo overstress
# ------------------------------------------------------------------
def _simo_pk2(base, Fbar2, h_prev_flat, kappa, bparams, g_i, tau_i, g_inf, dt, distortion_j_crit: float = 0.0):
    M = g_i.shape[0]
    F3 = jnp.eye(3).at[:2, :2].set(Fbar2)
    C = F3.T @ F3
    J = jnp.linalg.det(F3)                       # = det Fbar2 (F33 = 1)
    Cinv = jnp.linalg.inv(C + 1e-15 * jnp.eye(3))
    I1 = jnp.trace(C)
    
    # Classical Simo & Armero (1992) / Holzapfel (2000) Volumetric PK2 Stress:
    # S_vol = 0.5 * kappa * (J - 1/J) * C^{-1}
    # Provides smooth, infinite physical hydrostatic barrier as J -> 0, preventing element inversion
    # naturally from pure hyperelastic strain energy potential without artificial distortion flags.
    J_safe = jnp.maximum(J, 1e-4)
    vol_factor = 0.5 * kappa * (J_safe - 1.0 / J_safe)
    S_vol = vol_factor * Cinv

    # Optional Distortion Control (Abaqus-grade *SECTION CONTROLS, DISTORTION CONTROL=YES)
    distortion = jnp.where(
        distortion_j_crit > 0.0,
        jnp.maximum(0.0, (distortion_j_crit - J) / jnp.maximum(distortion_j_crit, 1e-12)),
        0.0
    )
    S_distort = (5000.0 * kappa) * (distortion ** 3) * Cinv
    S_vol = S_vol + S_distort

    # Isochoric PK2 (Flory split, Holzapfel Eq. 6.88-6.91):
    #   S_iso = 2 W1 J^{-2/3} ( I - (1/3) I1 C^{-1} )  -> 0 at F = I.
    I1b = (J_safe ** (-2.0 / 3.0)) * I1
    I1b_safe = jnp.clip(I1b, 3.0, 50.0)
    W1 = _W1(base, I1b_safe, bparams)
    S_iso = 2.0 * W1 * (J_safe ** (-2.0 / 3.0)) * (jnp.eye(3) - (I1 / 3.0) * Cinv)

    # Overstress recurrence on the isochoric stress (Simo 1987)
    h_prev = jnp.stack([_voigt6_to_tensor(h_prev_flat[6 * i:6 * i + 6])
                        for i in range(M + 1)])
    S_iso_prev = h_prev[M]
    dS = S_iso - S_iso_prev

    S_eff = S_vol + g_inf * S_iso
    h_new = []
    for i in range(M):
        ratio = dt / jnp.maximum(tau_i[i], 1e-30)
        beta_i = jnp.where(ratio < 1e-12, 1.0 - ratio, jnp.exp(-ratio))
        gamma_i = jnp.where(ratio < 1e-12, 1.0, (1.0 - beta_i) / ratio)
        h_i = beta_i * h_prev[i] + g_i[i] * gamma_i * dS
        S_eff = S_eff + h_i
        h_new.append(h_i)

    flat = [_tensor_to_voigt6(h_new[i]) for i in range(M)] + [_tensor_to_voigt6(S_iso)]
    h_new_flat = jnp.concatenate(flat)
    S_voigt = jnp.array([S_eff[0, 0], S_eff[1, 1], S_eff[0, 1]])
    return S_voigt, h_new_flat


def _internal_force(base, u_elem, coords, state_elem, kappa, bparams,
                    g_i, tau_i, g_inf, dt, thickness, distortion_j_crit: float = 0.0,
                    F_n=None):
    """Internal force.

    Updated-Lagrangian support (mirrors `q4_eas_jax`): when `F_n` (4,2,2) is
    supplied, `coords` is the LAST CONVERGED configuration and `u_elem` is the
    INCREMENTAL displacement, so the gradient computed here is F_inc and the
    total deformation gradient is `F_inc @ F_n`. Before this the viscoelastic
    kernels were TL-only while the corotational/J2 elements in the same model
    ran UL -- the adhesive rows therefore never got UL's per-increment
    protection and were evaluated from the flat reference all the way to full
    fold. Passing F_n = None (or identity) keeps the original TL behaviour.
    """
    _eye = jnp.eye(2)
    F_n = jnp.stack([_eye, _eye, _eye, _eye]) if F_n is None else F_n

    gX0, gY0, _ = _grads(0.0, 0.0, coords)
    F0i = _F_at(gX0, gY0, u_elem)
    F0 = F0i @ (0.25 * (F_n[0] + F_n[1] + F_n[2] + F_n[3]))
    J0 = F0[0, 0] * F0[1, 1] - F0[0, 1] * F0[1, 0]

    f_int = jnp.zeros(8)
    state_new = jnp.empty_like(state_elem)
    F_n_new = jnp.zeros((4, 2, 2))
    for gp in range(4):
        xi, eta = _GP2[gp]
        gX, gY, detJ = _grads(xi, eta, coords)
        w = detJ * _W2[gp] * thickness
        F_inc = _F_at(gX, gY, u_elem)
        F = F_inc @ F_n[gp]                     # total deformation gradient
        J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
        J_ratio = jnp.maximum(J0, 0.05) / jnp.maximum(J, 0.05)
        Fbar = F * jnp.sqrt(jnp.clip(J_ratio, 0.1, 10.0))   # F-bar locking control
        S_v, h_new = _simo_pk2(base, Fbar, state_elem[gp], kappa, bparams,
                               g_i, tau_i, g_inf, dt, distortion_j_crit=distortion_j_crit)
        # Work-conjugacy fix (found by opus architecture review, 2026-09-08,
        # dev_log/plan_abaqus_element_consolidation_20260908.md finding F4):
        # `S_v` is PK2 referred to the ORIGINAL (t=0) reference config --
        # `_simo_pk2` computes it from the TOTAL deformation gradient `F`.
        # `BL` below is built from `F_inc` and is the variation of the
        # INCREMENTAL Green-Lagrange strain relative to the LAST CONVERGED
        # (step-n) config, integrated against `w = detJ` from the SAME
        # step-n `coords`. These were being contracted directly
        # (`BL.T @ S_v`), which is only valid when F_n ~= I -- S_v needs
        # pushing forward to configuration n first. Derivation: F = F_inc
        # @ F_n (F_n fixed under variation) gives delta_E = F_n.T @
        # delta_E_inc @ F_n, and dV_0 = dV_n / det(F_n), so
        #   S : delta_E dV_0 = (F_n @ S @ F_n.T / det(F_n)) : delta_E_inc dV_n
        # i.e. the stress conjugate to delta_E_inc on the step-n volume is
        # S_n = F_n @ S @ F_n.T / det(F_n), NOT S_v itself. Measured error
        # from omitting this: up to 24.9% relative internal force on a
        # homogeneous deformation (F-bar/quadrature effects are identically
        # zero there, isolating this term) -- Newton still converges (K is
        # autodiffed from this same function) but to a slightly wrong
        # equilibrium, invisible to shape-level checks (AGENTS.md 4.9).
        Fn_gp = F_n[gp]
        S0_tensor = jnp.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
        detFn = jnp.maximum(jnp.abs(Fn_gp[0, 0] * Fn_gp[1, 1] - Fn_gp[0, 1] * Fn_gp[1, 0]), 1e-30)
        Sn_tensor = (Fn_gp @ S0_tensor @ Fn_gp.T) / detFn
        S_n_voigt = jnp.array([Sn_tensor[0, 0], Sn_tensor[1, 1], Sn_tensor[0, 1]])
        BL = _BL_columns(F_inc, gX, gY)
        f_int = f_int + BL.T @ S_n_voigt * w
        state_new = state_new.at[gp].set(h_new)
        F_n_new = F_n_new.at[gp].set(F)
    return f_int, state_new, F_n_new


@partial(jax.jit, static_argnames=("base",))
def compute_single(coords, u_elem, state_elem, kappa, bparams,
                   g_i, tau_i, g_inf, dt, thickness, distortion_j_crit: float = 0.0,
                   F_n_gps=None, base="neohookean"):
    """Finite-strain pluggable-base Simo viscoelastic F-bar hybrid element.

    Returns (f_int(8,), K_e(8,8), state_new). Tangent = consistent algorithmic
    modulus via autodiff at frozen history (NaN-safe; no eigendecomposition).
    """
    f_int, state_new, F_n_new = _internal_force(
        base, u_elem, coords, state_elem, kappa, bparams,
        g_i, tau_i, g_inf, dt, thickness, distortion_j_crit=distortion_j_crit,
        F_n=F_n_gps)
    K_e = jax.jacobian(
        lambda u: _internal_force(
            base, u, coords, state_elem, kappa, bparams,
            g_i, tau_i, g_inf, dt, thickness, distortion_j_crit=distortion_j_crit,
            F_n=F_n_gps)[0]
    )(u_elem)
    return f_int, K_e, state_new, F_n_new

