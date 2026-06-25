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
def _simo_pk2(base, Fbar2, h_prev_flat, kappa, bparams, g_i, tau_i, g_inf, dt):
    M = g_i.shape[0]
    F3 = jnp.eye(3).at[:2, :2].set(Fbar2)
    C = F3.T @ F3
    J = jnp.linalg.det(F3)                       # = det Fbar2 (F33 = 1)
    Cinv = jnp.linalg.inv(C)
    I1 = jnp.trace(C)
    lnJ = jnp.log(jnp.maximum(J, 1e-30))

    # Volumetric (elastic) PK2:  S_vol = kappa ln(J) C^{-1}
    S_vol = kappa * lnJ * Cinv

    # Isochoric PK2 (Flory split, Holzapfel Eq. 6.88-6.91):
    #   S_iso = 2 W1 J^{-2/3} ( I - (1/3) I1 C^{-1} )  -> 0 at F = I.
    I1b = J ** (-2.0 / 3.0) * I1
    W1 = _W1(base, I1b, bparams)
    S_iso = 2.0 * W1 * J ** (-2.0 / 3.0) * (jnp.eye(3) - (I1 / 3.0) * Cinv)

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
                    g_i, tau_i, g_inf, dt, thickness):
    gX0, gY0, _ = _grads(0.0, 0.0, coords)
    F0 = _F_at(gX0, gY0, u_elem)
    J0 = F0[0, 0] * F0[1, 1] - F0[0, 1] * F0[1, 0]

    f_int = jnp.zeros(8)
    state_new = jnp.empty_like(state_elem)
    for gp in range(4):
        xi, eta = _GP2[gp]
        gX, gY, detJ = _grads(xi, eta, coords)
        w = detJ * _W2[gp] * thickness
        F = _F_at(gX, gY, u_elem)
        J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
        Fbar = F * jnp.sqrt(jnp.maximum(J0 / J, 1e-12))   # F-bar locking control
        S_v, h_new = _simo_pk2(base, Fbar, state_elem[gp], kappa, bparams,
                               g_i, tau_i, g_inf, dt)
        BL = _BL_columns(Fbar, gX, gY)
        f_int = f_int + BL.T @ S_v * w
        state_new = state_new.at[gp].set(h_new)
    return f_int, state_new


@partial(jax.jit, static_argnames=("base",))
def compute_single(coords, u_elem, state_elem, kappa, bparams,
                   g_i, tau_i, g_inf, dt, thickness, base="neohookean"):
    """Finite-strain pluggable-base Simo viscoelastic F-bar hybrid element.

    Returns (f_int(8,), K_e(8,8), state_new). Tangent = consistent algorithmic
    modulus via autodiff at frozen history (NaN-safe; no eigendecomposition).
    """
    f_int, state_new = _internal_force(
        base, u_elem, coords, state_elem, kappa, bparams,
        g_i, tau_i, g_inf, dt, thickness)
    K_e = jax.jacobian(
        lambda u: _internal_force(
            base, u, coords, state_elem, kappa, bparams,
            g_i, tau_i, g_inf, dt, thickness)[0]
    )(u_elem)
    return f_int, K_e, state_new
