"""
dispsolver/element/kinematics/frame.py
=======================================
**The configuration ledger.** A per-Gauss-point record in which every field
names the configuration it is referred to, plus the ONE function permitted
to contract a stress with a B-operator.

Why this module exists
----------------------
Six defects found between 2026-09-08 and 2026-09-10 -- F4, F6, B1, B2, B3,
and `q4_eas.py`'s flagged residue -- are all the same sentence:

    "this quantity is referred to configuration X and I contracted it with
     something referred to configuration Y."

They were fixed one at a time, in nine files, and the class kept recurring
because nothing in the code said which configuration anything was in. The
remedy is not another correction; it is to make the sentence unwritable.

Three configurations, named
---------------------------
``REF0``     the original, undeformed geometry. **The material's frame.**
             `response_fn` is defined on the TOTAL deformation gradient and
             returns PK2 referred here, by definition of the Phase A
             interface (`material/response_interface.py`).
``REF_N``    the last converged configuration. **The element's frame** under
             NLGEOM: `coords` are config-n coordinates, the integration
             weight `detJ` is a config-n measure, and `B_L` is built from
             the INCREMENTAL deformation gradient, so it is conjugate to a
             stress on config n.
``CURRENT``  the trial iterate.

`REF0 == REF_N` exactly when `F_n = I`, which is Total Lagrangian -- so
every push-forward below is an exact no-op there and a TL path is
bit-identical with or without this module.

The rule
--------
    No element kernel may reference a coordinate array, a Jacobian, or a
    stress without going through a `GPKinematics` field whose name states
    its configuration; and no element kernel may form `B.T @ S` itself.

What that buys, defect by defect:

* **F4 / B1 cannot recur.** The `F_n S F_n^T / det F_n` push-forward is not
  something an element author has to remember -- `gp_internal_force` is the
  only route from a stress to a nodal force, and it applies it
  unconditionally.
* **B2 / B3 cannot recur.** A *partial* push-forward is unrepresentable. A
  mixed element gets `f_u` and `f_alpha` from the same
  `gp_internal_force` call on the same `GPKinematics`, so they are
  gradients of one potential by construction and `K_au = K_ua^T` follows
  rather than being asserted in a comment that later goes stale. (B2 was
  exactly this: F4 applied to `f_u` but deliberately not to `f_a`, which
  falsified the symmetry the static condensation assumes -- measured 4.0e-2
  TL / 7.8e-2 UL on the EAS path, 54% on CPE4IH.)
* **F6 becomes a single-site risk.** The enhanced-mode pull-back lives in
  one `cpe4i.py`, not five copies across JAX / NumPy / Numba.

What this module deliberately is NOT
------------------------------------
There is **no objective rate and no polar decomposition here.** Abaqus's
per-integration-point `DeltaF = DeltaR DeltaU` machinery exists to make
*rate-form* constitutive laws objective, and Abaqus says so directly:
"Objective rates are relevant only for rate form constitutive equations ...
For hyperelastic materials a total formulation is used; hence, the concept
of an objective rate is not relevant for the constitutive law" (Abaqus 2016
Theory Guide 1.5.3). **Every material in this repository is total-form** --
`plastic_jax.pk2_voigt_jax` is finite-strain multiplicative J2 taking the
total F, `_simo_pk2` likewise -- so a frame handler here would be a device
neither our materials nor Abaqus's treatment of them requires. When a
genuinely rate-form material arrives, that machinery belongs on the
material side of the interface: `material/rate_form_adapter.py`.

Tracing note
------------
`GPKinematics` is a plain frozen dataclass holding traced arrays. It is
built and consumed **inside** one traced function, never passed across a
`jax.jit` boundary, so it needs no pytree registration and costs nothing at
runtime -- under `jit` / `njit` it is fully unrolled away.

References
----------
* Work conjugacy of the pull-back: `S : dE_0 = (F_n S F_n^T) : dE_n` with
  `dE_0 = F_n^T dE_n F_n`, and `dV_0 = dV_n / det F_n`; see the derivation
  in `push_forward_stress`.
* Abaqus 2016 Theory Guide 1.4.3, 1.5.3, 1.5.4 (what Abaqus does instead,
  and why it does not need this for our material class).
* dev_log/eas_frame_consistency_benchmarks_20260910.md (B1-B4).
* AGENTS.md 4.14 (F4/F5/F6), 4.15 (B1-B4 and the standing FD test).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import jax.numpy as jnp


class Config(Enum):
    """Which configuration a kinematic quantity is referred to."""

    REF0 = 0     # original, undeformed -- the MATERIAL's frame
    REF_N = 1    # last converged -- the ELEMENT's frame under NLGEOM
    CURRENT = 2  # trial iterate

    def __str__(self) -> str:                       # pragma: no cover - debug aid
        return self.name


@dataclass(frozen=True)
class GPKinematics:
    """One Gauss point's kinematics, with every field's configuration named.

    Built once per Gauss point by the element; consumed only by
    `gp_internal_force`. An element that needs a quantity not in this record
    is telling you the record is missing a field, not that it should reach
    around it.

    Attributes
    ----------
    F_total : (2,2)
        Total deformation gradient, **referred to REF0**. This is what goes
        to `response_fn`: the material's frame is the original
        configuration, always, regardless of what the element is doing.
    F_inc : (2,2)
        CURRENT relative to REF_N. Equals `F_total` in Total Lagrangian.
    F_n : (2,2)
        REF_N relative to REF0. Identity in Total Lagrangian.
    B_L : (3,8)
        Green-Lagrange strain-displacement operator built from `F_inc`, so
        it is conjugate to a stress **on REF_N** -- never to the material's
        REF0 stress. Contracting the two directly is finding F4.
    weight : scalar
        `detJ * w_gp * thickness` on **REF_N** (the element integrates over
        config n because its `coords` are config-n coordinates).
    """

    F_total: jnp.ndarray
    F_inc: jnp.ndarray
    F_n: jnp.ndarray
    B_L: jnp.ndarray
    weight: jnp.ndarray

    # Stated, not assumed. Changing an element to integrate on a different
    # configuration means changing these, which means the push-forward in
    # `gp_internal_force` has to be revisited -- which is the point.
    F_total_config: Config = Config.REF0
    B_L_config: Config = Config.REF_N
    weight_config: Config = Config.REF_N


def _det2(A):
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]


def voigt_pullback_operator(F_n):
    """Voigt matrix `T` of the strain pull-back `E_n -> E_0 = F_n^T E_n F_n`.

    Voigt convention here is the one used throughout this codebase:
    strain `[E11, E22, gamma12]` with ENGINEERING shear `gamma = 2*E12`,
    stress `[S11, S22, S12]`. That pairing makes `S_v . E_v = S : E`, which
    is what lets the stress push-forward be exactly `T^T` (see
    `push_forward_stress`).

    `T = I` when `F_n = I`.
    """
    a11, a12 = F_n[0, 0], F_n[0, 1]
    a21, a22 = F_n[1, 0], F_n[1, 1]
    return jnp.array([
        [a11 * a11,       a21 * a21,       a11 * a21],
        [a12 * a12,       a22 * a22,       a12 * a22],
        [2.0 * a11 * a12, 2.0 * a21 * a22, a11 * a22 + a21 * a12],
    ])


def push_forward_stress(S_voigt_ref0, F_n):
    """PK2 on REF0 -> the work-conjugate stress on REF_N.

    `S_n = F_n S F_n^T / det(F_n)`.

    Derivation (this is the whole of finding F4, written out once):
    with `F_total = F_inc @ F_n`, `C_0 = F_n^T C_n F_n`, hence
    `dE_0 = F_n^T dE_n F_n`, so

        S : dE_0 = S : (F_n^T dE_n F_n) = (F_n S F_n^T) : dE_n

    and the element integrates on config n, where `dV_0 = dV_n / det F_n`.
    Dropping either factor is a first-order error in the stretch of `F_n`
    -- measured up to 24.9% on the internal force of the sibling kernels
    before the fix, and **exactly zero at `F_n = I`**, which is why every
    Total-Lagrangian test in the repo passed throughout.

    Equivalent to `T^T @ S_voigt / det(F_n)` with `T` from
    `voigt_pullback_operator`; written in tensor form here because it is
    cheaper and because the two forms cross-check each other (see
    `tests/element_contract/test_frame_ledger.py`).
    """
    S = jnp.array([[S_voigt_ref0[0], S_voigt_ref0[2]],
                   [S_voigt_ref0[2], S_voigt_ref0[1]]])
    detFn = jnp.maximum(jnp.abs(_det2(F_n)), 1e-30)
    Sn = (F_n @ S @ F_n.T) / detFn
    return jnp.array([Sn[0, 0], Sn[1, 1], Sn[0, 1]])


def push_forward_tangent(C_voigt_ref0, F_n):
    """Material tangent `dS/dE` on REF0 -> the tangent on REF_N.

    `C_n = T^T C T / det(F_n)`, the exact differential companion of
    `push_forward_stress`: pushing the stress without also pushing the
    tangent leaves `K_e` not equal to the Jacobian of `f_int` -- which is
    defect B1, measured at 97.6% relative error against a finite-difference
    Jacobian on a 39-degree-rotated reference, with the true Jacobian 151%
    ASYMMETRIC (complex eigenvalues: `f_alpha` was not the gradient of any
    potential). Identity at `F_n = I`.
    """
    T = voigt_pullback_operator(F_n)
    detFn = jnp.maximum(jnp.abs(_det2(F_n)), 1e-30)
    return (T.T @ C_voigt_ref0 @ T) / detFn


def gp_internal_force(kin: GPKinematics, S_voigt_ref0,
                      C_voigt_ref0: Optional[jnp.ndarray] = None):
    """**The only function permitted to contract a stress with a B-operator.**

    Parameters
    ----------
    kin : GPKinematics
        Whose `B_L` and `weight` live on REF_N (asserted by the record's own
        `*_config` fields, which exist so that a future element integrating
        on a different configuration cannot quietly reuse this routine).
    S_voigt_ref0 : (3,)
        The material's PK2, referred to REF0 -- by definition of the
        response interface, not by convention.
    C_voigt_ref0 : (3,3), optional
        The material's `dS/dE` on REF0. Omit it if the caller is building
        the tangent by autodiff of the force instead.

    Returns
    -------
    (f_gp(8,), K_gp(8,8) or None)
        The Gauss point's contribution. The push-forward is applied HERE,
        unconditionally and exactly once -- an element cannot apply it to
        the force and forget it on the tangent (B1), nor apply it to one
        block of a mixed element and not the other (B2/B3), because it
        never touches it at all.

    Note the returned `K_gp` is the MATERIAL stiffness only. The geometric
    (initial-stress) term is the element's own business: it depends on the
    element's shape-function gradients, which are element-specific, whereas
    everything here is not. Elements that want a consistent tangent should
    autodiff `f_int` instead and pass `C_voigt_ref0=None` -- with the
    compile-time caveat of AGENTS.md 4.4 for plasticity-coupled kernels.
    """
    if kin.B_L_config is not Config.REF_N or kin.weight_config is not Config.REF_N:
        raise ValueError(
            f"gp_internal_force assumes B_L and weight on {Config.REF_N}; got "
            f"B_L on {kin.B_L_config} and weight on {kin.weight_config}. "
            f"Pushing a REF0 stress onto a different configuration needs a "
            f"different transform -- do not reuse this one."
        )
    S_n = push_forward_stress(S_voigt_ref0, kin.F_n)
    f_gp = kin.B_L.T @ S_n * kin.weight
    if C_voigt_ref0 is None:
        return f_gp, None
    C_n = push_forward_tangent(C_voigt_ref0, kin.F_n)
    return f_gp, (kin.B_L.T @ C_n @ kin.B_L) * kin.weight
