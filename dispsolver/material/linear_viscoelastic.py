"""
linear_viscoelastic.py
======================
Small-strain linear viscoelasticity (Abaqus *VISCOELASTIC, TIME=PRONY) with
time-temperature superposition (*TRS, WLF) — designed for the Q1P0 hybrid
(mean-dilatation) element to avoid volumetric locking.

Formulation (plane strain, deviatoric/volumetric split)
-------------------------------------------------------
Bulk response is elastic:           p = K · θ            (θ = tr(ε))
Deviatoric response relaxes via a Prony series of internal stresses q_i:

    s(t+Δt) = 2·G_∞·e(t+Δt) + Σ_i q_i(t+Δt)

    q_i(t+Δt) = a_i·q_i(t) + 2·G_i·γ_i·( e(t+Δt) − e(t) )

    a_i = exp(−Δt / τ_i*) ,   γ_i = (1 − a_i)·τ_i* / Δt
    τ_i* = τ_i · a_T(T)                       (WLF shift, TTS)

with  G_∞ = G_0·(1 − Σ g_i),  G_i = G_0·g_i,  G_0 = E / (2(1+ν)),
      K   = E / (3(1−2ν)).

e is the (tensorial) deviatoric strain; in plane strain ε_zz = 0 so
    e = ε − (θ/3)·I  ⇒  e_zz = −θ/3 ≠ 0.

Algorithmic shear modulus (consistent tangent):
    G_alg = G_∞ + Σ_i G_i·γ_i ,   ds = 2·G_alg·de

Internal-variable layout per Gauss point (flat, length 4·(M+1)):
    [ e_prev(4) , q_1(4) , … , q_M(4) ]      components ordered (xx, yy, zz, xy)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .viscoelastic import wlf_shift


class LinearViscoelastic:
    """Small-strain linear viscoelastic material (Prony + WLF) for the hybrid element."""

    def __init__(
        self,
        E: float,
        nu: float,
        g_i: List[float],
        tau_i: List[float],
        wlf_params: Optional[Dict[str, float]] = None,
    ):
        if len(g_i) != len(tau_i):
            raise ValueError("g_i and tau_i must have same length")
        if sum(g_i) >= 1.0:
            raise ValueError("Sum of g_i must be < 1 (G_inf > 0 required)")

        self.E = float(E)
        self.nu = float(nu)
        self.G0 = E / (2.0 * (1.0 + nu))
        self.K = E / (3.0 * (1.0 - 2.0 * nu))

        self.M = len(g_i)
        self.g_i = np.asarray(g_i, dtype=np.float64)
        self.tau_i = np.asarray(tau_i, dtype=np.float64)
        self.g_inf = 1.0 - float(np.sum(g_i))
        self.G_inf = self.G0 * self.g_inf
        self.G_terms = self.G0 * self.g_i          # G_i per Prony term
        self.wlf_params = wlf_params

    # ------------------------------------------------------------------
    # State interface
    # ------------------------------------------------------------------

    @property
    def n_internal_vars(self) -> int:
        """e_prev(4) + q_i(4 each)."""
        return 4 * (self.M + 1)

    def initial_internal_vars(self) -> np.ndarray:
        return np.zeros(4 * (self.M + 1), dtype=np.float64)

    @staticmethod
    def internal_var_names(M: int) -> List[str]:
        comps = ["xx", "yy", "zz", "xy"]
        names = [f"e_prev_{c}" for c in comps]
        names += [f"q{i}_{c}" for i in range(M) for c in comps]
        return names

    # ------------------------------------------------------------------
    # Prony coefficients (shared by all Gauss points for a given Δt, T)
    # ------------------------------------------------------------------

    def prony_coeffs(self, dt: float, temperature: float = 20.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (a_i, gamma_i, G_alg) for the current step size and temperature."""
        aT = 1.0
        if self.wlf_params is not None:
            aT = wlf_shift(temperature,
                           self.wlf_params["C1"],
                           self.wlf_params["C2"],
                           self.wlf_params["T_ref"])
        a = np.empty(self.M, dtype=np.float64)
        gamma = np.empty(self.M, dtype=np.float64)
        for i in range(self.M):
            tau_eff = float(self.tau_i[i]) * aT
            ratio = dt / max(tau_eff, 1e-30)
            if ratio < 1e-12:
                a[i] = 1.0 - ratio
                gamma[i] = 1.0
            else:
                a[i] = float(np.exp(-ratio))
                gamma[i] = (1.0 - a[i]) / ratio
        G_alg = self.G_inf + float(np.sum(self.G_terms * gamma))
        return a, gamma, G_alg

    # ------------------------------------------------------------------
    # Deviatoric stress update at one Gauss point
    # ------------------------------------------------------------------

    def dev_update(
        self,
        e_dev_new: np.ndarray,      # (4,) tensorial dev strain [xx,yy,zz,xy]
        state_gp: np.ndarray,       # (4*(M+1),)
        a: np.ndarray,
        gamma: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (s_dev(4,), state_new(4*(M+1),)) using precomputed (a, gamma)."""
        M = self.M
        e_prev = state_gp[:4]
        de = e_dev_new - e_prev

        s_dev = 2.0 * self.G_inf * e_dev_new
        state_new = np.empty_like(state_gp)
        state_new[:4] = e_dev_new
        for i in range(M):
            q_prev = state_gp[4 * (i + 1):4 * (i + 2)]
            q_new = a[i] * q_prev + 2.0 * self.G_terms[i] * gamma[i] * de
            s_dev = s_dev + q_new
            state_new[4 * (i + 1):4 * (i + 2)] = q_new
        return s_dev, state_new

    def __repr__(self) -> str:
        trs = "WLF" if self.wlf_params else "isothermal"
        return (f"LinearViscoelastic(E={self.E}, nu={self.nu}, M={self.M}, "
                f"g_oo={self.g_inf:.4f}, {trs})")
