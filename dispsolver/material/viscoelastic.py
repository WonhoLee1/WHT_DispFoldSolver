"""
viscoelastic.py
===============
Viscoelastic material model via Prony series + WLF time-temperature superposition.

Wraps any hyperelastic MaterialModel with deviatoric stress relaxation.

Formulation (Simo 1987, Simo & Hughes 1998 Sec. 10.3)
------------------------------------------------------
Deviatoric/volumetric split via the isochoric deformation gradient:

    S_vol = J·p·C^{-1}      (p = dW_vol/dJ, fully elastic — no relaxation)
    S_dev = S_total - S_vol

Internal variables h_i (M Prony terms, overstress) evolve as:
    h_i(t+Δt) = β_i · h_i(t) + g_i · γ_i · ΔS_dev_el

    β_i = exp(-Δt / (τ_i · aT(T)))
    γ_i = (1 - β_i) · (τ_i · aT(T)) / Δt   (linearized midpoint rule)

    where g_∞ + Σg_i = 1
          ΔS_dev_el = S_dev_el(t+Δt) - S_dev_el(t)

The total effective deviatoric stress:
    S_eff_dev = g_∞ · S_dev_el(t+Δt) + Σ_i h_i(t+Δt)

Internal variable array layout (pytree-compatible):
    h_state : (M+1, 3, 3) where
        [0:M, :, :]  = h_i   Prony overstress
        [M,   :, :]  = S_dev_el_prev   previous deviatoric stress (for Δ)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import MaterialModel, Params


# ------------------------------------------------------------------
# WLF shift factor
# ------------------------------------------------------------------

def wlf_shift(temperature: float, C1: float, C2: float, T_ref: float) -> float:
    """Compute WLF shift factor aT."""
    delta_T = temperature - T_ref
    if delta_T <= -C2 + 1e-6:
        return 1e15
    log_aT = -C1 * delta_T / (C2 + delta_T)
    return 10.0 ** log_aT


# ------------------------------------------------------------------
# Volumetric stress helpers
# ------------------------------------------------------------------

def _extract_lam_mu(params: Params) -> Tuple[float, float]:
    mu = params.get('mu', None)
    lam = params.get('lambda', None)
    if mu is not None and lam is not None:
        return float(mu), float(lam)
    E = float(params['E'])
    nu = float(params['nu'])
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return mu, lam


def _embed_F(F_2d: np.ndarray) -> np.ndarray:
    F3 = np.eye(3, dtype=np.float64)
    F3[:2, :2] = F_2d
    return F3


def _C_and_J_from_F(F: np.ndarray) -> Tuple[np.ndarray, float]:
    F3 = _embed_F(F)
    C = F3.T @ F3
    J = float(np.linalg.det(F3))
    return C, J


def _volumetric_stress(F: np.ndarray, params: Params) -> np.ndarray:
    """Return S_vol as 3x3 tensor: J·p·C^{-1}."""
    C, J = _C_and_J_from_F(F)
    C_inv = np.linalg.inv(C)
    mu, lam = _extract_lam_mu(params)
    lnJ = np.log(max(J, 1e-30))
    p = (-mu + lam * lnJ) / J  # dW_vol/dJ
    return float(J * p) * C_inv


def _deviatoric_stress(S_full: np.ndarray, S_vol: np.ndarray) -> np.ndarray:
    return S_full - S_vol


# ------------------------------------------------------------------
# Batch state conversion helpers
# ------------------------------------------------------------------

_SYM_IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def _flat_batch_to_tensor_3d(flat: np.ndarray, M: int) -> np.ndarray:
    """(N, 6*(M+1)) -> (N, M+1, 3, 3) symmetric tensor batch."""
    N = flat.shape[0]
    tensor = np.zeros((N, M + 1, 3, 3), dtype=np.float64)
    for i in range(M + 1):
        for k, (r, c) in enumerate(_SYM_IDX):
            tensor[:, i, r, c] = flat[:, i * 6 + k]
            if r != c:
                tensor[:, i, c, r] = flat[:, i * 6 + k]
    return tensor


def _tensor_3d_batch_to_flat(tensor: np.ndarray, M: int) -> np.ndarray:
    """(N, M+1, 3, 3) -> (N, 6*(M+1)) symmetric tensor batch."""
    N = tensor.shape[0]
    flat = np.zeros((N, 6 * (M + 1)), dtype=np.float64)
    for i in range(M + 1):
        for k, (r, c) in enumerate(_SYM_IDX):
            flat[:, i * 6 + k] = tensor[:, i, r, c]
    return flat


# ------------------------------------------------------------------
# Viscoelastic material
# ------------------------------------------------------------------

class ViscoelasticMaterial:
    """Viscoelastic wrapper around a hyperelastic material.

    Internal variable layout: h_state is (M+1, 3, 3)
        h_state[0:M]  = Prony overstress h_i
        h_state[M]    = previous deviatoric elastic stress S_dev_el(t)
    """

    def __init__(
        self,
        base_material: MaterialModel,
        g_i: List[float],
        tau_i: List[float],
        wlf_params: Optional[Dict[str, float]] = None,
    ):
        if len(g_i) != len(tau_i):
            raise ValueError("g_i and tau_i must have same length")
        if sum(g_i) >= 1.0:
            raise ValueError("Sum of g_i must be < 1 (g_oo > 0 required)")

        self.base = base_material
        self.M = len(g_i)
        self.g_i = np.array(g_i, dtype=np.float64)
        self.tau_i = np.array(tau_i, dtype=np.float64)
        self.g_inf = 1.0 - np.sum(g_i)
        self.wlf_params = wlf_params

    # ------------------------------------------------------------------
    # Internal variable dimension
    # ------------------------------------------------------------------

    @property
    def n_internal_vars(self) -> int:
        """6 per Prony term + 6 for the prev deviatoric stress tensor."""
        return 6 * (self.M + 1)

    @staticmethod
    def internal_var_names(M: int) -> List[str]:
        comps = ["11", "22", "33", "12", "13", "23"]
        names = [f"h{i}_{c}" for i in range(M) for c in comps]
        names += [f"Sdev_prev_{c}" for c in comps]
        return names

    def initial_internal_vars(self) -> np.ndarray:
        """Return (M+1, 3, 3) zero-initialized state array."""
        return np.zeros((self.M + 1, 3, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # Voigt conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_voigt(S_3d: np.ndarray) -> np.ndarray:
        return np.array([S_3d[0, 0], S_3d[1, 1], S_3d[0, 1]], dtype=np.float64)

    # ------------------------------------------------------------------
    # Stress computation
    # ------------------------------------------------------------------

    def pk2_voigt(
        self,
        F: np.ndarray,
        params: Params,
        h_prev: np.ndarray,  # (M+1, 3, 3)
        dt: float,
        temperature: float = 20.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 2nd P-K stress with viscoelastic relaxation.

        Parameters
        ----------
        F : (2, 2) ndarray
        params : dict
        h_prev : (M+1, 3, 3) — [0:M]=Prony overstress, [M]=prev S_dev_el
        dt : float
        temperature : float

        Returns
        -------
        S_eff : (3,) ndarray — Voigt stress
        h_new : (M+1, 3, 3) — updated state
        """
        # 1. Full elastic stress
        S_full = np.asarray(self.base.pk2_tensor(F, params))

        # 2. Volumetric/deviatoric split
        S_vol = _volumetric_stress(F, params)
        S_dev_el = S_full - S_vol

        # 3. Deviatoric increment (from stored prev, or zero on first step)
        S_dev_prev = h_prev[self.M]  # zero on initial state
        dS_dev = S_dev_el - S_dev_prev

        # 4. WLF shift
        aT = 1.0
        if self.wlf_params is not None:
            aT = wlf_shift(temperature,
                           self.wlf_params["C1"],
                           self.wlf_params["C2"],
                           self.wlf_params["T_ref"])

        # 5. Update each Prony overstress term
        #    h_i(t+dt) = beta_i * h_i(t) + g_i * gamma_i * dS_dev
        #    beta_i = exp(-dt/tau_eff), gamma_i = (1-beta_i)*tau_eff/dt
        #    For dt/tau << 1: beta_i ~ 1 - dt/tau, gamma_i ~ 1
        #    (np.exp loses precision for very small arguments on Windows MSVC)
        h_new = np.zeros_like(h_prev)
        for i in range(self.M):
            tau_eff = self.tau_i[i] * aT
            ratio = dt / max(tau_eff, 1e-30)
            if ratio < 1e-12:
                beta_i = 1.0 - ratio
                gamma_i = 1.0
            else:
                beta_i = np.exp(-ratio)
                gamma_i = (1.0 - beta_i) / ratio
            h_new[i] = beta_i * h_prev[i] + self.g_i[i] * gamma_i * dS_dev

        # 6. Store S_dev_el for next step
        h_new[self.M] = S_dev_el

        # 7. Effective stress
        S_eff_3d = S_vol + self.g_inf * S_dev_el
        for i in range(self.M):
            S_eff_3d += h_new[i]

        return self._tensor_to_voigt(S_eff_3d), h_new

    # ------------------------------------------------------------------
    # Tangent (approximate)
    # ------------------------------------------------------------------

    def tangent_voigt(
        self,
        F: np.ndarray,
        params: Params,
        dt: float,
        temperature: float = 20.0,
    ) -> np.ndarray:
        """Approximate tangent: C_eff = C_vol + g_eff * (C_el - C_vol).

        g_eff = g_oo + sum_i g_i * beta_i  (instantaneous elastic factor)
        """
        C_el = np.asarray(self.base.tangent_voigt(F, params))

        aT = 1.0
        if self.wlf_params is not None:
            aT = wlf_shift(temperature,
                           self.wlf_params["C1"],
                           self.wlf_params["C2"],
                           self.wlf_params["T_ref"])

        g_eff = self.g_inf
        for i in range(self.M):
            tau_eff = self.tau_i[i] * aT
            beta_i = np.exp(-dt / max(tau_eff, 1e-30))
            g_eff += self.g_i[i] * beta_i

        K = (C_el[0, 0] + C_el[0, 1] + C_el[1, 0] + C_el[1, 1]) / 4.0
        m = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        C_vol = K * np.outer(m, m)
        C_dev = C_el - C_vol
        return C_vol + g_eff * C_dev

    # ------------------------------------------------------------------
    # Batch interface (N GPs simultaneously)
    # ------------------------------------------------------------------

    def _base_batch(self, F_batch: np.ndarray, params: Params):
        """Batched base-material (S_full, C_el) using the *numpy* analytical path.

        Critical: the sequential per-element path calls the base material's numpy
        methods (pk2_tensor / tangent_voigt). The JAX-autodiff path uses a different
        shear (Voigt) convention for C[2,2], so mixing them breaks Newton
        convergence. We therefore use the numpy batched methods when available,
        and fall back to a per-element numpy loop otherwise — never JAX.
        """
        N = F_batch.shape[0]
        if hasattr(self.base, 'pk2_tensor_batch') and hasattr(self.base, 'tangent_voigt_batch'):
            S_full = np.asarray(self.base.pk2_tensor_batch(F_batch, params))
            C_el = np.asarray(self.base.tangent_voigt_batch(F_batch, params))
        else:
            S_full = np.stack([np.asarray(self.base.pk2_tensor(F_batch[n], params)) for n in range(N)])
            C_el = np.stack([np.asarray(self.base.tangent_voigt(F_batch[n], params)) for n in range(N)])
        return S_full, C_el

    def pk2_tangent_voigt_batch(
        self,
        F_batch: np.ndarray,      # (N, 2, 2)
        params: Params,
        state_flat_batch: np.ndarray,  # (N, 6*(M+1))
        dt: float,
        temperature: float = 20.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized stress + tangent for N Gauss points simultaneously.

        Returns
        -------
        S_voigt       : (N, 3)
        C_voigt       : (N, 3, 3)
        state_new_flat: (N, 6*(M+1))
        """
        N = F_batch.shape[0]
        M = self.M

        # 1. Batched base material calls (numpy analytical — matches sequential)
        S_full, C_el = self._base_batch(F_batch, params)  # (N,3,3), (N,3,3)

        # 2. Volumetric/deviatoric split.
        # C3 = F3^T F3 is block-diagonal ([[C2d, 0], [0, 1]]) because F3 has a
        # unit (3,3) entry, so C3^{-1} reduces to the analytic 2x2 inverse of
        # C2d = F^T F. Computing it analytically (instead of np.linalg.inv on a
        # batch) never raises on a near-singular row — degenerate elements are
        # flagged `bad` and given a finite regularized response at the end.
        mu, lam = _extract_lam_mu(params)
        C2d = np.einsum('nki,nkj->nij', F_batch, F_batch)   # (N,2,2) = F^T F
        a = C2d[:, 0, 0]; b = C2d[:, 0, 1]; d = C2d[:, 1, 1]
        det2 = a * d - b * b                                  # = det(F)^2 >= 0
        J = F_batch[:, 0, 0] * F_batch[:, 1, 1] - F_batch[:, 0, 1] * F_batch[:, 1, 0]  # signed det(F)

        bad = (~np.isfinite(J)) | (~np.isfinite(det2)) | (det2 < 1e-12)
        det2_safe = np.where(bad, 1.0, det2)
        inv2 = 1.0 / det2_safe
        C3_inv = np.zeros((N, 3, 3), dtype=np.float64)
        C3_inv[:, 0, 0] = d * inv2
        C3_inv[:, 1, 1] = a * inv2
        C3_inv[:, 0, 1] = C3_inv[:, 1, 0] = -b * inv2
        C3_inv[:, 2, 2] = 1.0

        J_safe = np.where(bad, 1.0, J)
        lnJ = np.log(np.maximum(J_safe, 1e-30))   # matches sequential _volumetric_stress
        p   = (-mu + lam * lnJ) / J_safe
        S_vol = (J_safe * p)[:, None, None] * C3_inv     # (N, 3, 3)
        S_dev_el = S_full - S_vol                          # (N, 3, 3)

        # 3. Unpack state: (N, 6*(M+1)) -> (N, M+1, 3, 3)
        h_prev = _flat_batch_to_tensor_3d(state_flat_batch, M)  # (N, M+1, 3, 3)
        dS_dev = S_dev_el - h_prev[:, M]                  # (N, 3, 3)

        # 4. WLF shift (isothermal for now)
        aT = 1.0

        # 5. Prony update (batch numpy — same scalar beta/gamma for all N)
        h_new  = np.zeros_like(h_prev)
        g_eff  = self.g_inf
        for i in range(M):
            tau_eff = float(self.tau_i[i]) * aT
            ratio   = dt / max(tau_eff, 1e-30)
            if ratio < 1e-12:
                beta_i  = 1.0 - ratio
                gamma_i = 1.0
            else:
                beta_i  = float(np.exp(-ratio))
                gamma_i = (1.0 - beta_i) / ratio
            h_new[:, i] = beta_i * h_prev[:, i] + self.g_i[i] * gamma_i * dS_dev
            g_eff += self.g_i[i] * beta_i
        h_new[:, M] = S_dev_el

        # 6. Effective stress (batch)
        S_eff = S_vol + self.g_inf * S_dev_el
        for i in range(M):
            S_eff += h_new[:, i]
        S_voigt = np.stack([S_eff[:, 0, 0], S_eff[:, 1, 1], S_eff[:, 0, 1]], axis=1)

        # 7. Effective tangent (batch)
        K_mean = (C_el[:, 0, 0] + C_el[:, 0, 1] + C_el[:, 1, 0] + C_el[:, 1, 1]) / 4.0
        m      = np.array([1.0, 1.0, 0.0])
        C_vol  = K_mean[:, None, None] * (m[:, None] * m[None, :])
        C_eff  = C_vol + g_eff * (C_el - C_vol)           # (N, 3, 3)

        # 8. Pack state back: (N, M+1, 3, 3) -> (N, 6*(M+1))
        state_new_flat = _tensor_3d_batch_to_flat(h_new, M)

        # --- Finite regularization for degenerate elements ---
        # Zero stress + stable isotropic plane-strain tangent; history kept.
        if np.any(bad):
            C_iso = np.array([
                [lam + 2.0 * mu, lam, 0.0],
                [lam, lam + 2.0 * mu, 0.0],
                [0.0, 0.0, mu],
            ], dtype=np.float64)
            S_voigt[bad] = 0.0
            C_eff[bad] = C_iso
            state_new_flat[bad] = state_flat_batch[bad]

        return S_voigt, C_eff, state_new_flat

    def __repr__(self) -> str:
        wlf = "WLF" if self.wlf_params else "isothermal"
        return f"ViscoelasticMaterial(base={self.base}, M={self.M}, g_oo={self.g_inf:.4f}, {wlf})"
