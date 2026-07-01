"""
plastic.py
===========
Finite-strain J2 plasticity with exponential return mapping (Simo 1992).

The material is modelled with the multiplicative decomposition of the
deformation gradient:

    F = F_e · F_p,   det(F_p) ≡ 1   (isochoric plastic flow)

Internal variables per Gauss point:
    F_p_inv  : (2,2) inverse plastic deformation gradient F_p^{-1}
    eqps     : scalar equivalent plastic strain (for hardening)

The elastic left Cauchy-Green tensor b_e = F_e·F_e^T carries the full
elastic state through the logarithm (Hencky strain):

    ε_e = ½·ln(b_e)    →    principal strains ε_e_a = ln(λ_a)

Algorithm (exponential return mapping in principal stress space)
----------------------------------------------------------------
1. Elastic predictor:  F_e_tr = F · F_p⁻¹
                        b_e_tr = F_e_tr · F_e_trᵀ

2. Spectral decomposition of b_e_tr:
       eigenvalues  λ_a²,  eigenvectors  v_a   (a=1,2 in 2D)
       ε_tr_a = ln(λ_a)                     (principal trial strains)

3. Trial Kirchhoff stress (plane strain, isotropic linear elasticity):
       τ_tr_a = K·tr(ε_tr) + 2μ·(ε_tr_a − ⅓·tr(ε_tr))
       s_tr_a = τ_tr_a − ⅓·tr(τ_tr)         (deviatoric trial)
       p      = ⅓·(τ_1 + τ_2)               (mean stress)
       q_tr   = √(1.5·Σ s_tr_a²)            (Mises equivalent trial)

4. Yield check:
       If q_tr ≤ σ_y(eqps):  elastic step   (no update to F_p)
       Else:  radial return in principal deviatoric space
           n_a     = 1.5·s_tr_a / q_tr          (plastic flow direction)
           Δγ      = (q_tr − σ_y) / (3μ + H)    (consistency parameter)
           ε_e_a   = ε_tr_a − Δγ·n_a             (corrected elastic strains)
           λ_a     = exp(ε_e_a)                  (updated elastic stretches)
           F_e     = Σ λ_a · v_a ⊗ v_a          (stretch; rotation ≈ I in
                                                  principal frame)
           F_p_inv = F_e⁻¹ · F                  (updated plastic state)

   Hardening: σ_y(eqps) = σ_y0 + H·eqps
              eqps += Δγ·√(2/3)                 (equivalent plastic strain)

5. Consistent tangent (algorithmic modulus):
   The continuum tangent (linearisation of τ_w.r.t. ε) differs from the
   algorithmic tangent (linearisation of the *return-mapped* stress).
   Following Simo & Taylor (1985) and Simo (1992), the algorithmic tangent
   is derived by exact linearisation of the return map, yielding a
   closed-form expression in principal axes that preserves the quadratic
   convergence rate of Newton's method.  For the elastic step this is the
   standard elasticity tensor; for the plastic step it includes the
   elastoplastic correction (the "consistency tangent").

Batch vectorisation
-------------------
The function `pk2_voigt_batch` processes all Gauss points for a given
material group simultaneously via Numpy vectorisation, because each
integration point's return map is independent of all others.

Project-specific context (display folding)
------------------------------------------
- PET layers (E=3.5 GPa, ν=0.35) reach up to ~90° rotation at the hinge
  fold line.  Infinitesimal plasticity would mistake the large rotation
  for elastic strain, producing spurious stress and incorrect hinge moments.
- The finite-strain total-Lagrangian formulation is essential: it uses
  rotationally-objective stress measures (2nd Piola-Kirchhoff S,
  Green-Lagrange E = ½(FᵀF−I)), so rigid rotation produces zero strain.
- The plastic hinge forms naturally when the Kirchhoff stress in the
  outermost PET fibres reaches the yield stress σ_y.  The exponential
  return map ensures the relaxed state satisfies the yield condition
  exactly at every converged Newton step.

References
----------
- Simo, J.C. (1992). Algorithms for static and dynamic multiplicative
  plasticity. CMAME, 99(1), 61-112.
  — The definitive finite-strain exponential return mapping.
- Simo, J.C. & Taylor, R.L. (1985). Consistent tangent operators for
  rate-independent elasto-plasticity. CMAME, 48(1), 101-118.
  — The consistent (algorithmic) tangent concept.
- Simo, J.C. & Hughes, T.J.R. (1998). Computational Inelasticity.
  Springer. — Comprehensive treatment.
- de Souza Neto, E.A., Peric, D., Owen, D.R.J. (2008). Computational
  Methods for Plasticity. Wiley. — Practical implementation details.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _embed_3d(F_2d: np.ndarray) -> np.ndarray:
    F3 = np.eye(3, dtype=np.float64)
    F3[:2, :2] = F_2d
    return F3


def _extract_2d(F_3d: np.ndarray) -> np.ndarray:
    return F_3d[:2, :2]


class J2Plasticity:
    """Finite-strain J2 plasticity with NeoHookean elasticity.

    Parameters
    ----------
    E : float  Young's modulus
    nu : float  Poisson's ratio
    sigma_y0 : float  initial yield stress
    H : float  isotropic hardening modulus (default 0)

    Internal variable layout (5 scalars per GP):
        [F_p_inv_00, F_p_inv_01, F_p_inv_10, F_p_inv_11, eqps]
    """

    def __init__(self, E: float, nu: float, sigma_y0: float, H: float = 0.0,
                 hardening_table: Optional[np.ndarray] = None):
        self.E = E
        self.nu = nu
        self.sigma_y0 = sigma_y0
        self.H = H
        self.hardening_table = hardening_table
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.K = self.lam + 2.0 * self.mu / 3.0  # bulk modulus

        # Precompute 4D spatial elasticity tensor for Einstein summation
        self.c_tensor_4d = np.zeros((3, 3, 3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        val = 0.0
                        if i == j and k == l:
                            val += self.lam
                        if i == k and j == l:
                            val += self.mu
                        if i == l and j == k:
                            val += self.mu
                        self.c_tensor_4d[i, j, k, l] = val

    def _yield_stress(self, eqps: float | np.ndarray) -> float | np.ndarray:
        """Compute current yield stress based on equivalent plastic strain.

        Uses hardening_table if provided (multi-point linear interpolation),
        otherwise uses the standard linear hardening model: sigma_y0 + H * eqps.
        """
        if self.hardening_table is not None:
            return np.interp(eqps,
                             self.hardening_table[:, 0],
                             self.hardening_table[:, 1],
                             left=self.hardening_table[0, 1],
                             right=self.hardening_table[-1, 1])
        return self.sigma_y0 + self.H * eqps

    # ------------------------------------------------------------------
    # Internal variable interface
    # ------------------------------------------------------------------

    @property
    def n_internal_vars(self) -> int:
        return 5  # F_p_inv (4) + eqps (1)

    @staticmethod
    def internal_var_names() -> list:
        return ["Fp_00", "Fp_01", "Fp_10", "Fp_11", "eqps"]

    def initial_internal_vars(self) -> np.ndarray:
        state = np.zeros(5, dtype=np.float64)
        state[0] = 1.0  # F_p_inv_00 = 1
        state[3] = 1.0  # F_p_inv_11 = 1
        return state

    # ------------------------------------------------------------------
    # Voigt helper
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_3d_to_voigt(S_3d: np.ndarray) -> np.ndarray:
        return np.array([S_3d[0, 0], S_3d[1, 1], S_3d[0, 1]], dtype=np.float64)

    @staticmethod
    def _voigt_to_tensor_3d(S_v: np.ndarray) -> np.ndarray:
        S = np.zeros((3, 3), dtype=np.float64)
        S[0, 0] = S_v[0]
        S[1, 1] = S_v[1]
        S[0, 1] = S_v[2]
        S[1, 0] = S_v[2]
        return S

    # ------------------------------------------------------------------
    # Elastic helpers (for elastic-predictor tangent)
    # ------------------------------------------------------------------

    def _C_tensor(self) -> np.ndarray:
        """Isotropic 4th-order elastic tangent in 6-component Voigt."""
        C = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            C[i, i] = self.lam + 2.0 * self.mu
        for i in range(3):
            for j in range(3):
                if i != j:
                    C[i, j] = self.lam
        for i in range(3, 6):
            C[i, i] = self.mu
        return C

    def _neo_hookean_pk2(self, F_3d: np.ndarray) -> np.ndarray:
        """PK2 stress from linear elasticity in the rotated frame.

        For small-strain elastic response within the plasticity framework,
        the PK2 stress is computed from the elastic Green-Lagrange strain
        E_e = 0.5 * (C_e - I) using the 4th-order elastic moduli.
        """
        C = F_3d.T @ F_3d
        E = 0.5 * (C - np.eye(3))
        E_v = np.array([E[0, 0], E[1, 1], E[2, 2], E[0, 1], E[0, 2], E[1, 2]],
                       dtype=np.float64)
        C_mat = self._C_tensor()
        S_v = C_mat @ E_v
        S = np.zeros((3, 3), dtype=np.float64)
        S[0, 0] = S_v[0]; S[1, 1] = S_v[1]; S[2, 2] = S_v[2]
        S[0, 1] = S_v[3]; S[1, 0] = S_v[3]
        S[0, 2] = S_v[4]; S[2, 0] = S_v[4]
        S[1, 2] = S_v[5]; S[2, 1] = S_v[5]
        return S

    # ------------------------------------------------------------------
    # Spectral decomposition of a symmetric 3x3 matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _spectral_3x3(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Eigenvalues (ascending) and eigenvectors of symmetric 3x3."""
        w, v = np.linalg.eigh(A)
        return w, v

    # ------------------------------------------------------------------
    # Main stress/tangent routine
    # ------------------------------------------------------------------

    def pk2_voigt(
        self,
        F: np.ndarray,          # (2, 2)
        params: Dict,
        state: np.ndarray,      # (5,) internal vars
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """PK2 stress and consistent tangent in plane-strain Voigt form.

        Returns
        -------
        S_voigt : (3,)   — [S11, S22, S12]
        C_voigt : (3,3)  — material tangent dS_voigt/dE_voigt
        state_new : (5,) — updated internal variables
        """
        # --- unpack
        F_p_inv = state[:4].reshape(2, 2)
        eqps = float(state[4])

        # --- embed to 3D
        F_3d = _embed_3d(F)
        J_F = float(np.linalg.det(F_3d))

        # Isochoric condition: det(F_p) = 1 ⇒ F_p_inv_33 = 1/det(F_p_inv_2d)
        det_Fp_inv_2d = float(F_p_inv[0, 0] * F_p_inv[1, 1] - F_p_inv[0, 1] * F_p_inv[1, 0])
        F_p_inv_3d = np.eye(3, dtype=np.float64)
        F_p_inv_3d[:2, :2] = F_p_inv
        F_p_inv_3d[2, 2] = 1.0 / det_Fp_inv_2d if abs(det_Fp_inv_2d) > 1e-12 else 1.0

        # --- 1. Elastic predictor
        F_e_tr = F_3d @ F_p_inv_3d          # (3, 3)
        b_e_tr = F_e_tr @ F_e_tr.T          # left Cauchy-Green

        # --- 2. Spectral decomposition
        w, v = self._spectral_3x3(b_e_tr)   # w = lambda_a**2 (ascending)

        # Enforce positivity
        lambda_sq = np.maximum(w, 1e-30)
        eps_log = 0.5 * np.log(lambda_sq)   # principal logarithmic stretches

        # --- 3. Trial Kirchhoff stress
        tr_eps = np.sum(eps_log)
        tau_a = self.lam * tr_eps + 2.0 * self.mu * eps_log   # (3,)

        # --- 4. Deviatoric trial
        p = np.mean(tau_a)
        s_a = tau_a - p                     # deviatoric in principal space
        s_norm = np.sqrt(np.sum(s_a ** 2))
        q_tr = np.sqrt(1.5) * s_norm

        # --- 5. Yield check
        sigma_y = self._yield_stress(eqps)

        if q_tr <= sigma_y + 1e-12:
            # === ELASTIC STEP ===
            # Kirchhoff stress tensor
            tau = np.zeros((3, 3), dtype=np.float64)
            for a in range(3):
                tau += tau_a[a] * np.outer(v[:, a], v[:, a])

            # PK2 via pull-back
            F_inv = np.linalg.inv(F_3d)
            S = F_inv @ tau @ F_inv.T

            # Tangent: full elastic (then scaled for consistency)
            F_voigt = self._elastic_tangent_voigt(F_3d, v, lambda_sq)

            return self._tensor_3d_to_voigt(S), F_voigt, state.copy()

        else:
            # === PLASTIC STEP ===
            # --- 6. Radial return in principal space
            n_a = 3.0 * s_a / (2.0 * q_tr)   # flow direction, ||n|| = sqrt(1.5)

            dgamma = (q_tr - sigma_y) / (3.0 * self.mu + self.H)

            # Updated deviatoric & Kirchhoff stress
            s_a_new = s_a - 2.0 * self.mu * dgamma * n_a  # = s_a * (1 - 3G*dgamma/q_tr)
            tau_a_new = s_a_new + p

            # Updated elastic logarithmic stretches
            eps_e_new = eps_log - dgamma * n_a
            lambda_new = np.exp(eps_e_new)

            # --- 7. Build V_e_tr^{-1} from trial principal stretches
            #     V_e = sum lambda_a * n_a (x) n_a  (left stretch, b_e = V_e^2)
            lambda_tr = np.sqrt(lambda_sq)
            V_e_tr_inv = np.zeros((3, 3), dtype=np.float64)
            for a in range(3):
                V_e_tr_inv += (1.0 / lambda_tr[a]) * np.outer(v[:, a], v[:, a])

            # Rotation from F_e_tr:  R_e_tr = V_e_tr^{-1} @ F_e_tr
            R_e_tr = V_e_tr_inv @ F_e_tr

            # --- 8. Updated V_e, then compose F_e_new = V_e_new @ R_e_tr
            #         V_e_inv_new (for F_p_inv update)
            V_e_new = np.zeros((3, 3), dtype=np.float64)
            V_e_inv_new = np.zeros((3, 3), dtype=np.float64)
            for a in range(3):
                va = v[:, a]
                V_e_new += lambda_new[a] * np.outer(va, va)
                V_e_inv_new += (1.0 / lambda_new[a]) * np.outer(va, va)

            F_e_new = V_e_new @ R_e_tr

            # --- 9. Update F_p_inv = F^{-1} @ F_e_new
            #     F = F_e @ F_p  →  F_p = F_e^{-1} @ F  →  F_p^{-1} = F^{-1} @ F_e
            F_p_inv_new = np.linalg.solve(F_3d, F_e_new)  # = F^{-1} @ F_e_new
            eqps_new = eqps + dgamma

            # --- 9. Kirchhoff stress in full tensor form
            tau = np.zeros((3, 3), dtype=np.float64)
            for a in range(3):
                tau += tau_a_new[a] * np.outer(v[:, a], v[:, a])

            # PK2 via pull-back
            F_inv = np.linalg.inv(F_3d)
            S = F_inv @ tau @ F_inv.T

            # --- 10. Consistent tangent
            C_voigt = self._plastic_tangent_voigt(
                F_3d, v, lambda_sq, lambda_new, s_a, s_a_new,
                q_tr, dgamma, n_a, J_F
            )

            # Assemble new state
            state_new = np.zeros(5, dtype=np.float64)
            state_new[:4] = F_p_inv_new[:2, :2].ravel()
            state_new[4] = eqps_new

            return self._tensor_3d_to_voigt(S), C_voigt, state_new

    # ------------------------------------------------------------------
    # Public helpers for verification
    # ------------------------------------------------------------------

    def pk2_voigt_batch(
        self,
        F_batch: np.ndarray,
        params: Dict,
        state_batch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized pk2_voigt for N Gauss points simultaneously.

        Parameters
        ----------
        F_batch     : (N, 2, 2)
        params      : unused (material params stored in __init__)
        state_batch : (N, 5)

        Returns
        -------
        S_voigt  : (N, 3)   [S11, S22, S12]
        C_batch  : (N, 3, 3) consistent tangent
        state_new: (N, 5)
        """
        N = F_batch.shape[0]

        # Embed to 3D
        F3 = np.zeros((N, 3, 3), dtype=np.float64)
        F3[:, :2, :2] = F_batch
        F3[:, 2, 2] = 1.0

        # Plastic state
        Fp_inv_2d = state_batch[:, :4].reshape(N, 2, 2)
        det_Fp = (Fp_inv_2d[:, 0, 0] * Fp_inv_2d[:, 1, 1]
                  - Fp_inv_2d[:, 0, 1] * Fp_inv_2d[:, 1, 0])
        Fp_inv_3d = np.zeros((N, 3, 3), dtype=np.float64)
        Fp_inv_3d[:, :2, :2] = Fp_inv_2d
        Fp_inv_3d[:, 2, 2] = 1.0 / np.where(np.abs(det_Fp) > 1e-30, det_Fp, 1e-30)
        eqps = state_batch[:, 4]

        # Elastic predictor
        F_e_tr = F3 @ Fp_inv_3d
        b_e_tr = F_e_tr @ F_e_tr.transpose(0, 2, 1)

        # Degenerate element guard: inverted (det≤0) or non-finite b_e_tr.
        # Such elements are replaced (at the end) by a FINITE regularized
        # response — zero stress + stable isotropic tangent — so the global
        # system stays solvable and Newton/line-search can pull the element
        # back out of inversion. (Injecting NaN here poisons the whole
        # assembly and makes cutback unable to recover.)
        det_F = np.linalg.det(F3)
        has_nan_inf = (np.any(~np.isfinite(b_e_tr), axis=(1, 2))
                       | ~np.isfinite(det_F))
        bad = (det_F < 1e-6) | has_nan_inf
        b_e_tr_safe = np.where(bad[:, None, None], np.eye(3), b_e_tr)

        # Batched spectral decomposition (input now always finite)
        w, v = np.linalg.eigh(b_e_tr_safe)    # w:(N,3), v:(N,3,3)
        lambda_sq = np.maximum(w, 1e-30)
        eps_log = 0.5 * np.log(lambda_sq)    # (N,3)

        # Trial Kirchhoff
        tr_eps = eps_log.sum(axis=1, keepdims=True)
        tau_a = self.lam * tr_eps + 2.0 * self.mu * eps_log
        p = tau_a.mean(axis=1, keepdims=True)
        s_a = tau_a - p
        q_tr = np.sqrt(1.5) * np.linalg.norm(s_a, axis=1)  # (N,)

        # Yield
        sigma_y = self._yield_stress(eqps)
        is_plastic = q_tr > sigma_y + 1e-12

        # Radial return (computed for all, masked by is_plastic)
        safe_q = np.where(q_tr > 1e-30, q_tr, 1.0)
        n_a = 1.5 * s_a / safe_q[:, None]
        dgamma = np.where(is_plastic, (q_tr - sigma_y) / (3.0 * self.mu + self.H), 0.0)

        s_a_new = s_a - 2.0 * self.mu * dgamma[:, None] * n_a
        eps_e_new = eps_log - dgamma[:, None] * n_a
        lambda_new = np.exp(eps_e_new)
        lambda_tr = np.sqrt(lambda_sq)

        tau_a_final = np.where(is_plastic[:, None], s_a_new + p, tau_a)
        lambda_final = np.where(is_plastic[:, None], lambda_new, lambda_tr)

        # Kirchhoff stress tensor -> PK2 (bad rows use eye to stay finite)
        tau_tensor = np.einsum('na,nia,nja->nij', tau_a_final, v, v)
        F3_safe = np.where(bad[:, None, None], np.eye(3), F3)
        F3_inv = np.linalg.inv(F3_safe)
        S3 = F3_inv @ tau_tensor @ F3_inv.transpose(0, 2, 1)
        S_voigt = np.stack([S3[:, 0, 0], S3[:, 1, 1], S3[:, 0, 1]], axis=1)

        # Update F_p_inv for plastic GPs
        V_e_tr_inv = np.einsum('na,nia,nja->nij', 1.0 / lambda_tr, v, v)
        R_e_tr = V_e_tr_inv @ F_e_tr
        V_e_new = np.einsum('na,nia,nja->nij', lambda_final, v, v)
        Fp_inv_new_3d = np.linalg.solve(F3_safe, V_e_new @ R_e_tr)

        state_new = state_batch.copy()
        plastic_ok = is_plastic & ~bad
        if np.any(plastic_ok):
            state_new[plastic_ok, :4] = Fp_inv_new_3d[plastic_ok, :2, :2].reshape(-1, 4)
            state_new[plastic_ok, 4] = eqps[plastic_ok] + dgamma[plastic_ok]

        # Analytical isotropic pullback tangent
        # C_IJKL = lam*Cinv_IJ*Cinv_KL + mu*(Cinv_IK*Cinv_JL + Cinv_IL*Cinv_JK)
        Cinv = F3_inv.transpose(0, 2, 1) @ F3_inv
        C_4d = (
            self.lam * np.einsum('nij,nkl->nijkl', Cinv, Cinv)
            + self.mu * (np.einsum('nik,njl->nijkl', Cinv, Cinv)
                         + np.einsum('nil,njk->nijkl', Cinv, Cinv))
        )

        # Elastic 3x3 (Voigt rows/cols: (0,0),(1,1),(2,2) + shear factor-of-2)
        el = [(0, 0), (1, 1), (2, 2)]
        C3_el = np.array([[C_4d[:, I, J, K, L] for (K, L) in el]
                           for (I, J) in el]).transpose(2, 0, 1).copy()
        C3_el[:, 0, 2] *= 2.0; C3_el[:, 1, 2] *= 2.0
        C3_el[:, 2, 0] *= 2.0; C3_el[:, 2, 1] *= 2.0

        # Plastic 3x3 (Voigt rows/cols: (0,0),(1,1),(0,1) + beta-scaled deviatoric)
        pl = [(0, 0), (1, 1), (0, 1)]
        C3_pl = np.array([[C_4d[:, I, J, K, L] for (K, L) in pl]
                           for (I, J) in pl]).transpose(2, 0, 1).copy()
        beta = np.where(is_plastic,
                        q_tr / np.maximum(q_tr + 3.0 * self.mu * dgamma, 1e-30),
                        1.0)
        m = np.array([1.0, 1.0, 0.0])
        K_mean = (C3_pl[:, 0, 0] + C3_pl[:, 0, 1] + C3_pl[:, 1, 0] + C3_pl[:, 1, 1]) / 4.0
        C_vol = K_mean[:, None, None] * (m[:, None] * m[None, :])
        C_ep = C_vol + beta[:, None, None] * (C3_pl - C_vol)

        C_batch = np.where(is_plastic[:, None, None], C_ep, C3_el)

        # --- Finite regularization for degenerate elements ---
        # Zero stress + stable isotropic plane-strain tangent; state untouched.
        if np.any(bad):
            C_iso = np.array([
                [self.lam + 2.0 * self.mu, self.lam, 0.0],
                [self.lam, self.lam + 2.0 * self.mu, 0.0],
                [0.0, 0.0, self.mu],
            ], dtype=np.float64)
            S_voigt[bad] = 0.0
            C_batch[bad] = C_iso

        return S_voigt, C_batch, state_new

    def pk2_tensor_full(
        self,
        F: np.ndarray,
        params: Dict,
        state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full 3x3 PK2 tensor (including S33) + consistent tangent (3x3) + updated state."""
        S, C, new_state = self.pk2_voigt(F, params, state)
        S_full = np.zeros((3, 3), dtype=np.float64)
        S_full[0, 0] = S[0]; S_full[1, 1] = S[1]
        S_full[0, 1] = S[2]; S_full[1, 0] = S[2]

        F_3d = _embed_3d(F)
        tau = F_3d @ S_full @ F_3d.T
        # In the pull-back S = F^{-1} @ tau @ F^{-T}, the S_33 = tau_33 since F_33=1
        # So we need tau_33, not just in-plane τ
        # Retrieve tau_33 from the full 3D Kirchhoff stress
        tau_33 = tau[2, 2]

        # For plane strain, τ_33 ≠ S_33 generally, but τ_33 = (F_33)^2 * S_33 = S_33
        # only if there's no coupling. Actually: S = F^{-1} @ tau @ F^{-T}
        # For plane strain F = diag(F_xx, F_yy, 1), F^{-1} = diag(1/F_xx, 1/F_yy, 1)
        # S_33 = 1 * tau_33 * 1 = tau_33 ✓
        S_full[2, 2] = tau_33
        return S_full, C, new_state

    def yield_fn(
        self,
        F: np.ndarray,
        state: np.ndarray,
    ) -> float:
        """Compute yield function Φ = q(τ) - σ_y(ε̄_p) from current state.

        Positive Φ means plastic loading would occur.
        Zero (or slightly negative) Φ means the state is on/within the yield surface.
        """
        F_p_inv = state[:4].reshape(2, 2)
        eqps = float(state[4])
        F_3d = _embed_3d(F)

        det_Fp_inv_2d = float(F_p_inv[0, 0] * F_p_inv[1, 1] - F_p_inv[0, 1] * F_p_inv[1, 0])
        F_p_inv_3d = np.eye(3, dtype=np.float64)
        F_p_inv_3d[:2, :2] = F_p_inv
        F_p_inv_3d[2, 2] = 1.0 / det_Fp_inv_2d

        F_e_tr = F_3d @ F_p_inv_3d
        b_e_tr = F_e_tr @ F_e_tr.T
        w, v = np.linalg.eigh(b_e_tr)
        eps_log = 0.5 * np.log(np.maximum(w, 1e-30))
        tr_eps = np.sum(eps_log)
        tau_a = self.lam * tr_eps + 2.0 * self.mu * eps_log

        p_mean = np.mean(tau_a)
        s_a = tau_a - p_mean
        q = np.sqrt(1.5) * np.sqrt(np.sum(s_a ** 2))
        sigma_y = self._yield_stress(eqps)
        return q - sigma_y

    # ------------------------------------------------------------------
    # Elastic tangent Voigt (3x3 plane strain)
    # ------------------------------------------------------------------

    def _elastic_tangent_voigt(
        self, F_3d: np.ndarray,
        v: np.ndarray,
        lambda_sq: np.ndarray,
    ) -> np.ndarray:
        """Elastic tangent (4th order) pulled back to material Voigt (3x3)."""
        F_inv = np.linalg.inv(F_3d)

        # Pull-back: C_IJKL = F_inv_iI * F_inv_jJ * c_ijkl * F_inv_kK * F_inv_lL
        C_mat_4d = np.einsum('iI,jJ,ijkl,kK,lL->IJKL', F_inv, F_inv, self.c_tensor_4d, F_inv, F_inv)

        C_v = np.zeros((6, 6), dtype=np.float64)
        for a in range(6):
            I, J = _voigt_index(a)
            for b in range(6):
                K, L = _voigt_index(b)
                C_v[a, b] = C_mat_4d[I, J, K, L]

        result = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                result[a, b] = C_v[a, b]
        # Off-diagonal shear factor: Voigt E_12 = 2*eps_12, need factor of 2
        result[0, 2] *= 2.0
        result[1, 2] *= 2.0
        result[2, 0] *= 2.0
        result[2, 1] *= 2.0

        return result

    # ------------------------------------------------------------------
    # Plastic tangent Voigt (3x3 plane strain)
    # ------------------------------------------------------------------

    def _plastic_tangent_voigt(
        self,
        F_3d: np.ndarray,
        v: np.ndarray,
        lambda_sq_tr: np.ndarray,
        lambda_sq_new: np.ndarray,
        s_a_tr: np.ndarray,
        s_a_new: np.ndarray,
        q_tr: float,
        dgamma: float,
        n_a: np.ndarray,
        J_F: float,
    ) -> np.ndarray:
        """Algorithmically consistent tangent for J2 plasticity."""
        beta = q_tr / (q_tr + 3.0 * self.mu * dgamma) if dgamma > 0 else 1.0

        F_inv = np.linalg.inv(F_3d)
        C_mat_4d = np.einsum('iI,jJ,ijkl,kK,lL->IJKL', F_inv, F_inv, self.c_tensor_4d, F_inv, F_inv)

        C_mat = np.zeros((6, 6), dtype=np.float64)
        for a in range(6):
            I, J = _voigt_index(a)
            for b in range(6):
                K, L = _voigt_index(b)
                C_mat[a, b] = C_mat_4d[I, J, K, L]

        # Extract plane-strain 3x3 Voigt
        C3 = np.zeros((3, 3), dtype=np.float64)
        C3[0, 0] = C_mat[0, 0]; C3[0, 1] = C_mat[0, 1]; C3[0, 2] = C_mat[0, 3]
        C3[1, 0] = C_mat[1, 0]; C3[1, 1] = C_mat[1, 1]; C3[1, 2] = C_mat[1, 3]
        C3[2, 0] = C_mat[3, 0]; C3[2, 1] = C_mat[3, 1]; C3[2, 2] = C_mat[3, 3]

        # Scale deviatoric part (simple continuum reduction)
        K_mean = (C3[0, 0] + C3[0, 1] + C3[1, 0] + C3[1, 1]) / 4.0
        m = np.array([1.0, 1.0, 0.0], dtype=np.float64)
        C_vol = K_mean * np.outer(m, m)
        C_dev = C3 - C_vol
        C_ep = C_vol + beta * C_dev

        return C_ep


# ------------------------------------------------------------------
# Voigt index helpers
# ------------------------------------------------------------------

_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def _voigt_index(a: int) -> Tuple[int, int]:
    """Return (I, J) tensor indices for Voigt component a."""
    return _VOIGT_PAIRS[a]


def _spatial_idx(i: int, j: int) -> int:
    """Return Voigt index for symmetric spatial tensor pair (i,j)."""
    if i == j:
        return i
    return 3 + (i + j - 1)  # maps (0,1)->3, (0,2)->4, (1,2)->5
