"""
linear_viscoelastic.py
======================
Small-strain linear viscoelasticity (Abaqus *VISCOELASTIC, TIME=PRONY) with
time-temperature superposition (*TRS, WLF) — designed for the Q1P0 hybrid
(mean-dilatation) element to avoid volumetric locking.

Formulation (plane strain, deviatoric/volumetric split)
-------------------------------------------------------
Bulk response is elastic:           p  = K · θ             (θ = tr(ε))
Deviatoric response relaxes via a Prony series of internal stresses q_i:

    s(t+Δt)  =  2·G_∞·e(t+Δt)  +  Σ_i q_i(t+Δt)

    q_i(t+Δt)  =  a_i·q_i(t)  +  2·G_i·γ_i·( e(t+Δt) − e(t) )

    a_i  =  exp(−Δt / τ_i*)           (recurrence coefficient)
    γ_i  =  (1 − a_i)·τ_i* / Δt      (integration scaling)

    τ_i* =  τ_i · a_T(T)              (WLF shift, time-temperature superposition)

    log₁₀(a_T)  =  −C₁·(T−Tᵣ) / (C₂ + T−Tᵣ)       (WLF equation)

    G_∞ = G₀·(1 − Σ g_i),   G_i = G₀·g_i
    G₀  = E / (2(1+ν)),     K  = E / (3(1−2ν))

    e = ε − (θ/3)·I         (deviatoric strain, tensorial components)
    e_zz = −θ/3  (in plane strain ε_zz=0 but e_zz ≠ 0)

Algorithmic shear modulus (consistent tangent)
----------------------------------------------
    G_alg = G_∞ + Σ_i G_i·γ_i      ds = 2·G_alg·de

This is the exact linearisation of the recurrence, giving the correct
consistent tangent for Newton-Raphson convergence.  The tangent is isotropic
(deviatoric) because all q_i terms are proportional to the deviatoric strain
increment through the scalar γ_i.

Internal-variable layout per Gauss point (flat, length 4·(M+1)):
    [ e_prev(4) , q_1(4) , … , q_M(4) ]
Tensor components ordered: (xx, yy, zz, xy)   — these are full (3×3) tensor
components, NOT Voigt engineering shear.  This means the stored "xy" is the
tensor shear ε¹², NOT γ¹² = 2ε¹².  This convention is used throughout the
codebase for consistency with the 3D finite-strain formulation.

Stability:  The recurrence a_i = exp(−Δt/τ_i) is unconditionally stable for
Δt > 0 (A-stable, Dahlquist 1963).  For Δt ≫ τ_i the term reduces to zero
(full relaxation within one step), which is physically correct but can cause
oscillations in the internal stress if step sizes vary erratically.

WLF reference temperature Tᵣ is set at construction.  If T = Tᵣ (isothermal),
a_T = 1 and no time-scaling occurs.  Currently the folding simulation is
isothermal (room temperature), but the WLF infrastructure is retained for
thermal folding studies.

Project-specific context (display folding)
------------------------------------------
- PSA (E = 1.0–10 MPa, ν = 0.49) is nearly incompressible and requires the
  Q1P0 hybrid (mean-dilatation) element at the ELEMENT level (not the
  material level) to avoid volumetric locking.
- Prony series set to one term: g_i = [0.8], τ_i = [1.0 s].  This gives
  G_∞/G₀ = 0.2 — i.e., the shear modulus relaxes by 80% within ~1 s of
  sustained loading, modelling the PSA creep observed experimentally.
- The viscoelastic relaxation allows the PSA to accommodate shear strain
  between PET layers during folding, preventing unrealistic stress build-up
  at the PET/PSA interface.
- Time-dependent amplitude: the 90° fold is applied over t_total ≈ 1 s,
  which is comparable to τ₁ = 1.0 s, so the viscoelastic effects are active
  throughout the folding simulation (not in the "long-time" asymptotic limit).

References
----------
- Abaqus 2022 *VISCOELASTIC documentation (TIME=PRONY).
- Christensen, R.M. (1982). Theory of Viscoelasticity, 2nd ed., Academic Press.
- Williams, M.L., Landel, R.F., Ferry, J.D. (1955). The temperature dependence
  of relaxation mechanisms in amorphous polymers. JACS, 77(14), 3701-3707.
  — The WLF equation (original paper).
- Ferry, J.D. (1980). Viscoelastic Properties of Polymers, 3rd ed., Wiley.
  — Full treatment of Prony series/master curve / WLF.
- Simo, J.C. & Hughes, T.J.R. (1998). Computational Inelasticity. Springer.
  Ch. 9: viscoelasticity and the algorithmic tangent.
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

    # ------------------------------------------------------------------
    # Batch processing for vectorized assembly
    # ------------------------------------------------------------------

    def pk2_tangent_voigt_batch(
        self,
        F_flat: np.ndarray,      # (N_tot, 2, 2) deformation gradients
        params: dict,
        state_flat: np.ndarray,  # (N_tot, n_vars) internal variables
        dt: float,
        temperature: float = 20.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch processing for multiple Gauss points at once.

        Parameters
        ----------
        F_flat : (N_tot, 2, 2) deformation gradients
        params : dict (unused, for interface compatibility)
        state_flat : (N_tot, n_vars) internal variables
        dt : time increment
        temperature : temperature for WLF shift

        Returns
        -------
        S_flat : (N_tot, 3) PK2 stress in Voigt order [S11, S22, S12]
        C_flat : (N_tot, 3, 3) tangent stiffness
        state_new_flat : (N_tot, n_vars) updated internal variables
        """
        N_tot = F_flat.shape[0]
        n_vars = self.n_internal_vars
        M = self.M

        # Get Prony coefficients for current dt and temperature
        a, gamma, G_alg = self.prony_coeffs(dt, temperature)

        # Initialize output arrays
        S_flat = np.zeros((N_tot, 3), dtype=np.float64)
        C_flat = np.zeros((N_tot, 3, 3), dtype=np.float64)
        state_new_flat = np.zeros((N_tot, n_vars), dtype=np.float64)

        # Bulk modulus (volumetric response is elastic)
        K = self.K

        # Process all Gauss points in a vectorized manner
        # Extract strains from deformation gradients (small-strain assumption)
        # F = I + grad_u => eps = (grad_u + grad_u^T)/2
        # For plane strain: eps_zz = 0

        # Previous deviatoric strain: e_prev = state[:, :4]
        e_prev = state_flat[:, :4]  # (N_tot, 4)

        # Current total strain from F (small strain)
        # eps_xx = F[0,0] - 1, eps_yy = F[1,1] - 1, eps_xy = (F[0,1] + F[1,0])/2
        eps_xx = F_flat[:, 0, 0] - 1.0
        eps_yy = F_flat[:, 1, 1] - 1.0
        eps_xy = 0.5 * (F_flat[:, 0, 1] + F_flat[:, 1, 0])

        # Volumetric strain (trace)
        theta = eps_xx + eps_yy  # plane strain: eps_zz = 0

        # Deviatoric strain: e = eps - (theta/3) * I
        # For plane strain with eps_zz = 0:
        # e_xx = eps_xx - theta/3
        # e_yy = eps_yy - theta/3
        # e_zz = 0 - theta/3 = -theta/3
        # e_xy = eps_xy
        e_xx = eps_xx - theta / 3.0
        e_yy = eps_yy - theta / 3.0
        e_zz = -theta / 3.0
        e_xy = eps_xy

        e_dev_new = np.column_stack([e_xx, e_yy, e_zz, e_xy])  # (N_tot, 4)

        # Deviatoric stress update (vectorized Prony series)
        de = e_dev_new - e_prev  # (N_tot, 4)

        # s_dev = 2 * G_inf * e_dev_new + sum_i q_i
        s_dev = 2.0 * self.G_inf * e_dev_new  # (N_tot, 4)

        # Update Prony internal variables q_i
        state_new_flat[:, :4] = e_dev_new
        for i in range(M):
            q_prev = state_flat[:, 4 * (i + 1):4 * (i + 2)]  # (N_tot, 4)
            q_new = a[i] * q_prev + 2.0 * self.G_terms[i] * gamma[i] * de
            s_dev += q_new
            state_new_flat[:, 4 * (i + 1):4 * (i + 2)] = q_new

        # Convert deviatoric stress to Voigt form [S11, S22, S12]
        S_flat[:, 0] = s_dev[:, 0]  # S_xx
        S_flat[:, 1] = s_dev[:, 1]  # S_yy
        S_flat[:, 2] = s_dev[:, 3]  # S_xy

        # Add volumetric stress: p = K * theta
        p = K * theta
        S_flat[:, 0] += p
        S_flat[:, 1] += p

        # Tangent stiffness (consistent tangent)
        # For linear viscoelasticity:
        # C_dev = 2 * G_alg * I_dev (deviatoric part)
        # C_vol = K * I_vol (volumetric part)
        G2 = 2.0 * G_alg

        # Deviatoric tangent components (plane strain)
        # C_dev[0,0] = C_dev[1,1] = 4/3 * G2
        # C_dev[0,1] = C_dev[1,0] = -2/3 * G2
        # C_dev[2,2] = G2
        C43 = 4.0 / 3.0 * G2
        C23 = 2.0 / 3.0 * G2

        C_flat[:, 0, 0] = C43 + K
        C_flat[:, 0, 1] = K - C23
        C_flat[:, 0, 2] = 0.0
        C_flat[:, 1, 0] = K - C23
        C_flat[:, 1, 1] = C43 + K
        C_flat[:, 1, 2] = 0.0
        C_flat[:, 2, 0] = 0.0
        C_flat[:, 2, 1] = 0.0
        C_flat[:, 2, 2] = G2

        return S_flat, C_flat, state_new_flat

    def __repr__(self) -> str:
        trs = "WLF" if self.wlf_params else "isothermal"
        return (f"LinearViscoelastic(E={self.E}, nu={self.nu}, M={self.M}, "
                f"g_oo={self.g_inf:.4f}, {trs})")
