"""
test_hyperelastic.py
====================
Verification tests for hyperelastic material models.

AC-3: All three models match small-strain linear elastic limit
      (relative error < 1e-4 at strain = 1e-5).

Uses small uniaxial tension (σ_yy = 0) in plane strain:
    ε_yy = -ν/(1-ν) · ε_xx
    S_11 ≈ σ_xx = E/(1-ν²) · ε_xx
    S_22 ≈ 0, S_12 ≈ 0
"""

import jax
jax.config.update("jax_enable_x64", True)  # enable float64 for accurate comparison

import jax.numpy as jnp
import numpy as np
import pytest

from dispsolver.material import NeoHookean, Yeoh, ArrudaBoyce


# ------------------------------------------------------------------
# Material constants
# ------------------------------------------------------------------
E_YOUNG = 1000.0
NU = 0.3
EPSILON = 1e-5  # applied small strain
REL_TOL = 1e-4   # AC-3: relative error < 1e-4

# Lamé parameters for Neo-Hookean
MU = E_YOUNG / (2.0 * (1.0 + NU))
LAM = E_YOUNG * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
K_BULK = LAM + 2.0 * MU / 3.0  # bulk modulus

NH_PARAMS = {"E": E_YOUNG, "nu": NU}

# Yeoh: C10 = μ/2 (so 2*C10 = μ, matching NH shear modulus)
#       D1 = 2/K  (so K = 2/D1, matching NH bulk modulus)
YEOH_PARAMS = {"C10": MU / 2.0, "C20": 0.0, "C30": 0.0, "D1": 2.0 / K_BULK}

# Arruda-Boyce: mu = μ, lambda_m large → matches NH at small strain
# At λ_m → ∞, the series collapses to just the C1 term → G₀ = μ
# Use λ_m = 100 so higher-order terms are negligible (< 1e-6 effect)
AB_PARAMS = {"mu": MU, "lambda_m": 100.0, "K": K_BULK}


def _plane_strain_uniaxial_F(eps_xx: float = EPSILON, nu: float = NU):
    """Deformation gradient for plane strain uniaxial tension (σ_yy = 0)."""
    eps_yy = -nu / (1.0 - nu) * eps_xx
    return jnp.array([
        [1.0 + eps_xx, 0.0],
        [0.0, 1.0 + eps_yy],
    ])


def _linear_elastic_S11(eps_xx: float = EPSILON, E: float = E_YOUNG, nu: float = NU):
    """Analytical 2nd P-K stress S_11 for plane strain uniaxial tension (small strain).

    σ_xx = E / (1 - ν²) · ε_xx.  For small strain S_11 ≈ σ_11.
    """
    return E / (1.0 - nu ** 2) * eps_xx


# ------------------------------------------------------------------
# Small-strain limit tests
# ------------------------------------------------------------------

class TestSmallStrainLimit:
    """All material models match linear elasticity at small strain."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.F = _plane_strain_uniaxial_F()
        self.S11_exact = _linear_elastic_S11()

    # --- Neo-Hookean ---

    def test_nh_strain_energy_zero_at_I(self):
        model = NeoHookean()
        W = model.strain_energy(jnp.eye(2), NH_PARAMS)
        assert abs(float(W)) < 1e-15

    def test_nh_small_strain_s11(self):
        model = NeoHookean()
        S = model.pk2_voigt(self.F, NH_PARAMS)
        rel_err = abs(float(S[0]) - self.S11_exact) / abs(self.S11_exact)
        assert rel_err < REL_TOL, f"NeoHookean S_11 rel error {rel_err:.2e}"

    def test_nh_small_strain_s22_zero(self):
        model = NeoHookean()
        S = model.pk2_voigt(self.F, NH_PARAMS)
        assert abs(float(S[1])) < EPSILON, f"NeoHookean S_22 = {S[1]:.2e}"

    def test_nh_small_strain_s12_zero(self):
        model = NeoHookean()
        S = model.pk2_voigt(self.F, NH_PARAMS)
        assert abs(float(S[2])) < 1e-15

    def test_nh_tangent_symmetry(self):
        model = NeoHookean()
        C = model.tangent_voigt(self.F, NH_PARAMS)
        diff = float(jnp.abs(C - C.T).max())
        assert diff < 1e-10, f"NeoHookean tangent symmetry error {diff:.2e}"

    # --- YEOH ---

    def test_yeoh_strain_energy_zero_at_I(self):
        model = Yeoh()
        W = model.strain_energy(jnp.eye(2), YEOH_PARAMS)
        assert abs(float(W)) < 1e-15

    def test_yeoh_small_strain_s11(self):
        model = Yeoh()
        S = model.pk2_voigt(self.F, YEOH_PARAMS)
        rel_err = abs(float(S[0]) - self.S11_exact) / abs(self.S11_exact)
        assert rel_err < REL_TOL, f"Yeoh S_11 rel error {rel_err:.2e}"

    def test_yeoh_small_strain_s22_zero(self):
        model = Yeoh()
        S = model.pk2_voigt(self.F, YEOH_PARAMS)
        assert abs(float(S[1])) < EPSILON, f"Yeoh S_22 = {S[1]:.2e}"

    def test_yeoh_tangent_symmetry(self):
        model = Yeoh()
        C = model.tangent_voigt(self.F, YEOH_PARAMS)
        diff = float(jnp.abs(C - C.T).max())
        assert diff < 1e-10, f"Yeoh tangent symmetry error {diff:.2e}"

    # --- Arruda-Boyce ---

    def test_ab_strain_energy_zero_at_I(self):
        model = ArrudaBoyce()
        W = model.strain_energy(jnp.eye(2), AB_PARAMS)
        assert abs(float(W)) < 1e-15

    def test_ab_small_strain_s11(self):
        model = ArrudaBoyce()
        S = model.pk2_voigt(self.F, AB_PARAMS)
        rel_err = abs(float(S[0]) - self.S11_exact) / abs(self.S11_exact)
        assert rel_err < REL_TOL, f"ArrudaBoyce S_11 rel error {rel_err:.2e}"

    def test_ab_small_strain_s22_zero(self):
        model = ArrudaBoyce()
        S = model.pk2_voigt(self.F, AB_PARAMS)
        assert abs(float(S[1])) < EPSILON, f"ArrudaBoyce S_22 = {S[1]:.2e}"

    def test_ab_tangent_symmetry(self):
        model = ArrudaBoyce()
        C = model.tangent_voigt(self.F, AB_PARAMS)
        diff = float(jnp.abs(C - C.T).max())
        assert diff < 1e-10, f"ArrudaBoyce tangent symmetry error {diff:.2e}"


# ------------------------------------------------------------------
# Large deformation sanity checks
# ------------------------------------------------------------------

class TestLargeDeformation:
    """Verify monotonic/convex response at large strain."""

    @pytest.fixture(autouse=True)
    def setup(self):
        jax.config.update("jax_enable_x64", True)

    @staticmethod
    def _uniaxial_F(lam):
        """Incompressible plane strain uniaxial: J = F_xx * F_yy * 1 = 1 → F_yy = 1/F_xx."""
        return jnp.array([[lam, 0.0], [0.0, 1.0 / lam]])

    def test_nh_stress_monotonic(self):
        model = NeoHookean()
        stretches = np.linspace(1.0, 2.0, 20)
        S11_vals = [float(model.pk2_voigt(self._uniaxial_F(lam), NH_PARAMS)[0])
                    for lam in stretches]
        diffs = np.diff(S11_vals)
        assert np.all(diffs > 0), "NeoHookean S_11 not monotonic"

    def test_yeoh_stress_monotonic(self):
        model = Yeoh()
        stretches = np.linspace(1.0, 1.5, 10)
        S11_vals = [float(model.pk2_voigt(self._uniaxial_F(lam), YEOH_PARAMS)[0])
                    for lam in stretches]
        diffs = np.diff(S11_vals)
        assert np.all(diffs > 0), "Yeoh S_11 not monotonic"

    def test_ab_stress_monotonic(self):
        """Arruda-Boyce with λ_m=3: stiffening at large stretch."""
        model = ArrudaBoyce()
        # Use λ_m=3 for visible stiffening
        ab_params = {"mu": MU, "lambda_m": 3.0, "K": K_BULK * 10}
        stretches = np.linspace(1.0, 2.0, 15)
        S11_vals = [float(model.pk2_voigt(self._uniaxial_F(lam), ab_params)[0])
                    for lam in stretches]
        diffs = np.diff(S11_vals)
        assert np.all(diffs > 0), "ArrudaBoyce S_11 not monotonic"
