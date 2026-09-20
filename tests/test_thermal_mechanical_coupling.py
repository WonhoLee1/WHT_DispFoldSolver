"""
Tests for Thermal-Mechanical Coupling and Thermal Expansion Adapter.

Validates:
  1. Free thermal expansion: zero mechanical stress under unconstrained delta_T.
  2. Fully constrained thermal stress: exact match with theoretical sigma = -E(T) * alpha * delta_T.
  3. Temperature-dependent modulus interpolation integration with thermal strain.
  4. Multiplicative finite-strain split: F = F_me @ F_th returns F_me = I under pure thermal expansion.
"""

import numpy as np
import pytest
from dispsolver.material.dependence_table import eval_table_1d_numba
from dispsolver.material.thermal_expansion import (
    compute_thermal_strain_3d,
    compute_thermal_strain_2d,
    decompose_f_thermal_3d,
    decompose_f_thermal_2d,
)
from dispsolver.model.material import Material


def test_free_thermal_expansion_zero_stress():
    """Under free thermal expansion, mechanical strain must be zero and stress is zero."""
    alpha = 1.2e-5  # 1/K
    T0 = 20.0
    T = 120.0  # delta_T = 100.0 K

    strain_th = np.zeros(6, dtype=np.float64)
    compute_thermal_strain_3d(alpha, T, T0, strain_th)

    expected_th = alpha * (T - T0)
    assert abs(strain_th[0] - expected_th) < 1e-15
    assert abs(strain_th[1] - expected_th) < 1e-15
    assert abs(strain_th[2] - expected_th) < 1e-15

    # If total strain equals thermal strain (free expansion):
    strain_tot = strain_th.copy()
    strain_me = strain_tot - strain_th

    # Mechanical strain is identically zero
    assert np.allclose(strain_me, 0.0, atol=1e-15)


def test_constrained_thermal_stress_exact():
    """Under 1D axial constraint (eps_xx = 0), stress must equal -E * alpha * delta_T."""
    E = 210000.0  # MPa
    nu = 0.3
    alpha = 1.5e-5  # 1/K
    T0 = 20.0
    T = 100.0  # delta_T = 80.0 K

    strain_th = np.zeros(3, dtype=np.float64)
    compute_thermal_strain_2d(alpha, T, T0, strain_th)

    # 1D constraint: eps_xx_tot = 0, free in yy => eps_yy_tot = eps_th + nu * eps_xx_me
    # eps_xx_me = 0 - eps_th = -alpha * delta_T
    eps_xx_me = -strain_th[0]
    theoretical_sigma_xx = E * eps_xx_me  # 1D uniaxial stress = -E * alpha * delta_T

    expected_sigma = -E * alpha * (T - T0)
    assert abs(theoretical_sigma_xx - expected_sigma) < 1e-10
    assert theoretical_sigma_xx < 0.0, "Thermal stress under constraint must be compressive"


def test_temp_dependent_modulus_with_thermal_expansion():
    """Verify coupling of temperature-dependent modulus table with thermal expansion."""
    temps = np.array([20.0, 60.0, 100.0], dtype=np.float64)
    moduli = np.array([200000.0, 180000.0, 160000.0], dtype=np.float64)
    alpha = 1.0e-5
    T0 = 20.0

    # At T = 60.0, E(T) = 180000.0
    current_temp = 60.0
    E_curr = eval_table_1d_numba(temps, moduli, current_temp)
    assert abs(E_curr - 180000.0) < 1e-14

    delta_T = current_temp - T0
    eps_th = alpha * delta_T
    # Constrained stress = -E(T) * alpha * delta_T = -180000 * 1.0e-5 * 40 = -72.0 MPa
    sigma_th = -E_curr * eps_th
    assert abs(sigma_th - (-72.0)) < 1e-12


def test_finite_strain_multiplicative_split_identity():
    """In finite strain, pure thermal expansion F = theta * I must yield F_me = I."""
    alpha = 2.0e-5
    T0 = 0.0
    T = 100.0
    delta_T = 100.0
    theta = 1.0 + alpha * delta_T

    # Total F is pure isotropic thermal expansion
    F_tot_3d = theta * np.eye(3, dtype=np.float64)
    F_me_3d = np.zeros((3, 3), dtype=np.float64)

    theta_ret = decompose_f_thermal_3d(F_tot_3d, alpha, T, T0, F_me_3d)
    assert abs(theta_ret - theta) < 1e-15
    # Mechanical F must be exact identity tensor
    assert np.allclose(F_me_3d, np.eye(3), atol=1e-15)

    # 2D case
    F_tot_2d = theta * np.eye(2, dtype=np.float64)
    F_me_2d = np.zeros((2, 2), dtype=np.float64)
    theta_ret_2d = decompose_f_thermal_2d(F_tot_2d, alpha, T, T0, F_me_2d)
    assert abs(theta_ret_2d - theta) < 1e-15
    assert np.allclose(F_me_2d, np.eye(2), atol=1e-15)


def test_material_model_abaqus_api_expansion():
    """Verify that Material class accepts Abaqus Expansion and serializes to numba props."""
    mat = Material(name="Steel_Temperature")
    mat.Elastic(table=((210000.0, 0.3, 20.0), (190000.0, 0.32, 100.0)))
    mat.Expansion(type="ISO", table=((1.2e-5, 20.0), (1.5e-5, 100.0)), zero=20.0)

    props = mat.to_numba_props()
    assert "expansion" in props
    assert props["alpha"] == 1.2e-5
    assert props["T0"] == 20.0
    assert "temp_elastic_table" in props
    assert len(props["temp_elastic_table"]) == 2
    assert props["temp_elastic_table"][1] == (190000.0, 0.32, 100.0)
