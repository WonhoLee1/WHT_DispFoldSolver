"""
Tests for Abaqus *DEPENDENCIES Table Interpolation (dependence_table.py).

Verifies exact nodal evaluation, midpoint interpolation precision (< 1e-12),
and boundary clamping according to Abaqus theory manual specification.
"""

import numpy as np
import pytest
from dispsolver.material.dependence_table import (
    eval_table_1d_numba,
    eval_table_2d_numba,
)


def test_1d_nodal_exactness():
    """Verify that interpolated values at nodal points match exactly (< 1e-15)."""
    temps = np.array([20.0, 50.0, 100.0, 200.0], dtype=np.float64)
    moduli = np.array([210000.0, 205000.0, 195000.0, 180000.0], dtype=np.float64)

    for t, e in zip(temps, moduli):
        val = eval_table_1d_numba(temps, moduli, t)
        assert abs(val - e) < 1e-14, f"Failed at T={t}: expected {e}, got {val}"


def test_1d_midpoint_interpolation():
    """Verify midpoint linear interpolation precision (< 1e-12)."""
    temps = np.array([0.0, 100.0], dtype=np.float64)
    moduli = np.array([200.0, 100.0], dtype=np.float64)

    # At T=50.0, expected = 150.0
    val_mid = eval_table_1d_numba(temps, moduli, 50.0)
    assert abs(val_mid - 150.0) < 1e-14

    # At T=25.0, expected = 175.0
    val_q1 = eval_table_1d_numba(temps, moduli, 25.0)
    assert abs(val_q1 - 175.0) < 1e-14


def test_1d_boundary_clamping():
    """Verify that points outside the range are clamped to boundary values."""
    temps = np.array([20.0, 100.0], dtype=np.float64)
    moduli = np.array([210000.0, 190000.0], dtype=np.float64)

    # Below lower bound
    val_low = eval_table_1d_numba(temps, moduli, -50.0)
    assert abs(val_low - 210000.0) < 1e-14

    # Above upper bound
    val_high = eval_table_1d_numba(temps, moduli, 500.0)
    assert abs(val_high - 190000.0) < 1e-14


def test_2d_bilinear_interpolation():
    """Verify 2D bilinear interpolation on plastic strain - temperature grid."""
    eps_pl = np.array([0.0, 0.1, 0.5], dtype=np.float64)
    temps = np.array([20.0, 100.0], dtype=np.float64)
    # sigma_y grid: shape (3, 2)
    # T=20: [200, 300, 400]
    # T=100: [150, 250, 350]
    sigma_grid = np.array([
        [200.0, 150.0],
        [300.0, 250.0],
        [400.0, 350.0],
    ], dtype=np.float64)

    # Exact vertex
    val_00 = eval_table_2d_numba(eps_pl, temps, sigma_grid, 0.0, 20.0)
    assert abs(val_00 - 200.0) < 1e-14

    # Center of first cell: eps=0.05, T=60.0 -> expected = (200 + 300 + 150 + 250)/4 = 225.0
    val_mid = eval_table_2d_numba(eps_pl, temps, sigma_grid, 0.05, 60.0)
    assert abs(val_mid - 225.0) < 1e-14

    # Clamp check
    val_clamped = eval_table_2d_numba(eps_pl, temps, sigma_grid, 1.0, 200.0)
    assert abs(val_clamped - 350.0) < 1e-14
