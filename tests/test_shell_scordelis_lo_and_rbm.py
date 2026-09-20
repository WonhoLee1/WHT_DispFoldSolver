"""
test_shell_scordelis_lo_and_rbm.py
===================================
Automated Test Suite for 3D Shell Elements (MITC4+ and MITC3+)
Validates:
  1. Rigid Body Modes (6 zero eigenvalues)
  2. Scordelis-Lo Cylindrical Roof Academic Benchmark (< 2% error for MITC4+)
"""

import pytest
import numpy as np
from dispsolver.element_shell.mitc4_plus_numba import compute_mitc4_plus_stiffness
from dispsolver.element_shell.mitc3_plus_numba import compute_mitc3_plus_stiffness
from benchmark_element.benchmark_shell_mitc import run_scordelis_lo_benchmark


def test_mitc4_plus_rigid_body_modes():
    """Test MITC4+ single element has exactly 6 zero eigenvalues (RBM=6)."""
    E = 2.0e11
    nu = 0.3
    thickness = 0.01
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.2],
        [0.0, 1.0, 0.2]
    ])
    K = compute_mitc4_plus_stiffness(coords, E, nu, thickness)
    eigvals = np.linalg.eigvalsh(K)
    zero_eigs = np.sum(np.abs(eigvals) < 1e-4)
    assert zero_eigs == 6, f"Expected 6 RBMs for MITC4+, got {zero_eigs}"


def test_mitc3_plus_rigid_body_modes():
    """Test MITC3+ single element has exactly 6 zero eigenvalues (RBM=6)."""
    E = 2.0e11
    nu = 0.3
    thickness = 0.01
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.2],
        [0.0, 1.0, -0.1]
    ])
    K = compute_mitc3_plus_stiffness(coords, E, nu, thickness)
    eigvals = np.linalg.eigvalsh(K)
    zero_eigs = np.sum(np.abs(eigvals) < 1e-4)
    assert zero_eigs == 6, f"Expected 6 RBMs for MITC3+, got {zero_eigs}"


def test_scordelis_lo_mitc4_plus_convergence():
    """Test Scordelis-Lo Cylindrical Roof with MITC4+ achieves error < 2.0% on 16x16 mesh."""
    res_16 = run_scordelis_lo_benchmark(elem_type="MITC4+", nx=16, ny=16)
    # Theoretical value: -0.3024 m
    assert res_16["err_percent"] < 2.0, f"MITC4+ 16x16 error {res_16['err_percent']:.2f}% exceeds 2.0%"
    assert abs(res_16["w_computed"] - (-0.3024)) < 0.01
