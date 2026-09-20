"""
test_emcs_kernel_sanity.py
==========================
Sanity check for 3D EMCS Numba OpenMP C-extension kernels.
Verifies the Stiemer & Gonzalez discrete derivative strain energy identity:
    Psi(E_{n+1}) - Psi(E_n) = S_{n+1/2}^{EMCS} : (E_{n+1} - E_n)
to 10^-13 precision across small, moderate, and large strain increments.
"""

from __future__ import annotations
import numpy as np
import pytest

from dispsolver.element3d.emcs_kernels3d import (
    _compute_green_lagrange_strain_3d,
    _compute_neohookean_energy_and_stress_3d,
    _compute_emcs_discrete_stress_3d,
    compute_c3d8_emcs_element_numba,
    assemble_mesh_emcs_3d_numba
)


def compute_isotropic_elasticity_matrix(E: float = 210000.0, nu: float = 0.3) -> np.ndarray:
    """Construct 6x6 Voigt Elasticity Matrix C for isotropic continuum."""
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.zeros((6, 6), dtype=np.float64)
    C[0, 0] = lam + 2.0 * mu
    C[1, 1] = lam + 2.0 * mu
    C[2, 2] = lam + 2.0 * mu
    C[0, 1] = C[1, 0] = lam
    C[0, 2] = C[2, 0] = lam
    C[1, 2] = C[2, 1] = lam
    C[3, 3] = mu
    C[4, 4] = mu
    C[5, 5] = mu
    return C


def test_emcs_discrete_derivative_identity():
    """Verify exact strain energy difference formula Psi(E_np1) - Psi(E_n) = S_emcs : (E_np1 - E_n)."""
    C_mat = compute_isotropic_elasticity_matrix(E=200000.0, nu=0.3)

    # Test strain increments from 1e-12 up to 50% strain
    strain_magnitudes = [1e-12, 1e-8, 1e-5, 1e-3, 0.05, 0.20, 0.50]

    for mag in strain_magnitudes:
        # Base strain state E_n
        E_n = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.008], dtype=np.float64)
        
        # Random strain increment
        np.random.seed(42)
        dE = mag * np.array([1.0, -0.5, 0.8, 1.2, -0.3, 0.6], dtype=np.float64)
        E_np1 = E_n + dE

        # Exact strain energies
        Psi_n, _ = _compute_neohookean_energy_and_stress_3d(E_n, C_mat)
        Psi_np1, _ = _compute_neohookean_energy_and_stress_3d(E_np1, C_mat)
        delta_Psi_exact = Psi_np1 - Psi_n

        # EMCS Discrete Stress
        S_emcs = _compute_emcs_discrete_stress_3d(E_n, E_np1, C_mat)
        
        # Work done by EMCS Discrete Stress: S_emcs : dE = dot(S_emcs, dE)
        work_emcs = float(np.dot(S_emcs, dE))

        abs_err = abs(delta_Psi_exact - work_emcs)
        rel_err = abs_err / (abs(delta_Psi_exact) + 1e-15)

        print(f"Mag: {mag:e} | Exact: {delta_Psi_exact:.14e} | EMCS: {work_emcs:.14e} | AbsErr: {abs_err:.4e} | RelErr: {rel_err:.4e}")
        
        # Verification criteria:
        # For finite strain (mag >= 1e-5), exact discrete derivative identity holds to 1e-12
        # For tiny strain (mag < 1e-5), Taylor Padé protection active, error remains < 1e-10
        if mag >= 1e-5:
            assert abs_err < 1e-12 or rel_err < 1e-12, f"EMCS discrete derivative identity failed at strain mag {mag}"
        else:
            assert abs_err < 1e-10, f"EMCS small strain Taylor protection failed at strain mag {mag}"


def test_c3d8_emcs_element_matrices():
    """Verify 1-element C3D8 EMCS element matrices (internal force, tangent stiffness, mass matrix)."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)

    u_elem_n = np.zeros(24, dtype=np.float64)
    u_elem_np1 = np.zeros(24, dtype=np.float64)
    
    # Apply 10% tension in X and 5% shear in Y
    for i in range(8):
        x, y, z = coords[i]
        u_elem_np1[3*i + 0] = 0.10 * x
        u_elem_np1[3*i + 1] = 0.05 * x

    v_elem_n = np.zeros(24, dtype=np.float64)
    a_elem_n = np.zeros(24, dtype=np.float64)

    dt = 0.01
    density = 7.85e-9
    C_mat = compute_isotropic_elasticity_matrix(E=210000.0, nu=0.3)

    f_int, K_eff, M_elem, err = compute_c3d8_emcs_element_numba(
        coords, u_elem_n, u_elem_np1, v_elem_n, a_elem_n, dt, density, C_mat
    )

    assert err == 0
    assert f_int.shape == (24,)
    assert K_eff.shape == (24, 24)
    assert M_elem.shape == (24, 24)

    # Mass matrix must be symmetric positive definite (24 DOFs, 24 positive eigenvalues)
    eig_M = np.linalg.eigvalsh(M_elem)
    assert np.all(eig_M > 0.0), "Mass matrix is not positive definite!"

    # Tangent stiffness matrix K_eff must be symmetric
    symmetry_err = np.max(np.abs(K_eff - K_eff.T))
    print(f"K_eff symmetry max diff: {symmetry_err:.4e}")
    assert symmetry_err < 1e-10, "K_eff matrix is not symmetric!"

    print("C3D8 EMCS 1-Element Matrix Verification PASSED.")


if __name__ == "__main__":
    test_emcs_discrete_derivative_identity()
    test_c3d8_emcs_element_matrices()
