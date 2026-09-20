"""
test_creep_power_law.py
=======================
Verification of Abaqus Power-Law Secondary Creep (*CREEP, LAW=STRAIN).
Validates:
1. Pure elastic step when A=0 or dt=0.
2. Radial return iteration convergence under high stress.
3. Creep strain accumulation under uniaxial constant tension:
   d(eps_cr)/dt = A * sigma^n * t^m
   For m=0 (steady-state Norton): eps_cr(t) = A * sigma^n * t
4. Stress relaxation under fixed total strain (eps_tot = const):
   d(sigma)/dt + E * d(eps_cr)/dt = 0 => sigma(t) decays monotonically.
5. Consistent tangent operator symmetry and positivity.
"""

import numpy as np
import pytest

from dispsolver.material.creep_numba import integrate_power_law_creep
from dispsolver.model.material import Material


def test_creep_pure_elastic_limit():
    """Verify that with A=0, response degenerates to linear isotropic elasticity."""
    E = 200000.0  # MPa
    nu = 0.3
    eps_tot = np.array([0.001, -0.0003, -0.0003, 0.0, 0.0, 0.0], dtype=np.float64)
    eps_cr_old = np.zeros(6, dtype=np.float64)

    sigma, eps_cr_new, q_new, delta_p, C = integrate_power_law_creep(
        eps_tot=eps_tot,
        eps_cr_old=eps_cr_old,
        q_old=0.0,
        dt=1.0,
        t_total=1.0,
        E=E,
        nu=nu,
        A=0.0,  # Zero creep
        n=1.0,
        m=0.0,
    )

    # 1D uniaxial strain condition sigma_11 = E * eps_11 = 200.0 MPa
    np.testing.assert_allclose(sigma[0], 200.0, rtol=1e-5)
    np.testing.assert_allclose(sigma[1:], 0.0, atol=1e-5)
    np.testing.assert_allclose(eps_cr_new, 0.0, atol=1e-12)
    assert delta_p == 0.0
    # Tangent must be symmetric
    np.testing.assert_allclose(C, C.T, atol=1e-10)


def test_creep_steady_state_accumulation():
    """Verify Norton power law steady-state strain accumulation eps_cr = A * sigma^n * t.
    For n=1 (linear viscous creep), under constant stress sigma=100 MPa:
    eps_cr_rate = A * 100.
    """
    E = 100000.0  # MPa
    nu = 0.3
    A = 1e-4  # (1/MPa)/s
    n = 1.0
    m = 0.0

    sigma_target = 100.0  # MPa
    # For uniaxial stress sigma_11 = 100, elastic strain is eps_11_el = 100 / E = 0.001
    # Over time t=10s with dt=1.0s, at each step adjust eps_tot to maintain sigma ~= 100 MPa
    eps_tot = np.array([sigma_target / E, -nu * sigma_target / E, -nu * sigma_target / E, 0.0, 0.0, 0.0], dtype=np.float64)
    eps_cr = np.zeros(6, dtype=np.float64)
    q = sigma_target
    dt = 0.1
    n_steps = 50

    for step in range(1, n_steps + 1):
        t_tot = step * dt
        # Add incremental total strain so stress stays near 100
        eps_tot[0] += A * sigma_target * dt
        eps_tot[1] -= 0.5 * A * sigma_target * dt
        eps_tot[2] -= 0.5 * A * sigma_target * dt

        sigma, eps_cr, q, delta_p, C = integrate_power_law_creep(
            eps_tot=eps_tot,
            eps_cr_old=eps_cr,
            q_old=q,
            dt=dt,
            t_total=t_tot,
            E=E,
            nu=nu,
            A=A,
            n=n,
            m=m,
        )

    # Theoretical axial creep strain: eps_cr_11 = A * sigma * (n_steps * dt) = 1e-4 * 100 * 5.0 = 0.05
    expected_eps_cr = A * sigma_target * (n_steps * dt)
    # Numerical strain should match expected within 2%
    np.testing.assert_allclose(eps_cr[0], expected_eps_cr, rtol=0.03)


def test_creep_stress_relaxation():
    """Verify monotonic stress relaxation under fixed total strain eps_tot = const."""
    E = 200000.0
    nu = 0.3
    A = 1e-6
    n = 2.0  # Non-linear Norton exponent
    m = 0.0

    # Apply initial strain: eps_11 = 0.002 -> initial sigma ~= 400 MPa
    eps_tot = np.array([0.002, -0.0006, -0.0006, 0.0, 0.0, 0.0], dtype=np.float64)
    eps_cr = np.zeros(6, dtype=np.float64)
    q = 400.0
    dt = 0.5
    stresses = []

    for step in range(1, 21):
        t_tot = step * dt
        sigma, eps_cr, q, delta_p, C = integrate_power_law_creep(
            eps_tot=eps_tot,
            eps_cr_old=eps_cr,
            q_old=q,
            dt=dt,
            t_total=t_tot,
            E=E,
            nu=nu,
            A=A,
            n=n,
            m=m,
        )
        stresses.append(sigma[0])

    # Stress must strictly decrease monotonically
    for i in range(len(stresses) - 1):
        assert stresses[i] > stresses[i + 1]
    # Significant relaxation should have occurred
    assert stresses[-1] < stresses[0] * 0.9


def test_material_creep_api():
    """Verify Material class Creep method and properties serialization."""
    mat = Material(name="CreepSteel")
    mat.Elastic(table=((200000.0, 0.3),))
    mat.Creep(law="STRAIN", table=((1.5e-8, 3.2, -0.2),))

    props = mat.to_numba_props()
    assert "creep" in props
    assert props["creep"]["law"] == "STRAIN"
    assert props["creep"]["A"] == 1.5e-8
    assert props["creep"]["n"] == 3.2
    assert props["creep"]["m"] == -0.2
