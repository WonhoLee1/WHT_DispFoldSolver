"""
test_plastic_hardening.py
=========================
Verification suite for advanced plasticity models:
1. Bauschinger effect & reverse yielding under linear kinematic hardening:
   sigma_rev = sigma_fwd - 2 * sigma_y0
2. Strict degeneracy of Johnson-Cook model to quasi-static isothermal hardening:
   T = T_room, eps_dot <= eps_dot_0 ==> sigma_y == A + B * (eq_p)^n
3. Cowper-Symonds rate-dependent dynamic yield scaling:
   eps_dot = 0 ==> sigma_y == sigma_0; eps_dot = D ==> sigma_y == 2 * sigma_0
"""

import numpy as np
import pytest

from dispsolver.material.plastic_hardening import (
    return_mapping_linear_kinematic,
    evaluate_johnson_cook_yield,
    evaluate_cowper_symonds_yield,
    compute_von_mises_equivalent,
)


def test_linear_kinematic_hardening_bauschinger_effect():
    """Verify Bauschinger effect in linear kinematic hardening.
    After forward tensile yielding to sigma_fwd, reversing load to compression
    must yield at sigma_rev = sigma_fwd - 2 * sigma_y0.
    """
    E = 200000.0  # MPa
    nu = 0.3
    sigma_y0 = 250.0  # MPa
    H_iso = 0.0  # Pure kinematic hardening
    C_kin = 20000.0  # MPa

    alpha = np.zeros(6, dtype=np.float64)
    ep = np.zeros(6, dtype=np.float64)
    eq_p = 0.0

    # 1. Forward uniaxial tension to total strain e11 = 0.015 (yield strain is 250 / 200000 = 0.00125)
    e_forward = np.array([0.015, -nu * 0.015, -nu * 0.015, 0.0, 0.0, 0.0], dtype=np.float64)
    S_fwd, alpha_fwd, ep_fwd, eq_p_fwd, C_ep = return_mapping_linear_kinematic(
        e_forward, alpha, ep, eq_p, E, nu, sigma_y0, H_iso, C_kin
    )

    sigma_fwd = S_fwd[0]
    assert sigma_fwd > sigma_y0, f"Expected tensile yielding, got sigma_fwd={sigma_fwd}"
    assert np.linalg.norm(alpha_fwd) > 0.0, "Backstress should have evolved"

    # Theoretical forward stress:
    # d_gamma = (sigma_trial - sigma_y0) / (3*mu + C_kin) in 1D equivalent
    # Forward peak stress is reached with backstress alpha_11 > 0.
    # Theoretical reverse yield point under pure kinematic hardening:
    # Elastic unloading from sigma_fwd has an elastic range of exactly 2 * sigma_y0.
    # Therefore, yielding in compression occurs at sigma_rev = sigma_fwd - 2 * sigma_y0.
    expected_sigma_rev = sigma_fwd - 2.0 * sigma_y0

    # 2. Test elastic unloading state just before reverse yield
    # Unload by delta_sigma = 1.99 * sigma_y0 (still elastic)
    delta_e_elastic = -1.99 * sigma_y0 / E
    e_test_elastic = e_forward.copy()
    e_test_elastic[0] += delta_e_elastic
    e_test_elastic[1] -= nu * delta_e_elastic
    e_test_elastic[2] -= nu * delta_e_elastic

    S_el, alpha_el, ep_el, eq_p_el, _ = return_mapping_linear_kinematic(
        e_test_elastic, alpha_fwd, ep_fwd, eq_p_fwd, E, nu, sigma_y0, H_iso, C_kin
    )
    # Plastic strain and backstress must remain unchanged (purely elastic unload)
    assert eq_p_el == eq_p_fwd
    assert np.allclose(alpha_el, alpha_fwd)

    # 3. Test reverse yielding beyond 2 * sigma_y0
    # Unload by delta_sigma = 2.5 * sigma_y0 (definitely into compression yield)
    delta_e_plastic = -2.5 * sigma_y0 / E
    e_test_plastic = e_forward.copy()
    e_test_plastic[0] += delta_e_plastic
    e_test_plastic[1] -= nu * delta_e_plastic
    e_test_plastic[2] -= nu * delta_e_plastic

    S_rev, alpha_rev, ep_rev, eq_p_rev, _ = return_mapping_linear_kinematic(
        e_test_plastic, alpha_fwd, ep_fwd, eq_p_fwd, E, nu, sigma_y0, H_iso, C_kin
    )
    # Plastic strain must have increased
    assert eq_p_rev > eq_p_fwd, "Reverse yielding should accumulate plastic strain"
    # Backstress in 11-direction should have decreased (shifted toward negative)
    assert alpha_rev[0] < alpha_fwd[0]


def test_johnson_cook_strict_degeneracy():
    """Verify strict degeneracy of Johnson-Cook model at reference state:
    T = T_room and eps_dot <= eps_dot_0 ==> sigma_y == A + B * (eq_p)^n (error < 1e-12).
    """
    A = 350.0  # MPa
    B = 400.0  # MPa
    n = 0.35
    C_rate = 0.02
    m_temp = 1.1
    T_room = 293.15  # 20 degC (K)
    T_melt = 1793.15  # 1520 degC (K)
    eps_dot_0 = 1.0  # 1/s

    # Test array of equivalent plastic strains
    eq_p_vals = [0.0, 0.005, 0.02, 0.1, 0.5]
    for eq_p in eq_p_vals:
        # At reference condition (T = T_room, eps_dot <= eps_dot_0)
        sigma_jc = evaluate_johnson_cook_yield(
            A, B, n, C_rate, m_temp, T_room, T_melt, eps_dot_0,
            eq_p=eq_p, eps_dot=0.1, T=T_room
        )
        expected = A if eq_p == 0.0 else A + B * (eq_p ** n)
        assert abs(sigma_jc - expected) < 1e-12, f"Degeneracy failed: expected {expected}, got {sigma_jc}"


def test_johnson_cook_rate_and_temperature_effects():
    """Verify strain rate hardening and thermal softening in Johnson-Cook model."""
    A = 350.0
    B = 400.0
    n = 0.35
    C_rate = 0.05
    m_temp = 1.0
    T_room = 300.0
    T_melt = 1500.0
    eps_dot_0 = 1.0
    eq_p = 0.1

    sigma_ref = evaluate_johnson_cook_yield(
        A, B, n, C_rate, m_temp, T_room, T_melt, eps_dot_0,
        eq_p=eq_p, eps_dot=1.0, T=300.0
    )

    # 1. High strain rate (eps_dot = 1000.0 /s, ln(1000) = 6.90775)
    sigma_high_rate = evaluate_johnson_cook_yield(
        A, B, n, C_rate, m_temp, T_room, T_melt, eps_dot_0,
        eq_p=eq_p, eps_dot=1000.0, T=300.0
    )
    expected_rate_factor = 1.0 + C_rate * np.log(1000.0)
    assert abs(sigma_high_rate - sigma_ref * expected_rate_factor) < 1e-10
    assert sigma_high_rate > sigma_ref

    # 2. Elevated temperature (T = 900 K, halfway to melt: T* = (900-300)/(1500-300) = 0.5)
    sigma_hot = evaluate_johnson_cook_yield(
        A, B, n, C_rate, m_temp, T_room, T_melt, eps_dot_0,
        eq_p=eq_p, eps_dot=1.0, T=900.0
    )
    assert abs(sigma_hot - sigma_ref * 0.5) < 1e-10
    assert sigma_hot < sigma_ref

    # 3. Complete melting (T >= T_melt ==> sigma_y == 0.0)
    sigma_melt = evaluate_johnson_cook_yield(
        A, B, n, C_rate, m_temp, T_room, T_melt, eps_dot_0,
        eq_p=eq_p, eps_dot=1.0, T=1500.0
    )
    assert sigma_melt == 0.0


def test_cowper_symonds_rate_scaling():
    """Verify Cowper-Symonds rate scaling characteristics:
    1. eps_dot = 0 ==> sigma_y == sigma_0
    2. eps_dot = D ==> sigma_y == 2 * sigma_0
    3. Monotonic increase with eps_dot
    """
    sigma_0 = 300.0  # MPa
    D = 40.0  # 1/s (typical for mild steel)
    p = 5.0

    # 1. Static limit
    sigma_static = evaluate_cowper_symonds_yield(sigma_0, D, p, eps_dot=0.0)
    assert abs(sigma_static - sigma_0) < 1e-12

    # 2. Characteristic rate eps_dot = D ==> factor is 1 + (D/D)^(1/p) = 2.0
    sigma_char = evaluate_cowper_symonds_yield(sigma_0, D, p, eps_dot=D)
    assert abs(sigma_char - 2.0 * sigma_0) < 1e-12

    # 3. Higher rate monotonic test
    sigma_high = evaluate_cowper_symonds_yield(sigma_0, D, p, eps_dot=1000.0)
    assert sigma_high > sigma_char


def test_material_cae_advanced_plasticity_interface():
    """Verify Material class setup for Johnson-Cook, Cowper-Symonds, and Kinematic Hardening."""
    from dispsolver.model.material import Material

    # 1. Johnson-Cook material
    mat_jc = Material(name="Ti6Al4V_JC")
    mat_jc.Elastic(E=110000.0, nu=0.34)
    mat_jc.JohnsonCook(A=862.0, B=331.0, n=0.34, C=0.012, m=0.8, T_room=293.15, T_melt=1878.15)
    props_jc = mat_jc.to_numba_props()
    assert "johnson_cook" in props_jc
    assert props_jc["johnson_cook"]["A"] == 862.0
    assert props_jc["johnson_cook"]["m"] == 0.8

    # 2. Cowper-Symonds rate dependent material
    mat_cs = Material(name="Steel_CS")
    mat_cs.Elastic(E=210000.0, nu=0.3)
    mat_cs.Plastic(table=((300.0, 0.0), (450.0, 0.2)))
    mat_cs.RateDependent(type="COWPER_SYMONDS", table=((40.0, 5.0),))
    props_cs = mat_cs.to_numba_props()
    assert "rate_dependent" in props_cs
    assert props_cs["rate_dependent"]["D"] == 40.0
    assert props_cs["rate_dependent"]["p"] == 5.0

    # 3. Kinematic hardening material
    mat_kin = Material(name="Alu_Kinematic")
    mat_kin.Elastic(E=70000.0, nu=0.33)
    mat_kin.Plastic(table=((150.0, 0.0), (250.0, 0.05)), hardening="KINEMATIC")
    props_kin = mat_kin.to_numba_props()
    assert props_kin.get("hardening_type") == "KINEMATIC"

