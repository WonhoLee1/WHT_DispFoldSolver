"""
test_mullins_and_bergstrom_boyce.py
===================================
Verification suite for flexible display adhesive / elastomer advanced models:
1. Mullins Effect (Ogden-Roxburgh continuous damage mechanics):
   - Virgin loading: eta == 1.0 (no damage, exact base hyperelastic curve)
   - Unloading: eta < 1.0 (damage softening & energy dissipation)
   - Reloading: restores exactly to virgin curve once U >= U_max
2. Bergstrom-Boyce 2-Network Model:
   - Instantaneous response: carries combined Network A + Network B stiffness
   - Long-term equilibrium: Network B relaxes, asymptotically converging to Network A
"""

import numpy as np
import pytest

from dispsolver.material.mullins_damage import (
    evaluate_mullins_damage_factor,
    apply_mullins_damage_to_pk2,
)
from dispsolver.material.bergstrom_boyce import (
    arruda_boyce_pk2_and_tangent,
    evaluate_bergstrom_boyce,
)


def test_mullins_virgin_loading_no_damage():
    """Verify that virgin loading experiences zero damage (eta == 1.0)."""
    r = 2.5
    m = 0.5
    beta = 0.1

    U_dev = 1.2  # MPa (strain energy)
    U_max = 1.2  # Virgin state (current energy equals max)

    eta, d_eta = evaluate_mullins_damage_factor(U_dev, U_max, r, m, beta)
    assert eta == 1.0, f"Virgin loading should have eta=1.0, got {eta}"
    assert d_eta == 0.0


def test_mullins_unloading_damage_softening():
    """Verify that unloading from historical maximum softens the stress response (eta < 1.0)."""
    r = 2.0
    m = 0.2
    beta = 0.05

    U_max = 2.0  # Previously reached maximum
    U_dev = 0.5  # Unloaded state

    eta, d_eta = evaluate_mullins_damage_factor(U_dev, U_max, r, m, beta)
    assert eta < 1.0, f"Unloading must soften (eta < 1.0), got {eta}"
    assert eta > 0.0, f"Damage factor must remain positive, got {eta}"
    assert d_eta > 0.0, f"d_eta/dU must be positive during softening, got {d_eta}"

    # Verify stress scaling
    S0 = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    C0 = np.eye(6, dtype=np.float64) * 100.0

    S_mullins, C_mullins, U_max_new = apply_mullins_damage_to_pk2(
        S0, C0, U_dev, U_max, r, m, beta
    )
    # Deviatoric stress must be reduced by eta
    assert S_mullins[0] < S0[0], "Damaged stress must be lower than undamaged stress"
    assert U_max_new == U_max, "Historical maximum should not change during unloading"


def test_mullins_reloading_restores_at_umax():
    """Verify that reloading up to and beyond U_max restores the virgin response."""
    r = 2.5
    m = 0.3
    beta = 0.1

    U_max_old = 1.5
    # Reloading to a new peak U_dev = 2.0 > U_max_old
    U_dev_new = 2.0

    eta, d_eta = evaluate_mullins_damage_factor(U_dev_new, U_dev_new, r, m, beta)
    assert eta == 1.0, "Reloading beyond historical maximum must have eta=1.0"


def test_bergstrom_boyce_equilibrium_relaxation():
    """Verify Bergstrom-Boyce instantaneous stiffening and long-term relaxation to Network A."""
    mu_A = 0.2  # MPa
    lambda_m = 3.0
    K = 10.0
    s_ratio = 1.5  # Network B is 1.5x stiffer than Network A
    gamma_dot_0 = 0.5
    C_exp = 0.0
    m_exp = 1.0  # Linear viscous limit for test
    tau_hat = 0.05

    # 10% uniaxial stretch: lambda_1 = 1.1, lambda_2 = lambda_3 = 1 / sqrt(1.1)
    lam1 = 1.1
    lam2 = 1.0 / np.sqrt(lam1)
    C_rc = np.diag([lam1**2, lam2**2, lam2**2]).astype(np.float64)

    # 1. Pure equilibrium Network A stress
    S_A, _, _ = arruda_boyce_pk2_and_tangent(C_rc, mu_A, lambda_m, K)

    # 2. Instantaneous step (dt=0): Network A + Network B active
    S_B_init = np.zeros(6, dtype=np.float64)
    S_inst, _, S_B_step1 = evaluate_bergstrom_boyce(
        C_rc, S_B_init, dt=0.0,
        mu_A=mu_A, lambda_m=lambda_m, K=K, s_stiffness_ratio=s_ratio,
        gamma_dot_0=gamma_dot_0, C_exp=C_exp, m_exp=m_exp, tau_hat=tau_hat
    )
    assert S_inst[0] > S_A[0], "Instantaneous stress must exceed equilibrium stress"

    # 3. Relaxation holding deformation constant over multiple time steps (dt = 5.0 s each)
    S_B_cur = S_B_step1.copy()
    for _ in range(20):
        S_rel, _, S_B_cur = evaluate_bergstrom_boyce(
            C_rc, S_B_cur, dt=5.0,
            mu_A=mu_A, lambda_m=lambda_m, K=K, s_stiffness_ratio=s_ratio,
            gamma_dot_0=gamma_dot_0, C_exp=C_exp, m_exp=m_exp, tau_hat=tau_hat
        )

    # Long term relaxed stress should be much closer to equilibrium Network A stress
    diff_inst = abs(S_inst[0] - S_A[0])
    diff_relaxed = abs(S_rel[0] - S_A[0])
    assert diff_relaxed < 0.2 * diff_inst, f"Stress relaxation failed: diff_relaxed={diff_relaxed}, diff_inst={diff_inst}"


def test_material_cae_mullins_and_bb_interface():
    """Verify Material class definitions for Mullins effect and Bergstrom-Boyce model."""
    from dispsolver.model.material import Material

    # 1. Hyperelastic with Mullins damage
    mat_mullins = Material(name="Rubber_Mullins")
    mat_mullins.Hyperelastic(model="NEO_HOOKEAN", table=((0.5, 0.01),))
    mat_mullins.MullinsEffect(r=2.5, m=0.3, beta=0.1)
    props_m = mat_mullins.to_numba_props()
    assert "mullins" in props_m
    assert props_m["mullins"]["r"] == 2.5

    # 2. Bergstrom-Boyce 2-network model
    mat_bb = Material(name="PSA_BB")
    mat_bb.BergstromBoyce(mu_A=0.15, lambda_m=3.5, K=8.0, s=1.8, gamma_dot_0=0.05, tau_hat=0.4)
    props_bb = mat_bb.to_numba_props()
    assert "bergstrom_boyce" in props_bb
    assert props_bb["bergstrom_boyce"]["s"] == 1.8

