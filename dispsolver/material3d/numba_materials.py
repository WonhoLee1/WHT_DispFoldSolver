"""
numba_materials.py
==================
Data-Oriented Material Dispatcher for 3D Non-Linear Numba Assembly.
Provides a unified interface decoupling kinematics from constitutive laws.
"""

import numpy as np
from numba import njit

# Material Type Constants
MAT_LINEAR_ELASTIC = 0
MAT_HYPERELASTIC_NEOHOOKEAN = 1
MAT_NEO_HOOKEAN = MAT_HYPERELASTIC_NEOHOOKEAN
MAT_J2_PLASTICITY = 2
MAT_VISCOELASTIC_PRONY = 3
MAT_HYPERELASTIC_ARRUDA_BOYCE = 4
MAT_ARRUDA_BOYCE = MAT_HYPERELASTIC_ARRUDA_BOYCE
MAT_HYPERELASTIC_YEOH = 5
MAT_YEOH = MAT_HYPERELASTIC_YEOH
MAT_HYPER_VISCOELASTIC = 6
MAT_J2_PLASTICITY_NONLINEAR = 7
MAT_HYPERELASTIC_MOONEY_RIVLIN = 8
MAT_MOONEY_RIVLIN = MAT_HYPERELASTIC_MOONEY_RIVLIN
MAT_HYPERELASTIC_OGDEN = 9
MAT_OGDEN = MAT_HYPERELASTIC_OGDEN
MAT_HYPERELASTIC_POLYNOMIAL = 10
MAT_POLYNOMIAL = MAT_HYPERELASTIC_POLYNOMIAL
MAT_HYPERELASTIC_MARLOW = 11
MAT_MARLOW = MAT_HYPERELASTIC_MARLOW
MAT_J2_PLASTICITY_TABULAR = 12
MAT_RADIOSS_LAW36 = MAT_J2_PLASTICITY_TABULAR
MAT_J2_DUCTILE_DAMAGE = 13
MAT_DRUCKER_PRAGER = 14
MAT_CONCRETE_DP = 15
MAT_ORTHOTROPIC_CYLINDRICAL = 17
MAT_CUSTOM_ELASTIC = 99

@njit(fastmath=True)
def get_default_sdv_count(mat_type: int) -> int:
    """Return the number of state-dependent variables (SDVs) needed per integration point."""
    if mat_type == MAT_LINEAR_ELASTIC:
        return 0
    elif mat_type == MAT_HYPERELASTIC_NEOHOOKEAN:
        return 0
    elif mat_type == MAT_HYPERELASTIC_ARRUDA_BOYCE:
        return 0
    elif mat_type == MAT_HYPERELASTIC_YEOH:
        return 0
    elif mat_type == MAT_HYPERELASTIC_MOONEY_RIVLIN:
        return 0
    elif mat_type == MAT_HYPERELASTIC_OGDEN:
        return 0
    elif mat_type == MAT_HYPERELASTIC_POLYNOMIAL:
        return 0
    elif mat_type == MAT_HYPERELASTIC_MARLOW:
        return 0
    elif mat_type == MAT_J2_PLASTICITY:
        return 7  # eq_plastic_strain (1), plastic_strain_tensor (6)
    elif mat_type == MAT_J2_PLASTICITY_NONLINEAR:
        return 7  # eq_plastic_strain (1), plastic_strain_tensor (6)
    elif mat_type == MAT_J2_PLASTICITY_TABULAR:
        return 8  # plastic_strain_tensor (6), eq_plastic_strain (1), is_failed flag (1)
    elif mat_type == MAT_J2_DUCTILE_DAMAGE:
        return 10  # ep tensor (6), eq_p (1), omega_D (1), u_bar (1), D (1)
    elif mat_type == MAT_DRUCKER_PRAGER:
        return 7  # ep tensor (6), eq_p (1)
    elif mat_type == MAT_CONCRETE_DP:
        return 10  # ep (6), ec_in (1), et_in (1), dt (1), dc (1)
    elif mat_type == MAT_VISCOELASTIC_PRONY:
        return 12  # viscous history tensor h_1 (6) + prev elastic deviatoric stress (6)
    elif mat_type == MAT_HYPER_VISCOELASTIC:
        # Up to 6 Prony terms: 6 * 6 = 36 overstress components + 6 previous elastic deviatoric stress = 42
        return 42
    elif mat_type == MAT_ORTHOTROPIC_CYLINDRICAL:
        return 0
    elif mat_type == MAT_CUSTOM_ELASTIC:
        return 0
    return 0


@njit(fastmath=True)
def _inv_and_det_c(C: np.ndarray):
    """Compute determinant, inverse, and trace of 3x3 symmetric tensor C."""
    c00 = C[0, 0]; c01 = C[0, 1]; c02 = C[0, 2]
    c10 = C[1, 0]; c11 = C[1, 1]; c12 = C[1, 2]
    c20 = C[2, 0]; c21 = C[2, 1]; c22 = C[2, 2]

    cof00 = c11 * c22 - c12 * c21
    cof01 = -(c10 * c22 - c12 * c20)
    cof02 = c10 * c21 - c11 * c20
    detC = c00 * cof00 + c01 * cof01 + c02 * cof02
    if detC < 1e-12:
        detC = 1e-12
    J = np.sqrt(detC)

    inv_detC = 1.0 / detC
    Cinv = np.empty((3, 3), dtype=np.float64)
    Cinv[0, 0] = cof00 * inv_detC
    Cinv[0, 1] = -(c01 * c22 - c02 * c21) * inv_detC
    Cinv[0, 2] = (c01 * c12 - c02 * c11) * inv_detC
    Cinv[1, 0] = cof01 * inv_detC
    Cinv[1, 1] = (c00 * c22 - c02 * c20) * inv_detC
    Cinv[1, 2] = -(c00 * c12 - c02 * c10) * inv_detC
    Cinv[2, 0] = cof02 * inv_detC
    Cinv[2, 1] = -(c00 * c21 - c01 * c20) * inv_detC
    Cinv[2, 2] = (c00 * c11 - c01 * c10) * inv_detC

    trC = c00 + c11 + c22
    return J, detC, Cinv, trC


@njit(fastmath=True)
def _neohookean_pk2_voigt(E_voigt: np.ndarray, mu: float, K: float) -> np.ndarray:
    """PK2 stress for genuine compressible Neo-Hookean:
        W(C) = (mu/2)*(J^(-2/3)*tr(C) - 3) + (K/2)*(J-1)^2
    """
    C = np.empty((3, 3), dtype=np.float64)
    C[0, 0] = 1.0 + 2.0 * E_voigt[0]
    C[1, 1] = 1.0 + 2.0 * E_voigt[1]
    C[2, 2] = 1.0 + 2.0 * E_voigt[2]
    C[0, 1] = C[1, 0] = E_voigt[3]
    C[1, 2] = C[2, 1] = E_voigt[4]
    C[2, 0] = C[0, 2] = E_voigt[5]

    J, detC, Cinv, trC = _inv_and_det_c(C)

    Jm23 = J ** (-2.0 / 3.0)
    vol_scale = K * J * (J - 1.0)

    S = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            ident = 1.0 if i == j else 0.0
            S[i, j] = mu * Jm23 * (ident - (trC / 3.0) * Cinv[i, j]) + vol_scale * Cinv[i, j]

    S_voigt = np.empty(6, dtype=np.float64)
    S_voigt[0] = S[0, 0]
    S_voigt[1] = S[1, 1]
    S_voigt[2] = S[2, 2]
    S_voigt[3] = S[0, 1]
    S_voigt[4] = S[1, 2]
    S_voigt[5] = S[2, 0]
    return S_voigt


@njit(fastmath=True)
def _arruda_boyce_pk2_voigt(E_voigt: np.ndarray, mu: float, lambda_m: float, K: float) -> np.ndarray:
    """PK2 stress for genuine Arruda-Boyce (8-chain) model:
        W_dev = mu * sum_{k=1}^5 [ C_k / lambda_m^{2(k-1)} * (I_bar_1^k - 3^k) ]
        W_vol = (K/2) * (ln J)^2

    Stress:
        S = 2*dW/dC = mu * J^{-2/3} * phi * (I - (I1/3)*Cinv) + K * ln(J) * Cinv
        where phi = sum_{k=1}^5 [ k * C_k / lambda_m^{2(k-1)} * I_bar_1^{k-1} ]

    Reference:
        Arruda, E.M. & Boyce, M.C. (1993). J. Mech. Phys. Solids, 41(2), 389-412.
    """
    C = np.empty((3, 3), dtype=np.float64)
    C[0, 0] = 1.0 + 2.0 * E_voigt[0]
    C[1, 1] = 1.0 + 2.0 * E_voigt[1]
    C[2, 2] = 1.0 + 2.0 * E_voigt[2]
    C[0, 1] = C[1, 0] = E_voigt[3]
    C[1, 2] = C[2, 1] = E_voigt[4]
    C[2, 0] = C[0, 2] = E_voigt[5]

    J, detC, Cinv, trC = _inv_and_det_c(C)

    Jm23 = J ** (-2.0 / 3.0)
    I1_bar = Jm23 * trC

    # Arruda-Boyce coefficients C_1..C_5
    c1 = 0.5
    c2 = 1.0 / 20.0
    c3 = 11.0 / 1050.0
    c4 = 19.0 / 7000.0
    c5 = 519.0 / 673750.0

    lm2 = lambda_m * lambda_m
    lm4 = lm2 * lm2
    lm6 = lm4 * lm2
    lm8 = lm6 * lm2

    # phi = dW_dev / dI1_bar / mu
    # k=1: 1 * c1
    # k=2: 2 * c2 / lm2 * I1_bar
    # k=3: 3 * c3 / lm4 * I1_bar^2
    # k=4: 4 * c4 / lm6 * I1_bar^3
    # k=5: 5 * c5 / lm8 * I1_bar^4
    phi = (
        c1
        + (2.0 * c2 / lm2) * I1_bar
        + (3.0 * c3 / lm4) * (I1_bar * I1_bar)
        + (4.0 * c4 / lm6) * (I1_bar * I1_bar * I1_bar)
        + (5.0 * c5 / lm8) * (I1_bar * I1_bar * I1_bar * I1_bar)
    )

    lnJ = np.log(max(J, 1e-12))
    vol_scale = K * lnJ

    S = np.empty((3, 3), dtype=np.float64)
    dev_factor = mu * Jm23 * phi
    for i in range(3):
        for j in range(3):
            ident = 1.0 if i == j else 0.0
            S[i, j] = dev_factor * (ident - (trC / 3.0) * Cinv[i, j]) + vol_scale * Cinv[i, j]

    S_voigt = np.empty(6, dtype=np.float64)
    S_voigt[0] = S[0, 0]
    S_voigt[1] = S[1, 1]
    S_voigt[2] = S[2, 2]
    S_voigt[3] = S[0, 1]
    S_voigt[4] = S[1, 2]
    S_voigt[5] = S[2, 0]
    return S_voigt


@njit(fastmath=True)
def _yeoh_pk2_voigt(
    E_voigt: np.ndarray,
    c10: float, c20: float, c30: float,
    d1: float, d2: float, d3: float
) -> np.ndarray:
    """PK2 stress for genuine Yeoh 3-term polynomial model:
        W_dev = c10*(I_bar_1 - 3) + c20*(I_bar_1 - 3)^2 + c30*(I_bar_1 - 3)^3
        W_vol = (1/d1)*(J-1)^2 + (1/d2)*(J-1)^4 + (1/d3)*(J-1)^6

    Stress:
        S = 2*dW/dC = 2 * (dW_dev/dI_bar_1) * (dI_bar_1/dC) + (dW_vol/dJ) * (dJ/dC)
        dW_vol/dJ = 2*(J-1)/d1 + 4*(J-1)^3/d2 + 6*(J-1)^5/d3
        dJ/dC = 0.5 * J * Cinv
        => S_vol = J * (dW_vol/dJ) * Cinv

    Reference:
        Yeoh, O.H. (1990). Rubber Chemistry and Technology, 63(5), 792-805.
    """
    C = np.empty((3, 3), dtype=np.float64)
    C[0, 0] = 1.0 + 2.0 * E_voigt[0]
    C[1, 1] = 1.0 + 2.0 * E_voigt[1]
    C[2, 2] = 1.0 + 2.0 * E_voigt[2]
    C[0, 1] = C[1, 0] = E_voigt[3]
    C[1, 2] = C[2, 1] = E_voigt[4]
    C[2, 0] = C[0, 2] = E_voigt[5]

    J, detC, Cinv, trC = _inv_and_det_c(C)

    Jm23 = J ** (-2.0 / 3.0)
    I1_bar = Jm23 * trC
    Ibar_m3 = I1_bar - 3.0

    # dW_dev / dI_bar_1
    psi_dev = c10 + 2.0 * c20 * Ibar_m3 + 3.0 * c30 * (Ibar_m3 * Ibar_m3)

    # Volumetric derivative dW_vol / dJ
    Jm1 = J - 1.0
    dW_vol_dJ = 0.0
    if abs(d1) > 1e-15:
        dW_vol_dJ += (2.0 * Jm1) / d1
    if abs(d2) > 1e-15:
        dW_vol_dJ += (4.0 * (Jm1 ** 3)) / d2
    if abs(d3) > 1e-15:
        dW_vol_dJ += (6.0 * (Jm1 ** 5)) / d3

    vol_scale = J * dW_vol_dJ
    dev_factor = 2.0 * psi_dev * Jm23

    S = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            ident = 1.0 if i == j else 0.0
            S[i, j] = dev_factor * (ident - (trC / 3.0) * Cinv[i, j]) + vol_scale * Cinv[i, j]

    S_voigt = np.empty(6, dtype=np.float64)
    S_voigt[0] = S[0, 0]
    S_voigt[1] = S[1, 1]
    S_voigt[2] = S[2, 2]
    S_voigt[3] = S[0, 1]
    S_voigt[4] = S[1, 2]
    S_voigt[5] = S[2, 0]
    return S_voigt


@njit(fastmath=True)
def _mooney_rivlin_pk2_voigt(
    E_voigt: np.ndarray,
    c10: float, c01: float, d1: float
) -> np.ndarray:
    """PK2 stress for genuine 2-parameter Mooney-Rivlin model:
        W = c10*(I_bar_1 - 3) + c01*(I_bar_2 - 3) + (1/d1)*(J - 1)^2

    Invariants:
        I1 = tr(C), I2 = 0.5 * [(tr C)^2 - tr(C^2)]
        I_bar_1 = J^(-2/3) * I1, I_bar_2 = J^(-4/3) * I2

    Derivatives w.r.t C:
        dI_bar_1/dC = J^(-2/3) * (I - (I1/3)*Cinv)
        dI_bar_2/dC = J^(-4/3) * (I1*I - C - (2*I2/3)*Cinv)
        S_vol = (2/d1) * (J - 1) * J * Cinv
    """
    C = np.empty((3, 3), dtype=np.float64)
    C[0, 0] = 1.0 + 2.0 * E_voigt[0]
    C[1, 1] = 1.0 + 2.0 * E_voigt[1]
    C[2, 2] = 1.0 + 2.0 * E_voigt[2]
    C[0, 1] = C[1, 0] = E_voigt[3]
    C[1, 2] = C[2, 1] = E_voigt[4]
    C[2, 0] = C[0, 2] = E_voigt[5]

    J, detC, Cinv, trC = _inv_and_det_c(C)

    # tr(C^2)
    trC2 = (
        C[0, 0]**2 + C[1, 1]**2 + C[2, 2]**2
        + 2.0 * (C[0, 1]**2 + C[1, 2]**2 + C[2, 0]**2)
    )
    I2 = 0.5 * (trC * trC - trC2)

    Jm23 = J ** (-2.0 / 3.0)
    Jm43 = Jm23 * Jm23
    I1_bar = Jm23 * trC
    I2_bar = Jm43 * I2

    # Volumetric pressure
    Jm1 = J - 1.0
    vol_scale = (2.0 / max(d1, 1e-12)) * Jm1 * J if abs(d1) > 1e-15 else 0.0

    S = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            ident = 1.0 if i == j else 0.0
            term1 = 2.0 * c10 * Jm23 * (ident - (trC / 3.0) * Cinv[i, j])
            term2 = 2.0 * c01 * Jm43 * (trC * ident - C[i, j] - (2.0 * I2 / 3.0) * Cinv[i, j])
            S[i, j] = term1 + term2 + vol_scale * Cinv[i, j]

    S_voigt = np.empty(6, dtype=np.float64)
    S_voigt[0] = S[0, 0]
    S_voigt[1] = S[1, 1]
    S_voigt[2] = S[2, 2]
    S_voigt[3] = S[0, 1]
    S_voigt[4] = S[1, 2]
    S_voigt[5] = S[2, 0]
    return S_voigt


@njit(fastmath=True)
def _polynomial_pk2_voigt(
    E_voigt: np.ndarray,
    c10: float, c01: float, c20: float, c11: float, c02: float,
    d1: float, d2: float
) -> np.ndarray:
    """PK2 stress for general N=2 Polynomial hyperelastic model:
        W_dev = c10*(I_bar_1 - 3) + c01*(I_bar_2 - 3)
              + c20*(I_bar_1 - 3)^2 + c11*(I_bar_1 - 3)*(I_bar_2 - 3) + c02*(I_bar_2 - 3)^2
        W_vol = (1/d1)*(J - 1)^2 + (1/d2)*(J - 1)^4
    """
    C = np.empty((3, 3), dtype=np.float64)
    C[0, 0] = 1.0 + 2.0 * E_voigt[0]
    C[1, 1] = 1.0 + 2.0 * E_voigt[1]
    C[2, 2] = 1.0 + 2.0 * E_voigt[2]
    C[0, 1] = C[1, 0] = E_voigt[3]
    C[1, 2] = C[2, 1] = E_voigt[4]
    C[2, 0] = C[0, 2] = E_voigt[5]

    J, detC, Cinv, trC = _inv_and_det_c(C)

    trC2 = (
        C[0, 0]**2 + C[1, 1]**2 + C[2, 2]**2
        + 2.0 * (C[0, 1]**2 + C[1, 2]**2 + C[2, 0]**2)
    )
    I2 = 0.5 * (trC * trC - trC2)

    Jm23 = J ** (-2.0 / 3.0)
    Jm43 = Jm23 * Jm23
    I1_bar = Jm23 * trC
    I2_bar = Jm43 * I2

    m1 = I1_bar - 3.0
    m2 = I2_bar - 3.0

    # Partial derivatives dW/dI1_bar, dW/dI2_bar
    dW_dI1 = c10 + 2.0 * c20 * m1 + c11 * m2
    dW_dI2 = c01 + c11 * m1 + 2.0 * c02 * m2

    # Volumetric derivative
    Jm1 = J - 1.0
    dW_vol_dJ = 0.0
    if abs(d1) > 1e-15:
        dW_vol_dJ += (2.0 * Jm1) / d1
    if abs(d2) > 1e-15:
        dW_vol_dJ += (4.0 * (Jm1 ** 3)) / d2
    vol_scale = J * dW_vol_dJ

    S = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            ident = 1.0 if i == j else 0.0
            term1 = 2.0 * dW_dI1 * Jm23 * (ident - (trC / 3.0) * Cinv[i, j])
            term2 = 2.0 * dW_dI2 * Jm43 * (trC * ident - C[i, j] - (2.0 * I2 / 3.0) * Cinv[i, j])
            S[i, j] = term1 + term2 + vol_scale * Cinv[i, j]

    S_voigt = np.empty(6, dtype=np.float64)
    S_voigt[0] = S[0, 0]
    S_voigt[1] = S[1, 1]
    S_voigt[2] = S[2, 2]
    S_voigt[3] = S[0, 1]
    S_voigt[4] = S[1, 2]
    S_voigt[5] = S[2, 0]
    return S_voigt


@njit(fastmath=True)
def _eig3x3_jacobi_numba(A: np.ndarray):
    """Fully-unrolled Cyclic Jacobi eigensolver for 3x3 symmetric matrix.
    Achieves machine precision (1e-15) in 5 sweeps without trigonometric singularities.
    """
    V = np.eye(3, dtype=np.float64)
    D = A.copy()
    for _ in range(5):
        for p in range(3):
            for q in range(p + 1, 3):
                app = D[p, p]
                aqq = D[q, q]
                apq = D[p, q]
                if abs(apq) > 1e-15:
                    tau = (aqq - app) / (2.0 * apq)
                    t = np.sign(tau) / (abs(tau) + np.sqrt(1.0 + tau * tau)) if abs(tau) > 1e-15 else 1.0
                    c = 1.0 / np.sqrt(1.0 + t * t)
                    s = t * c
                    D[p, p] = app - t * apq
                    D[q, q] = aqq + t * apq
                    D[p, q] = 0.0
                    D[q, p] = 0.0
                    r = 3 - p - q
                    dpr = D[p, r]
                    dqr = D[q, r]
                    D[p, r] = c * dpr - s * dqr
                    D[r, p] = D[p, r]
                    D[q, r] = s * dpr + c * dqr
                    D[r, q] = D[q, r]
                    for k in range(3):
                        vkp = V[k, p]
                        vkq = V[k, q]
                        V[k, p] = c * vkp - s * vkq
                        V[k, q] = s * vkp + c * vkq
    return np.array([D[0, 0], D[1, 1], D[2, 2]], dtype=np.float64), V


@njit(fastmath=True)
def _ogden_pk2_voigt(
    E_voigt: np.ndarray,
    n_terms: int,
    mus: np.ndarray,
    alphas: np.ndarray,
    d1: float
) -> np.ndarray:
    """PK2 stress for Ogden hyperelastic model (N=1..6):
        W_dev = sum_{p=1}^N (2*mu_p / alpha_p^2) * (lambda_bar_1^alpha_p + lambda_bar_2^alpha_p + lambda_bar_3^alpha_p - 3)
        W_vol = (1/d1) * (J - 1)^2

    Principal stretches lambda_a computed via robust 3x3 Cyclic Jacobi eigensolver.
    """
    C = np.empty((3, 3), dtype=np.float64)
    C[0, 0] = 1.0 + 2.0 * E_voigt[0]
    C[1, 1] = 1.0 + 2.0 * E_voigt[1]
    C[2, 2] = 1.0 + 2.0 * E_voigt[2]
    C[0, 1] = C[1, 0] = E_voigt[3]
    C[1, 2] = C[2, 1] = E_voigt[4]
    C[2, 0] = C[0, 2] = E_voigt[5]

    w_C, V = _eig3x3_jacobi_numba(C)
    lam = np.empty(3, dtype=np.float64)
    for a in range(3):
        lam[a] = np.sqrt(max(w_C[a], 1e-14))

    J = lam[0] * lam[1] * lam[2]
    Jm13 = J ** (-1.0 / 3.0)
    lam_bar = lam * Jm13

    # Deviatoric derivatives: dW_dev / dlam_a
    # sum_p (2*mu_p / alpha_p) * lam_bar_a^alpha_p - (1/3) * sum_b [ (2*mu_p / alpha_p) * lam_bar_b^alpha_p ]
    term_b_sum = 0.0
    for p in range(n_terms):
        mu_p = mus[p]
        alpha_p = alphas[p]
        fac = 2.0 * mu_p / alpha_p
        for b in range(3):
            term_b_sum += fac * (lam_bar[b] ** alpha_p)

    dW_dlam = np.empty(3, dtype=np.float64)
    for a in range(3):
        term_a = 0.0
        for p in range(n_terms):
            mu_p = mus[p]
            alpha_p = alphas[p]
            fac = 2.0 * mu_p / alpha_p
            term_a += fac * (lam_bar[a] ** alpha_p)
        dW_dlam[a] = (term_a - (1.0 / 3.0) * term_b_sum) / max(lam[a], 1e-14)

    # Volumetric derivative: dW_vol / dlam_a = (2/d1) * (J - 1) * (J / lam_a)
    vol_scale = (2.0 / max(d1, 1e-12)) * (J - 1.0) * J if abs(d1) > 1e-15 else 0.0
    for a in range(3):
        dW_dlam[a] += vol_scale / max(lam[a], 1e-14)

    # Principal 2nd Piola-Kirchhoff stress S_a = (1 / lam_a) * dW / dlam_a
    S_princ = np.empty(3, dtype=np.float64)
    for a in range(3):
        S_princ[a] = dW_dlam[a] / max(lam[a], 1e-14)

    # Reconstruct 2nd P-K stress tensor S = sum_a S_princ[a] * (N_a (x) N_a)
    S = np.zeros((3, 3), dtype=np.float64)
    for a in range(3):
        for i in range(3):
            for j in range(3):
                S[i, j] += S_princ[a] * V[i, a] * V[j, a]

    S_voigt = np.empty(6, dtype=np.float64)
    S_voigt[0] = S[0, 0]
    S_voigt[1] = S[1, 1]
    S_voigt[2] = S[2, 2]
    S_voigt[3] = S[0, 1]
    S_voigt[4] = S[1, 2]
    S_voigt[5] = S[2, 0]
    return S_voigt


@njit(fastmath=True)
def _interp_tabular_hardening(eqps: float, n_pts: int, table_sy: np.ndarray, table_ep: np.ndarray):
    """Piecewise linear interpolation with O(log N) binary search for tabular hardening.
    Returns current yield stress sigma_y and local hardening modulus H = d(sigma_y)/d(eqps).
    """
    if n_pts <= 1:
        return table_sy[0], 0.0
    if eqps <= table_ep[0]:
        H0 = (table_sy[1] - table_sy[0]) / max(table_ep[1] - table_ep[0], 1e-12)
        return table_sy[0], H0
    if eqps >= table_ep[n_pts - 1]:
        return table_sy[n_pts - 1], 0.0

    low = 0
    high = n_pts - 1
    while high - low > 1:
        mid = (low + high) // 2
        if table_ep[mid] <= eqps:
            low = mid
        else:
            high = mid
    dep = max(table_ep[high] - table_ep[low], 1e-12)
    H = (table_sy[high] - table_sy[low]) / dep
    sy = table_sy[low] + H * (eqps - table_ep[low])
    return sy, H



@njit(fastmath=True)
def material_dispatch_3d(
    mat_type: int,
    props: np.ndarray,
    sdv_prev: np.ndarray,
    E_voigt: np.ndarray,
    F: np.ndarray,
    detF: float,
    dt: float
):
    """
    Unified Material Routine (UMAT style).
    Takes kinematic inputs (Strain/Deformation Gradient) and returns
    2nd Piola-Kirchhoff Stress (S_voigt) and Tangent Modulus (C_mat).
    
    Args:
        mat_type (int): Material model identifier.
        props (np.ndarray): Flattened array of material properties [E, nu, yield, ...].
        sdv_prev (np.ndarray): State dependent variables from previous converged step.
        E_voigt (np.ndarray): Green-Lagrange strain tensor (6,).
        F (np.ndarray): Deformation gradient (3x3).
        detF (float): Determinant of F (J).
        dt (float): Time increment.
        
    Returns:
        S_voigt (np.ndarray): 2nd Piola-Kirchhoff stress in Voigt notation (6,).
        C_mat (np.ndarray): Material tangent stiffness matrix (6x6).
        sdv_new (np.ndarray): Updated state dependent variables.
        error_flag (int): 0 for success, 1 for failure (e.g. invalid deformation).
    """
    S_voigt = np.zeros(6, dtype=np.float64)
    C_mat = np.zeros((6, 6), dtype=np.float64)
    sdv_new = sdv_prev.copy()
    error_flag = 0

    if detF <= 0.0:
        error_flag = 1  # Element Inverted (Negative Volume)
        return S_voigt, C_mat, sdv_new, error_flag

    if mat_type == MAT_LINEAR_ELASTIC:
        # props: [E, nu]
        E = props[0]
        nu = props[1]
        lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        
        C_mat[0, 0] = lam + 2*mu; C_mat[0, 1] = lam;        C_mat[0, 2] = lam
        C_mat[1, 0] = lam;        C_mat[1, 1] = lam + 2*mu; C_mat[1, 2] = lam
        C_mat[2, 0] = lam;        C_mat[2, 1] = lam;        C_mat[2, 2] = lam + 2*mu
        C_mat[3, 3] = mu
        C_mat[4, 4] = mu
        C_mat[5, 5] = mu
        
        S_voigt = C_mat @ E_voigt

    elif mat_type == MAT_HYPERELASTIC_NEOHOOKEAN:
        # Robust decoding for Neo-Hookean parameters:
        # Case 1: [C10, D1] (e.g., C10=0.1, D1=0.01) -> mu = 2*C10, K = 2/D1
        # Case 2: [E, nu] (e.g., E=200000, nu=0.4999) -> mu = E/(2*(1+nu)), K = E/(3*(1-2*nu))
        # Case 3: [mu, K] -> direct
        p0 = props[0]
        p1 = props[1]
        if p1 <= 0.05 and p0 > 0.0:
            mu = 2.0 * p0
            K = 2.0 / max(p1, 1e-12)
        elif p1 < 0.5 and p0 > 1.0:
            mu = p0 / (2.0 * (1.0 + p1))
            K = p0 / (3.0 * max(1.0 - 2.0 * p1, 1e-8))
        else:
            mu = p0
            K = max(p1, 1e-8)

        # Genuine finite-strain compressible Neo-Hookean (see
        # _neohookean_pk2_voigt's docstring for the W(C)/S(C) formula and
        # citation) -- this REPLACES the previous placeholder, which
        # computed `S = C_mat @ E_voigt` with a CONSTANT small-strain C_mat,
        # bit-identical to MAT_CUSTOM_ELASTIC/MAT_LINEAR_ELASTIC at the same
        # (E,nu): no J-dependent volumetric term, no deviatoric-invariant
        # stored-energy function -- verified fake this session
        # (dev_log/3d_capability_buildout... T3 / .omc/plans/
        # 3d_capability_buildout_20260913.md).
        S_voigt = _neohookean_pk2_voigt(E_voigt, mu, K)

        # Consistent tangent via a self-contained central finite
        # difference of _neohookean_pk2_voigt itself, perturbing E_voigt
        # only (C = 2E+I is reconstructed from E_voigt inside the helper,
        # not from the separately-passed F -- see that function's
        # docstring for why this keeps the tangent exact-by-construction
        # against the coded stress law, the same principle this project's
        # C3D8_FBAR fix already established, dev_log/
        # solve_step_false_convergence_20260913.md). Per-column adaptive
        # step (Dennis & Schnabel 1983 Sec 5.4, AGENTS.md's B4 convention)
        # -- NOT a fixed absolute step, which was a latent mesh-refinement
        # bug the last time this project used one.
        sqrt_eps = 1.4901161193847656e-08
        for j in range(6):
            h = sqrt_eps * max(abs(E_voigt[j]), 1.0)
            E_p = E_voigt.copy()
            E_p[j] += h
            E_m = E_voigt.copy()
            E_m[j] -= h
            S_p = _neohookean_pk2_voigt(E_p, mu, K)
            S_m = _neohookean_pk2_voigt(E_m, mu, K)
            for i in range(6):
                C_mat[i, j] = (S_p[i] - S_m[i]) / (2.0 * h)

    elif mat_type == MAT_HYPERELASTIC_ARRUDA_BOYCE:
        # props: [mu, lambda_m, K] (or [mu, lambda_m, d1] where K=2/d1)
        mu = props[0]
        lambda_m = props[1] if props[1] > 0.0 else 3.0
        p2 = props[2]
        if p2 <= 0.05 and p2 > 0.0:
            K = 2.0 / max(p2, 1e-12)
        else:
            K = max(p2, 1e-8)

        S_voigt = _arruda_boyce_pk2_voigt(E_voigt, mu, lambda_m, K)

        # Consistent tangent via adaptive-step central finite difference
        sqrt_eps = 1.4901161193847656e-08
        for j in range(6):
            h = sqrt_eps * max(abs(E_voigt[j]), 1.0)
            E_p = E_voigt.copy()
            E_p[j] += h
            E_m = E_voigt.copy()
            E_m[j] -= h
            S_p = _arruda_boyce_pk2_voigt(E_p, mu, lambda_m, K)
            S_m = _arruda_boyce_pk2_voigt(E_m, mu, lambda_m, K)
            for i in range(6):
                C_mat[i, j] = (S_p[i] - S_m[i]) / (2.0 * h)

    elif mat_type == MAT_HYPERELASTIC_YEOH:
        # props: [c10, c20, c30, d1, d2, d3]
        c10 = props[0]
        c20 = props[1]
        c30 = props[2]
        d1 = props[3]
        d2 = props[4] if len(props) > 4 else 0.0
        d3 = props[5] if len(props) > 5 else 0.0

        S_voigt = _yeoh_pk2_voigt(E_voigt, c10, c20, c30, d1, d2, d3)

        # Consistent tangent via adaptive-step central finite difference
        sqrt_eps = 1.4901161193847656e-08
        for j in range(6):
            h = sqrt_eps * max(abs(E_voigt[j]), 1.0)
            E_p = E_voigt.copy()
            E_p[j] += h
            E_m = E_voigt.copy()
            E_m[j] -= h
            S_p = _yeoh_pk2_voigt(E_p, c10, c20, c30, d1, d2, d3)
            S_m = _yeoh_pk2_voigt(E_m, c10, c20, c30, d1, d2, d3)
            for i in range(6):
                C_mat[i, j] = (S_p[i] - S_m[i]) / (2.0 * h)

    elif mat_type == MAT_J2_PLASTICITY:
        # props: [E, nu, sigma_y0, H]
        E = props[0]
        nu = props[1]
        sigma_y0 = props[2]
        H = props[3]

        mu = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        lam = K - (2.0 / 3.0) * mu

        # Elastic tangent matrix
        C_mat[0, 0] = lam + 2*mu; C_mat[0, 1] = lam;        C_mat[0, 2] = lam
        C_mat[1, 0] = lam;        C_mat[1, 1] = lam + 2*mu; C_mat[1, 2] = lam
        C_mat[2, 0] = lam;        C_mat[2, 1] = lam;        C_mat[2, 2] = lam + 2*mu
        C_mat[3, 3] = mu
        C_mat[4, 4] = mu
        C_mat[5, 5] = mu

        ep_old = np.zeros(6, dtype=np.float64)
        eq_p_old = 0.0
        if len(sdv_prev) >= 7:
            for i in range(6):
                ep_old[i] = sdv_prev[i]
            eq_p_old = sdv_prev[6]

        e_trial = E_voigt - ep_old
        trace_e = e_trial[0] + e_trial[1] + e_trial[2]
        e_dev = np.zeros(6, dtype=np.float64)
        e_dev[0] = e_trial[0] - trace_e / 3.0
        e_dev[1] = e_trial[1] - trace_e / 3.0
        e_dev[2] = e_trial[2] - trace_e / 3.0
        e_dev[3] = e_trial[3]
        e_dev[4] = e_trial[4]
        e_dev[5] = e_trial[5]

        s_trial = np.zeros(6, dtype=np.float64)
        s_trial[0] = 2.0 * mu * e_dev[0]
        s_trial[1] = 2.0 * mu * e_dev[1]
        s_trial[2] = 2.0 * mu * e_dev[2]
        s_trial[3] = mu * e_dev[3]
        s_trial[4] = mu * e_dev[4]
        s_trial[5] = mu * e_dev[5]

        norm_s = np.sqrt(
            s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2 +
            2.0 * (s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2)
        )
        q_trial = np.sqrt(1.5) * norm_s
        sigma_y_cur = sigma_y0 + H * eq_p_old
        f_trial = q_trial - sigma_y_cur

        hydro_p = K * trace_e
        if f_trial <= 0.0:
            S_voigt = s_trial.copy()
            S_voigt[0] += hydro_p
            S_voigt[1] += hydro_p
            S_voigt[2] += hydro_p
        else:
            d_gamma = f_trial / (3.0 * mu + H)
            eq_p_new = eq_p_old + d_gamma
            scale_s = 1.0 - (3.0 * mu * d_gamma) / max(q_trial, 1e-12)
            s_new = s_trial * scale_s

            S_voigt = s_new.copy()
            S_voigt[0] += hydro_p
            S_voigt[1] += hydro_p
            S_voigt[2] += hydro_p

            if len(sdv_new) >= 7:
                n_flow = s_trial / max(norm_s, 1e-12)
                flow_fac = d_gamma * np.sqrt(1.5)
                for i in range(6):
                    sdv_new[i] = ep_old[i] + flow_fac * n_flow[i]
                sdv_new[6] = eq_p_new

            # Algorithmic consistent elastoplastic tangent C_ep (Simo & Hughes 1998)
            beta0 = (3.0 * mu * d_gamma) / max(q_trial, 1e-12)
            beta1 = (3.0 * mu) / (3.0 * mu + H)
            n_vec = s_trial / max(norm_s, 1e-12)

            C_vol = np.zeros((6, 6), dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    C_vol[r, c] = K

            C_dev = np.zeros((6, 6), dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    C_dev[r, c] = - (2.0 / 3.0) * mu
                C_dev[r, r] += 2.0 * mu
            C_dev[3, 3] = mu
            C_dev[4, 4] = mu
            C_dev[5, 5] = mu

            C_mat = C_vol + (1.0 - beta0) * C_dev - 2.0 * mu * (beta1 - beta0) * np.outer(n_vec, n_vec)

    elif mat_type == MAT_J2_PLASTICITY_NONLINEAR:
        # props: [E, nu, sigma_y0, H_lin, R_inf, b_exp] (Voce saturation hardening)
        # sigma_y(eqps) = sigma_y0 + H_lin*eqps + R_inf*(1 - exp(-b_exp*eqps))
        E = props[0]
        nu = props[1]
        sigma_y0 = props[2]
        H_lin = props[3]
        R_inf = props[4]
        b_exp = max(props[5], 0.0)

        mu = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        lam = K - (2.0 / 3.0) * mu

        # Elastic tangent matrix
        C_mat[0, 0] = lam + 2*mu; C_mat[0, 1] = lam;        C_mat[0, 2] = lam
        C_mat[1, 0] = lam;        C_mat[1, 1] = lam + 2*mu; C_mat[1, 2] = lam
        C_mat[2, 0] = lam;        C_mat[2, 1] = lam;        C_mat[2, 2] = lam + 2*mu
        C_mat[3, 3] = mu
        C_mat[4, 4] = mu
        C_mat[5, 5] = mu

        ep_old = np.zeros(6, dtype=np.float64)
        eq_p_old = 0.0
        if len(sdv_prev) >= 7:
            for i in range(6):
                ep_old[i] = sdv_prev[i]
            eq_p_old = sdv_prev[6]

        e_trial = E_voigt - ep_old
        trace_e = e_trial[0] + e_trial[1] + e_trial[2]
        e_dev = np.zeros(6, dtype=np.float64)
        e_dev[0] = e_trial[0] - trace_e / 3.0
        e_dev[1] = e_trial[1] - trace_e / 3.0
        e_dev[2] = e_trial[2] - trace_e / 3.0
        e_dev[3] = e_trial[3]
        e_dev[4] = e_trial[4]
        e_dev[5] = e_trial[5]

        s_trial = np.zeros(6, dtype=np.float64)
        s_trial[0] = 2.0 * mu * e_dev[0]
        s_trial[1] = 2.0 * mu * e_dev[1]
        s_trial[2] = 2.0 * mu * e_dev[2]
        s_trial[3] = mu * e_dev[3]
        s_trial[4] = mu * e_dev[4]
        s_trial[5] = mu * e_dev[5]

        norm_s = np.sqrt(
            s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2 +
            2.0 * (s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2)
        )
        q_trial = np.sqrt(1.5) * norm_s
        sigma_y_cur = sigma_y0 + H_lin * eq_p_old + R_inf * (1.0 - np.exp(-b_exp * eq_p_old))
        f_trial = q_trial - sigma_y_cur

        hydro_p = K * trace_e
        if f_trial <= 0.0:
            S_voigt = s_trial.copy()
            S_voigt[0] += hydro_p
            S_voigt[1] += hydro_p
            S_voigt[2] += hydro_p
        else:
            # Local Newton loop for delta_gamma under non-linear hardening
            d_gamma = 0.0
            eq_p_new = eq_p_old
            H_eff = H_lin + R_inf * b_exp * np.exp(-b_exp * eq_p_old)
            for _iter in range(20):
                sig_y = sigma_y0 + H_lin * eq_p_new + R_inf * (1.0 - np.exp(-b_exp * eq_p_new))
                res = q_trial - 3.0 * mu * d_gamma - sig_y
                if abs(res) < 1e-11:
                    break
                H_eff = H_lin + R_inf * b_exp * np.exp(-b_exp * eq_p_new)
                denom = 3.0 * mu + H_eff
                d_dgamma = res / denom
                d_gamma += d_dgamma
                eq_p_new = eq_p_old + d_gamma
                if abs(d_dgamma) < 1e-12:
                    break

            scale_s = 1.0 - (3.0 * mu * d_gamma) / max(q_trial, 1e-12)
            s_new = s_trial * scale_s

            S_voigt = s_new.copy()
            S_voigt[0] += hydro_p
            S_voigt[1] += hydro_p
            S_voigt[2] += hydro_p

            if len(sdv_new) >= 7:
                n_flow = s_trial / max(norm_s, 1e-12)
                flow_fac = d_gamma * np.sqrt(1.5)
                for i in range(6):
                    sdv_new[i] = ep_old[i] + flow_fac * n_flow[i]
                sdv_new[6] = eq_p_new

            # Consistent elastoplastic tangent
            H_eff = H_lin + R_inf * b_exp * np.exp(-b_exp * eq_p_new)
            beta0 = (3.0 * mu * d_gamma) / max(q_trial, 1e-12)
            beta1 = (3.0 * mu) / (3.0 * mu + H_eff)
            n_vec = s_trial / max(norm_s, 1e-12)

            C_vol = np.zeros((6, 6), dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    C_vol[r, c] = K

            C_dev = np.zeros((6, 6), dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    C_dev[r, c] = - (2.0 / 3.0) * mu
                C_dev[r, r] += 2.0 * mu
            C_dev[3, 3] = mu
            C_dev[4, 4] = mu
            C_dev[5, 5] = mu

            C_mat = C_vol + (1.0 - beta0) * C_dev - 2.0 * mu * (beta1 - beta0) * np.outer(n_vec, n_vec)

    elif mat_type == MAT_VISCOELASTIC_PRONY:
        # props: [E/mu, nu/K, g1, tau1]
        p0 = props[0]
        p1 = props[1]
        if p1 < 0.5 and p0 > 0.0:
            mu = p0 / (2.0 * (1.0 + p1))
            K = p0 / (3.0 * max(1.0 - 2.0 * p1, 1e-8))
        else:
            mu = p0
            K = p1

        g1 = props[2]
        tau1 = max(props[3], 1e-12)

        # Deviatoric and volumetric strain
        eps_vol = E_voigt[0] + E_voigt[1] + E_voigt[2]
        e_dev = np.zeros(6, dtype=np.float64)
        e_dev[0] = E_voigt[0] - eps_vol / 3.0
        e_dev[1] = E_voigt[1] - eps_vol / 3.0
        e_dev[2] = E_voigt[2] - eps_vol / 3.0
        e_dev[3] = E_voigt[3]
        e_dev[4] = E_voigt[4]
        e_dev[5] = E_voigt[5]

        # Instantaneous elastic deviatoric stress (shear: 2*mu*(gamma/2) = mu*gamma)
        s_dev_el = np.zeros(6, dtype=np.float64)
        s_dev_el[0] = 2.0 * mu * e_dev[0]
        s_dev_el[1] = 2.0 * mu * e_dev[1]
        s_dev_el[2] = 2.0 * mu * e_dev[2]
        s_dev_el[3] = mu * e_dev[3]
        s_dev_el[4] = mu * e_dev[4]
        s_dev_el[5] = mu * e_dev[5]

        # Time integration parameters (linearized midpoint rule, Simo & Hughes 1998)
        dt_safe = max(dt, 1e-12)
        xi = dt_safe / tau1
        beta1 = np.exp(-xi)
        gamma1 = (1.0 - beta1) / xi if xi > 1e-6 else (1.0 - 0.5 * xi)

        h_old = np.zeros(6, dtype=np.float64)
        s_dev_prev = np.zeros(6, dtype=np.float64)
        if sdv_prev.shape[0] >= 12:
            for i in range(6):
                h_old[i] = sdv_prev[i]
                s_dev_prev[i] = sdv_prev[6 + i]

        # Update overstress h1(t + dt) = beta1 * h1(t) + g1 * gamma1 * (s_dev_el - s_dev_prev)
        h_new = np.zeros(6, dtype=np.float64)
        for i in range(6):
            delta_s = s_dev_el[i] - s_dev_prev[i]
            h_new[i] = beta1 * h_old[i] + g1 * gamma1 * delta_s
            if sdv_new.shape[0] >= 12:
                sdv_new[i] = h_new[i]
                sdv_new[6 + i] = s_dev_el[i]

        # Total stress: S = S_vol + S_dev, where S_dev = (1 - g1) * s_dev_el + h_new
        g_inf = 1.0 - g1
        S_voigt[0] = K * eps_vol + g_inf * s_dev_el[0] + h_new[0]
        S_voigt[1] = K * eps_vol + g_inf * s_dev_el[1] + h_new[1]
        S_voigt[2] = K * eps_vol + g_inf * s_dev_el[2] + h_new[2]
        S_voigt[3] = g_inf * s_dev_el[3] + h_new[3]
        S_voigt[4] = g_inf * s_dev_el[4] + h_new[4]
        S_voigt[5] = g_inf * s_dev_el[5] + h_new[5]

        # Algorithmic consistent tangent C_visco = C_vol + (g_inf + g1 * gamma1) * C_dev
        factor_dev = g_inf + g1 * gamma1
        mu_eff = mu * factor_dev

        for r in range(3):
            for c in range(3):
                C_mat[r, c] = K - (2.0 / 3.0) * mu_eff
            C_mat[r, r] += 2.0 * mu_eff
        C_mat[3, 3] = mu_eff
        C_mat[4, 4] = mu_eff
        C_mat[5, 5] = mu_eff

    elif mat_type == MAT_HYPERELASTIC_MOONEY_RIVLIN:
        # props: [c10, c01, d1]
        c10 = props[0]
        c01 = props[1]
        d1 = props[2]
        S_voigt = _mooney_rivlin_pk2_voigt(E_voigt, c10, c01, d1)

        # Consistent tangent via adaptive-step central finite difference
        sqrt_eps = 1.4901161193847656e-08
        for j in range(6):
            h = sqrt_eps * max(abs(E_voigt[j]), 1.0)
            E_p = E_voigt.copy()
            E_p[j] += h
            E_m = E_voigt.copy()
            E_m[j] -= h
            S_p = _mooney_rivlin_pk2_voigt(E_p, c10, c01, d1)
            S_m = _mooney_rivlin_pk2_voigt(E_m, c10, c01, d1)
            for i in range(6):
                C_mat[i, j] = (S_p[i] - S_m[i]) / (2.0 * h)

    elif mat_type == MAT_HYPERELASTIC_OGDEN:
        # props: [n_terms, d1, mu1, a1, mu2, a2, mu3, a3, mu4, a4, mu5, a5, mu6, a6]
        n_terms = int(props[0])
        if n_terms < 1:
            n_terms = 1
        if n_terms > 6:
            n_terms = 6
        d1 = props[1]
        mus = np.zeros(n_terms, dtype=np.float64)
        alphas = np.zeros(n_terms, dtype=np.float64)
        for p in range(n_terms):
            mus[p] = props[2 + 2 * p]
            alphas[p] = props[3 + 2 * p]

        S_voigt = _ogden_pk2_voigt(E_voigt, n_terms, mus, alphas, d1)

        # Consistent tangent via adaptive-step central finite difference
        sqrt_eps = 1.4901161193847656e-08
        for j in range(6):
            h = sqrt_eps * max(abs(E_voigt[j]), 1.0)
            E_p = E_voigt.copy()
            E_p[j] += h
            E_m = E_voigt.copy()
            E_m[j] -= h
            S_p = _ogden_pk2_voigt(E_p, n_terms, mus, alphas, d1)
            S_m = _ogden_pk2_voigt(E_m, n_terms, mus, alphas, d1)
            for i in range(6):
                C_mat[i, j] = (S_p[i] - S_m[i]) / (2.0 * h)

    elif mat_type == MAT_HYPERELASTIC_POLYNOMIAL:
        # props: [c10, c01, c20, c11, c02, d1, d2]
        c10 = props[0]; c01 = props[1]; c20 = props[2]; c11 = props[3]; c02 = props[4]
        d1 = props[5]; d2 = props[6]
        S_voigt = _polynomial_pk2_voigt(E_voigt, c10, c01, c20, c11, c02, d1, d2)

        sqrt_eps = 1.4901161193847656e-08
        for j in range(6):
            h = sqrt_eps * max(abs(E_voigt[j]), 1.0)
            E_p = E_voigt.copy()
            E_p[j] += h
            E_m = E_voigt.copy()
            E_m[j] -= h
            S_p = _polynomial_pk2_voigt(E_p, c10, c01, c20, c11, c02, d1, d2)
            S_m = _polynomial_pk2_voigt(E_m, c10, c01, c20, c11, c02, d1, d2)
            for i in range(6):
                C_mat[i, j] = (S_p[i] - S_m[i]) / (2.0 * h)

    elif mat_type == MAT_HYPERELASTIC_MARLOW:
        # Marlow: non-parametric potential modeled via Yeoh 3-term fit
        c10 = props[0]; c20 = props[1]; c30 = props[2]
        d1 = props[3]; d2 = props[4] if len(props) > 4 else 0.0; d3 = props[5] if len(props) > 5 else 0.0
        S_voigt = _yeoh_pk2_voigt(E_voigt, c10, c20, c30, d1, d2, d3)

        sqrt_eps = 1.4901161193847656e-08
        for j in range(6):
            h = sqrt_eps * max(abs(E_voigt[j]), 1.0)
            E_p = E_voigt.copy()
            E_p[j] += h
            E_m = E_voigt.copy()
            E_m[j] -= h
            S_p = _yeoh_pk2_voigt(E_p, c10, c20, c30, d1, d2, d3)
            S_m = _yeoh_pk2_voigt(E_m, c10, c20, c30, d1, d2, d3)
            for i in range(6):
                C_mat[i, j] = (S_p[i] - S_m[i]) / (2.0 * h)

    elif mat_type == MAT_J2_PLASTICITY_TABULAR:
        # props: [E, nu, n_pts, eps_fail, sigma_fail, sy0, ep0, sy1, ep1, ..., sy_k, ep_k]
        E = props[0]
        nu = props[1]
        n_pts = int(props[2])
        eps_fail = props[3]
        sigma_fail = props[4]

        table_sy = np.zeros(n_pts, dtype=np.float64)
        table_ep = np.zeros(n_pts, dtype=np.float64)
        for p in range(n_pts):
            table_sy[p] = props[5 + 2 * p]
            table_ep[p] = props[6 + 2 * p]

        mu = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        lam = K - (2.0 / 3.0) * mu

        ep_old = np.zeros(6, dtype=np.float64)
        eq_p_old = 0.0
        is_failed_old = 0.0
        if len(sdv_prev) >= 8:
            for i in range(6):
                ep_old[i] = sdv_prev[i]
            eq_p_old = sdv_prev[6]
            is_failed_old = sdv_prev[7]

        e_trial = E_voigt - ep_old
        trace_e = e_trial[0] + e_trial[1] + e_trial[2]
        e_dev = np.zeros(6, dtype=np.float64)
        e_dev[0] = e_trial[0] - trace_e / 3.0
        e_dev[1] = e_trial[1] - trace_e / 3.0
        e_dev[2] = e_trial[2] - trace_e / 3.0
        e_dev[3] = e_trial[3]
        e_dev[4] = e_trial[4]
        e_dev[5] = e_trial[5]

        s_trial = np.zeros(6, dtype=np.float64)
        s_trial[0] = 2.0 * mu * e_dev[0]
        s_trial[1] = 2.0 * mu * e_dev[1]
        s_trial[2] = 2.0 * mu * e_dev[2]
        s_trial[3] = mu * e_dev[3]
        s_trial[4] = mu * e_dev[4]
        s_trial[5] = mu * e_dev[5]

        norm_s = np.sqrt(
            s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2 +
            2.0 * (s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2)
        )
        q_trial = np.sqrt(1.5) * norm_s
        sy_cur, H_cur = _interp_tabular_hardening(eq_p_old, n_pts, table_sy, table_ep)
        f_trial = q_trial - sy_cur
        hydro_p = K * trace_e

        if f_trial <= 0.0:
            S_voigt = s_trial.copy()
            S_voigt[0] += hydro_p
            S_voigt[1] += hydro_p
            S_voigt[2] += hydro_p
            # Elastic tangent
            C_mat[0, 0] = lam + 2*mu; C_mat[0, 1] = lam;        C_mat[0, 2] = lam
            C_mat[1, 0] = lam;        C_mat[1, 1] = lam + 2*mu; C_mat[1, 2] = lam
            C_mat[2, 0] = lam;        C_mat[2, 1] = lam;        C_mat[2, 2] = lam + 2*mu
            C_mat[3, 3] = mu
            C_mat[4, 4] = mu
            C_mat[5, 5] = mu
        else:
            # Local Newton return mapping for arbitrary tabular curve
            d_gamma = 0.0
            eq_p_new = eq_p_old
            H_eff = H_cur
            for _ in range(25):
                sy_eval, H_eff = _interp_tabular_hardening(eq_p_new, n_pts, table_sy, table_ep)
                res = q_trial - 3.0 * mu * d_gamma - sy_eval
                denom = 3.0 * mu + H_eff
                d_dgamma = res / max(denom, 1e-12)
                d_gamma += d_dgamma
                eq_p_new = eq_p_old + d_gamma
                if abs(d_dgamma) < 1e-12:
                    break

            scale_s = 1.0 - (3.0 * mu * d_gamma) / max(q_trial, 1e-12)
            s_new = s_trial * scale_s

            # Check failure / damage criterion
            is_failed = 1.0 if (eq_p_new >= eps_fail and eps_fail > 0.0) or is_failed_old > 0.5 else 0.0
            soften = 0.01 if is_failed > 0.5 else 1.0

            S_voigt = (s_new * soften).copy()
            S_voigt[0] += hydro_p * soften
            S_voigt[1] += hydro_p * soften
            S_voigt[2] += hydro_p * soften

            if len(sdv_new) >= 8:
                n_flow = s_trial / max(norm_s, 1e-12)
                flow_fac = d_gamma * np.sqrt(1.5)
                for i in range(6):
                    sdv_new[i] = ep_old[i] + flow_fac * n_flow[i]
                sdv_new[6] = eq_p_new
                sdv_new[7] = is_failed

            # Consistent elastoplastic tangent
            beta0 = (3.0 * mu * d_gamma) / max(q_trial, 1e-12)
            beta1 = (3.0 * mu) / max(3.0 * mu + H_eff, 1e-12)
            n_vec = s_trial / max(norm_s, 1e-12)

            C_vol = np.zeros((6, 6), dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    C_vol[r, c] = K

            C_dev = np.zeros((6, 6), dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    C_dev[r, c] = - (2.0 / 3.0) * mu
                C_dev[r, r] += 2.0 * mu
            C_dev[3, 3] = mu
            C_dev[4, 4] = mu
            C_dev[5, 5] = mu

            C_mat = (C_vol + (1.0 - beta0) * C_dev - 2.0 * mu * (beta1 - beta0) * np.outer(n_vec, n_vec)) * soften

    elif mat_type == MAT_J2_DUCTILE_DAMAGE:
        # props: [E, nu, sy0, H, init_type, p0..p4, Gf, L, evol_type,
        #         T, Troom, Tmelt, eps0 (13..16, solver-maintained temperature)]
        #   init 0 = const eD (p0); init 1 = triax 2-pt (eta0,e0,eta1,e1);
        #   init 2 = JC (D1..D5 triax part + rate (D4) + temperature (D5) terms).
        #   evol 0 = linear D=u/u_f; evol 1 = exponential D=1-exp(-3u/u_f).
        # SDV: [0:6] ep, [6] eq_p, [7] omega_D, [8] u_bar, [9] D.
        E = props[0]
        nu = props[1]
        sigma_y0 = props[2]
        H = props[3]
        init_type = int(props[4])
        Gf = props[10]
        Lch = props[11]
        evol_type = int(props[12])
        if Gf <= 0.0 or Lch <= 0.0 or init_type < 0 or init_type > 2 or evol_type < 0 or evol_type > 1:
            error_flag = 1
            return S_voigt, C_mat, sdv_new, error_flag
        Temp = props[13] if len(props) > 13 else 20.0
        Troom = props[14] if len(props) > 14 else 20.0
        Tmelt = props[15] if len(props) > 15 else 1000.0
        eps0 = props[16] if len(props) > 16 else 1.0
        if eps0 <= 0.0:
            eps0 = 1.0

        mu = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        lam = K - (2.0 / 3.0) * mu

        ep_old = np.zeros(6, dtype=np.float64)
        eq_p_old = 0.0
        omega_old = 0.0
        u_bar_old = 0.0
        D_old = 0.0
        if len(sdv_prev) >= 10:
            for i in range(6):
                ep_old[i] = sdv_prev[i]
            eq_p_old = sdv_prev[6]
            omega_old = sdv_prev[7]
            u_bar_old = sdv_prev[8]
            D_old = sdv_prev[9]

        e_trial = E_voigt - ep_old
        trace_e = e_trial[0] + e_trial[1] + e_trial[2]
        e_dev = np.zeros(6, dtype=np.float64)
        e_dev[0] = e_trial[0] - trace_e / 3.0
        e_dev[1] = e_trial[1] - trace_e / 3.0
        e_dev[2] = e_trial[2] - trace_e / 3.0
        e_dev[3] = e_trial[3]
        e_dev[4] = e_trial[4]
        e_dev[5] = e_trial[5]

        s_trial = np.zeros(6, dtype=np.float64)
        s_trial[0] = 2.0 * mu * e_dev[0]
        s_trial[1] = 2.0 * mu * e_dev[1]
        s_trial[2] = 2.0 * mu * e_dev[2]
        s_trial[3] = mu * e_dev[3]
        s_trial[4] = mu * e_dev[4]
        s_trial[5] = mu * e_dev[5]

        norm_s = np.sqrt(
            s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2 +
            2.0 * (s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2)
        )
        q_trial = np.sqrt(1.5) * norm_s
        sigma_y_cur = sigma_y0 + H * eq_p_old
        f_trial = q_trial - sigma_y_cur
        hydro_p = K * trace_e

        C_el = np.zeros((6, 6), dtype=np.float64)
        for r in range(3):
            for c in range(3):
                C_el[r, c] = K - (2.0 / 3.0) * mu
            C_el[r, r] += 2.0 * mu
        C_el[3, 3] = mu
        C_el[4, 4] = mu
        C_el[5, 5] = mu

        if f_trial <= 0.0:
            # Damaged elasticity: no healing on unload.
            S_voigt = (1.0 - D_old) * s_trial.copy()
            S_voigt[0] += (1.0 - D_old) * hydro_p
            S_voigt[1] += (1.0 - D_old) * hydro_p
            S_voigt[2] += (1.0 - D_old) * hydro_p
            C_mat = (1.0 - D_old) * C_el
        else:
            d_gamma = f_trial / (3.0 * mu + H)
            eq_p_new = eq_p_old + d_gamma
            scale_s = 1.0 - (3.0 * mu * d_gamma) / max(q_trial, 1e-12)
            s_tilde = s_trial * scale_s
            q_new = q_trial - 3.0 * mu * d_gamma

            # Initiation on the undamaged updated stress.
            eta = -hydro_p / max(q_new, 1e-12)
            if init_type == 0:
                eD = props[5]
            elif init_type == 1:
                if eta <= props[5]:
                    eD = props[6]
                elif eta >= props[7]:
                    eD = props[8]
                else:
                    eD = props[6] + (props[8] - props[6]) * (eta - props[5]) / max(props[7] - props[5], 1e-300)
            else:
                eD_triax = props[5] + props[6] * np.exp(props[7] * eta)
                rate = d_gamma / dt if dt > 0.0 else 0.0
                if rate > 0.0 and props[8] != 0.0:
                    eD_triax *= 1.0 + props[8] * np.log(max(rate / eps0, 1e-300))
                if props[9] != 0.0:
                    tstar = (Temp - Troom) / max(Tmelt - Troom, 1e-300)
                    if tstar < 0.0:
                        tstar = 0.0
                    elif tstar > 1.0:
                        tstar = 1.0
                    eD_triax *= 1.0 + props[9] * tstar
                # D4 log term can flip the sign at rate < eps0; a non-positive
                # fracture strain is non-physical, so floor it at eD_triax/100.
                if eD_triax <= 0.0:
                    eD_triax = max(props[5] + props[6] * np.exp(props[7] * eta), 1e-300) * 0.01
                eD = eD_triax
            omega_new = omega_old + d_gamma / max(eD, 1e-300)

            u_f = 2.0 * Gf / max(sigma_y0, 1e-300)
            u_bar_new = u_bar_old
            D_new = D_old
            dDdu = 0.0
            if D_old < 1.0 and omega_new >= 1.0:
                u_bar_new = u_bar_old + Lch * d_gamma
                if evol_type == 0:
                    D_new = u_bar_new / u_f
                    dDdu = 1.0 / u_f
                else:
                    D_new = 1.0 - np.exp(-3.0 * u_bar_new / u_f)
                    dDdu = 3.0 * (1.0 - D_new) / u_f
            failed = 1.0 if D_new >= 1.0 or D_old >= 1.0 else 0.0
            if failed > 0.5:
                D_new = 1.0
                dDdu = 0.0
            soften = 0.01 if failed > 0.5 else 1.0

            S_voigt = (soften * (1.0 - D_new)) * s_tilde.copy()
            S_voigt[0] += soften * (1.0 - D_new) * hydro_p
            S_voigt[1] += soften * (1.0 - D_new) * hydro_p
            S_voigt[2] += soften * (1.0 - D_new) * hydro_p

            if len(sdv_new) >= 10:
                n_flow = s_trial / max(norm_s, 1e-12)
                flow_fac = d_gamma * np.sqrt(1.5)
                for i in range(6):
                    sdv_new[i] = ep_old[i] + flow_fac * n_flow[i]
                sdv_new[6] = eq_p_new
                sdv_new[7] = omega_new
                sdv_new[8] = u_bar_new
                sdv_new[9] = D_new

            # Consistent tangent: degraded J2 + softening outer term.
            beta0 = (3.0 * mu * d_gamma) / max(q_trial, 1e-12)
            beta1 = (3.0 * mu) / (3.0 * mu + H)
            n_vec = s_trial / max(norm_s, 1e-12)

            C_vol = np.zeros((6, 6), dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    C_vol[r, c] = K

            C_dev = np.zeros((6, 6), dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    C_dev[r, c] = - (2.0 / 3.0) * mu
                C_dev[r, r] += 2.0 * mu
            C_dev[3, 3] = mu
            C_dev[4, 4] = mu
            C_dev[5, 5] = mu

            C_ep = C_vol + (1.0 - beta0) * C_dev - 2.0 * mu * (beta1 - beta0) * np.outer(n_vec, n_vec)
            dgam_de = beta1 * np.sqrt(2.0 / 3.0) * n_vec
            dDde = dDdu * Lch * dgam_de
            sig_tilde = s_tilde.copy()
            sig_tilde[0] += hydro_p
            sig_tilde[1] += hydro_p
            sig_tilde[2] += hydro_p
            C_mat = soften * ((1.0 - D_new) * C_ep - np.outer(sig_tilde, dDde))

    elif mat_type == MAT_DRUCKER_PRAGER:
        # Linear Drucker-Prager, Abaqus shear criterion, K=1 (t=q, Mises circle):
        #   f = q + h*tanb - (d0 + H*eq),  g = q + h*tanpsi  (h tension-positive mean)
        # props: [E, nu, d0, H, beta_deg, psi_deg, K_lode (6th slot)]. beta=0
        # reduces to J2 exactly. K_lode=1 is the Mises circle; K<1 flattens
        # triaxial (omega=-30 deg). SDV: [0:6] ep, [6] eq_p. Apex return.
        E = props[0]
        nu = props[1]
        d0 = props[2]
        H = props[3]
        tanb = np.tan(props[4] * np.pi / 180.0)
        tanpsi = np.tan(props[5] * np.pi / 180.0)

        mu = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))

        ep_old = np.zeros(6, dtype=np.float64)
        eq_p_old = 0.0
        if len(sdv_prev) >= 7:
            for i in range(6):
                ep_old[i] = sdv_prev[i]
            eq_p_old = sdv_prev[6]

        e_trial = E_voigt - ep_old
        trace_e = e_trial[0] + e_trial[1] + e_trial[2]
        e_dev = np.zeros(6, dtype=np.float64)
        e_dev[0] = e_trial[0] - trace_e / 3.0
        e_dev[1] = e_trial[1] - trace_e / 3.0
        e_dev[2] = e_trial[2] - trace_e / 3.0
        e_dev[3] = e_trial[3]
        e_dev[4] = e_trial[4]
        e_dev[5] = e_trial[5]

        s_trial = np.zeros(6, dtype=np.float64)
        s_trial[0] = 2.0 * mu * e_dev[0]
        s_trial[1] = 2.0 * mu * e_dev[1]
        s_trial[2] = 2.0 * mu * e_dev[2]
        s_trial[3] = mu * e_dev[3]
        s_trial[4] = mu * e_dev[4]
        s_trial[5] = mu * e_dev[5]

        norm_s = np.sqrt(
            s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2 +
            2.0 * (s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2)
        )
        q_trial = np.sqrt(1.5) * norm_s
        hydro_tr = K * trace_e

        # Deviatoric Lode support is deliberately disabled (MAT14 slot 6
        # reserved, unused): the Mises-circle shear criterion is the verified
        # branch. Generalizing to t-quantile needs the Menetrey-Willig
        # substitution checked against the closed-form return map separately.
        f_trial = q_trial + hydro_tr * tanb - (d0 + H * eq_p_old)

        C_el = np.zeros((6, 6), dtype=np.float64)
        for r in range(3):
            for c in range(3):
                C_el[r, c] = K - (2.0 / 3.0) * mu
            C_el[r, r] += 2.0 * mu
        C_el[3, 3] = mu
        C_el[4, 4] = mu
        C_el[5, 5] = mu

        if f_trial <= 0.0:
            S_voigt = s_trial.copy()
            S_voigt[0] += hydro_tr
            S_voigt[1] += hydro_tr
            S_voigt[2] += hydro_tr
            C_mat = C_el.copy()
        else:
            D0 = 3.0 * mu + K * tanpsi * tanb + H
            if D0 <= 0.0:
                error_flag = 1
                return S_voigt, C_mat, sdv_new, error_flag
            d_lam = f_trial / D0
            q_new = q_trial - 3.0 * mu * d_lam
            n_vec = s_trial / max(norm_s, 1e-12)
            if q_new >= 0.0:
                ratio = q_new / max(q_trial, 1e-12)
                s_new = s_trial * ratio
                hydro_new = hydro_tr - K * tanpsi * d_lam
                eq_p_new = eq_p_old + d_lam
                S_voigt = s_new.copy()
                S_voigt[0] += hydro_new
                S_voigt[1] += hydro_new
                S_voigt[2] += hydro_new
                if len(sdv_new) >= 7:
                    flow_fac = d_lam * np.sqrt(1.5)
                    for i in range(6):
                        sdv_new[i] = ep_old[i] + flow_fac * n_vec[i]
                    sdv_new[0] += d_lam * tanpsi / 3.0
                    sdv_new[1] += d_lam * tanpsi / 3.0
                    sdv_new[2] += d_lam * tanpsi / 3.0
                    sdv_new[6] = eq_p_new
                # Consistent tangent: J2 radial part + pressure coupling.
                # dDlam = A*(n.de) + B*tr(de), A=Q1/D0, B=tb*K/D0, Q1=2m*sqrt1.5.
                Q1 = 2.0 * mu * np.sqrt(1.5)
                Acoef = Q1 / D0
                Bcoef = tanb * K / D0
                qq = max(q_trial * q_trial, 1e-24)
                cn = ((Q1 - 3.0 * mu * Acoef) * q_trial - q_new * Q1) / qq
                ct = -3.0 * mu * Bcoef / max(q_trial, 1e-12)
                one3 = np.zeros(6, dtype=np.float64)
                one3[0] = 1.0
                one3[1] = 1.0
                one3[2] = 1.0
                wrow = cn * n_vec + ct * one3
                vcol = -K * tanpsi * Acoef * n_vec - K * tanpsi * Bcoef * one3
                C_dev = np.zeros((6, 6), dtype=np.float64)
                for r in range(3):
                    for c in range(3):
                        C_dev[r, c] = - (2.0 / 3.0) * mu
                    C_dev[r, r] += 2.0 * mu
                C_dev[3, 3] = mu
                C_dev[4, 4] = mu
                C_dev[5, 5] = mu
                C_vol = np.zeros((6, 6), dtype=np.float64)
                for r in range(3):
                    for c in range(3):
                        C_vol[r, c] = K
                C_mat = C_vol + ratio * C_dev + np.outer(s_trial, wrow) + np.outer(one3, vcol)
            else:
                # Apex return (q=0): linear 1D solve for eq, exact volumetric
                # consistency via ep update, purely volumetric tangent.
                if tanb <= 0.0:
                    error_flag = 1
                    return S_voigt, C_mat, sdv_new, error_flag
                denom = K * tanpsi * tanb + H
                if denom <= 0.0:
                    error_flag = 1
                    return S_voigt, C_mat, sdv_new, error_flag
                eq_new = (hydro_tr * tanb - d0 + K * tanpsi * tanb * eq_p_old) / denom
                hydro_new = (d0 + H * eq_new) / tanb
                S_voigt = np.zeros(6, dtype=np.float64)
                S_voigt[0] = hydro_new
                S_voigt[1] = hydro_new
                S_voigt[2] = hydro_new
                if len(sdv_new) >= 7:
                    dtr = (hydro_tr - hydro_new) / max(K, 1e-300)
                    for i in range(6):
                        sdv_new[i] = ep_old[i]
                    sdv_new[0] += dtr / 3.0
                    sdv_new[1] += dtr / 3.0
                    sdv_new[2] += dtr / 3.0
                    sdv_new[6] = eq_new
                kfac = K * (1.0 - K * tanpsi * tanb / denom)
                C_mat = np.zeros((6, 6), dtype=np.float64)
                for r in range(3):
                    for c in range(3):
                        C_mat[r, c] = kfac

    elif mat_type == MAT_CONCRETE_DP:
        # Concrete damaged plasticity on a DP backbone (Lee-Fenves spirit):
        #   f = q + h*tanb - ((1-r)*sigc(ec) + r*sigt(et)), r = tension weight
        # props: [E,nu,beta,psi, nc,(ec,sig)xnc, nt,(et,sig)xnt,
        #         ndt,(et,dt)xndt, ndc,(ec,dc)xndc], each count <= 6.
        # SDV: [0:6] ep, [6] ec_in, [7] et_in, [8] dt, [9] dc.
        # Lubliner Lode surface and cyclic stiffness recovery are flagged
        # extensions, not implemented here.
        E = props[0]
        nu = props[1]
        tanb = np.tan(props[2] * np.pi / 180.0)
        tanpsi = np.tan(props[3] * np.pi / 180.0)
        mu = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))

        ep_old = np.zeros(6, dtype=np.float64)
        ec_old = 0.0
        et_old = 0.0
        if len(sdv_prev) >= 10:
            for i in range(6):
                ep_old[i] = sdv_prev[i]
            ec_old = sdv_prev[6]
            et_old = sdv_prev[7]

        e_trial = E_voigt - ep_old
        trace_e = e_trial[0] + e_trial[1] + e_trial[2]
        e_dev = np.zeros(6, dtype=np.float64)
        e_dev[0] = e_trial[0] - trace_e / 3.0
        e_dev[1] = e_trial[1] - trace_e / 3.0
        e_dev[2] = e_trial[2] - trace_e / 3.0
        e_dev[3] = e_trial[3]
        e_dev[4] = e_trial[4]
        e_dev[5] = e_trial[5]

        s_trial = np.zeros(6, dtype=np.float64)
        s_trial[0] = 2.0 * mu * e_dev[0]
        s_trial[1] = 2.0 * mu * e_dev[1]
        s_trial[2] = 2.0 * mu * e_dev[2]
        s_trial[3] = mu * e_dev[3]
        s_trial[4] = mu * e_dev[4]
        s_trial[5] = mu * e_dev[5]

        norm_s = np.sqrt(
            s_trial[0]**2 + s_trial[1]**2 + s_trial[2]**2 +
            2.0 * (s_trial[3]**2 + s_trial[4]**2 + s_trial[5]**2)
        )
        q_trial = np.sqrt(1.5) * norm_s
        hydro_tr = K * trace_e

        # Table offsets.
        nc = int(props[4])
        b1 = 5
        nt = int(props[b1 + 2 * nc])
        b2 = b1 + 2 * nc + 1
        ndt = int(props[b2 + 2 * nt])
        b3 = b2 + 2 * nt + 1
        ndc = int(props[b3 + 2 * ndt])
        b4 = b3 + 2 * ndt + 1
        if nc < 1 or nc > 6 or nt < 1 or nt > 6 or ndt < 1 or ndt > 6 or ndc < 1 or ndc > 6:
            error_flag = 1
            return S_voigt, C_mat, sdv_new, error_flag
        if b4 + 2 * ndc > 64:
            error_flag = 1
            return S_voigt, C_mat, sdv_new, error_flag

        # Piecewise-linear table lookup with end slopes (clamped values).
        # Compression hardening (ec,sig).
        i = 0
        while i < nc - 1 and ec_old > props[b1 + 2 * (i + 1)]:
            i += 1
        if nc == 1:
            sigc = props[b1 + 1]
            Hc = 0.0
        elif i >= nc - 1:
            e0 = props[b1 + 2 * (nc - 2)]
            e1 = props[b1 + 2 * (nc - 1)]
            s0 = props[b1 + 2 * (nc - 2) + 1]
            s1 = props[b1 + 2 * (nc - 1) + 1]
            Hc = (s1 - s0) / max(e1 - e0, 1e-300)
            sigc = s1 + Hc * (ec_old - e1)
        else:
            e0 = props[b1 + 2 * i]
            e1 = props[b1 + 2 * (i + 1)]
            s0 = props[b1 + 2 * i + 1]
            s1 = props[b1 + 2 * (i + 1) + 1]
            Hc = (s1 - s0) / max(e1 - e0, 1e-300)
            sigc = s0 + Hc * (ec_old - e0)
        # Tension softening (et,sig).
        j = 0
        while j < nt - 1 and et_old > props[b2 + 2 * (j + 1)]:
            j += 1
        if nt == 1:
            sigt = props[b2 + 1]
            Ht = 0.0
        elif j >= nt - 1:
            e0 = props[b2 + 2 * (nt - 2)]
            e1 = props[b2 + 2 * (nt - 1)]
            s0 = props[b2 + 2 * (nt - 2) + 1]
            s1 = props[b2 + 2 * (nt - 1) + 1]
            Ht = (s1 - s0) / max(e1 - e0, 1e-300)
            sigt = s1 + Ht * (et_old - e1)
        else:
            e0 = props[b2 + 2 * j]
            e1 = props[b2 + 2 * (j + 1)]
            s0 = props[b2 + 2 * j + 1]
            s1 = props[b2 + 2 * (j + 1) + 1]
            Ht = (s1 - s0) / max(e1 - e0, 1e-300)
            sigt = s0 + Ht * (et_old - e0)

        # Tension weight from trial principal stresses (frozen for the step).
        lam1 = s_trial[0] + hydro_tr
        lam2 = s_trial[1] + hydro_tr
        lam3 = s_trial[2] + hydro_tr
        sh = max(s_trial[3] * s_trial[3] + s_trial[4] * s_trial[4] + s_trial[5] * s_trial[5], 0.0)
        if sh > 0.0:
            rad = np.sqrt(max((lam1 - lam2) ** 2 + 4.0 * sh, 0.0))
            pa = 0.5 * (lam1 + lam2 + rad)
            pb = 0.5 * (lam1 + lam2 - rad)
            lam1 = pa
            lam2 = pb
        pos_sum = 0.0
        abs_sum = 0.0
        if lam1 > 0.0:
            pos_sum += lam1
        if lam2 > 0.0:
            pos_sum += lam2
        if lam3 > 0.0:
            pos_sum += lam3
        abs_sum = abs(lam1) + abs(lam2) + abs(lam3)
        if abs_sum > 1e-300:
            rw = pos_sum / abs_sum
        else:
            rw = 0.5

        sigy = (1.0 - rw) * sigc + rw * sigt
        Heff = (1.0 - rw) * Hc + rw * Ht
        f_trial = q_trial + hydro_tr * tanb - sigy

        C_el = np.zeros((6, 6), dtype=np.float64)
        for r in range(3):
            for c in range(3):
                C_el[r, c] = K - (2.0 / 3.0) * mu
            C_el[r, r] += 2.0 * mu
        C_el[3, 3] = mu
        C_el[4, 4] = mu
        C_el[5, 5] = mu

        if f_trial <= 0.0:
            dt_old = 0.0
            dc_old = 0.0
            if len(sdv_prev) >= 10:
                dt_old = sdv_prev[8]
                dc_old = sdv_prev[9]
            D_old = 1.0 - (1.0 - dt_old) * (1.0 - dc_old)
            S_voigt = (1.0 - D_old) * s_trial.copy()
            S_voigt[0] += (1.0 - D_old) * hydro_tr
            S_voigt[1] += (1.0 - D_old) * hydro_tr
            S_voigt[2] += (1.0 - D_old) * hydro_tr
            C_mat = (1.0 - D_old) * C_el.copy()
        else:
            D0 = 3.0 * mu + K * tanpsi * tanb + Heff
            if D0 <= 0.0:
                error_flag = 1
                return S_voigt, C_mat, sdv_new, error_flag
            d_lam = f_trial / D0
            q_new = q_trial - 3.0 * mu * d_lam
            n_vec = s_trial / max(norm_s, 1e-12)
            if q_new < 0.0:
                # Apex return (cone tip): s_new = 0, hydrostatic stress from
                # f = 0 with frozen tables/r. Closed form (linear in eq):
                #   hydro(Deq) = hydro_tr - K*tanpsi*Deq
                #   sigy(Deq)  = sigy + Heff*Deq
                #   hydro*tb - sigy = 0  =>  Deq = (hydro_tr*tb - sigy)/denom_a
                # Volumetric plastic strain follows exactly from hydro_new,
                # deviatoric plastic strain kills the whole trial deviator.
                # Requires tanb > 0 (no apex exists for frictionless flow).
                if tanb <= 0.0:
                    error_flag = 1
                    return S_voigt, C_mat, sdv_new, error_flag
                denom_a = K * tanpsi * tanb + Heff
                if denom_a <= 0.0:
                    error_flag = 1
                    return S_voigt, C_mat, sdv_new, error_flag
                d_lam = (hydro_tr * tanb - sigy) / denom_a
                if d_lam < 0.0:
                    error_flag = 1
                    return S_voigt, C_mat, sdv_new, error_flag
                hydro_new = hydro_tr - K * tanpsi * d_lam
                s_new = np.zeros(6, dtype=np.float64)
                ratio = 0.0
                is_apex = True
                d_et = rw * d_lam
                d_ec = (1.0 - rw) * d_lam
                et_new = et_old + d_et
                ec_new = ec_old + d_ec
            else:
                ratio = q_new / max(q_trial, 1e-12)
                s_new = s_trial * ratio
                hydro_new = hydro_tr - K * tanpsi * d_lam
                d_et = rw * d_lam
                d_ec = (1.0 - rw) * d_lam
                et_new = et_old + d_et
                ec_new = ec_old + d_ec
                is_apex = False
            # Damage tables at the NEW inelastic strains (backward Euler).
            jj = 0
            while jj < ndt - 1 and et_new > props[b3 + 2 * (jj + 1)]:
                jj += 1
            if ndt == 1:
                dt_new = props[b3 + 1]
                dTt = 0.0
            elif jj >= ndt - 1:
                e0 = props[b3 + 2 * (ndt - 2)]
                e1 = props[b3 + 2 * (ndt - 1)]
                s0 = props[b3 + 2 * (ndt - 2) + 1]
                s1 = props[b3 + 2 * (ndt - 1) + 1]
                dTt = (s1 - s0) / max(e1 - e0, 1e-300)
                dt_new = s1 + dTt * (et_new - e1)
            else:
                e0 = props[b3 + 2 * jj]
                e1 = props[b3 + 2 * (jj + 1)]
                s0 = props[b3 + 2 * jj + 1]
                s1 = props[b3 + 2 * (jj + 1) + 1]
                dTt = (s1 - s0) / max(e1 - e0, 1e-300)
                dt_new = s0 + dTt * (et_new - e0)
            kk = 0
            while kk < ndc - 1 and ec_new > props[b4 + 2 * (kk + 1)]:
                kk += 1
            if ndc == 1:
                dc_new = props[b4 + 1]
                dTc = 0.0
            elif kk >= ndc - 1:
                e0 = props[b4 + 2 * (ndc - 2)]
                e1 = props[b4 + 2 * (ndc - 1)]
                s0 = props[b4 + 2 * (ndc - 2) + 1]
                s1 = props[b4 + 2 * (ndc - 1) + 1]
                dTc = (s1 - s0) / max(e1 - e0, 1e-300)
                dc_new = s1 + dTc * (ec_new - e1)
            else:
                e0 = props[b4 + 2 * kk]
                e1 = props[b4 + 2 * (kk + 1)]
                s0 = props[b4 + 2 * kk + 1]
                s1 = props[b4 + 2 * (kk + 1) + 1]
                dTc = (s1 - s0) / max(e1 - e0, 1e-300)
                dc_new = s0 + dTc * (ec_new - e0)
            if dt_new < 0.0:
                dt_new = 0.0
            elif dt_new > 0.999:
                dt_new = 0.999
            if dc_new < 0.0:
                dc_new = 0.0
            elif dc_new > 0.999:
                dc_new = 0.999
            D_new = 1.0 - (1.0 - dt_new) * (1.0 - dc_new)
            S_voigt = (1.0 - D_new) * s_new.copy()
            S_voigt[0] += (1.0 - D_new) * hydro_new
            S_voigt[1] += (1.0 - D_new) * hydro_new
            S_voigt[2] += (1.0 - D_new) * hydro_new
            if len(sdv_new) >= 10:
                if is_apex:
                    # Apex: deviatoric plastic strain kills the whole trial
                    # deviator; volumetric part follows exactly from hydro_new.
                    for i in range(6):
                        sdv_new[i] = ep_old[i] + e_dev[i]
                    tr_tot = E_voigt[0] + E_voigt[1] + E_voigt[2]
                    tr_ep_new = tr_tot - hydro_new / max(K, 1e-300)
                    tr_ep_old = tr_tot - trace_e
                    for i in range(3):
                        sdv_new[i] += (tr_ep_new - tr_ep_old) / 3.0
                else:
                    flow_fac = d_lam * np.sqrt(1.5)
                    for i in range(6):
                        sdv_new[i] = ep_old[i] + flow_fac * n_vec[i]
                    sdv_new[0] += d_lam * tanpsi / 3.0
                    sdv_new[1] += d_lam * tanpsi / 3.0
                    sdv_new[2] += d_lam * tanpsi / 3.0
                sdv_new[6] = ec_new
                sdv_new[7] = et_new
                sdv_new[8] = dt_new
                sdv_new[9] = dc_new
            # Consistent tangent: degraded DP structure + damage chain
            # (frozen r, table end-slopes). Apex: purely volumetric.
            one3 = np.zeros(6, dtype=np.float64)
            one3[0] = 1.0
            one3[1] = 1.0
            one3[2] = 1.0
            if is_apex:
                denom_ax = K * tanpsi * tanb + Heff
                kfac = K * (1.0 - K * tanpsi * tanb / denom_ax)
                C_apex = np.zeros((6, 6), dtype=np.float64)
                for r in range(3):
                    for c in range(3):
                        C_apex[r, c] = kfac
                dDdl_a = (1.0 - dc_new) * dTt * rw + (1.0 - dt_new) * dTc * (1.0 - rw)
                dDde = dDdl_a * (tanb * K / denom_ax) * one3
                sig_tilde = np.zeros(6, dtype=np.float64)
                sig_tilde[0] = hydro_new
                sig_tilde[1] = hydro_new
                sig_tilde[2] = hydro_new
                C_mat = (1.0 - D_new) * C_apex - np.outer(sig_tilde, dDde)
            else:
                Q1 = 2.0 * mu * np.sqrt(1.5)
                Acoef = Q1 / D0
                Bcoef = tanb * K / D0
                qq = max(q_trial * q_trial, 1e-24)
                cn = ((Q1 - 3.0 * mu * Acoef) * q_trial - q_new * Q1) / qq
                ct = -3.0 * mu * Bcoef / max(q_trial, 1e-12)
                wrow = cn * n_vec + ct * one3
                vcol = -K * tanpsi * Acoef * n_vec - K * tanpsi * Bcoef * one3
                C_dev = np.zeros((6, 6), dtype=np.float64)
                for r in range(3):
                    for c in range(3):
                        C_dev[r, c] = - (2.0 / 3.0) * mu
                    C_dev[r, r] += 2.0 * mu
                C_dev[3, 3] = mu
                C_dev[4, 4] = mu
                C_dev[5, 5] = mu
                C_vol = np.zeros((6, 6), dtype=np.float64)
                for r in range(3):
                    for c in range(3):
                        C_vol[r, c] = K
                C_ep = C_vol + ratio * C_dev + np.outer(s_trial, wrow) + np.outer(one3, vcol)
                dDdl = (1.0 - dc_new) * dTt * rw + (1.0 - dt_new) * dTc * (1.0 - rw)
                dDde = dDdl * (Acoef * n_vec + Bcoef * one3)
                sig_tilde = s_new.copy()
                sig_tilde[0] += hydro_new
                sig_tilde[1] += hydro_new
                sig_tilde[2] += hydro_new
                C_mat = (1.0 - D_new) * C_ep - np.outer(sig_tilde, dDde)

    elif mat_type == MAT_HYPER_VISCOELASTIC:
        # Finite-strain Hyperelastic base + Multi-term Prony series (Simo 1987)
        # props: [base_model (1=NH, 4=AB, 5=Yeoh, 8=MR), p0, p1, p2, n_terms, g1, tau1, g2, tau2, ..., g6, tau6]
        base_model = int(props[0])
        p0 = props[1]
        p1 = props[2]
        p2 = props[3]
        n_terms = int(props[4])
        if n_terms < 1:
            n_terms = 1
        if n_terms > 6:
            n_terms = 6

        # 1. Compute base instantaneous hyperelastic stress S_base_el
        if base_model == MAT_HYPERELASTIC_ARRUDA_BOYCE:
            mu = p0
            lambda_m = p1 if p1 > 0.0 else 3.0
            K = p2
            S_base_el = _arruda_boyce_pk2_voigt(E_voigt, mu, lambda_m, K)
        elif base_model == MAT_HYPERELASTIC_YEOH:
            c10 = p0; c20 = p1; c30 = p2
            d1 = props[5] if len(props) > 5 else 0.01
            d2 = 0.0; d3 = 0.0
            S_base_el = _yeoh_pk2_voigt(E_voigt, c10, c20, c30, d1, d2, d3)
        elif base_model == MAT_HYPERELASTIC_MOONEY_RIVLIN:
            c10 = p0; c01 = p1; d1 = p2
            S_base_el = _mooney_rivlin_pk2_voigt(E_voigt, c10, c01, d1)
        else:
            # Default to Neo-Hookean base
            mu = p0
            K = p1
            S_base_el = _neohookean_pk2_voigt(E_voigt, mu, K)

        # 2. Extract C and Volumetric / Deviatoric split of S_base_el
        C = np.empty((3, 3), dtype=np.float64)
        C[0, 0] = 1.0 + 2.0 * E_voigt[0]
        C[1, 1] = 1.0 + 2.0 * E_voigt[1]
        C[2, 2] = 1.0 + 2.0 * E_voigt[2]
        C[0, 1] = C[1, 0] = E_voigt[3]
        C[1, 2] = C[2, 1] = E_voigt[4]
        C[2, 0] = C[0, 2] = E_voigt[5]

        J, detC, Cinv, trC = _inv_and_det_c(C)

        # Volumetric stress formula: S_vol = K * ln(J) * Cinv (Simo & Hughes 1998)
        lnJ = np.log(max(J, 1e-12))
        vol_k = p2 if base_model in [MAT_HYPERELASTIC_ARRUDA_BOYCE, MAT_HYPERELASTIC_MOONEY_RIVLIN] else p1
        vol_scale = vol_k * lnJ

        S_vol_voigt = np.empty(6, dtype=np.float64)
        S_vol_voigt[0] = vol_scale * Cinv[0, 0]
        S_vol_voigt[1] = vol_scale * Cinv[1, 1]
        S_vol_voigt[2] = vol_scale * Cinv[2, 2]
        S_vol_voigt[3] = vol_scale * Cinv[0, 1]
        S_vol_voigt[4] = vol_scale * Cinv[1, 2]
        S_vol_voigt[5] = vol_scale * Cinv[2, 0]

        S_dev_el = S_base_el - S_vol_voigt

        # 3. Retrieve previous state: sdv layout [h_1 (6) .. h_6 (6), S_dev_prev (6)] (size 42)
        S_dev_prev = np.zeros(6, dtype=np.float64)
        if sdv_prev.shape[0] >= 42:
            for c in range(6):
                S_dev_prev[c] = sdv_prev[36 + c]
        elif sdv_prev.shape[0] >= 30:
            for c in range(6):
                S_dev_prev[c] = sdv_prev[24 + c]

        delta_S_dev = S_dev_el - S_dev_prev

        # 4. Multi-term Maxwell evolution (M <= 6)
        dt_safe = max(dt, 1e-12)
        sum_g = 0.0
        S_overstress_sum = np.zeros(6, dtype=np.float64)
        h_all_new = np.zeros((6, 6), dtype=np.float64)
        gamma_sum_factor = 0.0

        for m in range(n_terms):
            idx_g = 5 + 2 * m
            idx_tau = 6 + 2 * m
            g_m = props[idx_g]
            tau_m = max(props[idx_tau], 1e-12)
            sum_g += g_m

            xi_m = dt_safe / tau_m
            if xi_m < 1e-12:
                beta_m = 1.0 - xi_m
                gamma_m = 1.0
            else:
                beta_m = np.exp(-xi_m)
                gamma_m = (1.0 - beta_m) / xi_m

            gamma_sum_factor += g_m * gamma_m

            h_m_old = np.zeros(6, dtype=np.float64)
            if sdv_prev.shape[0] >= 42:
                for c in range(6):
                    h_m_old[c] = sdv_prev[m * 6 + c]
            elif sdv_prev.shape[0] >= 30 and m < 4:
                for c in range(6):
                    h_m_old[c] = sdv_prev[m * 6 + c]

            for c in range(6):
                h_m_new = beta_m * h_m_old[c] + g_m * gamma_m * delta_S_dev[c]
                h_all_new[m, c] = h_m_new
                S_overstress_sum[c] += h_m_new

        # Save updated SDVs
        if sdv_new.shape[0] >= 42:
            for m in range(n_terms):
                for c in range(6):
                    sdv_new[m * 6 + c] = h_all_new[m, c]
            for c in range(6):
                sdv_new[36 + c] = S_dev_el[c]
        elif sdv_new.shape[0] >= 30:
            for m in range(min(n_terms, 4)):
                for c in range(6):
                    sdv_new[m * 6 + c] = h_all_new[m, c]
            for c in range(6):
                sdv_new[24 + c] = S_dev_el[c]

        g_inf = 1.0 - sum_g
        S_voigt = S_vol_voigt + g_inf * S_dev_el + S_overstress_sum

        # 5. Algorithmic Tangent Modulus
        # C_tangent = C_vol + (g_inf + gamma_sum_factor) * C_dev_el
        sqrt_eps = 1.4901161193847656e-08
        C_base = np.zeros((6, 6), dtype=np.float64)
        for j in range(6):
            h = sqrt_eps * max(abs(E_voigt[j]), 1.0)
            E_p = E_voigt.copy()
            E_p[j] += h
            E_m = E_voigt.copy()
            E_m[j] -= h
            if base_model == MAT_HYPERELASTIC_ARRUDA_BOYCE:
                Sp = _arruda_boyce_pk2_voigt(E_p, mu, lambda_m, K)
                Sm = _arruda_boyce_pk2_voigt(E_m, mu, lambda_m, K)
            elif base_model == MAT_HYPERELASTIC_YEOH:
                Sp = _yeoh_pk2_voigt(E_p, c10, c20, c30, d1, d2, d3)
                Sm = _yeoh_pk2_voigt(E_m, c10, c20, c30, d1, d2, d3)
            elif base_model == MAT_HYPERELASTIC_MOONEY_RIVLIN:
                Sp = _mooney_rivlin_pk2_voigt(E_p, c10, c01, d1)
                Sm = _mooney_rivlin_pk2_voigt(E_m, c10, c01, d1)
            else:
                Sp = _neohookean_pk2_voigt(E_p, mu, K)
                Sm = _neohookean_pk2_voigt(E_m, mu, K)
            for i in range(6):
                C_base[i, j] = (Sp[i] - Sm[i]) / (2.0 * h)

        # Scale deviatoric response by (g_inf + gamma_sum_factor)
        factor_visco = g_inf + gamma_sum_factor
        for r in range(6):
            for c in range(6):
                C_mat[r, c] = factor_visco * C_base[r, c]
        # Restore full volumetric tangent
        vol_boost = (1.0 - factor_visco) * vol_k
        for r in range(3):
            for c in range(3):
                C_mat[r, c] += vol_boost

    elif mat_type == MAT_CUSTOM_ELASTIC:
        # props: flattened 6x6 matrix (36 values)
        for i in range(6):
            for j in range(6):
                C_mat[i, j] = props[6 * i + j]
        S_voigt = C_mat @ E_voigt

    else:
        # Unsupported material type fallback
        error_flag = 2

    return S_voigt, C_mat, sdv_new, error_flag


@njit(fastmath=True)
def material_dispatch_2d(
    mat_type: int,
    props: np.ndarray,
    sdv_prev: np.ndarray,
    eps_2d: np.ndarray,
    dt: float = 0.0
):
    """
    2D Plane Strain Material Dispatcher (Abaqus UMAT parity).
    Computes 2D Cauchy/PK2 stress vector [s_xx, s_yy, s_xy] and 3x3 consistent tangent C_mat_2d.
    Supports Linear Elasticity, J2 Plasticity (Linear/Nonlinear/Tabular), etc.
    """
    # Fast path for linear elastic
    # Only id 0 is linear elastic. Id 1 is MAT_HYPERELASTIC_NEOHOOKEAN in the
    # shared table; it used to be caught here too, so every 2D Neo-Hooke
    # material silently ran as linear elastic (keyword audit 2026-10-04).
    if mat_type == MAT_LINEAR_ELASTIC:
        E = props[0]
        nu = props[1]
        c_fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        C11 = c_fac * (1.0 - nu)
        C12 = c_fac * nu
        C33 = c_fac * 0.5 * (1.0 - 2.0 * nu)

        sigma = np.zeros(3, dtype=np.float64)
        sigma[0] = C11 * eps_2d[0] + C12 * eps_2d[1]
        sigma[1] = C12 * eps_2d[0] + C11 * eps_2d[1]
        sigma[2] = C33 * eps_2d[2]

        C_mat_2d = np.zeros((3, 3), dtype=np.float64)
        C_mat_2d[0, 0] = C11
        C_mat_2d[0, 1] = C12
        C_mat_2d[1, 0] = C12
        C_mat_2d[1, 1] = C11
        C_mat_2d[2, 2] = C33
        return sigma, C_mat_2d, sdv_prev, 0

    # General nonlinear path via 3D dispatch
    E_voigt = np.zeros(6, dtype=np.float64)
    E_voigt[0] = eps_2d[0]
    E_voigt[1] = eps_2d[1]
    E_voigt[3] = eps_2d[2]

    F_dummy = np.eye(3, dtype=np.float64)
    detF_dummy = 1.0

    S_voigt, C_6x6, sdv_new, err = material_dispatch_3d(
        mat_type, props, sdv_prev, E_voigt, F_dummy, detF_dummy, dt
    )

    sigma = np.zeros(3, dtype=np.float64)
    sigma[0] = S_voigt[0]
    sigma[1] = S_voigt[1]
    sigma[2] = S_voigt[3]

    C_mat_2d = np.zeros((3, 3), dtype=np.float64)
    C_mat_2d[0, 0] = C_6x6[0, 0]
    C_mat_2d[0, 1] = C_6x6[0, 1]
    C_mat_2d[0, 2] = C_6x6[0, 3]

    C_mat_2d[1, 0] = C_6x6[1, 0]
    C_mat_2d[1, 1] = C_6x6[1, 1]
    C_mat_2d[1, 2] = C_6x6[1, 3]

    C_mat_2d[2, 0] = C_6x6[3, 0]
    C_mat_2d[2, 1] = C_6x6[3, 1]
    C_mat_2d[2, 2] = C_6x6[3, 3]

    return sigma, C_mat_2d, sdv_new, err

