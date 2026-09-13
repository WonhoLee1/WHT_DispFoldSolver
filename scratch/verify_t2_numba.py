"""T2 for Numba kernels: UL == TL, plus numba vs jax agreement."""
import numpy as np
from dispsolver.element.q4_visco_hybrid_reduced_numba import compute_single_reduced_hybrid_numba
from dispsolver.element.q4_visco_eas_numba import compute_single_eas_numba
from dispsolver.element.q4_visco_hybrid_up_numba import compute_visco_hybrid_up_single_numba

KAPPA = 8.3333
BPARAMS = np.array([0.015614, 3.0])
G_I = np.array([0.20]); TAU_I = np.array([3.33]); G_INF = 0.80; DT = 0.5
EYE2 = np.eye(2)
UNIT = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])
F_N4_ID = np.stack([EYE2]*4)
N_STATE = 12

def affine_u(coords, F0):
    H = F0 - np.eye(2)
    u = np.zeros(8)
    for a in range(4):
        d = H @ coords[a]
        u[2*a]=d[0]; u[2*a+1]=d[1]
    return u

F_n_case = np.array([[1.20,0.0],[0.0,0.90]])
F_extra = np.array([[1.0,0.03],[0.0,1.0]])
F_total = F_extra @ F_n_case
coords_n = np.stack([F_n_case @ UNIT[a] for a in range(4)])
u_inc = affine_u(coords_n, F_extra)
u_tl = affine_u(UNIT, F_total)
F_n4_case = np.stack([F_n_case]*4)

def report(name, f_tl, f_ul):
    err = np.max(np.abs(f_tl-f_ul))/max(np.max(np.abs(f_tl)),1e-12)
    print(f"{name:20s} max|f_TL|={np.max(np.abs(f_tl)):.4e}  rel err={err:.4e}")

state = np.zeros((4, N_STATE))

# CPE4RH numba
f_tl, _, _, _ = compute_single_reduced_hybrid_numba(2, UNIT, u_tl, state, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0, F_N4_ID)
f_ul, _, _, _ = compute_single_reduced_hybrid_numba(2, coords_n, u_inc, state, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0, F_n4_case)
report("CPE4RH numba", f_tl, f_ul)

# CPE4I numba
alpha0 = np.zeros(4)
f_tl, _, _, _, _ = compute_single_eas_numba(2, UNIT, u_tl, alpha0, state, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0, F_N4_ID)
f_ul, _, _, _, _ = compute_single_eas_numba(2, coords_n, u_inc, alpha0, state, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0, F_n4_case)
report("CPE4I numba", f_tl, f_ul)

# CPE4H numba
f_tl, _, _, _ = compute_visco_hybrid_up_single_numba(2, UNIT, u_tl, state, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0, F_N4_ID)
f_ul, _, _, _ = compute_visco_hybrid_up_single_numba(2, coords_n, u_inc, state, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0, F_n4_case)
report("CPE4H numba", f_tl, f_ul)
