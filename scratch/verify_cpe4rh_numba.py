import numpy as np
import jax.numpy as jnp
from dispsolver.element.q4_visco_hybrid_reduced_jax import compute_single_reduced_hybrid_jax
from dispsolver.element.q4_visco_hybrid_reduced_numba import compute_single_reduced_hybrid_numba

KAPPA = 8.3333
BPARAMS = np.array([0.015614, 3.0])
G_I = np.zeros(0); TAU_I = np.zeros(0); G_INF = 1.0; DT = 1.0
EYE2 = np.eye(2)
F_N_ID = np.stack([EYE2]*4)

def state_zeros():
    return np.zeros((4, 6))

def affine_u(coords, F0):
    H = F0 - np.eye(2)
    u = np.zeros(8)
    for a in range(4):
        d = H @ coords[a]
        u[2*a] = d[0]; u[2*a+1] = d[1]
    return u

UNIT = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])

print("=== Rigid rotation canary (Numba) ===")
for theta in [0.001, 1.0, 30.0, 90.0]:
    th = np.deg2rad(theta)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c,-s],[s,c]])
    u = affine_u(UNIT, R)
    f_e, K_e, s_new, fn_new = compute_single_reduced_hybrid_numba(
        2, UNIT, u, state_zeros(), KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0, F_N_ID)
    print(f"theta={theta:7.3f} deg  max|f_e|={np.max(np.abs(f_e)):.3e}")

print()
print("=== JAX vs Numba force/tangent agreement ===")
for name, F0 in [
    ("uniaxial", np.array([[1.2,0.0],[0.0,0.9]])),
    ("biaxial",  np.array([[1.1,0.0],[0.0,1.1]])),
    ("shear",    np.array([[1.0,0.15],[0.0,1.0]])),
    ("general",  np.array([[0.95,0.08],[0.05,1.08]])),
]:
    u = affine_u(UNIT, F0)
    f_j, K_j, _, _ = compute_single_reduced_hybrid_jax(
        "arruda", jnp.array(UNIT), jnp.array(u), jnp.array(state_zeros()),
        KAPPA, jnp.array(BPARAMS), jnp.array(G_I), jnp.array(TAU_I), G_INF, DT,
        1.0, jnp.array(F_N_ID), 0.0)
    f_n, K_n, _, _ = compute_single_reduced_hybrid_numba(
        2, UNIT, u, state_zeros(), KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0, F_N_ID)
    f_j = np.asarray(f_j); K_j = np.asarray(K_j)
    f_err = np.max(np.abs(f_j - f_n)) / max(np.max(np.abs(f_j)), 1e-12)
    K_err = np.max(np.abs(K_j - K_n)) / max(np.max(np.abs(K_j)), 1e-12)
    print(f"{name:10s}  force rel err={f_err:.3e}  tangent rel err={K_err:.3e}")
