import numpy as np
import jax.numpy as jnp
from dispsolver.element.q4_visco_hybrid_reduced_jax import compute_single_reduced_hybrid_jax
from dispsolver.element.q4_visco_hybrid_reduced_numba import compute_single_reduced_hybrid_numba

KAPPA = 8.3333
BPARAMS = np.array([0.015614, 3.0])
G_I = np.array([0.20]); TAU_I = np.array([3.33]); G_INF = 0.80; DT = 0.5
EYE2 = np.eye(2)

UNIT = np.array([[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]])

def compare(name, coords, u, F_n, state0, tag):
    n_state = 6*(len(G_I)+1)
    state = np.zeros((4, n_state)); state[:] = state0
    f_j, K_j, sn_j, fn_j = compute_single_reduced_hybrid_jax(
        "arruda", jnp.array(coords), jnp.array(u), jnp.array(state),
        KAPPA, jnp.array(BPARAMS), jnp.array(G_I), jnp.array(TAU_I), G_INF, DT,
        1.0, jnp.array(F_n), 0.0)
    f_n, K_n, sn_n, fn_n = compute_single_reduced_hybrid_numba(
        2, coords, u, state, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0, F_n)
    f_j = np.asarray(f_j); K_j = np.asarray(K_j)
    sn_j = np.asarray(sn_j); fn_j = np.asarray(fn_j)
    f_err = np.max(np.abs(f_j - f_n)) / max(np.max(np.abs(f_j)), 1e-12)
    K_err = np.max(np.abs(K_j - K_n)) / max(np.max(np.abs(K_j)), 1e-12)
    s_err = np.max(np.abs(sn_j - sn_n)) / max(np.max(np.abs(sn_j)), 1e-12)
    fn_err = np.max(np.abs(fn_j - fn_n)) / max(np.max(np.abs(fn_j)), 1e-12)
    print(f"{name:28s} [{tag}] f_err={f_err:.3e} K_err={K_err:.3e} state_err={s_err:.3e} F_n_new_err={fn_err:.3e}")

F_N_ID = np.stack([EYE2]*4)

# 1. Random NON-AFFINE nodal displacement (breaks affine symmetry) -- TL
rng = np.random.default_rng(42)
u_rand = rng.uniform(-0.05, 0.05, size=8)
state0 = rng.uniform(-1e-3, 1e-3, size=6*2)
compare("random non-affine u", UNIT, u_rand, F_N_ID, state0, "TL")

# 2. Random non-affine u with nonzero prior Prony state
u_rand2 = rng.uniform(-0.08, 0.08, size=8)
state1 = rng.uniform(-5e-3, 5e-3, size=6*2)
compare("random non-affine u #2", UNIT, u_rand2, F_N_ID, state1, "TL")

# 3. UL case: F_n = a real prior rotation+stretch (not identity), incremental u random
th = np.deg2rad(25.0)
c, s = np.cos(th), np.sin(th)
R25 = np.array([[c,-s],[s,c]])
stretch = np.array([[1.1,0.0],[0.0,0.95]])
F_prior = R25 @ stretch
F_n_rot = np.stack([F_prior]*4)
u_inc = rng.uniform(-0.02, 0.02, size=8)
state2 = rng.uniform(-2e-3, 2e-3, size=6*2)
compare("UL (F_n=rot+stretch), rand u_inc", UNIT, u_inc, F_n_rot, state2, "UL")

# 4. UL + near-collapsed element (severe distortion, stress-tests J_safe/detJ guards)
u_collapse = np.array([0.0,0.0, -0.7,0.05, -0.65,0.02, 0.0,0.0])
state3 = np.zeros(6*2)
compare("near-collapsed element", UNIT, u_collapse, F_N_ID, state3, "TL")

# 5. Irregular (non-unit-square) element geometry, random u
coords_irreg = np.array([[3.1,-0.4],[4.6,-0.1],[4.4,1.3],[3.0,1.1]])
u_irreg = rng.uniform(-0.1, 0.1, size=8)
state4 = rng.uniform(-1e-3,1e-3, size=6*2)
compare("irregular geometry, random u", coords_irreg, u_irreg, F_N_ID, state4, "TL")
