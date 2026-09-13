"""T2 verification: UL == TL consistency, for every fixed viscoelastic UL kernel."""
import numpy as np
import jax.numpy as jnp

from dispsolver.element.q4_visco_hybrid_reduced_jax import compute_single_reduced_hybrid_jax
from dispsolver.element.q4_visco_eas_jax import compute_single_eas, compute_single_hybrid
from dispsolver.element.q4_visco_simo_fs_jax import compute_single as visco_simo_compute_single

KAPPA = 8.3333
BPARAMS = jnp.array([0.015614, 3.0])
G_I = jnp.array([0.20]); TAU_I = jnp.array([3.33]); G_INF = 0.80; DT = 0.5
EYE2 = jnp.eye(2, dtype=jnp.float64)
UNIT = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
F_N4_ID = jnp.stack([EYE2] * 4)


def affine_u(coords, F0):
    H = F0 - jnp.eye(2, dtype=jnp.float64)
    u = jnp.zeros(8, dtype=jnp.float64)
    for a in range(4):
        d = H @ coords[a]
        u = u.at[2 * a].set(d[0])
        u = u.at[2 * a + 1].set(d[1])
    return u


F_n_case = jnp.array([[1.20, 0.0], [0.0, 0.90]])
F_extra = jnp.array([[1.0, 0.03], [0.0, 1.0]])
F_total = F_extra @ F_n_case
coords_n = jnp.stack([F_n_case @ UNIT[a] for a in range(4)])
u_inc = affine_u(coords_n, F_extra)
u_tl = affine_u(UNIT, F_total)
F_n4_case = jnp.stack([F_n_case] * 4)


def report(name, f_tl, f_ul):
    f_tl = np.asarray(f_tl); f_ul = np.asarray(f_ul)
    err = np.max(np.abs(f_tl - f_ul)) / max(np.max(np.abs(f_tl)), 1e-12)
    print(f"{name:20s} max|f_TL|={np.max(np.abs(f_tl)):.4e}  rel err={err:.4e}")


print("=== T2: UL == TL consistency (after F4 push-forward fix) ===")

# CPE4RH
state6 = jnp.zeros((4, 12))  # M=1 -> 6*(M+1)=12
f_tl, _, _, _ = compute_single_reduced_hybrid_jax(
    "arruda", UNIT, u_tl, state6, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0, F_N4_ID, 0.0)
f_ul, _, _, _ = compute_single_reduced_hybrid_jax(
    "arruda", coords_n, u_inc, state6, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0, F_n4_case, 0.0)
report("CPE4RH", f_tl, f_ul)

# CPE4I (compute_single_eas: coords, u_elem, alpha, state, kappa, bparams, g_i, tau_i, g_inf, dt, thickness, distortion_j_crit, F_n_gps, base)
alpha0 = jnp.zeros(4)
f_tl, _, _, _, _ = compute_single_eas(
    UNIT, u_tl, alpha0, state6, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0,
    distortion_j_crit=0.0, F_n_gps=F_N4_ID, base="arruda")
f_ul, _, _, _, _ = compute_single_eas(
    coords_n, u_inc, alpha0, state6, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0,
    distortion_j_crit=0.0, F_n_gps=F_n4_case, base="arruda")
report("CPE4I", f_tl, f_ul)

# CPE4H (compute_single_hybrid: coords, u_elem, alpha, state, kappa, bparams, g_i, tau_i, g_inf, dt, thickness, distortion_j_crit, F_n_gps, base, use_eas)
q0 = jnp.zeros(5)
f_tl, _, _, _, _ = compute_single_hybrid(
    UNIT, u_tl, q0, state6, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0,
    distortion_j_crit=0.0, F_n_gps=F_N4_ID, base="arruda", use_eas=False)
f_ul, _, _, _, _ = compute_single_hybrid(
    coords_n, u_inc, q0, state6, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0,
    distortion_j_crit=0.0, F_n_gps=F_n4_case, base="arruda", use_eas=False)
report("CPE4H", f_tl, f_ul)

# Q4_VISCO_SIMO (compute_single: coords, u_elem, state, kappa, bparams, g_i, tau_i, g_inf, dt, thickness, distortion_j_crit, F_n_gps, base)
f_tl, _, _, _ = visco_simo_compute_single(
    UNIT, u_tl, state6, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0,
    distortion_j_crit=0.0, F_n_gps=F_N4_ID, base="arruda")
f_ul, _, _, _ = visco_simo_compute_single(
    coords_n, u_inc, state6, KAPPA, BPARAMS, G_I, TAU_I, G_INF, DT, 1.0,
    distortion_j_crit=0.0, F_n_gps=F_n4_case, base="arruda")
report("Q4_VISCO_SIMO", f_tl, f_ul)
