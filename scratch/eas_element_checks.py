"""Element-level theoretical checks for q4_visco_eas_jax (CPE4I).

1. Rigid-body invariance at finite rotation  -> f_int ~ 0, alpha ~ 0
2. Eigen-spectrum of K_e (3 zero modes, none negative), at zero strain and
   under compression (EAS elements are known to go unstable in compression)
3. alpha saturation fraction under increasing pure bending
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from dispsolver.element.q4_visco_eas_jax import compute_single_eas
from dispsolver.fold_model_config import FoldModelConfig
from dispsolver.material.factory import build_material_instance

cfg = FoldModelConfig()
vmat, mp = build_material_instance(cfg.materials.definitions["PSA"])
base, bparams, kappa = vmat.simo_fs_args(mp)
ARGS = (float(kappa), jnp.asarray(bparams), jnp.asarray(vmat.g_i),
        jnp.asarray(vmat.tau_i), float(vmat.g_inf))
NV = 6 * (len(vmat.g_i) + 1)
ST0 = jnp.zeros((4, NV))
A0 = jnp.zeros(4)

# real free-span PSA row: 0.25 x 0.03 mm  (AR = 8.3)
COORDS = jnp.array([[0., 0.], [0.25, 0.], [0.25, 0.03], [0., 0.03]])


def call(u, alpha=A0, coords=COORDS):
    return compute_single_eas(coords, u, alpha, ST0, *ARGS, 1.0, 1.0, base=base)


print("=" * 72)
print("1. RIGID BODY INVARIANCE (finite rotation, zero strain)")
for deg in (5.0, 45.0, 90.0, 180.0):
    th = np.deg2rad(deg)
    R = jnp.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    u = (COORDS @ R.T - COORDS).flatten()
    f, K, a, _ = call(u)
    # scale reference: stiffness * element size
    ref = float(jnp.max(jnp.abs(K))) * 0.25
    print(f"   rot={deg:6.1f}deg  |f_int|={float(jnp.linalg.norm(f)):.3e}"
          f"  (rel {float(jnp.linalg.norm(f))/max(ref,1e-30):.2e})"
          f"  max|alpha|={float(jnp.max(jnp.abs(a))):.3e}")

print()
print("2. EIGEN-SPECTRUM of K_e  (expect 3 zero modes, none negative)")
for label, u in (
    ("undeformed        ", jnp.zeros(8)),
    ("tension  +5%      ", jnp.array([0., 0., .0125, 0., .0125, 0., 0., 0.])),
    ("compression -5%   ", jnp.array([0., 0., -.0125, 0., -.0125, 0., 0., 0.])),
    ("compression -20%  ", jnp.array([0., 0., -.05, 0., -.05, 0., 0., 0.])),
    ("compression -40%  ", jnp.array([0., 0., -.10, 0., -.10, 0., 0., 0.])),
    ("shear      +10%   ", jnp.array([0., 0., 0., 0., .003, 0., .003, 0.])),
):
    f, K, a, _ = call(u)
    w = np.linalg.eigvalsh(np.asarray(0.5 * (K + K.T)))
    kmax = max(abs(w).max(), 1e-30)
    n_zero = int(np.sum(np.abs(w) < 1e-9 * kmax))
    n_neg = int(np.sum(w < -1e-9 * kmax))
    print(f"   {label} zero_modes={n_zero} negative={n_neg} "
          f"min={w[0]:+.3e} max={w[-1]:+.3e} max|alpha|={float(jnp.max(jnp.abs(a))):.3e}")

print()
print("3. ALPHA SATURATION under pure bending (|alpha| cap = 0.05)")
print("   curvature applied as u_x = kappa*x_rel*y_rel on the 4 nodes")
for kap in (1.0, 5.0, 20.0, 50.0, 100.0, 200.0):
    xc, yc = 0.125, 0.015
    u = []
    for (x, y) in np.asarray(COORDS):
        u += [kap * (x - xc) * (y - yc), 0.0]
    u = jnp.array(u)
    f, K, a, _ = call(u)
    sat = float(jnp.max(jnp.abs(a))) / 0.05
    print(f"   kappa={kap:6.1f} 1/mm  max|alpha|={float(jnp.max(jnp.abs(a))):.4e}"
          f"  saturation={sat*100:6.2f}%  |f|={float(jnp.linalg.norm(f)):.3e}")
