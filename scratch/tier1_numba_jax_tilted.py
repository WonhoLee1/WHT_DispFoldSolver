"""Tier 1: numba vs JAX A/B on a TILTED AND DISTORTED element.

The existing validation (dynamic.py comment: force 1e-13, tangent 2e-6)
was taken on axis-aligned geometry, where BOTH the J0^-1 bug and a wrong
"fix" cancel identically -- it cannot distinguish correct from incorrect.
This runs the same comparison where the transpose actually matters.
"""
import numpy as np, jax.numpy as jnp
from dispsolver.element.q4_visco_eas_jax import compute_single_eas_status
from dispsolver.element.q4_visco_eas_numba import compute_single_eas_numba_status

KAPPA=8.3333; BP=np.array([0.015614,3.0]); GI=np.array([0.20]); TAU=np.array([3.33])
GINF=0.80; DT=0.5; NSTATE=12
EYE=np.eye(2); FN_ID=np.stack([EYE]*4)

def compare(name, coords, u, F_n):
    st = np.zeros((4,NSTATE)); a0=np.zeros(4)
    fj,Kj,aj,sj,fnj,stj = compute_single_eas_status(
        jnp.asarray(coords), jnp.asarray(u), jnp.asarray(a0), jnp.asarray(st),
        KAPPA, jnp.asarray(BP), jnp.asarray(GI), jnp.asarray(TAU), GINF, DT, 1.0,
        distortion_j_crit=0.0, F_n_gps=jnp.asarray(F_n), base="arruda")
    fn_,Kn,an,sn,fnn,stn = compute_single_eas_numba_status(
        2, coords, u, a0, st, KAPPA, BP, GI, TAU, GINF, DT, 1.0, F_n)
    fj=np.asarray(fj); Kj=np.asarray(Kj); aj=np.asarray(aj)
    ferr = np.max(np.abs(fj-fn_))/max(np.max(np.abs(fj)),1e-30)
    Kerr = np.max(np.abs(Kj-np.asarray(Kn)))/max(np.max(np.abs(Kj)),1e-30)
    aerr = np.max(np.abs(aj-np.asarray(an)))
    print(f"{name:34s} f_err={ferr:.3e}  K_err={Kerr:.3e}  alpha_err={aerr:.3e}  max|alpha|={np.max(np.abs(aj)):.3e}  status_jax={float(stj):.2e} status_numba={float(stn):.2e}")

rng = np.random.default_rng(3)
# TILTED + DISTORTED reference element (not a rotated rectangle: a real trapezoid, tilted)
th = np.deg2rad(37.0); c,s = np.cos(th), np.sin(th); R = np.array([[c,-s],[s,c]])
base = np.array([[0.0,0.0],[1.0,0.05],[0.92,0.17],[-0.06,0.13]])   # distorted quad
coords_td = base @ R.T
u = rng.uniform(-1,1,size=8)*8e-3

print("=== Tier 1: numba vs JAX, CPE4I ===")
rect = np.array([[0.,0.],[1.,0.],[1.,0.15],[0.,0.15]])
Fn = np.stack([R @ np.array([[1.1,0.03],[0.0,0.95]])]*4)
for scale,label in [(8e-3,"large (used to saturate the old clamp)"), (8e-4,"moderate"), (8e-5,"small")]:
    uu = u*(scale/8e-3)
    print(f"-- perturbation {label} --")
    compare("  axis-aligned rect", rect, uu, FN_ID)
    compare("  TILTED+DISTORTED TL", coords_td, uu, FN_ID)
    compare("  TILTED+DISTORTED UL", coords_td, uu, Fn)
