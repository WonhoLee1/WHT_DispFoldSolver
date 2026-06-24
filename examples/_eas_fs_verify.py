"""Finite-strain EAS verification: patch test + consistent-tangent (FD) check."""
import numpy as np
from dispsolver.material.plastic import J2Plasticity
from dispsolver.element.q4_eas import compute_eas_j2_contributions

np.set_printoptions(precision=3, suppress=True, linewidth=140)

# distorted element so the J0-transform matters
coords = np.array([[0.0, 0.0], [1.7, -0.1], [1.9, 1.2], [0.2, 1.0]])

def make_mat(elastic=True):
    # huge yield -> stays elastic (tests finite-strain elastic EAS) or normal yield
    return J2Plasticity(E=2000.0, nu=0.3, sigma_y0=1e9 if elastic else 5.0, H=50.0)

def f_only(u, alpha, mat, state):
    f, K, a_new, s_new = compute_eas_j2_contributions(
        coords, u, alpha, state, mat, {}, thickness=1.0)
    return f, K, a_new, s_new

for elastic in (True, False):
    mat = make_mat(elastic)
    nvars = mat.n_internal_vars
    state = np.tile(mat.initial_internal_vars(), (4, 1))

    # ---- Patch test: homogeneous deformation u = G x ----
    G = np.array([[0.08, 0.03], [-0.02, 0.05]])
    u = np.zeros(8)
    for i in range(4):
        x, y = coords[i]
        u[2 * i] = G[0, 0] * x + G[0, 1] * y
        u[2 * i + 1] = G[1, 0] * x + G[1, 1] * y
    f, K, a_new, _ = f_only(u, np.zeros(4), mat, state)
    print(f"[{'elastic' if elastic else 'plastic'}] patch: |alpha|={np.linalg.norm(a_new):.3e} "
          f"(should be ~0 for homogeneous F)")

    # ---- Consistent tangent via finite difference ----
    rng = np.random.default_rng(0)
    u = 0.05 * rng.standard_normal(8)
    f0, K0, a0, _ = f_only(u, np.zeros(4), mat, state)
    Kfd = np.zeros((8, 8))
    eps = 1e-7
    for j in range(8):
        up = u.copy(); up[j] += eps
        fp, _, _, _ = f_only(up, np.zeros(4), mat, state)
        um = u.copy(); um[j] -= eps
        fm, _, _, _ = f_only(um, np.zeros(4), mat, state)
        Kfd[:, j] = (fp - fm) / (2 * eps)
    err = np.linalg.norm(K0 - Kfd) / (np.linalg.norm(Kfd) + 1e-30)
    print(f"[{'elastic' if elastic else 'plastic'}] tangent FD rel.err = {err:.3e} "
          f"(pass if < 1e-5)")
