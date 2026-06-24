"""Quick EAS verification: patch test + bending (shear-locking) benchmark."""
import numpy as np
from dispsolver.element import q4
from dispsolver.element.q4_eas import compute_eas_linear_K

E, nu = 1000.0, 0.3
D = q4.plane_strain_D(E, nu)

# ---- 1. Patch test: a parallelogram element under uniform strain ----
# A constant-strain (linear displacement) field must be reproduced exactly,
# i.e. EAS stiffness must equal an exactly-integrated constant-strain response.
coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.5, 1.0], [0.5, 1.0]])
K_eas = compute_eas_linear_K(coords, D)

# Impose a linear displacement field u = a + Gx ; constant strain -> residual 0
G = np.array([[1e-3, 2e-4], [3e-4, -5e-4]])  # arbitrary const displacement grad
u = np.zeros(8)
for i in range(4):
    x, y = coords[i]
    u[2 * i] = G[0, 0] * x + G[0, 1] * y
    u[2 * i + 1] = G[1, 0] * x + G[1, 1] * y
f = K_eas @ u
# Expected nodal forces from exact constant stress
eps = np.array([G[0, 0], G[1, 1], G[0, 1] + G[1, 0]])
sig = D @ eps
f_exact = np.zeros(8)
for k in range(4):
    xi, eta = q4._GP2[k]
    _, detJ, invJ = q4.jacobian(xi, eta, coords)
    B = q4.B_matrix(xi, eta, invJ)
    f_exact += B.T @ sig * detJ * q4._W2[k]
patch_err = np.linalg.norm(f - f_exact) / (np.linalg.norm(f_exact) + 1e-30)
print(f"[Patch test] residual rel.err = {patch_err:.3e}  (pass if < 1e-10)")

# ---- 2. Cantilever bending: 1 element through thickness ----
# Thin beam L=10, h=1, tip moment. Compare tip deflection of standard B-bar Q4
# vs EAS. EAS should be far closer to Euler-Bernoulli (locking removed).
def beam_tip_disp(n_elem_x, use_eas):
    L, h = 10.0, 1.0
    nx = n_elem_x + 1
    xs = np.linspace(0, L, nx)
    ndof = nx * 2 * 2  # 2 rows of nodes
    K = np.zeros((ndof, ndof))

    def nid(i, j):  # i along x, j in {0,1} bottom/top
        return j * nx + i

    for e in range(n_elem_x):
        n1, n2 = nid(e, 0), nid(e + 1, 0)
        n3, n4 = nid(e + 1, 1), nid(e, 1)
        ec = np.array([[xs[e], 0], [xs[e + 1], 0], [xs[e + 1], h], [xs[e], h]])
        Ke = compute_eas_linear_K(ec, D) if use_eas else q4.compute_K_elem(ec, E, nu)
        dofs = []
        for n in (n1, n2, n3, n4):
            dofs += [2 * n, 2 * n + 1]
        for a in range(8):
            for b in range(8):
                K[dofs[a], dofs[b]] += Ke[a, b]

    # Clamp x=0 nodes, apply tip shear couple at x=L
    fixed = [2 * nid(0, 0), 2 * nid(0, 0) + 1, 2 * nid(0, 1), 2 * nid(0, 1) + 1]
    F = np.zeros(ndof)
    P = 1.0
    F[2 * nid(n_elem_x, 0) + 1] = -P / 2
    F[2 * nid(n_elem_x, 1) + 1] = -P / 2
    free = [d for d in range(ndof) if d not in fixed]
    Kff = K[np.ix_(free, free)]
    uf = np.linalg.solve(Kff, F[free])
    u = np.zeros(ndof)
    u[free] = uf
    tip = 0.5 * (u[2 * nid(n_elem_x, 0) + 1] + u[2 * nid(n_elem_x, 1) + 1])
    return tip

I = 1.0 ** 3 / 12.0
euler = -1.0 * 10.0 ** 3 / (3.0 * E * I)  # PL^3/3EI cantilever tip (sign down)
for nex in (10,):
    d_std = beam_tip_disp(nex, use_eas=False)
    d_eas = beam_tip_disp(nex, use_eas=True)
    print(f"[Bending nx={nex}] Euler={euler:.4e}  Bbar={d_std:.4e} ({d_std/euler*100:4.1f}%)  "
          f"EAS={d_eas:.4e} ({d_eas/euler*100:4.1f}%)")
