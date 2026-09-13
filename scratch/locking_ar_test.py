"""
Locking check at the REAL free-hinge-span aspect ratios.

Cantilever strip, one material layer, pure end-moment-equivalent tip load,
small deflection so Euler-Bernoulli theory is the exact reference.
Compares tip deflection vs theory for each element formulation at several
element aspect ratios (dx / dy), including the ones the teardrop free span
actually uses (dx=0.25mm, layer row dy=0.02-0.03mm -> AR 8-12.5).

delta_theory = P L^3 / (3 E' I),  E' = E/(1-nu^2) for plane strain,
I = h^3/12 per unit width (thickness=1).
"""
import numpy as np
from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity
from dispsolver.solver import DynamicSolver

E, NU = 4000.0, 0.3


def build_strip(L, h, nx, ny):
    mesh = Mesh()
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, h, ny + 1)
    nid = 1
    ids = {}
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(nid, x, y)
            ids[(j, i)] = nid
            nid += 1
    eid = 1
    for j in range(ny):
        for i in range(nx):
            mesh.add_element(eid, [ids[(j, i)], ids[(j, i + 1)],
                                   ids[(j + 1, i + 1)], ids[(j + 1, i)]], "Q4", 0)
            eid += 1
    return mesh, ids, xs, ys


def run_case(elem_type, L=5.0, h=0.02, nx=20, ny=1, P=1e-6):
    mesh, ids, xs, ys = build_strip(L, h, nx, ny)
    mat = J2Plasticity(E=E, nu=NU, sigma_y0=1e9, H=0.0)  # keep elastic
    solver = DynamicSolver(mesh, material=mat, element_type=elem_type,
                           mode="quasistatic", rho=1e-9, verbose=False)
    solver.sta_status = False
    nid_to_idx = mesh.node_id_to_index()

    fixed = []
    for j in range(ny + 1):
        n = ids[(j, 0)]
        k = nid_to_idx[n]
        fixed += [2 * k, 2 * k + 1]
    solver.set_prescribed_dofs(np.array(fixed, dtype=np.int32),
                               bc_vals=np.zeros(len(fixed)))

    f_ext = np.zeros(solver.n_dofs)
    tip_nodes = [ids[(j, nx)] for j in range(ny + 1)]
    for n in tip_nodes:
        f_ext[2 * nid_to_idx[n] + 1] = P / len(tip_nodes)
    solver.f_ext = f_ext

    code = solver.solve_step(1.0)
    tip = np.mean([solver.u[2 * nid_to_idx[n] + 1] for n in tip_nodes])

    E_ps = E / (1.0 - NU ** 2)
    I = h ** 3 / 12.0
    delta_th = P * L ** 3 / (3.0 * E_ps * I)
    ratio = delta_th / tip if abs(tip) > 1e-30 else float("inf")
    return code, tip, delta_th, ratio


if __name__ == "__main__":
    cases = [
        ("AR=1   (dx=0.02, h=0.02)", dict(L=0.4, h=0.02, nx=20, ny=1)),
        ("AR=4   (dx=0.08, h=0.02)", dict(L=1.6, h=0.02, nx=20, ny=1)),
        ("AR=12.5(dx=0.25, h=0.02)  <- free span PET row", dict(L=5.0, h=0.02, nx=20, ny=1)),
        ("AR=8.3 (dx=0.25, h=0.03)  <- free span PSA row", dict(L=5.0, h=0.03, nx=20, ny=1)),
    ]
    etypes = ["Q4_COROTATIONAL", "Q4_COROTATIONAL_EAS", "Q4_EAS", "Q4_COROTATIONAL_SRI"]
    print(f"{'case':44s} {'element':24s} {'tip_fem':>12s} {'tip_theory':>12s} {'stiffening':>11s}")
    for label, kw in cases:
        for et in etypes:
            try:
                code, tip, th, ratio = run_case(et, **kw)
                flag = "  <== LOCKED" if ratio > 1.5 else ""
                print(f"{label:44s} {et:24s} {tip:12.4e} {th:12.4e} {ratio:10.2f}x{flag}")
            except Exception as e:
                print(f"{label:44s} {et:24s}  ERROR: {type(e).__name__}: {str(e)[:60]}")
