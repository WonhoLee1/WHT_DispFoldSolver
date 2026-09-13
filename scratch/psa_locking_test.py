"""
Bending-stiffness (locking) check for the PSA element formulations at the
real free-hinge-span row aspect ratio.

PSA = Arruda-Boyce (mu=0.016779, lambda_m=3.0, K=0.83333) + Prony/WLF,
row thickness 0.03 mm, free-span dx 0.25 mm -> AR ~= 8.3.

Cantilever, tip load applied in N increments (so a flexible strip does not
fail from a single huge step), compared against small-deflection theory
delta = P L^3 / (3 E' I), E' = E/(1-nu^2), E = 2 mu (1+nu), nu from K/mu.
"""
import numpy as np
from dispsolver.mesh import Mesh
from dispsolver.solver import DynamicSolver
from dispsolver.material.factory import build_material_instance
from dispsolver.fold_model_config import FoldModelConfig

MU = 0.016779
K_BULK = 0.83333
NU = (3.0 * K_BULK - 2.0 * MU) / (2.0 * (3.0 * K_BULK + MU))
E_MOD = 2.0 * MU * (1.0 + NU)


def build_strip(L, h, nx, ny):
    mesh = Mesh()
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, h, ny + 1)
    nid, ids = 1, {}
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
    return mesh, ids


def run_case(elem_type, L=2.0, h=0.03, nx=8, ny=1, P=2e-9, n_steps=10):
    cfg = FoldModelConfig()
    mdef = cfg.materials.definitions["PSA"]
    mat, mparams = build_material_instance(mdef)
    mparams = dict(mparams)
    # Q4_UP path wants E/nu present as well
    mparams.setdefault("E", E_MOD)
    mparams.setdefault("nu", NU)
    mesh, ids = build_strip(L, h, nx, ny)
    solver = DynamicSolver(mesh, material=mat, material_params=mparams,
                           element_type=elem_type, mode="quasistatic",
                           rho=1e-9, verbose=False)
    solver.sta_status = False
    n2i = mesh.node_id_to_index()

    fixed = []
    for j in range(ny + 1):
        k = n2i[ids[(j, 0)]]
        fixed += [2 * k, 2 * k + 1]
    solver.set_prescribed_dofs(np.array(fixed, dtype=np.int32), bc_vals=np.zeros(len(fixed)))

    tip_nodes = [ids[(j, nx)] for j in range(ny + 1)]
    codes = []
    for s in range(1, n_steps + 1):
        f_ext = np.zeros(solver.n_dofs)
        for n in tip_nodes:
            f_ext[2 * n2i[n] + 1] = (P * s / n_steps) / len(tip_nodes)
        solver.f_ext = f_ext
        c = solver.solve_step(1.0 / n_steps)
        codes.append(c)
        if c < 0:
            break
    tip = float(np.mean([solver.u[2 * n2i[n] + 1] for n in tip_nodes]))
    E_ps = E_MOD / (1.0 - NU ** 2)
    I = h ** 3 / 12.0
    delta_th = P * L ** 3 / (3.0 * E_ps * I)
    ratio = delta_th / tip if abs(tip) > 1e-30 else float("inf")
    return codes, tip, delta_th, ratio


if __name__ == "__main__":
    print(f"PSA: mu={MU}, K={K_BULK} -> nu={NU:.4f}, E={E_MOD:.6f} MPa")
    for label, kw in [
        ("AR=8.3  (dx=0.25,h=0.03)  free span", dict(L=2.0, h=0.03, nx=8, ny=1)),
        ("AR=2.0  (dx=0.06,h=0.03)  refined  ", dict(L=0.48, h=0.03, nx=8, ny=1)),
        ("AR=1.0  (dx=0.03,h=0.03)  square   ", dict(L=0.24, h=0.03, nx=8, ny=1)),
    ]:
        for et in ("Q4_VISCO_SIMO", "Q4_UP", "Q4_HYBRID_EAS", "Q4_COROTATIONAL_HYBRID_EAS",
                   "Q4_HYBRID_SRI", "Q4_COROTATIONAL_HYBRID_SRI", "Q4_HYBRID",
                   "Q4_COROTATIONAL_SRI", "Q4_COROTATIONAL_EAS"):
            try:
                codes, tip, th, ratio = run_case(et, **kw)
                ok = all(c >= 0 for c in codes)
                flag = "" if ratio < 1.5 else "  <== STIFF"
                print(f"{label} {et:22s} conv={'OK ' if ok else 'FAIL'} tip={tip:11.4e} th={th:11.4e} ratio={ratio:8.2f}{flag}")
            except Exception as e:
                print(f"{label} {et:22s} ERROR {type(e).__name__}: {str(e)[:60]}")
