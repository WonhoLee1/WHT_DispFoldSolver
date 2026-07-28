"""
fd_jacobian_check.py
====================
Phase 1: FD verification of the global KKT tangent.

Key convention in this solver:
  R = f_ext - f_int  (positive residual)
  J_solver = K_T     (positive stiffness)
  Newton: J_solver * du = R

  FD computes: J_fd = dR/du = -K_T
  So: J_solver = -J_fd  (for the mechanical block)

We account for this sign convention below.
"""

import os
import sys
import numpy as np
import scipy.sparse as sps

os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, LinearViscoelastic
from dispsolver.constraint import RBE2HingeConstraint, PenaltyContactConstraint
from dispsolver.solver import DynamicSolver
from dispsolver.solver import dynamic as dyn_mod


def build_small_mesh():
    mesh = Mesh()
    nx = 8
    xs = np.linspace(-15.0, 15.0, nx)
    ys_list = [0.0]
    row_pids = []
    current_y = 0.0
    for layer in range(3):
        dy = 0.05 / 3.0 if layer % 2 == 0 else 0.05
        for _ in range(3 if layer % 2 == 0 else 1):
            current_y += dy
            ys_list.append(current_y)
            row_pids.append(layer)
    ys = np.array(ys_list)
    ny = len(ys)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(j * nx + i, x, y)
    elem_idx = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i
            mesh.add_element(elem_idx, [n1, n1 + 1, n1 + nx + 1, n1 + nx], "QUAD4", pid=row_pids[j])
            elem_idx += 1
    mesh.add_node(99999, -15.0, 0.0)
    mesh.add_node(99998, 15.0, 0.0)
    return mesh, nx, ny, xs, row_pids


def residual_raw(solver, u, u_ext, lam, dt):
    """Raw KKT residual R = [R_u, R_ext, R_lam] without BC elimination."""
    f_int, _, _ = solver._assemble(u, dt)
    R_u = solver.f_ext.copy() - f_int

    n_extra = solver.n_extra
    n_lambdas = solver.n_lambdas
    R_ext = np.zeros(n_extra)
    R_lam = np.zeros(n_lambdas)

    lam_offset = 0
    for c in solver.constraints:
        n_lam_c = c.n_multipliers()
        if n_lam_c > 0:
            r_u, c_u, v_u, r_ext, c_ext, v_ext, g = c.assemble(u, u_ext)
            for i in range(len(v_u)):
                eq_idx = r_u[i] + lam_offset
                dof_idx = c_u[i]
                R_u[dof_idx] -= v_u[i] * lam[eq_idx]
            for i in range(len(v_ext)):
                eq_idx = r_ext[i] + lam_offset
                ext_idx = c_ext[i]
                R_ext[ext_idx] -= v_ext[i] * lam[eq_idx]
            for i in range(n_lam_c):
                R_lam[lam_offset + i] = -g[i]
        lam_offset += n_lam_c

    return np.concatenate([R_u, R_ext, R_lam])


def jacobian_analytical(solver, u, u_ext, lam, dt):
    """Raw KKT Jacobian matching solve_step's assembly (no BC elimination)."""
    n_dofs = solver.n_dofs
    n_extra = solver.n_extra
    n_total = solver.n_total

    _, K_T, _ = solver._assemble(u, dt)
    K_eff = K_T.copy()

    K_eff_coo = K_eff.tocoo()
    g_row = list(K_eff_coo.row)
    g_col = list(K_eff_coo.col)
    g_val = list(K_eff_coo.data)

    C_row_u, C_col_u, C_val_u = [], [], []
    C_row_ext, C_col_ext, C_val_ext = [], [], []

    lam_offset = 0
    for c in solver.constraints:
        n_lam_c = c.n_multipliers()
        if n_lam_c > 0:
            r_u, c_u, v_u, r_ext, c_ext, v_ext, g = c.assemble(u, u_ext)
            for i in range(len(v_u)):
                C_row_u.append(r_u[i] + lam_offset)
                C_col_u.append(c_u[i])
                C_val_u.append(v_u[i])
            for i in range(len(v_ext)):
                C_row_ext.append(r_ext[i] + lam_offset)
                C_col_ext.append(c_ext[i])
                C_val_ext.append(v_ext[i])
        lam_offset += n_lam_c

    lam_start = n_dofs + n_extra
    ext_start = n_dofs

    for r, c, v in zip(C_row_u, C_col_u, C_val_u):
        g_row.append(r + lam_start); g_col.append(c);          g_val.append(v)
        g_row.append(c);              g_col.append(r + lam_start); g_val.append(v)

    for r, c, v in zip(C_row_ext, C_col_ext, C_val_ext):
        g_row.append(r + lam_start); g_col.append(c + ext_start); g_val.append(v)
        g_row.append(c + ext_start); g_col.append(r + lam_start); g_val.append(v)

    k_extra_geo = np.zeros(n_extra, dtype=np.float64)
    lam_off = 0
    for c in solver.constraints:
        n_lam_c = c.n_multipliers()
        if n_lam_c > 0 and hasattr(c, "extra_geometric_stiffness"):
            off, kval = c.extra_geometric_stiffness(u_ext, lam[lam_off:lam_off + n_lam_c])
            k_extra_geo[off] += kval
        lam_off += n_lam_c
    for i in range(n_extra):
        g_row.append(ext_start + i)
        g_col.append(ext_start + i)
        g_val.append(k_extra_geo[i] + 1e-12)

    return sps.coo_matrix((g_val, (g_row, g_col)), shape=(n_total, n_total)).tocsr()


def jacobian_fd(solver, u, u_ext, lam, dt, h=1e-7):
    """Central-difference Jacobian: J_fd[i,j] = dR_i/dx_j."""
    n_total = solver.n_total
    n_dofs = solver.n_dofs
    n_extra = solver.n_extra

    J_fd = np.zeros((n_total, n_total), dtype=np.float64)

    for j in range(n_total):
        u_p, ue_p, lam_p = u.copy(), u_ext.copy(), lam.copy()
        u_m, ue_m, lam_m = u.copy(), u_ext.copy(), lam.copy()

        if j < n_dofs:
            u_p[j] += h; u_m[j] -= h
        elif j < n_dofs + n_extra:
            ue_p[j - n_dofs] += h; ue_m[j - n_dofs] -= h
        else:
            lam_p[j - n_dofs - n_extra] += h; lam_m[j - n_dofs - n_extra] -= h

        R_p = residual_raw(solver, u_p, ue_p, lam_p, dt)
        R_m = residual_raw(solver, u_m, ue_m, lam_m, dt)
        J_fd[:, j] = (R_p - R_m) / (2.0 * h)

    return J_fd


def compare(J_solver, J_fd, solver, label=""):
    """Compare solver's KKT Jacobian J_solver with FD Jacobian J_fd.

    Convention: solver uses J_solver = -dR/du (positive stiffness convention).
    So J_solver = -J_fd for correctly implemented blocks.
    """
    n_dofs = solver.n_dofs
    n_extra = solver.n_extra
    n_total = solver.n_total

    if sps.issparse(J_solver):
        J_solver = J_solver.toarray()

    J_neg_fd = -J_fd

    print(f"\n  [{label}]")
    print(f"    ||J_solver|| = {np.linalg.norm(J_solver):.6e}")
    print(f"    ||J_fd||     = {np.linalg.norm(J_fd):.6e}")
    print(f"    ||J_solver - (-J_fd)|| = {np.linalg.norm(J_solver - J_neg_fd):.6e}")
    print(f"    ||J_solver - J_fd||    = {np.linalg.norm(J_solver - J_fd):.6e}")

    blocks = [
        ("K_eff (u,u)",        slice(0, n_dofs),               slice(0, n_dofs)),
        ("C_u (lam,u)",        slice(n_dofs+n_extra, n_total), slice(0, n_dofs)),
        ("C_u^T (u,lam)",      slice(0, n_dofs),               slice(n_dofs+n_extra, n_total)),
        ("C_th (lam,th)",      slice(n_dofs+n_extra, n_total), slice(n_dofs, n_dofs+n_extra)),
        ("C_th^T (th,lam)",    slice(n_dofs, n_dofs+n_extra),  slice(n_dofs+n_extra, n_total)),
        ("K_thth (th,th)",     slice(n_dofs, n_dofs+n_extra),  slice(n_dofs, n_dofs+n_extra)),
        ("CROSS (u,th)",       slice(0, n_dofs),               slice(n_dofs, n_dofs+n_extra)),
        ("CROSS (th,u)",       slice(n_dofs, n_dofs+n_extra),  slice(0, n_dofs)),
    ]

    print(f"\n    {'Block':<28s} {'||A||':>10s} {'||-Fd||':>10s} {'||A-(-Fd)||':>12s} {'||A-Fd||':>10s}  Best sign")
    print(f"    {'-'*85}")

    worst_name, worst_rel = "", 0.0
    for name, rs, cs in blocks:
        a = J_solver[rs, cs]
        f = J_fd[rs, cs]
        na = np.linalg.norm(a)
        nf = np.linalg.norm(f)
        diff_neg = np.linalg.norm(a + f)
        diff_pos = np.linalg.norm(a - f)

        if na < 1e-30 and nf < 1e-30:
            rel = 0.0
            sign = "both zero"
        elif diff_neg < diff_pos:
            rel = diff_neg / (max(na, nf) + 1e-30)
            sign = "NEGATIVE (J=-Fd)"
        else:
            rel = diff_pos / (max(na, nf) + 1e-30)
            sign = "POSITIVE (J=Fd)"

        st = "OK" if rel < 1e-6 else ("~OK" if rel < 1e-3 else ("WARN" if rel < 0.1 else "BROKEN"))
        print(f"    {name:<28s} {na:10.3e} {nf:10.3e} {diff_neg:12.4e} {diff_pos:10.4e}  {sign} {st}")

        if rel > worst_rel:
            worst_rel = rel
            worst_name = name

    print(f"    => Worst: {worst_name} (rel_err={worst_rel:.4e})")

    n_tc = min(n_extra, 2)
    n_uc = min(n_dofs, 24)

    cross_a = J_solver[:n_uc, n_dofs:n_dofs+n_tc]
    cross_f = J_fd[:n_uc, n_dofs:n_dofs+n_tc]
    nnz_a = np.abs(cross_a) > 1e-15
    nnz_f = np.abs(cross_f) > 1e-15
    print(f"\n    CROSS (u,th) sparsity: analytical={np.sum(nnz_a)} nonzero, FD={np.sum(nnz_f)} nonzero")
    if not np.array_equal(nnz_a, nnz_f):
        print(f"    *** SPLICITY MISMATCH ***")
        rows, cols = np.where(nnz_a != nnz_f)
        for r_, c_ in zip(rows[:10], cols[:10]):
            print(f"      [{r_},{c_}]: A={cross_a[r_,c_]:.4e}  Fd={cross_f[r_,c_]:.4e}")

    nonzero = np.abs(cross_a) > 1e-15
    if np.any(nonzero):
        rows, cols = np.where(nonzero)
        print(f"\n    Nonzero J[u,th] (analytical):")
        for r_, c_ in zip(rows, cols):
            print(f"      u-DOF {r_}, th {c_}: A={cross_a[r_,c_]:.6e}  Fd={cross_f[r_,c_]:.6e}")

    return worst_name, worst_rel


def main():
    print("=" * 85)
    print("FD JACOBIAN VERIFICATION - KKT Tangent Consistency Check")
    print("=" * 85)

    mesh, nx, ny, xs, row_pids = build_small_mesh()

    pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    psa_mat = LinearViscoelastic(E=10.0, nu=0.49, g_i=[0.8], tau_i=[1.0])
    materials = {layer: (pet_mat if layer % 2 == 0 else psa_mat) for layer in range(3)}

    tol = 1e-9
    slave_l = [i for i in range(nx) if xs[i] <= -15.0 + tol]
    slave_r = [i for i in range(nx) if xs[i] >= 15.0 - tol]
    rbe2_l = RBE2HingeConstraint(mesh, 99999, slave_l, extra_primal_offset=0)
    rbe2_r = RBE2HingeConstraint(mesh, 99998, slave_r, extra_primal_offset=1)
    c_nodes = list(range(nx)) + [(ny - 1) * nx + i for i in range(nx)]
    contact_c = PenaltyContactConstraint(mesh, contact_nodes=c_nodes, k_contact=1e6, d_0=0.2)

    elem_type = {layer: ("Q4_EAS" if layer % 2 == 0 else "Q4_UP") for layer in range(3)}
    solver = DynamicSolver(
        mesh, materials, rho=1000.0,
        material_params={layer: {} for layer in range(3)},
        constraints=[rbe2_l, rbe2_r],
        penalty_constraints=[contact_c],
        max_iter=30, tol=1e-2, atol=1e-7, verbose=False,
        element_type=elem_type,
        fast_assembly=True, mode="quasistatic",
    )

    print(f"  Mesh: {solver.n_elem} elem, {solver.n_nodes} nodes")
    print(f"  DOFs: u={solver.n_dofs}, theta={solver.n_extra}, lam={solver.n_lambdas}, total={solver.n_total}")

    u0 = np.zeros(solver.n_dofs)
    lam0 = np.zeros(solver.n_lambdas)
    dt = 0.001

    print("\n" + "#" * 85)
    print("TEST 1: u=0, theta=0, lambda=0")
    print("#" * 85)
    ue0 = np.zeros(solver.n_extra)
    J_a1 = jacobian_analytical(solver, u0, ue0, lam0, dt)
    J_f1 = jacobian_fd(solver, u0, ue0, lam0, dt, h=1e-7)
    compare(J_a1, J_f1, solver, "Test1")

    print("\n" + "#" * 85)
    print("TEST 2: u=0, theta=0.05, lambda=0")
    print("#" * 85)
    ue2 = np.array([0.05, -0.05])
    J_a2 = jacobian_analytical(solver, u0, ue2, lam0, dt)
    J_f2 = jacobian_fd(solver, u0, ue2, lam0, dt, h=1e-7)
    compare(J_a2, J_f2, solver, "Test2")

    print("\n" + "#" * 85)
    print("TEST 3: u=0, theta=0.05, lambda=100")
    print("#" * 85)
    lam3 = np.ones(solver.n_lambdas) * 100.0
    J_a3 = jacobian_analytical(solver, u0, ue2, lam3, dt)
    J_f3 = jacobian_fd(solver, u0, ue2, lam3, dt, h=1e-7)
    compare(J_a3, J_f3, solver, "Test3")

    print("\n" + "#" * 85)
    print("TEST 4: u=small_disp, theta=0.1, lambda=200")
    print("#" * 85)
    u4 = u0.copy()
    for i in range(solver.n_nodes):
        u4[2 * i + 1] = 0.001 * np.sin(np.pi * i / solver.n_nodes)
    ue4 = np.array([0.1, -0.1])
    lam4 = np.ones(solver.n_lambdas) * 200.0
    J_a4 = jacobian_analytical(solver, u4, ue4, lam4, dt)
    J_f4 = jacobian_fd(solver, u4, ue4, lam4, dt, h=1e-7)
    compare(J_a4, J_f4, solver, "Test4")


if __name__ == "__main__":
    main()
