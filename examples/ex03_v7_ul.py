"""
ex03_v7_ul.py
=============
U-bend folding with Updated Lagrangian (UL) Q4_EAS elements.

Key differences from ex03_v6:
  1. ALL j rows (not just j=0) are slave nodes — the entire wing cross-section
     rotates rigidly, driven by direct Dirichlet BCs (no RBE2 required).
  2. TARGET_ANGLE = 90 degrees.
  3. ul_mode=True in DynamicSolver — the Q4_EAS element uses an incremental
     (UL) deformation gradient, preventing the element inversion that caused
     NaN residuals in the Total-Lagrangian formulation at theta > 21 deg.

Why UL is needed:
  In TL, the deformation gradient F is always measured from the original
  undeformed reference.  At the slave/free boundary (xs=-10 column), F has
  det(F) < 0 once the slave wing has rotated past ~21 degrees, making stress
  computation pathological.  With UL, each step uses F_inc = I + grad(u_inc)
  where u_inc is the displacement from the LAST CONVERGED configuration.
  For small steps, det(F_inc) > 0 always holds.  The total deformation
  gradient F_total = F_inc @ F_n is then passed to the material model.
"""

import os
import numpy as np

os.environ["MKL_NUM_THREADS"] = "8"
os.environ["PARDISO_NUM_THREADS"] = "8"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, NeoHookean
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.solver import DynamicSolver
from dispsolver.export.vtkhdf_exporter import TransientVTKHDFExporter

# ── Parameters ───────────────────────────────────────────────────────────────
TARGET_ANGLE = np.pi / 2        # 90 degrees
T_TOTAL      = 1.0
DT_INITIAL   = 0.005
DT_MIN       = 1e-6
DT_MAX       = 0.02
MAX_ITER     = 60
TOL          = 2e-2
MAX_CUTBACKS = 50

LEFT_PIVOT   = np.array([-3.0, 0.1])
RIGHT_PIVOT  = np.array([ 3.0, 0.1])
SLAVE_THRESH = 10.0   # |xs| > this -> slave


class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
    def __call__(self, t):
        tau = np.clip((t - self.t0) / (self.t1 - self.t0), 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def run():
    print("=" * 70)
    print("ex03_v7_ul  -  Updated Lagrangian, ALL j rows slave, TARGET=90 deg")
    print("=" * 70)

    # ── 1. Mesh ──────────────────────────────────────────────────────────────
    mesh = Mesh()
    xs_left  = np.linspace(-40.0, -15.0, 21)[:-1]
    xs_mid   = np.linspace(-15.0,  15.0, 61)[:-1]
    xs_right = np.linspace( 15.0,  40.0, 21)
    xs = np.concatenate([xs_left, xs_mid, xs_right])
    nx = len(xs)

    ys_list  = [0.0]
    row_pids = []
    current_y = 0.0
    for layer in range(7):
        dy = 0.05 / 3.0 if layer % 2 == 0 else 0.05
        for _ in range(3 if layer % 2 == 0 else 1):
            current_y += dy
            ys_list.append(current_y)
            row_pids.append(layer)
    ys = np.array(ys_list)
    ny = len(ys)

    for j in range(ny):
        for i in range(nx):
            mesh.add_node(j * nx + i, xs[i], ys[j])

    elem_idx = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i
            mesh.add_element(elem_idx, [n1, n1+1, n1+nx+1, n1+nx], "QUAD4",
                             pid=row_pids[j])
            elem_idx += 1

    nid_to_idx = mesh.node_id_to_index()

    # ── 2. Materials ─────────────────────────────────────────────────────────
    pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    psa_base = NeoHookean()
    psa_mat = ViscoelasticMaterial(psa_base, g_i=[0.8], tau_i=[1.0])
    materials       = {l: (pet_mat if l % 2 == 0 else psa_mat) for l in range(7)}
    material_params = {l: ({} if l % 2 == 0 else {'E': 100.0, 'nu': 0.49}) for l in range(7)}

    # ── 3. Solver with Updated Lagrangian ────────────────────────────────────
    elem_type = {l: ("Q4_EAS" if l % 2 == 0 else "Q4_VISCO_SIMO") for l in range(7)}
    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params=material_params,
        constraints=[], penalty_constraints=[],
        max_iter=MAX_ITER, tol=TOL, atol=1e-7, rtol=5e-3,
        verbose=True,
        element_type=elem_type,
        fast_assembly=True,
        mode="quasistatic",
        ul_mode=True,       # <-- Updated Lagrangian
    )

    # ── 4. Boundary conditions: ALL j rows of slave regions ──────────────────
    bc_dofs = []
    bc_base = []

    slave_ux_dofs = []
    slave_uy_dofs = []
    slave_dx      = []
    slave_dy      = []
    slave_sign    = []   # -1 left, +1 right

    def _add_slave(nid, pivot, sign_val):
        idx = nid_to_idx[nid]
        x_ref = xs[nid % nx]
        y_ref = ys[nid // nx]
        dx = x_ref - pivot[0]
        dy = y_ref - pivot[1]
        slave_ux_dofs.append(idx * 2)
        slave_uy_dofs.append(idx * 2 + 1)
        slave_dx.append(dx)
        slave_dy.append(dy)
        slave_sign.append(sign_val)
        bc_dofs.extend([idx * 2, idx * 2 + 1])
        bc_base.extend([0.0, 0.0])

    for j in range(ny):
        for i, x in enumerate(xs):
            nid = j * nx + i
            if x < -SLAVE_THRESH:
                _add_slave(nid, LEFT_PIVOT, -1.0)
            elif x > SLAVE_THRESH:
                _add_slave(nid, RIGHT_PIVOT, +1.0)

    n_slave = len(slave_ux_dofs)
    slave_ux_dofs = np.array(slave_ux_dofs, dtype=np.int64)
    slave_uy_dofs = np.array(slave_uy_dofs, dtype=np.int64)
    slave_dx      = np.array(slave_dx)
    slave_dy      = np.array(slave_dy)
    slave_sign    = np.array(slave_sign)

    # Symmetry: UX=0 at xs~=0
    for j in range(ny):
        for i, x in enumerate(xs):
            if abs(x) < 0.01:
                nid = j * nx + i
                bc_dofs.append(nid_to_idx[nid] * 2)
                bc_base.append(0.0)

    solver.set_prescribed_dofs(bc_dofs, bc_base)
    solver.max_disp_limit = 300.0

    # Monitor nodes
    centre_nodes = [nid_to_idx[j * nx + i] for j in range(ny) for i, x in enumerate(xs)
                    if abs(x) < 5.0 and j == 0]
    wing_tip_idx = nid_to_idx[0 * nx + 0]

    print(f"  Slave BCs : {n_slave} nodes x2 = {2*n_slave}")
    print(f"  n_dofs    : {solver.n_dofs}")
    print(f"  UL mode   : {solver.ul_mode}")

    # ── 5. Output ────────────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    fp = os.path.join("output", "ex03_v7_ul.vtkhdf")
    exporter = TransientVTKHDFExporter(fp, mesh)
    exporter.add_step(0.0, solver.u)

    ampl = SmoothAmplitude(0.0, T_TOTAL)

    # ── 6. Time loop ─────────────────────────────────────────────────────────
    dt = DT_INITIAL
    step_count = 0
    cutbacks = 0

    while solver.time < T_TOTAL - 1e-12:
        dt = min(dt, T_TOTAL - solver.time)
        factor = ampl(solver.time + dt)
        theta  = -TARGET_ANGLE * factor

        # Update slave BCs (rigid rotation around pivot)
        cosL, sinL = np.cos(theta), np.sin(theta)
        cosR, sinR = np.cos(-theta), np.sin(-theta)
        is_left = slave_sign < 0
        ct = np.where(is_left, cosL, cosR)
        st = np.where(is_left, sinL, sinR)
        ux_vals = (ct - 1.0) * slave_dx - st * slave_dy
        uy_vals =  st        * slave_dx + (ct - 1.0) * slave_dy
        for k in range(n_slave):
            solver._bc_base_vals[2 * k]     = ux_vals[k]
            solver._bc_base_vals[2 * k + 1] = uy_vals[k]

        step_count += 1
        print(f"\n{'='*60}")
        print(f" STEP {step_count}  t={solver.time:.4f}->{solver.time+dt:.4f}"
              f"  theta={np.degrees(theta):.1f}deg")
        print(f"{'='*60}")

        saved = solver.save_state()
        n_iter = solver.solve_step(dt)

        if n_iter < 0:
            cutbacks += 1
            if cutbacks > MAX_CUTBACKS:
                print(f"  FATAL: {MAX_CUTBACKS} cutbacks at t={solver.time:.5f}")
                break
            dt = max(dt * 0.5, DT_MIN)
            print(f"  CUTBACK {cutbacks}: dt->{dt:.3e}")
            solver.restore_state(saved)
            step_count -= 1
            continue

        cutbacks = 0
        exporter.add_step(solver.time, solver.u)

        uy_tip = solver.u[wing_tip_idx * 2 + 1]
        uy_mid = (solver.u[centre_nodes[len(centre_nodes)//2] * 2 + 1]
                  if centre_nodes else 0.0)
        umax = float(np.max(np.abs(solver.u)))
        print(f"  [FOLD] theta={np.degrees(theta):.1f}deg  tip_UY={uy_tip:.2f}mm"
              f"  centre_UY={uy_mid:.4f}mm  max|u|={umax:.2f}mm", flush=True)

        if n_iter <= 3:
            dt = min(dt * 1.2, DT_MAX)
        elif n_iter > 15:
            dt = max(dt * 0.8, DT_MIN)

    print(f"\n{'='*60}")
    print(f"  DONE. Steps={step_count}  Cutbacks={cutbacks}")
    uy_mid_final = (solver.u[centre_nodes[len(centre_nodes)//2] * 2 + 1]
                    if centre_nodes else 0.0)
    print(f"  centre_UY_final={uy_mid_final:.4f}mm")
    if abs(uy_mid_final) > 1.0:
        print("  => U-BEND FORMED!")
    else:
        print("  => Centre still flat — further investigation needed.")
    print(f"{'='*60}")

    exporter.close()
    print(f"Output: {fp}")


if __name__ == "__main__":
    run()
