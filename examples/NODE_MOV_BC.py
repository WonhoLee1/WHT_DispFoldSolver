"""
NODE_MOV_BC.py
==============
Diagnostic test for the U-bend folding simulation.

**Purpose**
-----------
RBE2 only constrains the bottom row (j=0) of the slave region. The upper
rows (j=1 to ny-1) in the slave region must be dragged by element forces —
which may fail if the slave-free interface element inverts (det(F)<0).

This script DIRECTLY prescribes rigid-rotation BCs to ALL nodes in the
slave region (all j rows, xs < -10.0 and xs > 10.0). There is no RBE2
Lagrange constraint at all.

**What we learn from this test**
---------------------------------
- If center forms U-bend  →  RBE2 bug (only j=0 was constrained; upper
  rows in slave region were never driven, so no force reached the center).
- If center stays flat    →  element formulation issue: the boundary element
  at xs=-10.5 to xs=-10.0 inverts (det(F)<0) and fails to transmit force
  to the free center regardless of how the slave is prescribed.

**Geometry**
------------
Left  slave region : xs < -10.0  (all j rows)  → pivot (-3.0, 0.1)  θ_L < 0
Right slave region : xs >  10.0  (all j rows)  → pivot (+3.0, 0.1)  θ_R > 0
Free center        : -10.0 ≤ xs ≤ 10.0         (UX=0 at xs=0 only)
"""

import os
import numpy as np

os.environ["MKL_NUM_THREADS"]     = "8"
os.environ["PARDISO_NUM_THREADS"] = "8"
os.environ["XLA_FLAGS"]           = "--xla_cpu_multi_thread_eigen=false"

from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, NeoHookean
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.solver import DynamicSolver
from dispsolver.export.vtkhdf_exporter import TransientVTKHDFExporter

# ── Hyper-parameters ────────────────────────────────────────────────────────
TARGET_ANGLE = np.pi / 2          # 90°
T_TOTAL      = 1.0
DT_INITIAL   = 0.005
DT_MIN       = 1e-6
DT_MAX       = 0.02
MAX_ITER     = 60
TOL          = 2e-2
MAX_CUTBACKS = 200
STAB_FACTOR  = 5e-3

LEFT_PIVOT  = np.array([-3.0, 0.1])
RIGHT_PIVOT = np.array([ 3.0, 0.1])
SLAVE_X_THRESHOLD = -10.0   # xs < this → left slave
# right slave: xs > abs(SLAVE_X_THRESHOLD)


class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
    def __call__(self, t):
        tau = np.clip((t - self.t0) / (self.t1 - self.t0), 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def run():
    print("=" * 70)
    print("NODE_MOV_BC - Direct rotation BC (all j rows, no RBE2)")
    print("=" * 70)

    # ── 1. Mesh ──────────────────────────────────────────────────────────────
    mesh = Mesh()
    xs_left  = np.linspace(-40.0, -15.0, 21)[:-1]   # 20 pts, -40 to -16.25 (step 1.25mm)
    xs_mid   = np.linspace(-15.0,  15.0, 61)[:-1]   # 60 pts, step 0.5mm
    xs_right = np.linspace( 15.0,  40.0, 21)         # 21 pts, 15 to 40
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

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(j * nx + i, x, y)

    elem_idx = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i
            mesh.add_element(elem_idx, [n1, n1+1, n1+nx+1, n1+nx], "QUAD4",
                             pid=row_pids[j])
            elem_idx += 1

    nid_to_idx = mesh.node_id_to_index()

    # ── 2. Materials ─────────────────────────────────────────────────────────
    pet_mat  = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    psa_base = NeoHookean()
    psa_mat  = ViscoelasticMaterial(psa_base, g_i=[0.8], tau_i=[1.0])
    materials       = {lay: (pet_mat  if lay % 2 == 0 else psa_mat)  for lay in range(7)}
    material_params = {lay: ({}        if lay % 2 == 0 else {'E': 100.0, 'nu': 0.49})
                       for lay in range(7)}

    # ── 3. Solver with Co-rotational Q4 elements ──────────────────────────────
    element_type = {lay: "Q4_COROTATIONAL" for lay in range(7)}
    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params=material_params,
        constraints=[],          # no RBE2
        element_type=element_type,
        penalty_constraints=[],
        max_iter=MAX_ITER, tol=TOL, atol=1e-7, rtol=5e-3,
        verbose=True,
        fast_assembly=False,
        mode="quasistatic",
    )

    # ── 4. BCs ───────────────────────────────────────────────────────────────
    # Collect slave nodes and their reference offsets from the pivot.
    # We store (dof_ux, dof_uy, dx, dy, sign) where sign=-1 for left (θ<0)
    # and +1 for right (θ>0).
    bc_dofs  = []
    bc_base  = []

    # Parallel arrays for per-step update
    slave_ux_dofs = []
    slave_uy_dofs = []
    slave_dx      = []
    slave_dy      = []
    slave_sign    = []   # -1 → left (θ negative), +1 → right

    def _add_slave(node_id, pivot, sign_val):
        idx = nid_to_idx[node_id]
        x_ref, y_ref = xs[node_id % nx], ys[node_id // nx]
        dx = x_ref - pivot[0]
        dy = y_ref - pivot[1]
        dof_ux = idx * 2
        dof_uy = idx * 2 + 1
        slave_ux_dofs.append(dof_ux)
        slave_uy_dofs.append(dof_uy)
        slave_dx.append(dx)
        slave_dy.append(dy)
        slave_sign.append(sign_val)
        bc_dofs.extend([dof_ux, dof_uy])
        bc_base.extend([0.0, 0.0])   # updated each step

    for j in range(ny):
        for i, x in enumerate(xs):
            nid = j * nx + i
            if x < SLAVE_X_THRESHOLD:
                _add_slave(nid, LEFT_PIVOT, -1.0)   # θ_L < 0
            elif x > abs(SLAVE_X_THRESHOLD):
                _add_slave(nid, RIGHT_PIVOT, +1.0)  # θ_R > 0

    n_slave_bc = len(slave_ux_dofs)
    slave_ux_dofs = np.array(slave_ux_dofs, dtype=np.int64)
    slave_uy_dofs = np.array(slave_uy_dofs, dtype=np.int64)
    slave_dx      = np.array(slave_dx)
    slave_dy      = np.array(slave_dy)
    slave_sign    = np.array(slave_sign)

    # Symmetry: UX=0 at xs=0 for all rows
    sym_start_bc_idx = len(bc_dofs)
    for j in range(ny):
        for i, x in enumerate(xs):
            if abs(x) < 0.01:
                nid = j * nx + i
                idx = nid_to_idx[nid]
                bc_dofs.append(idx * 2)
                bc_base.append(0.0)

    solver.set_prescribed_dofs(bc_dofs, bc_base)
    solver.max_disp_limit = 300.0

    # ── 5. Monitor nodes ─────────────────────────────────────────────────────
    center_bot_nodes = [nid_to_idx[0 * nx + i] for i in range(nx)
                        if -5.0 <= xs[i] <= 5.0]
    wing_tip_idx = nid_to_idx[0 * nx + 0]   # j=0, xs=-40

    print(f"  Slave BCs : {n_slave_bc} nodes x 2 DOFs = {2*n_slave_bc} BCs")
    print(f"  Symmetry  : {len(bc_dofs) - 2*n_slave_bc} BCs")
    print(f"  Total BCs : {len(bc_dofs)}")

    # ── 6. Output ─────────────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    fp = os.path.join("output", "node_mov_bc.vtkhdf")
    exporter = TransientVTKHDFExporter(fp, mesh)
    exporter.add_step(0.0, solver.u)

    ampl = SmoothAmplitude(0.0, T_TOTAL)

    # ── 7. Time loop ──────────────────────────────────────────────────────────
    dt         = DT_INITIAL
    step_count = 0
    cutbacks   = 0
    total_iter = 0

    print(f"\n{'='*70}")
    print(" Starting time loop...")
    print(f"{'='*70}")

    while solver.time < T_TOTAL - 1e-12:
        dt = min(dt, T_TOTAL - solver.time)
        factor = ampl(solver.time + dt)
        theta  = -TARGET_ANGLE * factor       # left θ (negative)

        # Update slave BCs: UX = (cosθ-1)*dx - sinθ*dy
        #                   UY =  sinθ*dx  + (cosθ-1)*dy
        # For right side: θ_R = +TARGET_ANGLE * factor (mirror)
        cosL = np.cos(theta)
        sinL = np.sin(theta)
        cosR = np.cos(-theta)   # cosine is even
        sinR = np.sin(-theta)   # = -sinL

        # Vectorised update
        is_left  = slave_sign < 0
        is_right = ~is_left
        ct = np.where(is_left, cosL, cosR)
        st = np.where(is_left, sinL, sinR)

        ux_vals = (ct - 1.0) * slave_dx - st * slave_dy
        uy_vals =  st        * slave_dx + (ct - 1.0) * slave_dy

        for k in range(n_slave_bc):
            solver._bc_base_vals[2 * k]     = ux_vals[k]
            solver._bc_base_vals[2 * k + 1] = uy_vals[k]
        # solve_step calls _eval_bc_vals(t+dt) internally, which reads _bc_base_vals

        step_count += 1
        # Debug: verify BC is actually non-zero before solve
        if step_count <= 3 or factor > 0.005:
            print(f"  [DBG] _bc_base_vals[0:4] = {solver._bc_base_vals[:4]}")
            print(f"  [DBG] wing_tip_bc_uy (idx 1) = {solver._bc_base_vals[1]:.6f} mm")
        print(f"\n{'='*70}")
        print(f" STEP {step_count}  t={solver.time:.4f}->{solver.time+dt:.4f}"
              f"  theta_L={theta*180/np.pi:.1f}deg  factor={factor:.4f}")
        print(f"{'='*70}")

        saved  = solver.save_state()
        n_iter = solver.solve_step(dt)

        if n_iter < 0:
            cutbacks += 1
            if cutbacks > MAX_CUTBACKS:
                print(f"  FATAL: {MAX_CUTBACKS} cutbacks at t={solver.time:.5f}")
                break
            dt = max(dt * 0.5, DT_MIN)
            print(f"  CUTBACK {cutbacks}: dt→{dt:.3e}")
            solver.restore_state(saved)
            step_count -= 1
            continue

        cutbacks = 0
        total_iter += (n_iter + 1)
        exporter.add_step(solver.time, solver.u)

        uy_tip = solver.u[wing_tip_idx * 2 + 1]
        uy_mid = (solver.u[center_bot_nodes[len(center_bot_nodes) // 2] * 2 + 1]
                  if center_bot_nodes else 0.0)
        umax = float(np.max(np.abs(solver.u)))
        print(f"  [RESULT] θ={theta*180/np.pi:.1f}°  tip_UY={uy_tip:.3f}mm"
              f"  centre_UY={uy_mid:.4f}mm  max|u|={umax:.2f}mm", flush=True)

        # Warn if center is not forming U-bend
        expected_tip = abs(np.sin(theta)) * 35.0
        if factor > 0.3 and abs(uy_mid) < 0.5:
            print(f"  [WARNING] Centre still flat! "
                  f"tip={uy_tip:.1f}mm (expected~{expected_tip:.1f}mm)")
        elif factor > 0.1 and abs(uy_tip) > 1.0 and abs(uy_mid) > 0.1:
            print(f"  [U-BEND FORMING] Centre moving!")

        if n_iter <= 3:
            dt = min(dt * 1.2, DT_MAX)
        elif n_iter <= 8:
            pass
        elif n_iter <= 15:
            dt = max(dt * 0.85, DT_MIN)
        else:
            dt = max(dt * 0.7, DT_MIN)

    print(f"\n{'='*70}")
    print(f"  DONE. Steps={step_count}  Cutbacks={cutbacks}  Iter={total_iter}")
    umax = float(np.max(np.abs(solver.u)))
    uy_mid_final = (solver.u[center_bot_nodes[len(center_bot_nodes) // 2] * 2 + 1]
                    if center_bot_nodes else 0.0)
    print(f"  max|u|={umax:.3f}mm   centre_bottom_UY={uy_mid_final:.4f}mm")
    if abs(uy_mid_final) < 1.0:
        print("  => DIAGNOSIS: centre remains flat → element boundary issue")
        print("     (det(F)<0 at xs=-10.5 to xs=-10.0 interface)")
    else:
        print("  => DIAGNOSIS: centre forms U-bend → RBE2 was the problem!")
        print("     (RBE2 only constrained j=0; upper rows were not driven)")
    print(f"{'='*70}")

    exporter.close()
    print(f"Output: {fp}")


if __name__ == "__main__":
    run()
