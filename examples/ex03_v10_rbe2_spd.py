"""
ex03_v10_rbe2_spd.py
====================
TRUE FEA kinematic RBE2 condensation via DynamicSolver.
Solves K_red = T^T @ K_full @ T -- a pure SPD system with
no Lagrange multipliers. Left hinge at x=-12 rotates +90 deg,
right hinge at x=+12 rotates -90 deg. Center panel sags
naturally through FEA equilibrium.

KEY DEMONSTRATION:
  - `KinematicRBE2Constraint` for left/right hinges
  - `DynamicSolver(rbe2_constraints=[...])` activates condensation
  - θ extra DOFs prescribed via BCs (no penalty spring needed)
  - Newton solves the reduced SPD system K_red  *  du = R_red
  - BC residual correctly uses target - current for each step
"""

import os
import numpy as np

N_CORES = 4
MAX_ITER = 60
TOL = 1e-2
T_TOTAL = 1.0
DT_INITIAL = 0.005
DT_MIN = 1e-6
DT_MAX = 0.02
MAX_CUTBACKS = 20
FAST_ASSEMBLY = True

os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["PARDISO_NUM_THREADS"] = str(N_CORES)
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity
from dispsolver.material.neohookean import NeoHookean
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.constraint.rbe2_condensed import KinematicRBE2Constraint
from dispsolver.solver import DynamicSolver
from dispsolver.export.vtkhdf_exporter import TransientVTKHDFExporter


class SmoothAmplitude:
    """C2-continuous quintic step from t0 to t1."""
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1, self.T = t0, t1, t1 - t0

    def __call__(self, t):
        tau = np.clip((t - self.t0) / self.T, 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def build_mesh():
    """Build 7-layer PET/PSA panel with refined center section."""
    mesh = Mesh()

    xs_left = np.linspace(-40.0, -15.0, 21)[:-1]
    xs_mid = np.linspace(-15.0, 15.0, 61)[:-1]
    xs_right = np.linspace(15.0, 40.0, 21)
    xs = np.concatenate([xs_left, xs_mid, xs_right])
    nx = len(xs)

    ys_list = [0.0]
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
        pid = row_pids[j]
        for i in range(nx - 1):
            n1 = j * nx + i
            mesh.add_element(elem_idx, [n1, n1 + 1, n1 + nx + 1, n1 + nx], "QUAD4", pid=pid)
            elem_idx += 1

    return mesh, xs, nx, ny, row_pids


def run_rbe2_spd_condensation():
    print("=" * 70)
    print("ex03_v10 RBE2 KINEMATIC CONDENSATION (SPD) -- TRUE FEA EQUILIBRIUM")
    print("=" * 70)

    # 1. Mesh
    mesh, xs, nx, ny, row_pids = build_mesh()
    n_nodes = mesh.node_count()
    nid_to_idx = mesh.node_id_to_index()
    coords = mesh.nodes_array()
    print(f"Mesh: {n_nodes} nodes, {len(mesh.elements)} elements, {ny} layers, {nx} columns")

    # 2. Materials
    pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    psa_mat = ViscoelasticMaterial(NeoHookean(), g_i=[0.8], tau_i=[1.0])
    materials = {layer: pet_mat if layer % 2 == 0 else psa_mat for layer in range(7)}
    material_params = {layer: ({} if layer % 2 == 0 else {"E": 10.0, "nu": 0.49}) for layer in range(7)}
    element_type = {layer: "Q4_EAS" if layer % 2 == 0 else "Q4_VISCO_SIMO" for layer in range(7)}

    # 3. RBE2 constraints
    hinge_l_x = -12.0
    hinge_r_x = 12.0

    master_l_id = None
    master_r_id = None
    slave_l = []
    slave_r = []

    for j in range(ny):
        for i in range(nx):
            nid = j * nx + i
            x_val = xs[i]
            if abs(x_val - hinge_l_x) < 0.01:
                if j == 0:
                    master_l_id = nid
                else:
                    slave_l.append(nid)
            elif x_val < hinge_l_x:
                slave_l.append(nid)

    for j in range(ny):
        for i in range(nx):
            nid = j * nx + i
            x_val = xs[i]
            if abs(x_val - hinge_r_x) < 0.01:
                if j == 0:
                    master_r_id = nid
                else:
                    slave_r.append(nid)
            elif x_val > hinge_r_x:
                slave_r.append(nid)

    print(f"Left hinge: master={master_l_id}, {len(slave_l)} slaves")
    print(f"Right hinge: master={master_r_id}, {len(slave_r)} slaves")

    rbe2_L = KinematicRBE2Constraint(mesh, master_l_id, slave_l)
    rbe2_R = KinematicRBE2Constraint(mesh, master_r_id, slave_r)

    # 4. DynamicSolver with condensation
    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params=material_params,
        constraints=[],
        rbe2_constraints=[rbe2_L, rbe2_R],
        max_iter=MAX_ITER, tol=TOL, atol=1e-7, rtol=5e-3,
        verbose=True, element_type=element_type,
        fast_assembly=FAST_ASSEMBLY, mode="quasistatic",
    )

    print(f"  n_dofs={solver.n_dofs}, n_extra={solver.n_extra}, n_lambdas={solver.n_lambdas}")
    print(f"  Condensation: n_slaves={len(solver.condensation_mgr.slave_dofs)}, "
          f"n_independent={solver.condensation_mgr.n_independent}")

    # 5. BCs: symmetry + rigid body (θ is driven by penalty spring, not BCs)
    amplitude = SmoothAmplitude(0.0, T_TOTAL)
    THETA_MAX = np.pi / 2.0  # 90 deg
    THETA_PENALTY_K = 1e6    # stiff penalty spring drives θ to target

    bc_dofs = []
    bc_base = []
    bc_amps = []

    # Centreline UX=0 (all layers at x ≈ 0)
    for j in range(ny):
        for i in range(nx):
            if abs(xs[i]) < 0.01:
                node_idx = nid_to_idx[j * nx + i]
                bc_dofs.append(node_idx * 2)  # UX=0
                bc_base.append(0.0)
                bc_amps.append(None)

    # Bottom centre UY=0 (rigid-body mode)
    for i in range(nx):
        if abs(xs[i]) < 0.01:
            node_idx = nid_to_idx[0 * nx + i]
            bc_dofs.append(node_idx * 2 + 1)  # UY=0
            bc_base.append(0.0)
            bc_amps.append(None)
            break

    solver.set_prescribed_dofs(bc_dofs, bc_base, amplitudes=bc_amps)
    solver.theta_penalty_k = THETA_PENALTY_K
    solver.max_disp_limit = 300.0

    # 6. Time loop
    dt = DT_INITIAL
    step_count = 0
    cutbacks = 0
    total_iter = 0

    os.makedirs("output", exist_ok=True)
    fp = os.path.join("output", "ex03_v10_rbe2_spd.vtkhdf")
    exporter = TransientVTKHDFExporter(fp, mesh)
    exporter.add_step(0.0, solver.u)

    print(f"\n{'=' * 100}")
    print(f" RBE2 CONDENSATION: θ_target=±90 deg, no penalty, pure SPD solve")
    print(f" Left hinge: {len(slave_l)} slaves at x=-12")
    print(f" Right hinge: {len(slave_r)} slaves at x=+12")
    print(f" Theta penalty: k={THETA_PENALTY_K:.1e}")
    print(f" Initial dt={DT_INITIAL}, max dt={DT_MAX}, min dt={DT_MIN}")
    print(f"{'=' * 100}")

    while solver.time < T_TOTAL - 1e-12:
        dt = min(dt, T_TOTAL - solver.time)
        t_next = solver.time + dt

        amp_val = amplitude(t_next)
        theta_L_target = -THETA_MAX * amp_val
        theta_R_target = THETA_MAX * amp_val

        step_count += 1
        print(f"\n{'=' * 100}", flush=True)
        print(f" STEP {step_count}  t={solver.time:.5f}->{t_next:.5f}  dt={dt:.3e}  "
              f"θ=[{theta_L_target:.4f},{theta_R_target:.4f}]", flush=True)
        print(f"{'=' * 100}", flush=True)

        solver.theta_targets = {0: theta_L_target, 1: theta_R_target}
        saved = solver.save_state()
        n_iter = solver.solve_step(dt)

        if n_iter < 0:
            cutbacks += 1
            if cutbacks > MAX_CUTBACKS:
                print(f"\n  FATAL: {MAX_CUTBACKS} cutbacks at t={solver.time:.5f}", flush=True)
                break
            dt = max(dt * 0.5, DT_MIN)
            print(f"  CUTBACK {cutbacks}: dt->{dt:.3e}", flush=True)
            solver.restore_state(saved)
            step_count -= 1
            continue

        cutbacks = 0
        total_iter += (n_iter + 1)
        exporter.add_step(solver.time, solver.u)

        # Report actual θ
        if solver.u_extra is not None and len(solver.u_extra) >= 2:
            print(f"  θ actual: L={solver.u_extra[0]:.4f} R={solver.u_extra[1]:.4f}  "
                  f"error: L={theta_L_target - solver.u_extra[0]:.6f} R={theta_R_target - solver.u_extra[1]:.6f}",
                  flush=True)

        umax = float(np.max(np.abs(solver.u)))
        print(f"  max|u|={umax:.4f} mm", flush=True)

        # dt controller
        if n_iter <= 3:
            dt = min(dt * 1.2, DT_MAX)
        elif n_iter <= 8:
            pass
        elif n_iter <= 15:
            dt = max(dt * 0.85, DT_MIN)
        else:
            dt = max(dt * 0.7, DT_MIN)

    else:
        print(f"\n{'=' * 100}", flush=True)
        print(f"  ALL DONE. Steps: {step_count} Cutbacks: {cutbacks} Total iter: {total_iter}", flush=True)
        print(f"{'=' * 100}", flush=True)

    umax = float(np.max(np.abs(solver.u)))
    if solver.u_extra is not None and len(solver.u_extra) >= 2:
        theta_L_final = float(solver.u_extra[0])
        theta_R_final = float(solver.u_extra[1])
        print(f"  θ final: L={theta_L_final:.4f} rad ({np.degrees(theta_L_final):.1f} deg)  "
              f"R={theta_R_final:.4f} rad ({np.degrees(theta_R_final):.1f} deg)", flush=True)
    print(f"  max|u|={umax:.3f} mm", flush=True)

    # Report condensation efficiency
    reduction_pct = (1.0 - solver.condensation_mgr.n_independent / (solver.n_dofs + solver.n_extra)) * 100
    print(f"  Condensation: full={solver.n_dofs + solver.n_extra} → "
          f"independent={solver.condensation_mgr.n_independent} "
          f"({reduction_pct:.1f}% reduction)", flush=True)

    exporter.close()
    print(f"Output: {fp}")


if __name__ == "__main__":
    run_rbe2_spd_condensation()
