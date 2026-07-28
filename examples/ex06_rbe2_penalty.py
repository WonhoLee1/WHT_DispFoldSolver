"""
ex06_rbe2_penalty.py
=====================
Display folding via RBE2 hinge + θ-penalty spring — the θ extra DOF is
driven toward 90° by a stiff penalty spring (k=1e6) instead of prescribed
via Dirichlet BC.

KEY INGREDIENTS:
  1. Two RBE2HingeConstraints at x=-5 and x=+5 — each couples all layers
     of the wing to a master node at the hinge.
  2. θ NOT prescribed — solver.theta_penalty_k enforces a penalty spring
     on each θ extra DOF toward its target value.
  3. The penalty spring preserves KKT symmetry (diagonal +k) vs BC which
     zeroes rows asymmetrically and stalls at ~37°.
  4. Viscous stabilization for the thin-layer stack.
  5. U-bend curvature monitor at every step.

WHY THIS WORKS:
  - Boundary at the hinge (x=±5) → zero element shear regardless of θ.
  - Penalty-driven θ converges naturally through Newton iteration → no
    BC-elimination stalling.
  - The penalty diagonal (+k) keeps the KKT system well-conditioned even
    at large θ where cosθ → 0 and ∂gap/∂θ → 0.
"""

import os
import numpy as np
from dispsolver.solver import DynamicSolver

# ============================================================
# Hyper-parameters
# ============================================================
N_CORES = 8

MAX_ITER     = 60
TOL          = 1e-2
T_TOTAL      = 1.0
DT_INITIAL   = 0.005     # smaller initial step for penalty ramp (θ lags behind target)
DT_MIN       = 1e-6
DT_MAX       = 0.02      # cap for stability
MAX_CUTBACKS = 20
FAST_ASSEMBLY = True
STAB_FACTOR  = 8e-3      # increased for snap-through at hinge zone (~38 deg)

THETA_PENALTY_K = 1e6    # stiff spring: tracks target with <<0.1° error

os.environ["MKL_NUM_THREADS"]     = str(N_CORES)
os.environ["PARDISO_NUM_THREADS"] = str(N_CORES)
os.environ["XLA_FLAGS"]           = "--xla_cpu_multi_thread_eigen=false"

from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, LinearViscoelastic
from dispsolver.constraint import RBE2HingeConstraint, PenaltyContactConstraint
from dispsolver.export.vtkhdf_exporter import TransientVTKHDFExporter


class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
        self.T = t1 - t0
    def __call__(self, t):
        tau = np.clip((t - self.t0) / self.T, 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def run_v6_rbe2_penalty():
    print("=" * 60)
    print("ex06 RBE2 + theta-penalty spring - 90deg hinge fold")
    print("=" * 60)

    # 1. Mesh — single continuous mesh (101 × 13 = 1313 nodes)
    mesh = Mesh()
    xs_left  = np.linspace(-40.0, -15.0, 21)[:-1]
    xs_mid   = np.linspace(-15.0, 15.0, 61)[:-1]
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
        for i in range(nx - 1):
            n1 = j * nx + i
            mesh.add_element(elem_idx, [n1, n1+1, n1+nx+1, n1+nx], "QUAD4", pid=row_pids[j])
            elem_idx += 1

    nx = len(xs)   # re-read in case shifted
    n_nodes = len(xs) * len(ys)

    # 2. Materials
    pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    psa_mat = LinearViscoelastic(E=100.0, nu=0.49, g_i=[0.8], tau_i=[1.0])
    materials = {layer: (pet_mat if layer % 2 == 0 else psa_mat) for layer in range(7)}

    # 3. RBE2 constraints — left hinge at x=-5, right hinge at x=+5
    #    All layers of each wing are slaves.
    master_l_id = None
    master_r_id = None
    slave_l = []
    slave_r = []
    nid_to_idx = mesh.node_id_to_index()

    hinge_l_x = -5.0
    hinge_r_x = 5.0
    for j in range(ny):
        for i in range(nx):
            nid = j * nx + i
            if abs(xs[i] - hinge_l_x) < 0.01:
                if j == 0:
                    master_l_id = nid
                else:
                    slave_l.append(nid)
            elif xs[i] < -5.0:
                slave_l.append(nid)

    for j in range(ny):
        for i in range(nx):
            nid = j * nx + i
            if abs(xs[i] - hinge_r_x) < 0.01:
                if j == 0:
                    master_r_id = nid
                else:
                    slave_r.append(nid)
            elif xs[i] > 5.0:
                slave_r.append(nid)

    # Hinge at bottom-centre node (y=0)
    hinge_l = RBE2HingeConstraint(mesh, master_l_id, slave_l, extra_primal_offset=0)
    hinge_r = RBE2HingeConstraint(mesh, master_r_id, slave_r, extra_primal_offset=1)

    # 4. Contact constraint (self-contact)
    contact_nodes = [j * nx + i for j in [0, ny - 1] for i in range(nx)]
    contact = PenaltyContactConstraint(mesh, contact_nodes, k_contact=1e4, d_0=0.2)

    # 5. Solver — RBE2 constraints (θ-primal extra DOFs) + contact
    elem_type = {layer: ("Q4_EAS" if layer % 2 == 0 else "Q4_UP") for layer in range(7)}
    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params={layer: {} for layer in range(7)},
        constraints=[hinge_l, hinge_r],
        penalty_constraints=[contact],
        max_iter=MAX_ITER, tol=TOL, atol=1e-7, rtol=5e-3, force_atol=1e10, max_du_atol=1e-3,
        verbose=True, element_type=elem_type,
        fast_assembly=FAST_ASSEMBLY, mode="quasistatic",
        viscous_stab_factor=STAB_FACTOR,
    )

    # θ-penalty spring: k=1e6 drives θ toward target with negligible tracking error
    solver.theta_penalty_k = THETA_PENALTY_K

    # 6. BCs: centreline UX=0, one centre UY=0 (rigid-body mode)
    #    θ is NOT prescribed — it is driven by the penalty spring.
    amplitude = SmoothAmplitude(0.0, T_TOTAL)
    THETA_MAX = np.pi / 2.0   # 90°

    bc_dofs = []
    bc_base = []
    bc_amps = []

    # Centreline symmetry UX=0 (all layers at x≈0)
    for j in range(ny):
        for i in range(nx):
            if abs(xs[i]) < 0.01:
                node_idx = nid_to_idx[j * nx + i]
                bc_dofs.append(node_idx * 2)         # UX=0
                bc_base.append(0.0)
                bc_amps.append(None)

    # Bottom centre UY=0 (rigid-body mode)
    for i in range(nx):
        if abs(xs[i]) < 0.01:
            node_idx = nid_to_idx[0 * nx + i]         # j=0 (bottom layer)
            bc_dofs.append(node_idx * 2 + 1)          # UY=0
            bc_base.append(0.0)
            bc_amps.append(None)
            break

    solver.set_prescribed_dofs(bc_dofs, bc_base, amplitudes=bc_amps)
    solver.max_disp_limit = 300.0

    # U-bend curvature monitor
    def _curvature_monitor(u):
        ctr_bot = ctr_top = None
        hl_bot = hl_top = hr_bot = hr_top = None
        for j in [0, ny - 1]:
            for i in range(nx):
                node_id = j * nx + i
                idx = nid_to_idx[node_id]
                if abs(xs[i]) < 0.01:
                    if j == 0: ctr_bot = idx
                    if j == ny - 1: ctr_top = idx
                if abs(xs[i] - (-5.0)) < 0.01:
                    if j == 0: hl_bot = u[idx * 2 + 1]
                    if j == ny - 1: hl_top = u[idx * 2 + 1]
                if abs(xs[i] - 5.0) < 0.01:
                    if j == 0: hr_bot = u[idx * 2 + 1]
                    if j == ny - 1: hr_top = u[idx * 2 + 1]
        return (
            u[ctr_bot * 2 + 1] if ctr_bot is not None else 0.0,
            u[ctr_top * 2 + 1] if ctr_top is not None else 0.0,
            hl_bot or 0.0, hl_top or 0.0, hr_bot or 0.0, hr_top or 0.0,
        )

    # 7. Time loop
    dt = DT_INITIAL
    step_count = 0
    cutbacks = 0
    total_iter = 0

    os.makedirs("output", exist_ok=True)
    fp = os.path.join("output", "ex06_fold_penalty.vtkhdf")
    exporter = TransientVTKHDFExporter(fp, mesh)
    exporter.add_step(0.0, solver.u)

    print(f"\n{'='*100}")
    print(f" RBE2+θ-PENALTY: θ_target=90°, k_θ={THETA_PENALTY_K:.1e}")
    print(f" Left hinge master={master_l_id} ({len(slave_l)} slaves)")
    print(f" Right hinge master={master_r_id} ({len(slave_r)} slaves)")
    print(f" Contact: {len(contact_nodes)} nodes, k_c={contact.k_contact:.1e}, d_0={contact.d_0}")
    print(f" Stabilization: {STAB_FACTOR}")
    print(f" Initial dt={DT_INITIAL}, max dt={DT_MAX}, min dt={DT_MIN}")
    print(f"{'='*100}")

    while solver.time < T_TOTAL - 1e-12:
        dt = min(dt, T_TOTAL - solver.time)
        t_next = solver.time + dt

        # Set θ targets for this step: ramp from 0 → ±π/2
        amp_val = amplitude(t_next)
        theta_L_target = -THETA_MAX * amp_val   # left hinge: clockwise (-)
        theta_R_target =  THETA_MAX * amp_val   # right hinge: counter-clockwise (+)
        solver.theta_targets = {0: theta_L_target, 1: theta_R_target}

        step_count += 1
        print(f"\n{'='*100}", flush=True)
        print(f" STEP {step_count}  t={solver.time:.5f}->{t_next:.5f}  dt={dt:.3e}  "
              f"θ_target=[{theta_L_target:.4f},{theta_R_target:.4f}]", flush=True)
        print(f"{'='*100}", flush=True)

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

        # U-bend curvature monitor
        uy_cb, uy_ct, uy_hlb, uy_hlt, uy_hrb, uy_hrt = _curvature_monitor(solver.u)
        print(f"  U-bend: centre UY_bot={uy_cb:.4f} top={uy_ct:.4f}  "
              f"hinge_L UY_bot={uy_hlb:.4f} top={uy_hlt:.4f}  "
              f"hinge_R UY_bot={uy_hrb:.4f} top={uy_hrt:.4f}", flush=True)

        # Report actual θ
        if solver.u_extra is not None and len(solver.u_extra) >= 2:
            print(f"  θ actual: L={solver.u_extra[0]:.4f} R={solver.u_extra[1]:.4f}  "
                  f"error: L={theta_L_target-solver.u_extra[0]:.6f} R={theta_R_target-solver.u_extra[1]:.6f}",
                  flush=True)

        # Δt controller
        if n_iter <= 3:
            dt = min(dt * 1.2, DT_MAX)
        elif n_iter <= 8:
            pass
        elif n_iter <= 15:
            dt = max(dt * 0.85, DT_MIN)
        else:
            dt = max(dt * 0.7, DT_MIN)

    else:
        print(f"\n{'='*100}", flush=True)
        print(f"  ALL DONE. Steps: {step_count} Cutbacks: {cutbacks} Total iter: {total_iter}", flush=True)
        print(f"{'='*100}", flush=True)

    umax = float(np.max(np.abs(solver.u)))
    if solver.u_extra is not None and len(solver.u_extra) >= 2:
        theta_L_final = float(solver.u_extra[0])
        theta_R_final = float(solver.u_extra[1])
        print(f"  θ final: L={theta_L_final:.4f} rad ({np.degrees(theta_L_final):.1f}°)  "
              f"R={theta_R_final:.4f} rad ({np.degrees(theta_R_final):.1f}°)", flush=True)
    print(f"  max|u|={umax:.3f} mm", flush=True)
    exporter.close()
    print(f"Output: {fp}")


if __name__ == "__main__":
    run_v6_rbe2_penalty()
