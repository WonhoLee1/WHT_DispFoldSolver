"""
ex03_v6_prescribed_theta.py
============================
U-bend folding — RBE2 element with externally prescribed hinge rotation θ.

DRIVE: PRESCRIBED θ AT ELEMENT LEVEL (Abaqus-style).
  - RBE2HingeElement.set_prescribed_theta(theta) provides the rotation angle.
  - The element computes penalty + Augmented-Lagrange constraint forces
    acting on ALL element DOFs (master + slaves) — no master-only projection.
  - Master nodes are pinned (Ux=Uy=0 BC).
  - Wing slave nodes have NO prescribed BCs — they follow the RBE2 constraint.
  - The global system is purely displacement-based (no KKT, no λ block).

ARCHITECTURE
------------
  Time loop:
    1. Compute θ from SmoothAmplitude: θ_L = -TARGET * amp(t+dt)
    2. rbe2_left.set_prescribed_theta(θ_L)
    3. solver.solve_step(dt)
       → assembly calls element.compute_contributions()
       → element uses prescribed θ, returns f_e + K_e for ALL DOFs
       → solver adds to global [K_uu]{du} = -{f_int}
  No saddle-point, no extra DOFs, no ill-conditioned KKT regularization.
"""
import os
import numpy as np

from dispsolver.solver import DynamicSolver
from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, NeoHookean
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.element.rbe2 import RBE2HingeElement
from dispsolver.export.vtkhdf_exporter import TransientVTKHDFExporter

# ============================================================
# Hyper-parameters
# ============================================================
N_CORES = 8

MAX_ITER     = 30
TOL          = 1e-3
T_TOTAL      = 1.0
DT_INITIAL   = 0.01
DT_MIN       = 1e-6
DT_MAX       = 0.02
MAX_CUTBACKS = 20
TARGET_ANGLE = 0.523599  # ~30°
FAST_ASSEMBLY = True

os.environ["MKL_NUM_THREADS"]     = str(N_CORES)
os.environ["PARDISO_NUM_THREADS"] = str(N_CORES)
os.environ["XLA_FLAGS"]           = "--xla_cpu_multi_thread_eigen=false"


class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
        self.T = t1 - t0
    def __call__(self, t):
        tau = np.clip((t - self.t0) / self.T, 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def run_v6():
    print("=" * 60)
    print("ex03 v6 -- RBE2 element with prescribed theta (Abaqus-style)")
    print("=" * 60)

    # 1. Mesh
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

    nid_map = {}
    nid_counter = 0
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            nid_map[(j, i)] = nid_counter
            mesh.add_node(nid_counter, x, y)
            nid_counter += 1

    elem_idx = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = nid_map[(j, i)]
            mesh.add_element(elem_idx, [n1, n1+1, n1+nx+1, n1+nx], "QUAD4", pid=row_pids[j])
            elem_idx += 1

    # Hinge master nodes at (-3, 0) and (3, 0)
    master_L = nid_counter
    mesh.add_node(master_L, -3.0, 0.0)
    nid_counter += 1
    master_R = nid_counter
    mesh.add_node(master_R,  3.0, 0.0)
    nid_counter += 1

    coords_all = mesh.nodes_array()
    nid_to_idx = mesh.node_id_to_index()

    # 2. Materials
    pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    psa_base = NeoHookean()
    psa_mat = ViscoelasticMaterial(psa_base, g_i=[0.8], tau_i=[1.0])
    materials = {layer: (pet_mat if layer % 2 == 0 else psa_mat) for layer in range(7)}
    psa_params = {'E': 100.0, 'nu': 0.49}
    material_params = {layer: ({} if layer % 2 == 0 else psa_params) for layer in range(7)}

    # 3. RBE2 elements — bottom row (j=0), wings (x < -10, x > 10)
    slave_L_ids = [nid_map[(0, i)] for i in range(nx) if xs[i] < -10.0]
    slave_R_ids = [nid_map[(0, i)] for i in range(nx) if xs[i] >  10.0]

    rbe2_left  = RBE2HingeElement(master_L, slave_L_ids, coords_all, penalty=1e+4)
    rbe2_right = RBE2HingeElement(master_R, slave_R_ids, coords_all, penalty=1e+4)

    # 4. Solver — RBE2 elements added via rbe2_elements, NO constraints
    elem_type = {layer: ("Q4_EAS" if layer % 2 == 0 else "Q4_VISCO_SIMO") for layer in range(7)}
    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params=material_params,
        constraints=[],
        penalty_constraints=[],
        rbe2_elements=[rbe2_left, rbe2_right],
        max_iter=MAX_ITER, tol=TOL, atol=TOL,
        verbose=True, element_type=elem_type,
        fast_assembly=FAST_ASSEMBLY, mode="quasistatic",
    )

    print(f"\n  n_nodes={mesh.node_count():,}  n_dofs={solver.n_dofs:,}  "
          f"n_rbe2={len(solver.rbe2_elements)}  "
          f"n_extra={solver.n_extra}  n_lambdas={solver.n_lambdas}",
          flush=True)

    # 5. BCs — master nodes pinned, centre-line UX=0
    rotation_ampl = SmoothAmplitude(0.0, T_TOTAL)

    bc_dofs = [
        nid_to_idx[master_L]*2, nid_to_idx[master_L]*2+1,
        nid_to_idx[master_R]*2, nid_to_idx[master_R]*2+1,
    ]
    bc_vals = [0.0, 0.0, 0.0, 0.0]

    # Centre-line UX=0 (symmetry) — x≈0 only
    for j in range(ny):
        for i in range(nx):
            if abs(xs[i]) < 0.01:
                node_idx = nid_to_idx[nid_map[(j, i)]]
                bc_dofs.append(node_idx * 2)  # UX
                bc_vals.append(0.0)

    solver.set_prescribed_dofs(bc_dofs, bc_vals)

    # Wing-tip node IDs for progress reporting
    tip_nodes_L = [nid_map[(0, i)] for i in range(nx) if xs[i] < -10.0]
    tip_nodes_R = [nid_map[(0, i)] for i in range(nx) if xs[i] >  10.0]
    tip_L_nid = tip_nodes_L[0] if tip_nodes_L else 0
    tip_R_nid = tip_nodes_R[0] if tip_nodes_R else 0

    # 6. Time loop — prescribe θ at element level each step
    dt = DT_INITIAL
    step_count = 0
    cutbacks = 0
    total_iter = 0

    os.makedirs("output", exist_ok=True)
    fp = os.path.join("output", "ex03_fold_v6_prescribed_theta.vtkhdf")
    exporter = TransientVTKHDFExporter(fp, mesh)
    exporter.add_step(0.0, solver.u)

    print(f"\n{'='*100}")
    print(f" RBE2 ELEMENT: Abaqus-style prescribed θ (no KKT, no λ block)")
    print(f" θ_L = -TARGET * ampl(t)  θ_R = +TARGET * ampl(t)")
    print(f" TARGET_ANGLE = {TARGET_ANGLE:.4f} rad ({np.degrees(TARGET_ANGLE):.1f}°)")
    print(f"{'='*100}")

    while solver.time < T_TOTAL - 1e-12:
        dt = min(dt, T_TOTAL - solver.time)
        factor = rotation_ampl(solver.time + dt)

        theta_L = -TARGET_ANGLE * factor
        theta_R =  TARGET_ANGLE * factor

        # Prescribe θ at element level (Abaqus-style — no extra DOF)
        rbe2_left.set_prescribed_theta(theta_L)
        rbe2_right.set_prescribed_theta(theta_R)

        step_count += 1
        print(f"\n{'='*100}", flush=True)
        print(f" STEP {step_count}  t={solver.time:.5f}->{solver.time+dt:.5f} "
              f"factor={factor:.4f}", flush=True)
        print(f" θ: L={np.degrees(theta_L):.3f}°  R={np.degrees(theta_R):.3f}°",
              flush=True)
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

        # Step succeeded
        cutbacks = 0
        total_iter += (n_iter + 1)
        exporter.add_step(solver.time, solver.u)

        # Progress
        umax = float(np.max(np.abs(solver.u)))
        tip_UY_L = solver.u[nid_to_idx[tip_L_nid] * 2 + 1]
        tip_UY_R = solver.u[nid_to_idx[tip_R_nid] * 2 + 1]
        if step_count % 5 == 0 or step_count == 1:
            theta_L_actual = rbe2_left.state.theta_n
            print(f"  [FOLD] max|u|={umax:.2f}mm  θ_L_actual={np.degrees(theta_L_actual):.2f}°  "
                  f"tip_L_UY={tip_UY_L:.3f}  tip_R_UY={tip_UY_R:.3f}", flush=True)

        # Adaptive dt
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
    theta_L_actual = rbe2_left.state.theta_n
    theta_R_actual = rbe2_right.state.theta_n
    print(f"  Prescribed-θ drive  max|u|={umax:.3f} mm  "
          f"θ_L={np.degrees(theta_L_actual):.2f}°  θ_R={np.degrees(theta_R_actual):.2f}°",
          flush=True)

    exporter.close()
    print(f"Output: {fp}")


if __name__ == "__main__":
    run_v6()
