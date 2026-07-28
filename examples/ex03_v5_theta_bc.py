"""
ex03_v5_theta_bc.py
===================
U-bend folding — θ BC drive with RBE2 KKT constraints.

DRIVE: PRESCRIBED HINGE ROTATION θ (via extra DOF BC).
  - Extra DOFs: u_extra[0] = θ_L (left hinge), u_extra[1] = θ_R (right hinge).
  - θ_L, θ_R are prescribed via SmoothAmplitude (C² quintic ramp).
  - RBE2HingeConstraint (KKT, Lagrange multipliers) enforces rigid-body
    rotation of wing-tip slave nodes about the pinned master.
  - Wing nodes have NO direct displacement BCs — the RBE2 constraint
    translates θ into displacements that the structure can accommodate
    naturally, avoiding Q4 root-element distortion.

DESIGN RATIONALE
----------------
  v4 (prescribed-rotation drive) forced ALL wing-tip nodes along a rigid
  kinematic path: u_s = (R(θ)-I)·d₀  as displacement BCs.  This failed
  at θ ≈ -13° because Q4 bilinear elements at the hinge root could not
  accommodate the prescribed distortion.  v5 returns to the v3-1 approach:
  only θ is prescribed; the RBE2 constraint propagates rotation into wing
  displacements through KKT equilibrium — the same approach that reached
  -90° in the original simulation.
"""
import os
import numpy as np

from dispsolver.solver import DynamicSolver
from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, NeoHookean
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.constraint import RBE2HingeConstraint
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
TARGET_ANGLE = 0.523599  # ~30° (conservative first test)
FAST_ASSEMBLY = True

os.environ["MKL_NUM_THREADS"]     = str(N_CORES)
os.environ["PARDISO_NUM_THREADS"] = str(N_CORES)
os.environ["XLA_FLAGS"]           = "--xla_cpu_multi_thread_eigen=false"

# ----------------------------------------------------------------
# Smooth C2 amplitude (quintic polynomial)
# ----------------------------------------------------------------
class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
        self.T = t1 - t0
    def __call__(self, t):
        tau = np.clip((t - self.t0) / self.T, 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def run_v5():
    print("=" * 60)
    print("ex03 v5 -- θ BC drive (RBE2 KKT constraint)")
    print("=" * 60)

    # 1. Mesh — same as v4
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

    # 2. Materials — same as v4
    pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    psa_base = NeoHookean()
    psa_mat = ViscoelasticMaterial(psa_base, g_i=[0.8], tau_i=[1.0])
    materials = {layer: (pet_mat if layer % 2 == 0 else psa_mat) for layer in range(7)}
    psa_params = {'E': 100.0, 'nu': 0.49}
    material_params = {layer: ({} if layer % 2 == 0 else psa_params) for layer in range(7)}

    # 3. RBE2 constraints — bottom row (j=0), wings (x < -10, x > 10)
    slave_L = [nid_map[(0, i)] for i in range(nx) if xs[i] < -10.0]
    slave_R = [nid_map[(0, i)] for i in range(nx) if xs[i] >  10.0]

    # extra_primal_offset:
    #   0 → u_extra[0] = theta_L  (left hinge rotation)
    #   1 → u_extra[1] = theta_R  (right hinge rotation)
    rbe2_left  = RBE2HingeConstraint(mesh, master_L, slave_L, extra_primal_offset=0)
    rbe2_right = RBE2HingeConstraint(mesh, master_R, slave_R, extra_primal_offset=1)

    # 4. Solver — KKT saddle-point (RBE2 in constraints, no RBE2 elements)
    elem_type = {layer: ("Q4_EAS" if layer % 2 == 0 else "Q4_VISCO_SIMO") for layer in range(7)}
    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params=material_params,
        constraints=[rbe2_left, rbe2_right],
        penalty_constraints=[],
        rbe2_elements=[],
        max_iter=MAX_ITER, tol=TOL, atol=TOL,
        verbose=True, element_type=elem_type,
        fast_assembly=FAST_ASSEMBLY, mode="quasistatic",
    )

    n_total_dofs = solver.n_dofs
    n_lambdas = solver.n_lambdas
    print(f"\n  n_nodes={mesh.node_count():,}  "
          f"n_dofs={n_total_dofs:,}  n_extra={solver.n_extra}  n_lambdas={n_lambdas:,}",
          flush=True)

    # 5. BCs
    #  ┌─ 0 ─┐  ┌─ 1 ─┐  ┌─ 2 ─┐  ┌─ 3 ─┐  ┌── 4 ──┐  ┌── 5 ──┐
    #  │master│  │master│  │master│  │master│  │ extra  │  │ extra  │
    #  │L UX  │  │L UY  │  │R UX  │  │R UY  │  │theta_L │  │theta_R │
    #  └──────┘  └──────┘  └──────┘  └──────┘  └────────┘  └────────┘
    rotation_ampl = SmoothAmplitude(0.0, T_TOTAL)

    idx_L = nid_to_idx[master_L]
    idx_R = nid_to_idx[master_R]
    bc_dofs = [
        idx_L * 2,          # master_L UX
        idx_L * 2 + 1,      # master_L UY
        idx_R * 2,          # master_R UX
        idx_R * 2 + 1,      # master_R UY
        n_total_dofs + 0,   # theta_L (extra DOF)
        n_total_dofs + 1,   # theta_R (extra DOF)
    ]
    bc_base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    bc_amps = [None, None, None, None, None, None]  # theta updated manually

    BC_IDX_THETA_L = 4  # position in bc_dofs / bc_base
    BC_IDX_THETA_R = 5

    # Centerline UX=0 (symmetry) — x≈0 only
    for j in range(ny):
        for i in range(nx):
            if abs(xs[i]) < 0.01:
                node_idx = nid_to_idx[nid_map[(j, i)]]
                bc_dofs.append(node_idx * 2)  # UX
                bc_base.append(0.0)
                bc_amps.append(None)

    solver.set_prescribed_dofs(bc_dofs, bc_base, amplitudes=bc_amps)

    # Wing-tip node IDs for progress reporting
    tip_nodes_L = [nid_map[(0, i)] for i in range(nx) if xs[i] < -10.0]
    tip_nodes_R = [nid_map[(0, i)] for i in range(nx) if xs[i] >  10.0]
    tip_L_nid = tip_nodes_L[0] if tip_nodes_L else 0
    tip_R_nid = tip_nodes_R[0] if tip_nodes_R else 0

    # 6. Time loop — prescribe θ targets per step
    dt = DT_INITIAL
    step_count = 0
    cutbacks = 0
    total_iter = 0

    os.makedirs("output", exist_ok=True)
    fp = os.path.join("output", "ex03_fold_v5_theta_bc.vtkhdf")
    exporter = TransientVTKHDFExporter(fp, mesh)
    exporter.add_step(0.0, solver.u)

    print(f"\n{'='*100}")
    print(f" RBE2 KKT: Lagrange multiplier rigid rotation (θ BC drive)")
    print(f" Drive: theta_L = -TARGET * ampl(t)  theta_R = +TARGET * ampl(t)")
    print(f" TARGET_ANGLE = {TARGET_ANGLE:.4f} rad ({np.degrees(TARGET_ANGLE):.1f}°)")
    print(f"{'='*100}")

    while solver.time < T_TOTAL - 1e-12:
        dt = min(dt, T_TOTAL - solver.time)
        factor = rotation_ampl(solver.time + dt)

        step_count += 1
        # Prescribe θ BCs for this step
        theta_L_target = -TARGET_ANGLE * factor
        theta_R_target =  TARGET_ANGLE * factor
        solver._bc_base_vals[BC_IDX_THETA_L] = theta_L_target
        solver._bc_base_vals[BC_IDX_THETA_R] = theta_R_target

        theta_L = solver.u_extra[0]
        theta_R = solver.u_extra[1]
        print(f"\n{'='*100}", flush=True)
        print(f" STEP {step_count}  t={solver.time:.5f}->{solver.time+dt:.5f} "
              f"factor={factor:.4f}", flush=True)
        print(f" θ: L={np.degrees(theta_L):.2f}°→{np.degrees(theta_L_target):.2f}°  "
              f"R={np.degrees(theta_R):.2f}°→{np.degrees(theta_R_target):.2f}°",
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
            print(f"  [FOLD] max|u|={umax:.2f}mm  θ_L={np.degrees(solver.u_extra[0]):.2f}°  "
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
    print(f"  θ-BC drive  max|u|={umax:.3f} mm  "
          f"θ_L={np.degrees(solver.u_extra[0]):.2f}°  θ_R={np.degrees(solver.u_extra[1]):.2f}°",
          flush=True)

    exporter.close()
    print(f"Output: {fp}")


if __name__ == "__main__":
    run_v5()
