"""
ex03_v4_rbe2_element.py
========================
U-bend folding simulation — RBE2 element (static condensation) instead of KKT constraint.

KEY DIFFERENCES FROM v3-1 (KKT constraint):
  1. RBE2HingeElement replaces RBE2HingeConstraint — penalty-stabilised
     static condensation at element level, no KKT saddle-point system.
  2. No Lagrange multipliers (n_extra=0, n_lambdas=0) → purely
     displacement-based global system.
  3. Folding driven by prescribed wing-tip displacement (not θ BC).
  4. No force_atol=1e9 workaround needed — natural convergence.
"""

import os
import numpy as np
from dispsolver.solver import DynamicSolver

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
TARGET_ANGLE = 1.570796  # ~90°
FAST_ASSEMBLY = True

os.environ["MKL_NUM_THREADS"]     = str(N_CORES)
os.environ["PARDISO_NUM_THREADS"] = str(N_CORES)
os.environ["XLA_FLAGS"]           = "--xla_cpu_multi_thread_eigen=false"

from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, NeoHookean
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.element.rbe2 import RBE2HingeElement
from dispsolver.load import Amplitude
from dispsolver.export.vtkhdf_exporter import TransientVTKHDFExporter


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
def run_v4():
    print("=" * 60)
    print("ex03 v4 -- RBE2 element (static condensation)")
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

    # We need sequential node IDs for RBE2 element (which uses node ID as array index).
    # Build a mapping from (j, i) to sequential node ID.
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

    # 3. RBE2 elements — hinge on bottom row (j=0), wings (x < -10 and x > 10)
    slave_L_ids = [nid_map[(0, i)] for i in range(nx) if xs[i] < -10.0]
    slave_R_ids = [nid_map[(0, i)] for i in range(nx) if xs[i] >  10.0]

    rbe2_left  = RBE2HingeElement(master_L, slave_L_ids, coords_all)
    rbe2_right = RBE2HingeElement(master_R, slave_R_ids, coords_all)

    # 4. Solver
    elem_type = {layer: ("Q4_EAS" if layer % 2 == 0 else "Q4_VISCO_SIMO") for layer in range(7)}
    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params=material_params,
        constraints=[],
        rbe2_elements=[rbe2_left, rbe2_right],
        max_iter=MAX_ITER, tol=TOL, atol=TOL,
        verbose=True, element_type=elem_type,
        fast_assembly=FAST_ASSEMBLY, mode="quasistatic",
    )

    # 5. BCs: fix hinge masters, fix centre-line UX=0
    rotation_ampl = SmoothAmplitude(0.0, T_TOTAL)

    bc_dofs = [
        nid_to_idx[master_L]*2, nid_to_idx[master_L]*2+1,   # master_L Ux, Uy
        nid_to_idx[master_R]*2, nid_to_idx[master_R]*2+1,   # master_R Ux, Uy
    ]
    bc_base = [0.0, 0.0, 0.0, 0.0]

    # Centre-line UX=0 (symmetry)
    for j in range(ny):
        for i in range(nx):
            if abs(xs[i]) < 0.01:
                node_idx = nid_to_idx[nid_map[(j, i)]]
                bc_dofs.append(node_idx * 2)  # UX
                bc_base.append(0.0)

    solver.set_prescribed_dofs(bc_dofs, bc_base)

    # 6. Time loop — drive folding by applying moment via vertical tip forces
    dt = DT_INITIAL
    step_count = 0
    cutbacks = 0
    total_iter = 0

    # Wing-tip nodes for load application (bottom row, wing tips)
    tip_nodes_L = [nid_map[(0, i)] for i in range(nx) if xs[i] < -10.0]
    tip_nodes_R = [nid_map[(0, i)] for i in range(nx) if xs[i] >  10.0]

    os.makedirs("output", exist_ok=True)
    fp = os.path.join("output", "ex03_fold_rbe2_v4.vtkhdf")
    exporter = TransientVTKHDFExporter(fp, mesh)
    exporter.add_step(0.0, solver.u)

    print(f"\n{'='*100}")
    print(f" RBE2 ELEMENT: Static condensation (θ condensed at element level)")
    print(f" Drive: wing-tip vertical force")
    print(f"{'='*100}")

    while solver.time < T_TOTAL - 1e-12:
        dt = min(dt, T_TOTAL - solver.time)
        factor = rotation_ampl(solver.time + dt)

        # Apply vertical tip load (scaled by amplitude)
        # Force magnitude: increasing with amplitude to produce gradual folding
        force_mag = 200.0 * factor
        solver.apply_load(
            [nid_to_idx[n]*2+1 for n in tip_nodes_L + tip_nodes_R],
            [-force_mag] * len(tip_nodes_L) + [-force_mag] * len(tip_nodes_R)
        )

        step_count += 1
        print(f"\n{'='*100}", flush=True)
        print(f" STEP {step_count}  t={solver.time:.5f}->{solver.time+dt:.5f} "
              f"load={factor:.4f}  force={force_mag:.1f}", flush=True)
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
        if step_count % 5 == 0:
            print(f"  [FOLD] max|u|={umax:.2f}mm  tip_L_UY={tip_UY_L:.3f}  "
                  f"tip_R_UY={tip_UY_R:.3f}", flush=True)

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
    print(f"  RBE2 element drive  max|u|={umax:.3f} mm", flush=True)

    exporter.close()
    print(f"Output: {fp}")


if __name__ == "__main__":
    run_v4()
