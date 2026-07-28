"""
mode_comparison.py
==================
Compare all 4 time-integration modes on the ex03 folding problem.

Modes:
  transient   — α= 0.00, Newmark trapezoidal (no damping)
  moderate-1  — α=-0.05, light HF damping
  moderate-2  — α=-0.15, stronger HF damping (ex03 default)
  quasistatic — no inertia/damping, statics + dt for viscoelastic rate

For each mode:
  - Runs N_STEPS steps of the ex03 folding problem
  - Reports: steps, cutbacks, NR iterations, wall time, fold angle
  - Displays a summary table at the end
"""

import os, sys, time, numpy as np

# ── Config ──
N_CORES = 8
N_STEPS = 10          # profile first 10 steps (avoid long wall-clock)
os.environ["MKL_NUM_THREADS"]     = str(N_CORES)
os.environ["PARDISO_NUM_THREADS"] = str(N_CORES)

from dispsolver.solver import DynamicSolver
from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, LinearViscoelastic
from dispsolver.constraint import RBE2HingeConstraint

# Ex03 SmoothAmplitude
class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
        self.T = t1 - t0
    def __call__(self, t):
        tau = np.clip((t - self.t0) / self.T, 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))

MODES = ["transient", "moderate-1", "moderate-2", "quasistatic"]
TARGET_ANGLE = 1.570796  # 90 degrees

def build_solver(mode: str):
    """Build the ex03 mesh, materials, constraints and DynamicSolver."""
    mesh = Mesh()
    xs_left  = np.linspace(-40.0, -15.0, 21)[:-1]
    xs_mid   = np.linspace(-15.0,  15.0, 61)[:-1]
    xs_right = np.linspace( 15.0,  40.0, 21)
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
    mesh.add_node(99999, -15.0, 0.0)
    mesh.add_node(99998,  15.0, 0.0)

    pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    psa_mat = LinearViscoelastic(E=100.0, nu=0.49, g_i=[0.8], tau_i=[1.0])
    materials = {layer: (pet_mat if layer % 2 == 0 else psa_mat) for layer in range(7)}

    slave_l = [j * nx + i for j in range(ny) for i in range(nx) if xs[i] < -15.0]
    slave_r = [j * nx + i for j in range(ny) for i in range(nx) if xs[i] >  15.0]
    rbe2_left  = RBE2HingeConstraint(mesh, 99999, slave_l, extra_primal_offset=0)
    rbe2_right = RBE2HingeConstraint(mesh, 99998, slave_r, extra_primal_offset=1)

    elem_type = {layer: ("Q4_EAS" if layer % 2 == 0 else "Q4_UP") for layer in range(7)}
    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params={layer: {} for layer in range(7)},
        constraints=[rbe2_left, rbe2_right],
        penalty_constraints=[],
        max_iter=30, tol=1e-2, atol=1e-7, verbose=False, element_type=elem_type,
        fast_assembly=True, mode=mode,
        viscous_stab_factor=2e-4,
    )
    solver.theta_penalty_k = 1e9
    solver.max_disp_limit = 200.0

    nid_to_idx = mesh.node_id_to_index()
    bc_dofs = [nid_to_idx[99999]*2, nid_to_idx[99999]*2+1,
               nid_to_idx[99998]*2, nid_to_idx[99998]*2+1]
    solver.set_prescribed_dofs(bc_dofs, [0.0, 0.0, 0.0, 0.0], amplitudes=None)

    return solver

def run_mode(mode: str) -> dict:
    """Run N_STEPS in the given mode, return summary stats."""
    solver = build_solver(mode)
    rotation_ampl = SmoothAmplitude(0.0, 1.0)

    dt = 1e-3
    step_count = 0
    cutbacks = 0
    total_iter = 0
    MAX_CUTBACKS = 20
    DT_MIN = 1e-6
    DT_MAX = 0.01
    T_TOTAL = 1.0

    t0 = time.perf_counter()

    while solver.time < T_TOTAL - 1e-12 and step_count < N_STEPS:
        dt = min(dt, T_TOTAL - solver.time)
        factor = rotation_ampl(solver.time + dt)
        step_count += 1
        theta_L_target = -TARGET_ANGLE * factor
        theta_R_target =  TARGET_ANGLE * factor
        solver.theta_targets = {0: theta_L_target, 1: theta_R_target}

        saved = solver.save_state()
        n_iter = solver.solve_step(dt)

        if n_iter < 0:
            cutbacks += 1
            if cutbacks > MAX_CUTBACKS:
                break
            dt = max(dt * 0.5, DT_MIN)
            solver.restore_state(saved)
            step_count -= 1
            continue

        cutbacks = 0
        total_iter += (n_iter + 1)

        if n_iter <= 3:
            dt = min(dt * 1.5, DT_MAX)
        elif n_iter <= 8:
            pass
        elif n_iter <= 15:
            dt = max(dt * 0.85, DT_MIN)
        else:
            dt = max(dt * 0.7, DT_MIN)

    wall = time.perf_counter() - t0
    return {
        "mode": mode,
        "wall_s": wall,
        "steps": step_count,
        "cutbacks": cutbacks,
        "total_iter": total_iter,
        "avg_iter_per_step": total_iter / max(step_count, 1),
        "theta_L": float(solver.u_extra[0]),
        "theta_R": float(solver.u_extra[1]),
        "theta_deg": float(solver.u_extra[0]) * 180 / np.pi,
    }


if __name__ == "__main__":
    print("=" * 80)
    print("  Mode Comparison - ex03 folding problem")
    print(f"  {N_STEPS} steps per mode  |  N_CORES={N_CORES}")
    print("=" * 80)

    results = []
    for mode in MODES:
        print(f"\n  Running mode={mode!r} ... ", end="", flush=True)
        r = run_mode(mode)
        results.append(r)
        print(f" done ({r['wall_s']:.2f}s)")

    # Summary table
    print("\n" + "=" * 110)
    print(f"  {'Mode':<15s} {'Steps':<7s} {'Cutbacks':<10s} {'NR Iter':<8s} {'Avg/Step':<10s} "
          f"{'Wall (s)':<10s} {'θ_L (°)':<10s}")
    print("  " + "-" * 100)
    for r in results:
        print(f"  {r['mode']:<15s} {r['steps']:<7d} {r['cutbacks']:<10d} {r['total_iter']:<8d} "
              f"{r['avg_iter_per_step']:<10.2f} {r['wall_s']:<10.2f} {r['theta_deg']:<10.2f}")
    print("=" * 110)
    print()
