"""
ex03_optimized_v3.py
====================
Folding-beam convergence with KKT-equilibrated PARDISO solver.

CORE FIX (world-class convergence):
  1. _solve_linear_system now diagonal-equilibrates the KKT saddle-point matrix
     (|K_eff| ~10⁵ → 1,  |C| ~1 → 1) so PARDISO's pivot search sees uniform
     scaling instead of a 5-order-of-magnitude unit mismatch.
  2. PARDISO switched from mtype=11 (unsymmetric) to mtype=-2 (symmetric
     indefinite), enabling Bunch–Kaufman 1×1/2×2 pivot blocks that handle
     zero-diagonal LM rows without manual perturbation.
  3. Iterative refinement on the EQUILIBRATED system (iparm[8]=10, tol 1e-14)
     — better conditioned → refinement converges in fewer passes.
  4. solve_step adds a residual-based convergence criterion: if the KKT
     residual drops by 10+ orders vs the first iteration, accept the step
     even if du_norm/u_norm plateaus at ~0.006 (saddle-point geometry).

LM damping is kept as an **emergency-only** safety net: if the improved
solver still fails (solve_step returns < 0), per-DOF LM damping is activated.
At the exact bifurcation (det K_eff ≈ 0), even mtype=-2 can struggle —
LM shifts the zero eigenvalues, letting Newton "slide through".

WHY NOT:
  - Armijo line search: over-aggressive on indefinite KKT systems.
  - Modified Newton: destroys quadratic convergence.
  - Replacing solve_step entirely: fragile, prone to subtle bugs.
"""

import os
import numpy as np
from dispsolver.solver import DynamicSolver
from dispsolver.solver import dynamic as dyn_mod

# ============================================================
# Hyper-parameters
# ============================================================
N_CORES = 4

MAX_ITER     = 30
TOL          = 1e-2
T_TOTAL      = 1.0
DT_INITIAL   = 0.001
DT_MIN       = 5e-6
DT_MAX       = 0.05
MAX_CUTBACKS = 20
TARGET_ANGLE = 1.570796
FAST_ASSEMBLY = True

# LM damping (emergency only — see docstring above)
# Per-DOF scaling:  d_i  ->  d_i + _LM_ACTIVE * |d_i|
# Two-tier activation: first failure jumps straight to LM=0.5 (avoids
# wasting retries at 0.1→0.2→0.4).  After a successful step, LM decays
# SLOWLY (0.95×) and bottoms at 0.1 so it never dissipates entirely during
# the "easy" middle zone (Steps 7-10 in the folding beam).
# With residual-aware line search (dynamic.py), the solver can tolerate
# higher LM values without stalling — LM=1.0 is allowed as last resort.
LM_FIRST     = 0.5        # LM value on first failure (straight to max)
LM_GROW      = 1.5        # multiply LM on repeated failure
LM_MAX       = 1.0        # maximum LM (above this, cut dt instead)
LM_DECAY     = 0.95       # multiply LM after successful step (slow decay)
LM_FLOOR     = 0.2        # LM never drops below this during hard zone
LM_ITER_HIGH = 15         # if n_iter > this, keep LM active for next step

# Riks arc-length continuation (fallback when Newton fails near fold)
RIKS_ENABLED = True
RIKS_DS       = 0.005     # prescribed arc-length Δs
RIKS_PSI      = 1.0       # spherical constraint (1.0) vs load control (0.0)
RIKS_MAX_CORR = 15        # max corrector iterations
RIKS_TOL      = 1e-4      # arc-length constraint tolerance
RIKS_DS_MIN   = 5e-5      # minimum arc-length
RIKS_DS_MAX   = 0.02      # maximum arc-length

os.environ["MKL_NUM_THREADS"]     = str(N_CORES)
os.environ["PARDISO_NUM_THREADS"] = str(N_CORES)
os.environ["XLA_FLAGS"]           = "--xla_cpu_multi_thread_eigen=false"

from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, LinearViscoelastic
from dispsolver.constraint import RBE2HingeConstraint, PenaltyContactConstraint
from dispsolver.load import Amplitude
from dispsolver.export.vtkhdf_exporter import TransientVTKHDFExporter

# ----------------------------------------------------------------
# Emergency LM damping patch — ONLY activated when solve_step fails
# ----------------------------------------------------------------
_LM_ACTIVE = 0.0  # module-level: >0 activates LM damping

_orig_solve = dyn_mod._solve_linear_system

def _solve_lm(J, b, n_refine=4, tol=1e-14):
    """Original KKT-equilibrated solve + LM diagonal shift (emergency only).

    Per-DOF Levenberg-Marquardt:  d_i  ->  d_i + lambda * |d_i|.
    The KKT-equilibrated solver is tried first; LM is a last resort when
    the system is genuinely singular (det K_eff ≈ 0 at bifurcation).
    """
    global _LM_ACTIVE
    if _LM_ACTIVE > 0.0 and J.shape[0] == J.shape[1]:
        d = J.diagonal().copy()
        J = J.copy()
        J.setdiag(d + _LM_ACTIVE * np.abs(d))
    return _orig_solve(J, b, n_refine, tol)

dyn_mod._solve_linear_system = _solve_lm


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
def run_v3():
    global _LM_ACTIVE

    print("=" * 60)
    print("ex03 OPTIMIZED v3 -- KKT-equilibrated PARDISO + residual convergence")
    print("=" * 60)

    # 1. Mesh (identical to original)
    mesh = Mesh()
    xs_left = np.linspace(-40.0, -15.0, 21)[:-1]
    xs_mid  = np.linspace(-15.0, 15.0, 61)[:-1]
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
    mesh.add_node(99999, -15.0, 0.0)
    mesh.add_node(99998,  15.0, 0.0)

    # 2. Materials
    pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    psa_mat = LinearViscoelastic(E=10.0, nu=0.49, g_i=[0.8], tau_i=[1.0])
    materials = {layer: (pet_mat if layer % 2 == 0 else psa_mat) for layer in range(7)}

    # 3. Constraints
    tol = 1e-9
    slave_l = [i for i in range(nx) if xs[i] <= -15.0 + tol]
    slave_r = [i for i in range(nx) if xs[i] >=  15.0 - tol]
    rbe2_l = RBE2HingeConstraint(mesh, 99999, slave_l, extra_primal_offset=0)
    rbe2_r = RBE2HingeConstraint(mesh, 99998, slave_r, extra_primal_offset=1)
    c_nodes = list(range(nx)) + [(ny-1)*nx + i for i in range(nx)]
    contact_c = PenaltyContactConstraint(mesh, contact_nodes=c_nodes, k_contact=1e6, d_0=0.2)

    # 4. Solver
    elem_type = {layer: ("Q4_EAS" if layer % 2 == 0 else "Q4_UP") for layer in range(7)}
    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params={layer: {} for layer in range(7)},
        constraints=[rbe2_l, rbe2_r], penalty_constraints=[contact_c],
        max_iter=MAX_ITER, tol=TOL, verbose=True, element_type=elem_type,
        fast_assembly=FAST_ASSEMBLY, mode="moderate-2",
    )

    # 5. BCs with smooth amplitude
    nid_to_idx = mesh.node_id_to_index()
    rotation_ampl = SmoothAmplitude(0.0, T_TOTAL)
    idx_L, idx_R = nid_to_idx[99999], nid_to_idx[99998]
    idx_tL, idx_tR = solver.n_dofs, solver.n_dofs + 1
    bc_dofs = [idx_L*2, idx_L*2+1, idx_R*2, idx_R*2+1, idx_tL, idx_tR]
    bc_base = [0.0, 0.0, 0.0, 0.0, -TARGET_ANGLE, TARGET_ANGLE]
    bc_amps = [None, None, None, None, rotation_ampl, rotation_ampl]
    solver.set_prescribed_dofs(bc_dofs, bc_base, amplitudes=bc_amps)

    # 6. Time loop
    dt = DT_INITIAL
    step_count = 0
    cutbacks = 0
    total_iter = 0

    os.makedirs("output", exist_ok=True)
    fp = os.path.join("output", "ex03_fold_v3.vtkhdf")
    exporter = TransientVTKHDFExporter(fp, mesh)
    exporter.add_step(0.0, solver.u)

    print(f"\n{'='*100}")
    print(f" OPTIMIZED v3: KKT-equilibrated PARDISO + residual convergence + emergency LM")
    print(f" LM parameters: init={LM_FIRST} floor={LM_FLOOR} grow={LM_GROW}x max={LM_MAX} decay={LM_DECAY}x")
    print(f"{'='*100}")

    while solver.time < T_TOTAL - 1e-12:
        dt = min(dt, T_TOTAL - solver.time)
        factor = rotation_ampl(solver.time + dt)
        theta_L = -TARGET_ANGLE * factor
        theta_R =  TARGET_ANGLE * factor

        step_count += 1
        print(f"\n{'='*100}", flush=True)
        print(f" STEP {step_count}  t={solver.time:.5f}->{solver.time+dt:.5f} dt={dt:.3e} load={factor:.4f}", flush=True)
        print(f" Angles: L={theta_L*180/np.pi:.1f} R={theta_R*180/np.pi:.1f} LM_ACTIVE={_LM_ACTIVE:.2e}", flush=True)
        print(f"{'='*100}", flush=True)

        saved = solver.save_state()
        n_iter = solver.solve_step(dt)

        # --- Riks fallback: if Newton fails, try arc-length continuation ---
        if n_iter < 0 and RIKS_ENABLED:
            solver.restore_state(saved)
            ds_riks = RIKS_DS
            riks_result = solver.solve_step_riks(
                dt, ds_riks, psi=RIKS_PSI,
                max_corr=RIKS_MAX_CORR, tol=RIKS_TOL,
            )
            if riks_result > 0:
                n_iter = riks_result
                print(f"  => Riks arc-length rescued this step (ds={ds_riks:.4e})", flush=True)
            else:
                # Riks also failed — restore and let standard cutback handle it
                solver.restore_state(saved)

        if n_iter < 0:
            cutbacks += 1
            if cutbacks > MAX_CUTBACKS:
                print(f"\n  FATAL: {MAX_CUTBACKS} cutbacks at t={solver.time:.5f}", flush=True)
                break

            # Strategy: try LM first, cut dt only if LM maxed
            if _LM_ACTIVE < LM_MAX:
                if _LM_ACTIVE <= 0.0:
                    _LM_ACTIVE = LM_FIRST   # jump straight to 0.5
                elif _LM_ACTIVE * LM_GROW >= LM_MAX:
                    _LM_ACTIVE = LM_MAX     # clamp at max
                else:
                    _LM_ACTIVE = min(_LM_ACTIVE * LM_GROW, LM_MAX)
                print(f"  CUTBACK {cutbacks}: LM={_LM_ACTIVE:.3e}, keep dt={dt:.3e}", flush=True)
                solver.restore_state(saved)
                # Retry with same dt but LM active
                continue
            else:
                dt = max(dt * 0.5, DT_MIN)
                _LM_ACTIVE = 0.0
                print(f"  CUTBACK {cutbacks}: dt {dt*2:.3e}->{dt:.3e} (LM maxed, LM reset)", flush=True)
                solver.restore_state(saved)
                step_count -= 1
                continue

        # Step succeeded
        cutbacks = 0
        total_iter += (n_iter + 1)
        exporter.add_step(solver.time, solver.u)

        # Adaptive dt
        if n_iter <= 3:
            dt = min(dt * 1.5, DT_MAX)
        elif n_iter <= 8:
            pass
        elif n_iter <= 15:
            dt = max(dt * 0.85, DT_MIN)
        else:
            dt = max(dt * 0.7, DT_MIN)

        # LM management for next step
        if n_iter > LM_ITER_HIGH:
            # Hard step: ensure LM stays relevant for next step
            if _LM_ACTIVE <= 0.0:
                _LM_ACTIVE = LM_FIRST * 0.5
        elif _LM_ACTIVE > 0.0:
            # Easy step: slowly decay LM, but never below LM_FLOOR
            _LM_ACTIVE = max(_LM_ACTIVE * LM_DECAY, LM_FLOOR)

    else:
        print(f"\n{'='*100}", flush=True)
        print(f"  ALL DONE. Steps: {step_count} Cutbacks: {cutbacks} Total iter: {total_iter}", flush=True)
        print(f"{'='*100}", flush=True)

    exporter.close()
    print(f"Output: {fp}")


if __name__ == "__main__":
    run_v3()
