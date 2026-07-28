"""
profile_solver.py
=================
Instrumented profiling of ex03_optimized_v3 solver.
Monkey-patches DynamicSolver to time: assembly, PARDISO solve, constraint assembly,
line-search residual computation, and Newton iteration overhead.
Runs for a limited number of steps (10) to collect timing data.
"""

import os, sys, time, numpy as np
from collections import defaultdict

# ── Configure environment ──
N_CORES = 8
os.environ["MKL_NUM_THREADS"] = str(N_CORES)
os.environ["PARDISO_NUM_THREADS"] = str(N_CORES)

# ── Import solver ──
from dispsolver.solver import DynamicSolver
from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, LinearViscoelastic
from dispsolver.constraint import RBE2HingeConstraint

# ── Timing collectors ──
timers = defaultdict(float)
counters = defaultdict(int)

def _t(name):
    """Decorator factory: wraps a method to accumulate timing."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            dt = time.perf_counter() - t0
            timers[name] += dt
            counters[name] += 1
            return result
        return wrapper
    return decorator

# ── Build mesh (same as ex03_optimized_v3) ──
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

# ── Materials ──
pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
psa_mat = LinearViscoelastic(E=100.0, nu=0.49, g_i=[0.8], tau_i=[1.0])
materials = {layer: (pet_mat if layer % 2 == 0 else psa_mat) for layer in range(7)}

# ── Constraints ──
slave_l = [j * nx + i for j in range(ny) for i in range(nx) if xs[i] < -15.0]
slave_r = [j * nx + i for j in range(ny) for i in range(nx) if xs[i] >  15.0]
rbe2_left  = RBE2HingeConstraint(mesh, 99999, slave_l, extra_primal_offset=0)
rbe2_right = RBE2HingeConstraint(mesh, 99998, slave_r, extra_primal_offset=1)

# ── Build solver ──
elem_type = {layer: ("Q4_EAS" if layer % 2 == 0 else "Q4_UP") for layer in range(7)}
solver = DynamicSolver(
    mesh, materials, rho=1000.0, material_params={layer: {} for layer in range(7)},
    constraints=[rbe2_left, rbe2_right],
    penalty_constraints=[],
    max_iter=30, tol=1e-2, atol=1e-7, verbose=True, element_type=elem_type,
    fast_assembly=True, mode="quasistatic",
    viscous_stab_factor=2e-4,
)

# θ penalty drive
solver.theta_penalty_k = 1e9
solver.max_disp_limit = 200.0

nid_to_idx = mesh.node_id_to_index()
bc_dofs = [nid_to_idx[99999]*2, nid_to_idx[99999]*2+1,
           nid_to_idx[99998]*2, nid_to_idx[99998]*2+1]
bc_base = [0.0, 0.0, 0.0, 0.0]
solver.set_prescribed_dofs(bc_dofs, bc_base, amplitudes=None)

# ── Smooth amplitude ──
class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
        self.T = t1 - t0
    def __call__(self, t):
        tau = np.clip((t - self.t0) / self.T, 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))

rotation_ampl = SmoothAmplitude(0.0, 1.0)
TARGET_ANGLE = 1.570796  # 90 degrees

# ──────────────────────────────────────────────────────────────────
# ══ INSTRUMENTATION ══
# ──────────────────────────────────────────────────────────────────

# 1. Patch _solve_linear_system (PARDISO solve)
import dispsolver.solver.dynamic as _dyn_mod
_orig_solve = _dyn_mod._solve_linear_system
def _timed_solve(J, b, n_refine=4, tol=1e-14):
    t0 = time.perf_counter()
    result = _orig_solve(J, b, n_refine, tol)
    dt = time.perf_counter() - t0
    timers["pardiso_solve"] += dt
    counters["pardiso_solve"] += 1
    return result
_dyn_mod._solve_linear_system = _timed_solve

# 2. Patch _assemble (element assembly) via class method
_orig_assemble = DynamicSolver._assemble
def _timed_assemble(self, u, dt=None):
    t0 = time.perf_counter()
    f_int, K_T, state_new = _orig_assemble(self, u, dt)
    dt_elapsed = time.perf_counter() - t0
    timers["assembly"] += dt_elapsed
    counters["assembly"] += 1
    return f_int, K_T, state_new
DynamicSolver._assemble = _timed_assemble

# 3. Patch _assemble_multi_material_batch (if used)
_orig_batch = DynamicSolver._assemble_multi_material_batch
def _timed_batch(self, u, dt=None):
    t0 = time.perf_counter()
    f_int, K_T, state_new = _orig_batch(self, u, dt)
    dt = time.perf_counter() - t0
    timers["batch_assembly"] += dt
    counters["batch_assembly"] += 1
    return f_int, K_T, state_new
DynamicSolver._assemble_multi_material_batch = _timed_batch

# 4. Patch constraint assemble (RBE2) via class method
_orig_c_assemble = RBE2HingeConstraint.assemble
def _timed_c_assemble(self, u, u_ext):
    t0 = time.perf_counter()
    result = _orig_c_assemble(self, u, u_ext)
    dt = time.perf_counter() - t0
    timers["constraint_assemble"] += dt
    counters["constraint_assemble"] += 1
    return result
RBE2HingeConstraint.assemble = _timed_c_assemble

# 5. Patch _compute_R_total (line-search residual) via class method
_orig_R = DynamicSolver._compute_R_total
def _timed_R(self, u_k, u_ext_k, lam_k, u_n, v_n, a_n, u_ext_n, v_ext_n, a_ext_n, dt, inv_beta_dt2, beta):
    t0 = time.perf_counter()
    result = _orig_R(self, u_k, u_ext_k, lam_k, u_n, v_n, a_n, u_ext_n, v_ext_n, a_ext_n, dt, inv_beta_dt2, beta)
    dt = time.perf_counter() - t0
    timers["line_search_residual"] += dt
    counters["line_search_residual"] += 1
    return result
DynamicSolver._compute_R_total = _timed_R

# 6. Wrap constraint extra_geometric_stiffness — store original bound method
for c in solver.constraints:
    if hasattr(c, "extra_geometric_stiffness"):
        _orig_egs = c.extra_geometric_stiffness  # already bound
        def _make_timed_egs(orig_bound):
            def timed_egs(u_ext, lam):
                t0 = time.perf_counter()
                r = orig_bound(u_ext, lam)
                dt = time.perf_counter() - t0
                timers["extra_geo_stiffness"] += dt
                counters["extra_geo_stiffness"] += 1
                return r
            return timed_egs
        c.extra_geometric_stiffness = _make_timed_egs(_orig_egs)

# ──────────────────────────────────────────────────────────────────
# ══ RUN ══
# ──────────────────────────────────────────────────────────────────

N_STEPS = 10  # profile first 10 steps
dt = 1e-3
step_count = 0
cutbacks = 0
total_iter = 0
MAX_CUTBACKS = 20
T_TOTAL = 1.0
DT_MIN = 1e-6
DT_MAX = 0.01

# Additional timers
overall_t0 = time.perf_counter()

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
            print(f"\n  FATAL: {MAX_CUTBACKS} cutbacks at t={solver.time:.5f}", flush=True)
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

total_wall = time.perf_counter() - overall_t0

# ──────────────────────────────────────────────────────────────────
# ══ REPORT ══
# ──────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  TIMING BREAKDOWN -- ex03_optimized_v3 profile")
print("="*70)
print(f"\n  Total wall time:     {total_wall:.2f} s")
print(f"  Steps completed:     {step_count}/{N_STEPS}")
print(f"  Total NR iterations: {total_iter}")
print(f"  Avg time per step:   {total_wall/max(step_count,1):.3f} s")
print(f"  Avg time per NR it:  {total_wall/max(total_iter,1):.3f} s")
print()

# Sort by total time
sorted_items = sorted(timers.items(), key=lambda x: -x[1])
print(f"  {'Timer':<30s} {'Total (s)':<12s} {'Count':<8s} {'Avg (ms)':<10s} {'% of wall':<10s}")
print(f"  {'-'*70}")
for name, total in sorted_items:
    cnt = counters[name]
    avg_ms = total / max(cnt,1) * 1000
    pct = total / max(total_wall,1e-12) * 100
    print(f"  {name:<30s} {total:<12.3f} {cnt:<8d} {avg_ms:<10.2f} {pct:<9.1f}%")

print(f"\n  Memory diagnostics:")
print(f"    n_elem:   {solver.n_elem}")
print(f"    n_dofs:   {solver.n_dofs}")
print(f"    n_total:  {solver.n_total}")
print(f"    n_extra:  {solver.n_extra}")
print(f"    n_lambda: {solver.n_lambdas}")
print(f"    element_type: {elem_type}")
print(f"    use_multi_material_batch: {solver.use_multi_material_batch}")
print(f"    use_jax_grouped_vmap: {solver.use_jax_grouped_vmap}")
print(f"    use_j2_batch: {solver.use_j2_batch}")
print(f"    N_CORES:  {N_CORES}")
print()
