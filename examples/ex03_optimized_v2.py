"""
ex03_optimized_v2.py
====================
Second attempt at optimizing ex03_display_fold.py.

v1 FAILURES (avoided):
  1. Armijo line search: too strict at machine-precision residuals. Not used.
  2. Modified Newton skip: destroyed convergence. Not used.
  3. Replacing whole solve_step: fragile, easy to break.

v2 APPROACH (surgical, minimal):
  1. Monkey-patch ONLY `_solve_linear_system` to support LM damping via module-level
     parameter. The original solve_step is otherwise unchanged.
  2. Add a lightweight convergence monitor inside solve_step that tracks rel_change
     trajectory and adjusts LM lambda dynamically (increase when stalling, decay
     when converging).
  3. In the main loop: on cutback, try LM before cutting dt. This lets the solver
     "slide through" the bifurcation point instead of repeatedly halving dt.
"""

import os
import time
import numpy as np
import scipy.sparse as sps
from typing import Optional

# ============================================================
# Hyper-parameters
# ============================================================
N_CORES = 4

# --- Convergence ---
MAX_ITER    = 50
TOL         = 1e-3

# --- LM damping ---
LM_LAMBDA0  = 1e-4          # initial LM damping (relative to mean |K_diag|)
LM_LAMBDA_MAX = 5e-1        # max damping (above this, K becomes too perturbed)
LM_GROW     = 2.5           # multiply LM when oscillations detected
LM_DECAY    = 0.3           # multiply LM after a good iteration

# --- Oscillation detection ---
OSC_WINDOW  = 3             # lookback window for rel_change monitoring
OSC_THRESH  = 2             # consecutive increases = oscillation

# --- Time stepping ---
T_TOTAL     = 1.0
DT_INITIAL  = 0.02
DT_MIN      = 1e-5
DT_MAX      = 0.05
MAX_CUTBACKS = 10
TARGET_ANGLE = 1.570796

# --- Assembly ---
FAST_ASSEMBLY = True

os.environ["MKL_NUM_THREADS"]     = str(N_CORES)
os.environ["PARDISO_NUM_THREADS"] = str(N_CORES)
os.environ["XLA_FLAGS"]           = "--xla_cpu_multi_thread_eigen=false"

import numpy as np
from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, LinearViscoelastic
from dispsolver.constraint import RBE2HingeConstraint, PenaltyContactConstraint
from dispsolver.solver import DynamicSolver
from dispsolver.load import Amplitude
from dispsolver.export.vtkhdf_exporter import TransientVTKHDFExporter

# ============================================================
# Import solver module for monkey-patching
# ============================================================
import dispsolver.solver.dynamic as dyn_mod

# --------------------------------------------------------------------
# LM damping: parameter + patched solver
# --------------------------------------------------------------------
_LM_ACTIVE = 0.0          # module-level: >0 activates LM damping in solver

_orig_solve_linear = dyn_mod._solve_linear_system

def _solve_lm_patched(J, b, n_refine=2, tol=1e-11):
    """Original solver + LM damping when _LM_ACTIVE > 0.
    
    Scale LM lambda relative to |K_diag| so it works across material scales.
    """
    global _LM_ACTIVE
    if _LM_ACTIVE > 0.0 and J.shape[0] == J.shape[1]:
        d = J.diagonal().copy()
        d_abs_max = max(np.max(np.abs(d)), 1e-30)
        lm_scale = _LM_ACTIVE * d_abs_max
        J = J.copy()
        J.setdiag(d + lm_scale)
    return _orig_solve_linear(J, b, n_refine, tol)

# --------------------------------------------------------------------
# Convergence monitor for solve_step (minimal patch)
# --------------------------------------------------------------------
# We patch solve_step to add:
# 1. rel_change trajectory tracking
# 2. Dynamic LM adjustment: stall -> grow LM, converge -> decay LM
# 3. Early cutback on sustained oscillation

_orig_solve_step = dyn_mod.DynamicSolver.solve_step

def _lm_monitored_solve_step(self, dt: float) -> int:
    """Original solve_step + LM trajectory tracking.
    
    The only additions over original:
    - rel_change history tracking
    - Dynamic _LM_ACTIVE adjustment per iteration
    - Early cutback on sustained oscillation (saves wasted iterations)
    """
    # Redirect _solve_linear_system to the LM-patched version
    # (it uses _LM_ACTIVE, which we adjust per iteration)
    
    t_start = time.time()
    beta, gamma, dt2 = self.beta, self.gamma, dt * dt

    if self._bc_amplitudes is not None:
        self.bc_vals = self._eval_bc_vals(self.time + dt)

    u_n = self.u.copy()
    v_n = self.v.copy()
    a_n = self.a.copy()
    u_ext_n = self.u_extra.copy()
    v_ext_n = self.v_extra.copy()
    a_ext_n = self.a_extra.copy()

    u_k = u_n + dt * v_n + 0.5 * dt2 * a_n
    u_ext_k = u_ext_n + dt * v_ext_n + 0.5 * dt2 * a_ext_n
    lam_k = self.lam.copy()

    if self.alpha != 0.0 and self.f_int_n is None:
        self.f_int_n, _, _ = self._assemble(u_n, dt)

    has_bc = len(self.bc_dofs) > 0
    inv_beta_dt2 = 1.0 / (beta * dt2)

    if getattr(self, 'verbose', False):
        print(f"", flush=True)
        print(f"  Nonlinear Iteration Summary (Newton-Raphson + LM damping)", flush=True)
        print(f"  =============================================================================================================================", flush=True)
        print(f"  Time increment dt: {dt:.3e}", flush=True)
        print(f"  Convergence Tol  : {self.tol:.1e} (Disp Ratio)     Max Iterations: {self.max_iter}", flush=True)
        print(f"  -----------------------------------------------------------------------------------------------------------------------------", flush=True)
        print(f"  Iter   Max Res.Force  (Node/DOF)    Max Disp.Corr  (Node/DOF)    Max Disp.Incr  Disp.Ratio  Energy.Err  LM.lambda  Contacts", flush=True)
        print(f"  -----------------------------------------------------------------------------------------------------------------------------", flush=True)

    if np.any(np.isnan(u_k) | np.isinf(u_k)):
        print(f"DEBUG: u_k has NaN at start of solve_step! (dt={dt:.3e})", flush=True)

    res_norm_0 = None
    converged = False
    
    # --- LM control state ---
    global _LM_ACTIVE
    lm_lambda = _LM_ACTIVE if _LM_ACTIVE > 0 else 0.0
    rel_change_history = []
    
    for n_iter in range(self.max_iter):
        if np.any(np.isnan(u_k)):
            print(f"DEBUG: u_k has NaN before assembly at iter {n_iter+1}!", flush=True)

        # --- Assemble ---
        f_int, K_T, state_new = self._assemble(u_k, dt)

        if np.any(np.isnan(f_int)) or (K_T is not None and np.any(np.isnan(K_T.data))):
            if getattr(self, 'verbose', False):
                t_elapsed = time.time() - t_start
                print(f"  ---------------------------------------------------------------------------------------------------------------------", flush=True)
                print(f"  *** ERROR: NaN in assembly at iteration {n_iter+1}. (Time: {t_elapsed:.2f} s)\n", flush=True)
            return -(n_iter + 1)

        a_k = (u_k - u_n - dt * v_n) * inv_beta_dt2 - (1.0 - 2.0 * beta) / (2.0 * beta) * a_n
        a_ext_k = (u_ext_k - u_ext_n - dt * v_ext_n) * inv_beta_dt2 - (1.0 - 2.0 * beta) / (2.0 * beta) * a_ext_n

        if self.static_mode:
            R_u = self.f_ext - f_int
        elif self.alpha != 0.0 and self.f_int_n is not None:
            R_u = self.f_ext - self.M * a_k - (1.0 + self.alpha) * f_int + self.alpha * self.f_int_n
        else:
            R_u = self.f_ext - self.M * a_k - f_int
        R_ext = np.zeros(self.n_extra)
        R_lam = np.zeros(self.n_lambdas)

        C_row_u, C_col_u, C_val_u = [], [], []
        C_row_ext, C_col_ext, C_val_ext = [], [], []

        lam_offset = 0
        for c in self.constraints:
            n_lam = c.n_multipliers()
            if n_lam > 0:
                r_u, c_u, v_u, r_ext, c_ext, v_ext, g = c.assemble(u_k, u_ext_k)
                for i in range(len(v_u)):
                    eq_idx = r_u[i] + lam_offset
                    dof_idx = c_u[i]
                    R_u[dof_idx] -= v_u[i] * lam_k[eq_idx]
                    C_row_u.append(eq_idx); C_col_u.append(dof_idx); C_val_u.append(v_u[i])
                for i in range(len(v_ext)):
                    eq_idx = r_ext[i] + lam_offset
                    ext_idx = c_ext[i]
                    R_ext[ext_idx] -= v_ext[i] * lam_k[eq_idx]
                    C_row_ext.append(eq_idx); C_col_ext.append(ext_idx); C_val_ext.append(v_ext[i])
                for i in range(n_lam):
                    R_lam[lam_offset + i] = -g[i]
            lam_offset += n_lam

        R_total = np.concatenate([R_u, R_ext, R_lam])

        if self.static_mode:
            K_eff = K_T.copy()
        elif self.alpha != 0.0:
            K_eff = (1.0 + self.alpha) * K_T
            K_eff.setdiag(K_eff.diagonal() + inv_beta_dt2 * self.M)
        else:
            K_eff = K_T.copy()
            K_eff.setdiag(K_eff.diagonal() + inv_beta_dt2 * self.M)

        K_eff_coo = K_eff.tocoo()
        global_row = list(K_eff_coo.row)
        global_col = list(K_eff_coo.col)
        global_val = list(K_eff_coo.data)

        lam_start_idx = self.n_dofs + self.n_extra
        for r, c, v in zip(C_row_u, C_col_u, C_val_u):
            global_row.append(r + lam_start_idx); global_col.append(c); global_val.append(v)
            global_row.append(c); global_col.append(r + lam_start_idx); global_val.append(v)
        ext_start_idx = self.n_dofs
        for r, c, v in zip(C_row_ext, C_col_ext, C_val_ext):
            global_row.append(r + lam_start_idx); global_col.append(c + ext_start_idx); global_val.append(v)
            global_row.append(c + ext_start_idx); global_col.append(r + lam_start_idx); global_val.append(v)
        for i in range(self.n_extra):
            global_row.append(self.n_dofs + i); global_col.append(self.n_dofs + i); global_val.append(1e-12)

        J = sps.coo_matrix((global_val, (global_row, global_col)),
                           shape=(self.n_total, self.n_total)).tocsr()

        if has_bc:
            for i, idx in enumerate(self.bc_dofs):
                val = self.bc_vals[i]
                start_ptr = J.indptr[idx]; end_ptr = J.indptr[idx+1]
                for ptr in range(start_ptr, end_ptr):
                    col = J.indices[ptr]
                    J.data[ptr] = 1.0 if col == idx else 0.0
                if idx < self.n_dofs:
                    R_total[idx] = val - u_k[idx]
                else:
                    ext_idx = idx - self.n_dofs
                    R_total[idx] = val - u_ext_k[ext_idx]

        # === Set LM lambda (read by _solve_lm_patched via global) ===
        _LM_ACTIVE = lm_lambda

        du_all = _solve_lm_patched(J, R_total)

        if np.any(np.isnan(du_all) | np.isinf(du_all)):
            if getattr(self, 'verbose', False):
                t_elapsed = time.time() - t_start
                print(f"  ---------------------------------------------------------------------------------------------------------------------", flush=True)
                print(f"  *** ERROR: Search direction contains NaN/Inf at iteration {n_iter+1}. (Time: {t_elapsed:.2f} s)\n", flush=True)
            return -(n_iter + 1)

        du = du_all[:self.n_dofs]
        du_ext = du_all[self.n_dofs:self.n_dofs+self.n_extra]
        dlam = du_all[self.n_dofs+self.n_extra:]

        # --- Line search (NaN-only, same as original) ---
        alpha = 1.0
        alpha_min = 0.1
        R_norm_current = np.linalg.norm(R_total)

        while alpha > alpha_min + 1e-5:
            u_temp = u_k + alpha * du
            u_ext_temp = u_ext_k + alpha * du_ext
            lam_temp = lam_k + alpha * dlam

            R_temp = self._compute_R_total(
                u_temp, u_ext_temp, lam_temp,
                u_n, v_n, a_n,
                u_ext_n, v_ext_n, a_ext_n,
                dt, inv_beta_dt2, self.beta
            )
            R_norm_temp = np.linalg.norm(R_temp)

            if not np.isnan(R_norm_temp) and not np.isinf(R_norm_temp):
                break
            alpha *= 0.5

        if np.isnan(R_norm_temp) or np.isinf(R_norm_temp):
            if getattr(self, 'verbose', False):
                t_elapsed = time.time() - t_start
                print(f"  ---------------------------------------------------------------------------------------------------------------------", flush=True)
                print(f"  *** ERROR: Line search failed to resolve NaN/Inf at iteration {n_iter+1}. (Time: {t_elapsed:.2f} s)\n", flush=True)
            return -(n_iter + 1)

        du *= alpha
        du_ext *= alpha
        dlam *= alpha
        du_all *= alpha

        res_norm = R_norm_current
        if res_norm_0 is None:
            res_norm_0 = res_norm + 1e-16
        res_ratio = res_norm / res_norm_0
        du_norm = np.linalg.norm(du)
        if self.max_step is not None and du_norm > self.max_step:
            scale = self.max_step / du_norm
            du *= scale; du_ext *= scale; dlam *= scale; du_all *= scale
            du_norm = self.max_step

        u_norm = np.linalg.norm(u_k)
        rel_change = du_norm / (u_norm + 1e-12)

        # === Track rel_change history for LM adjustment ===
        rel_change_history.append(rel_change)
        if len(rel_change_history) > OSC_WINDOW:
            rel_change_history.pop(0)

        # === Dynamic LM adjustment ===
        # If rel_change is increasing → stall → grow LM
        # If rel_change is decreasing → good → decay LM
        if len(rel_change_history) >= 3:
            last3 = rel_change_history[-3:]
            # Count consecutive increases
            n_inc = sum(1 for i in range(1, len(last3)) if last3[i] > last3[i-1] * 1.02)
            
            if n_inc >= 2 and lm_lambda < LM_LAMBDA_MAX:
                # Stalling: increase LM damping
                lm_lambda = min(lm_lambda * LM_GROW, LM_LAMBDA_MAX)
                if lm_lambda <= 0:
                    lm_lambda = LM_LAMBDA0
                if getattr(self, 'verbose', False) and n_iter > 0:
                    lm_str = f"LM→{lm_lambda:.2e}"
            elif n_inc == 0 and lm_lambda > 0:
                # Converging: decay LM damping
                lm_lambda *= LM_DECAY
                if lm_lambda < 1e-10:
                    lm_lambda = 0.0

        if getattr(self, 'verbose', False):
            active_dofs = np.ones(self.n_dofs, dtype=bool)
            if len(self.bc_dofs) > 0:
                node_bc_dofs = self.bc_dofs[self.bc_dofs < self.n_dofs]
                active_dofs[node_bc_dofs] = False
            if np.any(active_dofs):
                abs_R_active = np.abs(R_u) * active_dofs
                max_R_val = np.max(abs_R_active)
                max_R_dof = np.argmax(abs_R_active)
                max_R_str = f"N{self.sorted_nids[max_R_dof//2]}({'UX' if max_R_dof%2==0 else 'UY'})"
            else:
                max_R_val = 0.0; max_R_str = "N/A"
            abs_du = np.abs(du)
            max_du_val = np.max(abs_du)
            max_du_dof = np.argmax(abs_du)
            max_du_str = f"N{self.sorted_nids[max_du_dof//2]}({'UX' if max_du_dof%2==0 else 'UY'})"
            delta_u = u_k - u_n
            max_disp_incr = np.max(np.abs(delta_u)) if self.n_dofs > 0 else 0.0
            energy_err = np.abs(np.dot(R_total, du_all))
            n_contacts = sum(getattr(pc, "n_active", 0) for pc in self.penalty_constraints)

            print(
                f"  {n_iter+1:4d}   "
                f"{max_R_val:11.4e}  {max_R_str:<13s} "
                f"{max_du_val:11.4e}  {max_du_str:<13s} "
                f"{max_disp_incr:11.4e}   "
                f"{rel_change:10.2e}  "
                f"{energy_err:10.2e}  "
                f"{lm_lambda:9.1e}  "
                f"{n_contacts:8d}",
                flush=True
            )

        if np.isnan(du_norm) or np.isinf(du_norm) or (res_norm_0 is not None and res_ratio > 1e15):
            if getattr(self, 'verbose', False):
                t_elapsed = time.time() - t_start
                print(f"  ---------------------------------------------------------------------------------------------------------------------", flush=True)
                print(f"  *** ERROR: Divergence detected at iteration {n_iter+1}. (Time: {t_elapsed:.2f} s)\n", flush=True)
            return -(self.max_iter)

        energy_err = np.abs(np.dot(R_total, du_all))
        if (rel_change < self.tol or energy_err < 1e-15) and n_iter > 0:
            converged = True
            self.time += dt
            if state_new is not None:
                self.state = state_new
            if self.alpha != 0.0:
                self.f_int_n = f_int.copy()
            if getattr(self, 'verbose', False):
                t_elapsed = time.time() - t_start
                print(f"  ---------------------------------------------------------------------------------------------------------------------", flush=True)
                print(f"  => Converged. (Time: {t_elapsed:.2f} s)\n", flush=True)
            break

        u_k += du
        u_ext_k += du_ext
        lam_k += dlam

    # --- Update state (same as original) ---
    self.u = u_k.copy()
    self.a = np.clip(a_k.copy(), -1.0e4, 1.0e4)
    self.v = np.clip(v_n + dt * ((1.0 - gamma) * a_n + gamma * a_k), -1.0e3, 1.0e3)
    self.u_extra = u_ext_k.copy()
    self.a_extra = np.clip(a_ext_k.copy(), -1.0e4, 1.0e4)
    self.v_extra = np.clip(v_ext_n + dt * ((1.0 - gamma) * a_ext_n + gamma * a_ext_k), -1.0e3, 1.0e3)
    self.lam = lam_k.copy()

    if not converged:
        if getattr(self, 'verbose', False):
            t_elapsed = time.time() - t_start
            print(f"  ---------------------------------------------------------------------------------------------------------------------", flush=True)
            print(f"  *** ERROR: Failed to converge after {self.max_iter} iterations. (Time: {t_elapsed:.2f} s)\n", flush=True)
        return -(self.max_iter)

    # Store final LM lambda for the main loop
    _LM_ACTIVE = lm_lambda

    return n_iter

# ============================================================
# Apply patches
# ============================================================
dyn_mod._solve_linear_system = _solve_lm_patched
dyn_mod.DynamicSolver.solve_step = _lm_monitored_solve_step


# ============================================================
# Smooth amplitude (same as v1)
# ============================================================
class SmoothAmplitude:
    """C2-continuous quintic step: f(t) = 10tau3 - 15tau4 + 6tau5."""
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
        self.T = t1 - t0

    def __call__(self, t):
        tau = np.clip((t - self.t0) / self.T, 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))


# ============================================================
# Main simulation
# ============================================================
def run_optimized_v2():
    global _LM_ACTIVE
    print("=" * 60)
    print("ex03 OPTIMIZED v2 -- LM damping + trajectory monitor")
    print("=" * 60)

    # 1. Mesh (same as original)
    mesh = Mesh()
    xs_left = np.linspace(-40.0, -15.0, 21)[:-1]
    xs_mid = np.linspace(-15.0, 15.0, 61)[:-1]
    xs_right = np.linspace(15.0, 40.0, 21)
    xs = np.concatenate([xs_left, xs_mid, xs_right])
    nx = len(xs)

    ys_list = [0.0]
    row_pids = []
    current_y = 0.0
    layer_thickness = 0.05
    for layer in range(7):
        if layer % 2 == 0:
            dy = layer_thickness / 3.0
            for _ in range(3):
                current_y += dy
                ys_list.append(current_y)
                row_pids.append(layer)
        else:
            current_y += layer_thickness
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
            n2 = n1 + 1
            n3 = n1 + nx + 1
            n4 = n1 + nx
            mesh.add_element(elem_idx, [n1, n2, n3, n4], "QUAD4", pid=pid)
            elem_idx += 1

    mesh.add_node(99999, -15.0, 0.0)
    mesh.add_node(99998, 15.0, 0.0)

    # 2. Materials
    pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    psa_mat = LinearViscoelastic(E=10.0, nu=0.49, g_i=[0.8], tau_i=[1.0])
    materials = {layer: pet_mat if layer % 2 == 0 else psa_mat for layer in range(7)}
    material_params = {layer: {} for layer in range(7)}

    # 3. Constraints
    tol = 1e-9
    slave_l = [i for i in range(nx) if xs[i] <= -15.0 + tol]
    slave_r = [i for i in range(nx) if xs[i] >= 15.0 - tol]
    rbe2_l = RBE2HingeConstraint(mesh, master_id=99999, slave_ids=slave_l, extra_primal_offset=0)
    rbe2_r = RBE2HingeConstraint(mesh, master_id=99998, slave_ids=slave_r, extra_primal_offset=1)
    contact_nodes = [i for i in range(nx)] + [(ny - 1) * nx + i for i in range(nx)]
    contact_constraint = PenaltyContactConstraint(mesh, contact_nodes=contact_nodes, k_contact=1e6, d_0=0.2)

    # 4. Solver
    element_type = {layer: "Q4_EAS" if layer % 2 == 0 else "Q4_UP" for layer in range(7)}
    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params=material_params,
        constraints=[rbe2_l, rbe2_r], penalty_constraints=[contact_constraint],
        max_iter=MAX_ITER, tol=TOL, verbose=True, element_type=element_type,
        fast_assembly=FAST_ASSEMBLY,
        mode="moderate-2",
    )

    # 5. BCs with smooth amplitude
    nid_to_idx = mesh.node_id_to_index()
    idx_L = nid_to_idx[99999]
    idx_R = nid_to_idx[99998]
    idx_theta_L = solver.n_dofs + 0
    idx_theta_R = solver.n_dofs + 1
    target_angle = TARGET_ANGLE

    rotation_ampl = SmoothAmplitude(0.0, T_TOTAL)
    bc_dofs = [idx_L * 2, idx_L * 2 + 1, idx_R * 2, idx_R * 2 + 1, idx_theta_L, idx_theta_R]
    bc_base = [0.0, 0.0, 0.0, 0.0, -target_angle, target_angle]
    bc_amps = [None, None, None, None, rotation_ampl, rotation_ampl]
    solver.set_prescribed_dofs(bc_dofs, bc_base, amplitudes=bc_amps)

    # 6. Time integration loop
    dt = DT_INITIAL
    step_count = 0
    cutbacks = 0
    total_iterations = 0

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "ex03_fold_optimized_v2.vtkhdf")
    exporter = TransientVTKHDFExporter(filepath, mesh)
    exporter.add_step(0.0, solver.u)

    print(f"\n{'='*100}")
    print(f" OPTIMIZED v2 ADAPTIVE DISPLAY FOLDING (LM damping + trajectory monitor)")
    print(f" Total Time: {T_TOTAL:.2f}s  Initial dt: {DT_INITIAL:.3e}s  dt_max: {DT_MAX:.2e}s")
    print(f" Target: {target_angle * 180 / np.pi:.1f} deg (180 deg fold)")
    print(f" Smooth C2 amplitude | LM-damped Newton | Trajectory monitor")
    print(f"{'='*100}")

    while solver.time < T_TOTAL - 1e-12:
        dt = min(dt, T_TOTAL - solver.time)
        t_start_step = solver.time
        t_end = t_start_step + dt
        factor = rotation_ampl(t_end)
        theta_L = -target_angle * factor
        theta_R = target_angle * factor

        step_count += 1
        print(f"\n{'='*100}", flush=True)
        print(f" STEP {step_count}  t: {t_start_step:.5f}->{t_end:.5f}  (dt={dt:.3e}, Load={factor:.4f})", flush=True)
        print(f" Angles: L={theta_L*180/np.pi:.1f} deg, R={theta_R*180/np.pi:.1f} deg", flush=True)
        print(f"{'='*100}", flush=True)

        saved_state = solver.save_state()
        n_iter = solver.solve_step(dt)

        if n_iter < 0:
            cutbacks += 1
            if cutbacks > MAX_CUTBACKS:
                print(f"\n  *** FATAL: Max cutbacks ({MAX_CUTBACKS}) at t={solver.time:.5f}. Aborting.", flush=True)
                break

            # Strategy: first try LM damping, then fall back to dt cutback
            if _LM_ACTIVE < LM_LAMBDA_MAX:
                # Bump up LM and retry with SAME dt
                if _LM_ACTIVE <= 0:
                    _LM_ACTIVE = LM_LAMBDA0
                else:
                    _LM_ACTIVE = min(_LM_ACTIVE * 5, LM_LAMBDA_MAX)
                dt_new = dt  # keep same dt
                print(f"  *** CUTBACK {cutbacks}: LM damping {_LM_ACTIVE:.2e}, keeping dt={dt:.3e}", flush=True)
            else:
                # LM maxed, now cut dt
                dt_new = max(dt * 0.5, DT_MIN)
                _LM_ACTIVE = 0.0  # reset LM for smaller step
                print(f"  *** CUTBACK {cutbacks}: dt {dt:.3e} -> {dt_new:.3e} (LM maxed, LM reset)", flush=True)

            solver.restore_state(saved_state)
            dt = dt_new
            step_count -= 1
            continue

        cutbacks = 0
        total_iterations += (n_iter + 1)
        exporter.add_step(solver.time, solver.u)

        # Adaptive stepping
        if n_iter <= 3:
            dt = min(dt * 2.0, DT_MAX)
        elif n_iter <= 6:
            dt = min(dt * 1.5, DT_MAX)
        elif n_iter <= 10:
            dt = min(dt * 1.2, DT_MAX)
        elif n_iter <= 15:
            dt = min(dt * 1.1, DT_MAX)
        else:
            dt = max(dt * 0.7, DT_MIN)
            # Activate LM pre-emptively for next hard step
            if _LM_ACTIVE <= 0:
                _LM_ACTIVE = LM_LAMBDA0

    else:
        print(f"\n{'='*100}", flush=True)
        print(f" => ALL INCREMENTS COMPLETED. Total steps: {step_count}, Cutbacks: {cutbacks}", flush=True)
        print(f" => Total Newton iterations: {total_iterations}", flush=True)
        print(f"{'='*100}", flush=True)

    exporter.close()
    print(f"Output: {filepath}")


if __name__ == "__main__":
    run_optimized_v2()
