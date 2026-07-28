"""
ex03_v9_mesh_refined.py
=======================
U-bend folding — UL + tanh-clustered mesh at hinge + thicker PSA.

Mesh improvements over v8:
  1. Tanh-clustered x-spacing at hinge region (±6mm around pivots at x=±3mm)
     → elements as small as ~0.07mm at pivot, vs uniform 0.5mm in v8
  2. PSA layers: 1→2 through-thickness elements for better shear resolution
  3. Hinge region: 60→80 elements for better curvature capture

This targets pushing past the 35° fold-angle barrier caused by Q4 element
inversion at the hinge (det(F) ≤ 0 at x≈±3mm when θ > 35°).
"""

import os
import numpy as np

os.environ["MKL_NUM_THREADS"] = "8"
os.environ["PARDISO_NUM_THREADS"] = "8"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"

from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, NeoHookean
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.solver import DynamicSolver
from dispsolver.export.vtkhdf_exporter import TransientVTKHDFExporter

# ── Parameters ────────────────────────────────────────────────────────────────
TARGET_ANGLE = np.pi / 2   # 90 degrees
T_TOTAL      = 1.0
DT_INITIAL   = 0.005
DT_MIN       = 1e-6
DT_MAX       = 0.02
MAX_ITER     = 60
TOL          = 2e-2
MAX_CUTBACKS = 50

LEFT_PIVOT   = np.array([-3.0, 0.0])
RIGHT_PIVOT  = np.array([ 3.0, 0.0])
SLAVE_THRESH = 3.5

# ── Tanh clustering helpers ───────────────────────────────────────────────────
def _tanh_nodes(n, a, b, dense_at_start=True, beta=3.5):
    """Generate n+1 node positions in [a, b] with tanh density at one end.

    When dense_at_start=True, elements are concentrated near x=a.
    When dense_at_start=False, elements are concentrated near x=b.
    beta controls clustering strength (higher = more concentrated).
    Always returns monotonically increasing coordinates.
    """
    xi = np.linspace(-1.0, 1.0, n + 1)
    t = np.tanh(beta * xi) / np.tanh(beta)          # [-1, 1], dense at ends
    if dense_at_start:
        # t=-1 (dense) → x=a, t=1 (sparse) → x=b
        x = a + (t + 1.0) / 2.0 * (b - a)
    else:
        # t=-1 (dense) → x=b, then reversed for increasing order
        x = a + (1.0 - t) / 2.0 * (b - a)
        x = x[::-1]
    return x


def _build_hinge_xs(hinge_left=-6.0, hinge_right=6.0,
                    x_min=-40.0, x_max=40.0,
                    n_left=16, n_right=16,
                    n_left_hinge=20, n_center_hinge=40, n_right_hinge=20,
                    beta=3.5):
    """Build x-coordinates with tanh clustering around hinge pivots (x=±3mm).

    Region breakdown:
      [x_min, hinge_left]         — coarse outer left
      [hinge_left, -3.0]           — dense at -3 (left pivot)
      [-3.0, 3.0]                  — dense at both ±3 (center hinge)
      [3.0, hinge_right]           — dense at 3 (right pivot)
      [hinge_right, x_max]        — coarse outer right
    """
    left_outer  = np.linspace(x_min, hinge_left,  n_left + 1)[:-1]
    left_hinge  = _tanh_nodes(n_left_hinge,  hinge_left,  -3.0, dense_at_start=False, beta=beta)[:-1]
    center      = _tanh_nodes(n_center_hinge, -3.0,  3.0, dense_at_start=True,  beta=beta)[:-1]
    right_hinge = _tanh_nodes(n_right_hinge, 3.0,  hinge_right, dense_at_start=True,  beta=beta)[:-1]
    right_outer = np.linspace(hinge_right, x_max, n_right + 1)
    return np.concatenate([left_outer, left_hinge, center, right_hinge, right_outer])


def _build_layers(pet_ny=3, psa_ny=2):
    """Build y-coordinates and PID list for 7-layer stack.

    PET layers (0, 2, 4, 6): pet_ny elements each,  total thickness 0.05mm
    PSA layers (1, 3, 5):     psa_ny elements each,  total thickness 0.05mm
    """
    ys_list  = [0.0]
    row_pids = []
    current_y = 0.0
    for layer in range(7):
        n = pet_ny if layer % 2 == 0 else psa_ny
        dy = 0.05 / n
        for _ in range(n):
            current_y += dy
            ys_list.append(current_y)
            row_pids.append(layer)
    return np.array(ys_list), row_pids


class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
    def __call__(self, t):
        tau = np.clip((t - self.t0) / (self.t1 - self.t0), 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def run():
    print("=" * 70)
    print("ex03_v9  -  UL, tanh-clustered hinge mesh, PSA 2elt, TARGET=90 deg")
    print("=" * 70)

    # ── 1. Mesh ───────────────────────────────────────────────────────────────
    xs = _build_hinge_xs(
        hinge_left=-6.0, hinge_right=6.0,
        x_min=-40.0, x_max=40.0,
        n_left=16, n_right=16,
        n_left_hinge=20, n_center_hinge=40, n_right_hinge=20,
        beta=3.5,
    )
    nx = len(xs)

    ys, row_pids = _build_layers(pet_ny=3, psa_ny=2)
    ny = len(ys)

    print(f"  Mesh: {nx}×{ny} = {nx * ny} nodes, {(nx - 1) * (ny - 1)} elements")
    print(f"  Hinge region xs min spacing: {np.min(np.diff(xs[(xs >= -6.0) & (xs <= 6.0)])):.4f} mm")

    mesh = Mesh()
    nid_map = {}
    for j in range(ny):
        for i in range(nx):
            nid = j * nx + i
            nid_map[(j, i)] = nid
            mesh.add_node(nid, xs[i], ys[j])

    elem_idx = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            n1 = j * nx + i
            mesh.add_element(elem_idx, [n1, n1+1, n1+nx+1, n1+nx], "QUAD4",
                             pid=row_pids[j])
            elem_idx += 1

    nid_to_idx = mesh.node_id_to_index()

    # ── 2. Materials ──────────────────────────────────────────────────────────
    pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    psa_base = NeoHookean()
    psa_mat = ViscoelasticMaterial(psa_base, g_i=[0.8], tau_i=[1.0])
    materials       = {l: (pet_mat if l % 2 == 0 else psa_mat) for l in range(7)}
    material_params = {l: ({} if l % 2 == 0 else {'E': 100.0, 'nu': 0.49}) for l in range(7)}

    # ── 3. Solver — Updated Lagrangian ───────────────────────────────────────
    elem_type = {l: ("Q4_EAS" if l % 2 == 0 else "Q4_VISCO_SIMO") for l in range(7)}
    solver = DynamicSolver(
        mesh, materials, rho=1000.0, material_params=material_params,
        constraints=[], penalty_constraints=[],
        max_iter=MAX_ITER, tol=TOL, atol=1e-7, rtol=5e-3,
        verbose=True,
        element_type=elem_type,
        fast_assembly=True,
        mode="quasistatic",
        ul_mode=True,
    )

    # ── 4. Boundary conditions ────────────────────────────────────────────────
    bc_dofs = []
    bc_base = []

    slave_ux_dofs = []
    slave_uy_dofs = []
    slave_dx_arr  = []
    slave_dy_arr  = []
    slave_sign    = []

    def _add_slave(j, i, pivot, sign_val):
        nid = nid_map[(j, i)]
        idx = nid_to_idx[nid]
        dx = xs[i] - pivot[0]
        dy = ys[j] - pivot[1]
        slave_ux_dofs.append(idx * 2)
        slave_uy_dofs.append(idx * 2 + 1)
        slave_dx_arr.append(dx)
        slave_dy_arr.append(dy)
        slave_sign.append(sign_val)
        bc_dofs.extend([idx * 2, idx * 2 + 1])
        bc_base.extend([0.0, 0.0])

    # j=0 only (bottom row slave)
    for i, x in enumerate(xs):
        if x < -SLAVE_THRESH:
            _add_slave(0, i, LEFT_PIVOT,  -1.0)
        elif x > SLAVE_THRESH:
            _add_slave(0, i, RIGHT_PIVOT, +1.0)

    n_slave = len(slave_ux_dofs)
    slave_ux_dofs = np.array(slave_ux_dofs, dtype=np.int64)
    slave_uy_dofs = np.array(slave_uy_dofs, dtype=np.int64)
    slave_dx_arr  = np.array(slave_dx_arr)
    slave_dy_arr  = np.array(slave_dy_arr)
    slave_sign    = np.array(slave_sign)

    # Symmetry: UX=0 at xs≈0
    for j in range(ny):
        for i, x in enumerate(xs):
            if abs(x) < 0.01:
                bc_dofs.append(nid_to_idx[nid_map[(j, i)]] * 2)
                bc_base.append(0.0)

    # Pivot pins: j=0, xs closest to ±3 → rigid body fix
    for pivot_x in [-3.0, 3.0]:
        closest_i = int(np.argmin(np.abs(xs - pivot_x)))
        nid = nid_map[(0, closest_i)]
        idx = nid_to_idx[nid]
        bc_dofs.extend([idx * 2, idx * 2 + 1])
        bc_base.extend([0.0, 0.0])

    solver.set_prescribed_dofs(bc_dofs, bc_base)
    solver.max_disp_limit = 300.0

    print(f"  Slave BCs : {n_slave} nodes × 2 = {2 * n_slave} dofs  (j=0 only)")
    print(f"  SLAVE_THRESH = {SLAVE_THRESH}")
    print(f"  n_dofs    : {solver.n_dofs}")
    print(f"  UL mode   : {solver.ul_mode}")
    print(f"  ny={ny}  nx={nx}  n_elem={solver.n_elem}")

    # ── 5. Output ─────────────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    fp = os.path.join("output", "ex03_v9_mesh_refined.vtkhdf")
    exporter = TransientVTKHDFExporter(fp, mesh)
    exporter.add_step(0.0, solver.u)

    ampl = SmoothAmplitude(0.0, T_TOTAL)

    wing_tip_idx = nid_to_idx[nid_map[(0, 0)]]
    centre_j0 = [nid_to_idx[nid_map[(0, i)]] for i, x in enumerate(xs) if abs(x) < 2.0]

    # ── 6. Time loop ──────────────────────────────────────────────────────────
    dt = DT_INITIAL
    step_count = 0
    cutbacks = 0

    while solver.time < T_TOTAL - 1e-12:
        dt = min(dt, T_TOTAL - solver.time)
        factor = ampl(solver.time + dt)
        theta  = -TARGET_ANGLE * factor

        # Update slave BCs (rigid rotation about each pivot)
        cosL, sinL = np.cos(theta),  np.sin(theta)
        cosR, sinR = np.cos(-theta), np.sin(-theta)
        is_left = slave_sign < 0
        ct = np.where(is_left, cosL, cosR)
        st = np.where(is_left, sinL, sinR)
        ux_vals = (ct - 1.0) * slave_dx_arr - st * slave_dy_arr
        uy_vals =  st        * slave_dx_arr + (ct - 1.0) * slave_dy_arr

        for k in range(n_slave):
            solver._bc_base_vals[2 * k]     = ux_vals[k]
            solver._bc_base_vals[2 * k + 1] = uy_vals[k]

        step_count += 1
        print(f"\n{'='*60}")
        print(f" STEP {step_count}  t={solver.time:.4f}->{solver.time+dt:.4f}"
              f"  theta={np.degrees(theta):.1f}deg")
        print(f"{'='*60}")

        saved = solver.save_state()
        n_iter = solver.solve_step(dt)

        if n_iter < 0:
            cutbacks += 1
            if cutbacks > MAX_CUTBACKS:
                print(f"  FATAL: {MAX_CUTBACKS} cutbacks at t={solver.time:.5f}")
                break
            dt = max(dt * 0.5, DT_MIN)
            print(f"  CUTBACK {cutbacks}: dt->{dt:.3e}")
            solver.restore_state(saved)
            step_count -= 1
            continue

        cutbacks = 0
        exporter.add_step(solver.time, solver.u)

        uy_tip = solver.u[wing_tip_idx * 2 + 1]
        uy_mid = (np.mean([solver.u[idx * 2 + 1] for idx in centre_j0])
                  if centre_j0 else 0.0)
        umax = float(np.max(np.abs(solver.u)))
        print(f"  [FOLD] theta={np.degrees(theta):.1f}deg  tip_UY={uy_tip:.2f}mm"
              f"  centre_UY={uy_mid:.4f}mm  max|u|={umax:.2f}mm", flush=True)

        # Adaptive dt
        if n_iter <= 3:
            dt = min(dt * 1.2, DT_MAX)
        elif n_iter > 15:
            dt = max(dt * 0.8, DT_MIN)

    print(f"\n{'='*60}")
    print(f"  DONE. Steps={step_count}  Cutbacks={cutbacks}")
    uy_mid_final = (np.mean([solver.u[idx * 2 + 1] for idx in centre_j0])
                    if centre_j0 else 0.0)
    uy_tip_final = solver.u[wing_tip_idx * 2 + 1]
    print(f"  centre_UY_final = {uy_mid_final:.4f} mm")
    print(f"  tip_UY_final    = {uy_tip_final:.4f} mm")
    if abs(uy_mid_final) > 1.0:
        print("  => U-BEND FORMED!")
    else:
        print("  => Centre still flat.")
    print(f"{'='*60}")

    exporter.close()
    print(f"Output: {fp}")


if __name__ == "__main__":
    run()
