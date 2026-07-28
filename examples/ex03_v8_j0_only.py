"""
ex03_v8_j0_only.py
==================
U-bend folding — Updated Lagrangian, **j=0 (bottom row) slave only**.

핵심 설계 원칙
--------------
1. Slave BC는 바닥 행(j=0)만 적용.
   내부 노드(j>0)는 재료 강성(얇고 뻣뻣한 PET wing)으로 자연스럽게 따라감.
   → 외팔보 고정단과 동일 원리: 고정단 경계에서 변위=0, 내부는 탄성 자유 변형.

2. SLAVE_THRESH = 3.5  (pivot xs=±3 근처)
   pivot 근처는 변위 ≈ 0 → slave/free 경계에서 변위 구배 ≈ 0
   → 외팔보 고정단과 동일 → EAS NaN 없음.

3. UL mode (Updated Lagrangian) — 대변형에서 det(F_inc)>0 항상 보장.

ex03_v7_ul.py의 오류:
   전체 j행을 slave로 잡아 wing이 완전한 rigid body가 됨.
   → xs=-10.5(slave) vs xs=-10.0(free) 사이에 거대한 변위 불연속
   → 모든 두께 방향에 걸쳐 심각한 NaN.
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
SLAVE_THRESH = 3.5   # pivot 근처 — 변위≈0 경계 → NaN 억제


class SmoothAmplitude:
    def __init__(self, t0=0.0, t1=1.0):
        self.t0, self.t1 = t0, t1
    def __call__(self, t):
        tau = np.clip((t - self.t0) / (self.t1 - self.t0), 0.0, 1.0)
        return tau**3 * (10.0 + tau * (-15.0 + 6.0 * tau))


def run():
    print("=" * 70)
    print("ex03_v8  -  UL, j=0 only slave, SLAVE_THRESH=3.5, TARGET=90 deg")
    print("=" * 70)

    # ── 1. Mesh ───────────────────────────────────────────────────────────────
    mesh = Mesh()
    xs_left  = np.linspace(-40.0, -15.0, 21)[:-1]
    xs_mid   = np.linspace(-15.0,  15.0, 61)[:-1]
    xs_right = np.linspace( 15.0,  40.0, 21)
    xs = np.concatenate([xs_left, xs_mid, xs_right])
    nx = len(xs)

    ys_list  = [0.0]
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
    #   - Slave: j=0 만, |xs| > SLAVE_THRESH (pivot 바깥 바닥 행)
    #   - 내부 노드(j>0): 자유 — 재료 강성으로 j=0을 따라감
    #   - Symmetry: xs≈0 컬럼 UX=0
    bc_dofs = []
    bc_base = []

    slave_ux_dofs = []
    slave_uy_dofs = []
    slave_dx_arr  = []
    slave_dy_arr  = []
    slave_sign    = []   # -1.0=left, +1.0=right

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

    # j=0만! (바닥 행만 slave)
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

    # Pivot 핀: j=0에서 xs가 ±3에 가장 가까운 노드 UX=UY=0
    # (rigid body 이동 방지 — ex03_v6의 master node 역할)
    for pivot_x in [-3.0, 3.0]:
        closest_i = int(np.argmin(np.abs(xs - pivot_x)))
        nid = nid_map[(0, closest_i)]
        idx = nid_to_idx[nid]
        bc_dofs.extend([idx * 2, idx * 2 + 1])
        bc_base.extend([0.0, 0.0])

    solver.set_prescribed_dofs(bc_dofs, bc_base)
    solver.max_disp_limit = 300.0

    print(f"  Slave BCs : {n_slave} nodes × 2 = {2*n_slave} dofs  (j=0 only)")
    print(f"  SLAVE_THRESH = {SLAVE_THRESH}  LEFT_PIVOT={LEFT_PIVOT}  RIGHT_PIVOT={RIGHT_PIVOT}")
    print(f"  n_dofs    : {solver.n_dofs}")
    print(f"  UL mode   : {solver.ul_mode}")
    print(f"  ny={ny}  nx={nx}  n_elem={solver.n_elem}")

    # ── 5. Output ─────────────────────────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    fp = os.path.join("output", "ex03_v8_j0_only.vtkhdf")
    exporter = TransientVTKHDFExporter(fp, mesh)
    exporter.add_step(0.0, solver.u)

    ampl = SmoothAmplitude(0.0, T_TOTAL)

    # Monitor: wing tip (j=0, i=0), centre bottom (j=0, xs≈0)
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
