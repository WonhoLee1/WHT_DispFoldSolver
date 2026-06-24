"""
ex03_display_fold.py
====================
Example showing a detailed display folding test:
- 7-layer Display Panel: Alternating PET (J2Plasticity) and PSA (ViscoelasticMaterial).
- Outer edge boundary nodes coupled to offset rotational hinges via RBE2 constraints.
- Hinge rotation driven up to 90 degrees in opposite directions (180-degree folding).
- JAX-based Penalty self-contact constraint dynamically prevents panel self-penetration.
- Adaptive time stepping and Backtracking Line Search for robust convergence.
"""

import os

N_CORES = 4  # ← 사용할 CPU 코어 수

# --- 수렴 제어 ---
MAX_ITER    = 50      # Newton-Raphson 최대 반복 수
TOL         = 1e-3   # 수렴 판정 (disp ratio) — 상용 S/W(Abaqus) 수준. 값↑ = 완화

# --- 시간 적분 ---
T_TOTAL     = 1.0    # 전체 해석 시간 [s]
DT_INITIAL  = 0.02   # 초기 시간 증분
DT_MIN      = 1e-5   # 최소 시간 증분 (컷백 하한)
DT_MAX      = 0.05   # 최대 시간 증분
MAX_CUTBACKS = 10    # 최대 컷백 횟수

# --- 하중 ---
TARGET_ANGLE = 1.570796  # 힌지 회전 목표각 [rad] (90°)

# --- 조립 경로 ---
# True: 배치 가속 조립 (순차와 기계정밀도로 일치하도록 검증됨 → 동일 수렴, 더 빠름)
# False: 순차 조립 (참조용 / 디버깅용)
FAST_ASSEMBLY = True

os.environ["MKL_NUM_THREADS"]     = str(N_CORES)  # MKL 전체 (PARDISO 포함)
os.environ["PARDISO_NUM_THREADS"] = str(N_CORES)  # PARDISO 명시적 오버라이드
os.environ["XLA_FLAGS"]           = f"--xla_cpu_multi_thread_eigen=false"

import numpy as np
from dispsolver.mesh import Mesh
from dispsolver.material import J2Plasticity, LinearViscoelastic
from dispsolver.constraint import RBE2HingeConstraint, PenaltyContactConstraint
from dispsolver.solver import DynamicSolver
from dispsolver.load import Amplitude
from dispsolver.export import export_vtkhdf
from dispsolver.export.vtkhdf_exporter import TransientVTKHDFExporter

def run_display_fold():
    print("Setting up multilayer display fold example with rotational hinges and contact...")
    
    # 1. Mesh setup
    # Display Panel: x in [-40, 40] mm, y in [0, 0.35] mm (7 layers of 0.05 mm thickness)
    mesh = Mesh()
    
    # X direction: fine mesh at hinge zone [-15, 15] and coarse mesh at ends
    xs_left = np.linspace(-40.0, -15.0, 21)[:-1]      # dx = 1.25 mm
    xs_mid = np.linspace(-15.0, 15.0, 61)[:-1]        # dx = 0.5 mm (dense hinge zone)
    xs_right = np.linspace(15.0, 40.0, 21)             # dx = 1.25 mm
    xs = np.concatenate([xs_left, xs_mid, xs_right])
    nx = len(xs)
    
    # Y direction: 7 layers
    # PET layers (0, 2, 4, 6) -> 3 elements through thickness
    # PSA layers (1, 3, 5)    -> 1 element through thickness
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
    
    # Add nodes
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            nid = j * nx + i
            mesh.add_node(nid, x, y)
            
    # Add Elements: pid corresponds to layer id (0 to 6)
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
            
    # Add Hinge Master Nodes (바닥면과 동일 평면 y=0)
    # Hinge L: at x = -15, y = 0.0 mm
    # Hinge R: at x = 15, y = 0.0 mm
    mesh.add_node(99999, -15.0, 0.0)
    mesh.add_node(99998, 15.0, 0.0)
    
    # 2. Material setup
    # PET: Elasto-plastic (J2Plasticity)
    # E=4000 MPa, 항복 80 MPa, 100% 소성변형 시 700 MPa까지 선형 하드닝
    # σy = σy0 + H·ε̄p  →  H = (700 - 80) / 1.0 = 620 MPa
    pet_mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)
    # PSA: 선형 점탄성 (소변형 Prony + WLF), Q1P0 하이브리드로 체적 락킹 완화
    # instantaneous E=10.0, nu=0.49 (near-incompressible 접착층)
    psa_mat = LinearViscoelastic(E=10.0, nu=0.49, g_i=[0.8], tau_i=[1.0])

    materials = {
        layer: pet_mat if layer % 2 == 0 else psa_mat
        for layer in range(7)
    }

    material_params = {
        layer: {} for layer in range(7)
    }
    
    # 3. Constraint setup
    # 바닥면(y=0, j=0 행)을 두 개의 강체 플랩으로 힌지에 연결한다.
    #   - 좌측 플랩 바닥면 (x <= -15) → 힌지 L (master 99999)
    #   - 우측 플랩 바닥면 (x >= +15) → 힌지 R (master 99998)
    #   - 가운데 힌지 영역(-15 < x < 15)의 바닥 노드는 자유 → 굽힘 허용
    # 바닥 행 노드의 전역 인덱스는 nid = 0*nx + i = i.
    tol = 1e-9
    slave_l = [i for i in range(nx) if xs[i] <= -15.0 + tol]
    slave_r = [i for i in range(nx) if xs[i] >=  15.0 - tol]
    rbe2_l = RBE2HingeConstraint(mesh, master_id=99999, slave_ids=slave_l, extra_primal_offset=0)
    rbe2_r = RBE2HingeConstraint(mesh, master_id=99998, slave_ids=slave_r, extra_primal_offset=1)
    
    # JAX Penalty self-contact constraint for panel top (y=0.35) and bottom (y=0.0) nodes
    # Contact thickness (threshold) d_0 = 0.2 mm
    contact_nodes = [i for i in range(nx)] + [(ny - 1) * nx + i for i in range(nx)]
    contact_constraint = PenaltyContactConstraint(mesh, contact_nodes=contact_nodes, k_contact=1e6, d_0=0.2)
    
    # 4. Solver setup — 재료(pid)별 요소 타입 혼용:
    #   PET (Even layers, J2 소성) → Q4_EAS (Enhanced Assumed Strain, 굽힘 락킹 제거)
    #   PSA (Odd layers, 선형 점탄성) → Q4_UP Q1P0 하이브리드 (체적 락킹 제거)
    element_type = {
        layer: "Q4_EAS" if layer % 2 == 0 else "Q4_UP"
        for layer in range(7)
}
    solver = DynamicSolver(
        mesh,
        materials,
        rho=1000.0,
        material_params=material_params,
        constraints=[rbe2_l, rbe2_r],
        penalty_constraints=[contact_constraint],
        max_iter=MAX_ITER,
        tol=TOL,
        verbose=True,
        element_type=element_type,
        fast_assembly=FAST_ASSEMBLY,
    )
    
    # 5. Boundary conditions and rotation drive
    # Global node indices for Hinge L and R masters
    nid_to_idx = mesh.node_id_to_index()
    idx_L = nid_to_idx[99999]
    idx_R = nid_to_idx[99998]
    
    # Extra DOFs indices for rotation angles: theta_L at solver.n_dofs, theta_R at solver.n_dofs + 1
    idx_theta_L = solver.n_dofs + 0
    idx_theta_R = solver.n_dofs + 1

    # Target rotation angle: 90 degrees in opposite directions (1.570796 rad)
    # Hinge L rotates clockwise (-theta), Hinge R rotates counter-clockwise (+theta)
    target_angle = TARGET_ANGLE

    # Adaptive incremental time stepping parameters
    t_total      = T_TOTAL
    dt_initial   = DT_INITIAL
    dt_min       = DT_MIN
    dt_max       = DT_MAX
    max_cutbacks = MAX_CUTBACKS

    # --- Time-dependent boundary conditions via Amplitude (Abaqus *AMPLITUDE) ---
    # The hinge rotation is driven by a single amplitude curve ramping 0 -> 1 over
    # the analysis time. Translations stay fixed (amplitude None). The solver
    # evaluates BC = base * amplitude(t) each increment automatically — no manual
    # per-step rescaling needed.
    rotation_ampl = Amplitude([0.0, t_total], [0.0, 1.0], method="linear")
    bc_dofs = [idx_L * 2, idx_L * 2 + 1, idx_R * 2, idx_R * 2 + 1, idx_theta_L, idx_theta_R]
    bc_base = [0.0, 0.0, 0.0, 0.0, -target_angle, target_angle]
    bc_amps = [None, None, None, None, rotation_ampl, rotation_ampl]
    solver.set_prescribed_dofs(bc_dofs, bc_base, amplitudes=bc_amps)

    dt = dt_initial
    step_count = 0
    cutbacks = 0

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Transient Exporter
    filepath = os.path.join(output_dir, "ex03_fold_transient_v4.vtkhdf")
    exporter = TransientVTKHDFExporter(filepath, mesh)
    
    # Export initial state (t=0)
    exporter.add_step(0.0, solver.u)

    print(f"\n{'='*100}")
    print(f" ADAPTIVE INCREMENTAL DISPLAY FOLDING SIMULATION")
    print(f" Total Time: {t_total:.2f} s   Initial dt: {dt_initial:.3e} s   dt_min: {dt_min:.1e} s   dt_max: {dt_max:.2e} s")
    print(f" Hinge Rotation Angle: {target_angle * 180 / np.pi:.1f} degrees (Total 180-degree Fold)")
    print(f" Driven by Amplitude curve: {rotation_ampl}")
    print(f"{'='*100}")

    while solver.time < t_total - 1e-12:
        dt = min(dt, t_total - solver.time)
        t_start_step = solver.time
        t_end = t_start_step + dt
        factor = rotation_ampl(t_end)            # current amplitude (load factor)
        theta_L = -target_angle * factor
        theta_R = target_angle * factor

        step_count += 1
        print(f"\n{'='*100}", flush=True)
        print(f" STEP {step_count}  Time: {t_start_step:.5f} -> {t_end:.5f}  (dt={dt:.3e}, Load Factor: {factor:.4f})", flush=True)
        print(f" Hinge Angles: L = {theta_L * 180 / np.pi:.1f} deg, R = {theta_R * 180 / np.pi:.1f} deg", flush=True)
        print(f"{'='*100}", flush=True)

        # Save state for rollback
        saved_state = solver.save_state()
        
        n_iter = solver.solve_step(dt)
        
        if n_iter < 0:
            # Step failed - cutback dt and retry
            cutbacks += 1
            if cutbacks > max_cutbacks:
                print(f"\n  *** FATAL: Maximum cutbacks ({max_cutbacks}) exceeded at t={solver.time:.5f}. Aborting.", flush=True)
                break
            
            dt_new = dt * 0.5
            if dt_new < dt_min:
                print(f"\n  *** FATAL: dt ({dt_new:.2e}) below minimum ({dt_min:.1e}). Aborting.", flush=True)
                break
            
            print(f"  *** CUTBACK {cutbacks}: dt {dt:.3e} -> {dt_new:.3e}", flush=True)
            
            # Rollback solver state and retry with smaller dt
            solver.restore_state(saved_state)
            dt = dt_new
            step_count -= 1
            continue
            
        # Step succeeded - solver.time advanced internally on convergence
        cutbacks = 0
        
        # Export frame
        exporter.add_step(solver.time, solver.u)
        
        # Adaptive time step adjustments based on NR iterations
        if n_iter <= 5:
            dt = min(dt * 1.5, dt_max)
        elif n_iter <= 10:
            dt = min(dt * 1.2, dt_max)
        elif n_iter >= 25:
            dt = max(dt * 0.5, dt_min)
            
    else:
        print(f"\n{'='*100}", flush=True)
        print(f" => ALL INCREMENTS COMPLETED SUCCESSFULLY. Total steps: {step_count}", flush=True)
        print(f"{'='*100}", flush=True)
        
    exporter.close()
    print(f"Successfully exported all transient frames to {filepath}")

if __name__ == "__main__":
    run_display_fold()
