import numpy as np
import time
import os

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d import DynamicSolver3D
from dispsolver.material3d.numba_materials import MAT_NEO_HOOKEAN, MAT_J2_PLASTICITY, MAT_LINEAR_ELASTIC

class AdaptiveDtController:
    def __init__(self, dt_init=0.01, dt_min=1e-5, dt_max=0.05):
        self.dt = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
    
    def adjust_dt(self, converged: bool, iters: int):
        if not converged:
            self.dt = max(self.dt_min, self.dt * 0.25)
            return False
        elif iters <= 4:
            self.dt = min(self.dt_max, self.dt * 1.5)
        return True

def create_4layer_thin_bar():
    mesh = Mesh3D()
    
    # 4 layers (Y-direction), 3 elements in Z (width-resolved), 40 elements in X
    nx = 40
    ny = 4
    nz = 3
    
    L = 40.0
    W = 1.0
    H = 0.4
    
    xs = np.linspace(-L/2, L/2, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)
    zs = np.linspace(0.0, W, nz + 1)
    
    node_map = {}
    nid = 1
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                mesh.add_node(nid, xs[i], ys[j], zs[k])
                node_map[(i, j, k)] = nid
                nid += 1
                
    eid = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n0 = node_map[(i, j, k)]
                n1 = node_map[(i+1, j, k)]
                n2 = node_map[(i+1, j+1, k)]
                n3 = node_map[(i, j+1, k)]
                n4 = node_map[(i, j, k+1)]
                n5 = node_map[(i+1, j, k+1)]
                n6 = node_map[(i+1, j+1, k+1)]
                n7 = node_map[(i, j+1, k+1)]
                
                # PID 0: PET (J2 Plasticity) -> C3D8_CR
                # PID 1: PSA (Hyperelastic)   -> C3D8H (Hybrid Element to prevent locking & inversion)
                pid = 0 if j % 2 == 0 else 1
                elem_type = "C3D8_CR" if pid == 0 else "C3D8H"
                
                elem = mesh.add_element(eid, [n0, n1, n2, n3, n4, n5, n6, n7], elem_type)
                elem.pid = pid
                eid += 1
                
    return mesh

def main():
    print("Building 4-layer thin bar 3D mesh with nz=3 and C3D8H hybrid elements...")
    mesh = create_4layer_thin_bar()
    
    # Materials setup
    # PID 0: PET (J2 Plasticity) E=4000, nu=0.3, Sy=80, H=400
    mat_pet = {"type": "j2_plasticity", "E": 4000.0, "nu": 0.3, "sigma_y0": 80.0, "H": 400.0}
    # PID 1: PSA (Neo-Hookean) C10=0.1, D1=0.01 -> mu=0.2 MPa, K=200 MPa
    mat_psa = {"type": "neo_hookean", "C10": 0.1, "D1": 0.01}
    
    materials = {0: mat_pet, 1: mat_psa}
    
    print("Initializing DOD Solver3D with heterogeneous multi-element support...")
    solver = DynamicSolver3D(mesh, materials=materials)
    
    # Enforce plane-strain by fixing all nodes in Z
    for nid in mesh.nodes:
        solver.fix_dof(nid, 2, 0.0)

    # Canonical two-pivot foldable display kinematics:
    # Left pivot at (-3.0, 0.2), Right pivot at (+3.0, 0.2)
    # Left wing (X <= -8.0) rotates by +theta around Left Pivot
    # Right wing (X >= +8.0) rotates by -theta around Right Pivot
    # Middle span (-8.0 < X < 8.0) bends into a smooth, natural U-shape loop
    LEFT_PIVOT = np.array([-3.0, 0.2, 0.0])
    RIGHT_PIVOT = np.array([ 3.0, 0.2, 0.0])

    left_nids = [n.id for n in mesh.nodes.values() if n.x <= -7.99]
    right_nids = [n.id for n in mesh.nodes.values() if n.x >= 7.99]

    def apply_kinematic_fold(theta_rad):
        # Left wing rotation around LEFT_PIVOT by +theta
        cos_L, sin_L = np.cos(theta_rad), np.sin(theta_rad)
        for nid in left_nids:
            n = mesh.nodes[nid]
            dx = n.x - LEFT_PIVOT[0]
            dy = n.y - LEFT_PIVOT[1]
            ux = dx * cos_L - dy * sin_L - dx
            uy = dx * sin_L + dy * cos_L - dy
            solver.fix_dof(nid, 0, ux)
            solver.fix_dof(nid, 1, uy)
            solver.fix_dof(nid, 2, 0.0)

        # Right wing rotation around RIGHT_PIVOT by -theta
        cos_R, sin_R = np.cos(-theta_rad), np.sin(-theta_rad)
        for nid in right_nids:
            n = mesh.nodes[nid]
            dx = n.x - RIGHT_PIVOT[0]
            dy = n.y - RIGHT_PIVOT[1]
            ux = dx * cos_R - dy * sin_R - dx
            uy = dx * sin_R + dy * cos_R - dy
            solver.fix_dof(nid, 0, ux)
            solver.fix_dof(nid, 1, uy)
            solver.fix_dof(nid, 2, 0.0)

    dt_ctrl = AdaptiveDtController(dt_init=0.02)
    t = 0.0
    t_end = 1.0
    target_theta_max = np.deg2rad(90.0) # 90 degrees each side -> U-shape 180 total
    
    step = 0
    draw_in_history = []
    theta_history = []
    
    while t < t_end:
        step += 1
        
        # Save previous state in case of cutback
        u_prev = solver.u.copy()
        t_prev = t
        
        t += dt_ctrl.dt
        if t > t_end:
            t = t_end
            dt_ctrl.dt = t - t_prev
            
        # Ramp load smoothly
        ramp = 3 * (t**2) - 2 * (t**3)
        theta_current = target_theta_max * ramp
        
        apply_kinematic_fold(theta_current)
        
        t_solve_start = time.perf_counter()
        conv, iters = solver.solve_step(dt=dt_ctrl.dt, max_iters=25)
        t_solve = time.perf_counter() - t_solve_start
        
        # Measure draw-in displacement on the right end
        nid_map = mesh.node_id_to_index()
        u_draw_in = np.mean([solver.u[3 * nid_map[nid] + 0] for nid in right_nids])
        
        print(f"Step {step}: t={t:.3f}, dt={dt_ctrl.dt:.4f}, theta={np.rad2deg(theta_current):.1f} deg | Draw-In UX={u_draw_in:.4f} mm")
        
        if conv:
            print(f"  -> Converged in {iters} iters ({t_solve:.2f}s)")
            dt_ctrl.adjust_dt(True, iters)
            if not hasattr(solver, 'history_u'):
                solver.history_u = []
            solver.history_u.append(solver.u.copy())
            draw_in_history.append(u_draw_in)
            theta_history.append(np.rad2deg(theta_current))
        else:
            print(f"  -> DIVERGED! Cutback triggered. (Iters={iters}, Time={t_solve:.2f}s)")
            solver.u = u_prev
            t = t_prev
            dt_ctrl.adjust_dt(False, iters)
            if dt_ctrl.dt <= dt_ctrl.dt_min:
                print("Minimum dt reached. Aborting.")
                break

    print("Simulation completed.")
    print(f"Final Right End Draw-In Displacement: {draw_in_history[-1]:.4f} mm")
    
    # Save results
    res_path = "examples/ex15_result.pkl"
    print(f"Saving results to {res_path}...")
    import pickle
    result_data = {
        'mesh': mesh,
        'displacement': solver.u.copy(),
        'history': solver.history_u if hasattr(solver, 'history_u') else [solver.u.copy()],
        'draw_in_history': draw_in_history,
        'theta_history': theta_history
    }
    with open(res_path, 'wb') as f:
        pickle.dump(result_data, f)
    print("Result saved successfully.")
    
    # Trigger visualization
    try:
        from examples.ex15_visualize_multilayer import main as run_viz
        print("Generating visualizations...")
        run_viz()
    except Exception as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    main()
