"""
ex15_multilayer_thin_bar_fold.py
================================
3D Multilayer Thin Display Bar Folding Simulation.
Demonstrates full commercial CAE model encapsulation:
  - Model -> Part -> Section -> Assembly -> Step architecture
  - Numba @njit C-Kernel UserFunctionBC for kinematic pivot rotation
  - SmoothStepAmplitude C² ramp (90 deg per wing, 180 deg total fold)
  - Real-time Sensor & IterationHook monitoring
  - Fully encapsulated DynamicSolver3D.solve(step) execution engine
"""

import numpy as np
import time
import os
import pickle
from numba import njit

from dispsolver.model.model import Model
from dispsolver.model.sensor import Sensor, IterationHook, ControlAction


# Pivot locations for canonical 2-pivot folding
# Pivot locations for canonical 2-pivot folding
LEFT_PIVOT = np.array([-3.0, 0.2, 0.0], dtype=np.float64)
RIGHT_PIVOT = np.array([ 3.0, 0.2, 0.0], dtype=np.float64)
TARGET_THETA_MAX = np.deg2rad(90.0)


@njit(fastmath=True)
def fold_left_numba(coords: np.ndarray, t: float, step_time: float):
    """Numba-compiled C-kernel evaluating left wing kinematic pivot rotation at time t."""
    if coords[0] > -7.99:
        return (None, None, 0.0)
    
    # Ramp load smoothly using Smoothstep: s(t) = 3t^2 - 2t^3
    t_clamped = max(0.0, min(1.0, t))
    ramp = 3.0 * (t_clamped ** 2) - 2.0 * (t_clamped ** 3)
    theta = TARGET_THETA_MAX * ramp

    cos_L = np.cos(theta)
    sin_L = np.sin(theta)

    dx = coords[0] - LEFT_PIVOT[0]
    dy = coords[1] - LEFT_PIVOT[1]

    ux = dx * cos_L - dy * sin_L - dx
    uy = dx * sin_L + dy * cos_L - dy
    return (ux, uy, 0.0)


@njit(fastmath=True)
def fold_right_numba(coords: np.ndarray, t: float, step_time: float):
    """Numba-compiled C-kernel evaluating right wing kinematic pivot rotation at time t."""
    if coords[0] < 7.99:
        return (None, None, 0.0)
    
    t_clamped = max(0.0, min(1.0, t))
    ramp = 3.0 * (t_clamped ** 2) - 2.0 * (t_clamped ** 3)
    theta = -TARGET_THETA_MAX * ramp

    cos_R = np.cos(theta)
    sin_R = np.sin(theta)

    dx = coords[0] - RIGHT_PIVOT[0]
    dy = coords[1] - RIGHT_PIVOT[1]

    ux = dx * cos_R - dy * sin_R - dx
    uy = dx * sin_R + dy * cos_R - dy
    return (ux, uy, 0.0)


def build_ex15_model():
    """Build complete CAE Model hierarchy for 4-layer thin bar 3D folding."""
    model = Model(name="Ex15_Multilayer_Fold", dim=3)
    part = model.Part(name="ThinBar")

    nx, ny, nz = 40, 4, 3
    L, H, W = 40.0, 0.4, 1.0

    xs = np.linspace(-L/2, L/2, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)
    zs = np.linspace(0.0, W, nz + 1)

    node_map = {}
    nid = 1
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                part.Node(nid, xs[i], ys[j], zs[k])
                node_map[(i, j, k)] = nid
                nid += 1

    eid = 1
    pet_elems = []
    psa_elems = []

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

                pid = 0 if j % 2 == 0 else 1
                elem_type = "C3D8_CR" if pid == 0 else "C3D8H"
                
                part.Element(eid, elem_type, [n0, n1, n2, n3, n4, n5, n6, n7])
                if pid == 0:
                    pet_elems.append(eid)
                else:
                    psa_elems.append(eid)
                eid += 1

    part.ElementSet("ELSET_PET", pet_elems)
    part.ElementSet("ELSET_PSA", psa_elems)
    part.create_set("ALL", elements=list(part.elements.keys()), nodes=list(part.nodes.keys()))

    right_wing_nids = [node_map[(i, j, k)] for i in range(nx + 1) for j in range(ny + 1) for k in range(nz + 1) if xs[i] >= 7.99]
    part.create_set("RIGHT_WING", nodes=right_wing_nids)

    # Define Materials
    mat_pet = model.Material(name="PET", mat_type="J2_PLASTICITY")
    mat_pet.E = 4000.0
    mat_pet.nu = 0.3
    mat_pet.sigma_y0 = 80.0
    mat_pet.H = 400.0

    mat_psa = model.Material(name="PSA", mat_type="NEO_HOOKEAN")
    mat_psa.C10 = 0.1
    mat_psa.D1 = 0.01

    sec_pet = model.SolidSection(name="SecPET", material="PET")
    sec_psa = model.SolidSection(name="SecPSA", material="PSA")

    part.SectionAssignment(region="ELSET_PET", sectionName="SecPET")
    part.SectionAssignment(region="ELSET_PSA", sectionName="SecPSA")

    # Instance Part into Assembly
    model.root_assembly.Instance(name="BAR_INST", part=part)

    # Define Step and BCs
    step = model.Step(name="FoldStep", time_period=1.0, dt_init=0.02, dt_min=1e-5, dt_max=0.05)

    model.UserFunctionBC(
        name="FoldLeftBC",
        createStepName="FoldStep",
        region="BAR_INST.ALL",
        numba_func=fold_left_numba
    )

    model.UserFunctionBC(
        name="FoldRightBC",
        createStepName="FoldStep",
        region="BAR_INST.ALL",
        numba_func=fold_right_numba
    )

    # Attach Sensor for Right Wing Draw-In
    draw_in_sensor = model.Sensor(
        name="DrawIn_UX",
        entity_type="set",
        entity_id="BAR_INST.RIGHT_WING",
        variable="U",
        comp="1"
    )
    step.sensors["DrawIn"] = draw_in_sensor

    return model, step


def main():
    print("Building 4-layer thin bar 3D model using CAE Abaqus API...", flush=True)
    model, step = build_ex15_model()

    print("Creating solver and building sparse system...", flush=True)
    solver, sys = model.create_solver3d(step_name="FoldStep")

    print("Starting encapsulated solver execution (180 deg fold)...", flush=True)
    t_start = time.perf_counter()
    success = solver.solve(step=step, sys=sys, verbose=True)
    t_wall = time.perf_counter() - t_start

    print(f"\nSimulation finished in {t_wall:.2f} s | Status: {'SUCCESS' if success else 'FAILED'}", flush=True)

    # Extract history
    history_u = solver.history_u if hasattr(solver, "history_u") and solver.history_u else [solver.u.copy()]
    history_t = solver.history_t if hasattr(solver, "history_t") and solver.history_t else [1.0]

    theta_history = []
    draw_in_history = []
    right_indices = sys.global_nsets.get("BAR_INST.RIGHT_WING", [])

    for idx, t_val in enumerate(history_t):
        ramp = 3.0 * (t_val ** 2) - 2.0 * (t_val ** 3)
        theta_history.append(np.rad2deg(TARGET_THETA_MAX * ramp))
        u_step = history_u[idx]
        u_nodes = u_step.reshape(-1, 3)
        ux_val = float(np.mean(u_nodes[right_indices, 0])) if len(right_indices) > 0 else 0.0
        draw_in_history.append(ux_val)

    if draw_in_history:
        print(f"Final Right End Draw-In Displacement: {draw_in_history[-1]:.4f} mm", flush=True)

    # Save results
    res_path = "examples/ex15_result.pkl"
    print(f"Saving results to {res_path}...", flush=True)
    mesh3d = sys.to_mesh3d()
    result_data = {
        'mesh': mesh3d,
        'displacement': solver.u.copy(),
        'history': history_u if history_u else [solver.u.copy()],
        'draw_in_history': draw_in_history,
        'theta_history': theta_history
    }
    with open(res_path, 'wb') as f:
        pickle.dump(result_data, f)
    print("Result saved successfully.", flush=True)

    # Trigger visualization
    try:
        from examples.ex15_visualize_multilayer import main as run_viz
        print("Generating visualizations...", flush=True)
        run_viz()
    except Exception as e:
        print(f"Visualization error: {e}", flush=True)


if __name__ == "__main__":
    main()
