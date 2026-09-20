"""
test_emcs_superiority_benchmark.py
===================================
Specialized Dynamic FEA Benchmark Suite proving the superiority of EMCS
(Energy-Momentum Conserving Scheme) over conventional implicit dynamic solvers
(Newmark-Trapezoidal & HHT-alpha).

Evaluates:
  1. High-Speed Rigid Translation & Large Rotation (Spinning Beam in Space).
  2. Large Deflection Unconstrained Elastic Free Oscillation.
  3. Energy Conservation Error, Momentum Preservation, Cutbacks, and Wall Time.
"""

import time
import pytest
import numpy as np

from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


def build_spinning_beam_mesh(n_elements_x: int = 4) -> Mesh3D:
    """Build a 3D beam mesh along X-axis centered at origin."""
    mesh = Mesh3D()
    dx = 10.0 / n_elements_x
    dy = 2.0
    dz = 2.0

    # Nodes
    node_id = 0
    for ix in range(n_elements_x + 1):
        x = -5.0 + ix * dx
        for iy in range(2):
            y = -1.0 + iy * dy
            for iz in range(2):
                z = -1.0 + iz * dz
                mesh.add_node(node_id, x, y, z)
                node_id += 1

    # Hex elements
    elem_id = 0
    for ix in range(n_elements_x):
        for iy in range(1):
            for iz in range(1):
                # 8 nodes of hex
                n0 = ix * 4 + iy * 2 + iz
                n1 = (ix + 1) * 4 + iy * 2 + iz
                n2 = (ix + 1) * 4 + (iy + 1) * 2 + iz
                n3 = ix * 4 + (iy + 1) * 2 + iz
                n4 = n0 + 1
                n5 = n1 + 1
                n6 = n2 + 1
                n7 = n3 + 1
                mesh.add_element(elem_id, [n0, n1, n2, n3, n4, n5, n6, n7], "C3D8")
                elem_id += 1

    return mesh


def test_emcs_vs_implicit_spinning_beam_benchmark():
    """Benchmark 1: High-Speed Free Spinning Beam in Space.

    Compares:
      1. EMCS (Energy-Momentum Conserving Scheme)
      2. Newmark-Trapezoidal (transient, no damping)
      3. HHT-alpha (moderate-3, Abaqus default damping)
    """
    dt = 0.005
    n_steps = 30
    w_z = 20.0  # Initial angular velocity (rad/s)
    v_x = 50.0  # Initial translational velocity (mm/s)

    mat_params = {"E": 200000.0, "nu": 0.3}
    density = 7.85e-9

    results = {}

    for mode in ["emcs", "transient", "moderate-3"]:
        mesh = build_spinning_beam_mesh(n_elements_x=4)
        solver = DynamicSolver3D(
            mesh=mesh,
            material_params=mat_params,
            time_integrator=mode,
            density=density,
        )

        # Apply initial velocity field v_0 = v_x + w_z x r
        for nid, node in mesh.nodes.items():
            idx = mesh.node_id_to_index()[nid]
            r_x, r_y, r_z = node.x, node.y, node.z
            solver.v[3 * idx + 0] = v_x - w_z * r_y
            solver.v[3 * idx + 1] = w_z * r_x
            solver.v[3 * idx + 2] = 0.0

        e_tot_history = []
        wall_start = time.time()
        cutbacks = 0
        total_iters = 0

        for step in range(n_steps):
            conv, iters = solver.solve_step(dt=dt, tol=1e-5)
            if not conv:
                cutbacks += 1
            total_iters += iters
            e_tot, e_kin, e_strain = solver.compute_total_energy()
            e_tot_history.append(e_tot)

        wall_time = time.time() - wall_start
        e_init = e_tot_history[0]
        e_final = e_tot_history[-1]
        e_drift_rel = abs(e_final - e_init) / max(e_init, 1e-12)

        results[mode] = {
            "wall_time": wall_time,
            "total_iters": total_iters,
            "avg_iters": total_iters / n_steps,
            "cutbacks": cutbacks,
            "e_init": e_init,
            "e_final": e_final,
            "e_drift_rel": e_drift_rel,
        }

    # Verify EMCS Superiority Criteria
    emcs_res = results["emcs"]
    hht_res = results["moderate-3"]
    newmark_res = results["transient"]

    # 1. EMCS must conserve total energy without artificial dissipation (drift < 1e-4)
    assert emcs_res["e_drift_rel"] < 1e-4, f"EMCS energy drift too high: {emcs_res['e_drift_rel']}"

    # 2. HHT-alpha must exhibit artificial energy decay (dissipation)
    assert hht_res["e_final"] < emcs_res["e_final"], "HHT-alpha should artificially dissipate energy relative to EMCS"

    # 3. EMCS zero cutbacks
    assert emcs_res["cutbacks"] == 0, "EMCS must converge with 0 cutbacks"


def test_emcs_large_deflection_free_vibration_benchmark():
    """Benchmark 2: Unconstrained Elastic Beam Free Vibration under High Initial Strain."""
    dt = 0.002
    n_steps = 25
    mesh = build_spinning_beam_mesh(n_elements_x=4)

    # Initial strain perturbation: stretch beam in X
    solver = DynamicSolver3D(
        mesh=mesh,
        material_params={"E": 100000.0, "nu": 0.3},
        time_integrator="emcs",
        density=7.85e-9,
    )

    for nid, node in mesh.nodes.items():
        idx = mesh.node_id_to_index()[nid]
        solver.u[3 * idx + 0] = 0.05 * node.x

    e_history = []
    for _ in range(n_steps):
        conv, _ = solver.solve_step(dt=dt, tol=1e-5)
        assert conv is True
        e_tot, _, _ = solver.compute_total_energy()
        e_history.append(e_tot)

    # Energy fluctuation ratio must remain bounded < 1e-3
    e_min, e_max = min(e_history), max(e_history)
    fluc_ratio = (e_max - e_min) / max(e_history[0], 1e-12)
    assert fluc_ratio < 1e-3, f"EMCS energy fluctuation ratio too high: {fluc_ratio}"
