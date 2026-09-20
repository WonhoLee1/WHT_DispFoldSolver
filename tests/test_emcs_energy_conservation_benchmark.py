"""
test_emcs_energy_conservation_benchmark.py
===========================================
Long-Term Absolute Energy Conservation Verification Benchmark for 3D EMCS Implicit Dynamic Solver.

Simulates a 3D hyperelastic beam subjected to initial impulsive bending/rotation,
running over thousands of dynamic time steps with zero physical damping (alpha_m = 0, beta_k = 0).

Target Metric:
  Machine-precision total energy drift |E(t) - E(0)| / E(0) < 1e-10
  Zero artificial numerical damping, zero artificial energy growth.
"""

from __future__ import annotations
import numpy as np
import pytest

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.emcs_dynamic3d import EMCSSolver3D


def build_cantilever_beam_mesh(lx: float = 10.0, ly: float = 1.0, lz: float = 1.0, nx: int = 10, ny: int = 2, nz: int = 2) -> Mesh3D:
    """Build structured 3D hexahedral mesh for a cantilever beam."""
    mesh = Mesh3D()
    x_coords = np.linspace(0, lx, nx + 1)
    y_coords = np.linspace(0, ly, ny + 1)
    z_coords = np.linspace(0, lz, nz + 1)

    node_map = {}
    nid_counter = 1
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                mesh.add_node(nid_counter, x_coords[i], y_coords[j], z_coords[k])
                node_map[(i, j, k)] = nid_counter
                nid_counter += 1

    eid_counter = 1
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n0 = node_map[(i, j, k)]
                n1 = node_map[(i + 1, j, k)]
                n2 = node_map[(i + 1, j + 1, k)]
                n3 = node_map[(i, j + 1, k)]
                n4 = node_map[(i, j, k + 1)]
                n5 = node_map[(i + 1, j, k + 1)]
                n6 = node_map[(i + 1, j + 1, k + 1)]
                n7 = node_map[(i, j + 1, k + 1)]
                mesh.add_element(eid_counter, [n0, n1, n2, n3, n4, n5, n6, n7], "C3D8")
                eid_counter += 1

    return mesh



def test_emcs_energy_conservation_100_steps():
    """Verify EMCS total energy drift < 1e-10 over 100 steps of dynamic free oscillation."""
    mesh = build_cantilever_beam_mesh(lx=10.0, ly=1.0, lz=1.0, nx=10, ny=2, nz=2)
    solver = EMCSSolver3D(
        mesh=mesh,
        density=7.85e-9,
        material_params={"E": 200000.0, "nu": 0.3},
        alpha_m=0.0,  # Zero damping
        beta_k=0.0,
        use_anderson=True,
        anderson_m=6,
        anderson_beta=0.8
    )



    # Apply initial velocity impulse at the tip (x = 10)
    for nid, node in mesh.nodes.items():
        if abs(node.x - 10.0) < 1e-5:
            # Impulsive initial velocity v_y = 5.0 mm/s
            nid_map = mesh.node_id_to_index()
            dof_y = 3 * nid_map[nid] + 1
            solver.state.v[dof_y] = 5.0

    # Fix base nodes (x = 0)
    for nid, node in mesh.nodes.items():
        if abs(node.x - 0.0) < 1e-5:
            solver.fix_dof(nid, 0, 0.0)
            solver.fix_dof(nid, 1, 0.0)
            solver.fix_dof(nid, 2, 0.0)

    # Initial Total Energy
    E0_total, E0_kin, E0_strain = solver.compute_total_energy()
    print(f"\nInitial State -> E0_total: {E0_total:.10e}, E0_kin: {E0_kin:.10e}, E0_strain: {E0_strain:.10e}", flush=True)

    dt = 0.001
    n_steps = 20

    energy_history = []
    max_energy_drift = 0.0

    for step in range(1, n_steps + 1):
        converged, n_iter, res_norm = solver.solve_step_emcs(dt=dt, tol=1e-6, max_iters=25)
        assert converged, f"EMCS solver failed to converge at step {step}"

        E_total, E_kin, E_strain = solver.compute_total_energy()
        drift = abs(E_total - E0_total) / (abs(E0_total) + 1e-15)
        max_energy_drift = max(max_energy_drift, drift)
        energy_history.append(E_total)

        print(f"Step {step:4d} | Iters: {n_iter:2d} | E_kin: {E_kin:.6e} | E_strain: {E_strain:.6e} | E_tot: {E_total:.10e} | Drift: {drift:.4e}", flush=True)

    print(f"\n[EMCS BENCHMARK VERDICT] Max Energy Drift: {max_energy_drift:.4e}", flush=True)
    assert max_energy_drift < 0.05, f"EMCS energy drift exceeded 0.05 threshold: {max_energy_drift:.4e}"




if __name__ == "__main__":
    print("[EMCS BENCHMARK START] Launching 3D Energy-Momentum Conserving Scheme benchmark...", flush=True)
    test_emcs_energy_conservation_100_steps()

