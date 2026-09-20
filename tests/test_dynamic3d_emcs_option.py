"""
test_dynamic3d_emcs_option.py
=============================
Verification of the Unified DynamicSolver3D Universal Interface with time_integrator="emcs" option.
"""

from __future__ import annotations
import numpy as np

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d import DynamicSolver3D


def test_dynamic3d_emcs_unified_option():
    """Verify DynamicSolver3D(..., time_integrator='emcs') unified flag option."""
    mesh = Mesh3D()
    mesh.add_node(1, 0.0, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0, 0.0)
    mesh.add_node(3, 1.0, 1.0, 0.0)
    mesh.add_node(4, 0.0, 1.0, 0.0)
    mesh.add_node(5, 0.0, 0.0, 1.0)
    mesh.add_node(6, 1.0, 0.0, 1.0)
    mesh.add_node(7, 1.0, 1.0, 1.0)
    mesh.add_node(8, 0.0, 1.0, 1.0)
    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], "C3D8")

    # Instantiate unified DynamicSolver3D with time_integrator="emcs"
    solver = DynamicSolver3D(
        mesh=mesh,
        material_params={"E": 200000.0, "nu": 0.3},
        time_integrator="emcs"
    )

    # Fix base node 1
    solver.fix_dof(1, 0, 0.0)
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(1, 2, 0.0)

    # Solve step using unified interface
    conv, iters = solver.solve_step(dt=0.001, tol=1e-6, use_anderson_accel=True)
    assert conv, "DynamicSolver3D with time_integrator='emcs' failed!"

    E_tot, E_kin, E_strain = solver.compute_total_energy()
    print(f"\n[UNIFIED INTERFACE PASS] Step 1 Iterations: {iters} | Total Energy: {E_tot:.6e}")


if __name__ == "__main__":
    test_dynamic3d_emcs_unified_option()
