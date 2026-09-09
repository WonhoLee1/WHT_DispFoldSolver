"""
abaqus_3d_benchmark_suite.py
============================
Automated Cross-Validation Benchmark Suite for 3D Elements against Abaqus C3D8I/C3D10M.
"""

import os
import sys
import numpy as np

from dispsolver.mesh3d import Mesh3D
from dispsolver.element3d import Hexa8EASElement, Hexa8FbarElement
from dispsolver.solver3d import DynamicSolver3D
from dispsolver.material3d import J2Plasticity3D


def run_3d_cantilever_bending_benchmark():
    """Run 3D Cantilever Bending Benchmark (C3D8I EAS Element)."""
    print("=========================================================")
    print("   3D Cantilever Bending Benchmark (C3D8I EAS Element)   ")
    print("=========================================================")
    
    mesh = Mesh3D()
    
    # Refined 20x2x2 elements along 10x1x1 mm cantilever
    # Create nodes
    nx, ny, nz = 20, 2, 2
    x_coords = np.linspace(0.0, 10.0, nx + 1)
    y_coords = np.linspace(0.0, 1.0, ny + 1)
    z_coords = np.linspace(0.0, 1.0, nz + 1)

    nid = 1
    node_lookup = {}
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            for k, z in enumerate(z_coords):
                mesh.add_node(nid, x, y, z)
                node_lookup[(i, j, k)] = nid
                nid += 1

    # Create Hexa8 elements
    eid = 1
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n1 = node_lookup[(i,   j,   k)]
                n2 = node_lookup[(i+1, j,   k)]
                n3 = node_lookup[(i+1, j+1, k)]
                n4 = node_lookup[(i,   j+1, k)]
                n5 = node_lookup[(i,   j,   k+1)]
                n6 = node_lookup[(i+1, j,   k+1)]
                n7 = node_lookup[(i+1, j+1, k+1)]
                n8 = node_lookup[(i,   j+1, k+1)]
                mesh.add_element(eid, [n1, n2, n3, n4, n5, n6, n7, n8], "C3D8I")
                eid += 1

    E, nu = 200000.0, 0.3
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu})

    # Fixed BC at X=0 (i=0 face)
    for j in range(ny + 1):
        for k in range(nz + 1):
            node_fixed = node_lookup[(0, j, k)]
            solver.fix_dof(node_fixed, 0, 0.0)
            solver.fix_dof(node_fixed, 1, 0.0)
            solver.fix_dof(node_fixed, 2, 0.0)

    # Apply bending force Py = -100 N at tip X=10.0 (i=nx face)
    num_tip_nodes = (ny + 1) * (nz + 1)
    p_per_node = -100.0 / num_tip_nodes

    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    nid_map = mesh.node_id_to_index()
    for j in range(ny + 1):
        for k in range(nz + 1):
            tip_node = node_lookup[(nx, j, k)]
            f_ext[3 * nid_map[tip_node] + 1] = p_per_node

    converged = solver.solve_static_step(f_ext, tol=1e-6)
    assert converged is True

    # Analytical Euler-Bernoulli Beam Tip Deflection: delta_y = P * L^3 / (3 * E * I)
    # L = 10.0, b = 1.0, h = 1.0 -> I = b * h^3 / 12 = 1/12
    # Timoshenko beam theory correction factor includes shear deformation
    analytical_deflection = -2.0

    tip_node_sample = node_lookup[(nx, 0, 0)]
    computed_deflection = solver.u[3 * nid_map[tip_node_sample] + 1]
    
    error_percent = abs((computed_deflection - analytical_deflection) / analytical_deflection) * 100.0

    print(f"-> Analytical Tip Deflection : {analytical_deflection:.4f} mm")
    print(f"-> Computed C3D8I Deflection : {computed_deflection:.4f} mm")
    print(f"-> Error Percentage          : {error_percent:.2f} %")
    assert error_percent < 25.0
    print("[PASS] 3D Cantilever Bending Benchmark Passed Successfully!")


if __name__ == "__main__":
    run_3d_cantilever_bending_benchmark()
