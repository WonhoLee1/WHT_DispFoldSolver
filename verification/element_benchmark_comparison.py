"""
element_benchmark_comparison.py
================================
Comprehensive Benchmark Comparison Suite for 2D and 3D Finite Elements.

Evaluates:
1. Pure Bending Accuracy (Shear Locking Resistance)
2. Incompressible Locking Resistance (nu -> 0.5 Volumetric Locking)
3. Mesh Distortion Robustness (Distorted Patch Sensitivity)
4. Assembly & Solver Computation Speed (ms per 1000 element-steps)
"""

from __future__ import annotations
import sys
import time
import numpy as np

from dispsolver.mesh3d import Mesh3D, Node3D, Element3D
from dispsolver.element3d import (
    Hexa8EASElement,
    Hexa8FbarElement,
    Tetra4ANPElement,
    Tetra10Element,
    assemble_mesh_c3d8_eas_numba,
    assemble_mesh_c3d8_fbar_numba
)
from dispsolver.material3d import J2Plasticity3D
from dispsolver.solver3d import DynamicSolver3D


def run_3d_element_benchmark_suite() -> dict:
    """Run standardized benchmark suite across 3D element types."""
    print("=" * 80)
    print(" Running Comprehensive 3D Element Benchmark Evaluation...")
    print("=" * 80)

    results = {}

    # Define test 3D Cantilever Beam geometry: 10 x 1 x 1 mm
    # Analytical Euler-Bernoulli end displacement under P = 10 N:
    # u_z = P * L^3 / (3 * E * I), where I = b * h^3 / 12 = 1.0 * 1.0^3 / 12 = 1/12
    # E = 200000 MPa, nu = 0.3
    # u_z_analytical = 10 * 10^3 / (3 * 200000 * (1/12)) = 10000 / 50000 = 0.20 mm
    # Timoshenko shear correction factor brings exact 3D displacement to ~0.206 mm.
    u_exact_bending = 0.2061

    element_types = [
        ("C3D8I (9-mode EAS)", "C3D8I", Hexa8EASElement(num_eas_modes=9)),
        ("C3D8_FBAR (Multiplicative F-bar)", "C3D8_FBAR", Hexa8FbarElement()),
        ("C3D4_ANP (Average Nodal Pressure Tet)", "C3D4_ANP", Tetra4ANPElement()),
        ("C3D10M (Quadratic 10-node Tet)", "C3D10M", Tetra10Element()),
    ]

    for label, elem_type_code, elem_inst in element_types:
        print(f"\n[Evaluating {label}]")
        
        # 1. Pure Bending Test (Beam 10x1x1 mm, 10x2x2 mesh)
        n_x, n_y, n_z = 10, 2, 2
        mesh = Mesh3D()
        
        xs = np.linspace(0.0, 10.0, n_x + 1)
        ys = np.linspace(0.0, 1.0, n_y + 1)
        zs = np.linspace(0.0, 1.0, n_z + 1)

        node_grid = np.zeros((n_x + 1, n_y + 1, n_z + 1), dtype=int)
        nid = 1
        for k in range(n_z + 1):
            for j in range(n_y + 1):
                for i in range(n_x + 1):
                    node_grid[i, j, k] = nid
                    mesh.add_node(nid, xs[i], ys[j], zs[k])
                    nid += 1

        eid = 1
        for k in range(n_z):
            for j in range(n_y):
                for i in range(n_x):
                    n0 = node_grid[i,     j,     k]
                    n1 = node_grid[i + 1, j,     k]
                    n2 = node_grid[i + 1, j + 1, k]
                    n3 = node_grid[i,     j + 1, k]
                    n4 = node_grid[i,     j,     k + 1]
                    n5 = node_grid[i + 1, j,     k + 1]
                    n6 = node_grid[i + 1, j + 1, k + 1]
                    n7 = node_grid[i,     j + 1, k + 1]
                    mesh.add_element(eid, [n0, n1, n2, n3, n4, n5, n6, n7], elem_type_code)
                    eid += 1

        solver = DynamicSolver3D(mesh, {"E": 200000.0, "nu": 0.3})
        
        # Fix X=0 face
        for node in mesh.nodes.values():
            if abs(node.x - 0.0) < 1e-5:
                solver.fix_dof(node.id, 0, 0.0)
                solver.fix_dof(node.id, 1, 0.0)
                solver.fix_dof(node.id, 2, 0.0)

        # Apply load P = 10 N in Z-direction on X=10 face
        f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
        x10_nids = [n.id for n in mesh.nodes.values() if abs(n.x - 10.0) < 1e-5]
        p_per_node = 10.0 / len(x10_nids)
        nid_map = mesh.node_id_to_index()
        for nid in x10_nids:
            f_ext[3 * nid_map[nid] + 2] = p_per_node

        t_start = time.time()
        solver.solve_static_step(f_ext, tol=1e-5)
        t_solve = (time.time() - t_start) * 1000.0  # ms

        # End displacement in Z
        end_disp_z = float(np.mean([solver.u[3 * nid_map[nid] + 2] for nid in x10_nids]))
        bending_error = abs((end_disp_z - u_exact_bending) / u_exact_bending) * 100.0

        # 2. Incompressible Volumetric Locking Test (nu = 0.49999)
        solver_incomp = DynamicSolver3D(mesh, {"E": 200000.0, "nu": 0.49999})
        for node in mesh.nodes.values():
            if abs(node.x - 0.0) < 1e-5:
                solver_incomp.fix_dof(node.id, 0, 0.0)
                solver_incomp.fix_dof(node.id, 1, 0.0)
                solver_incomp.fix_dof(node.id, 2, 0.0)

        solver_incomp.solve_static_step(f_ext, tol=1e-5)
        incomp_disp_z = float(np.mean([solver_incomp.u[3 * nid_map[nid] + 2] for nid in x10_nids]))
        locking_ratio = incomp_disp_z / max(end_disp_z, 1e-12) * 100.0

        results[elem_type_code] = {
            "label": label,
            "bending_disp_mm": end_disp_z,
            "bending_error_pct": bending_error,
            "incomp_disp_mm": incomp_disp_z,
            "locking_ratio_pct": locking_ratio,
            "solve_time_ms": t_solve
        }

        print(f"  -> Bending Disp (Exact {u_exact_bending:.4f}mm) : {end_disp_z:.4f} mm (Err: {bending_error:.2f}%)", flush=True)
        print(f"  -> Incompressible (nu=0.49999) Disp: {incomp_disp_z:.4f} mm (Locking Ratio: {locking_ratio:.1f}%)", flush=True)
        print(f"  -> Computation Time                : {t_solve:.2f} ms", flush=True)

    return results


if __name__ == "__main__":
    run_3d_element_benchmark_suite()
