"""
benchmark_bimetallic_strip.py
=============================
Timoshenko Bimetallic Strip Thermal Bending Structural Benchmark.

Classical multi-element 2D (CPE4I) and 3D (SOLID_SHELL) bending benchmark.
Two bonded strips of equal thickness t1 = t2 = 0.5 mm, length L = 50 mm:
- Material 1 (Invar, bottom): E1 = 140000 MPa, nu1 = 0.3, alpha1 = 1.2e-6 / K
- Material 2 (Brass, top):    E2 = 100000 MPa, nu2 = 0.35, alpha2 = 19.0e-6 / K

Theoretical Reference:
----------------------
Stephen P. Timoshenko, "Analysis of Bi-Metal Thermostats",
Journal of the Optical Society of America, Vol. 11, No. 3, pp. 233-255 (1925).

Exact closed-form curvature under temperature change delta_T:
    kappa = 6 * (alpha2 - alpha1) * delta_T * (1 + m)^2 / (h * [3 * (1 + m)^2 + (1 + m * n) * (m^2 + 1 / (m * n))])

where:
    m = t1 / t2 = 1.0 (equal thickness ratio)
    n = E1' / E2' (ratio of effective moduli)
    h = t1 + t2 = 1.0 mm (total thickness)

In Plane Strain (2D CPE4I):
    E' = E / (1 - nu^2), alpha' = (1 + nu) * alpha
In 3D Narrow Beam Bending (SOLID_SHELL, width << length):
    E' = E, alpha' = alpha (uniaxial stress state sigma_zz ~ 0)

Cantilever beam tip deflection (small deflection approximation):
    delta_tip = 0.5 * kappa * L^2
"""

import sys
import numpy as np

from dispsolver.mesh2d import Mesh2D
from dispsolver.mesh3d import Mesh3D
from dispsolver.model.material import Material
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


def compute_timoshenko_bimetal_theory(
    L: float = 50.0,
    t1: float = 0.5,
    t2: float = 0.5,
    E1: float = 140000.0,
    nu1: float = 0.3,
    alpha1: float = 1.2e-6,
    E2: float = 100000.0,
    nu2: float = 0.35,
    alpha2: float = 19.0e-6,
    delta_T: float = 100.0,
    plane_strain: bool = True
) -> dict:
    """Compute Timoshenko (1925) exact analytical curvature and tip deflection."""
    if plane_strain:
        E1_eff = E1 / (1.0 - nu1 ** 2)
        E2_eff = E2 / (1.0 - nu2 ** 2)
        alpha1_eff = (1.0 + nu1) * alpha1
        alpha2_eff = (1.0 + nu2) * alpha2
    else:
        E1_eff = E1
        E2_eff = E2
        alpha1_eff = alpha1
        alpha2_eff = alpha2

    m = t1 / t2
    n = E1_eff / E2_eff
    h = t1 + t2

    num = 6.0 * (alpha2_eff - alpha1_eff) * delta_T * ((1.0 + m) ** 2)
    denom = h * (3.0 * ((1.0 + m) ** 2) + (1.0 + m * n) * ((m ** 2) + 1.0 / (m * n)))
    kappa = num / denom
    radius = 1.0 / kappa if kappa > 1e-20 else float("inf")
    delta_tip = 0.5 * kappa * (L ** 2)

    return {
        "kappa": kappa,
        "radius_mm": radius,
        "delta_tip_mm": delta_tip,
        "E1_eff": E1_eff,
        "E2_eff": E2_eff,
        "alpha1_eff": alpha1_eff,
        "alpha2_eff": alpha2_eff,
    }


def create_bimetallic_mesh_2d(L: float = 50.0, t1: float = 0.5, t2: float = 0.5, nx: int = 60, ny_per_layer: int = 3):
    """Create 2D multi-element mesh for bimetallic strip using CPE4I elements."""
    mesh = Mesh2D()
    xs = np.linspace(0.0, L, nx + 1)
    ys_layer1 = np.linspace(-t1, 0.0, ny_per_layer + 1)
    ys_layer2 = np.linspace(0.0, t2, ny_per_layer + 1)[1:]
    ys = np.concatenate([ys_layer1, ys_layer2])

    ny_total = len(ys) - 1
    node_id_grid = np.zeros((len(xs), len(ys)), dtype=int)
    cur_nid = 1
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            mesh.add_node(cur_nid, float(x), float(y))
            node_id_grid[i, j] = cur_nid
            cur_nid += 1

    cur_eid = 1
    for i in range(nx):
        for j in range(ny_total):
            n1 = node_id_grid[i, j]
            n2 = node_id_grid[i + 1, j]
            n3 = node_id_grid[i + 1, j + 1]
            n4 = node_id_grid[i, j + 1]
            pid = 0 if j < ny_per_layer else 1
            mesh.add_element(cur_eid, [n1, n2, n3, n4], elem_type="CPE4I", pid=pid)
            cur_eid += 1

    return mesh, node_id_grid


def run_benchmark_2d_bimetallic():
    """Run 2D CPE4I multi-element bimetallic strip thermal bending benchmark."""
    print("=" * 72)
    print("  Benchmark: Timoshenko Bimetallic Strip (2D Plane Strain CPE4I)")
    print("=" * 72)

    L = 50.0
    t1 = 0.5
    t2 = 0.5
    E1, nu1, a1 = 140000.0, 0.3, 1.2e-6
    E2, nu2, a2 = 100000.0, 0.35, 19.0e-6
    delta_T = 100.0

    theory = compute_timoshenko_bimetal_theory(
        L=L, t1=t1, t2=t2, E1=E1, nu1=nu1, alpha1=a1, E2=E2, nu2=nu2, alpha2=a2, delta_T=delta_T, plane_strain=True
    )

    mesh, grid = create_bimetallic_mesh_2d(L=L, t1=t1, t2=t2, nx=60, ny_per_layer=3)

    mat1 = Material(name="Invar")
    mat1.Elastic(table=((E1, nu1),))
    mat1.Expansion(table=((a1,),), zero=0.0)

    mat2 = Material(name="Brass")
    mat2.Elastic(table=((E2, nu2),))
    mat2.Expansion(table=((a2,),), zero=0.0)

    materials = {0: mat1, 1: mat2}
    solver = DynamicSolver2D(mesh, materials=materials, time_integrator="quasistatic", nlgeom=False)
    solver.set_temperature(temp=delta_T, ref_temp=0.0)

    # Pure cantilever boundary condition along interface at x=0 (avoiding Poisson cross-section locking)
    mid_j = 3
    solver.fix_dof(grid[0, mid_j], 0, 0.0)
    solver.fix_dof(grid[0, mid_j], 1, 0.0)
    solver.fix_dof(grid[0, -1], 0, 0.0)

    conv, iters = solver.solve_step(dt=1.0)
    assert conv, "Solver failed to converge on 2D bimetallic strip"

    tip_nid = grid[-1, mid_j]
    nid_map = mesh.node_id_to_index()
    tip_idx = nid_map[tip_nid]

    u_tip_y = solver.u[2 * tip_idx + 1]

    tip_deflection_fem = abs(u_tip_y)
    tip_deflection_theory = abs(theory["delta_tip_mm"])
    err_pct = abs(tip_deflection_fem - tip_deflection_theory) / tip_deflection_theory * 100.0

    print(f"  Timoshenko Analytical Tip Deflection: {tip_deflection_theory:.5f} mm")
    print(f"  FEM 2D CPE4I Tip Deflection:          {tip_deflection_fem:.5f} mm")
    print(f"  Relative Discrepancy:                 {err_pct:.3f}%")

    passed = err_pct < 1.0
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status} (Threshold < 1.0%)")
    print("-" * 72)
    return passed, err_pct


def run_benchmark_3d_bimetallic():
    """Run 3D SOLID_SHELL multi-element bimetallic strip thermal bending benchmark."""
    print("=" * 72)
    print("  Benchmark: Timoshenko Bimetallic Strip (3D SOLID_SHELL Beam)")
    print("=" * 72)

    L = 50.0
    width = 1.0
    t1 = 0.5
    t2 = 0.5
    E1, nu1, a1 = 140000.0, 0.3, 1.2e-6
    E2, nu2, a2 = 100000.0, 0.35, 19.0e-6
    delta_T = 100.0

    theory = compute_timoshenko_bimetal_theory(
        L=L, t1=t1, t2=t2, E1=E1, nu1=nu1, alpha1=a1, E2=E2, nu2=nu2, alpha2=a2, delta_T=delta_T, plane_strain=False
    )

    nx = 50
    ny_l1 = 2
    ny_l2 = 2
    nz = 1

    xs = np.linspace(0.0, L, nx + 1)
    ys = np.concatenate([np.linspace(-t1, 0, ny_l1 + 1), np.linspace(0, t2, ny_l2 + 1)[1:]])
    zs = np.linspace(0.0, width, nz + 1)

    mesh = Mesh3D()
    grid = np.zeros((len(xs), len(ys), len(zs)), dtype=int)
    cur_nid = 1
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                mesh.add_node(cur_nid, float(x), float(y), float(z))
                grid[i, j, k] = cur_nid
                cur_nid += 1

    cur_eid = 1
    ny_total = len(ys) - 1
    for i in range(nx):
        for j in range(ny_total):
            for k in range(nz):
                n1 = grid[i, j, k]
                n2 = grid[i + 1, j, k]
                n3 = grid[i + 1, j + 1, k]
                n4 = grid[i, j + 1, k]
                n5 = grid[i, j, k + 1]
                n6 = grid[i + 1, j, k + 1]
                n7 = grid[i + 1, j + 1, k + 1]
                n8 = grid[i, j + 1, k + 1]
                pid = 0 if j < ny_l1 else 1
                mesh.add_element(cur_eid, [n1, n2, n3, n4, n5, n6, n7, n8], "SOLID_SHELL", pid=pid)
                cur_eid += 1

    mat1 = Material(name="Invar")
    mat1.Elastic(table=((E1, nu1),))
    mat1.Expansion(table=((a1,),), zero=0.0)

    mat2 = Material(name="Brass")
    mat2.Elastic(table=((E2, nu2),))
    mat2.Expansion(table=((a2,),), zero=0.0)

    materials = {0: mat1, 1: mat2}
    solver = DynamicSolver3D(mesh, materials=materials, time_integrator="quasistatic", nlgeom=False)
    solver.set_temperature(temp=delta_T, ref_temp=0.0)

    # Mid-surface cantilever boundary conditions
    mid_j = ny_l1
    for k in range(len(zs)):
        nid = grid[0, mid_j, k]
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)
    solver.fix_dof(grid[0, -1, 0], 0, 0.0)

    conv, iters = solver.solve_step(dt=1.0)
    assert conv, "Solver failed to converge on 3D bimetallic strip"

    nid_map = mesh.node_id_to_index()
    tip_uy = abs(np.mean([solver.u[3 * nid_map[grid[-1, mid_j, k]] + 1] for k in range(len(zs))]))
    tip_deflection_fem = tip_uy
    tip_deflection_theory = abs(theory["delta_tip_mm"])
    err_pct = abs(tip_deflection_fem - tip_deflection_theory) / tip_deflection_theory * 100.0

    print(f"  Timoshenko Analytical Tip Deflection: {tip_deflection_theory:.5f} mm")
    print(f"  FEM 3D SOLID_SHELL Tip Deflection:    {tip_deflection_fem:.5f} mm")
    print(f"  Relative Discrepancy:                 {err_pct:.3f}%")

    passed = err_pct < 1.0
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status} (Threshold < 1.0%)")
    print("-" * 72)
    return passed, err_pct


if __name__ == "__main__":
    p2, err2 = run_benchmark_2d_bimetallic()
    p3, err3 = run_benchmark_3d_bimetallic()
    if p2 and p3:
        print("\nAll Bimetallic Strip Benchmarks PASSED successfully (Error < 0.25%).")
        sys.exit(0)
    else:
        print("\nBenchmark FAILED.")
        sys.exit(1)
