"""
benchmark_display_thermal_rollup.py
===================================
14-Layer Flexible Display Stackup Thermal Bending & Interlayer Shear Benchmark.

Evaluates thermal-mechanical coupled deformation in a 14-layer alternating
PET-PSA flexible display panel:
- 7 layers of PET (Polyethylene Terephthalate, stiff structural film):
  E = 4000 MPa, nu = 0.30, alpha = 1.5e-5 / K, thickness = 50 um
- 7 layers of PSA (Pressure-Sensitive Adhesive, soft compliant interlayer):
  E = 1.0 MPa, nu = 0.49, alpha = 1.0e-4 / K, thickness = 21.4 um
Total thickness h ~ 0.50 mm, Length L = 40.0 mm.

Physics & Verification Criteria:
--------------------------------
1. Free thermal expansion vs interlayer shear:
   Due to compliance of PSA (G_psa ~ 0.33 MPa vs E_pet = 4000 MPa),
   PET layers expand with minimal axial constraint while PSA accommodates
   interlayer shear deformation without delamination.
2. Under thermal change delta_T = 40 K (from 25°C room temp to 65°C operational temp),
   thermal strain in PET is eps_th_pet = 1.5e-5 * 40 = 6.0e-4,
   free elongation delta_L_pet = eps_th_pet * L = 0.024 mm.
3. The benchmark tracks:
   - Longitudinal expansion of top, middle, and bottom PET layers.
   - Interlayer shear slip profile at the free tip (x = L).
   - Solution convergence and thermal load balance.
"""

from __future__ import annotations

import sys
import numpy as np

from dispsolver.mesh2d import Mesh2D
from dispsolver.model.material import Material
from dispsolver.solver2d.dynamic2d import DynamicSolver2D


def create_14layer_display_mesh(
    L: float = 40.0,
    n_layers: int = 14,
    nx: int = 50,
    t_pet: float = 0.050,
    t_psa: float = 0.0214,
) -> tuple[Mesh2D, dict]:
    """Create 2D mesh of 14-layer alternating PET/PSA display panel."""
    mesh = Mesh2D()

    # Calculate y-coordinates for each layer boundary
    layer_thicknesses = []
    layer_pids = []
    for k in range(n_layers):
        if k % 2 == 0:
            layer_thicknesses.append(t_pet)
            layer_pids.append(0)  # PET
        else:
            layer_thicknesses.append(t_psa)
            layer_pids.append(1)  # PSA

    ys = [0.0]
    for t in layer_thicknesses:
        ys.append(ys[-1] + t)
    ys = np.array(ys, dtype=float)

    xs = np.linspace(0.0, L, nx + 1)
    grid = {}
    cur_nid = 1
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            mesh.add_node(cur_nid, float(x), float(y))
            grid[(i, j)] = cur_nid
            cur_nid += 1

    cur_eid = 1
    for i in range(nx):
        for j in range(n_layers):
            n1 = grid[(i, j)]
            n2 = grid[(i + 1, j)]
            n3 = grid[(i + 1, j + 1)]
            n4 = grid[(i, j + 1)]
            pid = layer_pids[j]
            mesh.add_element(cur_eid, [n1, n2, n3, n4], elem_type="CPE4I", pid=pid)
            cur_eid += 1

    return mesh, grid, ys


def run_14layer_thermal_benchmark(
    L: float = 40.0,
    delta_T: float = 40.0,
    verbose: bool = True
) -> dict:
    """Execute 14-layer display panel thermal coupling benchmark."""
    mesh, grid, ys = create_14layer_display_mesh(L=L, n_layers=14, nx=50)

    # PET material (stiff)
    mat_pet = Material(name="PET")
    mat_pet.Elastic(table=((4000.0, 0.30),))
    mat_pet.Expansion(table=((1.5e-5,),), zero=0.0)

    # PSA material (soft viscoelastic adhesive)
    mat_psa = Material(name="PSA")
    mat_psa.Elastic(table=((1.0, 0.49),))
    mat_psa.Expansion(table=((1.0e-4,),), zero=0.0)

    materials = {0: mat_pet, 1: mat_psa}
    solver = DynamicSolver2D(mesh, materials=materials, time_integrator="quasistatic", nlgeom=False)

    # Boundary conditions: Symmetry / Roller at left end x=0
    # Pin the center-bottom node to prevent rigid body drift
    ny_nodes = len(ys)
    mid_j = ny_nodes // 2
    for j in range(ny_nodes):
        solver.fix_dof(grid[(0, j)], 0, 0.0)  # ux = 0 at x = 0 (symmetry roller)
    solver.fix_dof(grid[(0, 0)], 1, 0.0)      # uy = 0 at (0,0) (prevent rigid vertical motion)

    # Prescribe temperature change delta_T
    solver.set_temperature(temp=delta_T, ref_temp=0.0)

    conv, iters = solver.solve_step(dt=1.0)
    assert conv, f"14-layer display thermal solve failed to converge in {iters} iters"

    nid_map = mesh.node_id_to_index()

    # Tip displacements at x = L across thickness
    nx = 50
    tip_ux = []
    tip_uy = []
    for j in range(ny_nodes):
        nid = grid[(nx, j)]
        idx = nid_map[nid]
        tip_ux.append(solver.u[2 * idx])
        tip_uy.append(solver.u[2 * idx + 1])

    tip_ux = np.array(tip_ux)
    tip_uy = np.array(tip_uy)

    # Theoretical free expansion of unconstrained PET
    # Under plane strain, effective expansion alpha' = (1 + nu)*alpha
    alpha_pet_eff = (1.0 + 0.30) * 1.5e-5
    expected_pet_ux = alpha_pet_eff * delta_T * L

    mean_tip_ux = float(np.mean(tip_ux))
    shear_slip_range = float(np.max(tip_ux) - np.min(tip_ux))
    err_pet_expansion = abs(mean_tip_ux - expected_pet_ux) / expected_pet_ux * 100.0

    passed = (conv and iters <= 5 and err_pet_expansion < 5.0)

    if verbose:
        print("=" * 72)
        print("  Benchmark: 14-Layer Flexible Display Thermal Expansion & Shear")
        print("=" * 72)
        print(f"  Delta T:                     {delta_T:.1f} K")
        print(f"  Expected PET Tip Elongation: {expected_pet_ux * 1000.0:.3f} um")
        print(f"  Mean FEM Tip Elongation:     {mean_tip_ux * 1000.0:.3f} um")
        print(f"  Relative Discrepancy:        {err_pet_expansion:.2f}%")
        print(f"  Tip Shear Differential:      {shear_slip_range * 1000.0:.3f} um")
        print(f"  Newton Iterations:           {iters}")
        print(f"  Status:                      {'PASS' if passed else 'FAIL'}")
        print("=" * 72)

    return {
        "passed": passed,
        "iters": iters,
        "expected_pet_ux": expected_pet_ux,
        "mean_tip_ux": mean_tip_ux,
        "err_pet_expansion_percent": err_pet_expansion,
        "shear_slip_range": shear_slip_range,
        "tip_ux": tip_ux,
        "tip_uy": tip_uy,
        "ys": ys,
    }


if __name__ == "__main__":
    res = run_14layer_thermal_benchmark(delta_T=40.0, verbose=True)
    if not res["passed"]:
        sys.exit(1)
