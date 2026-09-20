# -*- coding: utf-8 -*-
"""benchmark_field_ef1.py: field-variable-dependent elasticity/plasticity benchmarks.

Mirrors benchmark_abaqus_material_inp.py's temperature pair, with the
independent variable replaced by Abaqus field variable FV1 (*FIELD):

1. field_elastic_ef1: E(f=0) = 200000 MPa, E(f=1) = 160000 MPa, nu = 0.3.
   Unit C3D8 cube, prescribed axial strain eps_11 = 0.001:
   f=0   -> F = 200.0 N,  f=1 -> F = 160.0 N (area 1 mm^2).
   Midpoint f=0.5 -> F = 180.0 N (linear table interp check).
2. field_plastic_ef1: sy(f=0) = 250 MPa, sy(f=1) = 200 MPa, H = 2000 MPa.
   Same 0.005 trial strain as temp_plastic_mises; elevated field value
   must lower the hardened flow stress.
"""

from __future__ import annotations

import numpy as np

from dispsolver.mesh3d import Mesh3D
from dispsolver.model.material import Material
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.material.dependence_table import eval_table_1d_numba
from dispsolver.material.plastic_hardening import return_mapping_linear_kinematic


def create_unit_c3d8_cube() -> Mesh3D:
    """Create a single 1x1x1 mm C3D8 element mesh."""
    mesh = Mesh3D()
    coords = [
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
    ]
    for nid, (x, y, z) in enumerate(coords, start=1):
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, list(range(1, 9)), "C3D8", pid=0)
    return mesh


def _uniaxial_reaction(field_value: float) -> float:
    mesh = create_unit_c3d8_cube()
    mat = Material(name="Field_Elastic_Steel")
    mat.Elastic(field_table=((200000.0, 0.3, 0.0), (160000.0, 0.3, 1.0)))
    solver = DynamicSolver3D(mesh, materials={0: mat}, time_integrator="quasistatic")
    solver.set_field_variable(field_value)
    for nid in [1, 4, 5, 8]:
        solver.fix_dof(nid, 0, 0.0)
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(1, 2, 0.0)
    for nid in [2, 3, 6, 7]:
        solver.fix_dof(nid, 0, 0.001)
    solver.solve_step(dt=1.0)
    rf = solver.compute_reaction_forces()
    nid_map = mesh.node_id_to_index()
    return float(sum(rf[3 * nid_map[n] + 0] for n in [2, 3, 6, 7]))


def run_benchmark_field_elastic_ef1() -> bool:
    """Benchmark 1: field_elastic_ef1. PASS if all three loads < 1% error."""
    print("\n[Benchmark 1/2] field_elastic_ef1 (field-dependent elasticity)...")
    cases = ((0.0, 200.0), (0.5, 180.0), (1.0, 160.0))
    ok = True
    for f_val, ref in cases:
        F = _uniaxial_reaction(f_val)
        err = abs(F - ref) / ref
        print(f"  f={f_val}: Computed F = {F:.4f} N (Ref: {ref:.1f} N, Error: {err * 100:.3f}%)")
        ok = ok and err < 0.01
    print(f"  --> Verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def run_benchmark_field_plastic_ef1() -> bool:
    """Benchmark 2: field_plastic_ef1. PASS if elevated field lowers flow stress."""
    print("\n[Benchmark 2/2] field_plastic_ef1 (field-dependent J2 plasticity)...")
    E, nu, H = 200000.0, 0.3, 2000.0
    sy_tab = np.array([[250.0, 0.0], [200.0, 1.0]], dtype=np.float64)
    e_trial = np.array([0.005, -nu * 0.005, -nu * 0.005, 0.0, 0.0, 0.0], dtype=np.float64)
    alpha = np.zeros(6, dtype=np.float64)
    ep = np.zeros(6, dtype=np.float64)

    sigs = []
    for f_val in (0.0, 1.0):
        sy = float(eval_table_1d_numba(sy_tab[:, 1], sy_tab[:, 0], f_val))
        S, _, _, _, _ = return_mapping_linear_kinematic(
            e_trial, alpha, ep, 0.0, E, nu, sigma_y0=sy, H_iso=H, C_kin=0.0
        )
        sigs.append(float(S[0]))
        print(f"  f={f_val}: Yield = {sy:.1f} MPa => Hardened Stress = {S[0]:.2f} MPa")
    ok = sigs[1] < sigs[0]
    print(f"  --> Verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> bool:
    """Run the field ef1 benchmark pair."""
    ok1 = run_benchmark_field_elastic_ef1()
    ok2 = run_benchmark_field_plastic_ef1()
    print(f"\nfield_ef1 pair: {'2/2 PASS' if ok1 and ok2 else 'FAIL'}")
    return ok1 and ok2


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
