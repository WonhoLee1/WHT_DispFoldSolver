"""
benchmark_abaqus_3delem.py
==========================
Abaqus 2024 Official Verification Benchmark: simaver-c-3delem.htm
Reference input file: SIMAINPRefResources/ec38sfs2.inp

Problem:
  Single 3D solid element (2.0 x 2.0 x 1.0) subjected to combined 3D
  triaxial normal stress and shear stresses:
  sigma_xx = sigma_yy = sigma_zz = sigma_xy = sigma_yz = sigma_xz = -1000.0

Exact Analytical Solution (Abaqus Verification Manual):
  E = 30.0e6, nu = 0.3
  eps_xx = eps_yy = eps_zz = -1.3333333333333333e-5
  gamma_xy = gamma_yz = gamma_xz = -8.6666666666666667e-5

  Displacement field:
    u_x = x * eps_xx + y * gamma_xy
    u_y = y * eps_yy + z * gamma_yz
    u_z = z * eps_zz + x * gamma_xz
"""

from __future__ import annotations
import sys
import numpy as np
from typing import Dict, List, Tuple, Any

from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

# Analytical material and response
E = 30.0e6
nu = 0.3
EPS = -1.3333333333333333e-5
GAMMA = -8.6666666666666667e-5

def exact_u(x: float, y: float, z: float) -> Tuple[float, float, float]:
    ux = x * EPS + y * GAMMA
    uy = y * EPS + z * GAMMA
    uz = z * EPS + x * GAMMA
    return ux, uy, uz

def make_single_hex_mesh(elem_type: str = "C3D8") -> Mesh3D:
    mesh = Mesh3D()
    # 8 nodes of [0, 2] x [0, 2] x [0, 1]
    coords = [
        (1, 0.0, 0.0, 0.0),
        (2, 2.0, 0.0, 0.0),
        (3, 2.0, 2.0, 0.0),
        (4, 0.0, 2.0, 0.0),
        (5, 0.0, 0.0, 1.0),
        (6, 2.0, 0.0, 1.0),
        (7, 2.0, 2.0, 1.0),
        (8, 0.0, 2.0, 1.0),
    ]
    for nid, x, y, z in coords:
        mesh.add_node(nid, x, y, z)
        
    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], elem_type=elem_type)
    return mesh

def build_ext_forces() -> Dict[Tuple[int, int], float]:
    """Compute total nodal equivalent loads from DLOAD (P1..P6 = 1000) and CLOAD.
    Returns dict: {(nid, dof): force_val}
    dof: 0=x, 1=y, 2=z
    """
    forces: Dict[Tuple[int, int], float] = {}
    def add_f(nid: int, dof: int, val: float):
        key = (nid, dof)
        forces[key] = forces.get(key, 0.0) + val

    # DLOAD P1..P6 = 1000.0
    # Pressure P1 (Bottom, z=0, outward norm -z): load is inwards -> +z
    # Area = 2*2 = 4. Total force = 4000. Each node (1,2,3,4) gets +1000 in z.
    for n in [1, 2, 3, 4]:
        add_f(n, 2, 1000.0)

    # Pressure P2 (Top, z=1, outward norm +z): load is inwards -> -z
    # Area = 4. Total force = -4000. Each node (5,6,7,8) gets -1000 in z.
    for n in [5, 6, 7, 8]:
        add_f(n, 2, -1000.0)

    # Pressure P3 (Front, y=0, outward norm -y): load inwards -> +y
    # Area = 2*1 = 2. Total force = 2000. Nodes (1,2,6,5) get +500 in y.
    for n in [1, 2, 6, 5]:
        add_f(n, 1, 500.0)

    # Pressure P4 (Right, x=2, outward norm +x): load inwards -> -x
    # Area = 2*1 = 2. Total force = -2000. Nodes (2,3,7,6) get -500 in x.
    for n in [2, 3, 7, 6]:
        add_f(n, 0, -500.0)

    # Pressure P5 (Back, y=2, outward norm +y): load inwards -> -y
    # Area = 2*1 = 2. Total force = -2000. Nodes (4,3,7,8) get -500 in y.
    for n in [4, 3, 7, 8]:
        add_f(n, 1, -500.0)

    # Pressure P6 (Left, x=0, outward norm -x): load inwards -> +x
    # Area = 2*1 = 2. Total force = 2000. Nodes (1,4,8,5) get +500 in x.
    for n in [1, 4, 8, 5]:
        add_f(n, 0, 500.0)

    # CLOAD from ec38sfs2.inp
    cloads = [
        (2, 0,  1500.00),
        (3, 0,   500.00),
        (3, 1,   500.00),
        (3, 2, -1000.00),
        (4, 0,   500.00),
        (4, 1,  1500.00),
        (5, 1,  -500.00),
        (5, 2,  1000.00),
        (6, 0,  -500.00),
        (6, 1, -1500.00),
        (7, 0, -1500.00),
        (7, 1, -1500.00),
        (7, 2, -1000.00),
        (8, 0, -1500.00),
        (8, 1,  -500.00),
    ]
    for nid, dof, val in cloads:
        add_f(nid, dof, val)

    return forces

def run_test_for_element(elem_type: str):
    mesh = make_single_hex_mesh(elem_type)
    nid_map = mesh.node_id_to_index()
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=False)

    # Boundary conditions from ec38sfs2.inp:
    # 1, 1, 3 (ux=0, uy=0, uz=0)
    # 2, 2    (uy=0)
    # 4, 3    (uz=0)
    # 5, 1    (ux=0)
    solver.fix_dof(1, 0, 0.0)
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(1, 2, 0.0)
    solver.fix_dof(2, 1, 0.0)
    solver.fix_dof(4, 2, 0.0)
    solver.fix_dof(5, 0, 0.0)

    # Build external force vector
    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    forces = build_ext_forces()
    for (nid, dof), val in forces.items():
        idx = nid_map[nid]
        f_ext[3 * idx + dof] += val

    conv, iters = solver.solve_step(dt=1.0, f_ext=f_ext, max_iters=25)
    if not conv:
        print(f"[{elem_type}] DIVERGED!")
        return

    # Check displacements against analytical solution at all 8 nodes
    max_disp_err = 0.0
    err_list = []
    for nid in range(1, 9):
        idx = nid_map[nid]
        node = mesh.nodes[nid]
        ux_ex, uy_ex, uz_ex = exact_u(node.x, node.y, node.z)
        ux_fe = solver.u[3 * idx + 0]
        uy_fe = solver.u[3 * idx + 1]
        uz_fe = solver.u[3 * idx + 2]
        
        diff = np.sqrt((ux_fe - ux_ex)**2 + (uy_fe - uy_ex)**2 + (uz_fe - uz_ex)**2)
        norm_ex = np.sqrt(ux_ex**2 + uy_ex**2 + uz_ex**2)
        rel_err = diff / max(norm_ex, 1e-12) * 100.0
        err_list.append(rel_err)
        max_disp_err = max(max_disp_err, rel_err)

    print(f"[{elem_type:<10}] CONVERGED (iters={iters:2d}) | Max Displacement Error vs Abaqus Exact = {max_disp_err:.6e}%")

def main():
    print("==========================================================================")
    print("Abaqus 2024 Official Benchmark simaver-c-3delem (ec38sfs2.inp)")
    print("Combined Triaxial Normal + Shear Loading: sigma_ij = -1000.0 psi")
    print("==========================================================================")
    
    test_elements = ["C3D8", "C3D8I", "C3D8R", "C3D8H", "C3D8_CR"]
    for etype in test_elements:
        run_test_for_element(etype)

if __name__ == "__main__":
    main()
