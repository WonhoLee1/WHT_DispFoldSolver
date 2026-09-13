"""
benchmark_mesh_refinement_3d.py
===============================
Comprehensive 3D Mesh Refinement & Runtime Scaling Benchmark for 5-Layer Composite Cantilever.

Investigates:
1. Through-Thickness Refinement: 1, 2, 4 elements per layer (up to 800 elements, 5,166 DOFs).
2. True 3D Width Refinement: Ny = 1, 5, 10, 20 elements across width with free uy (up to 4,000 elements, 15,498 DOFs).
3. Wall-clock runtime scaling (Pure solve time vs DOFs, PARDISO performance).
4. Physical 3D effects: Poisson anticlastic curvature and edge effect.

Best element pairing: C3D8I (PET) + C3D8H (PSA) with Conformal Node-Sharing.
"""

from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

# Problem Constants
L_BEAM = 40.0       # mm
T_PET = 0.05        # mm (50 um)
T_PSA = 0.03        # mm (30 um)
TOTAL_H = 3 * T_PET + 2 * T_PSA  # 0.21 mm (210 um)
WIDTH = 1.0         # mm

E_PET = 4000.0      # MPa
NU_PET = 0.3

MU_PSA = 0.05       # MPa
NU_PSA = 0.499
E_PSA = 2.0 * MU_PSA * (1.0 + NU_PSA)  # ~0.15 MPa

P_TIP = 0.0015      # N (1.5 mN)
PID_PET = 1
PID_PSA = 2


def make_5layer_mesh_3d_parametric(
    nx: int = 40,
    ny: int = 1,
    nz_per_layer: int = 1,
    pet_elem_type: str = "C3D8I",
    psa_elem_type: str = "C3D8H"
) -> Tuple[Mesh3D, List[int], List[int], Dict[str, Any]]:
    """Parametric 3D conformal structured mesh generator for 5-layer composite."""
    mesh = Mesh3D()
    layer_thicknesses = [T_PET, T_PSA, T_PET, T_PSA, T_PET]
    layer_types = [pet_elem_type, psa_elem_type, pet_elem_type, psa_elem_type, pet_elem_type]
    layer_pids = [PID_PET, PID_PSA, PID_PET, PID_PSA, PID_PET]

    dx = L_BEAM / float(nx)
    dy = WIDTH / float(ny)

    # Build z coordinates
    z_subdivisions = []
    z_elem_pids = []
    z_elem_types = []
    z_coords = [0.0]

    for lay_idx, t in enumerate(layer_thicknesses):
        dz = t / float(nz_per_layer)
        for _ in range(nz_per_layer):
            z_coords.append(z_coords[-1] + dz)
            z_elem_pids.append(layer_pids[lay_idx])
            z_elem_types.append(layer_types[lay_idx])

    nz_total = len(z_elem_pids)
    num_z_nodes = nz_total + 1

    # Node grid: (nx + 1, ny + 1, num_z_nodes)
    grid = np.zeros((nx + 1, ny + 1, num_z_nodes), dtype=np.int32)
    node_id = 1
    root_nodes = []
    tip_nodes = []

    for k in range(num_z_nodes):
        z_val = z_coords[k]
        for j in range(ny + 1):
            y_val = j * dy
            for i in range(nx + 1):
                x_val = i * dx
                mesh.add_node(node_id, x_val, y_val, z_val)
                grid[i, j, k] = node_id
                if i == 0:
                    root_nodes.append(node_id)
                if i == nx:
                    tip_nodes.append(node_id)
                node_id += 1

    # Element generation
    elem_id = 1
    for k in range(nz_total):
        etype = z_elem_types[k]
        pid = z_elem_pids[k]
        for j in range(ny):
            for i in range(nx):
                n1 = int(grid[i, j, k])
                n2 = int(grid[i + 1, j, k])
                n3 = int(grid[i + 1, j + 1, k])
                n4 = int(grid[i, j + 1, k])
                n5 = int(grid[i, j, k + 1])
                n6 = int(grid[i + 1, j, k + 1])
                n7 = int(grid[i + 1, j + 1, k + 1])
                n8 = int(grid[i, j + 1, k + 1])
                mesh.add_element(elem_id, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type=etype, pid=pid)
                elem_id += 1

    meta = {
        "grid": grid,
        "z_coords": z_coords,
        "nx": nx,
        "ny": ny,
        "nz_total": nz_total,
        "nz_per_layer": nz_per_layer
    }
    return mesh, root_nodes, tip_nodes, meta


def run_parametric_case(
    nx: int = 40,
    ny: int = 1,
    nz_per_layer: int = 1,
    fix_uy: bool = True,
    label: str = ""
) -> Dict[str, Any]:
    """Execute single parametric 3D case and return detailed metrics."""
    mesh, root_nodes, tip_nodes, meta = make_5layer_mesh_3d_parametric(
        nx=nx, ny=ny, nz_per_layer=nz_per_layer, pet_elem_type="C3D8I", psa_elem_type="C3D8H"
    )

    materials = {
        PID_PET: {"E": E_PET, "nu": NU_PET},
        PID_PSA: {"E": E_PSA, "nu": NU_PSA}
    }

    solver = DynamicSolver3D(mesh, material_params={"E": E_PET, "nu": NU_PET}, nlgeom=True)
    solver.materials = materials
    solver._setup_numba_topology()

    # Apply Boundary Conditions
    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)  # fix ux
        solver.fix_dof(nid, 1, 0.0)  # fix uy
        solver.fix_dof(nid, 2, 0.0)  # fix uz

    if fix_uy:
        # Pseudo-2D plane strain condition
        for nid in mesh.nodes.keys():
            solver.fix_dof(nid, 1, 0.0)
    else:
        # True 3D: only prevent rigid-body lateral shift at root centerline
        # Already root is fixed in uy, which prevents rigid body translation/rotation!
        pass

    nid_map = mesh.node_id_to_index()
    n_substeps = 15
    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    tip_p_per_node = -P_TIP / float(len(tip_nodes))

    p_hist = [0.0]
    uz_hist = [0.0]
    ux_hist = [0.0]
    iters_total = 0

    t_solve_start = time.perf_counter()

    for s in range(1, n_substeps + 1):
        frac = float(s) / float(n_substeps)
        ramp = frac * frac * (3.0 - 2.0 * frac)
        for nid in tip_nodes:
            idx = nid_map[nid]
            f_ext[3 * idx + 2] = tip_p_per_node * ramp

        conv, iters = solver.solve_step(dt=1.0, f_ext=f_ext, max_iters=25)
        iters_total += iters

        tip_uzs = [solver.u[3 * nid_map[nid] + 2] for nid in tip_nodes]
        tip_uxs = [solver.u[3 * nid_map[nid]] for nid in tip_nodes]
        p_hist.append(ramp * P_TIP)
        uz_hist.append(float(np.mean(tip_uzs)))
        ux_hist.append(float(np.mean(tip_uxs)))

    t_solve_wall = time.perf_counter() - t_solve_start

    # Deformed centerline profile
    grid = meta["grid"]
    center_j = ny // 2
    profile_x = []
    profile_z = []
    for i in range(nx + 1):
        # Average across z at center_j
        x_pts = [mesh.nodes[int(grid[i, center_j, k])].x + solver.u[3 * nid_map[int(grid[i, center_j, k])]] for k in range(meta["nz_total"] + 1)]
        z_pts = [mesh.nodes[int(grid[i, center_j, k])].z + solver.u[3 * nid_map[int(grid[i, center_j, k])] + 2] for k in range(meta["nz_total"] + 1)]
        profile_x.append(float(np.mean(x_pts)))
        profile_z.append(float(np.mean(z_pts)))

    # Free-end staircase shear slip: ux(top PET) - ux(bot PET)
    # Average across width at tip
    top_nids = [int(grid[-1, j, -1]) for j in range(ny + 1)]
    bot_nids = [int(grid[-1, j, 0]) for j in range(ny + 1)]
    ux_top = np.mean([solver.u[3 * nid_map[n]] for n in top_nids])
    ux_bot = np.mean([solver.u[3 * nid_map[n]] for n in bot_nids])
    shear_slip_um = abs(ux_top - ux_bot) * 1000.0  # in um

    # 3D Anticlastic curvature: difference in uz across width at tip
    uz_tip_y0 = solver.u[3 * nid_map[int(grid[-1, 0, meta['nz_total'] // 2])] + 2]
    uz_tip_ymid = solver.u[3 * nid_map[int(grid[-1, center_j, meta['nz_total'] // 2])] + 2]
    anticlastic_delta_uz_um = abs(uz_tip_y0 - uz_tip_ymid) * 1000.0  # in um

    return {
        "label": label,
        "nx": nx,
        "ny": ny,
        "nz_per_layer": nz_per_layer,
        "fix_uy": fix_uy,
        "num_elements": mesh.num_elements,
        "num_nodes": mesh.num_nodes,
        "num_dofs": solver.num_dofs,
        "final_deflection_mm": float(abs(uz_hist[-1])),
        "shear_slip_um": float(shear_slip_um),
        "anticlastic_delta_uz_um": float(anticlastic_delta_uz_um),
        "total_iters": iters_total,
        "t_wall_sec": round(t_solve_wall, 3),
        "ms_per_step": round((t_solve_wall / n_substeps) * 1000.0, 1),
        "profile_x": profile_x,
        "profile_z": profile_z
    }


def run_all_refinements():
    print("=" * 80, flush=True)
    print("3D MESH REFINEMENT & RUNTIME SCALING BENCHMARK", flush=True)
    print("5-Layer Composite (PET-PSA-PET-PSA-PET) Cantilever Beam", flush=True)
    print("Element Pairing: C3D8I (PET) + C3D8H (PSA) Conformal", flush=True)
    print("=" * 80, flush=True)

    # Pre-warm JIT with a tiny 1-element dummy mesh to isolate compilation time
    print("\n>>> Warming up Numba JIT kernels ...", flush=True)
    _ = run_parametric_case(nx=2, ny=1, nz_per_layer=1, fix_uy=True, label="WARMUP")
    print("    JIT Warmup complete! Measuring pure solve times.\n", flush=True)

    results = {
        "thickness_study": [],
        "width_study_3d": []
    }

    # -------------------------------------------------------------
    # Study 1: Through-Thickness Refinement (1, 2, 4 elements/layer)
    # -------------------------------------------------------------
    print(">>> Stage 1: Through-Thickness Refinement (Nx=40, Ny=1, Plane-Strain) ...", flush=True)
    for nz in [1, 2, 4]:
        name = f"Thickness-{nz}elem/layer"
        print(f"    Running {name} ...", flush=True)
        res = run_parametric_case(nx=40, ny=1, nz_per_layer=nz, fix_uy=True, label=name)
        results["thickness_study"].append(res)
        print(f"      Elems: {res['num_elements']} | DOFs: {res['num_dofs']} | Defl: {res['final_deflection_mm']:.4f} mm | Slip: {res['shear_slip_um']:.2f} um | Time: {res['t_wall_sec']:.3f} s ({res['ms_per_step']} ms/step)", flush=True)

    # -------------------------------------------------------------
    # Study 2: True 3D Width Refinement (Ny = 1, 5, 10, 20 with Free uy)
    # -------------------------------------------------------------
    print("\n>>> Stage 2: True 3D Width Refinement (Nx=40, Nz=5, Free uy) ...", flush=True)
    for ny in [1, 5, 10, 20]:
        name = f"Width-Ny={ny}"
        print(f"    Running {name} ...", flush=True)
        res = run_parametric_case(nx=40, ny=ny, nz_per_layer=1, fix_uy=False, label=name)
        results["width_study_3d"].append(res)
        print(f"      Elems: {res['num_elements']} | DOFs: {res['num_dofs']} | Defl: {res['final_deflection_mm']:.4f} mm | Slip: {res['shear_slip_um']:.2f} um | Anticlastic: {res['anticlastic_delta_uz_um']:.2f} um | Time: {res['t_wall_sec']:.3f} s", flush=True)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "benchmark_mesh_refinement_3d.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] All refinement cases completed and saved to {json_path}", flush=True)
    return results


if __name__ == "__main__":
    run_all_refinements()
