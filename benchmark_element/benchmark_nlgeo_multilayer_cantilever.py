"""
benchmark_nlgeo_multilayer_cantilever.py
========================================
Comprehensive 5-Layer Composite (PET-PSA-PET-PSA-PET) Cantilever Benchmark.

Features:
1. Conformal Node-Sharing vs Surface-to-Surface Tie Coupling comparisons.
2. High-precision runtime profiling (Wall-clock time, step time, iters).
3. Exact element and degree-of-freedom (DOF) accounting.
4. 2D vs 3D solid element suitability evaluations.

Dimensions:
- Length L = 40.0 mm, Width = 1.0 mm
- Thickness: 3x PET (50 um) + 2x PSA (30 um) = 210 um (0.21 mm)
- Properties:
  * PET: E = 4000 MPa, nu = 0.3
  * PSA: Neo-Hookean mu = 0.05 MPa (E ~ 0.15 MPa), nu = 0.499
  * Modulus Ratio: 80,000!
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

from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.constraint.surface_tie import SurfaceTieConstraint
from dispsolver.constraint3d.surface_tie3d import SurfaceTieConstraint3D

# ====================================================================
# Problem Constants
# ====================================================================
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


# ====================================================================
# 2D Conformal Mesh Generator
# ====================================================================
def make_5layer_mesh_2d_conformal(
    pet_elem_type: str, psa_elem_type: str, nx: int = 40
) -> Tuple[Mesh2D, List[int], List[int]]:
    if pet_elem_type.upper() == "CPE6M":
        from dispsolver.mesh2d.conformal_cpe6m_cpe4h_builder import create_5layer_cpe6m_cpe4h_conformal_mesh
        mesh, sets = create_5layer_cpe6m_cpe4h_conformal_mesh(
            length=L_BEAM, t_pet=T_PET, t_psa=T_PSA, nx=nx, pid_pet=PID_PET, pid_psa=PID_PSA
        )
        return mesh, sets["LEFT_NODES"], sets["RIGHT_NODES"]

    mesh = Mesh2D()
    layer_thicknesses = [T_PET, T_PSA, T_PET, T_PSA, T_PET]
    layer_types = [pet_elem_type, psa_elem_type, pet_elem_type, psa_elem_type, pet_elem_type]
    layer_pids = [PID_PET, PID_PSA, PID_PET, PID_PSA, PID_PET]

    dx = L_BEAM / nx
    y_coords = [0.0]
    for t in layer_thicknesses:
        y_coords.append(y_coords[-1] + t)

    node_id = 1
    grid = np.zeros((nx + 1, 6), dtype=np.int32)
    for j in range(6):
        y_val = y_coords[j]
        for i in range(nx + 1):
            mesh.add_node(node_id, i * dx, y_val)
            grid[i, j] = node_id
            node_id += 1

    elem_id = 1
    for j in range(5):
        etype = layer_types[j]
        pid = layer_pids[j]
        for i in range(nx):
            n1 = int(grid[i, j])
            n2 = int(grid[i + 1, j])
            n3 = int(grid[i + 1, j + 1])
            n4 = int(grid[i, j + 1])
            mesh.add_element(elem_id, [n1, n2, n3, n4], elem_type=etype, pid=pid)
            elem_id += 1

    root_nodes = sorted(list(grid[0, :]))
    tip_nodes = sorted(list(grid[-1, :]))
    return mesh, root_nodes, tip_nodes


# ====================================================================
# 2D Surface Tie Mesh Generator (Non-Conformal / Duplicate Interface Nodes)
# ====================================================================
def make_5layer_mesh_2d_tie(
    pet_elem_type: str, psa_elem_type: str, nx: int = 40
) -> Tuple[Mesh2D, List[int], List[int], List[Tuple[List[int], List[int]]]]:
    mesh = Mesh2D()
    layer_thicknesses = [T_PET, T_PSA, T_PET, T_PSA, T_PET]
    layer_types = [pet_elem_type, psa_elem_type, pet_elem_type, psa_elem_type, pet_elem_type]
    layer_pids = [PID_PET, PID_PSA, PID_PET, PID_PSA, PID_PET]

    dx = L_BEAM / nx
    y_base = 0.0
    node_id = 1
    elem_id = 1

    root_nodes = []
    tip_nodes = []
    tie_pairs = []

    layer_top_nodes = []
    layer_bot_nodes = []

    for j in range(5):
        t = layer_thicknesses[j]
        etype = layer_types[j]
        pid = layer_pids[j]

        curr_bot = []
        curr_top = []

        if etype.upper() == "CPE6M":
            # Quadratic triangle layer: 3 rows of nodes
            bot_nids = []
            for i in range(2 * nx + 1):
                x_val = i * (dx / 2.0)
                mesh.add_node(node_id, x_val, y_base)
                bot_nids.append(node_id)
                curr_bot.append(node_id)
                if i == 0: root_nodes.append(node_id)
                if i == 2 * nx: tip_nodes.append(node_id)
                node_id += 1

            mid_nids = []
            for i in range(2 * nx + 1):
                x_val = i * (dx / 2.0)
                mesh.add_node(node_id, x_val, y_base + 0.5 * t)
                mid_nids.append(node_id)
                if i == 0: root_nodes.append(node_id)
                if i == 2 * nx: tip_nodes.append(node_id)
                node_id += 1

            top_nids = []
            for i in range(2 * nx + 1):
                x_val = i * (dx / 2.0)
                mesh.add_node(node_id, x_val, y_base + t)
                top_nids.append(node_id)
                curr_top.append(node_id)
                if i == 0: root_nodes.append(node_id)
                if i == 2 * nx: tip_nodes.append(node_id)
                node_id += 1

            for i in range(nx):
                v1 = bot_nids[2 * i]
                m1 = bot_nids[2 * i + 1]
                v2 = bot_nids[2 * i + 2]

                m4 = mid_nids[2 * i]
                md = mid_nids[2 * i + 1]
                m2 = mid_nids[2 * i + 2]

                v4 = top_nids[2 * i]
                m3 = top_nids[2 * i + 1]
                v3 = top_nids[2 * i + 2]

                mesh.add_element(elem_id, [v1, v2, v3, m1, m2, md], elem_type="CPE6M", pid=pid)
                elem_id += 1
                mesh.add_element(elem_id, [v1, v3, v4, md, m3, m4], elem_type="CPE6M", pid=pid)
                elem_id += 1
        else:
            # 4-node quad layer
            for i in range(nx + 1):
                mesh.add_node(node_id, i * dx, y_base)
                curr_bot.append(node_id)
                if i == 0: root_nodes.append(node_id)
                if i == nx: tip_nodes.append(node_id)
                node_id += 1

            for i in range(nx + 1):
                mesh.add_node(node_id, i * dx, y_base + t)
                curr_top.append(node_id)
                if i == 0: root_nodes.append(node_id)
                if i == nx: tip_nodes.append(node_id)
                node_id += 1

            for i in range(nx):
                n1 = curr_bot[i]
                n2 = curr_bot[i + 1]
                n3 = curr_top[i + 1]
                n4 = curr_top[i]
                mesh.add_element(elem_id, [n1, n2, n3, n4], elem_type=etype, pid=pid)
                elem_id += 1

        if j > 0:
            # Pair previous layer's top with current layer's bot
            prev_top = layer_top_nodes[-1]
            tie_pairs.append((curr_bot, prev_top))  # (slave, master)

        layer_bot_nodes.append(curr_bot)
        layer_top_nodes.append(curr_top)
        y_base += t

    return mesh, sorted(list(set(root_nodes))), sorted(list(set(tip_nodes))), tie_pairs


# ====================================================================
# 3D Conformal Mesh Generator
# ====================================================================
def make_5layer_mesh_3d_conformal(
    pet_elem_type: str, psa_elem_type: str, nx: int = 40
) -> Tuple[Mesh3D, List[int], List[int]]:
    mesh = Mesh3D()
    layer_thicknesses = [T_PET, T_PSA, T_PET, T_PSA, T_PET]
    layer_types = [pet_elem_type, psa_elem_type, pet_elem_type, psa_elem_type, pet_elem_type]
    layer_pids = [PID_PET, PID_PSA, PID_PET, PID_PSA, PID_PET]

    dx = L_BEAM / nx
    z_coords = [0.0]
    for t in layer_thicknesses:
        z_coords.append(z_coords[-1] + t)

    node_id = 1
    grid = np.zeros((nx + 1, 2, 6), dtype=np.int32)
    for k in range(6):
        z_val = z_coords[k]
        for j in range(2):
            y_val = j * WIDTH
            for i in range(nx + 1):
                mesh.add_node(node_id, i * dx, y_val, z_val)
                grid[i, j, k] = node_id
                node_id += 1

    elem_id = 1
    for k in range(5):
        etype = layer_types[k]
        pid = layer_pids[k]
        for i in range(nx):
            n1 = int(grid[i, 0, k])
            n2 = int(grid[i + 1, 0, k])
            n3 = int(grid[i + 1, 1, k])
            n4 = int(grid[i, 1, k])
            n5 = int(grid[i, 0, k + 1])
            n6 = int(grid[i + 1, 0, k + 1])
            n7 = int(grid[i + 1, 1, k + 1])
            n8 = int(grid[i, 1, k + 1])
            mesh.add_element(elem_id, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type=etype, pid=pid)
            elem_id += 1

    root_nodes = sorted(list(grid[0, :, :].ravel()))
    tip_nodes = sorted(list(grid[-1, :, :].ravel()))
    return mesh, root_nodes, tip_nodes


# ====================================================================
# 3D Surface Tie Mesh Generator
# ====================================================================
def make_5layer_mesh_3d_tie(
    pet_elem_type: str, psa_elem_type: str, nx: int = 40
) -> Tuple[Mesh3D, List[int], List[int], List[Tuple[List[int], List[int]]]]:
    mesh = Mesh3D()
    layer_thicknesses = [T_PET, T_PSA, T_PET, T_PSA, T_PET]
    layer_types = [pet_elem_type, psa_elem_type, pet_elem_type, psa_elem_type, pet_elem_type]
    layer_pids = [PID_PET, PID_PSA, PID_PET, PID_PSA, PID_PET]

    dx = L_BEAM / nx
    z_base = 0.0
    node_id = 1
    elem_id = 1

    root_nodes = []
    tip_nodes = []
    tie_pairs = []

    layer_top_nodes = []

    for k in range(5):
        t = layer_thicknesses[k]
        etype = layer_types[k]
        pid = layer_pids[k]

        grid = np.zeros((nx + 1, 2, 2), dtype=np.int32)
        for kz in range(2):
            z_val = z_base + kz * t
            for j in range(2):
                y_val = j * WIDTH
                for i in range(nx + 1):
                    mesh.add_node(node_id, i * dx, y_val, z_val)
                    grid[i, j, kz] = node_id
                    if i == 0: root_nodes.append(node_id)
                    if i == nx: tip_nodes.append(node_id)
                    node_id += 1

        for i in range(nx):
            n1 = int(grid[i, 0, 0])
            n2 = int(grid[i + 1, 0, 0])
            n3 = int(grid[i + 1, 1, 0])
            n4 = int(grid[i, 1, 0])
            n5 = int(grid[i, 0, 1])
            n6 = int(grid[i + 1, 0, 1])
            n7 = int(grid[i + 1, 1, 1])
            n8 = int(grid[i, 1, 1])
            mesh.add_element(elem_id, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type=etype, pid=pid)
            elem_id += 1

        curr_bot = list(grid[:, :, 0].ravel())
        curr_top_faces = []
        for i in range(nx):
            face = (int(grid[i, 0, 1]), int(grid[i + 1, 0, 1]), int(grid[i + 1, 1, 1]), int(grid[i, 1, 1]))
            curr_top_faces.append(face)

        if k > 0:
            prev_top_faces = layer_top_nodes[-1]
            tie_pairs.append((curr_bot, prev_top_faces))

        layer_top_nodes.append(curr_top_faces)
        z_base += t

    return mesh, sorted(root_nodes), sorted(tip_nodes), tie_pairs


# ====================================================================
# Simulation Execution Engine
# ====================================================================
def run_simulation_2d(pet_type: str, psa_type: str, use_tie: bool = False, nx: int = 40) -> Dict[str, Any]:
    t_start = time.perf_counter()
    if use_tie:
        mesh, root_nodes, tip_nodes, tie_pairs = make_5layer_mesh_2d_tie(pet_type, psa_type, nx=nx)
    else:
        mesh, root_nodes, tip_nodes = make_5layer_mesh_2d_conformal(pet_type, psa_type, nx=nx)
        tie_pairs = []

    materials = {
        PID_PET: {"E": E_PET, "nu": NU_PET},
        PID_PSA: {"E": E_PSA, "nu": NU_PSA}
    }
    solver = DynamicSolver2D(mesh, materials=materials, nlgeom=True)

    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)

    # Attach SurfaceTie constraints if requested
    nid_map = mesh.node_id_to_index()
    coords = mesh.nodes_array()
    for pair_idx, (slaves, masters) in enumerate(tie_pairs):
        tie = SurfaceTieConstraint(
            slave_node_ids=slaves,
            master_node_ids=masters,
            nid_to_idx=nid_map,
            coords=coords,
            penalty_stiffness=1e4,
            name=f"TIE_{pair_idx}"
        )
        solver.constraints.append(tie)

    n_substeps = 15
    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    tip_p_per_node = -P_TIP / float(len(tip_nodes))

    p_hist = [0.0]
    uy_hist = [0.0]
    ux_hist = [0.0]
    iters_total = 0

    for s in range(1, n_substeps + 1):
        frac = float(s) / float(n_substeps)
        ramp = frac * frac * (3.0 - 2.0 * frac)
        for nid in tip_nodes:
            idx = nid_map[nid]
            f_ext[2 * idx + 1] = tip_p_per_node * ramp

        conv, iters = solver.solve_step(dt=1.0, f_ext=f_ext, max_iters=25)
        iters_total += iters

        tip_uys = [solver.u[2 * nid_map[nid] + 1] for nid in tip_nodes]
        tip_uxs = [solver.u[2 * nid_map[nid]] for nid in tip_nodes]
        p_hist.append(ramp * P_TIP)
        uy_hist.append(float(np.mean(tip_uys)))
        ux_hist.append(float(np.mean(tip_uxs)))

    t_wall = time.perf_counter() - t_start

    # Deformed centerline profile
    nodes_by_x: Dict[float, List[int]] = {}
    for nid, node in mesh.nodes.items():
        xr = round(node.x, 3)
        nodes_by_x.setdefault(xr, []).append(nid)

    profile_x = []
    profile_y = []
    for xr in sorted(nodes_by_x.keys()):
        nids = nodes_by_x[xr]
        cx = np.mean([mesh.nodes[n].x + solver.u[2 * nid_map[n]] for n in nids])
        cy = np.mean([mesh.nodes[n].y + solver.u[2 * nid_map[n] + 1] for n in nids])
        profile_x.append(float(cx))
        profile_y.append(float(cy))

    # Calculate free-end staircase shear slip: ux(top PET) - ux(bot PET)
    tip_nodes_sorted_y = sorted(tip_nodes, key=lambda n: mesh.nodes[n].y)
    ux_tip_bot = solver.u[2 * nid_map[tip_nodes_sorted_y[0]]]
    ux_tip_top = solver.u[2 * nid_map[tip_nodes_sorted_y[-1]]]
    shear_slip_um = abs(ux_tip_top - ux_tip_bot) * 1000.0  # in um

    n_pet_elems = sum(1 for e in mesh.elements.values() if e.pid == PID_PET)
    n_psa_elems = sum(1 for e in mesh.elements.values() if e.pid == PID_PSA)

    return {
        "p_history": p_hist,
        "uy_history": uy_hist,
        "ux_history": ux_hist,
        "profile_x": profile_x,
        "profile_y": profile_y,
        "final_uy": uy_hist[-1],
        "shear_slip_um": float(shear_slip_um),
        "total_iters": iters_total,
        "t_wall_sec": round(t_wall, 3),
        "ms_per_step": round((t_wall / n_substeps) * 1000.0, 1),
        "num_dofs": solver.num_dofs,
        "num_nodes": mesh.num_nodes,
        "num_elems_total": mesh.num_elements,
        "num_elems_pet": n_pet_elems,
        "num_elems_psa": n_psa_elems
    }


def run_simulation_3d(pet_type: str, psa_type: str, use_tie: bool = False, nx: int = 40) -> Dict[str, Any]:
    t_start = time.perf_counter()
    if use_tie:
        mesh, root_nodes, tip_nodes, tie_pairs = make_5layer_mesh_3d_tie(pet_type, psa_type, nx=nx)
    else:
        mesh, root_nodes, tip_nodes = make_5layer_mesh_3d_conformal(pet_type, psa_type, nx=nx)
        tie_pairs = []

    materials = {
        PID_PET: {"E": E_PET, "nu": NU_PET},
        PID_PSA: {"E": E_PSA, "nu": NU_PSA}
    }
    solver = DynamicSolver3D(mesh, material_params={"E": E_PET, "nu": NU_PET}, nlgeom=True)
    solver.materials = materials
    solver._setup_numba_topology()

    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)

    for nid in mesh.nodes.keys():
        solver.fix_dof(nid, 1, 0.0)

    if "R" in pet_type and hasattr(solver, "elem_controls") and solver.elem_controls is not None:
        solver.elem_controls[:, 0] = 0.5

    nid_map = mesh.node_id_to_index()
    coords = mesh.nodes_array()
    for pair_idx, (slaves, masters) in enumerate(tie_pairs):
        tie = SurfaceTieConstraint3D(
            slave_node_ids=slaves,
            master_faces=masters,
            nid_to_idx=nid_map,
            coords=coords,
            penalty_stiffness=1e4,
            name=f"TIE3D_{pair_idx}"
        )
        solver.constraints.append(tie)

    n_substeps = 15
    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    tip_p_per_node = -P_TIP / float(len(tip_nodes))

    p_hist = [0.0]
    uz_hist = [0.0]
    ux_hist = [0.0]
    iters_total = 0

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

    t_wall = time.perf_counter() - t_start

    nodes_by_x: Dict[float, List[int]] = {}
    for nid, node in mesh.nodes.items():
        xr = round(node.x, 3)
        nodes_by_x.setdefault(xr, []).append(nid)

    profile_x = []
    profile_z = []
    for xr in sorted(nodes_by_x.keys()):
        nids = nodes_by_x[xr]
        cx = np.mean([mesh.nodes[n].x + solver.u[3 * nid_map[n]] for n in nids])
        cz = np.mean([mesh.nodes[n].z + solver.u[3 * nid_map[n] + 2] for n in nids])
        profile_x.append(float(cx))
        profile_z.append(float(cz))

    tip_nodes_sorted_z = sorted(tip_nodes, key=lambda n: mesh.nodes[n].z)
    ux_tip_bot = solver.u[3 * nid_map[tip_nodes_sorted_z[0]]]
    ux_tip_top = solver.u[3 * nid_map[tip_nodes_sorted_z[-1]]]
    shear_slip_um = abs(ux_tip_top - ux_tip_bot) * 1000.0  # in um

    n_pet_elems = sum(1 for e in mesh.elements.values() if e.pid == PID_PET)
    n_psa_elems = sum(1 for e in mesh.elements.values() if e.pid == PID_PSA)

    return {
        "p_history": p_hist,
        "uy_history": uz_hist,
        "ux_history": ux_hist,
        "profile_x": profile_x,
        "profile_y": profile_z,
        "final_uy": uz_hist[-1],
        "shear_slip_um": float(shear_slip_um),
        "total_iters": iters_total,
        "t_wall_sec": round(t_wall, 3),
        "ms_per_step": round((t_wall / n_substeps) * 1000.0, 1),
        "num_dofs": solver.num_dofs,
        "num_nodes": mesh.num_nodes,
        "num_elems_total": mesh.num_elements,
        "num_elems_pet": n_pet_elems,
        "num_elems_psa": n_psa_elems
    }


# ====================================================================
# Driver Execution Function
# ====================================================================
def run_all_benchmarks():
    print("=" * 80)
    print("5-Layer Composite (PET-PSA-PET-PSA-PET) Cantilever Benchmark Suite")
    print("Evaluating Conformal vs SurfaceTie, Solve Times, and Element Counts")
    print(f"L={L_BEAM}mm, H={TOTAL_H*1000:.1f}um (PET 50um, PSA 30um), P_tip={P_TIP*1000:.1f}mN")
    print("=" * 80)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "benchmark_nlgeo_multilayer_cantilever.json"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {"2D": {}, "3D": {}}
    if "2D" not in results: results["2D"] = {}
    if "3D" not in results: results["3D"] = {}

    # 2D Candidate Suite
    suite_2d = [
        ("2D-Opt1 (CPE4I+CPE4H Conformal)", "CPE4I", "CPE4H", False),
        ("2D-Tie  (CPE4I+CPE4H SurfaceTie)", "CPE4I", "CPE4H", True),
        ("2D-Opt2 (CPE4R+CPE4H Conformal)", "CPE4R", "CPE4H", False),
        ("2D-CPE6M (CPE6M+CPE4H Conformal)", "CPE6M", "CPE4H", False),
        ("2D-CPE6M-Tie (CPE6M+CPE4H SurfaceTie)", "CPE6M", "CPE4H", True),
        ("2D-Base (CPE4+CPE4 Baseline)",    "CPE4",  "CPE4",  False),
    ]

    print(f"\n>>> Running 2D Candidate Suite ({len(suite_2d)} Cases) ...", flush=True)
    for label, pet_e, psa_e, tie in suite_2d:
        print(f"\n[2D] {label} ...", flush=True)
        res = run_simulation_2d(pet_e, psa_e, use_tie=tie, nx=40)
        results["2D"][label] = res
        print(f"     Deflection: {abs(res['final_uy']):.4f} mm | Slip: {res['shear_slip_um']:.2f} um", flush=True)
        print(f"     Time: {res['t_wall_sec']:.3f} s ({res['ms_per_step']} ms/step) | DOFs: {res['num_dofs']} | Elems: {res['num_elems_total']}", flush=True)

    # 3D Candidate Suite
    suite_3d = [
        ("3D-Opt1 (C3D8I+C3D8H Conformal)", "C3D8I", "C3D8H", False),
        ("3D-Tie  (C3D8I+C3D8H SurfaceTie)", "C3D8I", "C3D8H", True),
        ("3D-Opt2 (C3D8R+C3D8R Conformal)", "C3D8R", "C3D8R", False),
        ("3D-Base (C3D8+C3D8 Baseline)",    "C3D8",  "C3D8",  False),
    ]

    print("\n>>> Running 3D Candidate Suite (4 Cases) ...", flush=True)
    for label, pet_e, psa_e, tie in suite_3d:
        if label in results["3D"]:
            cached = results["3D"][label]
            print(f"\n[3D CACHED] {label} ...", flush=True)
            print(f"     Deflection: {abs(cached['final_uy']):.4f} mm | Slip: {cached['shear_slip_um']:.2f} um", flush=True)
            print(f"     Time: {cached['t_wall_sec']:.3f} s ({cached['ms_per_step']} ms/step) | DOFs: {cached['num_dofs']} | Elems: {cached['num_elems_total']}", flush=True)
            continue

        print(f"\n[3D] {label} ...", flush=True)
        res = run_simulation_3d(pet_e, psa_e, use_tie=tie, nx=40)
        results["3D"][label] = res
        print(f"     Deflection: {abs(res['final_uy']):.4f} mm | Slip: {res['shear_slip_um']:.2f} um", flush=True)
        print(f"     Time: {res['t_wall_sec']:.3f} s ({res['ms_per_step']} ms/step) | DOFs: {res['num_dofs']} | Elems: {res['num_elems_total']}", flush=True)

    # Save Results
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] All benchmark cases completed and saved to {json_path}", flush=True)
    return results


if __name__ == "__main__":
    run_all_benchmarks()
