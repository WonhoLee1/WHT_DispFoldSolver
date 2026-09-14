"""
benchmark_macneal_harder.py
===========================
MacNeal-Harder (1985) Standard Finite Element Benchmark Suite.

Official References:
- R. H. MacNeal and R. L. Harder (1985), "A proposed standard set of problems to test
  finite element accuracy", Finite Elements in Analysis and Design, 1(1), 3-20.
- NAFEMS Standard Benchmarks & Abaqus Verification Manual.

Benchmarks Included:
1. 3D Twisted Beam Problem (MacNeal-Harder Problem 3)
   - 90-degree twisted ribbon under out-of-plane and in-plane tip shear loads.
   - Out-of-plane target tip deflection: 0.005424
   - In-plane target tip deflection: 0.001754
2. Cook's Membrane Shear Bending Benchmark (2D & 3D)
   - Skewed tapered panel under end shear load to evaluate combined bending-shear
     and mesh distortion sensitivity.
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Enforce UTF-8 stdout encoding on Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from benchmark_element.mechanics_patches import (
    make_twisted_beam_mesh,
    make_cooks_membrane_mesh_3d,
)
from benchmark_element.mechanics_patches_2d import (
    make_cooks_membrane_mesh_2d,
)


ALL_3D_ELEMENTS = [
    "C3D8",
    "C3D8I",
    "C3D8_FBAR",
    "C3D8_CR",
    "C3D8H",
    "C3D8R",
    "C3D4",
    "C3D4_ANP",
    "C3D10",
    "C3D10M",
    "C3D6",
]

ALL_2D_ELEMENTS = [
    "CPE4",
    "CPE4I",
    "CPE4R",
    "CPE4H",
    "CPE4_FBAR",
    "CPE4_CR",
    "CPE3",
    "CPE6",
    "CPE6M",
    "CPE8",
]


# =========================================================================
# Suite 1: MacNeal-Harder 3D Twisted Beam
# =========================================================================

def run_twisted_beam_test(
    elem_type: str,
    load_case: str = "out_of_plane",
    nx: int = 12,
    ny: int = 1,
    nz: int = 1
) -> Dict[str, Any]:
    """Execute MacNeal-Harder 3D Twisted Beam benchmark.
    
    Geometry:
      L = 12.0, W = 1.1, t = 0.32, 90 deg linear twist about X-axis.
      Root clamped at x=0 (u_x = u_y = u_z = 0).
      
    Material:
      E = 29.0e6, nu = 0.22
      
    Load cases (MacNeal & Harder 1985 canonical):
      - 'out_of_plane': Unit shear force P_z = 1.0 at tip. Target disp = 0.005424
      - 'in_plane':     Unit shear force P_y = 1.0 at tip. Target disp = 0.001754
    """
    E = 29.0e6
    nu = 0.22
    L = 12.0
    W = 1.1
    t = 0.32

    if load_case == "out_of_plane":
        target_disp = 0.005424
        dof_idx = 2  # Z-direction: transverse out-of-plane bending
    elif load_case == "in_plane":
        target_disp = 0.001754
        dof_idx = 1  # Y-direction: in-plane bending
    else:
        raise ValueError(f"Unknown load_case: {load_case}")

    mesh, root_nodes, tip_nodes, tip_center_nid = make_twisted_beam_mesh(
        elem_type=elem_type, L=L, W=W, t=t, twist_deg=90.0, nx=nx, ny=ny, nz=nz
    )
    nid_map = mesh.node_id_to_index()
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)

    # Fully clamp root face at x = 0
    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)

    # Distribute total load P = 1.0 across tip nodes
    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    p_per_node = 1.0 / len(tip_nodes)
    for nid in tip_nodes:
        idx = nid_map[nid]
        f_ext[3 * idx + dof_idx] = p_per_node

    converged, iters = solver.solve_step(dt=1.0, f_ext=f_ext, max_iters=50)
    if not converged:
        return {
            "status": "DIVERGED",
            "load_case": load_case,
            "tip_disp": float("nan"),
            "target": target_disp,
            "ratio": float("nan"),
            "error_pct": float("nan"),
            "iters": iters,
        }

    # Extract tip displacement at center node (or mean if center not found)
    if tip_center_nid > 0 and tip_center_nid in nid_map:
        idx = nid_map[tip_center_nid]
        computed_disp = float(solver.u[3 * idx + dof_idx])
    else:
        # Fallback to mean tip displacement
        tip_disps = [solver.u[3 * nid_map[n] + dof_idx] for n in tip_nodes]
        computed_disp = float(np.mean(tip_disps))

    ratio = computed_disp / target_disp
    err_pct = (ratio - 1.0) * 100.0

    return {
        "status": "PASS",
        "load_case": load_case,
        "tip_disp": computed_disp,
        "target": target_disp,
        "ratio": ratio,
        "error_pct": err_pct,
        "iters": iters,
    }


# =========================================================================
# Suite 2: Cook's Membrane 3D
# =========================================================================

def run_cooks_membrane_test_3d(
    elem_type: str,
    nx: int = 8,
    ny: int = 8,
    nz: int = 1,
    thickness: float = 1.0
) -> Dict[str, Any]:
    """Execute Cook's Membrane 3D benchmark.
    
    Tapered clamped panel (left edge x=0 clamped, right edge x=48 loaded in shear).
    Total shear load P_y = 1.0 N applied on right edge.
    Linear elastic properties: E = 1.0, nu = 0.3333333333333333.
    """
    E = 1.0
    nu = 1.0 / 3.0

    mesh, root_nodes, tip_nodes = make_cooks_membrane_mesh_3d(
        elem_type=elem_type, thickness=thickness, nx=nx, ny=ny, nz=nz
    )
    nid_map = mesh.node_id_to_index()
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)

    # Clamped boundary condition at left edge (x = 0)
    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)

    # Distribute vertical shear load P = 1.0 across right edge tip nodes
    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    p_per_node = 1.0 / len(tip_nodes)
    for nid in tip_nodes:
        idx = nid_map[nid]
        f_ext[3 * idx + 1] = p_per_node

    converged, iters = solver.solve_step(dt=1.0, f_ext=f_ext, max_iters=50)
    if not converged:
        return {
            "status": "DIVERGED",
            "tip_uy": float("nan"),
            "iters": iters,
        }

    # Find the top-right corner node (max y among tip nodes)
    top_right_nid = max(tip_nodes, key=lambda n: mesh.nodes[n].y)
    top_right_uy = float(solver.u[3 * nid_map[top_right_nid] + 1])

    # Reference value for Cook's membrane with E=1, nu=1/3:
    # Highly refined reference solution tip deflection is ~23.91
    ref_val = 23.91
    ratio = top_right_uy / ref_val

    return {
        "status": "PASS",
        "tip_uy": top_right_uy,
        "ref_value": ref_val,
        "ratio": ratio,
        "iters": iters,
    }


# =========================================================================
# Suite 3: Cook's Membrane 2D
# =========================================================================

def run_cooks_membrane_test_2d(elem_type: str, nx: int = 8, ny: int = 8) -> Dict[str, Any]:
    """Execute Cook's Membrane 2D benchmark on plane-strain solid elements."""
    E = 1.0
    nu = 1.0 / 3.0

    mesh, root_nodes, tip_nodes = make_cooks_membrane_mesh_2d(elem_type, nx=nx, ny=ny)
    nid_map = mesh.node_id_to_index()
    solver = DynamicSolver2D(mesh, {"E": E, "nu": nu}, nlgeom=True)

    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)

    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    p_per_node = 1.0 / len(tip_nodes)
    for nid in tip_nodes:
        idx = nid_map[nid]
        f_ext[2 * idx + 1] = p_per_node

    converged, iters = solver.solve_step(dt=1.0, f_ext=f_ext, max_iters=50)
    if not converged:
        return {
            "status": "DIVERGED",
            "tip_uy": float("nan"),
            "iters": iters,
        }

    top_right_nid = max(tip_nodes, key=lambda n: mesh.nodes[n].y)
    top_right_uy = float(solver.u[2 * nid_map[top_right_nid] + 1])
    ref_val = 23.91
    ratio = top_right_uy / ref_val

    return {
        "status": "PASS",
        "tip_uy": top_right_uy,
        "ref_value": ref_val,
        "ratio": ratio,
        "iters": iters,
    }


# =========================================================================
# Main Runner & CLI
# =========================================================================

def run_all_macneal_harder_benchmarks() -> Dict[str, Any]:
    """Run all MacNeal-Harder benchmarks across all 2D and 3D solid elements."""
    print("=" * 80)
    print(" MacNeal-Harder (1985) Standard Finite Element Benchmark Suite")
    print("=" * 80)

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "twisted_beam_3d": {},
        "cooks_membrane_3d": {},
        "cooks_membrane_2d": {},
    }

    # 1. Twisted Beam 3D
    print("\n[*] Running 3D Twisted Beam Benchmark (11 Solid Elements)...")
    print("-" * 80)
    print(f"{'Element':<12} | {'Out-of-Plane (Target 0.005424)':<30} | {'In-Plane (Target 0.001754)':<30}")
    print("-" * 80)

    for elem in ALL_3D_ELEMENTS:
        t0 = time.time()
        res_oop = run_twisted_beam_test(elem, load_case="out_of_plane")
        res_ip = run_twisted_beam_test(elem, load_case="in_plane")
        elapsed = time.time() - t0

        results["twisted_beam_3d"][elem] = {
            "out_of_plane": res_oop,
            "in_plane": res_ip,
            "elapsed_s": elapsed,
        }

        oop_str = f"{res_oop['tip_disp']:.6f} ({res_oop['ratio']*100:.1f}%)" if res_oop['status'] == "PASS" else "DIVERGED"
        ip_str = f"{res_ip['tip_disp']:.6f} ({res_ip['ratio']*100:.1f}%)" if res_ip['status'] == "PASS" else "DIVERGED"
        print(f"{elem:<12} | {oop_str:<30} | {ip_str:<30} (took {elapsed:.2f}s)")

    # 2. Cook's Membrane 3D
    print("\n[*] Running 3D Cook's Membrane Benchmark (11 Solid Elements)...")
    print("-" * 80)
    print(f"{'Element':<12} | {'Tip Deflection (Ref ~23.91)':<30} | {'Ratio':<12} | {'Iters':<6}")
    print("-" * 80)

    for elem in ALL_3D_ELEMENTS:
        t0 = time.time()
        res_c3d = run_cooks_membrane_test_3d(elem)
        elapsed = time.time() - t0
        results["cooks_membrane_3d"][elem] = res_c3d

        if res_c3d["status"] == "PASS":
            c_str = f"{res_c3d['tip_uy']:.4f}"
            r_str = f"{res_c3d['ratio']*100:.1f}%"
            it_str = f"{res_c3d['iters']}"
        else:
            c_str, r_str, it_str = "DIVERGED", "N/A", "N/A"
        print(f"{elem:<12} | {c_str:<30} | {r_str:<12} | {it_str:<6}")

    # 3. Cook's Membrane 2D
    print("\n[*] Running 2D Cook's Membrane Benchmark (10 Solid Elements)...")
    print("-" * 80)
    print(f"{'Element':<12} | {'Tip Deflection (Ref ~23.91)':<30} | {'Ratio':<12} | {'Iters':<6}")
    print("-" * 80)

    for elem in ALL_2D_ELEMENTS:
        t0 = time.time()
        res_c2d = run_cooks_membrane_test_2d(elem)
        elapsed = time.time() - t0
        results["cooks_membrane_2d"][elem] = res_c2d

        if res_c2d["status"] == "PASS":
            c_str = f"{res_c2d['tip_uy']:.4f}"
            r_str = f"{res_c2d['ratio']*100:.1f}%"
            it_str = f"{res_c2d['iters']}"
        else:
            c_str, r_str, it_str = "DIVERGED", "N/A", "N/A"
        print(f"{elem:<12} | {c_str:<30} | {r_str:<12} | {it_str:<6}")

    # Save results to disk
    out_dir = Path("benchmark_element/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "macneal_harder_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results saved to {out_file}")

    return results


if __name__ == "__main__":
    run_all_macneal_harder_benchmarks()
