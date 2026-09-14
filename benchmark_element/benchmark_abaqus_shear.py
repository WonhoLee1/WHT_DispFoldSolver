"""
benchmark_abaqus_shear.py
=========================
Abaqus 2024 Official Verification Benchmark: simaver-c-shear.htm
Reference input files:
  - shear_cpe4r.inp (2D CPE4R, up to 300% nominal shear)
  - shear_c3d8r.inp (3D C3D8R, up to 300% nominal shear)

Dienes (1979) Analytical Reference:
  J. K. Dienes (1979), "On the analysis of rotation and stress rate in deforming bodies",
  Acta Mechanica, 32, 217-232.
  Formula for Green-Naghdi / Co-rotational Hypoelasticity:
    tan(2*beta) = gamma  ==>  beta = 0.5 * arctan(gamma)
    S11 = 4 * mu * [cos(2*beta)*ln(cos(beta)) + beta*sin(2*beta) - sin^2(beta)]
    S12 = 2 * mu * cos(2*beta) * [2*beta - 2*tan(2*beta)*ln(cos(beta)) - tan(beta)]
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def dienes_analytical(gamma: float, mu: float = 0.5) -> Tuple[float, float]:
    """Dienes (1979) analytical shear stress S12 and normal stress S11."""
    beta = 0.5 * np.arctan(gamma)
    cb = np.cos(beta)
    c2b = np.cos(2.0 * beta)
    s2b = np.sin(2.0 * beta)
    t2b = np.tan(2.0 * beta)
    tb = np.tan(beta)

    s11 = 4.0 * mu * (c2b * np.log(cb) + beta * s2b - np.sin(beta)**2)
    s12 = 2.0 * mu * c2b * (2.0 * beta - 2.0 * t2b * np.log(cb) - tb)
    return float(s11), float(s12)


def run_simple_shear_2d(elem_type: str, n_steps: int = 30) -> Dict[str, Any]:
    """Run 2D Simple Shear up to gamma = 3.0 (300%)."""
    E = 1.0
    nu = 0.0
    mu = E / (2.0 * (1.0 + nu)) # 0.5

    mesh = Mesh2D()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(3, 1.0, 1.0)
    mesh.add_node(4, 0.0, 1.0)
    mesh.add_element(1, [1, 2, 3, 4], elem_type=elem_type)

    nid_map = mesh.node_id_to_index()
    solver = DynamicSolver2D(mesh, {"E": E, "nu": nu}, nlgeom=True)

    # Bottom edge fixed
    solver.fix_dof(1, 0, 0.0)
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(2, 0, 0.0)
    solver.fix_dof(2, 1, 0.0)
    # Top edge y fixed
    solver.fix_dof(3, 1, 0.0)
    solver.fix_dof(4, 1, 0.0)

    gammas = np.linspace(0.1, 3.0, n_steps)
    history_gamma = []
    history_s12_fe = []
    history_s12_dienes = []
    history_s11_dienes = []
    history_iters = []

    diverged = False
    for g in gammas:
        solver.fix_dof(3, 0, g)
        solver.fix_dof(4, 0, g)
        conv, iters = solver.solve_step(dt=1.0, max_iters=25)
        if not conv:
            diverged = True
            break

        R = solver.compute_reaction_forces()
        idx3 = nid_map[3]
        idx4 = nid_map[4]
        tau_fe = float(R[2 * idx3] + R[2 * idx4]) # Top shear reaction force (Area = 1.0)
        s11_d, s12_d = dienes_analytical(g, mu=mu)

        history_gamma.append(float(g))
        history_s12_fe.append(tau_fe)
        history_s12_dienes.append(s12_d)
        history_s11_dienes.append(s11_d)
        history_iters.append(int(iters))

    return {
        "status": "PASS" if not diverged else "DIVERGED",
        "elem_type": elem_type,
        "max_gamma_reached": history_gamma[-1] if history_gamma else 0.0,
        "gamma": history_gamma,
        "s12_fe": history_s12_fe,
        "s12_dienes": history_s12_dienes,
        "s11_dienes": history_s11_dienes,
        "iters": history_iters,
    }


def run_simple_shear_3d(elem_type: str, n_steps: int = 30) -> Dict[str, Any]:
    """Run 3D Simple Shear up to gamma = 3.0 (300%)."""
    E = 1.0
    nu = 0.0
    mu = E / (2.0 * (1.0 + nu)) # 0.5

    mesh = Mesh3D()
    # 1x1x1 cube
    coords = [
        (1, 0.0, 0.0, 0.0),
        (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0),
        (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0),
        (6, 1.0, 0.0, 1.0),
        (7, 1.0, 1.0, 1.0),
        (8, 0.0, 1.0, 1.0),
    ]
    for nid, x, y, z in coords:
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], elem_type=elem_type)

    nid_map = mesh.node_id_to_index()
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)

    # Fix bottom face (nodes 1, 2, 5, 6: y=0)
    for n in [1, 2, 5, 6]:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fix_dof(n, 2, 0.0)

    # Top face (nodes 3, 4, 7, 8: y=1): constrain y and z
    for n in [3, 4, 7, 8]:
        solver.fix_dof(n, 1, 0.0)
        solver.fix_dof(n, 2, 0.0)

    gammas = np.linspace(0.1, 3.0, n_steps)
    history_gamma = []
    history_s12_fe = []
    history_s12_dienes = []
    history_iters = []

    diverged = False
    for g in gammas:
        for n in [3, 4, 7, 8]:
            solver.fix_dof(n, 0, g)
        conv, iters = solver.solve_step(dt=1.0, max_iters=25)
        if not conv:
            diverged = True
            break

        R = solver.compute_reaction_forces()
        tau_fe = float(sum(R[3 * nid_map[n] + 0] for n in [3, 4, 7, 8])) # Top shear force (Area = 1.0)
        _, s12_d = dienes_analytical(g, mu=mu)

        history_gamma.append(float(g))
        history_s12_fe.append(tau_fe)
        history_s12_dienes.append(s12_d)
        history_iters.append(int(iters))

    return {
        "status": "PASS" if not diverged else "DIVERGED",
        "elem_type": elem_type,
        "max_gamma_reached": history_gamma[-1] if history_gamma else 0.0,
        "gamma": history_gamma,
        "s12_fe": history_s12_fe,
        "s12_dienes": history_s12_dienes,
        "iters": history_iters,
    }


def main():
    print("==========================================================================")
    print("Abaqus 2024 Official Verification Benchmark: simaver-c-shear (300% Shear)")
    print("==========================================================================")

    results_2d = {}
    elements_2d = ["CPE4_CR", "CPE4I_CR", "CPE4R_CR", "CPE4H_CR"]
    for et in elements_2d:
        res = run_simple_shear_2d(et)
        results_2d[et] = res
        print(f"2D [{et:<10}]: status={res['status']}, max_gamma={res['max_gamma_reached']:.1f} (300%), avg_iters={np.mean(res['iters']):.1f}")

    results_3d = {}
    elements_3d = ["C3D8_CR", "C3D8I_CR", "C3D8R_CR", "C3D8H_CR"]
    for et in elements_3d:
        res = run_simple_shear_3d(et)
        results_3d[et] = res
        print(f"3D [{et:<10}]: status={res['status']}, max_gamma={res['max_gamma_reached']:.1f} (300%), avg_iters={np.mean(res['iters']):.1f}")

    out_file = RESULTS_DIR / "abaqus_shear_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"2d": results_2d, "3d": results_3d}, f, indent=2)
    print(f"\n[Saved JSON]: {out_file}")


if __name__ == "__main__":
    main()
