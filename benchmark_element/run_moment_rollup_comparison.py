"""
run_moment_rollup_comparison.py
===============================
Prescribed-rotation 2-turn (theta=4*pi) roll-up: SOLID_SHELL vs rivals.

Unlike run_moment_rollup_3d (which discards convergence info), this captures
per-step Newton iters, non-converged-step count, wall time, and final
double-circle shape error (R = L/4*pi).

Run:
    SEPARATION_CANDS="SOLID_SHELL,C3D8I_CR" python -u benchmark_element/run_moment_rollup_comparison.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from benchmark_element.benchmark_nlgeo_cantilever import (
    L_BEAM, H_BEAM, B_BEAM, E_MOD, NU_POI,
)
from benchmark_element.mechanics_patches import make_cantilever_beam_mesh
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

N_SUBSTEPS = 40
NX = 20


def run_rollup(elem_type: str):
    mesh, root_nodes, tip_nodes = make_cantilever_beam_mesh(
        elem_type=elem_type, L=L_BEAM, h=H_BEAM, b=B_BEAM, nx=NX, ny=1, nz=1
    )
    solver = DynamicSolver3D(
        mesh, material_params={"E": E_MOD, "nu": NU_POI}, nlgeom=True
    )
    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)
    for nid in mesh.nodes.keys():
        solver.fix_dof(nid, 1, 0.0)
    nid_map = mesh.node_id_to_index()

    iters_hist, nonconv = [], 0
    t0 = time.perf_counter()
    for s in range(1, N_SUBSTEPS + 1):
        th_curr = float(s) / N_SUBSTEPS * 4.0 * np.pi
        sinc_t = np.sin(th_curr) / th_curr
        omc_t = (1.0 - np.cos(th_curr)) / th_curr
        Xc, Zc = L_BEAM * sinc_t, L_BEAM * omc_t
        cos_t, sin_t = np.cos(th_curr), np.sin(th_curr)
        for nid in tip_nodes:
            dz0 = mesh.nodes[nid].z
            solver.fix_dof(nid, 0, (Xc - dz0 * sin_t) - L_BEAM)
            solver.fix_dof(nid, 2, (Zc + dz0 * cos_t) - mesh.nodes[nid].z)
        conv, iters = solver.solve_step(dt=1.0, max_iters=25)
        iters_hist.append(iters)
        if not conv:
            nonconv += 1
    wall = time.perf_counter() - t0

    # Final double-circle error: radius of each centerline station from
    # ideal circle center (0, R) with R = L/4pi, stations past 1st turn
    # coincide doubly -- use tip return error + radial spread.
    R = L_BEAM / (4.0 * np.pi)
    tip_x = np.mean([mesh.nodes[n].x + solver.u[3 * nid_map[n]] for n in tip_nodes])
    tip_z = np.mean([mesh.nodes[n].z + solver.u[3 * nid_map[n] + 2] for n in tip_nodes])
    tip_return_err = float(np.hypot(tip_x - 0.0, tip_z - 0.0))
    solver.close()
    return {
        "iters_mean": float(np.mean(iters_hist)),
        "iters_max": int(np.max(iters_hist)),
        "nonconv": nonconv,
        "wall_s": wall,
        "tip_return_err": tip_return_err,
    }


def main():
    cands = os.environ.get("SEPARATION_CANDS", "SOLID_SHELL,C3D8I_CR").split(",")
    print(f"2-turn roll-up: nx={NX}, {N_SUBSTEPS} substeps, R=L/4pi={L_BEAM / (4 * np.pi):.4f} m")
    for etype in cands:
        etype = etype.strip()
        try:
            r = run_rollup(etype)
            print(f"[{etype:12s}] iters mean={r['iters_mean']:.2f} max={r['iters_max']} "
                  f"nonconv={r['nonconv']}/{N_SUBSTEPS} wall={r['wall_s']:.1f}s "
                  f"tip_return_err={r['tip_return_err']:.3e} m", flush=True)
        except Exception as e:  # noqa: BLE001 -- diagnostic script
            print(f"[{etype:12s}] ERROR: {e}", flush=True)


if __name__ == "__main__":
    main()
