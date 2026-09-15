"""
run_distorted_nlgeo_separation.py
=================================
Distorted-mesh NLGEOM cantilever separation test: C3D8I_CR vs SOLID_SHELL vs C3D8_CR.

Background: on a perfect brick mesh SOLID_SHELL_CR measures bit-identical
(1e-13) to C3D8I_CR. That is EXPECTED (MITC tying exact on undistorted bricks
+ same EAS-9), not a dispatch bug -- but it means the undistorted cantilever
cannot separate the two kernels. This script applies a deterministic
trapezoidal distortion (zero at root/tip faces, max mid-span, so BC/load
meaning is preserved) and checks:
  1. separation: |uy(C3D8I_CR) - uy(SOLID_SHELL)| >> 1e-13 on distorted mesh
  2. robustness: distorted-vs-clean drift per element vs Bisshopp-Drucker theory

Run:
    python -u benchmark_element/run_distorted_nlgeo_separation.py
"""

import os
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from benchmark_element.benchmark_nlgeo_cantilever import (
    BisshoppDruckerElastica,
    E_MOD, I_SECTION, L_BEAM, H_BEAM, B_BEAM, P_TIP, NU_POI,
)
from benchmark_element.mechanics_patches import make_cantilever_beam_mesh
from dispsolver.solver3d.dynamic3d import DynamicSolver3D

DISTORT_AMP = 0.15 * H_BEAM  # 15% of section height, mid-span peak


def apply_trapezoid_distortion(mesh) -> float:
    """Zigzag-free trapezoidal distortion: opposite z-shift per thickness
    layer with a sin(pi*x/L) envelope (zero at root x=0 and tip x=L).

    Returns max nodal shift applied.
    """
    max_shift = 0.0
    for nid, node in mesh.nodes.items():
        if node.x <= 0.0 or node.x >= L_BEAM:
            continue
        s = np.sign(node.z) if abs(node.z) > 1e-15 else 0.0
        dz = DISTORT_AMP * np.sin(np.pi * node.x / L_BEAM) * s
        node.z += dz
        max_shift = max(max_shift, abs(dz))
    return max_shift


def solve_cantilever(mesh, elem_label: str):
    """Same solve loop as run_3d_cantilever_transverse (15 substeps, ramp)."""
    solver = DynamicSolver3D(
        mesh, material_params={"E": E_MOD, "nu": NU_POI}, nlgeom=True
    )
    root_nodes = [nid for nid, n in mesh.nodes.items() if n.x == 0.0]
    tip_nodes = [nid for nid, n in mesh.nodes.items() if n.x == L_BEAM]
    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)
    for nid in mesh.nodes.keys():
        solver.fix_dof(nid, 1, 0.0)

    n_substeps = 15
    f_ext = np.zeros(solver.num_dofs, dtype=np.float64)
    tip_p_per_node = P_TIP / float(len(tip_nodes))
    nid_map = mesh.node_id_to_index()
    cutbacks = 0
    for s in range(1, n_substeps + 1):
        frac = float(s) / float(n_substeps)
        ramp = frac * frac * (3.0 - 2.0 * frac)
        for nid in tip_nodes:
            idx = nid_map[nid]
            f_ext[3 * idx + 2] = tip_p_per_node * ramp
        conv, _ = solver.solve_step(dt=1.0, f_ext=f_ext, max_iters=25)
        if not conv:
            cutbacks += 1
    tip_uz = float(np.mean([solver.u[3 * nid_map[n] + 2] for n in tip_nodes]))
    solver.close()
    return tip_uz, cutbacks


def main():
    elastica = BisshoppDruckerElastica(E_MOD, I_SECTION, L_BEAM, P_TIP)
    _, _, uy_exact = elastica.solve_at_load(P_TIP)
    print(f"Theory uy = {uy_exact:.6f} m")

    candidates = ["C3D8I_CR", "SOLID_SHELL", "C3D8_CR"]
    rows = {}
    amp_scales = [float(s) for s in os.environ.get("SEPARATION_AMPS", "1.0").split(",")]
    for amp in amp_scales:
        globals()["DISTORT_AMP"] = amp * 0.15 * H_BEAM
        for distorted in (False, True):
            tag = f"x{amp:g}-" + ("DISTORTED" if distorted else "CLEAN    ")
        for etype in candidates:
            mesh, _, _ = make_cantilever_beam_mesh(
                elem_type=etype, L=L_BEAM, h=H_BEAM, b=B_BEAM,
                nx=10, ny=1, nz=1,
            )
            shift = apply_trapezoid_distortion(mesh) if distorted else 0.0
            try:
                uy, cb = solve_cantilever(mesh, etype)
                err = abs(uy - uy_exact) / abs(uy_exact) * 100.0
                rows[(etype, distorted)] = (uy, err, cb)
                print(f"[{tag}] {etype:12s} uy={uy:10.6f} err={err:7.3f}% "
                      f"cutbacks={cb} maxshift={shift:.5f}", flush=True)
            except Exception as e:  # noqa: BLE001 -- diagnostic script
                rows[(etype, distorted)] = None
                print(f"[{tag}] {etype:12s} ERROR: {e}", flush=True)

    a = rows.get(("C3D8I_CR", True))
    b = rows.get(("SOLID_SHELL", True))
    if a is not None and b is not None:
        sep = abs(a[0] - b[0])
        print(f"\nSeparation |uy(C3D8I_CR)-uy(SOLID_SHELL)| distorted = {sep:.3e} m")
        print("SEPARATED" if sep > 1e-9 else "STILL IDENTICAL -- dispatch suspect")


if __name__ == "__main__":
    main()
