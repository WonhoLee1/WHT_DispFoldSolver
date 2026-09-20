"""Probe4: linear bending stiffness at flat state, CPE6M-mesh vs Opt1-mesh. 1 deg tip rotation."""
import sys, os
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from benchmark_element.benchmark_pure_moment_rollup import (
    make_2d_mesh, L_BEAM, T_PET, T_PSA, PID_PET, PID_PSA, E_PET, NU_PET, E_PSA, NU_PSA,
    Z_MID, compute_roll_up_displacements,
)
from dispsolver.mesh2d.conformal_cpe6m_cpe4h_builder import create_5layer_cpe6m_cpe4h_conformal_mesh
from dispsolver.solver2d.dynamic2d import DynamicSolver2D

TH = np.radians(1.0)
for case in ["Opt1", "CPE6M"]:
    if case == "CPE6M":
        mesh, sets = create_5layer_cpe6m_cpe4h_conformal_mesh(
            length=L_BEAM, t_pet=T_PET, t_psa=T_PSA, nx=40, pid_pet=PID_PET, pid_psa=PID_PSA)
        root_nodes, tip_nodes = sets["LEFT_NODES"], sets["RIGHT_NODES"]
    else:
        mesh, root_nodes, tip_nodes, _ = make_2d_mesh("CPE4I", "CPE4H")
    materials = {PID_PET: {"E": E_PET, "nu": NU_PET}, PID_PSA: {"E": E_PSA, "nu": NU_PSA}}
    solver = DynamicSolver2D(mesh, materials=materials, nlgeom=True)
    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
    nid_map = mesh.node_id_to_index()
    for nid in tip_nodes:
        node = mesh.nodes[nid]
        ux_t, uy_t = compute_roll_up_displacements(node.y, TH)
        solver.fix_dof(nid, 0, ux_t)
        solver.fix_dof(nid, 1, uy_t)
    conv, it = solver.solve_step(dt=1.0, max_iters=35, tol=1e-5)
    K_g, f = solver.assemble_system(solver.u, dt=1.0)
    free = np.ones(solver.num_dofs, bool)
    free[list(solver.fixed_dofs.keys())] = False
    r = float(np.linalg.norm(np.asarray(f)[free]))
    rx = solver.compute_section_reactions(root_nodes, center_pt=(0.0, Z_MID))
    print(f"{case}: conv={conv} it={it} r_free={r:.3e} M_root={rx['Mz']:.8f} M/theta={rx['Mz'] / TH:.6f}", flush=True)
