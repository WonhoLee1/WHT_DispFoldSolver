"""An increment is not accepted on the displacement ratio alone.

Regression for 2026-10-09 (dev_log/ex12_fold_wall_root_cause_20261009.md): the legacy
DynamicSolver accepted an iteration when |du| / |u_TOTAL| < tol, by itself. The ratio is
measured against the total accumulated displacement, so a model that has already moved a lot
accepts almost any correction -- ex12 increments "converged" with the residual rising.

Probe: Cook's membrane (Neo-Hookean, large deflection) solved once as is and once after a
rigid pre-translation. A rigid translation changes no stress, so the converged tip deflection
must be the same; with the old criterion the translated run stopped after the first correction
because |u_total| was inflated by the translation.
"""
import numpy as np

from dispsolver.material import NeoHookean
from dispsolver.solver import DynamicSolver

from tests.test_static import cooks_membrane_mesh

SHIFT = 1.0e3          # rigid pre-translation [mm]
TOL = 1.0e-2           # the displacement-ratio tolerance ex12 runs with


def _tip_deflection(shift: float) -> float:
    mesh = cooks_membrane_mesh(nx=4, ny=4)
    solver = DynamicSolver(mesh, NeoHookean(), rho=1e-6, material_params={"mu": 96.15, "lambda": 144.23},
                           max_iter=30, tol=TOL, static_mode=True)
    idx = mesh.node_id_to_index()
    left = sorted(mesh.get_nodeset("left").node_ids)
    bc_dofs, bc_vals = [], []
    for nid in left:
        bc_dofs += [2 * idx[nid], 2 * idx[nid] + 1]
        bc_vals += [shift, 0.0]
    solver.set_prescribed_dofs(np.array(bc_dofs), bc_vals=np.array(bc_vals))
    solver.u[0::2] = shift                       # the whole body starts rigidly translated
    right = sorted(mesh.get_nodeset("right").node_ids)
    solver.apply_load([2 * idx[n] + 1 for n in right], [100.0 / len(right)] * len(right))
    assert solver.solve_step(dt=1.0) > 0
    return float(solver.u[2 * idx[24] + 1])


def test_rigid_pre_translation_does_not_change_the_converged_answer():
    ref = _tip_deflection(0.0)
    moved = _tip_deflection(SHIFT)
    assert 5.0 < ref < 8.0
    assert abs(moved - ref) <= 1e-4 * abs(ref), (ref, moved)
