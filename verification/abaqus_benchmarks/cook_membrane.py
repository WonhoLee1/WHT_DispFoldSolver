"""
bmk_cook_membrane.py
=====================
Abaqus official benchmark: Cook's membrane (SIMACAEBMKRefMap/simabmk-c-cookmembrane).

Fetched directly this session from the local Abaqus doc mirror
(http://desktop-whl:4040), plus the actual `cook_2d.inp` deck (saved at
verification/abaqus_benchmarks/reference/cook_2d.inp) and the digitized convergence figure
(verification/abaqus_benchmarks/reference/cook2d_convergence.png) -- NOT from memory, and
NOT reusing the pre-existing (unaudited, other-agent) 3D benchmark suite's
reference number (23.96 mm), which does not match this figure and is
believed wrong for this exact 1.0N/C10=0.4/D1=2.5e-4 case.

Geometry (from cook_2d.inp node coords, confirmed bilinear):
  x(xi,eta) = 48*xi
  y_bot(xi) = 44*xi,  y_top(xi) = 44 + 16*xi
  y(xi,eta) = y_bot + eta*(y_top - y_bot)
  xi, eta in [0,1], xi=0 is the fixed (left) edge, xi=1 the loaded (right) edge.

Material: Abaqus *Hyperelastic, neo hooke: C10=0.4, D1=2.5e-4.
  Abaqus neo-Hookean: U = C10*(Ibar1-3) + (1/D1)*(J-1)^2  =>  mu = 2*C10 = 0.8.
  Our q4_visco_simo_fs_jax "neohookean" base uses the SAME isochoric term
  (mu/2)*(Ibar1-3) but a different (Simo-Taylor) volumetric potential --
  both linearize to the same bulk modulus, so kappa = 2/D1 = 8000 is the
  correct correspondence for the near-incompressible limit being tested
  here; the two volumetric PENALTY FUNCTIONS are not bit-identical, so an
  exact digit match to Abaqus's own curve is not expected -- see dev_log
  writeup for this caveat.

Load: 1.0 N concentrated force in +Y via a UNIFORM distributing coupling
  on the right (xi=1) edge -- per Abaqus docs, UNIFORM weighting splits
  the load evenly across the coupled nodes, so this is applied directly
  as f_ext[dof_y] = 1.0/(n+1) per right-edge node (n+1 nodes on that edge).

Digitized reference (verification/abaqus_benchmarks/reference/cook2d_convergence.png),
vertical displacement U2 of the top-right corner node vs elements/side:

    n:        4      8      16     32
    CPE3H:   3.35   3.55   3.75   4.10   (poor -- doc: "particularly stiff")
    CPE4H:   6.15   6.70   6.85   6.90   (full integration, locks at coarse mesh)
    CPE4IH:  6.55   6.75   6.85   6.90   (incompatible modes + hybrid pressure)
    CPE4RH:  6.65   6.80   6.87   6.90   (reduced integration + hybrid pressure)
    CPE6H/CPE6MH: ~6.85-6.95 at every n (converge fastest)

Our two comparable element types:
    "Q4_VISCO_SIMO"  -- full 2x2 Gauss + F-bar (centroid J0 dilatation).
                        Closest analog to CPE4H (full integration, no
                        incompatible modes) though the volumetric
                        treatment differs from CPE4H's hybrid pressure.
    "CPE4I"          -- 4-mode EAS incompatible modes (this session's
                        redesigned element: F6 transpose fix, F4 push-
                        forward fix, no more magnitude clamp -- line-
                        search damped local Newton instead).  Closest
                        analog to CPE4IH, though ours carries NO hybrid
                        pressure DOF -- incompressibility is handled by
                        the EAS modes alone (see q4_visco_eas_jax.py's own
                        docstring). This benchmark is a real stress test
                        of that "EAS alone is enough" claim at a
                        genuinely high K/mu ratio (8000/0.8 = 10000).
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dispsolver.mesh.mesh import Mesh
from dispsolver.material.neohookean import NeoHookean
from dispsolver.material.viscoelastic import ViscoelasticMaterial
from dispsolver.solver.dynamic import DynamicSolver
from dispsolver.solver.dt_controller import AdaptiveDtController

MU = 0.8        # 2*C10
KAPPA = 8000.0  # 2/D1


def build_cook_mesh(n: int, elem_type: str):
    mesh = Mesh()
    grid = {}
    nid = 1
    for j in range(n + 1):
        eta = j / n
        for i in range(n + 1):
            xi = i / n
            x = 48.0 * xi
            y_bot = 44.0 * xi
            y_top = 44.0 + 16.0 * xi
            y = y_bot + eta * (y_top - y_bot)
            mesh.add_node(nid, x, y)
            grid[(i, j)] = nid
            nid += 1
    eid = 1
    for j in range(n):
        for i in range(n):
            n1, n2 = grid[(i, j)], grid[(i + 1, j)]
            n3, n4 = grid[(i + 1, j + 1)], grid[(i, j + 1)]
            mesh.add_element(eid, [n1, n2, n3, n4], elem_type, pid=0)
            eid += 1
    left_nodes = [grid[(0, j)] for j in range(n + 1)]
    right_nodes = [grid[(n, j)] for j in range(n + 1)]
    corner_node = grid[(n, n)]
    return mesh, left_nodes, right_nodes, corner_node


def run_one(n: int, elem_type: str, num_steps: int = 30, verbose_fail=True, elem_jit: str = "jax"):
    mesh, left_nodes, right_nodes, corner_node = build_cook_mesh(n, elem_type)

    base_material = NeoHookean()
    material = ViscoelasticMaterial(base_material, g_i=[], tau_i=[])
    material_params = {"mu": MU, "K": KAPPA}

    # Dict form is required: the multi-material batch assembler (which is
    # the only path that knows how to dispatch CPE4I/Q4_VISCO_SIMO against
    # a ViscoelasticMaterial) only activates when `element_type_by_pid`
    # is not None -- see dynamic.py's constructor, `elif
    # (self.element_type_by_pid is not None or self.element_type in (...))`.
    # A single-pid model must still go through the dict form for this.
    solver = DynamicSolver(
        mesh=mesh,
        material={0: material},
        material_params={0: material_params},
        rho=1.0,
        element_type={0: elem_type},
        mode="quasistatic",
        nlgeom=True,
        elem_jit=elem_jit,
        tol=1e-5,
        max_iter=30,
        verbose=False,
    )
    solver.sta_status = False

    bc_dofs = []
    bc_vals = []
    for nid in left_nodes:
        idx = solver.nid_to_idx[nid]
        bc_dofs += [2 * idx, 2 * idx + 1]
        bc_vals += [0.0, 0.0]
    solver.set_prescribed_dofs(np.array(bc_dofs, dtype=np.int32), np.array(bc_vals, dtype=np.float64))

    base_f_ext = np.zeros(solver.n_dofs, dtype=np.float64)
    w = 1.0 / len(right_nodes)
    for nid in right_nodes:
        idx = solver.nid_to_idx[nid]
        base_f_ext[2 * idx + 1] = w

    corner_idx = solver.nid_to_idx[corner_node]

    dt_ctrl = AdaptiveDtController(dt_init=1.0 / num_steps, dt_min=1e-6, dt_max=1.0 / num_steps, target_iters=6)
    t = 0.0
    cutbacks = 0
    max_total_cutbacks = 60
    while t < 1.0 - 1e-9:
        dt = min(dt_ctrl.dt, 1.0 - t)
        solver.f_ext = base_f_ext * (t + dt)
        # Checkpoint/restore rollback (dev_log/session_20260730_reduced_
        # integration_failure.md): a failed Newton attempt leaves
        # solver.eas_alpha (and _ul_F_n) corrupted at whatever bad value
        # the last rejected iteration wrote -- without restoring, the
        # RETRY at a smaller dt still warm-starts alpha from that garbage,
        # which can make the local EAS Newton fail unpredictably even at
        # arbitrarily small load. Missing this was a real bug in this
        # script, not the element kernel.
        checkpoint = solver.save_state()
        n_iter = solver.solve_step(dt)
        if n_iter is not None and n_iter >= 0:
            t += dt
            dt_ctrl.dt = dt_ctrl.update(n_iter, True)
        else:
            solver.restore_state(checkpoint)
            cutbacks += 1
            dt_ctrl.dt = dt_ctrl.update(abs(n_iter) if n_iter is not None else 30, False)
            if cutbacks > max_total_cutbacks or dt_ctrl.dt < dt_ctrl.dt_min * 1.01:
                if verbose_fail:
                    print(f"    [FAIL] n={n} {elem_type}: stalled at t={t:.4f}, cutbacks={cutbacks}")
                return None, t, cutbacks
    u2 = float(solver.u[2 * corner_idx + 1])
    return u2, 1.0, cutbacks


REF = {
    4:  {"CPE4H": 6.15, "CPE4IH": 6.55},
    8:  {"CPE4H": 6.70, "CPE4IH": 6.75},
    16: {"CPE4H": 6.85, "CPE4IH": 6.85},
    32: {"CPE4H": 6.90, "CPE4IH": 6.90},
}

if __name__ == "__main__":
    print("=== Cook's membrane benchmark (Abaqus official, C10=0.4, D1=2.5e-4, 1.0N) ===")
    print(f"{'n':>4} {'elem':>14} {'backend':>8} {'U2_top_right':>14} {'ref(H/IH)':>12} {'err%':>7} {'t_reached':>10} {'cutbacks':>9}")
    for n in (4, 8, 16, 32):
        for elem_type, ref_key in (("Q4_VISCO_SIMO", "CPE4H"), ("CPE4I", "CPE4IH")):
            for backend in ("jax", "numba"):
                u2, t_reached, cb = run_one(n, elem_type, elem_jit=backend)
                ref = REF[n][ref_key]
                u2s = f"{u2:.4f}" if u2 is not None else "FAILED"
                errs = f"{(u2/ref-1)*100:+.2f}" if u2 is not None else "-"
                print(f"{n:>4} {elem_type:>14} {backend:>8} {u2s:>14} {ref:>12.2f} {errs:>7} {t_reached:>10.4f} {cb:>9}", flush=True)
