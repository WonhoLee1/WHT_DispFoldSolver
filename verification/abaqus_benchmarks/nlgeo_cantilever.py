"""
bmk_nlgeo_cantilever.py
=========================
Abaqus official benchmark: Geometrically nonlinear cantilever, transverse
tip load case (SIMACAEBMKRefMap/simabmk-c-nlgeocantilever), CPS4I element
-- the plane-STRESS incompatible-mode sibling of our plane-STRAIN CPE4I/
Q4_EAS. Real input decks fetched this session and saved at
verification/abaqus_benchmarks/reference/nlgeocantilever_{10,20}cps4i_tload.inp.

Also stands in for `simabmk-c-cantilevercaxasaxa` (per user instruction:
where the official benchmark needs a 3D-only element family -- CAXA/SAXA,
axisymmetric-with-Fourier, no plane-strain analog exists at all -- use the
plane-strain-comparable example instead). That benchmark tests the SAME
physical setup (cantilever, tip transverse load, compared to the SAME
Bisshopp & Drucker (1945) exact elastica solution, there via B22 beam
elements) with a different element family (CAXA/SAXA pipe elements); the
CPS4I case here already covers that comparison for a 2D solver.

Why nu=0 makes plane strain a valid stand-in for the real CPS4I (plane
stress) benchmark: cross-section properties only enter the elastica
solution through EI (no Poisson coupling for pure bending of a
rectangular section), and plane-strain vs plane-stress differ only by
the factor E/(1-nu^2) vs E -- identical when nu=0, which this exact
benchmark uses (*ELASTIC, 1.E8, <blank> -> nu defaults to 0 in Abaqus).
So CPE-family (plane strain) and CPS-family (plane stress) predictions
are mathematically identical for this specific benchmark.

Geometry (from the .inp): L=10, cross-section height h=0.1478,
thickness t=0.1 (out-of-plane, matches the doc's "100mm x 147.8mm" in
different units), E=1e8, nu=0. Fixed at one end (BOUNDARY 21,1,2 /
22,1,2 in the .inp -- flipped here to fix at x=0 for convention). Tip
load P=269.35 N in +Y, distributed 50/50 (weight 1,1 in the .inp's
*COUPLING/*DISTRIBUTING) onto the two tip corner nodes -- exact for this
geometry since the coupling reference node sits exactly on the section
mid-height, symmetric between the two corner nodes, so no net moment is
introduced by the equal-weight approximation.

Meshes: 1x10 (coarse, matches nlgeocantilever_10cps4i_tload.inp) and
1x20 (fine, matches nlgeocantilever_20cps4i_tload.inp).

No hyperelastic material available/needed here -- the real element uses
*ELASTIC (linear, small-strain constitutive) + NLGEOM (large-rotation
kinematics), i.e. classical elastica behavior. This codebase's finite-
strain J2Plasticity is used as a linear-elastic surrogate by setting
sigma_y0 absurdly high (never yields) -- same E/nu, zero plastic
correction, and its multiplicative-Hencky-strain kinematics reduce to
St-Venant-Kirchhoff-like large-rotation/small-strain behavior, which is
exactly the elastica assumption.

Element comparison (mirrors the doc's own CPS4 vs CPS4I finding, and
reproduces this session's own AGENTS.md sec 4.1 shear-locking finding
under an INDEPENDENT loading case):
    "Q4_COROTATIONAL" + J2(no yield) ~ CPS4  (full integration, no EAS
        -- expected stiffer than exact, shear-locked in bending)
    "Q4_EAS"          + J2(no yield) ~ CPS4I (EAS-4 incompatible modes,
        this session's F6/line-search-fixed element -- expected to track
        the exact elastica closely at both coarse and fine mesh)
"""
from __future__ import annotations

import os
import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dispsolver.mesh.mesh import Mesh
from dispsolver.material.plastic import J2Plasticity
from dispsolver.solver.dynamic import DynamicSolver
from dispsolver.solver.dt_controller import AdaptiveDtController

E = 1.0e8
NU = 0.0
L = 10.0
H = 0.1478
THICK = 0.1
P = 269.35


class BisshoppDruckerElastica:
    """Exact large-deflection cantilever tip-load elastica (1945).

    EI*theta'' = -P*cos(theta), theta(0)=0, theta'(L)=0 (moment-free tip).
    First integral: (EI/2)*theta'^2 = P*(sin(theta_tip) - sin(theta)).
    """

    def __init__(self, E: float, I: float, L: float, P: float):
        self.EI = E * I
        self.L = L
        self.P = P

    # Substitution th = theta_tip - s^2 (s in [0, sqrt(theta_tip)]) removes
    # the integrable inverse-sqrt singularity at th=theta_tip entirely: near
    # s=0, sin(theta_tip)-sin(th) ~ cos(theta_tip)*s^2, so the s-integrand
    # tends to a finite limit (2/sqrt(cos(theta_tip))) instead of blowing up
    # -- `quad` converges cleanly with no roundoff warning this way.
    @staticmethod
    def _s_integrand(s, theta_tip, weight):
        th = theta_tip - s * s
        d = np.sin(theta_tip) - np.sin(th)
        d = max(d, 1e-30)
        return weight(th) * 2.0 * s / np.sqrt(d)

    def _residual(self, theta_tip: float) -> float:
        beta = np.sqrt(2.0 * self.P / self.EI)
        smax = np.sqrt(theta_tip)
        val, _ = quad(self._s_integrand, 0.0, smax, args=(theta_tip, lambda th: 1.0), limit=200)
        return beta * self.L - val

    def solve(self):
        # theta_tip in (0, pi/2) for this load range; bracket and solve.
        lo, hi = 1e-6, np.pi / 2 - 1e-6
        theta_tip = brentq(self._residual, lo, hi, xtol=1e-13, rtol=1e-13)
        beta = np.sqrt(2.0 * self.P / self.EI)
        smax = np.sqrt(theta_tip)
        x_val, _ = quad(self._s_integrand, 0.0, smax, args=(theta_tip, np.cos), limit=200)
        y_val, _ = quad(self._s_integrand, 0.0, smax, args=(theta_tip, np.sin), limit=200)
        x_tip = x_val / beta
        y_tip = y_val / beta
        return theta_tip, x_tip, y_tip


def build_cantilever_mesh(nx: int, elem_type: str):
    mesh = Mesh()
    grid = {}
    nid = 1
    for j in range(2):  # ny=1 element through height
        y = j * H
        for i in range(nx + 1):
            x = L * i / nx
            mesh.add_node(nid, x, y)
            grid[(i, j)] = nid
            nid += 1
    eid = 1
    for i in range(nx):
        n1, n2 = grid[(i, 0)], grid[(i + 1, 0)]
        n3, n4 = grid[(i + 1, 1)], grid[(i, 1)]
        mesh.add_element(eid, [n1, n2, n3, n4], elem_type, pid=0)
        eid += 1
    fixed_nodes = [grid[(0, 0)], grid[(0, 1)]]
    tip_nodes = [grid[(nx, 0)], grid[(nx, 1)]]
    return mesh, fixed_nodes, tip_nodes


def run_one(nx: int, elem_type: str, num_steps: int = 40, elem_jit: str = "jax"):
    mesh, fixed_nodes, tip_nodes = build_cantilever_mesh(nx, elem_type)
    # sigma_y0 must stay "never yields" but not absurdly disproportionate to
    # E -- 1e14 (1e6x E) triggered a Newton/line-search collapse on the very
    # first (tiny) increment, likely a numerical-scaling issue in the smooth
    # tanh-blended yield surface (AGENTS.md sec 2) at that extreme ratio.
    # Expected peak bending stress here is ~7.4e6 (M_max*c/I at the base);
    # 1e9 is ~135x that margin while staying within a few orders of E.
    material = J2Plasticity(E=E, nu=NU, sigma_y0=1e9, H=0.0)

    solver = DynamicSolver(
        mesh=mesh,
        material={0: material},
        element_type={0: elem_type},
        rho=1.0,
        mode="quasistatic",
        nlgeom=True,
        elem_jit=elem_jit,
        section_thickness=THICK,
        tol=1e-5,
        max_iter=30,
    )
    solver.sta_status = False

    bc_dofs, bc_vals = [], []
    for nid in fixed_nodes:
        idx = solver.nid_to_idx[nid]
        bc_dofs += [2 * idx, 2 * idx + 1]
        bc_vals += [0.0, 0.0]
    solver.set_prescribed_dofs(np.array(bc_dofs, dtype=np.int32), np.array(bc_vals, dtype=np.float64))

    base_f_ext = np.zeros(solver.n_dofs, dtype=np.float64)
    for nid in tip_nodes:
        idx = solver.nid_to_idx[nid]
        base_f_ext[2 * idx + 1] = P / len(tip_nodes)

    tip_idx = [solver.nid_to_idx[nid] for nid in tip_nodes]

    dt_ctrl = AdaptiveDtController(dt_init=1.0 / num_steps, dt_min=1e-7, dt_max=1.0 / num_steps, target_iters=6)
    t = 0.0
    cutbacks = 0
    while t < 1.0 - 1e-9:
        dt = min(dt_ctrl.dt, 1.0 - t)
        solver.f_ext = base_f_ext * (t + dt)
        # Checkpoint/restore rollback (dev_log/session_20260730_reduced_
        # integration_failure.md pattern): a failed attempt leaves
        # solver.eas_alpha corrupted at the last rejected iterate; without
        # restoring, the retry warm-starts from garbage. Missing this was
        # a real bug in this script, not in the element kernel.
        checkpoint = solver.save_state()
        n_iter = solver.solve_step(dt)
        if n_iter is not None and n_iter >= 0:
            t += dt
            dt_ctrl.dt = dt_ctrl.update(n_iter, True)
        else:
            solver.restore_state(checkpoint)
            cutbacks += 1
            dt_ctrl.dt = dt_ctrl.update(abs(n_iter) if n_iter is not None else 30, False)
            if cutbacks > 60 or dt_ctrl.dt < dt_ctrl.dt_min * 1.01:
                return None, None, t, cutbacks

    uy = float(np.mean([solver.u[2 * i + 1] for i in tip_idx]))
    ux = float(np.mean([solver.u[2 * i] for i in tip_idx]))
    return uy, ux, 1.0, cutbacks


if __name__ == "__main__":
    I = THICK * H ** 3 / 12.0
    theory = BisshoppDruckerElastica(E, I, L, P)
    theta_tip, x_tip_exact, y_tip_exact = theory.solve()
    print(f"=== Bisshopp & Drucker (1945) exact elastica: EI={E*I:.4f}, beta=PL^2/EI={P*L**2/(E*I):.4f} ===")
    print(f"    theta_tip = {np.degrees(theta_tip):.3f} deg, "
          f"x_tip = {x_tip_exact:.4f} (horiz. shortening from L), "
          f"y_tip = {y_tip_exact:.4f} (vertical deflection)")
    print(f"    L - x_tip (final horizontal position from base) = {L - x_tip_exact:.4f}")
    print()
    print(f"{'nx':>4} {'elem':>16} {'backend':>8} {'uy':>10} {'err%':>7} {'ux':>10} {'err%':>7} {'cutbacks':>9}")
    for nx, label in ((10, "coarse (10el)"), (20, "fine (20el)")):
        for elem_type in ("Q4_COROTATIONAL", "Q4_EAS"):
            for backend in ("jax", "numba"):
                uy, ux, t_reached, cb = run_one(nx, elem_type, elem_jit=backend)
                if uy is None:
                    print(f"{nx:>4} {elem_type:>16} {backend:>8} {'FAILED':>10} {'-':>7} {'-':>10} {'-':>7} {cb:>9}")
                    continue
                uy_err = (uy / y_tip_exact - 1) * 100
                ux_exact = -(L - x_tip_exact)  # shortening, our ux is negative
                ux_err = (ux / ux_exact - 1) * 100 if abs(ux_exact) > 1e-9 else float("nan")
                print(f"{nx:>4} {elem_type:>16} {backend:>8} {uy:>10.4f} {uy_err:>+7.2f} {ux:>10.4f} {ux_err:>+7.2f} {cb:>9}", flush=True)
