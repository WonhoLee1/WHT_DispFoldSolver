"""
benchmark_3d_contact.py
========================
Quantitative Hertz-contact comparison for the Phase 1 contact
implementation (dev_log/3d_contact_implementation_design_20260913.md,
dispsolver/model/interaction.py + dispsolver/constraint3d/surface_contact3d.py).

This is the follow-up the design doc's own acceptance-test note flagged as
"not attempted": tests/test_contact_phase1.py's curved-block test only
checks qualitative sanity (converges, bounded penetration, localized
patch). This module derives the closed-form 2D Hertz line-contact
formulas fresh from first principles (see `hertz_line_contact_reference`'s
docstring for the derivation, cross-checked against Johnson (1985)
*Contact Mechanics* eq. 4.43-4.46 and Timoshenko & Goodier (1951) Theory
of Elasticity 2nd ed. §140 -- both give the identical result, as they
must since Johnson's is the standard modern restatement of the same
classical theory), and compares them against this codebase's own FEM
result at the SAME reaction force the FEM itself produces for a given
prescribed displacement -- exactly the methodology
reference_abaqus_docs/bmk_hertzcontact.txt's own Figure 2 uses ("the
contact pressure ... predicted by the ... model is in good agreement with
the analytical distribution" -- a comparison made AT the FEM's own
converged load, not by trying to independently predict load from
displacement, which has no closed form in 2D plane-strain contact).

Geometry: the same "D-cross-section" curved block as
tests/test_contact_phase1.py::test_hertz_inspired_curved_block_contact
(flat top, single circular arc of radius R=254mm on the bottom, matching
the real Abaqus benchmark's cylinder radius and material properties
exactly: E=206 GPa, nu=0.3, frictionless hard contact against a flat
rigid plane) -- confirmed against the source
(reference_abaqus_docs/bmk_hertzcontact.txt) to be the same "deformable
cylinder vs. rigid flat plane" symmetry reduction the real benchmark
itself uses, just with the far-field body simplified from a full solid
cylinder to a locally-curved block (this codebase's contact machinery only
needs correct LOCAL curvature near the contact patch -- far-field body
shape does not enter the Hertz formula at all, only R at the contact point
does).

Two things this benchmark honestly does NOT claim:
1. It does not reproduce the real problem's absolute displacement->force
   relationship -- no closed form exists for that in 2D plane-strain
   contact (the classical "log-singularity" of 2D elasticity), and the
   source document doesn't provide one either.
2. The far-field boundary (flat top, rigid loading platen, finite block
   height H) is NOT the real problem's far-field (a full circular
   cross-section, loaded via the opposite diametric cut) -- so this
   benchmark can only be trusted for LOCAL contact-patch quantities
   (half-width, peak pressure) at a load level where the contact patch is
   small compared to both R and H (Hertz's own small-contact-patch
   assumption), not for anything relating to the block's overall
   deflection.
"""

from __future__ import annotations
import numpy as np

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.model.set import GeneralSet
from dispsolver.model.interaction import ContactProperty, AnalyticalRigidSurface, ContactPair


def hertz_line_contact_reference(P_per_length: float, R: float, E: float, nu: float) -> dict:
    """Closed-form 2D (plane-strain) Hertz line-contact solution for an
    elastic cylinder of radius R pressed against a rigid flat plane by a
    line load P_per_length (force per unit axial length), frictionless.

    Derivation (fresh, not lifted from memory -- see module docstring):

    Effective radius, two bodies in contact, R2 -> infinity for a flat
    plane: 1/R_eff = 1/R1 + 1/R2 = 1/R  =>  R_eff = R.

    Effective modulus: 1/E* = (1-nu1^2)/E1 + (1-nu2^2)/E2. Rigid plane
    (E2 -> infinity) drops the second term entirely:
        E* = E / (1 - nu^2)

    Contact half-width (Johnson 1985 eq. 4.45; Timoshenko & Goodier 1951
    Theory of Elasticity 2nd ed. section 140, same result):
        b = sqrt( 4 * P_per_length * R_eff / (pi * E*) )

    Peak (centerline) contact pressure, from the elliptical pressure
    distribution p(x) = p0*sqrt(1-(x/b)^2) integrating to
    P_per_length = (pi/2)*p0*b:
        p0 = 2*P_per_length / (pi*b)
    which substituting b above gives the equivalent closed form
        p0 = sqrt( P_per_length * E* / (pi * R_eff) )
    (both forms verified algebraically identical here; the second is used
    as the reported reference value, the first as b_ref).

    Returns dict: E_star, R_eff, b (half-width), p0 (peak pressure).
    """
    if P_per_length <= 0.0:
        return {"E_star": E / (1.0 - nu ** 2), "R_eff": R, "b": 0.0, "p0": 0.0}
    E_star = E / (1.0 - nu ** 2)
    R_eff = R
    b = np.sqrt(4.0 * P_per_length * R_eff / (np.pi * E_star))
    p0 = 2.0 * P_per_length / (np.pi * b)
    # cross-check against the algebraically-equivalent second form
    p0_alt = np.sqrt(P_per_length * E_star / (np.pi * R_eff))
    assert abs(p0 - p0_alt) / max(p0, 1e-30) < 1e-9, "Hertz p0 formula internal inconsistency"
    return {"E_star": E_star, "R_eff": R_eff, "b": b, "p0": p0}


def _graded_x(half_w: float, x_fine: float, dx_fine: float, n_coarse: int) -> np.ndarray:
    """Symmetric x-grid, fine (dx_fine) within |x|<x_fine, coarse (linear,
    n_coarse segments) from x_fine out to half_w. Uniform nx=60 puts
    dx=1mm everywhere, an order of magnitude coarser than the actual
    Hertz half-width at practical load levels (b~0.4mm measured at the
    smallest tested load) -- a uniform mesh fine enough to resolve that
    directly would need hundreds of elements across the whole width for
    no benefit far from the contact patch. Grading concentrates resolution
    only where the patch actually forms.
    """
    n_fine = max(1, int(round(x_fine / dx_fine)))
    right_fine = np.linspace(0.0, x_fine, n_fine + 1)
    right_coarse = np.linspace(x_fine, half_w, n_coarse + 1)[1:]
    right = np.concatenate([right_fine, right_coarse])
    xs = np.concatenate([-right[::-1][:-1], right])
    return xs


def make_hertz_curved_block_mesh(
    R=254.0, half_w=30.0, H=30.0, depth=1.0,
    nx=60, ny=2, nz=8,
    x_fine=None, dx_fine=None, n_coarse=20,
):
    """D-cross-section block: flat top, single circular arc (radius R) on
    the bottom, touching the z=0 plane at x=0. Same construction as
    tests/test_contact_phase1.py::test_hertz_inspired_curved_block_contact.

    If x_fine/dx_fine are given, the x-grid is GRADED (see `_graded_x`):
    fine spacing dx_fine within |x|<x_fine (where the Hertz contact patch
    actually forms), coarser linear spacing (n_coarse segments) out to
    half_w -- needed because a uniform mesh fine enough to resolve a
    sub-millimeter contact patch directly (nx in the hundreds) would waste
    almost all its resolution far from the patch. If not given, falls back
    to the original uniform nx-element grid (used by
    tests/test_contact_phase1.py's qualitative sanity check, which doesn't
    need patch-resolving resolution).

    depth=1.0 (matching the source's own "cylinder of unit thickness")
    means total_normal_force read off the contact constraint IS directly
    the line load P_per_length -- no extra division needed.

    Returns (mesh, grid, xs) where grid[(i,j,k)] -> node id and xs is the
    array of x-coordinates (needed to identify which columns are active).
    """
    mesh = Mesh3D()
    if x_fine is not None and dx_fine is not None:
        xs = _graded_x(half_w, x_fine, dx_fine, n_coarse)
        nx = len(xs) - 1
    else:
        xs = np.linspace(-half_w, half_w, nx + 1)
    ys = np.linspace(0.0, depth, ny + 1)
    z_bottom = np.array([R - np.sqrt(max(R ** 2 - x ** 2, 0.0)) for x in xs])
    z_top = z_bottom + H

    nid = 1
    grid = {}
    for k in range(nz + 1):
        frac = k / nz
        for j in range(ny + 1):
            for i in range(nx + 1):
                z = z_bottom[i] + frac * (z_top[i] - z_bottom[i])
                mesh.add_node(nid, xs[i], ys[j], z)
                grid[(i, j, k)] = nid
                nid += 1

    eid = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                conn = [
                    grid[(i, j, k)], grid[(i + 1, j, k)], grid[(i + 1, j + 1, k)], grid[(i, j + 1, k)],
                    grid[(i, j, k + 1)], grid[(i + 1, j, k + 1)], grid[(i + 1, j + 1, k + 1)], grid[(i, j + 1, k + 1)],
                ]
                mesh.add_element(eid, conn, "C3D8")
                eid += 1

    return mesh, grid, xs


def _solve_with_cutback(solver, set_bc_fn, uz_target, n_steps=40, max_iters=80, min_frac=1.0 / 128.0, augmented=False,
                      use_anderson=False, anderson_m=4):
    import time
    d = uz_target / n_steps
    cur = 0.0
    u_saved = solver.u.copy()
    n_calls = 0
    total_iters = 0
    t_wall = 0.0
    while abs(cur - uz_target) > 1e-9 and abs(cur) < abs(uz_target):
        trial = cur + d
        if abs(trial) > abs(uz_target):
            trial = uz_target
        set_bc_fn(trial)
        if getattr(solver, "debug_sdi", False):
            print(f"[CUTBACK] attempt trial={trial:.8f} cur={cur:.8f} d={d:.8f}", flush=True)
        t0 = time.perf_counter()
        if augmented:
            converged, _aug_iters, _iters = solver.solve_step_augmented(dt=1.0, max_iters=max_iters)
        else:
            converged, _iters = solver.solve_step(dt=1.0, max_iters=max_iters,
                                                  use_anderson_accel=use_anderson,
                                                  anderson_m=anderson_m)
        t_wall += time.perf_counter() - t0
        total_iters += _iters
        n_calls += 1
        if getattr(solver, "debug_sdi", False):
            print(f"[CUTBACK] -> converged={converged} iters={_iters}", flush=True)
        if converged:
            cur = trial
            u_saved = solver.u.copy()
        else:
            solver.u = u_saved.copy()
            d /= 2.0
            if abs(d) < abs(uz_target) * min_frac:
                return False, cur, n_calls, total_iters, t_wall
    return abs(cur - uz_target) < 1e-6, cur, n_calls, total_iters, t_wall


def run_hertz_contact_benchmark(
    R=254.0, E=206000.0, nu=0.3, half_w=30.0, H=30.0, depth=1.0,
    nx=60, ny=2, nz=8, uz_target=-0.06, n_steps=40,
    x_fine=None, dx_fine=None, n_coarse=20, max_iters=120, min_frac=1.0 / 1024,
    stabilization_coefficient=0.0, debug_sdi=False, penalty_form="LINEAR",
    constraint_enforcement="PENALTY", use_anderson=False, anderson_m=4,
    chatter_stabilization=False, hysteresis_band=0.0, hyst_frac=0.0,
):
    """Run the curved-block-vs-rigid-plane FEM contact solve, read off the
    reaction line load, compute the Hertz reference half-width/peak
    pressure AT THAT SAME LOAD, and compare against the FEM's own measured
    contact-patch half-width.

    FEM half-width measurement: the outermost x-coordinate (on either
    side of the symmetric center) among slave nodes currently reporting
    nonzero contact force, i.e. the actual discretized extent of the
    active set -- not a fitted curve, just the raw active-node span. This
    is inherently quantized to the mesh's own x-resolution (dx = 2*half_w
    / nx), which is reported alongside the comparison so a mismatch within
    one element width is distinguishable from a real discrepancy.

    Peak pressure is NOT independently reconstructed from the coarse FE
    nodal forces (a tributary-area-based nodal-force-to-pressure
    conversion is noisy on a mesh this coarse relative to the contact
    patch, and would not be an independent check anyway -- see module
    docstring). Only the half-width, and the line-load-consistent p0
    computed from the SAME closed-form pair, are reported.
    """
    mesh, grid, xs = make_hertz_curved_block_mesh(
        R=R, half_w=half_w, H=H, depth=depth, nx=nx, ny=ny, nz=nz,
        x_fine=x_fine, dx_fine=dx_fine, n_coarse=n_coarse,
    )
    nx = len(xs) - 1
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)
    if debug_sdi:
        solver.debug_sdi = True
    nid_to_idx = mesh.node_id_to_index()

    bottom_nodes = [grid[(i, j, 0)] for i in range(nx + 1) for j in range(ny + 1)]
    top_nodes = [grid[(i, j, nz)] for i in range(nx + 1) for j in range(ny + 1)]
    y0_nodes = [grid[(i, 0, k)] for i in range(nx + 1) for k in range(nz + 1)]
    y1_nodes = [grid[(i, ny, k)] for i in range(nx + 1) for k in range(nz + 1)]

    slave_set = GeneralSet(name="ARC_BOTTOM", nodes=bottom_nodes)
    plane = AnalyticalRigidSurface(name="FLOOR", point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0])
    prop = ContactProperty(name="FRICTIONLESS_HARD", penalty_form=penalty_form)
    pair = ContactPair(
        name="CYLINDER_TO_FLOOR", master=plane, slave=slave_set, interaction_property=prop,
        constraint_enforcement=constraint_enforcement,
    )

    # Representative element stiffness scale for the penalty: use the
    # FINEST local element size near the contact patch (dx_fine when
    # grading is active), not the coarse far-field spacing -- what
    # matters for contact activation robustness is the stiffness of the
    # elements actually touching the plane. Only used for penalty_form=
    # "LINEAR" (an explicit override); "NONLINEAR" derives Ki/Kf from the
    # auto k_ref estimator instead (dev_log/contact_abaqus_grade_design_20260915.md
    # sec A.2), which is why `materials` is now passed through so that
    # estimator picks up the benchmark's actual E rather than its 200000
    # MPa fallback default.
    dx_local = float(np.min(np.diff(xs)))
    k_rep = E * dx_local
    band = hysteresis_band if hysteresis_band > 0.0 else hyst_frac * 0.1 * k_rep
    stab = chatter_stabilization or (hyst_frac > 0.0)
    contact = pair.build_runtime_constraint(
        mesh, nid_to_idx, penalty_stiffness=0.1 * k_rep,
        stabilization_coefficient=stabilization_coefficient,
        chatter_stabilization=stab,
        hysteresis_band=band,
        materials={0: {"E": E, "nu": nu}},
    )
    solver.constraints.append(contact)

    for n in y0_nodes:
        solver.fix_dof(n, 1, 0.0)
    for n in y1_nodes:
        solver.fix_dof(n, 1, 0.0)
    for n in top_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 2, 0.0)

    def set_uz(uz):
        for n in top_nodes:
            solver.fixed_dofs[3 * nid_to_idx[n] + 2] = uz

    reached, final_uz, n_calls, total_iters, t_wall = _solve_with_cutback(
        solver, set_uz, uz_target=uz_target, n_steps=n_steps,
        max_iters=max_iters, min_frac=min_frac,
        augmented=(constraint_enforcement == "AUGMENTED_LAGRANGE"),
        use_anderson=use_anderson, anderson_m=anderson_m,
    )

    f_contact, _, stats = contact.assemble(solver.u)

    P_per_length = stats["total_normal_force"] / depth

    active_x = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = grid[(i, j, 0)]
            idx = nid_to_idx[nid]
            f_node = f_contact[3 * idx: 3 * idx + 3]
            if np.linalg.norm(f_node) > 1e-9:
                active_x.append(xs[i])
    b_fem = max(abs(x) for x in active_x) if active_x else 0.0

    ref = hertz_line_contact_reference(P_per_length, R, E, nu)

    ratio_b = b_fem / ref["b"] if ref["b"] > 0 else float("nan")

    return {
        "status": "CONVERGED" if reached else "DIVERGED",
        "final_uz": final_uz,
        "n_calls": n_calls,
        "total_iters": total_iters,
        "t_wall": t_wall,
        "use_anderson": use_anderson,
        "P_per_length": P_per_length,
        "n_active": stats["n_active"],
        "n_candidates": stats["n_candidates"],
        "max_penetration": stats["max_penetration"],
        "b_fem": b_fem,
        "dx_mesh": dx_local,
        "b_ref": ref["b"],
        "p0_ref": ref["p0"],
        "E_star": ref["E_star"],
        "ratio_b_fem_to_ref": ratio_b,
    }


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 78)
    print("Hertz Line-Contact Benchmark (reference_abaqus_docs/bmk_hertzcontact.txt)")
    print("=" * 78)
    print("R=254mm, E=206000 MPa, nu=0.3, frictionless hard contact vs rigid plane")
    print()

    for uz_target in (-0.02, -0.04, -0.06, -0.10):
        res = run_hertz_contact_benchmark(uz_target=uz_target)
        print(f"uz_target={uz_target:+.3f} mm  status={res['status']}")
        print(f"  P_per_length (FEM reaction)   = {res['P_per_length']:.4f} N/mm")
        print(f"  b_fem  (active-node halfwidth) = {res['b_fem']:.4f} mm  (mesh dx={res['dx_mesh']:.4f} mm)")
        print(f"  b_ref  (Hertz, at same P)      = {res['b_ref']:.4f} mm")
        print(f"  ratio b_fem/b_ref              = {res['ratio_b_fem_to_ref']:.4f}")
        print(f"  p0_ref (Hertz peak pressure)   = {res['p0_ref']:.4f} MPa")
        print(f"  n_active/n_candidates          = {res['n_active']}/{res['n_candidates']}")
        print(f"  max_penetration                = {res['max_penetration']:.5f} mm")
        print()
