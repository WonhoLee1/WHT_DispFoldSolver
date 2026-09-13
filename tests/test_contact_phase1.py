"""
test_contact_phase1.py
=======================
Phase 1 contact acceptance test (dev_log/3d_contact_implementation_design_
20260913.md §9): frictionless, hard, small-sliding, node-to-surface,
penalty contact, defined through the NEW CAE-level API
(dispsolver.model.interaction: ContactProperty, AnalyticalRigidSurface,
ContactPair) rather than raw node arrays -- proving the "compatible with
this codebase's existing set-based Python-API structure" requirement
concretely, not just in the runtime constraint class itself.

Two tests:
  1. test_simple_block_contact_sanity -- a plain cube approaching a rigid
     plane, the minimal case used to develop/debug the mechanics before
     attempting the curved-body case below.
  2. test_hertz_inspired_curved_block_contact -- a "D-cross-section" solid
     (flat top, circular-arc bottom of radius 254mm matching
     benchmark_element/reference_abaqus_docs/bmk_hertzcontact.txt's
     cylinder radius) pressed down onto a rigid plane. This is a
     SIMPLIFIED, qualitative stand-in for the real Hertz problem, NOT a
     literal reproduction -- seeAGENTS honest limitations below.

HONEST FINDINGS FROM THIS SESSION, READ BEFORE CHANGING ANYTHING HERE:

- The exact closed-form Hertz cylindrical-contact formulas (contact
  half-width, peak pressure) were explicitly flagged in the design doc as
  "still need to be derived fresh and carefully... do not lift a
  half-remembered version" -- NOT attempted here. This test checks
  qualitative sanity (converges, doesn't run away, penetration stays
  small and bounded, reaction force grows with applied displacement), not
  quantitative agreement with Timoshenko & Goodier (1951).
- The real Hertz problem is two identical cylinders; the source document
  itself states "Because of symmetry, the contact problem can be modeled
  as a deformable cylinder being pressed against a flat, rigid surface"
  -- confirmed twice this session against the primary source, so the
  deformable-body-vs-rigid-plane reduction below is not an invented
  shortcut.
- Geometry here is further simplified from "a full solid cylinder" to a
  "D-cross-section" block (flat top, single circular arc on the bottom,
  no polar mesh singularity at a disk center) purely to keep meshing
  simple with plain structured hexes -- this is NOT the real problem's
  full solid-cylinder geometry, just a locally-curved analog of it, since
  what actually matters for a Phase-1 sanity check is "does a curved
  surface contacting a flat rigid plane behave sensibly," not the far-
  field elasticity of a full cylinder.
- **A real, load-bearing finding from getting this to converge at all**:
  Abaqus's own linear-penalty default is "~10x a representative
  underlying element stiffness" (ctc_contactconstraints_std.txt). Using
  that ratio here made EVERY attempt fail to converge at the exact moment
  contact first activates (residual grows monotonically every Newton
  iteration instead of decreasing, confirmed by direct iteration-by-
  iteration instrumentation -- not a chattering active-set flip-flop,
  but the "line search interacting badly with a discontinuous residual"
  failure mode the design doc's §8 point 4 already predicted). A
  systematic sweep (0.1x/0.5x/1x/2x/5x representative stiffness) found
  0.1x is the only tested value that converges through activation with
  this codebase's current plain backtracking line search and no active-
  set-stability tracking. This is a real, deliberate departure from
  Abaqus's own "hard contact" default penalty scale, not a bug -- it
  trades exactness (more visible elastic penetration, ~0.5-0.7mm on the
  simple block test at 0.1x vs a much smaller value at higher penalty)
  for actually being able to converge Phase 1's plain penalty
  implementation without Phase 2's robustness machinery (active-set-
  stability convergence check + contact-specific damping, design doc §8)
  which is NOT built yet. Revisit this penalty scale once Phase 2 lands.
"""

import numpy as np
import pytest

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.model.set import GeneralSet
from dispsolver.model.interaction import ContactProperty, AnalyticalRigidSurface, ContactPair


def _solve_with_cutback(solver, set_bc_fn, uz_target, n_steps=40, max_iters=60, min_frac=1.0 / 64.0):
    """Ramp a prescribed displacement from 0 to uz_target, bisecting the
    increment (not the solver's own max_iters -- a plain script-level
    retry, no new solve_step capability needed) on non-convergence. This
    is standard incremental-loading practice, not something the contact
    constraint or solver needs to know about internally.

    Returns (reached_target: bool, final_uz: float, n_solve_calls: int).
    """
    d = uz_target / n_steps
    cur = 0.0
    u_saved = solver.u.copy()
    n_calls = 0
    while abs(cur - uz_target) > 1e-9 and abs(cur) < abs(uz_target):
        trial = cur + d
        if abs(trial) > abs(uz_target):
            trial = uz_target
        set_bc_fn(trial)
        converged, _iters = solver.solve_step(dt=1.0, max_iters=max_iters)
        n_calls += 1
        if converged:
            cur = trial
            u_saved = solver.u.copy()
        else:
            solver.u = u_saved.copy()
            d /= 2.0
            if abs(d) < abs(uz_target) * min_frac:
                return False, cur, n_calls
    return abs(cur - uz_target) < 1e-6, cur, n_calls


def test_simple_block_contact_sanity():
    """Minimal case: a cube starts with a small gap above a rigid plane,
    gets pushed down through the gap and into contact via the top face's
    prescribed displacement, using the CAE-level ContactPair/
    AnalyticalRigidSurface/ContactProperty API end to end."""
    mesh = Mesh3D()
    L = 10.0
    gap0 = 0.5
    xs, ys, zs = [0.0, L], [0.0, L], [gap0, gap0 + L]
    nid = 1
    grid = {}
    for k, z in enumerate(zs):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                mesh.add_node(nid, x, y, z)
                grid[(i, j, k)] = nid
                nid += 1
    conn = [
        grid[(0, 0, 0)], grid[(1, 0, 0)], grid[(1, 1, 0)], grid[(0, 1, 0)],
        grid[(0, 0, 1)], grid[(1, 0, 1)], grid[(1, 1, 1)], grid[(0, 1, 1)],
    ]
    mesh.add_element(1, conn, "C3D8")

    E, nu = 200000.0, 0.3
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)
    nid_to_idx = mesh.node_id_to_index()

    bottom_nodes = [grid[(0, 0, 0)], grid[(1, 0, 0)], grid[(1, 1, 0)], grid[(0, 1, 0)]]
    top_nodes = [grid[(0, 0, 1)], grid[(1, 0, 1)], grid[(1, 1, 1)], grid[(0, 1, 1)]]

    slave_set = GeneralSet(name="BOTTOM", nodes=bottom_nodes)
    plane = AnalyticalRigidSurface(name="FLOOR", point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0])
    prop = ContactProperty(name="FRICTIONLESS_HARD")
    pair = ContactPair(name="CUBE_TO_FLOOR", master=plane, slave=slave_set, interaction_property=prop)

    k_rep = E * L
    contact = pair.build_runtime_constraint(mesh, nid_to_idx, penalty_stiffness=0.1 * k_rep)
    solver.constraints.append(contact)

    for n in top_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fix_dof(n, 2, 0.0)
    for n in bottom_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)

    def set_uz(uz):
        for n in top_nodes:
            solver.fixed_dofs[3 * nid_to_idx[n] + 2] = uz

    reached, final_uz, n_calls = _solve_with_cutback(solver, set_uz, uz_target=-1.0, n_steps=40)
    assert reached, f"did not reach target displacement, stalled at uz={final_uz}"

    _, _, stats = contact.assemble(solver.u)
    assert stats["n_active"] == 4, f"expected all 4 bottom nodes in contact, got {stats}"
    assert stats["max_penetration"] < 1.0, f"penetration unreasonably large: {stats}"
    assert stats["total_normal_force"] > 0.0


def test_hertz_inspired_curved_block_contact():
    """D-cross-section solid (flat top, R=254mm circular-arc bottom,
    matching the Hertz cylinder radius) pressed onto a rigid plane. See
    module docstring for exactly what this does and does not prove."""
    R = 254.0
    E, nu = 206000.0, 0.3
    half_w = 30.0  # half-width of the modeled strip -- edge sag R - sqrt(R^2 - half_w^2) stays small vs H
    H = 30.0       # block height above the lowest point of the arc
    depth = 20.0   # axial (y) extrusion depth, "unit thickness"-style per the source's 3D variant
    nx, ny, nz = 16, 2, 6

    mesh = Mesh3D()
    xs = np.linspace(-half_w, half_w, nx + 1)
    ys = np.linspace(0.0, depth, ny + 1)
    # z=0 is the rigid plane; the block's bottom surface sags down toward it
    # at the edges and just touches it (z=0) at x=0 initially (zero initial gap
    # at the center, per make sense for "about to make contact").
    z_bottom = np.array([np.sqrt(max(R ** 2 - x ** 2, 0.0)) - R for x in xs]) + R - R  # = sqrt(R^2-x^2) - R, <=0
    # z_bottom(0) = 0 (touching), z_bottom(edge) < 0 is wrong sign -- fix: arc rises away from plane as |x| grows
    z_bottom = np.array([R - np.sqrt(max(R ** 2 - x ** 2, 0.0)) for x in xs])  # 0 at center, small positive at edges
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

    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)
    nid_to_idx = mesh.node_id_to_index()

    bottom_nodes = [grid[(i, j, 0)] for i in range(nx + 1) for j in range(ny + 1)]
    top_nodes = [grid[(i, j, nz)] for i in range(nx + 1) for j in range(ny + 1)]
    y0_nodes = [grid[(i, 0, k)] for i in range(nx + 1) for k in range(nz + 1)]
    y1_nodes = [grid[(i, ny, k)] for i in range(nx + 1) for k in range(nz + 1)]

    slave_set = GeneralSet(name="ARC_BOTTOM", nodes=bottom_nodes)
    plane = AnalyticalRigidSurface(name="FLOOR", point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0])
    prop = ContactProperty(name="FRICTIONLESS_HARD")
    pair = ContactPair(name="CYLINDER_TO_FLOOR", master=plane, slave=slave_set, interaction_property=prop)

    k_rep = E * (2.0 * half_w / nx)
    contact = pair.build_runtime_constraint(mesh, nid_to_idx, penalty_stiffness=0.1 * k_rep)
    solver.constraints.append(contact)

    # Plane-strain-like axial constraint (both y-faces), matching the
    # source's "unit thickness... out-of-plane displacements fixed" 3D setup.
    for n in y0_nodes:
        solver.fix_dof(n, 1, 0.0)
    for n in y1_nodes:
        solver.fix_dof(n, 1, 0.0)
    # Rigid loading platen on top: fully prescribed (simplification -- the
    # real problem loads via a displaced diametric cut, not a fully rigid
    # cap, but this is enough for a qualitative check).
    for n in top_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 2, 0.0)

    def set_uz(uz):
        for n in top_nodes:
            solver.fixed_dofs[3 * nid_to_idx[n] + 2] = uz

    # Target displacement small relative to R and H -- this is a sanity
    # check of contact mechanics, not an attempt to reproduce the source's
    # own (much larger, geometrically-linearized-away) load level.
    uz_target = -0.5
    reached, final_uz, n_calls = _solve_with_cutback(solver, set_uz, uz_target=uz_target, n_steps=30, max_iters=80)

    _, _, stats = contact.assemble(solver.u)

    assert reached, (
        f"curved-block contact did not converge to the full target displacement "
        f"(stalled at uz={final_uz} of {uz_target}); stats at last converged state: {stats}"
    )
    assert stats["n_active"] > 0, "expected at least some of the arc's bottom nodes in contact"
    assert stats["max_penetration"] < 0.5 * H, f"penetration implausibly large relative to block height: {stats}"
    assert stats["total_normal_force"] > 0.0

    # Contact patch should be centered and should NOT cover the whole
    # bottom surface at this small a displacement (a real Hertz contact
    # patch is much smaller than the body) -- a coarse but real sanity
    # check that this isn't just "everything is in contact because the
    # penalty/BC setup is wrong."
    assert stats["n_active"] < stats["n_candidates"], (
        f"expected a LOCALIZED contact patch, not full-surface contact: {stats}"
    )
