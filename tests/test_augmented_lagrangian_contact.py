"""
test_augmented_lagrangian_contact.py
======================================
Phase 2 augmented-Lagrangian contact (design doc sec4;
dispsolver/constraint3d/surface_contact3d.py's SurfaceContactConstraint3D
with augmented_lagrange=True; dispsolver/solver3d/dynamic3d.py's
solve_step_augmented()).

**FIXED, 2026-09-13.** Previously: running repeated augmentation cycles at
a FIXED prescribed displacement (the Lagrange multiplier update is the
only thing changing between cycles) produced max_penetration growing
GEOMETRICALLY -- ratio ~1.422 essentially exactly repeated across 7
consecutive cycles (0.138 -> 0.200 -> 0.285 -> 0.405 -> 0.577 -> 0.820 ->
1.166), reproducing regardless of relaxation (omega swept 0.1/0.3/0.5/1.0,
all diverged).

Root cause: `SurfaceContactConstraint3D.assemble()` added the contact
force into `f_int_global` with the opposite sign from the "internal
force" convention every element (and the surface tie) already uses for
that same accumulator (verified by finite difference: the coded force's
own derivative did not match its own coded tangent; independently
confirmed by comparing the sign relationship between the converged
elastic force and the converged contact force against the true physical
equilibrium condition F_cube_physical == -F_contact_physical). This made
the solved equilibrium satisfy F_cube_physical == +F_contact_physical
instead -- plausible-looking (a positive force, `converged=True` on the
loose relative-residual tolerance) but not real force balance (measured
nodal residual ~1e3, not ~0) -- and it flipped the sign of d(gap)/d(lam)
from the theoretically-correct +1/(K_elastic+k_contact) to the observed
-1/(K_elastic-k_contact), i.e. the outer Uzawa update's feedback
direction was inverted, so each cycle made penetration worse instead of
better. Fixed by negating the force term in `assemble()` (see that
method's own comment for the full derivation); the tangent did not need
to change (the two sign flips cancel).

Measured after the fix, same single-C3D8-element case: max_penetration
now contracts geometrically toward zero (0.0771 -> 0.0594 -> 0.0458 ->
0.0353 -> 0.0272 -> 0.0210 -> 0.0162 -> 0.0125 -> 0.0096 -> 0.0074 over 10
cycles, ratio ~0.771/cycle, constant), and the per-node force-balance
residual is ~1e-10 every cycle (genuine equilibrium, not just the loose
tolerance). This matches the first-principles 1D spring re-derivation
that this docstring used to say was contradicted by observation -- the
derivation was right; the code was wrong.
"""

import numpy as np

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.model.set import GeneralSet
from dispsolver.model.interaction import ContactProperty, AnalyticalRigidSurface, ContactPair


def _build_cube_contact(constraint_enforcement="AUGMENTED_LAGRANGE"):
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
    pair = ContactPair(
        name="CUBE_TO_FLOOR", master=plane, slave=slave_set, interaction_property=prop,
        constraint_enforcement=constraint_enforcement,
    )
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

    for n in top_nodes:
        solver.fixed_dofs[3 * nid_to_idx[n] + 2] = -0.6

    return solver, contact


def test_augmented_lagrangian_converges():
    """Repeated augmentation cycles at a FIXED BC must converge
    max_penetration toward a small, stable value (standard Uzawa
    contraction). Measured 2026-09-13 after the f_node sign fix (see
    module docstring): max_penetration shrinks geometrically, ratio
    ~0.771/cycle, constant -- not just "eventually smaller", a genuine
    contraction every single cycle."""
    solver, contact = _build_cube_contact("AUGMENTED_LAGRANGE")

    penetrations = []
    for _ in range(10):
        converged, _iters = solver.solve_step(dt=1.0, max_iters=60)
        assert converged, "inner Newton solve itself should still converge each cycle"
        _, _, stats = contact.assemble(solver.u)
        penetrations.append(stats["max_penetration"])
        contact.update_augmented_multipliers(solver.u, omega=1.0)

    ratios = [penetrations[i + 1] / penetrations[i] for i in range(len(penetrations) - 1)]
    assert all(r < 1.0 for r in ratios), (
        f"expected every cycle to shrink max_penetration (contracting Uzawa iteration): "
        f"{penetrations} (ratios {ratios})"
    )
    assert penetrations[-1] < 0.1 * penetrations[0], (
        f"expected substantial convergence over 10 cycles: {penetrations}"
    )


def test_plain_penalty_contact_still_unaffected():
    """Sanity check: the augmented-Lagrangian scaffolding must not have
    changed plain PENALTY contact's behavior at all (augmented_lagrange
    defaults False on SurfaceContactConstraint3D, and ContactPair only
    sets it True for constraint_enforcement='AUGMENTED_LAGRANGE')."""
    solver, contact = _build_cube_contact("PENALTY")
    assert contact.augmented_lagrange is False
    converged, _iters = solver.solve_step(dt=1.0, max_iters=60)
    assert converged
    _, _, stats = contact.assemble(solver.u)
    assert stats["n_active"] == 4
    assert stats["total_normal_force"] > 0.0
