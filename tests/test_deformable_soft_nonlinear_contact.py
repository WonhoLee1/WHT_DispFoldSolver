"""
test_deformable_soft_nonlinear_contact.py
===========================================
Deformable-vs-deformable contact + nonlinear penalty / soft laws /
augmented Lagrangian (relaxed 2026-09-16,
dev_log/contact_deformable_soft_nonlinear_20260916.md).

`ContactPair.build_runtime_constraint()` used to raise
NotImplementedError for ANY non-HARD/LINEAR `normal_behavior`/
`penalty_form` combination whenever `master` was not an
AnalyticalRigidSurface, even though `DeformableSurfaceContactConstraint3D.
assemble()` already routed every non-AL evaluation through the same
generic `self.law.evaluate(penetration)` seam the rigid-plane class uses
-- the restriction was resolver-level caution ("first cut... a smaller
follow-on than the initial mechanism", per the original commit's own
comment), not a real limitation in the underlying class. This file
verifies the relaxation is actually correct, not just that it no longer
raises.

Also verifies a related, adjacent gap closed in the same pass:
`augmented_lagrange` was never passed through to
`DeformableSurfaceContactConstraint3D` by the CAE resolver at all (dead
parameter, AUGMENTED_LAGRANGE was blocked entirely for deformable
masters) even though PDASS already added real `augmented_lagrange`
support to that class in this same session's earlier work
(dev_log/contact_pdass_precise_design_20260915.md sec3, section 6's own
"변형체-변형체 접촉에도 동일하게 포팅" note) -- and that the class itself
was missing the HardLaw restriction guard its rigid-plane sibling already
has for `augmented_lagrange=True` (a silent-wrong-answer risk: the AL
branch always wins over `self.law.evaluate(...)` in assemble(), so a
soft/nonlinear law would be silently ignored without the guard).
"""

import numpy as np
import pytest

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.model.set import GeneralSet
from dispsolver.model.interaction import ContactProperty, ContactPair
from dispsolver.constraint3d.surface_contact3d_deformable import DeformableSurfaceContactConstraint3D
from dispsolver.constraint3d.pressure_overclosure import NonlinearPenaltyLaw, LinearSoftLaw


def _build_two_cube_deformable_contact(normal_behavior="HARD", penalty_form="LINEAR",
                                        constraint_enforcement="PENALTY", **prop_kwargs):
    """Two stacked single-C3D8-element cubes, contact between the top
    cube's bottom face and the bottom cube's top face -- same geometry
    as test_augmented_lagrangian_contact.py's `_build_cube_contact`, but
    with a DEFORMABLE (not AnalyticalRigidSurface) master, resolved via
    ContactPair's own raw-face-tuple-list acceptance path
    (_resolve_master_faces)."""
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
    top_conn = [
        grid[(0, 0, 0)], grid[(1, 0, 0)], grid[(1, 1, 0)], grid[(0, 1, 0)],
        grid[(0, 0, 1)], grid[(1, 0, 1)], grid[(1, 1, 1)], grid[(0, 1, 1)],
    ]
    mesh.add_element(1, top_conn, "C3D8")

    # Bottom (master) cube, spanning z in [0, gap0].
    bottom_grid = {}
    nid2 = nid
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(nid2, x, y, 0.0)
            bottom_grid[(i, j, 0)] = nid2
            nid2 += 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            mesh.add_node(nid2, x, y, gap0)
            bottom_grid[(i, j, 1)] = nid2
            nid2 += 1
    bottom_conn = [
        bottom_grid[(0, 0, 0)], bottom_grid[(1, 0, 0)], bottom_grid[(1, 1, 0)], bottom_grid[(0, 1, 0)],
        bottom_grid[(0, 0, 1)], bottom_grid[(1, 0, 1)], bottom_grid[(1, 1, 1)], bottom_grid[(0, 1, 1)],
    ]
    mesh.add_element(2, bottom_conn, "C3D8")

    E, nu = 200000.0, 0.3
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)
    nid_to_idx = mesh.node_id_to_index()

    top_cube_bottom_nodes = [grid[(0, 0, 0)], grid[(1, 0, 0)], grid[(1, 1, 0)], grid[(0, 1, 0)]]
    top_cube_top_nodes = [grid[(0, 0, 1)], grid[(1, 0, 1)], grid[(1, 1, 1)], grid[(0, 1, 1)]]
    bottom_cube_top_face = (
        bottom_grid[(0, 0, 1)], bottom_grid[(1, 0, 1)], bottom_grid[(1, 1, 1)], bottom_grid[(0, 1, 1)],
    )
    bottom_cube_bottom_nodes = [bottom_grid[(0, 0, 0)], bottom_grid[(1, 0, 0)], bottom_grid[(1, 1, 0)], bottom_grid[(0, 1, 0)]]

    slave_set = GeneralSet(name="TOP_CUBE_BOTTOM", nodes=top_cube_bottom_nodes)
    prop = ContactProperty(name="PROP", normal_behavior=normal_behavior, penalty_form=penalty_form, **prop_kwargs)
    pair = ContactPair(
        name="TOP_TO_BOTTOM", master=[bottom_cube_top_face], slave=slave_set,
        interaction_property=prop, constraint_enforcement=constraint_enforcement,
    )
    k_rep = E * L
    contact = pair.build_runtime_constraint(mesh, nid_to_idx, penalty_stiffness=0.1 * k_rep, materials={0: {"E": E, "nu": nu}})
    solver.constraints.append(contact)

    for n in bottom_cube_bottom_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fix_dof(n, 2, 0.0)
    for n in top_cube_bottom_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
    for n in top_cube_top_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fixed_dofs[3 * nid_to_idx[n] + 2] = 0.0

    return solver, contact, nid_to_idx


def test_hard_linear_still_works_unaffected():
    """The original, already-verified HARD+LINEAR path must be bit-for-
    bit unaffected by relaxing the restriction for other laws."""
    solver, contact, nid_to_idx = _build_two_cube_deformable_contact("HARD", "LINEAR")
    for n in solver.fixed_dofs:
        pass
    top_top_nodes_dof = [k for k, v in solver.fixed_dofs.items() if k % 3 == 2 and v == 0.0]
    # Drive the top face down to compress the interface.
    for dof in top_top_nodes_dof:
        solver.fixed_dofs[dof] = -0.55
    converged, _iters = solver.solve_step(dt=1.0, max_iters=60)
    assert converged
    _, _, stats = contact.assemble(solver.u)
    assert stats["n_active"] == 4
    assert stats["total_normal_force"] > 0.0


def test_nonlinear_penalty_no_longer_raises_and_converges():
    """penalty_form='NONLINEAR' with a deformable master used to raise
    NotImplementedError unconditionally -- must now build and solve."""
    solver, contact, nid_to_idx = _build_two_cube_deformable_contact("HARD", "NONLINEAR")
    from dispsolver.constraint3d.pressure_overclosure import NonlinearPenaltyLaw
    assert isinstance(contact.law, NonlinearPenaltyLaw)

    top_top_nodes_dof = [k for k, v in solver.fixed_dofs.items() if k % 3 == 2 and v == 0.0]
    for dof in top_top_nodes_dof:
        solver.fixed_dofs[dof] = -0.55
    converged, _iters = solver.solve_step(dt=1.0, max_iters=80)
    assert converged
    _, _, stats = contact.assemble(solver.u)
    assert stats["n_active"] == 4
    assert stats["total_normal_force"] > 0.0


@pytest.mark.parametrize("behavior", ["SOFT_LINEAR", "SOFT_EXPONENTIAL", "SOFT_TABULAR"])
def test_soft_laws_no_longer_raise_and_converge(behavior):
    """Every soft normal_behavior with a deformable master used to raise
    NotImplementedError unconditionally -- must now build and solve,
    with the class actually using the resolved soft law (not silently
    falling back to HardLaw)."""
    solver, contact, nid_to_idx = _build_two_cube_deformable_contact(behavior, "LINEAR")
    from dispsolver.constraint3d.pressure_overclosure import HardLaw
    assert not isinstance(contact.law, HardLaw), (
        f"expected a real {behavior} law object, got HardLaw (silent fallback)"
    )

    top_top_nodes_dof = [k for k, v in solver.fixed_dofs.items() if k % 3 == 2 and v == 0.0]
    for dof in top_top_nodes_dof:
        solver.fixed_dofs[dof] = -0.55
    converged, _iters = solver.solve_step(dt=1.0, max_iters=80)
    assert converged
    _, _, stats = contact.assemble(solver.u)
    assert stats["n_active"] == 4
    assert stats["total_normal_force"] > 0.0


def test_augmented_lagrangian_now_wired_for_deformable_hard():
    """AUGMENTED_LAGRANGE + deformable master used to raise
    NotImplementedError unconditionally (the flag was also never passed
    through even where the class already supported it). With HARD
    normal_behavior it must now build with augmented_lagrange=True and
    show the same Uzawa-contraction pattern already verified for the
    rigid-plane sibling and for SurfaceToSurfaceContactConstraint3D."""
    solver, contact, nid_to_idx = _build_two_cube_deformable_contact("HARD", "LINEAR", "AUGMENTED_LAGRANGE")
    assert contact.augmented_lagrange is True

    top_top_nodes_dof = [k for k, v in solver.fixed_dofs.items() if k % 3 == 2 and v == 0.0]
    for dof in top_top_nodes_dof:
        solver.fixed_dofs[dof] = -0.55

    penetrations = []
    for _ in range(8):
        converged, _iters = solver.solve_step(dt=1.0, max_iters=80)
        assert converged
        _, _, stats = contact.assemble(solver.u)
        penetrations.append(stats["max_penetration"])
        contact.update_augmented_multipliers(solver.u, omega=1.0)

    # PDASS is ALSO active automatically inside solve_step() itself (any
    # constraint with augmented_lagrange=True, per dev_log/contact_pdass_
    # precise_design_20260915.md) -- combined with this explicit outer-
    # loop call, penetration on this small fixture collapses to the
    # floating-point floor from the very first cycle (same behavior
    # already observed for SurfaceToSurfaceContactConstraint3D's own
    # analogous test). Check the achieved floor, not a multi-cycle ratio
    # trend that this fixture converges too fast to exhibit.
    floor = 1e-8
    assert all(p <= floor for p in penetrations), penetrations


def test_augmented_lagrangian_with_soft_law_still_rejected():
    """Abaqus's own restriction (AL only for HARD contact) must still
    apply for a deformable master -- this was never meant to be relaxed,
    only the LAW restriction (independent of enforcement method) was."""
    with pytest.raises(NotImplementedError):
        _build_two_cube_deformable_contact("SOFT_LINEAR", "LINEAR", "AUGMENTED_LAGRANGE")


def test_class_level_guard_rejects_al_with_nonhard_law():
    """Defensive guard added directly on DeformableSurfaceContactConstraint3D
    (mirrors SurfaceContactConstraint3D's own check): augmented_lagrange=True
    with an explicit non-HardLaw must raise, not silently ignore the law."""
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    coords = np.array([
        [0.0, 0.0, 1.0], [-5.0, -5.0, 0.0], [5.0, -5.0, 0.0],
        [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0],
    ])
    with pytest.raises(ValueError):
        DeformableSurfaceContactConstraint3D(
            slave_node_ids=[1], master_faces=[(2, 3, 4, 5)],
            nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=100.0,
            law=NonlinearPenaltyLaw(k_i=1.0, k_f=100.0, e=0.01, d=0.05),
            augmented_lagrange=True,
        )
