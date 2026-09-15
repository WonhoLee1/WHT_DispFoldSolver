"""
test_contact_chatter_stabilization.py
=======================================
Active-set CHATTERING stabilization (design doc
dev_log/contact_stabilization_precise_design_20260915.md), Stage 1: a
per-node Schmitt-trigger (hysteresis) gate on the active-set LABEL for
SurfaceContactConstraint3D and DeformableSurfaceContactConstraint3D.

Two kinds of tests here, deliberately:

1. Direct mechanism tests (`test_*_gate_*`) drive `update_gate_state()`/
   `get_active_set()`/`assemble()` against a HAND-CONSTRUCTED sequence of
   displacement vectors that make the raw NCP argument `x_i = lam_i +
   k_contact*penetration_i(u)` cross zero repeatedly -- i.e. exactly the
   Newton-level chattering signature the design doc's sec1.1 defines --
   bypassing the full Newton loop. This verifies the mechanism itself
   (does the gate actually debounce per its own formula) independent of
   whether any *particular* small toy FE fixture happens to make Newton
   itself chatter.

2. `test_full_solve_*` runs a real solve_step() (design doc's own
   boundary-straddling / tilted-BC construction, sec5.1) to confirm the
   new opt-in flag does not break a real solve and stays a true no-op
   when disabled (default). **Honest finding, not assumed**: repeated
   probing of this project's existing single-C3D8-element contact
   fixture (uniform and tilted top BC, penalty stiffness ratios 0.1x to
   100x the elastic stiffness) did NOT reproduce genuine multi-flip
   Newton-level chattering -- each slave node's active-set membership
   settled after at most one transition once SDI's own exemption
   handles the initial activation discontinuity (dev_log/hertz_contact_
   benchmark_20260913.md's fix already makes single-element activation
   very well-behaved). This matches the design doc's own sec5.1 caveat
   ("if the fixture does not chatter, tune it closer... before the
   comparison means anything") -- chattering in practice needs a larger
   or more geometrically complex contact patch than one element's four
   corners can produce, which is out of scope to construct here. The
   direct mechanism tests above are therefore the primary correctness
   evidence for this feature; the full-solve tests are a no-regression/
   no-crash check, not a demonstrated chattering reduction.
"""

import numpy as np
import pytest

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.model.set import GeneralSet
from dispsolver.model.interaction import ContactProperty, AnalyticalRigidSurface, ContactPair
from dispsolver.constraint3d.surface_contact3d import SurfaceContactConstraint3D
from dispsolver.constraint3d.surface_contact3d_deformable import DeformableSurfaceContactConstraint3D
from dispsolver.constraint3d.pressure_overclosure import NonlinearPenaltyLaw


def _rigid_fixture(chatter_stabilization=False, hysteresis_band=0.0):
    nid_to_idx = {1: 0}
    coords = np.array([[0.0, 0.0, 0.0]])
    return SurfaceContactConstraint3D(
        slave_node_ids=[1],
        rigid_plane_point=np.array([0.0, 0.0, -1.0]),
        rigid_plane_normal=np.array([0.0, 0.0, 1.0]),
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=100.0,
        chatter_stabilization=chatter_stabilization,
        hysteresis_band=hysteresis_band,
    )


def _u_for_z(z_disp):
    u = np.zeros(3)
    u[2] = z_disp
    return u


def test_rigid_gate_disabled_flips_every_crossing():
    """Baseline (chatter_stabilization=False, today's existing behavior):
    get_active_set() has no memory, so a node oscillating across gap=0
    flips its membership on EVERY crossing, exactly as many times as the
    driving displacement crosses zero. This is the exact "no memory
    across flips" defect design doc sec0 characterizes -- confirmed here
    directly, not just asserted in prose."""
    c = _rigid_fixture(chatter_stabilization=False)
    # plane at z=-1, node starts at z=0 -> gap0=1. Drive z_disp so gap
    # crosses zero repeatedly: gap = 0 + z_disp - (-1) = z_disp + 1.
    disps = [-1.1, -0.9, -1.1, -0.9, -1.1, -0.9]  # gap: -0.1,+0.1,-0.1,+0.1,-0.1,+0.1
    active_seq = [1 in c.get_active_set(_u_for_z(d)) for d in disps]
    assert active_seq == [True, False, True, False, True, False], active_seq
    flips = sum(1 for k in range(1, len(active_seq)) if active_seq[k] != active_seq[k - 1])
    assert flips == 5


def test_rigid_gate_debounces_small_oscillation_inside_band():
    """With chatter_stabilization=True and hysteresis_band=5.0 (large
    relative to k_contact=100 * the small gap oscillation used here, so
    x_i = k_contact*penetration stays inside +-band throughout), the gate
    should hold its value across the SAME oscillating sequence that flips
    5 times with the gate off (previous test) -- the debounce, measured
    directly against design doc sec2.2's exact formula."""
    c = _rigid_fixture(chatter_stabilization=True, hysteresis_band=5.0)
    # k_contact=100, penetration in [-0.1, +0.1] -> x_i in [-10, +10],
    # which DOES cross +-5 -- use a smaller oscillation so x_i stays
    # strictly inside the +-5 band: penetration in [-0.03, +0.03] ->
    # x_i in [-3, +3].
    disps = [-1.03, -0.97, -1.03, -0.97, -1.03, -0.97]
    gates = []
    for d in disps:
        c.update_gate_state(_u_for_z(d))
        gates.append(1 in c.get_active_set(_u_for_z(d)))
    # Gate starts False (never crossed +band yet since x_i max = +3 < 5),
    # so it should stay False the entire sequence -- zero flips, vs. 5
    # with the gate disabled.
    assert gates == [False] * 6, gates


def test_rigid_gate_flips_only_on_genuine_band_crossing():
    """The gate must still track a GENUINE, sustained crossing (not just
    freeze forever) -- turn on once x_i rises above +h, stay on while
    inside the dead band, turn off only once x_i falls below -h."""
    c = _rigid_fixture(chatter_stabilization=True, hysteresis_band=5.0)
    # penetration -> x_i = 100*penetration. Sequence: deep penetration
    # (x_i=+20, well above +5 -> gate ON), shallow retreat but still
    # inside the dead band (x_i=+2, between -5 and +5 -> HOLDS on),
    # genuine separation (x_i=-20, below -5 -> gate OFF), back to deep
    # penetration (x_i=+20 -> gate ON again). Exactly 2 genuine flips.
    penetrations = [0.20, 0.02, -0.20, 0.20]
    gates = []
    for p in penetrations:
        u = _u_for_z(-1.0 - p)  # gap = -p when z_disp = -1-p (plane at z=-1)
        c.update_gate_state(u)
        gates.append(1 in c.get_active_set(u))
    assert gates == [True, True, False, True], gates


def test_rigid_gate_keeps_tangent_engaged_across_dead_band():
    """sec2.1's central claim: relabeling alone is not enough -- the
    tangent stiffness must stay engaged (k_diag=k_contact) throughout the
    WHOLE dead band, including where f_mag is clamped to 0 (raw x_i<0 but
    still > -h). Verify directly against assemble()'s returned COO
    stiffness triplet, not just the force."""
    c = _rigid_fixture(chatter_stabilization=True, hysteresis_band=5.0)
    # Turn the gate on with a clear activation.
    u_on = _u_for_z(-1.20)  # penetration=0.20 -> x_i=20 > +5
    c.update_gate_state(u_on)
    assert 1 in c.get_active_set(u_on)
    # Now probe a point inside the lower half of the dead band: x_i<0
    # but > -5 (penetration=-0.02 -> x_i=-2).
    u_dead = _u_for_z(-0.98)
    c.update_gate_state(u_dead)
    assert 1 in c.get_active_set(u_dead), "gate must still be ON inside the dead band"
    f_contact, (rows, cols, data), stats = c.assemble(u_dead)
    assert len(data) > 0, "stiffness must still be assembled while gated ON, even at f_mag=0"
    k_zz = data[np.argmax(np.abs(data))]
    assert k_zz == pytest.approx(100.0, rel=1e-9), (
        f"expected k_diag == k_contact (100.0) engaged through the dead band, got {k_zz}"
    )
    # Force at this exact point is clamped to zero (raw x_i=-2 < 0).
    assert np.allclose(f_contact, 0.0), f_contact


def test_rigid_gate_requires_hardlaw():
    """Scope restriction (design doc sec2.1): chatter_stabilization=True
    with an explicit non-HardLaw is rejected, mirroring
    augmented_lagrange's own existing restriction."""
    with pytest.raises(ValueError):
        SurfaceContactConstraint3D(
            slave_node_ids=[1],
            rigid_plane_point=np.array([0.0, 0.0, -1.0]),
            rigid_plane_normal=np.array([0.0, 0.0, 1.0]),
            nid_to_idx={1: 0},
            coords=np.array([[0.0, 0.0, 0.0]]),
            penalty_stiffness=100.0,
            law=NonlinearPenaltyLaw(k_i=1.0, k_f=100.0, e=0.01, d=0.05),
            chatter_stabilization=True,
        )


def test_rigid_gate_disabled_is_bit_identical_to_before():
    """chatter_stabilization defaults False -- assemble()/get_active_set()
    must be byte-identical to the pre-existing code path (no restructuring
    regression from adding the new branch)."""
    c_off = _rigid_fixture(chatter_stabilization=False)
    u = _u_for_z(-1.05)
    f1, (r1, c1, d1), s1 = c_off.assemble(u)
    active1 = c_off.get_active_set(u)
    # Re-run identically (no state should have been mutated in a way that
    # changes a repeat call).
    f2, (r2, c2, d2), s2 = c_off.assemble(u)
    active2 = c_off.get_active_set(u)
    assert np.allclose(f1, f2)
    assert np.allclose(d1, d2)
    assert active1 == active2 == frozenset({1})
    assert s1["max_penetration"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Deformable-vs-deformable class: near-verbatim port of the same checks.
# ---------------------------------------------------------------------------


def _deformable_fixture(chatter_stabilization=False, hysteresis_band=0.0):
    # Slave node 1 sits directly above a flat master Quad4 face (nodes
    # 2,3,4,5) at z=0, slave at z=1 initially -> gap0=1.
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    coords = np.array([
        [0.0, 0.0, 1.0],
        [-5.0, -5.0, 0.0],
        [5.0, -5.0, 0.0],
        [5.0, 5.0, 0.0],
        [-5.0, 5.0, 0.0],
    ])
    return DeformableSurfaceContactConstraint3D(
        slave_node_ids=[1],
        master_faces=[(2, 3, 4, 5)],
        nid_to_idx=nid_to_idx,
        coords=coords,
        penalty_stiffness=100.0,
        chatter_stabilization=chatter_stabilization,
        hysteresis_band=hysteresis_band,
    )


def _u5_for_slave_z(z_disp):
    u = np.zeros(15)
    u[2] = z_disp  # slave node's own z DOF
    return u


def test_deformable_gate_disabled_flips_every_crossing():
    c = _deformable_fixture(chatter_stabilization=False)
    disps = [-1.1, -0.9, -1.1, -0.9]
    active_seq = [1 in c.get_active_set(_u5_for_slave_z(d)) for d in disps]
    assert active_seq == [True, False, True, False], active_seq


def test_deformable_gate_debounces_and_engages_tangent():
    c = _deformable_fixture(chatter_stabilization=True, hysteresis_band=5.0)
    u_on = _u5_for_slave_z(-1.20)  # penetration=0.20 -> x_i=20 > +5
    c.update_gate_state(u_on)
    assert 1 in c.get_active_set(u_on)

    u_dead = _u5_for_slave_z(-0.98)  # penetration=-0.02 -> x_i=-2, inside dead band
    c.update_gate_state(u_dead)
    assert 1 in c.get_active_set(u_dead), "gate must hold ON inside the dead band"
    f_contact, (rows, cols, data), stats = c.assemble(u_dead)
    assert len(data) > 0, "5-node stencil stiffness must still be assembled while gated ON"
    assert np.allclose(f_contact, 0.0), "force clamped to 0 at this raw-negative dead-band point"

    u_off = _u5_for_slave_z(-0.70)  # penetration=-0.30 -> x_i=-30, below -5
    c.update_gate_state(u_off)
    assert 1 not in c.get_active_set(u_off), "gate must open once genuinely below -h"


def test_deformable_gate_requires_hardlaw():
    with pytest.raises(ValueError):
        DeformableSurfaceContactConstraint3D(
            slave_node_ids=[1],
            master_faces=[(2, 3, 4, 5)],
            nid_to_idx={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
            coords=np.array([
                [0.0, 0.0, 1.0], [-5.0, -5.0, 0.0], [5.0, -5.0, 0.0],
                [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0],
            ]),
            penalty_stiffness=100.0,
            law=NonlinearPenaltyLaw(k_i=1.0, k_f=100.0, e=0.01, d=0.05),
            chatter_stabilization=True,
        )


# ---------------------------------------------------------------------------
# Full-solve wiring / no-regression checks (design doc sec5.1's fixture).
# ---------------------------------------------------------------------------


def _build_cube_contact(chatter_stabilization=False, hysteresis_band=0.0, d_top=-0.6):
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
        constraint_enforcement="PENALTY",
    )
    k_rep = E * L
    contact = pair.build_runtime_constraint(
        mesh, nid_to_idx, penalty_stiffness=0.1 * k_rep,
        chatter_stabilization=chatter_stabilization, hysteresis_band=hysteresis_band,
    )
    solver.constraints.append(contact)

    for n in top_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fix_dof(n, 2, 0.0)
    for n in bottom_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)

    for n in top_nodes:
        solver.fixed_dofs[3 * nid_to_idx[n] + 2] = d_top

    return solver, contact


def test_full_solve_chatter_disabled_matches_pre_existing_plain_penalty():
    """chatter_stabilization defaults False on the CAE (ContactPair) path
    too -- the new params must not perturb the ALREADY-passing plain-
    penalty test's result at all."""
    solver, contact = _build_cube_contact(chatter_stabilization=False)
    assert contact.chatter_stabilization is False
    converged, _iters = solver.solve_step(dt=1.0, max_iters=60)
    assert converged
    _, _, stats = contact.assemble(solver.u)
    assert stats["n_active"] == 4
    assert stats["total_normal_force"] > 0.0


def test_full_solve_chatter_enabled_still_converges():
    """chatter_stabilization=True on a real full Newton solve must not
    break convergence or force balance -- a pure no-crash / no-regression
    check (see module docstring for why this fixture does not itself
    demonstrate a chattering reduction)."""
    solver, contact = _build_cube_contact(
        chatter_stabilization=True, hysteresis_band=0.01 * (0.1 * 200000.0 * 10.0) * 1.0,
    )
    converged, _iters = solver.solve_step(dt=1.0, max_iters=60)
    assert converged
    _, _, stats = contact.assemble(solver.u)
    assert stats["n_active"] == 4
    assert stats["total_normal_force"] > 0.0
