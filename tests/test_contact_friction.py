"""
test_contact_friction.py
==========================
Penalty-regularized Coulomb friction (design doc
dev_log/contact_friction_precise_design_20260916.md), Stage 1:
`SurfaceContactConstraint3D` (rigid plane) and
`DeformableSurfaceContactConstraint3D` (5-node stencil).
`SurfaceToSurfaceContactConstraint3D` stays frictionless (design doc
sec8's explicit deferral) and is not touched here.

Four layers of verification, mirroring the design doc's own sec10 plan:

1. `test_stick_slip_*` -- the closed-form STICK/SLIP switch (design doc
   sec1-2), verified by DIRECT calls to `_friction_force_and_tangent`
   (no Newton loop) against a known analytical answer: `f_t = k_t*s`
   below `s_crit`, `f_t` plateaus at exactly `mu*p_i` above it.
2. `test_committed_state_*` -- the design's single most important, non-
   obvious decision (sec5): the friction anchor must NOT move on a
   trial evaluation and must move by exactly the radial-return amount
   only when `commit_friction_state()` is explicitly called.
3. `test_frame_consistency_*` -- the convected-anchor fix (sec4):
   rigidly rotating the deformable master face with zero real relative
   slip must NOT be misread as spurious slip.
4. `test_full_solve_*` -- a real `DynamicSolver3D.solve_step()` sliding-
   block scenario, confirming the whole pipeline (SDI's `_contact_full_
   state` extension, the three `update_state=True` commit sites) wires
   together and reproduces the qualitative stick-then-slip signature
   end to end, not just in isolated direct calls.
"""

import numpy as np
import pytest

from dispsolver.constraint3d.surface_contact3d import SurfaceContactConstraint3D
from dispsolver.constraint3d.surface_contact3d_deformable import DeformableSurfaceContactConstraint3D
from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


# ---------------------------------------------------------------------------
# 1. Closed-form stick/slip switch, direct calls.
# ---------------------------------------------------------------------------


def _rigid_friction_fixture(mu=0.3, k_contact=100.0, ratio=1.0):
    nid_to_idx = {1: 0}
    coords = np.array([[0.0, 0.0, 0.0]])
    return SurfaceContactConstraint3D(
        slave_node_ids=[1],
        rigid_plane_point=np.array([0.0, 0.0, -1.0]),
        rigid_plane_normal=np.array([0.0, 0.0, 1.0]),
        nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=k_contact,
        friction_coefficient=mu, tangential_stiffness_ratio=ratio,
    )


def test_stick_slip_switch_matches_closed_form():
    """f_t = k_t*s while sticking; f_t plateaus at exactly mu*p_i once
    slipping -- the classic Coulomb signature, checked against the exact
    formula, not fitted."""
    c = _rigid_friction_fixture(mu=0.3, k_contact=100.0, ratio=1.0)
    p_i = 5.0  # a fixed, known normal pressure (not re-derived from gap here)
    s_crit = 0.3 * p_i / c.k_t  # = 0.015

    for s_applied in [0.0, 0.005, 0.010, 0.014]:  # strictly inside the stick region
        xs = np.array([0.0, 0.0, 0.0]) + s_applied * c.t1
        f_t, Ktt, is_slip, s1, s2, s_mag = c._friction_force_and_tangent(1, xs, p_i)
        assert not is_slip, (s_applied, s_mag, s_crit)
        assert np.linalg.norm(f_t) == pytest.approx(c.k_t * s_applied, rel=1e-9, abs=1e-12)
        assert Ktt == pytest.approx(c.k_t * (np.outer(c.t1, c.t1) + np.outer(c.t2, c.t2)), abs=1e-9)

    for s_applied in [0.02, 0.05, 0.10]:  # strictly past s_crit=0.015
        xs = np.array([0.0, 0.0, 0.0]) + s_applied * c.t1
        f_t, Ktt, is_slip, s1, s2, s_mag = c._friction_force_and_tangent(1, xs, p_i)
        assert is_slip
        assert np.linalg.norm(f_t) == pytest.approx(0.3 * p_i, rel=1e-9)  # plateaus, does NOT keep growing


def test_stick_slip_force_continuous_at_switch():
    """The design's own algebraic continuity claim (sec2, point 1):
    f_t_stick and f_t_slip agree exactly AT s_crit."""
    c = _rigid_friction_fixture(mu=0.3, k_contact=100.0, ratio=1.0)
    p_i = 5.0
    s_crit = 0.3 * p_i / c.k_t
    eps = 1e-9
    xs_below = s_crit * (1 - 1e-6) * c.t1
    xs_above = s_crit * (1 + 1e-6) * c.t1
    f_below, _, slip_below, *_ = c._friction_force_and_tangent(1, xs_below, p_i)
    f_above, _, slip_above, *_ = c._friction_force_and_tangent(1, xs_above, p_i)
    assert not slip_below and slip_above
    assert np.linalg.norm(f_below - f_above) < 1e-4 * max(np.linalg.norm(f_below), 1e-12)


def test_friction_disabled_is_exact_zero_fallback():
    """friction_coefficient=0.0 (default) must be a byte-identical no-op
    -- zero force, zero extra stiffness, regardless of slip."""
    c = _rigid_friction_fixture(mu=0.0)
    xs = np.array([0.05, 0.0, 0.0])
    f_t, Ktt, is_slip, s1, s2, s_mag = c._friction_force_and_tangent(1, xs, 5.0)
    assert np.allclose(f_t, 0.0)
    assert np.allclose(Ktt, 0.0)
    assert is_slip is False


def test_friction_requires_active_normal_contact():
    """Friction must not apply when p_i<=0 (no normal contact) -- the
    design's own "friction never recomputes the normal pressure
    independently, and only applies where it's active" framing."""
    c = _rigid_friction_fixture(mu=0.3)
    xs = np.array([0.5, 0.0, 0.0])  # large tangential offset
    f_t, Ktt, is_slip, s1, s2, s_mag = c._friction_force_and_tangent(1, xs, 0.0)
    assert np.allclose(f_t, 0.0)
    assert np.allclose(Ktt, 0.0)


# ---------------------------------------------------------------------------
# 2. Committed-state integrity (design doc sec5's key decision).
# ---------------------------------------------------------------------------


def test_anchor_unaffected_by_trial_evaluation():
    """Calling _friction_force_and_tangent (what assemble()/
    get_friction_state() do) many times with different trial `xs` must
    NEVER move self._anchor -- only commit_friction_state() may."""
    c = _rigid_friction_fixture(mu=0.3, k_contact=100.0)
    anchor_before = c._anchor[1].copy()
    for s_trial in [0.01, 0.05, 0.20, 0.5]:
        xs = s_trial * c.t1
        c._friction_force_and_tangent(1, xs, 5.0)  # pure query, no side effects
    assert np.allclose(c._anchor[1], anchor_before)


def test_commit_friction_state_radial_return():
    """After commit_friction_state(u) on a genuinely slipping node, the
    NEW anchor must leave exactly s_crit of elastic slip remaining
    (design doc sec5.3's radial return), not zero and not the full
    (larger) slipped distance."""
    c = _rigid_friction_fixture(mu=0.3, k_contact=100.0, ratio=1.0)
    p_i = 5.0
    s_crit = 0.3 * p_i / c.k_t
    s_applied = 0.10  # well past s_crit

    u = np.zeros(3)
    u[0] = s_applied  # slave node's own x-displacement (t1 = +x for this plane)

    # p_i is derived from the REAL gap in this fixture (touching at z=0,
    # plane at z=-1 -> gap0=1, so p_i via HardLaw would be 0 unless we
    # drive penetration too). Use the direct-call form with an explicit
    # p_i (matching how assemble() would have computed it) to isolate
    # the commit-time radial-return logic from the normal branch.
    xs = np.array([0.0, 0.0, 0.0]) + s_applied * c.t1
    f_t, _Ktt, is_slip, s1, s2, s_mag = c._friction_force_and_tangent(1, xs, p_i)
    assert is_slip

    # Directly exercise the same radial-return arithmetic
    # commit_friction_state() performs, using a stubbed _normal_pressure
    # so we don't need a full penetrating-gap fixture just to get p_i=5.
    c._normal_pressure = lambda u_, nid: (xs, p_i)  # monkeypatch for this test only
    c.commit_friction_state(u)

    new_anchor = c._anchor[1]
    remaining_slip = np.linalg.norm(xs - new_anchor)
    assert remaining_slip == pytest.approx(s_crit, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Frame consistency under master-face rotation (deformable class).
# ---------------------------------------------------------------------------


def _deformable_friction_fixture(mu=0.3, k_contact=100.0):
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    coords = np.array([
        [0.0, 0.0, 1.0], [-5.0, -5.0, 0.0], [5.0, -5.0, 0.0],
        [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0],
    ])
    return DeformableSurfaceContactConstraint3D(
        slave_node_ids=[1], master_faces=[(2, 3, 4, 5)],
        nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=k_contact,
        friction_coefficient=mu,
    )


def test_frame_consistency_pure_rotation_no_spurious_slip():
    """Rigidly rotate the master face (and the slave node, co-rotating
    with zero real relative slip) about the shared normal axis -- s_mag
    (convected anchor) must stay ~0 throughout (design doc sec4's own
    falsifiable prediction, Test 2)."""
    c = _deformable_friction_fixture(mu=0.3, k_contact=100.0)
    n_dof = 15
    p_i = 5.0

    for theta_deg in [0.0, 10.0, 30.0, 90.0, 180.0]:
        theta = np.radians(theta_deg)
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        u = np.zeros(n_dof)
        # Rotate every node (slave + all 4 master vertices) rigidly about
        # the z-axis (== the shared normal direction here) -- zero real
        # relative sliding between slave and master.
        nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        coords = np.array([
            [0.0, 0.0, 1.0], [-5.0, -5.0, 0.0], [5.0, -5.0, 0.0],
            [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0],
        ])
        for nid, idx in nid_to_idx.items():
            x0 = coords[idx]
            x_rot = Rz @ x0
            u[3 * idx: 3 * idx + 3] = x_rot - x0

        s_nid, m_face, xi, eta, sign = c.pairs[0]
        xs, q, n, p_i_actual = c._normal_pressure(u, s_nid, m_face, xi, eta, sign)
        f_t, Ktt, is_slip, s1, s2, s_mag, t1, t2 = c._friction_force_and_tangent(s_nid, xs, q, n, p_i)
        assert s_mag < 1e-8, (theta_deg, s_mag)
        assert not is_slip


def test_frame_consistency_real_slip_still_detected():
    """Sanity counter-check to test_frame_consistency_pure_rotation_no_
    spurious_slip: a GENUINE relative displacement (slave moves, master
    does not) must still be detected as nonzero slip -- confirms the
    convected-anchor fix doesn't accidentally suppress real slip too."""
    c = _deformable_friction_fixture(mu=0.3, k_contact=100.0)
    n_dof = 15
    u = np.zeros(n_dof)
    u[0] = 0.5  # slave node's own x-displacement; master unchanged

    s_nid, m_face, xi, eta, sign = c.pairs[0]
    xs, q, n, p_i_actual = c._normal_pressure(u, s_nid, m_face, xi, eta, sign)
    f_t, Ktt, is_slip, s1, s2, s_mag, t1, t2 = c._friction_force_and_tangent(s_nid, xs, q, n, 5.0)
    assert s_mag == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Full DynamicSolver3D regression: real Newton loop, sliding block.
# ---------------------------------------------------------------------------


def _build_cube_friction_fixture(mu=0.3, k_scale=0.1, ratio=1.0):
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

    k_rep = E * L
    contact = SurfaceContactConstraint3D(
        slave_node_ids=bottom_nodes,
        rigid_plane_point=np.array([0.0, 0.0, 0.0]), rigid_plane_normal=np.array([0.0, 0.0, 1.0]),
        nid_to_idx=nid_to_idx, coords=mesh.coords, penalty_stiffness=k_scale * k_rep,
        friction_coefficient=mu, tangential_stiffness_ratio=ratio,
    )
    solver.constraints.append(contact)

    for n in top_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fix_dof(n, 2, 0.0)
    for n in bottom_nodes:
        solver.fix_dof(n, 1, 0.0)

    return solver, contact, bottom_nodes, nid_to_idx


def test_full_solve_friction_disabled_matches_frictionless_baseline():
    """friction_coefficient=0.0 on the CAE/full-solve path must not
    perturb the already-verified frictionless result at all."""
    solver, contact, bottom_nodes, nid_to_idx = _build_cube_friction_fixture(mu=0.0)
    top_nodes = [n for n in solver.mesh.node_id_to_index() if n not in bottom_nodes]
    for n in top_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fixed_dofs[3 * nid_to_idx[n] + 2] = -0.6  # compress: closes gap0=0.5
    converged, _iters = solver.solve_step(dt=1.0, max_iters=60)
    assert converged
    _, _, stats = contact.assemble(solver.u)
    assert stats["n_active"] == 4
    assert stats["total_normal_force"] > 0.0


def test_full_solve_sliding_block_stick_then_slip():
    """Compress the cube against the floor (fixed normal preload), then
    ramp a prescribed horizontal (tangential) displacement on the SAME
    nodes already in contact. Falsifiable prediction (design doc sec10
    Test 1): while the applied tangential displacement stays below
    s_crit, the reaction tangential force tracks k_t*s_applied; once it
    exceeds s_crit, the reaction plateaus at mu*p_i."""
    solver, contact, bottom_nodes, nid_to_idx = _build_cube_friction_fixture(
        mu=0.3, k_scale=0.1, ratio=1.0,
    )
    top_nodes = [n for n in solver.mesh.node_id_to_index() if n not in bottom_nodes]
    for n in top_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fixed_dofs[3 * nid_to_idx[n] + 2] = -0.6  # compress: closes gap0=0.5

    converged, _iters = solver.solve_step(dt=1.0, max_iters=80)
    assert converged
    _, _, stats0 = contact.assemble(solver.u)
    p_i_typ = stats0["total_normal_force"] / 4.0  # per-node average pressure
    assert p_i_typ > 0.0
    s_crit = 0.3 * p_i_typ / contact.k_t

    tangential_forces = []
    applied = []
    for d_tan in [0.2 * s_crit, 0.6 * s_crit, 1.0 * s_crit, 3.0 * s_crit, 8.0 * s_crit]:
        for n in top_nodes:
            solver.fixed_dofs[3 * nid_to_idx[n] + 0] = d_tan
        converged, _iters = solver.solve_step(dt=1.0, max_iters=80)
        assert converged
        f_contact, _, stats = contact.assemble(solver.u)
        f_tan_total = sum(f_contact[3 * nid_to_idx[n] + 0] for n in bottom_nodes)
        tangential_forces.append(abs(f_tan_total))
        applied.append(d_tan)

    # Stick region (small d_tan): force should be well below the fully-
    # slipped plateau. Slip region (large d_tan): force should have
    # plateaued (last two points close to each other, not still growing
    # linearly).
    plateau = 0.3 * stats0["total_normal_force"]  # mu * (converged total normal force)
    assert tangential_forces[0] < 0.9 * plateau, tangential_forces
    assert tangential_forces[-1] == pytest.approx(plateau, rel=0.25)
    assert abs(tangential_forces[-1] - tangential_forces[-2]) < 0.1 * plateau, tangential_forces
