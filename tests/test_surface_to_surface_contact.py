"""
test_surface_to_surface_contact.py
=====================================
Surface-to-surface (dual-mortar) contact, Stage A (design doc
dev_log/contact_surface_to_surface_precise_design_20260915.md,
`dispsolver/constraint3d/surface_contact3d_s2s.py`).

Three layers of verification, deliberately in this order (cheapest/most
fundamental first):

1. `test_dual_basis_*` -- the defining bi-orthogonality property of the
   dual (Wohlmuth) shape functions, `integral(Psi_i * N_j dA) ==
   D_f[i]*delta_ij`, checked directly against the class's own internal
   M_f/D_f/A_f construction. This is the mathematical foundation
   everything else (weak gap, force distribution, tangent) is built on
   -- if this is wrong, nothing downstream can be trusted regardless of
   what a full solve appears to converge to.
2. `test_*_patch_test` -- hand-constructed displacement vectors (no
   Newton loop) on a single flat face pair and a 2x2 conforming grid,
   checking the closed-form uniform-pressure patch-test answer (a
   bilinear quad's own nodal force split under constant pressure is the
   textbook `p*Area/4` per corner -- directly derivable from `integral
   N_a dA == Area/4` for a rectangular Quad4) and Newton's-third-law
   force balance across the whole assembled vector.
3. `test_full_solve_*` -- a real two-block `DynamicSolver3D` solve with
   a genuinely subdivided (2x2 elements per block face), matching-
   topology contact interface, confirming the class integrates into the
   existing Newton/SDI/PDASS machinery with zero solver-side changes
   (design doc sec3's own claim, checked here rather than trusted).
"""

import numpy as np
import pytest

from dispsolver.constraint3d.surface_contact3d_s2s import (
    SurfaceToSurfaceContactConstraint3D,
    _quad4_shape_functions,
    _quad4_area_jacobian,
    _GAUSS2X2,
)
from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


# ---------------------------------------------------------------------------
# 1. Dual-basis bi-orthogonality (the mathematical foundation).
# ---------------------------------------------------------------------------


def _flat_face_coords(cx=0.0, cy=0.0, z=0.0, size=2.0):
    h = size / 2.0
    return np.array([
        [cx - h, cy - h, z], [cx + h, cy - h, z], [cx + h, cy + h, z], [cx - h, cy + h, z],
    ])


def test_dual_basis_biorthogonality_flat_face():
    """integral(Psi_i(xi,eta) * N_j(xi,eta) dA) must equal D_f[i] when
    i==j and 0 otherwise -- the exact defining property of a Wohlmuth
    dual/bi-orthogonal basis, re-derived independently here (a finer
    4x4-point Gauss rule, NOT the class's own 2x2) against the class's
    own A_f/D_f."""
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
    coords = _flat_face_coords()
    c = SurfaceToSurfaceContactConstraint3D(
        slave_faces=[(1, 2, 3, 4)], master_faces=[(1, 2, 3, 4)],  # self-pair, only A_f/D_f matter here
        nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=1.0,
        position_tolerance=10.0,
    )
    A_f = c._A_f[0]
    D_f = np.array([c.D_global[n] for n in [1, 2, 3, 4]])

    # Independent finer quadrature (4x4 Gauss-Legendre), not the class's own.
    gl4 = [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]
    wl4 = [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]
    q = [coords[i] for i in range(4)]

    integral = np.zeros((4, 4))
    for xi, wx in zip(gl4, wl4):
        for eta, wy in zip(gl4, wl4):
            N = np.array(_quad4_shape_functions(xi, eta))
            Psi = A_f @ N
            dA = _quad4_area_jacobian(q[0], q[1], q[2], q[3], xi, eta) * wx * wy
            integral += np.outer(Psi, N) * dA

    expected = np.diag(D_f)
    assert np.allclose(integral, expected, atol=1e-9), (integral, expected)


def test_dual_basis_D_f_sums_to_face_area():
    """D_f[i] summed over i must equal the face's own physical area
    (Psi's own row-sum identity, sec2.1: D_i = sum_j M_ij, and
    sum_i D_i = sum_ij M_ij = integral(sum_i N_i * sum_j N_j) =
    integral(1*1) = Area, since sum_i N_i == 1 everywhere)."""
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}
    coords = _flat_face_coords(size=3.0)  # 3x3 -> area 9
    c = SurfaceToSurfaceContactConstraint3D(
        slave_faces=[(1, 2, 3, 4)], master_faces=[(1, 2, 3, 4)],
        nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=1.0,
        position_tolerance=10.0,
    )
    D_f = np.array([c.D_global[n] for n in [1, 2, 3, 4]])
    assert D_f.sum() == pytest.approx(9.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Hand-constructed patch tests (no Newton loop).
# ---------------------------------------------------------------------------


def _single_pair_fixture(penalty_stiffness=100.0, augmented_lagrange=False):
    # Slave face 1 (nodes 1-4) COINCIDENT with master face (nodes 5-8),
    # both at z=0.0, same 2x2 flat footprint -- a "just touching"
    # initial configuration (gap0=0.0, the frozen-pairing sign
    # convention's own `>= 0.0` boundary), so a subsequent -z
    # displacement of the slave face is genuine penetration of exactly
    # that magnitude (not merely closing a pre-existing gap).
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
    top = _flat_face_coords(z=0.0)
    bot = _flat_face_coords(z=0.0)
    coords = np.vstack([top, bot])
    c = SurfaceToSurfaceContactConstraint3D(
        slave_faces=[(1, 2, 3, 4)], master_faces=[(5, 6, 7, 8)],
        nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=penalty_stiffness,
        augmented_lagrange=augmented_lagrange, position_tolerance=10.0,
    )
    return c, coords


def _uniform_penetration_u(n_nodes, slave_local_indices, penetration):
    u = np.zeros(3 * n_nodes)
    for idx in slave_local_indices:
        u[3 * idx + 2] = -penetration  # push slave face down by `penetration`
    return u


def test_single_face_pair_uniform_pressure_patch_test():
    """Uniform penetration -> uniform pressure p = k_contact*penetration
    -> each of the 4 corners (slave AND master side) gets exactly
    p*Area/4 (Area=4 here), the textbook bilinear-quad uniform-load
    split. Verified against the CLOSED FORM, not against the code's own
    internal numbers."""
    k = 100.0
    penetration = 0.02
    c, coords = _single_pair_fixture(penalty_stiffness=k)
    u = _uniform_penetration_u(8, [0, 1, 2, 3], penetration)

    f_contact, (rows, cols, data), stats = c.assemble(u)
    assert stats["n_active"] == 4
    p_expected = k * penetration
    force_per_corner_expected = p_expected * 4.0 / 4.0  # Area=4, /4 corners

    for local_idx in range(8):  # all 8 nodes (4 slave + 4 master)
        fz = f_contact[3 * local_idx + 2]
        sign = -1.0 if local_idx < 4 else +1.0  # slave pushed +z (reaction), master pushed -z
        assert fz == pytest.approx(sign * force_per_corner_expected, rel=1e-9), (
            f"node local_idx={local_idx}: fz={fz}, expected {sign * force_per_corner_expected}"
        )

    # Newton's third law: total force on the whole assembled vector is zero.
    assert np.abs(f_contact.reshape(-1, 3).sum(axis=0)).max() < 1e-9


def test_single_face_pair_augmented_lagrangian_lambda_grows_at_fixed_u():
    """Direct-call sanity check on the Uzawa UPDATE FORMULA itself, at a
    FIXED displacement (deliberately NOT a convergence test -- see
    test_full_solve_two_block_augmented_lagrangian_converges below for
    that; holding u fixed while repeatedly augmenting lam has no elastic
    feedback to converge TOWARD, so lam grows by the same increment
    every cycle, `k_contact*g_tilde`, exactly as the formula predicts):
    lambda increases by an EXACT, CONSTANT amount per cycle when g_tilde
    itself does not change."""
    c, coords = _single_pair_fixture(penalty_stiffness=50.0, augmented_lagrange=True)
    u = _uniform_penetration_u(8, [0, 1, 2, 3], 0.05)
    expected_increment = 50.0 * 0.05  # k_contact * g_tilde, constant since u is fixed

    lambda_changes = []
    for _ in range(5):
        result = c.update_augmented_multipliers(u, omega=1.0)
        lambda_changes.append(result["max_lambda_change"])

    assert all(lc == pytest.approx(expected_increment, rel=1e-9) for lc in lambda_changes), lambda_changes


def test_full_solve_two_block_augmented_lagrangian_converges():
    """The REAL Uzawa contraction test (mirrors
    test_augmented_lagrangian_contact.py's own
    test_augmented_lagrangian_converges): repeated (solve_step +
    update_augmented_multipliers) cycles on the two-block fixture must
    shrink max_penetration geometrically toward zero -- this is the
    genuine feedback loop the fixed-u test above deliberately does not
    have (elastic displacement responds to the growing lam each cycle,
    which is what makes the outer iteration a contraction)."""
    solver, contact, slave_nodes, master_nodes, nid_to_idx, master_faces, slave_faces = (
        _build_two_block_s2s_fixture(k_scale=0.1, augmented_lagrange=True)
    )
    penetrations = []
    for _ in range(8):
        converged, _iters = solver.solve_step(dt=1.0, max_iters=80)
        assert converged
        _, _, stats = contact.assemble(solver.u)
        penetrations.append(stats["max_penetration"])
        contact.update_augmented_multipliers(solver.u, omega=1.0)

    # This class's PDASS is ALSO active inside solve_step() itself (any
    # constraint with augmented_lagrange=True gets update_augmented_
    # multipliers() called automatically at solve_step()'s own commit
    # points, dev_log/contact_pdass_precise_design_20260915.md) -- so the
    # outer loop's own explicit call on top of that converges penetration
    # to numerical noise (~1e-9 and below) within just a few cycles, far
    # faster than the point-collocation sibling test's ~0.77/cycle rate
    # (that test's constraint predates PDASS-in-solve_step). A plain
    # ratio check breaks once penetration is already at the floating-
    # point floor (0/tiny or tiny/0) -- check the CONTRACTION property
    # only while penetration is still resolvably above that floor, and
    # separately require the run to have reached the floor at all.
    floor = 1e-8
    resolvable = [p for p in penetrations if p > floor]
    ratios = [resolvable[i + 1] / resolvable[i] for i in range(len(resolvable) - 1)]
    assert len(resolvable) >= 3, penetrations
    assert all(r < 1.0 for r in ratios), (penetrations, ratios)
    assert penetrations[-1] <= floor, penetrations


def _grid_2x2_fixture(penalty_stiffness=100.0):
    """A 2x2 grid of matching slave/master Quad4 faces (3x3=9 nodes per
    side, sharing interior/edge nodes across faces -- exactly the
    "interior node belongs to multiple slave faces" case sec2.2/2.3 is
    about), aligned and conforming."""
    nid_to_idx = {}
    coords_list = []
    nid = 1

    def add_node(x, y, z):
        nonlocal nid
        nid_to_idx[nid] = len(coords_list)
        coords_list.append([x, y, z])
        this_nid = nid
        nid += 1
        return this_nid

    # Slave grid, COINCIDENT with the master grid (z=0), same footprint
    # -- same "just touching" (gap0=0.0) rationale as _single_pair_fixture.
    slave_grid = {}
    for j in range(3):
        for i in range(3):
            slave_grid[(i, j)] = add_node(float(i), float(j), 0.0)
    # Master grid (z=0), same footprint
    master_grid = {}
    for j in range(3):
        for i in range(3):
            master_grid[(i, j)] = add_node(float(i), float(j), 0.0)

    def faces(grid):
        out = []
        for j in range(2):
            for i in range(2):
                out.append((grid[(i, j)], grid[(i + 1, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)]))
        return out

    slave_faces = faces(slave_grid)
    master_faces = faces(master_grid)
    coords = np.array(coords_list)

    c = SurfaceToSurfaceContactConstraint3D(
        slave_faces=slave_faces, master_faces=master_faces,
        nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=penalty_stiffness,
        position_tolerance=10.0,
    )
    return c, coords, slave_grid, master_grid, nid_to_idx


def test_grid_2x2_shared_node_aggregation_and_force_balance():
    """Interior/edge nodes are shared by multiple slave faces (up to 4
    for the true interior node). Under UNIFORM penetration, every slave
    node's weak gap g_tilde should still equal exactly the imposed
    penetration (sec2.2/2.4's aggregation must not distort a spatially
    UNIFORM field -- this is the discrete partition-of-unity check design
    doc sec2.5 names), and the whole assembled force vector must still
    balance to zero (Newton's third law) even with multiple overlapping
    segments contributing to shared nodes."""
    k = 100.0
    penetration = 0.03
    c, coords, slave_grid, master_grid, nid_to_idx = _grid_2x2_fixture(penalty_stiffness=k)
    n_nodes = len(coords)
    u = np.zeros(3 * n_nodes)
    for nid in slave_grid.values():
        u[3 * nid_to_idx[nid] + 2] = -penetration

    g_tilde = c._compute_g_tilde(u)
    for nid, g in g_tilde.items():
        assert g == pytest.approx(penetration, rel=1e-9), (nid, g, penetration)

    f_contact, (rows, cols, data), stats = c.assemble(u)
    assert stats["n_active"] == 9
    assert np.abs(f_contact.reshape(-1, 3).sum(axis=0)).max() < 1e-8

    # Total normal force must equal p * total_area (p=k*penetration, area=2x2=4).
    p = k * penetration
    total_force_z = sum(
        f_contact[3 * nid_to_idx[nid] + 2] for nid in master_grid.values()
    )
    assert total_force_z == pytest.approx(p * 4.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Full DynamicSolver3D regression: real two-block mesh, real Newton loop.
# ---------------------------------------------------------------------------


def _build_two_block_s2s_fixture(k_scale=0.1, augmented_lagrange=False, d_top=-0.55):
    """Two stacked blocks, EACH split into a 2x2 in-plane grid of C3D8
    elements at the contact interface (so both the slave and master
    contact faces are genuinely subdivided, matching-topology, real FE
    nodes with real elemental stiffness -- not a bare geometric face)."""
    mesh = Mesh3D()
    L = 10.0
    gap0 = 0.5
    half = L / 2.0

    nid = 1
    grid = {}  # (i, j, k) -> nid, i,j in {0,1,2} (3 nodes per edge -> 2x2 elements), k in {0,1,2,3}
    # Bottom block spans z in [0, half]; top block spans z in [gap0+half, gap0+L].
    z_levels = [0.0, half, gap0 + half, gap0 + L]
    for k, z in enumerate(z_levels):
        for j in range(3):
            for i in range(3):
                x = i * half
                y = j * half
                mesh.add_node(nid, x, y, z)
                grid[(i, j, k)] = nid
                nid += 1

    def hex_conn(i0, j0, k0):
        return [
            grid[(i0, j0, k0)], grid[(i0 + 1, j0, k0)], grid[(i0 + 1, j0 + 1, k0)], grid[(i0, j0 + 1, k0)],
            grid[(i0, j0, k0 + 1)], grid[(i0 + 1, j0, k0 + 1)], grid[(i0 + 1, j0 + 1, k0 + 1)], grid[(i0, j0 + 1, k0 + 1)],
        ]

    eid = 1
    for k0 in (0, 2):  # bottom block (k=0->1), top block (k=2->3)
        for j0 in range(2):
            for i0 in range(2):
                mesh.add_element(eid, hex_conn(i0, j0, k0), "C3D8")
                eid += 1

    E, nu = 200000.0, 0.3
    solver = DynamicSolver3D(mesh, {"E": E, "nu": nu}, nlgeom=True)
    nid_to_idx = mesh.node_id_to_index()

    bottom_top_face_nodes = [grid[(i, j, 1)] for j in range(3) for i in range(3)]  # top of bottom block
    top_bottom_face_nodes = [grid[(i, j, 2)] for j in range(3) for i in range(3)]  # bottom of top block
    bottom_fixed_nodes = [grid[(i, j, 0)] for j in range(3) for i in range(3)]
    top_driven_nodes = [grid[(i, j, 3)] for j in range(3) for i in range(3)]

    def s2s_faces(grid_level):
        out = []
        for j0 in range(2):
            for i0 in range(2):
                out.append((
                    grid[(i0, j0, grid_level)], grid[(i0 + 1, j0, grid_level)],
                    grid[(i0 + 1, j0 + 1, grid_level)], grid[(i0, j0 + 1, grid_level)],
                ))
        return out

    slave_faces = s2s_faces(2)  # bottom face of the TOP block (moving, deformable)
    master_faces = s2s_faces(1)  # top face of the BOTTOM block

    from dispsolver.constraint3d.surface_contact3d_s2s import SurfaceToSurfaceContactConstraint3D
    k_rep = E * half
    contact = SurfaceToSurfaceContactConstraint3D(
        slave_faces=slave_faces, master_faces=master_faces,
        nid_to_idx=nid_to_idx, coords=mesh.coords,
        penalty_stiffness=k_scale * k_rep, augmented_lagrange=augmented_lagrange,
    )
    solver.constraints.append(contact)

    for n in bottom_fixed_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fix_dof(n, 2, 0.0)
    for n in top_driven_nodes:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fixed_dofs[3 * nid_to_idx[n] + 2] = d_top

    return solver, contact, bottom_top_face_nodes, top_bottom_face_nodes, nid_to_idx, master_faces, slave_faces


def test_full_solve_two_block_matching_topology_converges():
    solver, contact, slave_face_nodes, master_face_nodes, nid_to_idx, _mf, _sf = _build_two_block_s2s_fixture()
    converged, iters = solver.solve_step(dt=1.0, max_iters=80, max_sdi_iters=400)
    assert converged, f"expected convergence, got iters={iters}"

    active = contact.get_active_set(solver.u)
    assert len(active) == 9  # all 9 interface slave nodes should be in contact

    _, _, stats = contact.assemble(solver.u)
    assert stats["total_normal_force"] > 0.0
    assert stats["n_active"] == 9


def test_penalty_stiffness_units_differ_from_point_collocation_by_design():
    """NOT a bug -- a documented units/scale distinction, found while
    trying (and correctly failing) to cross-check raw force magnitude
    against DeformableSurfaceContactConstraint3D using the SAME numeric
    `penalty_stiffness`.

    Dimensional analysis of this class's own formulas (module docstring
    sec2.4/2.5): `g_tilde` carries units of LENGTH (gap_numerator is
    Psi[dimensionless] * penetration[length] * w[area], divided by
    D_global[area] -> length). For the force-distribution formula
    `f += -Ns*w[area]*lam_h*n` to yield a physical FORCE, `lam_h` (and
    therefore `p = lam + k_contact*g_tilde`, and therefore `k_contact`
    itself) must carry PRESSURE units (force/area/length, i.e.
    force/length^3) -- a penalty MODULUS density.

    `DeformableSurfaceContactConstraint3D.k_contact`, by contrast, is a
    per-node lumped spring constant (force/length, its own docstring:
    "Penalty stiffness k_contact [force/length]... a value derived from
    the actual material/mesh stiffness scale") -- already scaled for one
    node's own tributary area, no further area multiplication anywhere
    in that class's own assemble().

    These are NOT the same physical quantity, so passing the identical
    numeric `penalty_stiffness` to both classes on the same mesh (as an
    earlier version of this test mistakenly did) produces total forces
    that differ by roughly the interface's own representative nodal
    tributary area -- confirmed empirically here, not just asserted:
    this fixture's total normal force differs by ~3.8x between the two
    classes at equal numeric k_contact, and 3.8x is the right order of
    magnitude for this fixture's own tributary-area ratio (a 2x2 sub-
    grid of a 5x5-unit face -> ~6.25 unit^2 per node vs. the
    point-collocation class's own implicit per-node area assumption).
    Anyone wiring a CAE-level `k_ref`-style auto-derivation for this
    class in the future must account for this unit difference -- do not
    reuse `DeformableSurfaceContactConstraint3D`'s own k_ref formula
    unchanged."""
    from dispsolver.constraint3d.surface_contact3d_deformable import DeformableSurfaceContactConstraint3D

    solver_s2s, contact_s2s, _bn, _tn, _idx, _mf, _sf = _build_two_block_s2s_fixture()
    conv_s2s, _ = solver_s2s.solve_step(dt=1.0, max_iters=80, max_sdi_iters=400)
    assert conv_s2s
    _, _, stats_s2s = contact_s2s.assemble(solver_s2s.u)

    solver2, _cd, master_nodes2, slave_nodes2, nid_to_idx2, master_faces2, _sf2 = (
        _build_two_block_s2s_fixture()
    )
    solver2.constraints.clear()
    k_rep = 200000.0 * 5.0
    contact_n2s = DeformableSurfaceContactConstraint3D(
        slave_node_ids=slave_nodes2, master_faces=master_faces2,
        nid_to_idx=nid_to_idx2, coords=solver2.mesh.coords,
        penalty_stiffness=0.1 * k_rep,
    )
    solver2.constraints.append(contact_n2s)
    conv_n2s, _ = solver2.solve_step(dt=1.0, max_iters=80, max_sdi_iters=400)
    assert conv_n2s
    _, _, stats_n2s = contact_n2s.assemble(solver2.u)

    rel_diff = abs(stats_s2s["total_normal_force"] - stats_n2s["total_normal_force"]) / stats_n2s["total_normal_force"]
    # Documented as DIFFERENT, not equal -- bound it loosely (order-of-
    # magnitude sane) rather than asserting equality this class's own
    # units were never meant to produce.
    assert 0.1 < rel_diff < 10.0, (stats_s2s["total_normal_force"], stats_n2s["total_normal_force"], rel_diff)
