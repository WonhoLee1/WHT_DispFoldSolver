"""Contact geometry follows the sliding formulation and the step's NLGEOM.

Abaqus/Standard (local copies: benchmark_element/reference_abaqus_docs/
ctc_contactpairform_std.txt, ctc_contactpair_std.txt):
  * *CONTACT PAIR without SMALL SLIDING is finite sliding (re-tracked);
  * SMALL SLIDING + NLGEOM: pairing fixed from the initial configuration,
    contact plane rotates with the master surface;
  * SMALL SLIDING without NLGEOM: infinitesimal sliding and rotation --
    reference geometry, gap linear in u.
Before 2026-10-05 the reader dropped SMALL SLIDING (every deck pair ran small
sliding), PENALTY/NITSCHE ignored `sliding`, and no contact class read NLGEOM.
"""
import types
import warnings

import numpy as np
import pytest

from dispsolver.constraint3d.contact_geometry import contact_mode
from dispsolver.constraint3d.nitsche_contact3d import NitscheContactConstraint3D
from dispsolver.constraint3d.surface_contact3d_deformable import DeformableSurfaceContactConstraint3D

# two master quads side by side in z = 0, normal +z; master body below (z < 0)
NODES = {1: (0, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0), 4: (0, 1, 0), 5: (1, 1, 0), 6: (2, 1, 0),
         7: (0.5, 0.5, 0.001)}
MASTER = [(1, 2, 5, 4), (2, 3, 6, 5)]
INTERIOR = {MASTER[0]: np.array([0.5, 0.5, -1.0]), MASTER[1]: np.array([1.5, 0.5, -1.0])}


def _setup():
    ids = sorted(NODES)
    nid_to_idx = {n: i for i, n in enumerate(ids)}
    coords = np.array([NODES[n] for n in ids], dtype=float)
    return nid_to_idx, coords


def _penalty(sliding):
    nid_to_idx, coords = _setup()
    return DeformableSurfaceContactConstraint3D([7], MASTER, nid_to_idx, coords, penalty_stiffness=1e3,
                                                master_interior=INTERIOR, sliding=sliding), nid_to_idx, coords


def _solver(nlgeom):
    return types.SimpleNamespace(nlgeom=nlgeom)


def test_mode_table():
    assert contact_mode("FINITE", True) == contact_mode("FINITE", False) == "finite"
    assert contact_mode("SMALL", True) == "small"
    assert contact_mode("SMALL", False) == "linear"


@pytest.mark.parametrize("nlgeom", [True, False])
def test_bind_solver_sets_mode_from_nlgeom(nlgeom):
    c, _, _ = _penalty("SMALL")
    c.bind_solver(_solver(nlgeom))
    assert c.mode == ("small" if nlgeom else "linear")
    c, _, _ = _penalty("FINITE")
    c.bind_solver(_solver(nlgeom))
    assert c.mode == "finite"


def test_finite_sliding_retracks_the_master_face():
    """Slave slides from over face A to over face B, slightly penetrating."""
    u = np.zeros(3 * len(NODES))
    u[3 * 6: 3 * 6 + 3] = [1.0, 0.0, -0.011]           # node 7 -> (1.5, 0.5, -0.01)
    for sliding, face in (("FINITE", MASTER[1]), ("SMALL", MASTER[0])):
        c, _, _ = _penalty(sliding)
        c.bind_solver(_solver(True))
        c.reproject_deformed(u)
        assert tuple(c.pairs[0][1]) == face
    c, _, _ = _penalty("FINITE")
    c.bind_solver(_solver(True))
    c.reproject_deformed(u)
    f, _, st = c.assemble(u)
    assert st["n_active"] == 1
    assert np.isclose(f[3 * 6 + 2], -1e3 * 0.01)      # f_int = -k*pen*n on the slave


def _rotate_all(coords, deg, axis=1):
    th = np.radians(deg)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) if axis == 1 else np.eye(3)
    return (coords @ R.T - coords).ravel()


def test_small_sliding_gap_is_objective_under_nlgeom():
    c, nid_to_idx, coords = _penalty("SMALL")
    c.bind_solver(_solver(True))
    u = _rotate_all(coords, 30.0)
    s_nid, face, xi, eta, sign = c.pairs[0]
    g0 = c._current_gap_and_normal(np.zeros_like(u), s_nid, face, xi, eta, sign)[0]
    g = c._current_gap_and_normal(u, s_nid, face, xi, eta, sign)[0]
    assert np.isclose(g, g0, atol=1e-12)


def test_small_sliding_without_nlgeom_is_linear_in_u():
    c, nid_to_idx, coords = _penalty("SMALL")
    c.bind_solver(_solver(False))
    rng = np.random.default_rng(3)
    u = 1e-3 * rng.standard_normal(3 * len(NODES))
    u[3 * 6 + 2] = -0.01                                 # keep the slave penetrating
    # gap(u) = g0 + a.u exactly (reference normal)
    s_nid, face, xi, eta, sign = c.pairs[0]
    g = lambda v: c._current_gap_and_normal(v, s_nid, face, xi, eta, sign)[0]
    g0 = g(np.zeros_like(u))
    assert np.isclose(g(2 * u) - g0, 2 * (g(u) - g0), rtol=1e-12, atol=1e-15)
    # ... and the linear-mode tangent has no normal-rotation terms: rigid
    # rotation of the whole pair changes the gap (small-strain contact).
    rot = _rotate_all(coords, 30.0)
    assert not np.isclose(g(rot), g0, atol=1e-6)


def test_linear_mode_tangent_matches_finite_difference():
    c, _, _ = _penalty("SMALL")
    c.bind_solver(_solver(False))
    rng = np.random.default_rng(5)
    u = 1e-3 * rng.standard_normal(3 * len(NODES))
    u[3 * 6 + 2] = -0.01
    f0, (r, cc, d), _ = c.assemble(u)
    K = np.zeros((len(u), len(u)))
    np.add.at(K, (r, cc), d)
    h = 1e-7
    Kfd = np.zeros_like(K)
    for j in range(len(u)):
        up = u.copy(); up[j] += h
        um = u.copy(); um[j] -= h
        Kfd[:, j] = (c.assemble(up)[0] - c.assemble(um)[0]) / (2 * h)
    assert np.allclose(K, Kfd, atol=1e-5 * np.abs(K).max())


def test_nitsche_finite_retracks_quadrature_points():
    ids = sorted(NODES) + [8, 9, 10]
    extra = {8: (0.75, 0.25, 0.001), 9: (0.75, 0.75, 0.001), 10: (0.25, 0.75, 0.001)}
    allnodes = {**NODES, **extra}
    allnodes[7] = (0.25, 0.25, 0.001)
    nid_to_idx = {n: i for i, n in enumerate(ids)}
    coords = np.array([allnodes[n] for n in ids], dtype=float)
    slave = [(7, 8, 9, 10)]
    c = NitscheContactConstraint3D(slave, MASTER, nid_to_idx, coords, gamma=1e3, consistency=False,
                                   master_interior=INTERIOR, sliding="FINITE")
    c.bind_solver(_solver(True))
    u = np.zeros(3 * len(ids))
    for n in (7, 8, 9, 10):
        u[3 * nid_to_idx[n]] = 1.0                       # slide the slave face onto face B
    c.reproject_deformed(u)
    assert all(qp["m_face"] == MASTER[1] for qp in c.pairs[0]["quad_points"])


def test_reader_contact_pair_sliding():
    from dispsolver.io.abaqus_reader import _parse_source
    base = """*NODE
1,0,0,0
2,1,0,0
3,1,1,0
4,0,1,0
5,0,0,1
6,1,0,1
7,1,1,1
8,0,1,1
*ELEMENT, TYPE=C3D8, ELSET=E
1,1,2,3,4,5,6,7,8
*SURFACE, NAME=A
E, S1
*SURFACE, NAME=B
E, S2
*SURFACE INTERACTION, NAME=I
*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD
"""
    for line, want in (("*CONTACT PAIR, INTERACTION=I\nA, B\n", "FINITE"),
                       ("*CONTACT PAIR, INTERACTION=I, SMALL SLIDING\nA, B\n", "SMALL")):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, model = _parse_source(base + line, dim=3)
        pairs = [o for o in model.interactions.values() if hasattr(o, "sliding")]
        pairs += [getattr(e, "entity", e) for st in model.steps.values() for e in st.interactions.values()
                  if hasattr(getattr(e, "entity", e), "sliding")]
        assert pairs and all(p.sliding == want for p in pairs)


@pytest.mark.parametrize("nlgeom", [True, False])
def test_small_sliding_mortar_follows_nlgeom(nlgeom):
    """Small-sliding mortar: the contact plane rotates with the master under
    NLGEOM (gap invariant to a rigid rotation of the pair) and stays in the
    reference configuration without it. It used the reference normal always."""
    from dispsolver.constraint3d.dual_mortar_clipping3d import DualMortarClippedContactConstraint3D
    nodes = {1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 1, 0), 4: (0, 1, 0),
             5: (0, 0, -0.01), 6: (1, 0, -0.01), 7: (1, 1, -0.01), 8: (0, 1, -0.01)}
    ids = sorted(nodes)
    nid_to_idx = {n: i for i, n in enumerate(ids)}
    coords = np.array([nodes[n] for n in ids], dtype=float)
    c = DualMortarClippedContactConstraint3D(slave_faces=[(5, 6, 7, 8)], master_faces=[(1, 2, 3, 4)],
                                             nid_to_idx=nid_to_idx, coords=coords, penalty_stiffness=1e3,
                                             mortar_type="DUAL")
    c.bind_solver(_solver(nlgeom))
    g0 = c._compute_g_tilde(np.zeros(coords.size))
    g = c._compute_g_tilde(_rotate_all(coords, 30.0))
    same = all(np.isclose(g[k], g0[k], atol=1e-12) for k in g0)
    assert same == nlgeom
