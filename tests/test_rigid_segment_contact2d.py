"""RotatingRigidSegmentContact2D: gap sign, segment limits, rotation, tangent == FD Jacobian."""
import numpy as np
import pytest
from scipy.sparse import lil_matrix

from dispsolver.constraint.rigid_segment_contact2d import RotatingRigidSegmentContact2D

# right plate of the ex12 fold: pivot (1.4, 0), top face y=0 for x in [7.5, 40], normal +y
PIVOT = (1.4, 0.0)


def _contact(xs, k=1.0e4):
    coords = np.array([[x, 0.0] for x in xs])
    nid = {i + 1: i for i in range(len(xs))}
    c = RotatingRigidSegmentContact2D(list(nid), nid, coords, pivot=PIVOT, tangent=(1.0, 0.0),
                                      normal=(0.0, 1.0), s_range=(7.5 - 1.4, 40.0 - 1.4),
                                      penalty_stiffness=k)
    return c, coords


def _assemble(c, u):
    f = np.zeros_like(u)
    K = lil_matrix((len(u), len(u)))
    c.apply_penalty(u, f, K)
    return f, K.toarray()


def test_no_force_when_separated_or_touching():
    c, _ = _contact([10.0, 20.0])
    u = np.array([0.0, 0.01, 0.0, 0.0])          # one above, one exactly on the face
    f, K = _assemble(c, u)
    assert not f.any() and not K.any()


def test_penetration_pushes_out_along_normal():
    c, _ = _contact([10.0])
    f, K = _assemble(c, np.array([0.0, -0.002]))
    assert f[1] == pytest.approx(1.0e4 * -0.002)   # f_int = k g n, so R = -f_int pushes +y
    assert f[0] == 0.0 and K[1, 1] == pytest.approx(1.0e4) and K[0, 0] == 0.0


def test_inactive_outside_the_segment():
    c, _ = _contact([5.0, 41.0])                   # beyond the inner edge 7.5 / outer edge 40
    f, _ = _assemble(c, np.array([0.0, -0.1, 0.0, -0.1]))
    assert not f.any()


def test_follows_the_plate_rotation():
    th = np.radians(30.0)
    c, coords = _contact([20.0])
    c.set_rotation(th)
    # a node exactly on the rotated face: no force; pushed 0.001 into it: force along the rotated normal
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    on_face = np.array(PIVOT) + R @ (coords[0] - np.array(PIVOT))
    n = R @ np.array([0.0, 1.0])
    f, _ = _assemble(c, on_face - coords[0])
    assert np.allclose(f, 0.0, atol=1e-9)
    f, _ = _assemble(c, on_face - 0.001 * n - coords[0])
    assert np.allclose(f, 1.0e4 * -0.001 * n, rtol=1e-9)


def test_tangent_is_the_jacobian_of_the_force():
    rng = np.random.default_rng(0)
    c, coords = _contact([9.0, 15.0, 30.0])
    c.set_rotation(np.radians(37.0))
    u = rng.normal(scale=0.3, size=6)
    # put every node clearly inside the plate so the active set is fixed under the perturbation
    for j in range(3):
        th = c.theta
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        x_face = np.array(PIVOT) + R @ (coords[j] - np.array(PIVOT))
        u[2 * j:2 * j + 2] = x_face - 0.01 * (R @ np.array([0.0, 1.0])) - coords[j] + 0.05 * (R @ np.array([1.0, 0.0]))
    f0, K = _assemble(c, u)
    h = 1e-7
    J = np.zeros((6, 6))
    for k in range(6):
        up = u.copy(); up[k] += h
        um = u.copy(); um[k] -= h
        J[:, k] = (_assemble(c, up)[0] - _assemble(c, um)[0]) / (2 * h)
    assert np.allclose(K, J, rtol=1e-6, atol=1e-6)
    assert c.stats(u)["n_active"] == 3 and c.stats(u)["max_penetration"] == pytest.approx(0.01)


def test_softened_law_is_c1_and_its_tangent_is_the_jacobian():
    g0 = 1e-3
    c, coords = _contact([10.0, 15.0, 20.0, 25.0])
    c.softening_gap = g0
    p, dp = c._pressure(np.array([1e-6, -1e-12, -(1 - 1e-9) * g0, -(1 + 1e-9) * g0, -0.01]))
    assert p[0] == 0.0 and p[1] < 1e-12 and abs(dp[1]) < 1e-3     # no load and no slope at touch
    assert abs(p[2] - p[3]) < 1e-6 and abs(dp[2] - dp[3]) < 1e-2  # continuous value and slope at -g0
    # nodes in the band (two), touching, and penetrated: the tangent is the FD Jacobian
    u = np.zeros(8)
    u[1], u[3], u[5], u[7] = 0.4 * g0, -0.3 * g0, -0.8 * g0, -0.01
    f0, K = _assemble(c, u)
    h = 1e-9
    J = np.zeros((8, 8))
    for k in range(8):
        up = u.copy(); up[k] += h
        um = u.copy(); um[k] -= h
        J[:, k] = (_assemble(c, up)[0] - _assemble(c, um)[0]) / (2 * h)
    assert np.allclose(K, J, rtol=1e-5, atol=1e-3)
    assert f0[1] == 0 and 0 > f0[3] > f0[5] > f0[7]              # open: none; grows with overclosure


def test_fillet_rounds_the_inner_edge_with_a_consistent_tangent():
    r = 0.5
    c, coords = _contact([7.5 + r - 1e-9, 7.5 + r + 1e-9, 7.6])   # just either side of the arc/flat joint, and on the arc
    c.fillet_radius = r
    c.softening_gap = 1e-3
    c.set_rotation(np.radians(20.0))
    th = c.theta
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    piv = np.array(PIVOT)
    # rigidly carry all nodes with the plate, then push 0.01 mm into the face along the local normal
    u = np.zeros(6)
    for j in range(3):
        x_rot = piv + R @ (coords[j] - piv)
        u[2 * j:2 * j + 2] = x_rot - coords[j]
    g, n, curv, on = c._geometry(u)
    assert on.all() and abs(g[0] - g[1]) < 1e-6                  # gap continuous across the joint
    assert curv[2] > 0 and curv[1] == 0.0                         # node 3 on the arc, node 2 on the flat
    # on the arc the face drops below the flat top, so a node carried rigidly is OPEN there
    assert g[2] > 0
    u[4:6] -= (g[2] + 0.01) * n[2]                                # push node 3 0.01 into the fillet
    f0, K = _assemble(c, u)
    h = 1e-8
    J = np.zeros((6, 6))
    for k in range(6):
        up = u.copy(); up[k] += h
        um = u.copy(); um[k] -= h
        J[:, k] = (_assemble(c, up)[0] - _assemble(c, um)[0]) / (2 * h)
    assert np.allclose(K[4:6, 4:6], J[4:6, 4:6], rtol=1e-5, atol=1e-3)
    assert np.dot(-f0[4:6], n[2]) > 0                             # residual pushes it back out along the arc normal
