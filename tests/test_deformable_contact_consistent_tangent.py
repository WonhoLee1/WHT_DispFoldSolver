"""
test_deformable_contact_consistent_tangent.py
===============================================
`DeformableSurfaceContactConstraint3D` stiffness must be the Jacobian of
its own contact force (AGENTS.md sec4.15's standing test, applied to the
contact constraint): K vs a central finite difference of f, on a TILTED,
warped master face with a slave that has slid off its projection point,
for the frictionless, stick and slip branches.

Before 2026-10-01 the block was a modified-Newton approximation (normal
held fixed, slip-branch mu*p coupling dropped, friction assembled with
the projection weights instead of the anchor's own): 3-41% off FD on the
fixture below. The default now makes the NORMAL part exact and keeps the
symmetric friction block (the exact friction block is opt-in via
consistent_friction_tangent; it made the Hertz roller SLIDE step diverge).
"""

import numpy as np
import pytest

from dispsolver.constraint3d.surface_contact3d_deformable import DeformableSurfaceContactConstraint3D


def _fixture(mu, consistent=True, k=1.0e4, ratio=0.1, consistent_friction=True):
    rng = np.random.default_rng(3)
    # master face: unit square, tilted and warped; slave just below its interior
    base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    c, s = np.cos(0.3), np.sin(0.3)
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    master = base @ R.T + 0.05 * rng.standard_normal((4, 3))
    slave = (0.35 * master[0] + 0.25 * master[1] + 0.2 * master[2] + 0.2 * master[3]) + np.array([0.0, 0.0, 0.02])
    coords = np.vstack([slave, master])
    nid_to_idx = {i + 1: i for i in range(5)}
    con = DeformableSurfaceContactConstraint3D(
        slave_node_ids=[1], master_faces=[(2, 3, 4, 5)], nid_to_idx=nid_to_idx, coords=coords,
        penalty_stiffness=k, friction_coefficient=mu, tangential_stiffness_ratio=ratio,
        consistent_tangent=consistent, consistent_friction_tangent=consistent_friction,
    )
    assert len(con.pairs) == 1
    return con


def _penetrated_state(con, slide):
    """Push the slave through the face along the face normal and slide it
    tangentially by `slide` (relative to its frozen anchor)."""
    s_nid, m_face, xi, eta, sign = con.pairs[0]
    u = np.zeros(15)
    _gap, n, _N, _si, _mi = con._current_gap_and_normal(u, s_nid, m_face, xi, eta, sign)
    t = np.cross(n, [0.3, 0.8, 0.5])
    t /= np.linalg.norm(t)
    u[0:3] = -0.05 * n + slide * t
    rng = np.random.default_rng(7)
    u[3:] = 0.01 * rng.standard_normal(12)   # deform the master face too
    return u


def _fd_jacobian(con, u, h=1e-7):
    J = np.zeros((15, 15))
    for j in range(15):
        up, um = u.copy(), u.copy()
        up[j] += h
        um[j] -= h
        J[:, j] = (con.assemble(up)[0] - con.assemble(um)[0]) / (2 * h)
    return J


def _K(con, u):
    _f, (r, c, d), _ = con.assemble(u)
    K = np.zeros((15, 15))
    np.add.at(K, (r, c), d)
    return K


@pytest.mark.parametrize("mu,slide,branch", [
    (0.0, 0.0, "frictionless"),
    (0.3, 1e-4, "stick"),
    (0.3, 0.3, "slip"),
])
def test_contact_tangent_is_fd_jacobian(mu, slide, branch):
    con = _fixture(mu)
    u = _penetrated_state(con, slide)
    s_nid, m_face, xi, eta, sign = con.pairs[0]
    xs, q, n, p = con._normal_pressure(u, s_nid, m_face, xi, eta, sign)
    assert p > 0.0
    if mu > 0.0:
        _ft, _Kt, is_slip, *_ = con._friction_force_and_tangent(s_nid, xs, q, n, p)
        assert is_slip == (branch == "slip")
    J = _fd_jacobian(con, u)
    err = np.linalg.norm(_K(con, u) - J) / np.linalg.norm(J)
    assert err < 1e-6, f"{branch}: |K - FD|/|FD| = {err:.3e}"


def test_modified_newton_block_is_measurably_inconsistent():
    """Keeps the reason for the change visible: the legacy block
    (consistent_tangent=False) is NOT the Jacobian on the same state."""
    con = _fixture(0.3, consistent=False)
    u = _penetrated_state(con, 0.3)
    J = _fd_jacobian(con, u)
    err = np.linalg.norm(_K(con, u) - J) / np.linalg.norm(J)
    assert err > 1e-3


def test_consistent_tangent_leaves_force_unchanged():
    for mu in (0.0, 0.3):
        a, b = _fixture(mu, True), _fixture(mu, False)
        u = _penetrated_state(a, 0.3)
        np.testing.assert_array_equal(a.assemble(u)[0], b.assemble(u)[0])


def test_default_friction_block_is_the_symmetric_ktt():
    """Default = exact normal part + the pre-2026-10-01 symmetric Ktt:
    it must equal the exact-normal/frictionless stiffness plus
    w_a*w_b*Ktt on every 3x3 block."""
    con = _fixture(0.3, consistent_friction=False)
    u = _penetrated_state(con, 0.3)
    s_nid, m_face, xi, eta, sign = con.pairs[0]
    xs, q, n, p = con._normal_pressure(u, s_nid, m_face, xi, eta, sign)
    _ft, Ktt, *_ = con._friction_force_and_tangent(s_nid, xs, q, n, p)
    mu = con.mu
    con.mu = 0.0
    K_normal = _K(con, u)
    con.mu = mu
    N = [(1 - xi) * (1 - eta) / 4, (1 + xi) * (1 - eta) / 4, (1 + xi) * (1 + eta) / 4, (1 - xi) * (1 + eta) / 4]
    w = [1.0, -N[0], -N[1], -N[2], -N[3]]
    expected = K_normal + np.kron(np.outer(w, w), Ktt)
    np.testing.assert_allclose(_K(con, u), expected, rtol=0, atol=1e-9 * np.abs(expected).max())


def test_frictionless_normal_part_is_fd_jacobian_by_default():
    con = _fixture(0.0, consistent_friction=False)
    u = _penetrated_state(con, 0.0)
    J = _fd_jacobian(con, u)
    assert np.linalg.norm(_K(con, u) - J) / np.linalg.norm(J) < 1e-6
