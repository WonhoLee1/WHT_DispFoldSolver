"""CPE4H on a linear elastic material is the Q1P0 mixed element (constant pressure, standard deviatoric part).

The reference below is the textbook definition written independently in numpy: deviatoric energy G e:e at the
2x2 Gauss points (e is the 3D deviatoric strain of the plane-strain state, so e_zz = -eps_vol/3) plus the
volumetric energy K eps_vol_avg^2 V / 2 with the ELEMENT-AVERAGE volumetric strain. The element kernel was
rewritten to take any material; this keeps its linear elastic answer unchanged.
"""
import numpy as np
import pytest

from dispsolver.element2d.cpe4h_numba import compute_cpe4h_element_numba


def q1p0_reference(coords, u, E, nu):
    G, K = E / (2 * (1 + nu)), E / (3 * (1 - 2 * nu))
    g = 1 / np.sqrt(3)
    pts = [(-g, -g), (g, -g), (g, g), (-g, g)]
    f, Kdev, vol_vec, V = np.zeros(8), np.zeros((8, 8)), np.zeros(8), 0.0
    for xi, eta in pts:
        dN = 0.25 * np.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                              [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
        J = dN @ coords
        dNx = np.linalg.solve(J, dN)
        w = np.linalg.det(J)
        B = np.zeros((3, 8))
        B[0, 0::2], B[1, 1::2], B[2, 0::2], B[2, 1::2] = dNx[0], dNx[1], dNx[1], dNx[0]
        bv = B[0] + B[1]
        vol_vec += bv * w
        V += w
        # strain components that carry deviatoric energy: xx, yy, zz, xy (engineering gamma/2 weight)
        Bd = np.vstack([B[0] - bv / 3, B[1] - bv / 3, -bv / 3, B[2]])
        W = np.diag([1.0, 1.0, 1.0, 0.5])
        Kdev += 2 * G * Bd.T @ W @ Bd * w
        f += 2 * G * Bd.T @ W @ (Bd @ u) * w
    eps_avg = vol_vec @ u / V
    f += K * eps_avg * vol_vec
    return f, Kdev + K / V * np.outer(vol_vec, vol_vec)


@pytest.mark.parametrize("E, nu", [(1000.0, 0.3), (0.05, 0.499), (4000.0, 0.45)])
def test_linear_elastic_cpe4h_is_q1p0(E, nu):
    rng = np.random.default_rng(11)
    for _ in range(8):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float) + rng.normal(0, 0.12, (4, 2))
        u = rng.normal(0, 0.02, 8)
        props = np.zeros(64)
        props[0], props[1] = E, nu
        f, K, err = compute_cpe4h_element_numba(coords, u, 0, props, np.zeros((4, 8)), 0.0, None)
        f_ref, K_ref = q1p0_reference(coords, u, E, nu)
        assert err == 0
        np.testing.assert_allclose(f, f_ref, rtol=1e-9, atol=1e-12 * max(1.0, np.abs(f_ref).max()))
        np.testing.assert_allclose(K, K_ref, rtol=1e-9, atol=1e-12 * np.abs(K_ref).max())
