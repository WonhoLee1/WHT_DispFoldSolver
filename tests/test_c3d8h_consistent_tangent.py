"""C3D8H tangent is the Jacobian of C3D8H's own force, including the element-frame derivative.

f_g = R(u) f_loc(u_loc(u)); the old tangent was R K_loc R^T, which freezes the edge-based frame R(u). In a
mode where that frame turns differently from the material (a soft layer rocking between stiff faces)
the dropped dR/du terms are stiffness-sized: measured 98% off for a sheared PSA-like cube, so Newton took
~35 iterations (line search stuck at 1/64 steps) instead of 4. A random direction does not excite such a
mode (it stays within a few percent), so the rocking direction is what discriminates.
"""
import numpy as np
import pytest

from dispsolver.element3d.c3d8_hybrid_numba import compute_c3d8_hybrid_element_umat_numba
from dispsolver.material3d.numba_materials import MAT_HYPERELASTIC_NEOHOOKEAN

COORDS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64)
PROPS = np.zeros(36)
PROPS[0], PROPS[1] = 0.1, 0.01   # Neo-Hookean C10, D1 (the PSA of the fold)


def _eval(u):
    f, K, err = compute_c3d8_hybrid_element_umat_numba(
        COORDS, u, MAT_HYPERELASTIC_NEOHOOKEAN, PROPS, np.zeros((8, 0)))
    assert err == 0
    return f, K


def _rocking():
    """Top face tilts about y (z of the top nodes +-, y +-), bottom face untouched."""
    d = np.zeros(24)
    for n, sz, sy in ((4, 1, -1), (5, -1, 1), (6, -1, -1), (7, 1, 1)):
        d[3 * n + 2] = 0.9 * sz
        d[3 * n + 1] = 0.2 * sy
    return d / np.linalg.norm(d)


def _shear(amp):
    u = np.zeros(24)
    for n in range(4, 8):
        u[3 * n] = amp
    return u


def _relerr(u, d, h=1e-6):
    _, K = _eval(u)
    fd = (_eval(u + h * d)[0] - _eval(u - h * d)[0]) / (2 * h)
    return np.linalg.norm(K @ d - fd) / np.linalg.norm(fd)


@pytest.mark.parametrize("amp,bound", [(0.005, 5e-3), (0.02, 1.5e-2), (0.05, 3e-2)])
def test_tangent_along_the_rocking_mode(amp, bound):
    assert _relerr(_shear(amp), _rocking()) < bound


def test_tangent_at_zero_state_is_exact():
    rng = np.random.default_rng(0)
    for _ in range(3):
        d = rng.standard_normal(24)
        assert _relerr(np.zeros(24), d / np.linalg.norm(d)) < 1e-6
    assert _relerr(np.zeros(24), _rocking()) < 1e-6


def test_tangent_at_a_rotated_sheared_state():
    th = np.radians(35.0)
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    u = _shear(0.02)
    d = np.zeros(24)
    base = _rocking()
    for i in range(8):
        x = COORDS[i] + u[3 * i:3 * i + 3]
        u[3 * i:3 * i + 3] = Rz @ x - COORDS[i]
        d[3 * i:3 * i + 3] = Rz @ base[3 * i:3 * i + 3]
    assert _relerr(u, d) < 1.5e-2
