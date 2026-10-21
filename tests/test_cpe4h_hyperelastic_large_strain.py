"""CPE4H on a hyperelastic law: f and K are an exact derivative pair at LARGE strain, and the result does not
depend on how the Neo-Hookean constants were spelled.

6456909 made CPE4H pass the Gauss points a Green-Lagrange strain (Hencky -> GL) for non-linear-elastic laws, but
(1) left the tangent without the chain factor dE/d eps, so K was 3-7% off the Jacobian of f at 5% strain and grows
with strain, (2) took the pressure stiffness from props[1], which is D1 for [C10, D1] (K = D1 instead of 2/D1) and
nu for J2, and (3) applied the conversion to J2 as well, moving its residual stress 2.2% off the closed form
(tests/test_plastic_history_2d.py). Small strains hide (1); only a large-strain state with a non-trivial direction
discriminates, which is what this file does.
"""
import numpy as np
import pytest

from dispsolver.element2d.cpe4h_numba import compute_cpe4h_element_numba
from dispsolver.material3d.numba_materials import MAT_HYPERELASTIC_NEOHOOKEAN

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
SKEWED = np.array([[0.0, 0.0], [1.2, 0.1], [1.1, 0.9], [-0.1, 1.0]])

# the same material, three spellings: [C10, D1] (mu = 2 C10, K = 2/D1), [E, nu], [mu, K]
MU, K = 2.0, 2000.0   # E = 6 > 1 so the decoder reads [E, nu] as [E, nu]
E_NU = (9.0 * K * MU / (3.0 * K + MU), (3.0 * K - 2.0 * MU) / (2.0 * (3.0 * K + MU)))
SPELLINGS = {"C10-D1": (1.0, 0.001), "E-nu": E_NU, "mu-K": (MU, K)}


def _props(spelling):
    p = np.zeros(8)
    p[0], p[1] = SPELLINGS[spelling]
    return p


def _eval(coords, u, spelling):
    f, Ke, err = compute_cpe4h_element_numba(
        coords, u, MAT_HYPERELASTIC_NEOHOOKEAN, _props(spelling), np.zeros((4, 4)), 1.0)
    assert err == 0
    return f, Ke


def _homogeneous(coords, a, b, c):
    """u = (F - I) X with F = [[1+a, b], [0, 1+c]]."""
    u = np.zeros(8)
    for i in range(4):
        x, y = coords[i]
        u[2 * i] = a * x + b * y
        u[2 * i + 1] = c * y
    return u


def _fd_relerr(coords, u, d, spelling, h=1e-7):
    _, Ke = _eval(coords, u, spelling)
    fp = _eval(coords, u + h * d, spelling)[0]
    fm = _eval(coords, u - h * d, spelling)[0]
    fd = (fp - fm) / (2 * h)
    return np.linalg.norm(Ke @ d - fd) / np.linalg.norm(fd)


@pytest.mark.parametrize("coords", [SQUARE, SKEWED], ids=["square", "skewed"])
@pytest.mark.parametrize("state", [(-0.15, 0.2, -0.1), (0.25, -0.15, 0.1), (-0.3, 0.3, -0.25)],
                         ids=["compress-shear", "stretch-shear", "deep-compression"])
def test_tangent_is_the_jacobian_at_large_strain(coords, state):
    u = _homogeneous(coords, *state)
    rng = np.random.default_rng(7)
    for _ in range(4):
        d = rng.standard_normal(8)
        assert _fd_relerr(coords, u, d / np.linalg.norm(d), "mu-K") < 2e-5


def test_tangent_is_exact_at_the_reference_state():
    rng = np.random.default_rng(1)
    for _ in range(3):
        d = rng.standard_normal(8)
        assert _fd_relerr(SQUARE, np.zeros(8), d / np.linalg.norm(d), "mu-K") < 1e-6


@pytest.mark.parametrize("spelling", ["C10-D1", "E-nu"])
def test_result_does_not_depend_on_how_the_constants_are_spelled(spelling):
    u = _homogeneous(SKEWED, -0.15, 0.2, -0.1)
    f0, K0 = _eval(SKEWED, u, "mu-K")
    f1, K1 = _eval(SKEWED, u, spelling)
    # [E, nu] -> K = E / (3 (1 - 2 nu)) loses ~1e-9 to cancellation, so compare in norm
    assert np.linalg.norm(f1 - f0) < 1e-8 * np.linalg.norm(f0)
    assert np.linalg.norm(K1 - K0) < 1e-8 * np.linalg.norm(K0)


def test_volumetric_stiffness_is_the_materials_not_a_props_slot():
    """Pure volume change: pressure force = K * eps_vol * B_vol, with K = 2000 (not D1 = 0.001 or nu)."""
    eps = 1e-4
    u = np.zeros(8)
    for i in range(4):
        u[2 * i] = 0.5 * eps * SQUARE[i, 0]
        u[2 * i + 1] = 0.5 * eps * SQUARE[i, 1]
    f, _ = _eval(SQUARE, u, "C10-D1")
    # pressure K*eps_vol acts on a unit edge, half on each of its two nodes
    assert abs(np.abs(f[2] + f[4]) / 2.0 - K * eps * 0.5) < 0.1 * K * eps * 0.5
