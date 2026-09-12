"""
test_neohookean_3d.py
======================
Verifies MAT_HYPERELASTIC_NEOHOOKEAN in dispsolver/material3d/numba_materials.py
is a genuine finite-strain compressible Neo-Hookean law, not the previous
placeholder (S = C_mat @ E_voigt, bit-identical to MAT_CUSTOM_ELASTIC at
any strain -- fixed 2026-09-13, .omc/plans/3d_capability_buildout_20260913.md T3).
"""

import numpy as np
import pytest

from dispsolver.material3d.numba_materials import (
    material_dispatch_3d,
    MAT_HYPERELASTIC_NEOHOOKEAN,
    MAT_CUSTOM_ELASTIC,
)

E, NU = 200000.0, 0.3
_LAM = (E * NU) / ((1.0 + NU) * (1.0 - 2.0 * NU))
_MU = E / (2.0 * (1.0 + NU))
_C_MAT = np.array([
    [_LAM + 2 * _MU, _LAM, _LAM, 0, 0, 0],
    [_LAM, _LAM + 2 * _MU, _LAM, 0, 0, 0],
    [_LAM, _LAM, _LAM + 2 * _MU, 0, 0, 0],
    [0, 0, 0, _MU, 0, 0],
    [0, 0, 0, 0, _MU, 0],
    [0, 0, 0, 0, 0, _MU],
])
_PROPS_CUSTOM = np.zeros(36)
_PROPS_CUSTOM[:36] = _C_MAT.ravel()[:36]
_PROPS_NH = np.zeros(36)
_PROPS_NH[0] = E
_PROPS_NH[1] = NU
_SDV = np.zeros(0)
_F_ID = np.eye(3)


def test_small_strain_recovers_linear_elasticity():
    """At small strain, Neo-Hookean should closely match the linear law
    it's parameterized from (not required to be exact -- it's a
    different model -- but should agree to a few tenths of a percent)."""
    eps = np.array([1e-4, -3e-5, -3e-5, 2e-5, 0.0, 0.0])
    s_custom, _, _, _ = material_dispatch_3d(MAT_CUSTOM_ELASTIC, _PROPS_CUSTOM, _SDV, eps, _F_ID, 1.0, 1.0)
    s_nh, _, _, _ = material_dispatch_3d(MAT_HYPERELASTIC_NEOHOOKEAN, _PROPS_NH, _SDV, eps, _F_ID, 1.0, 1.0)
    rel = np.max(np.abs(s_nh - s_custom)) / max(np.max(np.abs(s_custom)), 1e-10)
    assert rel < 0.01, f"small-strain relative diff too large: {rel:.4e}"


def test_large_strain_diverges_from_linear_elasticity():
    """The previous placeholder was bit-identical to linear elasticity at
    ANY strain -- this is the regression check that it no longer is."""
    lam_stretch = 1.3
    E11 = 0.5 * (lam_stretch ** 2 - 1.0)
    lam_lat = 1.0 / np.sqrt(lam_stretch)
    E22 = E33 = 0.5 * (lam_lat ** 2 - 1.0)
    eps = np.array([E11, E22, E33, 0.0, 0.0, 0.0])
    s_custom, _, _, _ = material_dispatch_3d(MAT_CUSTOM_ELASTIC, _PROPS_CUSTOM, _SDV, eps, _F_ID, 1.0, 1.0)
    s_nh, _, _, _ = material_dispatch_3d(MAT_HYPERELASTIC_NEOHOOKEAN, _PROPS_NH, _SDV, eps, _F_ID, 1.0, 1.0)
    rel = np.max(np.abs(s_nh - s_custom)) / max(np.max(np.abs(s_custom)), 1e-10)
    assert rel > 0.1, f"expected large divergence from linear elasticity at 30% stretch, got {rel:.4e}"


def test_tangent_matches_fd_jacobian_at_large_strain():
    """C8-contract style check (AGENTS.md §4.16): the returned tangent
    must be the FD Jacobian of the returned stress, independently
    recomputed here (not reusing the material routine's own internal FD
    loop)."""
    lam_stretch = 1.3
    E11 = 0.5 * (lam_stretch ** 2 - 1.0)
    lam_lat = 1.0 / np.sqrt(lam_stretch)
    E22 = E33 = 0.5 * (lam_lat ** 2 - 1.0)
    eps = np.array([E11, E22, E33, 0.0, 0.0, 0.0])

    _, C_analytic, _, _ = material_dispatch_3d(MAT_HYPERELASTIC_NEOHOOKEAN, _PROPS_NH, _SDV, eps, _F_ID, 1.0, 1.0)

    h = 1e-6
    fd = np.zeros((6, 6))
    for j in range(6):
        ep = eps.copy(); ep[j] += h
        em = eps.copy(); em[j] -= h
        sp, _, _, _ = material_dispatch_3d(MAT_HYPERELASTIC_NEOHOOKEAN, _PROPS_NH, _SDV, ep, _F_ID, 1.0, 1.0)
        sm, _, _, _ = material_dispatch_3d(MAT_HYPERELASTIC_NEOHOOKEAN, _PROPS_NH, _SDV, em, _F_ID, 1.0, 1.0)
        fd[:, j] = (sp - sm) / (2.0 * h)

    err = np.max(np.abs(fd - C_analytic)) / max(np.max(np.abs(fd)), 1e-10)
    assert err < 1e-6, f"tangent vs independent FD Jacobian relative error too large: {err:.3e}"
