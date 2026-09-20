# -*- coding: utf-8 -*-
"""Tests for the FLD necking criterion (Keeler-Brazier + tabular)."""
import numpy as np

from dispsolver.material.forming_limit import (
    keeler_flc0,
    keeler_major_limit,
    fld_margin,
    tabular_major_limit,
)


def test_flc0_reference():
    # t = 1.0 mm, n = 0.21 -> (23.3 + 14.13) * 0.21/0.21 / 100 = 0.3743.
    np.testing.assert_allclose(keeler_flc0(1.0, 0.21), 0.3743, rtol=1e-12)


def test_plane_strain_minimum():
    f0 = keeler_flc0(1.0, 0.21)
    assert keeler_major_limit(0.0, f0) == f0
    assert keeler_major_limit(-0.1, f0) > f0
    assert keeler_major_limit(0.1, f0) > f0


def test_draw_side_slope():
    f0 = 0.3
    np.testing.assert_allclose(keeler_major_limit(-0.1, f0) - f0, 0.1, rtol=1e-12)


def test_margin_sign():
    f0 = 0.3
    assert fld_margin(0.1, 0.0, f0) > 0.0
    assert fld_margin(0.5, 0.0, f0) < 0.0
    assert fld_margin(f0, 0.0, f0) == 0.0


def test_tabular_matches_keeler_nodes():
    minor = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    f0 = 0.3
    major = np.array([keeler_major_limit(m, f0) for m in minor])
    for m in (-0.15, 0.0, 0.07):
        np.testing.assert_allclose(tabular_major_limit(m, minor, major),
                                   keeler_major_limit(m, f0), rtol=1e-12)
    assert tabular_major_limit(-1.0, minor, major) == major[0]
    assert tabular_major_limit(1.0, minor, major) == major[-1]
