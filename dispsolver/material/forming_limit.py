# -*- coding: utf-8 -*-
"""forming_limit.py: forming-limit diagram (FLD) necking criterion (post-processing).

Keeler-Brazier FLC for sheet steel plus tabular Abaqus-style FLC input:

    FLC0 = (23.3 + 14.13*t) * n / 0.21          (t = thickness mm, n = hardening exponent)
    left side (minor < 0):  major = FLC0 - minor   (draw, slope -1 in % space)
    right side (minor > 0): major = FLC0 * (1 + minor)  (stretch, Keeler)

margin = major_limit(minor) - major_strain; negative => predicted necking.
Tabular form reuses dependence_table 1D interp (minor -> major limit).

This is an output-only criterion (same class as damage initiation): it never
alters stress or tangent. Solver coupling (per-GP evaluation + flag SDV) is
a documented Wave-3 extension, not part of this unit.
"""

from __future__ import annotations

import numpy as np

from dispsolver.material.dependence_table import eval_table_1d_numba


def keeler_flc0(thickness_mm: float, n_hardening: float) -> float:
    """Plane-strain FLC0 intercept (engineering strain, Keeler-Brazier)."""
    return (23.3 + 14.13 * float(thickness_mm)) * float(n_hardening) / 0.21 / 100.0


def keeler_major_limit(minor: float, flc0: float) -> float:
    """Major limit strain at given minor strain (engineering strains)."""
    if minor <= 0.0:
        return flc0 - minor
    return flc0 * (1.0 + minor)


def fld_margin(major: float, minor: float, flc0: float) -> float:
    """Safety margin to the FLC (negative => necking predicted)."""
    return keeler_major_limit(minor, flc0) - major


def tabular_major_limit(minor: float, minor_tab: np.ndarray,
                        major_tab: np.ndarray) -> float:
    """Abaqus-style tabular FLC lookup (linear + clamp)."""
    return float(eval_table_1d_numba(np.asarray(minor_tab, dtype=np.float64),
                                     np.asarray(major_tab, dtype=np.float64),
                                     float(minor)))
