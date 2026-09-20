# -*- coding: utf-8 -*-
"""Hill48 tests: isotropy reduction, direction dependence, flow, FD tangent."""
import numpy as np
from dispsolver.material.plasticity_hill import (
    hill48_phi, return_mapping_hill48)
from dispsolver.material.plastic_hardening import compute_von_mises_equivalent

E, NU, Y0, HI = 200000.0, 0.3, 250.0, 1000.0
ISO = (0.5, 0.5, 0.5, 1.5, 1.5, 1.5)


def test_isotropy_reduces_to_mises():
    # compute_von_mises_equivalent expects DEVIATORIC input; Hill phi is
    # pressure-independent, so compare on the deviatoric part.
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = rng.standard_normal(6) * 100.0
        s[:3] -= s[:3].mean()
        assert abs(hill48_phi(s, *ISO)
                   - compute_von_mises_equivalent(s)) < 1e-9


def test_uniaxial_yield_direction_dependence():
    # Rolling (11): phi = sqrt((G+H)) s11; transverse (22): sqrt((F+H)) s11-like
    F, G, H, L, M, N = 0.3, 0.7, 0.5, 1.5, 1.5, 1.5
    s_roll = np.array([100.0, 0, 0, 0, 0, 0], float)
    s_trans = np.array([0, 100.0, 0, 0, 0, 0], float)
    assert abs(hill48_phi(s_roll, F, G, H, L, M, N) - 100 * np.sqrt(G + H)) < 1e-9
    assert abs(hill48_phi(s_trans, F, G, H, L, M, N) - 100 * np.sqrt(F + H)) < 1e-9


def test_return_map_yields_on_phi_and_hardens():
    e = np.zeros(6)
    e[0], e[1] = 0.003, -0.001
    S, ep, qp, C = return_mapping_hill48(
        e, np.zeros(6), 0.0, E, NU, *ISO, Y0, HI)
    assert qp > 0.0
    assert abs(hill48_phi(S, *ISO) - (Y0 + HI * qp)) < 1e-9 * Y0


def test_elastic_step_identity():
    e = np.zeros(6)
    e[0] = 1e-5
    S, ep, qp, C = return_mapping_hill48(
        e, np.zeros(6), 0.0, E, NU, *ISO, Y0, HI)
    assert qp == 0.0 and np.allclose(ep, 0.0, atol=0.0, rtol=0.0)


def test_tangent_vs_finite_difference():
    F, G, H, L, M, N = 0.4, 0.6, 0.5, 1.5, 1.5, 2.0
    e = np.array([0.004, -0.001, 0.0005, 0.001, 0.0, -0.0005])
    S0, _, _, C = return_mapping_hill48(
        e, np.zeros(6), 0.0, E, NU, F, G, H, L, M, N, Y0, HI)
    h = 1e-7
    Kn = np.zeros((6, 6))
    for j in range(6):
        up, um = e.copy(), e.copy()
        up[j] += h
        um[j] -= h
        Sp, _, _, _ = return_mapping_hill48(
            up, np.zeros(6), 0.0, E, NU, F, G, H, L, M, N, Y0, HI)
        Sm, _, _, _ = return_mapping_hill48(
            um, np.zeros(6), 0.0, E, NU, F, G, H, L, M, N, Y0, HI)
        Kn[:, j] = (Sp - Sm) / (2 * h)
    assert np.linalg.norm(C - Kn) / np.linalg.norm(C) < 1e-6
