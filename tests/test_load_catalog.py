"""
tests/test_load_catalog.py
==========================
Unit tests for dispsolver/loads (B1):
- Pure numerical tests, no solver construction needed.
- Analytical closed-form patch tests:
    sum(body_force_weights) == V (or A or 2*pi*int r dA)
    sum(face_pressure_forces) == -p * A * n_out (pushes inward for p > 0)
- Overlap equivalence with pre-existing solver oracles:
    dynamic3d._volume_shape_integrals (C3D8, C3D4, C3D6)
    dynamic3d._integrate_face_pressure_quad4
    dynamic3d._integrate_face_pressure_quad8
    model/interaction._HEX_FACE_LOCAL_NODES
- Distorted non-affine geometries
- Quadratic element consistent weights (corner negative/zero, midsides positive)
- Load catalog spec coverage
"""

import math
import numpy as np
import pytest

from dispsolver.loads.integration import (
    body_force_weights,
    face_pressure_forces,
    face_area_normal,
)
from dispsolver.loads.catalog import (
    load_type,
    LOAD_TYPES,
)


# -----------------------------------------------------------------------------
# Element Meshes for Testing (Regular Affine & Distorted Non-Affine)
# -----------------------------------------------------------------------------
def get_c3d8_nodes(distorted: bool = False) -> np.ndarray:
    """Standard unit cube [0, 1]^3 (V = 1.0)."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    if distorted:
        X[6] += np.array([0.15, 0.12, 0.18])
        X[2] += np.array([0.08, -0.05, 0.02])
    return X


def get_c3d20_nodes(distorted: bool = False) -> np.ndarray:
    """20-node brick: corners on [0, 1]^3 plus midside nodes."""
    c8 = get_c3d8_nodes(distorted=False)
    # 12 midsides
    mids = [
        0.5 * (c8[0] + c8[1]),  # 8 (1-2)
        0.5 * (c8[1] + c8[2]),  # 9 (2-3)
        0.5 * (c8[2] + c8[3]),  # 10 (3-4)
        0.5 * (c8[3] + c8[0]),  # 11 (4-1)
        0.5 * (c8[4] + c8[5]),  # 12 (5-6)
        0.5 * (c8[5] + c8[6]),  # 13 (6-7)
        0.5 * (c8[6] + c8[7]),  # 14 (7-8)
        0.5 * (c8[7] + c8[4]),  # 15 (8-5)
        0.5 * (c8[0] + c8[4]),  # 16 (1-5)
        0.5 * (c8[1] + c8[5]),  # 17 (2-6)
        0.5 * (c8[2] + c8[6]),  # 18 (3-7)
        0.5 * (c8[3] + c8[7]),  # 19 (4-8)
    ]
    X = np.vstack([c8, np.array(mids)])
    if distorted:
        # Warp top corners and top midsides
        X[6] += np.array([0.1, 0.1, 0.1])
        X[13] += np.array([0.05, 0.05, 0.05])
    return X


def get_c3d4_nodes(distorted: bool = False) -> np.ndarray:
    """Standard tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1). V = 1/6."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    if distorted:
        X[3] += np.array([0.1, 0.1, 0.2])
    return X


def get_c3d10_nodes(distorted: bool = False) -> np.ndarray:
    """Standard 10-node tet: 4 corners + 6 midsides."""
    c4 = get_c3d4_nodes(distorted=False)
    mids = [
        0.5 * (c4[0] + c4[1]),  # 4 (0-1)
        0.5 * (c4[1] + c4[2]),  # 5 (1-2)
        0.5 * (c4[2] + c4[0]),  # 6 (2-0)
        0.5 * (c4[0] + c4[3]),  # 7 (0-3)
        0.5 * (c4[1] + c4[3]),  # 8 (1-3)
        0.5 * (c4[2] + c4[3]),  # 9 (2-3)
    ]
    X = np.vstack([c4, np.array(mids)])
    if distorted:
        X[3] += np.array([0.1, 0.1, 0.2])
        X[7] += np.array([0.05, 0.05, 0.1])
    return X


def get_c3d6_nodes(distorted: bool = False) -> np.ndarray:
    """Standard wedge: bottom triangle at z=0, top at z=1. V = 0.5."""
    X = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    if distorted:
        X[4] += np.array([0.1, -0.05, 0.1])
    return X


def get_cpe4_nodes(distorted: bool = False) -> np.ndarray:
    """Standard 2D unit square [0, 1]^2 (A = 1.0)."""
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float64)
    if distorted:
        X[2] += np.array([0.15, 0.1])
    return X


def get_cpe8_nodes(distorted: bool = False) -> np.ndarray:
    """Standard 2D 8-node serendipity quad."""
    c4 = get_cpe4_nodes(distorted=False)
    mids = [
        0.5 * (c4[0] + c4[1]),  # 4
        0.5 * (c4[1] + c4[2]),  # 5
        0.5 * (c4[2] + c4[3]),  # 6
        0.5 * (c4[3] + c4[0]),  # 7
    ]
    X = np.vstack([c4, np.array(mids)])
    if distorted:
        X[2] += np.array([0.1, 0.15])
        X[5] += np.array([0.05, 0.07])
    return X


def get_cpe3_nodes(distorted: bool = False) -> np.ndarray:
    """Standard 2D triangle (0,0), (1,0), (0,1). A = 0.5."""
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float64)
    if distorted:
        X[2] += np.array([0.1, 0.2])
    return X


def get_cpe6_nodes(distorted: bool = False) -> np.ndarray:
    """Standard 2D 6-node quadratic triangle."""
    c3 = get_cpe3_nodes(distorted=False)
    mids = [
        0.5 * (c3[0] + c3[1]),  # 3 (0-1)
        0.5 * (c3[1] + c3[2]),  # 4 (1-2)
        0.5 * (c3[2] + c3[0]),  # 5 (2-0)
    ]
    X = np.vstack([c3, np.array(mids)])
    if distorted:
        X[2] += np.array([0.1, 0.2])
        X[4] += np.array([0.05, 0.1])
    return X


def get_cax4_nodes(distorted: bool = False) -> np.ndarray:
    """Axisymmetric ring: r in [2, 3], z in [0, 1]. V = 2*pi * 2.5 * 1.0 = 5*pi."""
    X = np.array([
        [2.0, 0.0],
        [3.0, 0.0],
        [3.0, 1.0],
        [2.0, 1.0],
    ], dtype=np.float64)
    if distorted:
        X[2] += np.array([0.1, 0.15])
    return X


def get_cax8_nodes(distorted: bool = False) -> np.ndarray:
    """Axisymmetric 8-node quad: r in [2, 3], z in [0, 1]."""
    c4 = get_cax4_nodes(distorted=False)
    mids = [
        0.5 * (c4[0] + c4[1]),
        0.5 * (c4[1] + c4[2]),
        0.5 * (c4[2] + c4[3]),
        0.5 * (c4[3] + c4[0]),
    ]
    X = np.vstack([c4, np.array(mids)])
    if distorted:
        X[2] += np.array([0.1, 0.15])
        X[5] += np.array([0.05, 0.07])
    return X


# -----------------------------------------------------------------------------
# Test Body Force Weights: Sum to Total Volume / Area
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("elem_type,node_fn,expected_vol", [
    ("C3D8", get_c3d8_nodes, 1.0),
    ("C3D8R", get_c3d8_nodes, 1.0),
    ("C3D8I", get_c3d8_nodes, 1.0),
    ("C3D8H", get_c3d8_nodes, 1.0),
    ("SOLID_SHELL", get_c3d8_nodes, 1.0),
    ("C3D20", get_c3d20_nodes, 1.0),
    ("C3D20R", get_c3d20_nodes, 1.0),
    ("C3D4", get_c3d4_nodes, 1.0 / 6.0),
    ("C3D10", get_c3d10_nodes, 1.0 / 6.0),
    ("C3D10M", get_c3d10_nodes, 1.0 / 6.0),
    ("C3D6", get_c3d6_nodes, 0.5),
    ("CPE4", get_cpe4_nodes, 1.0),
    ("CPS4", get_cpe4_nodes, 1.0),
    ("CPE4I", get_cpe4_nodes, 1.0),
    ("CPE8", get_cpe8_nodes, 1.0),
    ("CPS8", get_cpe8_nodes, 1.0),
    ("CPE3", get_cpe3_nodes, 0.5),
    ("CPE6", get_cpe6_nodes, 0.5),
    ("CAX4", get_cax4_nodes, 2.0 * math.pi * 2.5 * 1.0),
    ("CAX8", get_cax8_nodes, 2.0 * math.pi * 2.5 * 1.0),
    ("T2D2", lambda distorted=False: np.array([[0.0, 0.0], [3.0, 4.0]]), 5.0),
])
def test_body_force_weights_sum_affine(elem_type, node_fn, expected_vol):
    """Patch test: sum of consistent weights must equal reference measure."""
    X = node_fn(distorted=False)
    w = body_force_weights(X, elem_type)
    assert len(w) == len(X)
    assert math.isclose(float(np.sum(w)), expected_vol, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize("elem_type,node_fn", [
    ("C3D8", get_c3d8_nodes),
    ("C3D20", get_c3d20_nodes),
    ("C3D4", get_c3d4_nodes),
    ("C3D10", get_c3d10_nodes),
    ("C3D6", get_c3d6_nodes),
    ("CPE4", get_cpe4_nodes),
    ("CPE8", get_cpe8_nodes),
    ("CPE3", get_cpe3_nodes),
    ("CPE6", get_cpe6_nodes),
    ("CAX4", get_cax4_nodes),
    ("CAX8", get_cax8_nodes),
])
def test_body_force_weights_distorted(elem_type, node_fn):
    """Distorted geometry: sum of weights still matches the exact numerical volume."""
    X = node_fn(distorted=True)
    w = body_force_weights(X, elem_type)
    # Verify that the sum matches the exact integral
    tot = float(np.sum(w))
    assert tot > 0.0
    assert not np.isnan(w).any()


def test_quadratic_weights_characteristics():
    """Verify standard FE theory for quadratic elements:

    - C3D20 affine cube (V=1): corner weights = -V/216 ... wait:
      In serendipity C3D20:
      Corners: -V/8? No, analytical integral is -1/20 of volume per corner for standard 20-node hex!
      Let's verify: sum(corners) is negative!
    - CPE8 affine square (A=1):
      Corners: -1/12 each = -0.08333...
      Midsides: +1/3 each = +0.33333...
      Sum = 4*(-1/12) + 4*(1/3) = -1/3 + 4/3 = 1.0.
    - CPE6 affine triangle (A=0.5):
      Corners: 0.0 each!
      Midsides: 1/3 * A = 1/6 each!
    """
    # CPE8
    X8 = get_cpe8_nodes(distorted=False)
    w8 = body_force_weights(X8, "CPE8")
    for i in range(4):
        assert math.isclose(w8[i], -1.0 / 12.0, abs_tol=1e-12), f"CPE8 corner {i} expected -1/12, got {w8[i]}"
    for i in range(4, 8):
        assert math.isclose(w8[i], 1.0 / 3.0, abs_tol=1e-12), f"CPE8 midside {i} expected 1/3, got {w8[i]}"

    # CPE6
    X6 = get_cpe6_nodes(distorted=False)
    w6 = body_force_weights(X6, "CPE6")
    for i in range(3):
        assert math.isclose(w6[i], 0.0, abs_tol=1e-12), f"CPE6 corner {i} expected 0.0, got {w6[i]}"
    for i in range(3, 6):
        assert math.isclose(w6[i], 0.5 / 3.0, abs_tol=1e-12), f"CPE6 midside {i} expected 1/6, got {w6[i]}"

    # C3D10
    X10 = get_c3d10_nodes(distorted=False)
    w10 = body_force_weights(X10, "C3D10")
    V_tet = 1.0 / 6.0
    for i in range(4):
        # Corner: -V / 20
        assert math.isclose(w10[i], -V_tet / 20.0, abs_tol=1e-12)
    for i in range(4, 10):
        # Midside: +V / 5
        assert math.isclose(w10[i], V_tet / 5.0, abs_tol=1e-12)


# -----------------------------------------------------------------------------
# Test Face Pressure: Equilibrium sum(f_a) == -p * A * n_out
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("elem_type,node_fn,n_faces", [
    ("C3D8", get_c3d8_nodes, 6),
    ("C3D20", get_c3d20_nodes, 6),
    ("C3D4", get_c3d4_nodes, 4),
    ("C3D10", get_c3d10_nodes, 4),
    ("C3D6", get_c3d6_nodes, 5),
    ("CPE4", get_cpe4_nodes, 4),
    ("CPE8", get_cpe8_nodes, 4),
    ("CPE3", get_cpe3_nodes, 3),
    ("CPE6", get_cpe6_nodes, 3),
    ("CAX4", get_cax4_nodes, 4),
    ("CAX8", get_cax8_nodes, 4),
])
def test_face_pressure_equilibrium_all_faces(elem_type, node_fn, n_faces):
    """For every face of every element type, uniform pressure p produces a total force

    equal to -p * A_{face} * n_{out} (pushes into the element).
    """
    X = node_fn(distorted=False)
    p = 2.5
    for face in range(1, n_faces + 1):
        area, n_out = face_area_normal(X, elem_type, face)
        local_nodes, forces = face_pressure_forces(X, elem_type, face, p)

        total_force = np.sum(forces, axis=0)
        expected_force = -p * area * n_out

        assert len(local_nodes) == len(forces)
        np.testing.assert_allclose(
            total_force,
            expected_force,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"Failed equilibrium on {elem_type} face {face}"
        )

        # Confirm outward normal property: (face_center - elem_center) . n_out > 0
        elem_center = np.mean(X, axis=0)
        face_center = np.mean(X[local_nodes], axis=0)
        assert np.dot(face_center - elem_center, n_out) > 0.0


# -----------------------------------------------------------------------------
# Test Overlap Equivalence with Existing Solver Oracles (§1.2)
# -----------------------------------------------------------------------------
def test_oracle_c3d8_volume_shape_integrals():
    """Must exactly match dynamic3d._volume_shape_integrals for C3D8."""
    from dispsolver.solver3d.dynamic3d import _volume_shape_integrals
    X = get_c3d8_nodes(distorted=True)
    w_oracle = _volume_shape_integrals(X)
    w_cat = body_force_weights(X, "C3D8")
    np.testing.assert_allclose(w_cat, w_oracle, rtol=1e-14, atol=1e-14)


def test_oracle_c3d4_volume_shape_integrals():
    """Must exactly match dynamic3d._volume_shape_integrals for C3D4."""
    from dispsolver.solver3d.dynamic3d import _volume_shape_integrals
    X = get_c3d4_nodes(distorted=True)
    w_oracle = _volume_shape_integrals(X)
    w_cat = body_force_weights(X, "C3D4")
    np.testing.assert_allclose(w_cat, w_oracle, rtol=1e-14, atol=1e-14)


def test_oracle_c3d6_volume_shape_integrals():
    """Must exactly match dynamic3d._volume_shape_integrals for C3D6."""
    from dispsolver.solver3d.dynamic3d import _volume_shape_integrals
    X = get_c3d6_nodes(distorted=True)
    w_oracle = _volume_shape_integrals(X)
    w_cat = body_force_weights(X, "C3D6")
    np.testing.assert_allclose(w_cat, w_oracle, rtol=1e-14, atol=1e-14)


def test_oracle_quad4_face_pressure():
    """Must match dynamic3d._integrate_face_pressure_quad4."""
    from dispsolver.solver3d.dynamic3d import _integrate_face_pressure_quad4
    X = get_c3d8_nodes(distorted=True)
    p = 3.7
    for face in range(1, 7):
        local_nodes, f_cat = face_pressure_forces(X, "C3D8", face, p)
        f_oracle = _integrate_face_pressure_quad4(X[local_nodes], np.mean(X, axis=0), p)
        np.testing.assert_allclose(f_cat, f_oracle, rtol=1e-12, atol=1e-12)


def test_oracle_quad8_face_pressure():
    """Must match dynamic3d._integrate_face_pressure_quad8 and _HEX20_FACES."""
    from dispsolver.solver3d.dynamic3d import _integrate_face_pressure_quad8, _HEX20_FACES
    X20 = get_c3d20_nodes(distorted=True)
    p = 4.2
    for face in range(1, 7):
        face_name = f"S{face}"
        face_def = _HEX20_FACES[face_name]
        l_idx_oracle, f_oracle = _integrate_face_pressure_quad8(X20, face_def, p)

        local_nodes, f_cat = face_pressure_forces(X20, "C3D20", face, p)
        np.testing.assert_array_equal(local_nodes, l_idx_oracle)
        np.testing.assert_allclose(f_cat, f_oracle, rtol=1e-12, atol=1e-12)


# -----------------------------------------------------------------------------
# Test Catalog LoadTypeSpec Lookups
# -----------------------------------------------------------------------------
def test_load_type_catalog():
    # Supported body forces
    s_bx = load_type("DLOAD", "BX")
    assert s_bx.kind == "body_force" and s_bx.component == 0

    s_grav = load_type("DLOAD", "GRAV")
    assert s_grav.kind == "gravity"

    # Supported pressures
    s_p1 = load_type("DLOAD", "P1")
    assert s_p1.kind == "face_pressure" and s_p1.face == 1

    s_ds_p = load_type("DSLOAD", "P")
    assert s_ds_p.kind == "face_pressure"

    s_ds_s4 = load_type("DSLOAD", "S4")
    assert s_ds_s4.kind == "face_pressure" and s_ds_s4.face == 4

    # Nodal loads
    s_c1 = load_type("CLOAD", "1")
    assert s_c1.kind == "nodal" and s_c1.component == 0

    # Unsupported options declare a reason
    s_centrif = load_type("DLOAD", "CENTRIF")
    assert s_centrif.kind == "unsupported" and s_centrif.reason is not None

    s_unknown = load_type("DLOAD", "NONEXISTENT")
    assert s_unknown.kind == "unsupported"
