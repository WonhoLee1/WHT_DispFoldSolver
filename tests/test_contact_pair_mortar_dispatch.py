"""
tests/test_contact_pair_mortar_dispatch.py
===========================================
Verification tests for ContactPair (SurfaceToSurfaceContactStd) routing to Mortar constraints:

1. 3D Small Sliding + DUAL_MORTAR -> DualMortarClippedContactConstraint3D (mortar_type="DUAL")
2. 3D Small Sliding + STANDARD_MORTAR -> DualMortarClippedContactConstraint3D (mortar_type="STANDARD")
3. 3D Finite Sliding + DUAL_MORTAR -> FiniteSlidingDualMortar3D (mortar_type="DUAL")
4. 3D Finite Sliding + STANDARD_MORTAR -> FiniteSlidingDualMortar3D (mortar_type="STANDARD")
5. 2D ContactPair + DUAL_MORTAR -> DualMortarContactConstraint2D (mortar_type="DUAL")
6. 2D ContactPair + STANDARD_MORTAR -> DualMortarContactConstraint2D (mortar_type="STANDARD")
"""

import numpy as np
import pytest

from dispsolver.model.interaction import (
    ContactPair,
    SurfaceToSurfaceContactStd,
    ContactProperty,
    DUAL_MORTAR,
    STANDARD_MORTAR,
)
from dispsolver.constraint3d.dual_mortar_clipping3d import DualMortarClippedContactConstraint3D
from dispsolver.constraint3d.finite_sliding_mortar3d import FiniteSlidingDualMortar3D
from dispsolver.constraint.dual_mortar_contact import DualMortarContactConstraint2D


def test_contact_pair_3d_small_sliding_mortar_dispatch():
    """Verify ContactPair with SMALL sliding dispatches to DualMortarClippedContactConstraint3D."""
    coords = np.array([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
    ], dtype=np.float64)
    nid_to_idx = {i + 1: i for i in range(8)}

    class MockMesh:
        def nodes_array(self):
            return coords

    mesh = MockMesh()
    prop = ContactProperty(name="P1")

    # DUAL
    cp_dual = ContactPair(
        name="CP_DUAL",
        main=[(5, 6, 7, 8)],
        secondary=[(1, 2, 3, 4)],
        interaction_property=prop,
        sliding="SMALL",
        constraint_enforcement=DUAL_MORTAR,
    )
    c_dual = cp_dual.build_runtime_constraint(mesh, nid_to_idx, coords)
    assert isinstance(c_dual, DualMortarClippedContactConstraint3D)
    assert c_dual.mortar_type == "DUAL"

    # STANDARD
    cp_std = ContactPair(
        name="CP_STD",
        main=[(5, 6, 7, 8)],
        secondary=[(1, 2, 3, 4)],
        interaction_property=prop,
        sliding="SMALL",
        constraint_enforcement=STANDARD_MORTAR,
    )
    c_std = cp_std.build_runtime_constraint(mesh, nid_to_idx, coords)
    assert isinstance(c_std, DualMortarClippedContactConstraint3D)
    assert c_std.mortar_type == "STANDARD"


def test_contact_pair_3d_finite_sliding_mortar_dispatch():
    """Verify ContactPair with FINITE sliding dispatches to FiniteSlidingDualMortar3D."""
    coords = np.array([
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],
    ], dtype=np.float64)
    nid_to_idx = {i + 1: i for i in range(8)}

    class MockMesh:
        def nodes_array(self):
            return coords

    mesh = MockMesh()
    prop = ContactProperty(name="P1")

    # FINITE DUAL
    cp_dual = SurfaceToSurfaceContactStd(
        name="CP_FINITE_DUAL",
        master=[(5, 6, 7, 8)],
        slave=[(1, 2, 3, 4)],
        interactionProperty=prop,
        sliding="FINITE",
        constraintEnforcement=DUAL_MORTAR,
    )
    c_dual = cp_dual.build_runtime_constraint(mesh, nid_to_idx, coords)
    assert isinstance(c_dual, FiniteSlidingDualMortar3D)
    assert c_dual.mortar_type == "DUAL"

    # FINITE STANDARD
    cp_std = SurfaceToSurfaceContactStd(
        name="CP_FINITE_STD",
        master=[(5, 6, 7, 8)],
        slave=[(1, 2, 3, 4)],
        interactionProperty=prop,
        sliding="FINITE",
        constraintEnforcement=STANDARD_MORTAR,
    )
    c_std = cp_std.build_runtime_constraint(mesh, nid_to_idx, coords)
    assert isinstance(c_std, FiniteSlidingDualMortar3D)
    assert c_std.mortar_type == "STANDARD"


def test_contact_pair_2d_mortar_dispatch():
    """Verify 2D ContactPair dispatches to DualMortarContactConstraint2D."""
    coords_2d = np.array([
        [0.0, 0.0], [1.0, 0.0],
        [0.0, 0.0], [1.0, 0.0],
    ], dtype=np.float64)
    nid_to_idx = {1: 0, 2: 1, 3: 2, 4: 3}

    class MockMesh2D:
        def nodes_array(self):
            return coords_2d

    mesh = MockMesh2D()
    prop = ContactProperty(name="P1")

    # 2D DUAL
    cp_2d_dual = ContactPair(
        name="CP_2D_DUAL",
        main=[(3, 4)],
        secondary=[(1, 2)],
        interaction_property=prop,
        constraint_enforcement=DUAL_MORTAR,
    )
    c_dual = cp_2d_dual.build_runtime_constraint(mesh, nid_to_idx, coords_2d)
    assert isinstance(c_dual, DualMortarContactConstraint2D)
    assert c_dual.mortar_type == "DUAL"

    # 2D STANDARD
    cp_2d_std = ContactPair(
        name="CP_2D_STD",
        main=[(3, 4)],
        secondary=[(1, 2)],
        interaction_property=prop,
        constraint_enforcement=STANDARD_MORTAR,
    )
    c_std = cp_2d_std.build_runtime_constraint(mesh, nid_to_idx, coords_2d)
    assert isinstance(c_std, DualMortarContactConstraint2D)
    assert c_std.mortar_type == "STANDARD"
