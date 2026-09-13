"""
test_terminology_main_secondary.py
==================================
Tests for modern CAE terminology adoption (main / secondary) and
backward compatibility for legacy terms (master / slave).
"""

import pytest
import numpy as np
from dispsolver.constraint import SurfaceTie, SurfaceTieConstraint, SurfaceTieConstraint3D
from dispsolver.model.constraint import Tie
from dispsolver.model.interaction import ContactPair, ContactProperty, AnalyticalRigidSurface
from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d.mesh3d import Mesh3D


def test_surface_tie_main_secondary_2d():
    mesh = Mesh2D()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(3, 0.0, 0.1)
    mesh.add_node(4, 1.0, 0.1)
    mesh.add_element(1, [1, 2, 4, 3], elem_type="CPE4")

    # New standard: secondary, main
    tie_new = SurfaceTie(
        secondary=[3, 4],
        main=[1, 2],
        mesh=mesh,
        penalty_stiffness=1e5,
        name="TIE_NEW"
    )
    assert isinstance(tie_new, SurfaceTieConstraint)
    assert tie_new.slave_node_ids == [3, 4]
    assert tie_new.master_node_ids == [1, 2]

    # Legacy alias: slave, master
    tie_legacy = SurfaceTie(
        slave=[3, 4],
        master=[1, 2],
        mesh=mesh,
        penalty_stiffness=1e5,
        name="TIE_LEGACY"
    )
    assert isinstance(tie_legacy, SurfaceTieConstraint)
    assert tie_legacy.slave_node_ids == [3, 4]
    assert tie_legacy.master_node_ids == [1, 2]


def test_surface_tie_main_secondary_3d():
    mesh = Mesh3D()
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
        (9, 0.0, 0.0, 1.1), (10, 1.0, 0.0, 1.1)
    ]
    for nid, x, y, z in nodes:
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], elem_type="C3D8")

    # New standard: secondary, main
    tie_new = SurfaceTie(
        secondary=[9, 10],
        main=[5, 6, 7, 8],
        mesh=mesh,
        penalty_stiffness=1e6
    )
    assert isinstance(tie_new, SurfaceTieConstraint3D)
    assert (5, 6, 7, 8) in tie_new.master_faces

    # Legacy alias: slave, master
    tie_legacy = SurfaceTie(
        slave=[9, 10],
        master=[5, 6, 7, 8],
        mesh=mesh,
        penalty_stiffness=1e6
    )
    assert isinstance(tie_legacy, SurfaceTieConstraint3D)
    assert (5, 6, 7, 8) in tie_legacy.master_faces


def test_model_tie_bidirectional_alias():
    # 1. Using new main / secondary
    t1 = Tie(name="TIE_1", main="MAIN_SURF", secondary="SEC_SURF")
    assert t1.main == "MAIN_SURF"
    assert t1.secondary == "SEC_SURF"
    assert t1.master == "MAIN_SURF"
    assert t1.slave == "SEC_SURF"

    # 2. Using legacy master / slave
    t2 = Tie(name="TIE_2", master="M_SURF", slave="S_SURF")
    assert t2.main == "M_SURF"
    assert t2.secondary == "S_SURF"
    assert t2.master == "M_SURF"
    assert t2.slave == "S_SURF"


def test_model_contact_pair_bidirectional_alias():
    cp_prop = ContactProperty(name="PROP_1")
    rigid = AnalyticalRigidSurface(name="RIGID_1", point=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0])

    # 1. Using main / secondary
    cp1 = ContactPair(name="CP1", interaction_property=cp_prop, main=rigid, secondary="DEFORM_SURF")
    assert cp1.main == rigid
    assert cp1.secondary == "DEFORM_SURF"
    assert cp1.master == rigid
    assert cp1.slave == "DEFORM_SURF"

    # 2. Using master / slave
    cp2 = ContactPair(name="CP2", interaction_property=cp_prop, master=rigid, slave="DEFORM_SURF")
    assert cp2.main == rigid
    assert cp2.secondary == "DEFORM_SURF"
    assert cp2.master == rigid
    assert cp2.slave == "DEFORM_SURF"
