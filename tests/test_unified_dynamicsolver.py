"""
test_unified_dynamicsolver.py
=============================
Tests for unified DynamicSolver facade (Abaqus API Parity).
Verifies automatic routing to DynamicSolver2D and DynamicSolver3D.
"""

import pytest
import numpy as np
from dispsolver.solver import DynamicSolver
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.model.model import Model
from dispsolver.model.part import Part
from dispsolver.model.material import Material


def test_unified_dynamicsolver_2d_dispatch():
    mesh = Mesh2D()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(3, 1.0, 1.0)
    mesh.add_node(4, 0.0, 1.0)
    mesh.add_element(1, [1, 2, 3, 4], elem_type="CPE4", pid=1)

    materials = {1: {"E": 1000.0, "nu": 0.3}}
    solver = DynamicSolver(mesh, materials=materials)

    assert isinstance(solver, DynamicSolver2D)
    assert solver.mesh is mesh
    assert solver.num_dofs == 8


def test_unified_dynamicsolver_3d_dispatch():
    mesh = Mesh3D()
    nodes = [
        (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0), (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0), (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0)
    ]
    for nid, x, y, z in nodes:
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], elem_type="C3D8", pid=1)

    materials = {1: {"E": 2000.0, "nu": 0.3}}
    solver = DynamicSolver(mesh, materials=materials)

    assert isinstance(solver, DynamicSolver3D)
    assert solver.mesh is mesh
    assert solver.num_dofs == 24


def test_model_create_solver_unified():
    model = Model(name="TestModel")
    mat = Material(name="Steel")
    mat.Elastic([[210000.0, 0.3]])
    model.materials["Steel"] = mat

    part = Part(name="Part-1", dim=3)
    part.Node(1, 0.0, 0.0, 0.0)
    part.Node(2, 1.0, 0.0, 0.0)
    part.Node(3, 1.0, 1.0, 0.0)
    part.Node(4, 0.0, 1.0, 0.0)
    part.Node(5, 0.0, 0.0, 1.0)
    part.Node(6, 1.0, 0.0, 1.0)
    part.Node(7, 1.0, 1.0, 1.0)
    part.Node(8, 0.0, 1.0, 1.0)
    part.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
    part.create_set(name="Part-1-Set", elements=[1], nodes=[1, 2, 3, 4, 5, 6, 7, 8])
    model.HomogeneousSolidSection(name="SolidSection", material="Steel")
    part.SectionAssignment(region="Part-1-Set", sectionName="SolidSection")
    model.parts["Part-1"] = part

    inst = model.root_assembly.Instance(name="Part-1-1", part=part)
    model.root_assembly.instances["Part-1-1"] = inst

    solver, sys = model.create_solver()
    assert isinstance(solver, DynamicSolver3D)


def test_model_create_solver_unified_2d():
    model = Model(name="TestModel2D", dim=2)
    mat = Material(name="Polymer")
    mat.Elastic([[3000.0, 0.35]])
    model.materials["Polymer"] = mat

    part = Part(name="Part-2D", dim=2)
    part.Node(1, 0.0, 0.0)
    part.Node(2, 1.0, 0.0)
    part.Node(3, 1.0, 1.0)
    part.Node(4, 0.0, 1.0)
    part.Element(1, "CPE4", [1, 2, 3, 4])
    part.create_set(name="Part-2D-Set", elements=[1], nodes=[1, 2, 3, 4])
    model.HomogeneousSolidSection(name="SolidSection2D", material="Polymer")
    part.SectionAssignment(region="Part-2D-Set", sectionName="SolidSection2D")
    model.parts["Part-2D"] = part

    inst = model.root_assembly.Instance(name="Part-2D-1", part=part)
    model.root_assembly.instances["Part-2D-1"] = inst

    solver, sys = model.create_solver()
    assert isinstance(solver, DynamicSolver2D)
