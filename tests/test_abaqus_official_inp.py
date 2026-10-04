"""Official-benchmark-equivalent inp reading checks.

Decks in benchmark_inp/ are continuum analogues built from the SIMULIA
verification-manual descriptions (the reference .txt pages carry no inp
text). These tests assert READING coverage, not benchmark numbers.
"""
import os

import pytest

from dispsolver.io.abaqus_reader import _parse_source

def _inp_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "benchmark_inp")


def _read(name, dim):
    reader, model = _parse_source(os.path.join(_inp_dir(), name), dim=dim)
    return reader, model


def test_cook_2d_reads():
    reader, model = _read("cook_2d_coarse.inp", dim=2)
    assert reader.unsupported_keywords == []
    part = list(model.parts.values())[0]
    assert len(part.nodes) == 25
    assert len(part.elements) == 16
    assert "SHEAR" in model.steps
    assert "CR" in model.constraints
    assert model.materials["RUBBER"].extra_params.get("trs") is None
    step = model.steps["SHEAR"]
    assert len(step.get_active_bcs()) == 1
    assert len(step.get_active_loads()) == 1


def test_cantilever_2d_reads():
    reader, model = _read("cantilever_2d.inp", dim=2)
    assert reader.unsupported_keywords == []
    part = list(model.parts.values())[0]
    assert len(part.nodes) == 22
    assert len(part.elements) == 10
    assert model.materials["STEEL"].density == 1.0
    assert "BEND" in model.steps


def test_cube_c3d8_reads():
    reader, model = _read("cube_c3d8.inp", dim=3)
    assert reader.unsupported_keywords == []
    assert reader.warnings == []
    part = list(model.parts.values())[0]
    assert len(part.nodes) == 8
    assert len(part.elements) == 1


def test_patch_2d_reads():
    reader, model = _read("patch_2d.inp", dim=2)
    assert reader.unsupported_keywords == []
    part = list(model.parts.values())[0]
    assert len(part.nodes) == 9
    assert len(part.elements) == 4
    assert "PATCHLOAD" in model.steps


def test_cantilever_moment_2d_reads():
    reader, model = _read("cantilever_moment_2d.inp", dim=2)
    assert reader.unsupported_keywords == []
    step = model.steps["BEND"]
    assert len(step.get_active_loads()) == 2


def test_thick_cylinder_2d_reads():
    reader, model = _read("thick_cylinder_2d.inp", dim=2)
    part = list(model.parts.values())[0]
    assert len(part.nodes) == 27
    assert len(part.elements) == 16
    step = model.steps["PRESS"]
    assert len(step.get_active_loads()) == 1


def test_tensile_bar_3d_reads():
    reader, model = _read("tensile_bar_3d.inp", dim=3)
    assert reader.unsupported_keywords == []
    part = list(model.parts.values())[0]
    assert len(part.nodes) == 12
    assert len(part.elements) == 2
    assert "TENSION" in model.steps


def test_shear_2d_reads():
    reader, model = _read("shear_2d.inp", dim=2)
    assert reader.unsupported_keywords == []
    assert reader.warnings == []
    part = list(model.parts.values())[0]
    assert len(part.nodes) == 4


def test_cook_3d_reads():
    reader, model = _read("cook_3d_coarse.inp", dim=3)
    assert reader.unsupported_keywords == []
    part = list(model.parts.values())[0]
    assert len(part.nodes) == 18
    assert len(part.elements) == 4


def test_contact_2block_2d_reads():
    reader, model = _read("contact_2block_2d.inp", dim=2)
    assert reader.unsupported_keywords == []
    assert "TOP" in model.parts and "BOT" in model.parts
    assert len(model.root_assembly.instances) == 2
    assert len(model.contact_properties) == 1
    assert model.contact_properties["FR"].friction == 0.2
    assert len(model.interactions) == 1


def test_assembly_rotated_2d_reads():
    reader, model = _read("assembly_rotated_2d.inp", dim=2)
    assert reader.unsupported_keywords == []
    part = model.parts["P"]
    assert "EDGE" in part.sets
    # a set defined inside *INSTANCE keeps its deck name on the instance's own
    # part (it used to be renamed I__ILOCAL, which INST.SET references and
    # sections could not find)
    assert "ILOCAL" in part.sets
