# -*- coding: utf-8 -*-
"""Tests for field-variable-dependent material response (FV1, Abaqus *FIELD parity)."""
import numpy as np

from dispsolver.material.dependence_table import eval_table_1d_numba
from dispsolver.model.material import Material
from benchmark_element.benchmark_field_ef1 import (
    run_benchmark_field_elastic_ef1,
    run_benchmark_field_plastic_ef1,
    _uniaxial_reaction,
)


def test_field_table_exact_and_clamp():
    f = np.array([0.0, 1.0])
    e = np.array([200000.0, 160000.0])
    assert eval_table_1d_numba(f, e, 0.0) == 200000.0
    assert eval_table_1d_numba(f, e, 1.0) == 160000.0
    assert eval_table_1d_numba(f, e, 0.5) == 180000.0
    assert eval_table_1d_numba(f, e, -2.0) == 200000.0
    assert eval_table_1d_numba(f, e, 5.0) == 160000.0


def test_material_field_table_storage():
    mat = Material(name="F")
    mat.Elastic(field_table=((200000.0, 0.3, 0.0), (160000.0, 0.3, 1.0)))
    assert mat.elastic == (200000.0, 0.3)
    assert mat.extra_params["field_elastic_table"] == [
        (200000.0, 0.3, 0.0), (160000.0, 0.3, 1.0)]
    mat2 = Material(name="P")
    mat2.Plastic(table=[(250.0, 0.0)], field_table=((250.0, 0.0, 0.0), (200.0, 0.0, 1.0)))
    assert mat2.extra_params["field_plastic_table"] == [
        (250.0, 0.0, 0.0), (200.0, 0.0, 1.0)]


def test_field_elastic_solver_parity():
    assert abs(_uniaxial_reaction(0.0) - 200.0) / 200.0 < 0.01
    assert abs(_uniaxial_reaction(1.0) - 160.0) / 160.0 < 0.01


def test_field_benchmark_pair_pass():
    assert run_benchmark_field_elastic_ef1()
    assert run_benchmark_field_plastic_ef1()
