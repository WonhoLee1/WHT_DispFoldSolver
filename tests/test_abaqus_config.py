"""
test_abaqus_config.py
======================
Unit test suite for Abaqus-style solver configuration engine, JSON/YAML IO,
key path dynamic overrides, preset factory, and DynamicSolver integration.
"""

import os
import tempfile
import pytest
import numpy as np

from dispsolver.solver.abaqus_config import (
    AbaqusSolverConfig,
    AbaqusStepConfig,
    AbaqusTimeIncrementationControls,
    AbaqusFieldControls,
    AbaqusDynamicControls,
    AbaqusSectionControls,
    AbaqusStabilizeControls,
)
from dispsolver.fold_model_config import FoldModelConfig, SolverTuningConfig, DEFAULT_CONFIG
from dispsolver.solver.dynamic import DynamicSolver
from dispsolver.mesh.mesh import Mesh


def test_abaqus_config_defaults():
    """Verify Abaqus-style default parameter values match standard FEA specifications."""
    cfg = AbaqusSolverConfig()
    assert cfg.step.procedure == "DYNAMIC"
    assert cfg.step.application == "MODERATE DISSIPATION"
    assert cfg.step.nlgeom is True
    assert cfg.step.dt_init == 0.0025
    assert cfg.step.t_total == 1.0

    assert cfg.time_incrementation.I_0 == 10
    assert cfg.time_incrementation.I_R == 16
    assert cfg.time_incrementation.I_P == 9
    assert cfg.time_incrementation.I_C == 20
    assert cfg.time_incrementation.I_L == 4
    assert cfg.time_incrementation.dt_growth_factor == 1.5
    assert cfg.time_incrementation.dt_cutback_factor == 0.25

    assert cfg.field_controls.R_n == 0.005
    assert cfg.field_controls.C_n == 0.01
    assert cfg.field_controls.rtol == 1e-4
    assert cfg.field_controls.atol == 1e-6
    assert cfg.field_controls.max_disp_corr == 0.03

    assert cfg.dynamic_controls.integration_mode == "moderate-4"
    assert cfg.dynamic_controls.alpha == -0.22

    assert cfg.section_controls.distortion_control is True
    assert cfg.section_controls.distortion_j_crit == 0.20


def test_inp_snippet_roundtrip():
    """Verify Abaqus .inp keyword snippet generation and parsing."""
    cfg_orig = AbaqusSolverConfig()
    cfg_orig.step.dt_max = 0.05
    cfg_orig.time_incrementation.I_C = 25
    cfg_orig.dynamic_controls.alpha = -0.15
    cfg_orig.field_controls.R_n = 0.001
    cfg_orig.section_controls.distortion_j_crit = 0.15

    inp_text = cfg_orig.to_inp_snippet()
    assert "*STEP, NLGEOM=YES" in inp_text
    assert "*DYNAMIC, APPLICATION=MODERATE DISSIPATION, ALPHA=-0.1500" in inp_text
    assert "*CONTROLS, PARAMETERS=TIME INCREMENTATION" in inp_text
    assert "*CONTROLS, PARAMETERS=FIELD" in inp_text
    assert "*SECTION CONTROLS, NAME=PSA_DISTORTION_CTRL, DISTORTION CONTROL=YES, LENGTH RATIO=0.15" in inp_text

    cfg_parsed = AbaqusSolverConfig.from_inp_snippet(inp_text)
    assert cfg_parsed.step.procedure == "DYNAMIC"
    assert cfg_parsed.step.dt_max == 0.05
    assert cfg_parsed.time_incrementation.I_C == 25
    assert pytest.approx(cfg_parsed.dynamic_controls.alpha, abs=1e-4) == -0.15
    assert pytest.approx(cfg_parsed.field_controls.R_n, abs=1e-4) == 0.001
    assert pytest.approx(cfg_parsed.section_controls.distortion_j_crit, abs=1e-2) == 0.15


def test_json_and_yaml_serialization():
    """Verify JSON and YAML save/load file roundtrip for FoldModelConfig and AbaqusSolverConfig."""
    cfg = FoldModelConfig.get_preset("optistruct_galpha")
    cfg.solver.max_cutbacks = 20
    cfg.solver.max_displacement_corr = 0.05

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "config.json")
        cfg.save(json_path)
        assert os.path.exists(json_path)

        loaded_json = FoldModelConfig.load(json_path)
        assert loaded_json.solver.integration_mode == "generalized-0.8"
        assert loaded_json.solver.max_cutbacks == 20
        assert loaded_json.solver.max_displacement_corr == 0.05

        yaml_path = os.path.join(tmpdir, "config.yaml")
        cfg.save(yaml_path)
        assert os.path.exists(yaml_path)

        loaded_yaml = FoldModelConfig.load(yaml_path)
        assert loaded_yaml.solver.integration_mode == "generalized-0.8"
        assert loaded_yaml.solver.max_cutbacks == 20


def test_dot_notation_key_path_mutation():
    """Verify set_path and get_path dynamic key-path access."""
    cfg = FoldModelConfig()
    cfg.set_path("solver.integration_mode", "generalized-0.7")
    cfg.set_path("solver.max_cutbacks", 20)
    cfg.set_path("drive.dt_max", 0.05)
    cfg.set_path("solver.distortion_control", False)

    assert cfg.get_path("solver.integration_mode") == "generalized-0.7"
    assert cfg.get_path("solver.max_cutbacks") == 20
    assert cfg.get_path("drive.dt_max") == 0.05
    assert cfg.get_path("solver.distortion_control") is False


def test_config_presets():
    """Verify predefined config presets."""
    teardrop = FoldModelConfig.get_preset("teardrop")
    assert teardrop.geometry.hinge_half_gap == 7.5
    assert teardrop.drive.dt_max == 0.1

    optistruct = FoldModelConfig.get_preset("optistruct_galpha")
    assert optistruct.solver.integration_mode == "generalized-0.8"

    quasistatic = FoldModelConfig.get_preset("quasistatic")
    assert quasistatic.solver.integration_mode == "quasistatic"


def test_dynamic_solver_from_config():
    """Verify DynamicSolver.from_config instantiation."""
    mesh = Mesh()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(3, 1.0, 1.0)
    mesh.add_node(4, 0.0, 1.0)
    mesh.add_element(1, [1, 2, 3, 4], "Q4", pid=0)

    mat_params = {0: {"E": 1000.0, "nu": 0.3}}
    cfg = FoldModelConfig.get_preset("moderate-4")
    cfg.solver.max_cutbacks = 20
    cfg.solver.max_iter = 10

    solver = DynamicSolver.from_config(
        mesh=mesh,
        material=mat_params[0],
        material_params=mat_params,
        config=cfg,
        elem_jit="numba",
        verbose=False,
    )

    assert solver.mode == "moderate-4"
    assert solver.max_iter == 10
    assert getattr(solver, "max_cutbacks", None) == 20
