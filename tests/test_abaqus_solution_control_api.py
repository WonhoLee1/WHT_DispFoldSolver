"""Tests for Abaqus-compatible Solution Control API and Presets."""

import pytest
from dispsolver.model.control import (
    FieldControl,
    TimeIncrementationControl,
    LineSearchControl,
    ActiveToleranceEngine,
    SolutionControl,
)
from dispsolver.model.model import Model
from dispsolver.model.step import Step


def test_field_and_time_controls_tuples():
    fc = FieldControl.from_tuple((0.008, 0.02, 1.5, 0.003, 0.04, 0.002))
    assert fc.R_n == 0.008
    assert fc.C_n == 0.02
    assert fc.q_0 == 1.5
    assert fc.c_0 == 0.003
    assert fc.R_p == 0.04
    assert fc.C_a == 0.002

    tc = TimeIncrementationControl.from_tuple((20, 20, 10, 30, 3, 5, 1.30, 0.30))
    assert tc.I_0 == 20
    assert tc.I_R == 20
    assert tc.I_P == 10
    assert tc.I_C == 30
    assert tc.I_L == 3
    assert tc.I_G == 5
    assert tc.D_G == 1.30
    assert tc.D_C == 0.30

    ls = LineSearchControl.from_tuple((6, 0.002, 1.05, 0.30))
    assert ls.N_ls == 6
    assert ls.s_ls == 0.002
    assert ls.f_ls == 1.05
    assert ls.eta_ls == 0.30


def test_solution_control_presets():
    # 1. STANDARD
    ctrl_std = SolutionControl("Std", preset="STANDARD")
    assert ctrl_std.preset == "STANDARD"
    assert ctrl_std.field.R_n == 0.005
    assert ctrl_std.field.C_n == 0.01
    assert ctrl_std.field.R_p == 0.02
    assert ctrl_std.field.C_a == 0.001
    assert ctrl_std.adaptive_convergence is False

    # 2. RELAXED
    ctrl_rel = SolutionControl("Rel", preset="RELAXED")
    assert ctrl_rel.preset == "RELAXED"
    assert ctrl_rel.field.R_n == 0.02
    assert ctrl_rel.field.C_n == 0.05
    assert ctrl_rel.field.R_p == 0.05
    assert ctrl_rel.field.C_a == 0.005

    # 3. CONCEPTUAL
    ctrl_con = SolutionControl("Con", preset="CONCEPTUAL")
    assert ctrl_con.preset == "CONCEPTUAL"
    assert ctrl_con.field.R_n == 0.05
    assert ctrl_con.field.C_n == 0.10
    assert ctrl_con.field.R_p == 0.10
    assert ctrl_con.field.C_a == 0.01

    # 4. ADAPTIVE
    ctrl_adp = SolutionControl("Adp", preset="ADAPTIVE")
    assert ctrl_adp.preset == "ADAPTIVE"
    assert ctrl_adp.field.R_n == 0.005
    assert ctrl_adp.field.C_n == 0.01
    assert ctrl_adp.adaptive_convergence is True


def test_active_tolerance_engine():
    engine = ActiveToleranceEngine(R_n_base=0.005, C_n_base=0.01, max_relax_factor=5.0)

    # When inactive: strictly returns base values
    engine.is_active = False
    r_n, c_n, r_p, c_a = engine.compute_tolerances(cutback_count=2, iter_count=10)
    assert r_n == 0.005
    assert c_n == 0.01

    # When active:
    engine.is_active = True
    # Normal smooth iteration:
    r_n0, c_n0, _, _ = engine.compute_tolerances(cutback_count=0, iter_count=2)
    assert r_n0 == 0.005
    assert c_n0 == 0.01

    # On cutback 1:
    r_n1, c_n1, _, _ = engine.compute_tolerances(cutback_count=1, iter_count=0)
    assert r_n1 == pytest.approx(0.010)
    assert c_n1 == pytest.approx(0.020)

    # Ceiling guard (max_relax_factor = 5.0)
    r_n_max, c_n_max, _, _ = engine.compute_tolerances(cutback_count=10, iter_count=0)
    assert r_n_max == pytest.approx(0.005 * 5.0)
    assert c_n_max == pytest.approx(0.01 * 5.0)


def test_model_and_step_integration():
    model = Model(name="TestModel", dim=2)
    assert "solutionControls" in dir(model)
    assert hasattr(model, "solution_controls")

    fast_ctrl = model.SolutionControl("FastCtrl", preset="ADAPTIVE", maxRelaxFactor=4.0)
    assert "FASTCTRL" in model.solution_controls or "FastCtrl" in model.solution_controls

    step = model.StaticStep("Step-1")
    assert hasattr(step, "control")
    assert step.control.preset == "STANDARD"

    # Bind via setValues
    step.setValues(solutionControls=fast_ctrl)
    assert step.control.preset == "ADAPTIVE"
    assert step.control.max_relax_factor == 4.0

    # Modify step.control directly via Abaqus API
    step.control.setValues(preset="RELAXED")
    assert step.control.preset == "RELAXED"
    assert step.control.field.R_n == 0.02


def test_to_inp_snippet():
    ctrl = SolutionControl("InpCtrl", preset="STANDARD")
    inp = ctrl.to_inp_snippet()
    assert "*CONTROLS, PARAMETERS=TIME INCREMENTATION" in inp
    assert "*CONTROLS, PARAMETERS=FIELD" in inp
    assert "*CONTROLS, PARAMETERS=LINE SEARCH" in inp
