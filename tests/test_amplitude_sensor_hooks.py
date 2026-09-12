"""Unit tests for Amplitude curves, Numba User Functions, Sensors, Iteration Hooks, and Solver Encapsulation."""

import pytest
import numpy as np
from numba import njit

from dispsolver.model.model import Model
from dispsolver.model.amplitude import TabularAmplitude, SmoothStepAmplitude, UserFunctionAmplitude
from dispsolver.model.sensor import Sensor, IterationHook, ControlAction, SensorManager
from dispsolver.model.bc import DisplacementBC, UserFunctionBC


def test_tabular_amplitude():
    amp = TabularAmplitude(name="Tab1", data=[(0.0, 0.0), (0.5, 10.0), (1.0, 20.0)])
    assert amp.evaluate(0.0) == 0.0
    assert amp.evaluate(0.25) == 5.0
    assert amp.evaluate(0.5) == 10.0
    assert amp.evaluate(1.0) == 20.0


def test_smoothstep_amplitude():
    amp = SmoothStepAmplitude(name="Smooth1", data=[(0.0, 0.0), (1.0, 100.0)])
    # t=0 -> 0, t=1 -> 100, t=0.5 -> 50 (s(0.5) = 10(1/8) - 15(1/16) + 6(1/32) = 0.5)
    assert pytest.approx(amp.evaluate(0.0), abs=1e-6) == 0.0
    assert pytest.approx(amp.evaluate(0.5), abs=1e-6) == 50.0
    assert pytest.approx(amp.evaluate(1.0), abs=1e-6) == 100.0


def test_user_function_amplitude_numba():
    @njit
    def my_numba_amp(t, step_time, total_time):
        return 5.0 * t**2

    amp = UserFunctionAmplitude(name="UserAmp", numba_func=my_numba_amp)
    assert pytest.approx(amp.evaluate(2.0), abs=1e-6) == 20.0


def test_model_amplitude_and_bc_registration():
    model = Model(name="TestModel", dim=3)
    part = model.Part(name="Bar")
    
    # Create nodes and elements
    for i in range(8):
        x = 10.0 if i in [1, 2, 5, 6] else 0.0
        y = 1.0 if i in [2, 3, 6, 7] else 0.0
        z = 1.0 if i in [4, 5, 6, 7] else 0.0
        part.Node(i + 1, x, y, z)
    part.Element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
    part.NodeSet("FIXED_FACE", [1, 4, 8, 5])
    part.NodeSet("MOVE_FACE", [2, 3, 7, 6])

    mat = model.Material(name="Steel")
    sec = model.SolidSection(name="Sec1", material="Steel")
    part.SectionAssignment(region="ALL", sectionName="Sec1")

    inst = model.root_assembly.Instance(name="BAR_1", part=part)

    amp = model.SmoothStepAmplitude(name="SmoothRamp", data=[(0.0, 0.0), (1.0, 1.0)])
    step = model.Step(name="Step-1", time_period=1.0)

    bc1 = model.DisplacementBC(name="FixedBC", createStepName="Step-1", region="BAR_1.FIXED_FACE", u1=0.0, u2=0.0, u3=0.0)
    bc2 = model.DisplacementBC(name="MoveBC", createStepName="Step-1", region="BAR_1.MOVE_FACE", u1=2.0, amplitude=amp)

    assert "FixedBC" in step.boundary_conditions
    assert "MoveBC" in step.boundary_conditions
    assert step.boundary_conditions["MoveBC"].entity.amplitude == "SmoothRamp"


def test_sensor_and_iteration_hook():
    model = Model(name="TestModel", dim=3)
    s1 = model.Sensor(name="Disp_Sensor", entity_type="node", entity_id=1, variable="U", comp="MAG")
    
    called = []
    def my_hook(iter_count, step_time, solver, sensor_vals):
        called.append((iter_count, step_time))
        return ControlAction.CONTINUE

    hook = model.IterationHook(name="Hook1", callback=my_hook)

    step = model.Step(name="Step-1")
    step.sensors["S1"] = s1
    step.iteration_hooks["H1"] = hook

    assert "S1" in step.sensors
    assert "H1" in step.iteration_hooks
