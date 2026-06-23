"""
test_amplitude.py
=================
Time-dependent loads/BCs via Amplitude curves (Abaqus *AMPLITUDE).
"""

import numpy as np
import pytest

from dispsolver.mesh import Mesh
from dispsolver.material import LinearViscoelastic
from dispsolver.solver import DynamicSolver
from dispsolver.load import Amplitude


# ------------------------------------------------------------------
# Amplitude interpolation
# ------------------------------------------------------------------

def test_amplitude_linear():
    a = Amplitude([0.0, 1.0, 2.0], [0.0, 1.0, 3.0], method="linear")
    assert a(0.0) == pytest.approx(0.0)
    assert a(0.5) == pytest.approx(0.5)
    assert a(1.0) == pytest.approx(1.0)
    assert a(1.5) == pytest.approx(2.0)
    # endpoint hold
    assert a(-5.0) == pytest.approx(0.0)
    assert a(99.0) == pytest.approx(3.0)


def test_amplitude_smooth_step():
    a = Amplitude([0.0, 1.0], [0.0, 1.0], method="smooth")
    assert a(0.0) == pytest.approx(0.0)
    assert a(1.0) == pytest.approx(1.0)
    assert a(0.5) == pytest.approx(0.5)        # symmetric midpoint
    # smooth step is flatter than linear near the ends
    assert a(0.1) < 0.1
    assert a(0.9) > 0.9


def test_amplitude_step():
    a = Amplitude([0.0, 1.0, 2.0], [10.0, 20.0, 30.0], method="step")
    assert a(0.5) == pytest.approx(10.0)
    assert a(1.5) == pytest.approx(20.0)
    assert a(2.5) == pytest.approx(30.0)


def test_amplitude_validation():
    with pytest.raises(ValueError):
        Amplitude([0.0, 0.0], [1.0, 2.0])          # non-increasing time
    with pytest.raises(ValueError):
        Amplitude([0.0], [1.0])                     # too few points


# ------------------------------------------------------------------
# Solver integration: time-dependent prescribed displacement
# ------------------------------------------------------------------

def _single_quad():
    mesh = Mesh()
    mesh.add_node(0, 0.0, 0.0); mesh.add_node(1, 1.0, 0.0)
    mesh.add_node(2, 1.0, 1.0); mesh.add_node(3, 0.0, 1.0)
    mesh.add_element(0, [0, 1, 2, 3], "QUAD4")
    mat = LinearViscoelastic(1000.0, 0.3, g_i=[0.5], tau_i=[1.0])
    return DynamicSolver(mesh, mat, rho=1000.0, material_params={},
                         element_type="Q4_UP", tol=1e-8)


def test_time_dependent_bc_ramp():
    solver = _single_quad()
    ramp = Amplitude([0.0, 1.0], [0.0, 1.0], method="linear")

    # base prescribed: top nodes pulled to uy=0.1 at full amplitude; bottom fixed
    bc_dofs, base_vals, amps = [], [], []
    for nid in [0, 1]:
        bc_dofs += [nid * 2, nid * 2 + 1]; base_vals += [0.0, 0.0]; amps += [None, None]
    for nid in [2, 3]:
        bc_dofs += [nid * 2, nid * 2 + 1]; base_vals += [0.0, 0.1]; amps += [None, ramp]
    solver.set_prescribed_dofs(bc_dofs, base_vals, amplitudes=amps)

    # Step 1 to t=0.5  -> amplitude 0.5 -> top uy ~ 0.05
    assert solver.solve_step(dt=0.5) >= 0
    assert solver.time == pytest.approx(0.5)
    top_uy = solver.u[2 * 2 + 1]               # node 2, uy
    assert top_uy == pytest.approx(0.05, rel=1e-6)

    # Step 2 to t=1.0  -> amplitude 1.0 -> top uy ~ 0.10
    assert solver.solve_step(dt=0.5) >= 0
    assert solver.time == pytest.approx(1.0)
    assert solver.u[2 * 2 + 1] == pytest.approx(0.10, rel=1e-6)


def test_time_not_advanced_on_failure():
    solver = _single_quad()
    solver.set_prescribed_dofs([0, 1, 2, 3], [0.0, 0.0, 0.0, 0.0])
    t0 = solver.time
    solver.solve_step(dt=0.1)
    assert solver.time == pytest.approx(t0 + 0.1)
