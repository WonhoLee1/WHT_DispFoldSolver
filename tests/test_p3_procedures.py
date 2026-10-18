"""
test_p3_procedures.py
=====================
Verification suite for Package P3:
1. *STATIC, STABILIZE keyword reader parsing (flag, direct factor, FACTOR=..., ALLSDVTOL=...).
2. Step & Model Python API stabilization configuration.
3. DynamicSolver2D stabilization on/off comparison (ALLSD calculation, bit-identical when off).
4. DynamicSolver3D stabilization on/off comparison (both solve_step and _solve_step_with_hooks).
5. Output pipeline integration (ALLSD recorded via HistoryOutputRequest).
"""

import pytest
import numpy as np

from dispsolver.io.abaqus_reader import read_inp
from dispsolver.model.model import Model
from dispsolver.mesh2d import Mesh2D
from dispsolver.mesh3d import Mesh3D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.solver.stabilization import AbaqusViscousStabilization


# ─── 1. Keyword Reader Parsing Tests ──────────────────────────────────────────

def test_static_keyword_without_stabilize():
    deck = """*HEADING
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=EALL
1, 1, 2, 3, 4, 5, 6, 7, 8
*SOLID SECTION, ELSET=EALL, MATERIAL=M1
*MATERIAL, NAME=M1
*ELASTIC
210000.0, 0.3
*STEP
*STATIC
0.1, 1.0, 1e-5, 0.1
*END STEP
"""
    model = read_inp(deck, dim=3)
    step = [s for s in model.steps.values() if s.name != "Initial"][0]
    assert step.procedure == "STATIC"
    assert step.stabilize is None


def test_static_keyword_with_bare_stabilize():
    deck = """*HEADING
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=EALL
1, 1, 2, 3, 4, 5, 6, 7, 8
*SOLID SECTION, ELSET=EALL, MATERIAL=M1
*MATERIAL, NAME=M1
*ELASTIC
210000.0, 0.3
*STEP
*STATIC, STABILIZE
0.1, 1.0, 1e-5, 0.1
*END STEP
"""
    model = read_inp(deck, dim=3)
    step = [s for s in model.steps.values() if s.name != "Initial"][0]
    assert step.procedure == "STATIC"
    assert step.stabilize is not None
    assert step.stabilize["enabled"] is True
    assert step.stabilize["factor"] == pytest.approx(2e-4)
    assert step.stabilize["allsdvtol"] == pytest.approx(0.05)


def test_static_keyword_with_factor_and_allsdvtol():
    deck = """*HEADING
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=EALL
1, 1, 2, 3, 4, 5, 6, 7, 8
*SOLID SECTION, ELSET=EALL, MATERIAL=M1
*MATERIAL, NAME=M1
*ELASTIC
210000.0, 0.3
*STEP
*STATIC, STABILIZE, FACTOR=0.0005, ALLSDVTOL=0.02
0.1, 1.0, 1e-5, 0.1
*END STEP
"""
    model = read_inp(deck, dim=3)
    step = [s for s in model.steps.values() if s.name != "Initial"][0]
    assert step.procedure == "STATIC"
    assert step.stabilize is not None
    assert step.stabilize["enabled"] is True
    assert step.stabilize["factor"] == pytest.approx(0.0005)
    assert step.stabilize["allsdvtol"] == pytest.approx(0.02)


def test_static_keyword_with_direct_factor_assignment():
    deck = """*HEADING
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
3, 1.0, 1.0, 0.0
4, 0.0, 1.0, 0.0
5, 0.0, 0.0, 1.0
6, 1.0, 0.0, 1.0
7, 1.0, 1.0, 1.0
8, 0.0, 1.0, 1.0
*ELEMENT, TYPE=C3D8, ELSET=EALL
1, 1, 2, 3, 4, 5, 6, 7, 8
*SOLID SECTION, ELSET=EALL, MATERIAL=M1
*MATERIAL, NAME=M1
*ELASTIC
210000.0, 0.3
*STEP
*STATIC, STABILIZE=0.001
0.1, 1.0, 1e-5, 0.1
*END STEP
"""
    model = read_inp(deck, dim=3)
    step = [s for s in model.steps.values() if s.name != "Initial"][0]
    assert step.procedure == "STATIC"
    assert step.stabilize is not None
    assert step.stabilize["enabled"] is True
    assert step.stabilize["factor"] == pytest.approx(0.001)


# ─── 2. Python API Model & Step Tests ─────────────────────────────────────────

def test_model_static_step_api():
    model = Model(name="TestModel")
    s1 = model.StaticStep(name="Step-1", stabilize=True)
    assert s1.stabilize is not None
    assert s1.stabilize["enabled"] is True
    assert s1.stabilize["factor"] == pytest.approx(2e-4)

    s2 = model.StaticStep(name="Step-2", stabilize=1e-3, allsdvtol=0.01)
    assert s2.stabilize is not None
    assert s2.stabilize["factor"] == pytest.approx(1e-3)
    assert s2.stabilize["allsdvtol"] == pytest.approx(0.01)

    s3 = model.StaticStep(name="Step-3")
    assert s3.stabilize is None
    s3.setValues(stabilize=True, factor=5e-4)
    assert s3.stabilize is not None
    assert s3.stabilize["factor"] == pytest.approx(5e-4)


# ─── 3. 2D Solver Stabilization (Mechanical Verification) ─────────────────────

def _make_2d_mesh():
    mesh = Mesh2D()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(3, 1.0, 1.0)
    mesh.add_node(4, 0.0, 1.0)
    mesh.add_element(1, [1, 2, 3, 4], "CPE4")
    return mesh


def test_2d_stabilization_on_vs_off():
    mesh = _make_2d_mesh()
    mat = {"E": 1000.0, "nu": 0.3, "density": 1.0}

    # Case A: Stabilize OFF
    solver_off = DynamicSolver2D(mesh=mesh, materials={1: mat}, nlgeom=False)
    # Fix left edge (nodes 1 and 4)
    solver_off.set_dirichlet_bc(0, 0.0)  # node 1 ux
    solver_off.set_dirichlet_bc(1, 0.0)  # node 1 uy
    solver_off.set_dirichlet_bc(6, 0.0)  # node 4 ux
    solver_off.set_dirichlet_bc(7, 0.0)  # node 4 uy
    # Pull node 2 and 3 in x
    solver_off.set_dirichlet_bc(2, 0.1)  # node 2 ux
    solver_off.set_dirichlet_bc(4, 0.1)  # node 3 ux

    ok_off, _ = solver_off.solve_step(dt=0.1)
    assert ok_off
    assert solver_off.energy_allsd == pytest.approx(0.0)
    u_off = solver_off.u.copy()

    # Case B: Stabilize ON
    mesh_on = _make_2d_mesh()
    solver_on = DynamicSolver2D(mesh=mesh_on, materials={1: mat}, nlgeom=False)
    solver_on.set_dirichlet_bc(0, 0.0)
    solver_on.set_dirichlet_bc(1, 0.0)
    solver_on.set_dirichlet_bc(6, 0.0)
    solver_on.set_dirichlet_bc(7, 0.0)
    solver_on.set_dirichlet_bc(2, 0.1)
    solver_on.set_dirichlet_bc(4, 0.1)

    solver_on._stabilization = AbaqusViscousStabilization(damping_factor=2e-4, enabled=True)
    ok_on, _ = solver_on.solve_step(dt=0.1)
    assert ok_on
    assert solver_on.energy_allsd > 0.0
    u_on = solver_on.u.copy()

    # Displacement difference is very small (O(damping_factor))
    assert np.linalg.norm(u_on - u_off) / np.linalg.norm(u_off) < 1e-3

    # Case C: Stabilize OFF must be bit-identical to another solve with no stab set
    mesh_c = _make_2d_mesh()
    solver_c = DynamicSolver2D(mesh=mesh_c, materials={1: mat}, nlgeom=False)
    solver_c.set_dirichlet_bc(0, 0.0)
    solver_c.set_dirichlet_bc(1, 0.0)
    solver_c.set_dirichlet_bc(6, 0.0)
    solver_c.set_dirichlet_bc(7, 0.0)
    solver_c.set_dirichlet_bc(2, 0.1)
    solver_c.set_dirichlet_bc(4, 0.1)
    solver_c._stabilization = AbaqusViscousStabilization(damping_factor=2e-4, enabled=False)
    ok_c, _ = solver_c.solve_step(dt=0.1)
    assert ok_c
    assert solver_c.energy_allsd == pytest.approx(0.0)
    np.testing.assert_array_equal(solver_c.u, u_off)


# ─── 4. 3D Solver Stabilization (Mechanical Verification) ─────────────────────

def _make_3d_mesh():
    mesh = Mesh3D()
    mesh.add_node(1, 0.0, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0, 0.0)
    mesh.add_node(3, 1.0, 1.0, 0.0)
    mesh.add_node(4, 0.0, 1.0, 0.0)
    mesh.add_node(5, 0.0, 0.0, 1.0)
    mesh.add_node(6, 1.0, 0.0, 1.0)
    mesh.add_node(7, 1.0, 1.0, 1.0)
    mesh.add_node(8, 0.0, 1.0, 1.0)
    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], "C3D8")
    return mesh


def test_3d_stabilization_solve_step_and_hooks():
    mesh = _make_3d_mesh()
    mat = {"E": 1000.0, "nu": 0.3, "density": 1.0}

    # Case A: solve_step path with stabilize ON
    solver = DynamicSolver3D(mesh=mesh, materials={1: mat}, nlgeom=False)
    # Fix x=0 face: nodes 1, 4, 5, 8
    for nid in [1, 4, 5, 8]:
        d = 3 * (nid - 1)
        solver.set_dirichlet_bc(d, 0.0)
        solver.set_dirichlet_bc(d + 1, 0.0)
        solver.set_dirichlet_bc(d + 2, 0.0)
    # Pull x=1 face: nodes 2, 3, 6, 7
    for nid in [2, 3, 6, 7]:
        d = 3 * (nid - 1)
        solver.set_dirichlet_bc(d, 0.05)

    solver._stabilization = AbaqusViscousStabilization(damping_factor=5e-4, enabled=True)
    ok, _ = solver.solve_step(dt=0.1)
    assert ok
    assert solver.energy_allsd > 0.0
    energy_step1 = solver.energy_allsd

    # Next increment accumulates more energy
    for nid in [2, 3, 6, 7]:
        d = 3 * (nid - 1)
        solver.set_dirichlet_bc(d, 0.10)
    ok2, _ = solver.solve_step(dt=0.1)
    assert ok2
    assert solver.energy_allsd > energy_step1

    # Case B: _solve_step_with_hooks path with stabilize ON
    mesh_b = _make_3d_mesh()
    solver_b = DynamicSolver3D(mesh=mesh_b, materials={1: mat}, nlgeom=False)
    for nid in [1, 4, 5, 8]:
        d = 3 * (nid - 1)
        solver_b.set_dirichlet_bc(d, 0.0)
        solver_b.set_dirichlet_bc(d + 1, 0.0)
        solver_b.set_dirichlet_bc(d + 2, 0.0)
    for nid in [2, 3, 6, 7]:
        d = 3 * (nid - 1)
        solver_b.set_dirichlet_bc(d, 0.05)

    solver_b._stabilization = AbaqusViscousStabilization(damping_factor=5e-4, enabled=True)
    ok_hooks, _ = solver_b._solve_step_with_hooks(dt=0.1, t_next=0.1, sys=None, sensor_mgr=None, hooks=[])
    assert ok_hooks
    assert solver_b.energy_allsd > 0.0
    assert solver_b.energy_allsd == pytest.approx(energy_step1, rel=1e-5)


# ─── 5. Output Pipeline Integration (ALLSD) ───────────────────────────────────

def test_output_pipeline_records_allsd():
    model = Model(name="PipeModel", dim=2)
    part = model.Part("Plate", dim=2)
    part.add_node(1, [0.0, 0.0])
    part.add_node(2, [1.0, 0.0])
    part.add_node(3, [1.0, 1.0])
    part.add_node(4, [0.0, 1.0])
    part.add_element(1, "CPE4", [1, 2, 3, 4])
    part.create_set("AllElems", elements=[1])
    part.SectionAssignment(region="AllElems", sectionName="Sec1")

    mat = model.Material("Mat1")
    mat.Elastic(E=1000.0, nu=0.3)
    model.SolidSection("Sec1", material="Mat1", thickness=1.0)

    part.create_set("FixedNodes", nodes=[1, 4])
    part.create_set("PulledNodes", nodes=[2, 3])

    inst = model.root_assembly.Instance("Plate-1", part)

    step = model.StaticStep(name="Step-1", time_period=0.2, dt_init=0.1, dt_min=0.01, dt_max=0.1,
                            stabilize=True, factor=1e-3)
    model.DisplacementBC("FixLeft", createStepName="Step-1", region="Plate-1.FixedNodes", u1=0.0, u2=0.0)
    model.DisplacementBC("PullRight", createStepName="Step-1", region="Plate-1.PulledNodes", u1=0.05)

    step.HistoryOutputRequest(name="H-Out", variables=["ALLSD", "ALLIE"], frequency=1)

    solver, sys = model.create_solver2d("Step-1")
    odb = solver.solve_with_odb(step=step, sys=sys, verbose=False)

    assert "Step-1" in odb.steps
    odb_step = odb.steps["Step-1"]
    assert "WholeModel" in odb_step.historyRegions
    hr = odb_step.historyRegions["WholeModel"]
    assert "ALLSD" in hr.historyOutputs

    ho_allsd = hr.historyOutputs["ALLSD"]
    assert len(ho_allsd.data) > 0
    # Final value should match solver.energy_allsd > 0
    assert ho_allsd.values[-1] == pytest.approx(solver.energy_allsd)
    assert solver.energy_allsd > 0.0
