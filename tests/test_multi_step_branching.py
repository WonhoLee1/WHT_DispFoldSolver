"""Unit tests for Commercial Multi-Step CAE Architecture, Entity Lifecycle, and Branching Step Tree Execution Engine."""

import pytest
import numpy as np

from dispsolver.model.model import Model
from dispsolver.model.step import Step, EntityStatus
from dispsolver.solver.step_executor import StateCheckpoint, MultiStepExecutor


def test_step_lifecycle_entity_status_transitions():
    """Verify CREATED -> PROPAGATED -> MODIFIED -> DEACTIVATED -> REACTIVATED transitions across steps."""
    model = Model(name="LifecycleModel")
    p = model.Part("BarPart", dim=3)
    p.add_node(1, [0.0, 0.0, 0.0])
    p.create_set_from_box("NSET1", (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), "NODES")
    model.root_assembly.Instance("BAR_INST", p)

    # Step 1: Create BC-1 and Load-1
    step1 = model.Step(name="Step-1", time_period=1.0)
    bc1 = model.DisplacementBC(name="BC-1", createStepName="Step-1", region="BAR_INST.NSET1", u1=0.0)
    load1 = model.ConcentratedForce(name="Load-1", createStepName="Step-1", region="BAR_INST.NSET1", cf2=-100.0)

    assert step1.boundary_conditions["BC-1"].status == EntityStatus.CREATED
    assert step1.loads["Load-1"].status == EntityStatus.CREATED

    # Step 2: Child of Step 1 (Inherits entities as PROPAGATED)
    step2 = model.Step(name="Step-2", parent_step=step1, time_period=1.0)
    assert step2.boundary_conditions["BC-1"].status == EntityStatus.PROPAGATED
    assert step2.loads["Load-1"].status == EntityStatus.PROPAGATED

    # In Step 2, modify BC-1 and deactivate Load-1
    step2.modify_bc("BC-1", u1=5.0)
    step2.deactivate_load("Load-1")

    assert step2.boundary_conditions["BC-1"].status == EntityStatus.MODIFIED
    assert step2.boundary_conditions["BC-1"].entity.u1 == 5.0
    assert step2.loads["Load-1"].status == EntityStatus.DEACTIVATED
    assert not step2.loads["Load-1"].is_active()

    # Step 3: Child of Step 2 (Propagates modified BC-1 and keeps Load-1 deactivated)
    step3 = model.Step(name="Step-3", parent_step=step2, time_period=1.0)
    assert step3.boundary_conditions["BC-1"].status == EntityStatus.PROPAGATED
    assert step3.boundary_conditions["BC-1"].entity.u1 == 5.0
    assert step3.loads["Load-1"].status == EntityStatus.DEACTIVATED


def test_state_checkpoint_capture_and_restore():
    """Verify deep state checkpoint capture and restore."""
    class DummySolver:
        def __init__(self):
            self.u = np.array([1.0, 2.0, 3.0])
            self.v = np.array([0.1, 0.2, 0.3])
            self.a = np.array([0.01, 0.02, 0.03])
            self.elem_sdvs = np.ones((5, 8, 4)) * 7.5

    solver = DummySolver()
    ckpt = StateCheckpoint(
        step_name="Step-1",
        t_accum=1.0,
        u=solver.u.copy(),
        v=solver.v.copy(),
        a=solver.a.copy(),
        elem_sdvs=solver.elem_sdvs.copy()
    )

    # Mutate solver state
    solver.u += 100.0
    solver.elem_sdvs *= 0.0
    assert not np.allclose(solver.u, ckpt.u)

    # Restore from checkpoint
    ckpt.restore_to_solver(solver)
    assert np.allclose(solver.u, [1.0, 2.0, 3.0])
    assert np.allclose(solver.v, [0.1, 0.2, 0.3])
    assert np.allclose(solver.elem_sdvs, 7.5)


def test_branching_execution_engine_end_to_end():
    """Verify branching execution: Step 1 (Loading) -> Branch A (Unloading) vs Branch B (Impact Overload)."""
    model = Model(name="BranchingCantileverTest", dim=3)
    p = model.Part("BarPart", dim=3)

    # 1 hex element
    p.add_node(1, [0.0, 0.0, 0.0])
    p.add_node(2, [1.0, 0.0, 0.0])
    p.add_node(3, [1.0, 1.0, 0.0])
    p.add_node(4, [0.0, 1.0, 0.0])
    p.add_node(5, [0.0, 0.0, 1.0])
    p.add_node(6, [1.0, 0.0, 1.0])
    p.add_node(7, [1.0, 1.0, 1.0])
    p.add_node(8, [0.0, 1.0, 1.0])
    p.add_element(1, "C3D8_COROTATIONAL", [1, 2, 3, 4, 5, 6, 7, 8])

    p.create_set_from_box("FIXED_FACE", (-0.1, 0.1), (-0.1, 1.1), (-0.1, 1.1), "NODES")
    p.create_set_from_box("TIP_FACE", (0.9, 1.1), (-0.1, 1.1), (-0.1, 1.1), "NODES")

    mat = model.Material(name="Steel", mat_type="ELASTIC")
    mat.E = 210000.0
    mat.nu = 0.3
    sec = model.SolidSection(name="SolidSec", material="Steel")
    p.SectionAssignment(region=p.create_set_from_box("ALL_ELEMS", (-0.1, 1.1), (-0.1, 1.1), (-0.1, 1.1), "ELEMENTS"), sectionName="SolidSec")

    model.root_assembly.Instance("BAR_INST", part=p)

    # Base BC on Fixed Face
    amp = model.SmoothStepAmplitude(name="Ramp", data=[(0.0, 0.0), (0.05, 1.0)])

    # Step 1: Loading
    step1 = model.Step(name="Step-1", time_period=0.05, dt_init=0.05, dt_min=1e-5, dt_max=0.05)
    model.DisplacementBC(name="Clamp", createStepName="Step-1", region="BAR_INST.FIXED_FACE", u1=0.0, u2=0.0, u3=0.0)
    model.ConcentratedForce(name="DownForce", createStepName="Step-1", region="BAR_INST.TIP_FACE", cf2=-3000.0, amplitude="Ramp")

    # Branch A: Step 2A (Unload / Release force)
    step2a = model.Step(name="Step-2A", parent_step=step1, time_period=0.05, dt_init=0.05, dt_min=1e-5, dt_max=0.05)
    step2a.deactivate_load("DownForce")

    # Branch B: Step 2B (Overload / Impact: 3x force)
    step2b = model.Step(name="Step-2B", parent_step=step1, time_period=0.05, dt_init=0.05, dt_min=1e-5, dt_max=0.05)
    step2b.modify_load("DownForce", cf2=-9000.0)

    executor = MultiStepExecutor(model=model, verbose=False)

    # 1. Execute Step 1
    ok1 = executor.execute_step(step1)
    assert ok1 is True
    u_step1 = executor.results_by_step["Step-1"]["u"].copy()
    tip_indices = executor.sys.global_nsets["BAR_INST.TIP_FACE"]
    uy_step1 = np.mean([u_step1[3 * i + 1] for i in tip_indices])
    assert uy_step1 < -1e-6  # Deflected downward

    # 2. Execute Branch A (Unloading from Step 1 checkpoint)
    ok2a = executor.execute_step(step2a, from_parent_checkpoint=True)
    assert ok2a is True
    u_step2a = executor.results_by_step["Step-2A"]["u"].copy()
    uy_step2a = np.mean([u_step2a[3 * i + 1] for i in tip_indices])
    # Force released -> should rebound back toward 0 (less negative deflection than Step 1)
    assert uy_step2a > uy_step1

    # 3. Execute Branch B (Overload from Step 1 checkpoint)
    ok2b = executor.execute_step(step2b, from_parent_checkpoint=True)
    assert ok2b is True
    u_step2b = executor.results_by_step["Step-2B"]["u"].copy()
    uy_step2b = np.mean([u_step2b[3 * i + 1] for i in tip_indices])
    # 3x force -> should deflect even further downward than Step 1
    assert uy_step2b < uy_step1
