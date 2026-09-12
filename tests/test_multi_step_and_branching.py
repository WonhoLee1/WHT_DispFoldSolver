"""Unit test suite for Multi-Step CAE modeling, Entity Lifecycle management,
PredefinedFields, and Branching Step Tree Execution Engine.
"""

import pytest
import numpy as np
import os
import pickle

from dispsolver.model import Model, Part, GeneralSet, SolidSection, DisplacementBC, Amplitude
from dispsolver.model.step import Step, InitialStep, EntityStatus, ConcentratedLoad, PredefinedField
from dispsolver.solver.step_executor import MultiStepExecutor, StateCheckpoint


def test_entity_lifecycle_status_transitions():
    """Verify BC/Load/Interaction status transitions across sequential and branching steps."""
    model = Model(name="LifecycleTestModel", dim=3)
    part = model.Part(name="Bar", dim=3)
    part.add_node(1, [0.0, 0.0, 0.0])
    part.add_node(2, [1.0, 0.0, 0.0])
    part.add_node(3, [2.0, 0.0, 0.0])
    part.add_node(4, [0.0, 1.0, 0.0])
    part.add_node(5, [1.0, 1.0, 0.0])
    part.add_node(6, [2.0, 1.0, 0.0])
    part.add_node(7, [0.0, 0.0, 1.0])
    part.add_node(8, [1.0, 0.0, 1.0])

    part.add_element(1, "C3D8", [1, 2, 5, 4, 7, 8, 6, 3])
    s_fix = part.create_node_set("FIX_NODES", [1, 4, 7])
    s_load = part.create_node_set("LOAD_NODES", [2, 5, 8])

    inst = model.root_assembly.Instance(name="BAR_1", part=part)

    # Initial Step
    bc_fix = DisplacementBC(name="FIX_BC", region="BAR_1.FIX_NODES", u1=0.0, u2=0.0, u3=0.0)
    model.initial_step.add_boundary_condition(bc_fix)

    pf_vel = model.PredefinedField(name="INIT_VEL", field_type="VELOCITY", region="ALL", values=[0.0, 0.0, 0.0])
    assert "INIT_VEL" in model.initial_step.predefined_fields

    # Step 1: Loading
    step1 = model.Step(name="Step-1", previous="Initial", time_period=1.0)
    # FIX_BC is propagated from Initial
    assert step1.boundary_conditions["FIX_BC"].status == EntityStatus.PROPAGATED

    load1 = ConcentratedLoad(name="PULL_LOAD", region="BAR_1.LOAD_NODES", f1=100.0)
    step1.add_load(load1)
    assert step1.loads["PULL_LOAD"].status == EntityStatus.CREATED

    # Step 2-1 (Branch A): Unloading (PULL_LOAD deactivated)
    step2_1 = model.create_step_branch(branch_name="Step-2-1", parent_step_name="Step-1", time_period=0.5)
    assert step2_1.loads["PULL_LOAD"].status == EntityStatus.PROPAGATED
    step2_1.deactivate_load("PULL_LOAD")
    assert step2_1.loads["PULL_LOAD"].status == EntityStatus.DEACTIVATED
    assert "PULL_LOAD" not in step2_1.get_active_loads()

    # Step 2-2 (Branch B): Overload (PULL_LOAD modified to 300.0)
    step2_2 = model.create_step_branch(branch_name="Step-2-2", parent_step_name="Step-1", time_period=0.5)
    step2_2.modify_load("PULL_LOAD", f1=300.0)
    assert step2_2.loads["PULL_LOAD"].status == EntityStatus.MODIFIED
    assert step2_2.loads["PULL_LOAD"].entity.f1 == 300.0
    assert "PULL_LOAD" in step2_2.get_active_loads()


def test_step_tree_hierarchy_and_checkpoint_rollback(tmp_path):
    """Verify Branching Step Tree Execution with Checkpoint snapshot and restore."""
    model = Model(name="BranchingExecutionTest", dim=3)
    mat = model.Material(name="Steel", mat_type="ELASTIC")
    mat.elastic = (210000.0, 0.3)
    sec = model.SolidSection(name="SteelSec", material="Steel")

    part = model.Part(name="Cube", dim=3)
    coords = [
        (1, 0.0, 0.0, 0.0),
        (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0),
        (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0),
        (6, 1.0, 0.0, 1.0),
        (7, 1.0, 1.0, 1.0),
        (8, 0.0, 1.0, 1.0),
    ]
    for nid, x, y, z in coords:
        part.add_node(nid, [x, y, z])

    part.add_element(1, "C3D8", [1, 2, 3, 4, 5, 6, 7, 8])
    part.create_element_set("ALL", [1])
    part.create_node_set("BOTTOM", [1, 2, 3, 4])
    part.create_node_set("TOP", [5, 6, 7, 8])
    part.assign_section("ALL", "SteelSec")

    inst = model.root_assembly.Instance(name="CUBE_1", part=part)

    # Initial Step
    bc_bot = DisplacementBC(name="BOT_FIX", region="CUBE_1.BOTTOM", u1=0.0, u2=0.0, u3=0.0)
    model.initial_step.add_boundary_condition(bc_bot)

    # Step 1: Pre-compression (Push TOP down by 0.1mm)
    step1 = model.Step(name="Step-1", previous="Initial", time_period=0.1, dt_init=0.05)
    bc_top_step1 = DisplacementBC(name="TOP_PUSH", region="CUBE_1.TOP", u3=-0.1)
    step1.add_boundary_condition(bc_top_step1)

    # Branch A: Step 2-1 (Tension: Pull TOP up to +0.2mm)
    step2_1 = model.create_step_branch(branch_name="Step-2-1", parent_step_name="Step-1", time_period=0.1, dt_init=0.05)
    step2_1.modify_bc("TOP_PUSH", u3=+0.2)

    # Branch B: Step 2-2 (Shear: Shift TOP sideways u1=+0.15mm)
    step2_2 = model.create_step_branch(branch_name="Step-2-2", parent_step_name="Step-1", time_period=0.1, dt_init=0.05)
    step2_2.modify_bc("TOP_PUSH", u1=+0.15, u3=None)

    # Execute Multi-Step Tree Engine
    executor = MultiStepExecutor(model=model)
    save_dir = str(tmp_path / "results")
    results = executor.run_step_tree(save_dir=save_dir, verbose=False)

    # Verification:
    # 1. Step-1, Step-2-1, Step-2-2 results created
    assert "Step-1" in results
    assert "Step-2-1" in results
    assert "Step-2-2" in results

    # 2. Checkpoint restoration verification
    # Step-2-1 and Step-2-2 must both start from Step-1's completed checkpoint!
    ckpt1 = executor.checkpoints["Step-1"]
    ckpt2_1 = executor.checkpoints["Step-2-1"]
    ckpt2_2 = executor.checkpoints["Step-2-2"]

    assert ckpt1.time_accumulated == pytest.approx(0.1, abs=1e-5)
    assert ckpt2_1.time_accumulated == pytest.approx(0.2, abs=1e-5)
    assert ckpt2_2.time_accumulated == pytest.approx(0.2, abs=1e-5)

    # 3. Branch output files exist
    assert os.path.exists(os.path.join(save_dir, "result_branch_Step-1.pkl"))
    assert os.path.exists(os.path.join(save_dir, "result_branch_Step-2-1.pkl"))
    assert os.path.exists(os.path.join(save_dir, "result_branch_Step-2-2.pkl"))
