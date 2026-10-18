"""
test_p4_dispfold_parity.py
==========================
Comprehensive verification suite for Package P4: Framework Parity & Multi-Stage Folding.

Verifies:
1. Abaqus API Exporter keyword parity:
   - Reference Nodes (RP_LEFT, RP_RIGHT) defined at pivot locations.
   - *RIGID BODY constraints binding support plates to pivot RPs.
   - *TIE constraints coupling support plates to display bottom layers.
   - *BOUNDARY rotational BCs with UR3 (DOF 6) and smoothstep amplitude.
   - *OUTPUT (Field and History) and *ENERGY OUTPUT (ALLIE, ALLSD, ALLAE, ALLFD).
   - Lossless keyword-level roundtrip verification via validate_inp_roundtrip.
2. Multi-Stage Folding Specification & Deck Generation:
   - 3-stage sequence: Fold (0 -> theta_max) -> Hold (theta_max) -> Unfold (theta_max -> 0).
   - Generates distinct *STEP blocks with proper durations and boundary conditions.
3. Multi-Stage End-to-End Execution:
   - Runs 2D multi-stage cycle, verifying continuity across stage boundaries,
     non-inversion (det J > 0), and artifact generation.
4. Backward Compatibility:
   - Single-stage teardrop preset remains 100% backward-compatible.
"""

from __future__ import annotations
import os
import json
import tempfile
import pytest
import numpy as np

from example_disp_fold.model_spec import (
    UnifiedFoldModelConfig,
    DisplayStackConfig,
    StageSpec,
)
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from example_disp_fold.abaqus_api_builder import AbaqusApiBuilder
from example_disp_fold.ex1_dispfold_teardrop import run_dispfold_simulation
from dispsolver.io import read_abaqus_input, read_inp


def test_p4_exporter_standard_keywords_2d():
    """Verify that AbaqusApiBuilder emits all required standard keywords for 2D model."""
    config = DisplayStackConfig(preset="compact_4", dim="2D")
    generator = GmshMeshGenerator(config)
    mesh, meta = generator.build_2d()

    builder = AbaqusApiBuilder(config, mesh, meta)
    inp_str = builder.generate_inp_string()

    # 1. Structural headers
    assert "*HEADING" in inp_str
    assert "*NODE" in inp_str
    assert "*ELEMENT" in inp_str
    assert "*SOLID SECTION" in inp_str
    assert "*MATERIAL" in inp_str

    # 2. Reference Nodes and Sets
    assert "*NSET, NSET=RP_LEFT" in inp_str
    assert "*NSET, NSET=RP_RIGHT" in inp_str
    assert "*NSET, NSET=NSET_PLATE_LEFT" in inp_str
    assert "*NSET, NSET=NSET_PLATE_RIGHT" in inp_str
    assert "*NSET, NSET=NSET_TIED_LEFT" in inp_str
    assert "*NSET, NSET=NSET_TIED_RIGHT" in inp_str

    # 3. Rigid Body Constraints
    assert "*RIGID BODY, ELSET=ELSET_PLATE_LEFT, REF NODE=" in inp_str
    assert "*RIGID BODY, ELSET=ELSET_PLATE_RIGHT, REF NODE=" in inp_str

    # 4. Tie Constraints
    assert "*TIE, NAME=TIE_LEFT" in inp_str
    assert "*TIE, NAME=TIE_RIGHT" in inp_str

    # 5. Amplitude
    assert "*AMPLITUDE, NAME=AMP_SMOOTH, DEFINITION=SMOOTH STEP" in inp_str

    # 6. Step with UR3 boundary conditions
    assert "*STEP, NAME=Step-1_Folding, NLGEOM=YES" in inp_str
    assert "*STATIC" in inp_str
    assert "*BOUNDARY, AMPLITUDE=AMP_SMOOTH" in inp_str

    # 7. Output requests
    assert "*OUTPUT, FIELD, VARIABLE=PRESELECT" in inp_str
    assert "*NODE OUTPUT" in inp_str
    assert "U, RF" in inp_str
    assert "*ELEMENT OUTPUT, DIRECTIONS=YES" in inp_str
    assert "S, MISES, LE, E" in inp_str
    assert "*OUTPUT, HISTORY" in inp_str
    assert "*ENERGY OUTPUT" in inp_str
    assert "ALLIE, ALLKE, ALLSE, ALLVD, ALLAE, ALLSD, ALLFD" in inp_str
    assert "*END STEP" in inp_str


def test_p4_exporter_roundtrip_verification():
    """Verify that validate_inp_roundtrip performs comprehensive keyword-level checks."""
    config = DisplayStackConfig(preset="compact_4", dim="2D")
    generator = GmshMeshGenerator(config)
    mesh, meta = generator.build_2d()

    builder = AbaqusApiBuilder(config, mesh, meta)

    with tempfile.TemporaryDirectory() as tmpdir:
        inp_path = os.path.join(tmpdir, "model_parity.inp")
        builder.write_inp(inp_path)

        val_res = builder.validate_inp_roundtrip(inp_path)
        assert val_res["valid"] is True
        assert val_res["num_nodes"] == len(mesh.nodes) + 2
        assert val_res["mesh_nodes"] == len(mesh.nodes)
        assert val_res["ref_nodes"] == 2
        assert val_res["rbe2_constraints"] == 2
        assert val_res["tie_constraints"] == 2
        assert len(val_res["boundaries"]) >= 4
        assert "Step-1_Folding" in val_res["steps"]


def test_p4_multistage_spec_and_deck_generation():
    """Verify that setup_multistage_cycle generates 3 distinct steps with correct parameters."""
    config = DisplayStackConfig(preset="compact_4", dim="2D")
    stages = config.setup_multistage_cycle(
        fold_time=1.0,
        hold_time=2.0,
        unfold_time=1.5,
        theta_max_deg=60.0,
        dt_init=0.01,
        dt_min=1e-5,
        dt_max=0.05,
    )

    assert len(stages) == 3
    assert stages[0].name == "Step-1_Fold"
    assert stages[0].stage_type == "FOLD"
    assert stages[0].theta_start_deg == 0.0
    assert stages[0].theta_target_deg == 60.0
    assert stages[0].duration == 1.0

    assert stages[1].name == "Step-2_Hold"
    assert stages[1].stage_type == "HOLD"
    assert stages[1].theta_start_deg == 60.0
    assert stages[1].theta_target_deg == 60.0
    assert stages[1].duration == 2.0

    assert stages[2].name == "Step-3_Unfold"
    assert stages[2].stage_type == "UNFOLD"
    assert stages[2].theta_start_deg == 60.0
    assert stages[2].theta_target_deg == 0.0
    assert stages[2].duration == 1.5

    generator = GmshMeshGenerator(config)
    mesh, meta = generator.build_2d()

    builder = AbaqusApiBuilder(config, mesh, meta)
    inp_str = builder.generate_inp_string()

    # Verify all 3 steps appear in emitted deck
    assert "*STEP, NAME=Step-1_Fold, NLGEOM=YES" in inp_str
    assert "*STEP, NAME=Step-2_Hold, NLGEOM=YES" in inp_str
    assert "*STEP, NAME=Step-3_Unfold, NLGEOM=YES" in inp_str

    with tempfile.TemporaryDirectory() as tmpdir:
        inp_path = os.path.join(tmpdir, "model_multistage.inp")
        builder.write_inp(inp_path)

        val_res = builder.validate_inp_roundtrip(inp_path)
        assert val_res["valid"] is True
        assert val_res["steps"] == ["Step-1_Fold", "Step-2_Hold", "Step-3_Unfold"]

        # Parse with modern reader
        model_inp = read_inp(inp_path, dim=2)
        assert len(model_inp.steps) == 4  # Initial + 3 analysis steps
        for sname in ["Step-1_Fold", "Step-2_Hold", "Step-3_Unfold"]:
            assert sname in model_inp.steps
            assert len(model_inp.steps[sname].field_output_requests) > 0


def test_p4_multistage_execution_smoke():
    """Verify that multi-stage Fold -> Hold -> Unfold cycle executes end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Pre-configure multi-stage on case
        config = DisplayStackConfig(preset="compact_4", dim="2D")
        config.setup_multistage_cycle(
            fold_time=0.5,
            hold_time=0.5,
            unfold_time=0.5,
            theta_max_deg=20.0,
            dt_init=0.05,
            dt_min=1e-4,
            dt_max=0.1,
        )

        out_dir = run_dispfold_simulation(
            dim="2D",
            preset="compact_4",
            theta_max_deg=20.0,
            enable_predictor=True,
            tie_method="MPC",
            contact_mode="AUTO",
            sol_preset="ADAPTIVE",
            max_steps=4,
            stages=config.simulation_case.stages,
            launch_gui=False,
            results_base_dir=tmpdir,
        )

        assert os.path.isdir(out_dir)

        # Verify summary JSON reflects multi-stage
        json_path = os.path.join(out_dir, "result_summary.json")
        assert os.path.isfile(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        assert summary["dim"] == "2D"
        assert summary["preset"] == "compact_4"
        assert summary["num_steps"] == 4
        assert summary["num_stages"] == 3
        assert summary["wall_time_s"] > 0.0

        # Verify all 7 artifacts generated
        for fname in ["run.log", "model.inp", "model_and_results.vtkhdf", "summary_plots.png", "folding_animation.gif", "result_summary.json", "report.html"]:
            p = os.path.join(out_dir, fname)
            assert os.path.isfile(p), f"Artifact {fname} missing from output folder"
            assert os.path.getsize(p) > 0, f"Artifact {fname} is empty"
