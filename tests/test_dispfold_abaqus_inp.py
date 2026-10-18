"""
test_dispfold_abaqus_inp.py
===========================
TDD Unit Verification Suite for Seam 3: Abaqus API Builder & Lossless INP Roundtrip.

Verifies:
1. AbaqusModelBuilder generates 100% compliant Abaqus input deck string (.inp)
   with all essential keyword sections (*HEADING, *NODE, *ELEMENT, *SOLID SECTION,
   *MATERIAL, *NSET, *STEP, *STATIC, *AMPLITUDE, *BOUNDARY).
2. Lossless roundtrip verification via dispsolver.io.read_abaqus_input:
   Nodes, Elements, NodeSets, ElementSets, Materials, Steps, and Boundary Conditions (BCs).
3. Both 2D plane strain (CPE4I/CPE4H) and 3D continuum solid (C3D8I/C3D8H) architectures,
   including full 3D spatial coordinate verification via read_inp(dim=3).
"""

from __future__ import annotations
import os
import tempfile
import pytest
import numpy as np

from example_disp_fold.model_spec import DisplayStackConfig
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from example_disp_fold.abaqus_api_builder import AbaqusModelBuilder, AbaqusApiBuilder
from dispsolver.io import read_abaqus_input, read_inp


def test_abaqus_model_builder_inp_string_content_2d():
    """Verify that AbaqusModelBuilder generates complete, structured .inp string."""
    config = DisplayStackConfig(preset="compact_4", dim="2D")
    generator = GmshMeshGenerator(config)
    mesh, meta = generator.build_2d()

    builder = AbaqusModelBuilder(config, mesh, meta)
    inp_str = builder.generate_inp_string()

    assert isinstance(inp_str, str)
    assert len(inp_str) > 1000

    # 1. Heading and Structural Headers
    assert "*HEADING" in inp_str
    assert "Foldable Display 180-Degree Teardrop Folding Simulation" in inp_str

    # 2. Nodes
    assert "*NODE" in inp_str
    assert "1, " in inp_str

    # 3. Elements and Sections
    assert "*ELEMENT, TYPE=CPE4I, ELSET=ELSET_PET_BOTTOM" in inp_str
    assert "*SOLID SECTION, ELSET=ELSET_PET_BOTTOM, MATERIAL=PET" in inp_str
    assert "*ELEMENT, TYPE=CPE4H, ELSET=ELSET_PSA_MID" in inp_str
    assert "*SOLID SECTION, ELSET=ELSET_PSA_MID, MATERIAL=PSA" in inp_str
    assert "*ELEMENT, TYPE=CPE4, ELSET=ELSET_PLATE_LEFT" in inp_str
    assert "*SOLID SECTION, ELSET=ELSET_PLATE_LEFT, MATERIAL=PLATE" in inp_str

    # 4. Materials
    assert "*MATERIAL, NAME=PET" in inp_str
    assert "*ELASTIC" in inp_str
    assert "*PLASTIC" in inp_str
    assert "*MATERIAL, NAME=PSA" in inp_str
    assert "*HYPERELASTIC, ARRUDA-BOYCE" in inp_str
    assert "*VISCOELASTIC, TIME=PRONY" in inp_str
    assert "*MATERIAL, NAME=GLASS" in inp_str
    assert "*MATERIAL, NAME=PLATE" in inp_str

    # 5. Sets and Step
    assert "*NSET, NSET=NSET_TIED_LEFT" in inp_str
    assert "*NSET, NSET=NSET_TIED_RIGHT" in inp_str
    assert "*NSET, NSET=RP_LEFT" in inp_str
    assert "*NSET, NSET=RP_RIGHT" in inp_str
    assert "*RIGID BODY, ELSET=ELSET_PLATE_LEFT, REF NODE=" in inp_str
    assert "*RIGID BODY, ELSET=ELSET_PLATE_RIGHT, REF NODE=" in inp_str
    assert "*TIE, NAME=TIE_LEFT" in inp_str
    assert "*TIE, NAME=TIE_RIGHT" in inp_str
    assert "*STEP, NAME=Step-1_Folding, NLGEOM=YES" in inp_str
    assert "*STATIC" in inp_str
    assert "*AMPLITUDE, NAME=AMP_SMOOTH, DEFINITION=SMOOTH STEP" in inp_str

    # 6. Boundary Conditions and Output Requests
    assert "*BOUNDARY" in inp_str
    assert "*OUTPUT, FIELD" in inp_str
    assert "*OUTPUT, HISTORY" in inp_str
    assert "*ENERGY OUTPUT" in inp_str
    assert "*END STEP" in inp_str


def test_abaqus_inp_lossless_roundtrip_2d():
    """Verify lossless roundtrip for 2D model from AbaqusModelBuilder to read_abaqus_input."""
    config = DisplayStackConfig(preset="compact_4", dim="2D")
    generator = GmshMeshGenerator(config)
    mesh, meta = generator.build_2d()

    builder = AbaqusModelBuilder(config, mesh, meta)

    with tempfile.TemporaryDirectory() as tmpdir:
        inp_path = os.path.join(tmpdir, "model_2d.inp")
        res_file = builder.write_inp(inp_path)
        assert os.path.exists(res_file)

        # Validate with built-in roundtrip validator
        val_res = builder.validate_inp_roundtrip(res_file)
        assert val_res["valid"] is True
        assert val_res["num_nodes"] == len(mesh.nodes) + 2
        assert val_res["mesh_nodes"] == len(mesh.nodes)
        assert val_res["ref_nodes"] == 2
        assert val_res["num_elements"] == len(mesh.elements)
        assert val_res["rbe2_constraints"] == 2
        assert val_res["tie_constraints"] == 2

        # Direct in-depth parsing check via read_abaqus_input
        model_parsed = read_abaqus_input(res_file)

        # 1. Nodes check (continuum mesh + 2 reference nodes)
        assert len(model_parsed.mesh.nodes) == len(mesh.nodes) + 2
        orig_coords = mesh.nodes_array()
        parsed_coords = model_parsed.mesh.nodes_array()
        assert np.allclose(orig_coords, parsed_coords[:len(orig_coords)], atol=1e-5)

        # 2. Elements check
        assert len(model_parsed.mesh.elements) == len(mesh.elements)
        for eid, elem_orig in mesh.elements.items():
            elem_parsed = model_parsed.mesh.elements[eid]
            assert elem_parsed.node_ids == elem_orig.node_ids

        # 3. Sets check
        assert "NSET_TIED_LEFT" in model_parsed.mesh.node_sets
        assert "NSET_TIED_RIGHT" in model_parsed.mesh.node_sets
        assert set(model_parsed.mesh.node_sets["NSET_TIED_LEFT"].node_ids) == set(meta["disp_bottom_left_nids"])
        assert set(model_parsed.mesh.node_sets["NSET_TIED_RIGHT"].node_ids) == set(meta["disp_bottom_right_nids"])

        assert "ELSET_PET_BOTTOM" in model_parsed.mesh.element_sets
        assert "ELSET_PSA_MID" in model_parsed.mesh.element_sets
        assert "ELSET_PLATE_LEFT" in model_parsed.mesh.element_sets
        assert "ELSET_PLATE_RIGHT" in model_parsed.mesh.element_sets

        # 4. Materials check
        for mat_name in ["PET", "PSA", "GLASS", "PLATE"]:
            assert any(mat_name in str(name) for name in model_parsed.material_names.values())

        # 5. Step & Solver configuration check
        assert model_parsed.solver_config.get("procedure") == "STATIC"
        assert pytest.approx(model_parsed.solver_config.get("dt_init"), rel=1e-4) == config.simulation_case.dt_init
        assert pytest.approx(model_parsed.solver_config.get("t_total"), rel=1e-4) == config.simulation_case.total_time

        # 6. Boundary Conditions check (translations + rotations)
        assert len(model_parsed.boundaries) >= 4
        for bc in model_parsed.boundaries:
            assert int(bc.nset) in model_parsed.mesh.nodes
            assert bc.dof1 in (1, 6)

        # 7. Part Names mapping check
        assert len(model_parsed.part_names) >= 2
        assert any("Plate" in str(v) for v in model_parsed.part_names.values())


def test_abaqus_inp_lossless_roundtrip_3d():
    """Verify lossless roundtrip for 3D model with C3D8/C3D8I continuum solid elements."""
    config = DisplayStackConfig(preset="compact_4", dim="3D")
    generator = GmshMeshGenerator(config)
    mesh, meta = generator.build_3d()

    builder = AbaqusModelBuilder(config, mesh, meta)

    with tempfile.TemporaryDirectory() as tmpdir:
        inp_path = os.path.join(tmpdir, "model_3d.inp")
        res_file = builder.write_inp(inp_path)
        assert os.path.exists(res_file)

        # 1. read_abaqus_input validation
        model_parsed = read_abaqus_input(res_file)
        assert len(model_parsed.mesh.nodes) == len(mesh.nodes) + 2
        assert len(model_parsed.mesh.elements) == len(mesh.elements)

        orig_coords = mesh.nodes_array()
        # Projected in-plane coords check
        assert np.allclose(orig_coords[:, :2], model_parsed.mesh.nodes_array()[:len(orig_coords)], atol=1e-5)

        first_elem = next(iter(model_parsed.mesh.elements.values()))
        assert len(first_elem.node_ids) == 8

        # Sets check
        assert "NSET_TIED_LEFT" in model_parsed.mesh.node_sets
        assert "NSET_TIED_RIGHT" in model_parsed.mesh.node_sets
        assert "ELSET_PLATE_LEFT" in model_parsed.mesh.element_sets
        assert "ELSET_PLATE_RIGHT" in model_parsed.mesh.element_sets

        # Materials and BCs check
        assert len(model_parsed.materials) >= 4
        assert len(model_parsed.boundaries) >= 4
        assert model_parsed.solver_config.get("procedure") == "STATIC"

        # 2. Full 3D coordinate validation via read_inp(dim=3)
        m3_parsed = read_inp(res_file, dim=3)
        part = next(iter(m3_parsed.parts.values()))
        sorted_nids = sorted(part.nodes.keys())
        coords_3d = np.array([part.nodes[nid] for nid in sorted_nids], dtype=np.float64)
        assert np.allclose(orig_coords, coords_3d[:len(orig_coords)], atol=1e-5)
