"""The folding framework hands the solver the materials it writes into model.inp (PRD: prd_2d_material_wiring).

Before 2026-10-18 ex1_dispfold_teardrop.py passed {"E", "nu"} dictionaries, so PET plasticity and the PSA
hyperelastic/viscoelastic law existed only in the exported deck and every layer was solved as linear elastic.
Expected values come from MaterialSpec (the single source of truth for the stack), not from the solver.
"""
import numpy as np
import pytest

from example_disp_fold.model_spec import DisplayStackConfig
from example_disp_fold.gmsh_mesh_generator import GmshMeshGenerator
from example_disp_fold.abaqus_api_builder import AbaqusApiBuilder
from dispsolver.material.catalog import material_id, is_linear_elastic
from dispsolver.solver2d.dynamic2d import DynamicSolver2D


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    config = DisplayStackConfig(preset="compact_4", dim="2D")
    mesh, meta = GmshMeshGenerator(config).build_2d()
    builder = AbaqusApiBuilder(config, mesh, meta)
    path = tmp_path_factory.mktemp("inp") / "model.inp"
    builder.write_inp(str(path))
    return config, mesh, builder, str(path)


def test_the_solver_materials_are_the_exported_deck_materials(stack):
    config, _, builder, path = stack
    mats = builder.solver_materials(path)
    assert sorted(mats) == [1, 2, 3, 4]                       # PET, PSA, GLASS, PLATE (the mesh's pid rule)
    pet, psa, glass, plate = mats[1], mats[2], mats[3], mats[4]
    assert not is_linear_elastic(material_id(pet))            # PET yields (MaterialSpec.yield_stress = 80 MPa)
    assert not is_linear_elastic(material_id(psa))            # PSA is hyperelastic + viscoelastic
    assert is_linear_elastic(material_id(glass)) and is_linear_elastic(material_id(plate))
    spec = config.materials
    # PSA volumetric stiffness from the spec (E, nu): K = E / (3 (1 - 2 nu)) = 8.33 MPa. The exporter used to write
    # a fixed D = 0.001 (K = 2/D = 2000 MPa), 240 times too stiff once the deck became the solver's input.
    psa_spec = spec["PSA"]
    assert psa.hyperelastic["K"] == pytest.approx(psa_spec.E / (3.0 * (1.0 - 2.0 * psa_spec.nu)), rel=1e-4)
    assert psa.hyperelastic["mu"] == pytest.approx(psa_spec.arruda_boyce_mu)
    assert [g for g, _, _ in psa.viscoelastic_prony] == pytest.approx(psa_spec.prony_g)
    assert pet.elastic[0] == pytest.approx(spec["PET"].E)
    assert glass.elastic[0] == pytest.approx(spec["GLASS"].E)
    assert plate.elastic[0] == pytest.approx(spec["PLATE"].E)


def test_the_solver_built_on_them_runs_nonlinear_materials(stack):
    config, mesh, builder, path = stack
    solver = DynamicSolver2D(mesh, materials=builder.solver_materials(path), nlgeom=True)
    ids = {int(m) for m in solver.elem_mat_types}
    assert any(not is_linear_elastic(i) for i in ids), f"every element is linear elastic: material ids {sorted(ids)}"
    pet_rows = [i for i, e in enumerate(mesh.elements.values()) if e.pid == 1]
    assert pet_rows and all(not is_linear_elastic(int(solver.elem_mat_types[i])) for i in pet_rows)
