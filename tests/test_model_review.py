from types import SimpleNamespace
import numpy as np
import pytest

from dispsolver.mesh.mesh import Element, Mesh, Node
from dispsolver.material.type_tags import J2_PLASTIC, NEOHOOKEAN
from dispsolver.postprocess.model_review import (
    compute_pid_thickness_from_mesh,
    compute_pid_thickness_from_result,
    format_model_review,
    print_model_review_from_builder_result,
    print_model_review_from_result,
)

def _build_test_mesh():
    mesh = Mesh()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(3, 1.0, 0.03)
    mesh.add_node(4, 0.0, 0.03)
    mesh.elements[1] = Element(1, [1, 2, 3, 4], 'QUAD4', pid=1)

    mesh.add_node(5, 0.0, 0.08)
    mesh.add_node(6, 1.0, 0.08)
    mesh.elements[2] = Element(2, [4, 3, 6, 5], 'QUAD4', pid=2)

    mesh.add_node(7, -5.0, -0.5)
    mesh.add_node(8, -1.0, -0.5)
    mesh.add_node(9, -1.0, 0.0)
    mesh.add_node(10, -5.0, 0.0)
    mesh.elements[3] = Element(3, [7, 8, 9, 10], 'QUAD4', pid=3)
    return mesh

def test_compute_pid_thickness_from_mesh():
    mesh = _build_test_mesh()
    thick = compute_pid_thickness_from_mesh(mesh)
    assert 1 in thick and 2 in thick and 3 in thick
    assert np.isclose(thick[1], 0.03, atol=1e-6)
    assert np.isclose(thick[2], 0.05, atol=1e-6)
    assert np.isclose(thick[3], 0.50, atol=1e-6)
    assert compute_pid_thickness_from_mesh(None) == {}
    assert compute_pid_thickness_from_mesh(SimpleNamespace()) == {}

def test_compute_pid_thickness_from_result():
    mock_res_with_meta = SimpleNamespace(meta={'pid_thickness': {'1': 0.03, '2': 0.05}})
    thick_from_meta = compute_pid_thickness_from_result(mock_res_with_meta)
    assert thick_from_meta == {1: 0.03, 2: 0.05}

    points = np.array([[0.,0.,0.],[1.,0.,0.],[1.,0.03,0.],[0.,0.03,0.],[1.,0.08,0.],[0.,0.08,0.]])
    conn = np.array([0, 1, 2, 3, 3, 2, 4, 5], dtype=np.int64)
    offsets = np.array([0, 4, 8], dtype=np.int64)
    element_pid = np.array([1, 2], dtype=np.int64)
    mock_res = SimpleNamespace(points=points, connectivity=conn, offsets=offsets, element_pid=element_pid, meta={})
    thick = compute_pid_thickness_from_result(mock_res)
    assert np.isclose(thick[1], 0.03, atol=1e-6)
    assert np.isclose(thick[2], 0.05, atol=1e-6)
    assert compute_pid_thickness_from_result(None) == {}
    assert compute_pid_thickness_from_result(SimpleNamespace()) == {}

def test_format_model_review_with_thickness():
    review_text = format_model_review(
        pid_element_count={1: 100, 2: 200},
        pid_names={1: 'PSA', 2: 'PET'},
        pid_params={1: {'E': 50, 'nu': 0.45}, 2: {'E': 4000, 'nu': 0.3, 'sigma_y0': 80, 'H': 400}},
        pid_types={1: NEOHOOKEAN, 2: J2_PLASTIC},
        pid_part_names={1: 'pid 1', 2: 'Layer 2'},
        pid_element_type={1: 'Q4_UP', 2: 'Q4_UP'},
        pid_thickness={1: 0.0300, 2: 0.0500},
    )
    assert 'MODEL REVIEW' in review_text
    assert 'thickness=0.0300 mm (30.0 um)' in review_text
    assert 'thickness=0.0500 mm (50.0 um)' in review_text
    assert '- pid 1: 100 elements, material=PSA, thickness=0.0300 mm (30.0 um), element=Q4_UP' in review_text
    assert '- Layer 2: 200 elements, material=PET, thickness=0.0500 mm (50.0 um), element=Q4_UP' in review_text

def test_print_model_review_from_builder_result(capsys):
    mesh = _build_test_mesh()
    builder_res = SimpleNamespace(
        mesh=mesh,
        material_names={1: 'PSA', 2: 'PET', 3: 'STEEL'},
        material_params={
            1: {'E': 50, 'nu': 0.45},
            2: {'E': 4000, 'nu': 0.3, 'sigma_y0': 80, 'H': 400},
            3: {'E': 200000, 'nu': 0.3},
        },
        material_types={1: NEOHOOKEAN, 2: J2_PLASTIC, 3: NEOHOOKEAN},
        part_names={3: 'Plate Left'},
    )
    print_model_review_from_builder_result(builder_res, pid_element_type={1: 'Q4_UP', 2: 'Q4_UP', 3: 'Q4'})
    out = capsys.readouterr().out
    assert 'MODEL REVIEW' in out
    assert '- pid 1: 1 elements, material=PSA, thickness=0.0300 mm (30.0 um), element=Q4_UP' in out
    assert '- pid 2: 1 elements, material=PET, thickness=0.0500 mm (50.0 um), element=Q4_UP' in out
    assert '- Plate Left: 1 elements, material=STEEL, thickness=0.5000 mm (500.0 um), element=Q4' in out
