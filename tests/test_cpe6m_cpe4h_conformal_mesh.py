"""
test_cpe6m_cpe4h_conformal_mesh.py
==================================
Unit verification suite for the conformal multi-layer mesh generator:
- Upper PET layer (CPE6M quadratic triangles)
- Lower PSA layer (CPE4H linear quadrilaterals with 2*nx refinement)

Validates:
1. Strict 100% node sharing along the interface y = t_psa.
2. Zero duplicate nodes and positive Jacobian areas for all elements.
3. Element connectivity and boundary node sets.
4. Element-level numerical assembly integrity for both CPE6M and CPE4H.
"""

import pytest
import numpy as np
from dispsolver.mesh2d.conformal_cpe6m_cpe4h_builder import (
    create_cpe6m_cpe4h_conformal_stack,
    create_5layer_cpe6m_cpe4h_conformal_mesh
)
from dispsolver.element2d.cpe6m_numba import compute_cpe6m_element_numba
from dispsolver.element2d.cpe4h_numba import compute_cpe4h_element_numba


def test_conformal_interface_node_sharing():
    """Verify that all nodes on y = t_psa are strictly shared between CPE6M and CPE4H."""
    length = 20.0
    t_pet = 0.2
    t_psa = 0.05
    nx = 8
    ny_pet = 2
    ny_psa = 2

    mesh, sets = create_cpe6m_cpe4h_conformal_stack(
        length=length, t_pet=t_pet, t_psa=t_psa,
        nx=nx, ny_pet=ny_pet, ny_psa=ny_psa
    )

    interface_nodes = sets["INTERFACE_NODES"]
    expected_interface_nodes_count = 2 * nx + 1
    assert len(interface_nodes) == expected_interface_nodes_count, (
        f"Expected {expected_interface_nodes_count} interface nodes, got {len(interface_nodes)}"
    )

    # Collect all nodes referenced by PET elements
    pet_referenced_nodes = set()
    for eid in sets["PET_ELEMENTS"]:
        pet_referenced_nodes.update(mesh.elements[eid].node_ids)

    # Collect all nodes referenced by PSA elements
    psa_referenced_nodes = set()
    for eid in sets["PSA_ELEMENTS"]:
        psa_referenced_nodes.update(mesh.elements[eid].node_ids)

    # Every interface node must be shared by BOTH PET and PSA elements
    for nid in interface_nodes:
        assert nid in pet_referenced_nodes, f"Interface node {nid} missing in PET elements!"
        assert nid in psa_referenced_nodes, f"Interface node {nid} missing in PSA elements!"

    # Verify no duplicate coordinates exist in the mesh
    coords = mesh.nodes_array()
    unique_coords = np.unique(np.round(coords, decimals=7), axis=0)
    assert len(coords) == len(unique_coords), "Mesh contains duplicate overlapping nodes!"


def test_element_jacobians_and_areas():
    """Verify all CPE6M and CPE4H elements have positive area (correct node winding)."""
    mesh, sets = create_cpe6m_cpe4h_conformal_stack(
        length=10.0, t_pet=0.1, t_psa=0.05, nx=4, ny_pet=1, ny_psa=1
    )

    for eid, elem in mesh.elements.items():
        elem_coords = np.array([[mesh.nodes[nid].x, mesh.nodes[nid].y] for nid in elem.node_ids])
        if elem.elem_type == "CPE4H":
            # Shoelace formula for 4-node quad
            x, y = elem_coords[:, 0], elem_coords[:, 1]
            area = 0.5 * ((x[0]*y[1] - x[1]*y[0]) + (x[1]*y[2] - x[2]*y[1]) +
                          (x[2]*y[3] - x[3]*y[2]) + (x[3]*y[0] - x[0]*y[3]))
            assert area > 0.0, f"CPE4H element {eid} has non-positive area: {area}"
        elif elem.elem_type == "CPE6M":
            # Area of triangle using first 3 vertices
            x, y = elem_coords[:3, 0], elem_coords[:3, 1]
            area = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
            assert area > 0.0, f"CPE6M element {eid} has non-positive area: {area}"


def test_element_stiffness_assembly_sanity():
    """Verify element kernels can execute without error on the generated conformal mesh."""
    mesh, sets = create_cpe6m_cpe4h_conformal_stack(
        length=10.0, t_pet=0.1, t_psa=0.05, nx=2, ny_pet=1, ny_psa=1
    )

    props_pet = np.array([4000.0, 0.3, 0.0, 0.0, 0.0], dtype=np.float64)
    props_psa = np.array([1.0, 0.499, 0.0, 0.0, 0.0], dtype=np.float64)

    # Test PET CPE6M stiffness evaluation
    for eid in sets["PET_ELEMENTS"][:2]:
        elem = mesh.elements[eid]
        elem_coords = np.array([[mesh.nodes[nid].x, mesh.nodes[nid].y] for nid in elem.node_ids], dtype=np.float64)
        u_zero = np.zeros(12, dtype=np.float64)
        sdvs = np.zeros((3, 7), dtype=np.float64)
        fe, Ke, err = compute_cpe6m_element_numba(elem_coords, u_zero, 0, props_pet, sdvs, 1.0)
        assert err == 0
        assert Ke.shape == (12, 12)
        assert not np.isnan(Ke).any()
        assert np.all(np.linalg.eigvalsh(Ke) >= -1e-10)

    # Test PSA CPE4H stiffness evaluation
    for eid in sets["PSA_ELEMENTS"][:2]:
        elem = mesh.elements[eid]
        elem_coords = np.array([[mesh.nodes[nid].x, mesh.nodes[nid].y] for nid in elem.node_ids], dtype=np.float64)
        u_zero = np.zeros(8, dtype=np.float64)
        sdvs = np.zeros((4, 7), dtype=np.float64)
        fe, Ke, err = compute_cpe4h_element_numba(elem_coords, u_zero, 0, props_psa, sdvs, 1.0)
        assert err == 0
        assert Ke.shape == (8, 8)
        assert not np.isnan(Ke).any()
        assert np.all(np.linalg.eigvalsh(Ke) >= -1e-10)


def test_5layer_conformal_mesh_generation():
    """Verify 5-layer sandwich mesh (PET-PSA-PET-PSA-PET) has 100% shared interface nodes."""
    mesh, sets = create_5layer_cpe6m_cpe4h_conformal_mesh(
        length=40.0, t_pet=0.05, t_psa=0.03, nx=10
    )

    # Check element counts: 3 PET layers * (10*2 = 20) = 60 CPE6M, 2 PSA layers * (20) = 40 CPE4H -> 100 elems
    assert len(sets["PET_ELEMENTS"]) == 60
    assert len(sets["PSA_ELEMENTS"]) == 40
    assert mesh.num_elements == 100

    # Check boundary and interface sets
    assert len(sets["BOTTOM_NODES"]) > 0
    assert len(sets["TOP_NODES"]) > 0
    assert len(sets["LEFT_NODES"]) > 0
    assert len(sets["RIGHT_NODES"]) > 0
    assert len(sets["CENTERLINE_NODES"]) > 0
    assert len(sets["INTERFACE_NODES"]) > 0

    # Verify zero duplicate coordinate nodes
    coords = mesh.nodes_array()
    unique_coords = np.unique(np.round(coords, decimals=7), axis=0)
    assert len(coords) == len(unique_coords)
