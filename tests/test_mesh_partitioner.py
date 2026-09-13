"""
test_mesh_partitioner.py
========================
Unit tests for DomainPartitioner in 2D and 3D FE meshes.
Verifies element partitioning, internal vs interface node classification, and mapping integrity.
"""

import pytest
import numpy as np

from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.parallel.partitioner import DomainPartitioner, Subdomain


def create_sample_2d_mesh(nx=10, ny=2):
    """Creates a regular 2D quad mesh (nx x ny)."""
    mesh = Mesh2D()
    nid = 1
    grid = np.zeros((nx + 1, ny + 1), dtype=int)
    for j in range(ny + 1):
        for i in range(nx + 1):
            mesh.add_node(nid, float(i), float(j))
            grid[i, j] = nid
            nid += 1

    eid = 1
    for j in range(ny):
        for i in range(nx):
            n1 = int(grid[i, j])
            n2 = int(grid[i + 1, j])
            n3 = int(grid[i + 1, j + 1])
            n4 = int(grid[i, j + 1])
            mesh.add_element(eid, [n1, n2, n3, n4], elem_type="CPE4")
            eid += 1

    return mesh


def create_sample_3d_mesh(nx=8, ny=2, nz=2):
    """Creates a regular 3D hex mesh (nx x ny x nz)."""
    mesh = Mesh3D()
    nid = 1
    grid = np.zeros((nx + 1, ny + 1, nz + 1), dtype=int)
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                mesh.add_node(nid, float(i), float(j), float(k))
                grid[i, j, k] = nid
                nid += 1

    eid = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n1 = int(grid[i, j, k])
                n2 = int(grid[i + 1, j, k])
                n3 = int(grid[i + 1, j + 1, k])
                n4 = int(grid[i, j + 1, k])
                n5 = int(grid[i, j, k + 1])
                n6 = int(grid[i + 1, j, k + 1])
                n7 = int(grid[i + 1, j + 1, k + 1])
                n8 = int(grid[i, j + 1, k + 1])
                mesh.add_element(eid, [n1, n2, n3, n4, n5, n6, n7, n8], elem_type="C3D8")
                eid += 1

    return mesh


def test_partition_2d_mesh_2_parts():
    mesh = create_sample_2d_mesh(nx=10, ny=2)
    total_elems = mesh.num_elements
    total_nodes = mesh.num_nodes

    subdomains = DomainPartitioner.partition(mesh, num_partitions=2, axis="auto")
    assert len(subdomains) == 2

    # Check element conservation and non-overlap
    elem_union = set(subdomains[0].elem_ids).union(set(subdomains[1].elem_ids))
    assert len(elem_union) == total_elems
    assert len(set(subdomains[0].elem_ids).intersection(set(subdomains[1].elem_ids))) == 0

    # Check interface nodes exist along the cut
    assert len(subdomains[0].interface_node_ids) > 0
    assert set(subdomains[0].interface_node_ids) == set(subdomains[1].interface_node_ids)

    # Check internal nodes do not overlap
    assert len(set(subdomains[0].internal_node_ids).intersection(set(subdomains[1].internal_node_ids))) == 0

    # Total nodes check
    all_nids = set(subdomains[0].node_ids).union(set(subdomains[1].node_ids))
    assert len(all_nids) == total_nodes


def test_partition_2d_mesh_4_parts():
    mesh = create_sample_2d_mesh(nx=12, ny=2)
    subdomains = DomainPartitioner.partition(mesh, num_partitions=4, axis="auto")
    assert len(subdomains) == 4

    total_eids = set()
    for sub in subdomains:
        total_eids.update(sub.elem_ids)
    assert len(total_eids) == mesh.num_elements

    # Check neighbor connectivity (1D slice should form chain 0-1-2-3)
    assert 1 in subdomains[0].neighbor_ranks
    assert 0 in subdomains[1].neighbor_ranks
    assert 2 in subdomains[1].neighbor_ranks
    assert 1 in subdomains[2].neighbor_ranks
    assert 3 in subdomains[2].neighbor_ranks
    assert 2 in subdomains[3].neighbor_ranks


def test_partition_3d_mesh_4_parts():
    mesh = create_sample_3d_mesh(nx=12, ny=2, nz=2)
    subdomains = DomainPartitioner.partition(mesh, num_partitions=4, axis="auto")
    assert len(subdomains) == 4

    total_eids = set()
    for sub in subdomains:
        total_eids.update(sub.elem_ids)
        assert sub.sub_mesh.num_elements == len(sub.elem_ids)
        assert sub.sub_mesh.num_nodes == len(sub.node_ids)

    assert len(total_eids) == mesh.num_elements


def test_partition_single_part():
    mesh = create_sample_2d_mesh(nx=4, ny=2)
    subdomains = DomainPartitioner.partition(mesh, num_partitions=1)
    assert len(subdomains) == 1
    assert len(subdomains[0].elem_ids) == mesh.num_elements
    assert len(subdomains[0].internal_node_ids) == mesh.num_nodes
    assert len(subdomains[0].interface_node_ids) == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
