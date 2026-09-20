"""
test_flat_bvh_tree3d.py
=======================
Unit tests for Flat Linear BVH (Morton LBVH) and Zero-Allocation Sutherland-Hodgman Numba Kernels.
"""

import numpy as np
import pytest
from dispsolver.contact.flat_bvh_tree3d import (
    FlatBVHTree3D,
    bvh_refit_flat_numba,
    sutherland_hodgman_clip_quad4_numba,
    integrate_dual_mortar_polygon_numba,
)


def test_flat_bvh_construction_and_refit():
    # 4 Quad4 faces forming a 2x2 flat plate on z=0
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
        [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0],
    ], dtype=np.float64)

    faces = [
        (0, 1, 4, 3),  # Face 0
        (1, 2, 5, 4),  # Face 1
        (3, 4, 7, 6),  # Face 2
        (4, 5, 8, 7),  # Face 3
    ]

    tree = FlatBVHTree3D(faces, coords, skin=0.02)
    assert tree.n_prims == 4
    assert tree.n_nodes == 7  # 2 * 4 - 1
    assert tree.nodes_bbox.shape == (7, 6)
    assert tree.nodes_topo.shape == (7, 4)

    # Deform node 8 in z direction by 0.5
    u = np.zeros(27, dtype=np.float64)
    u[8 * 3 + 2] = 0.5
    tree.refit(u=u)

    root_bbox = tree.nodes_bbox[tree.root_idx]
    assert root_bbox[5] >= 0.5 + 0.02 - 1e-12


def test_sutherland_hodgman_clipping_quad():
    # Overlapping quads: [0, 2] x [0, 2] and [1, 3] x [1, 3]
    qa_x = np.array([0.0, 2.0, 2.0, 0.0])
    qa_y = np.array([0.0, 0.0, 2.0, 2.0])
    qb_x = np.array([1.0, 3.0, 3.0, 1.0])
    qb_y = np.array([1.0, 1.0, 3.0, 3.0])

    out_x = np.zeros(16, dtype=np.float64)
    out_y = np.zeros(16, dtype=np.float64)

    n_clip = sutherland_hodgman_clip_quad4_numba(qa_x, qa_y, 4, qb_x, qb_y, 4, out_x, out_y)
    assert n_clip == 4

    # Resulting polygon should be [1, 2] x [1, 2]
    assert np.all(out_x[:n_clip] >= 1.0 - 1e-12)
    assert np.all(out_x[:n_clip] <= 2.0 + 1e-12)
    assert np.all(out_y[:n_clip] >= 1.0 - 1e-12)
    assert np.all(out_y[:n_clip] <= 2.0 + 1e-12)


def test_sutherland_hodgman_clipping_octagon():
    # Square [-1, 1] x [-1, 1] and rotated 45-deg diamond with radius 1.2
    sq_x = np.array([-1.0, 1.0, 1.0, -1.0])
    sq_y = np.array([-1.0, -1.0, 1.0, 1.0])
    r = 1.2
    dm_x = np.array([r, 0.0, -r, 0.0])
    dm_y = np.array([0.0, r, 0.0, -r])

    out_x = np.zeros(16, dtype=np.float64)
    out_y = np.zeros(16, dtype=np.float64)

    n_clip = sutherland_hodgman_clip_quad4_numba(dm_x, dm_y, 4, sq_x, sq_y, 4, out_x, out_y)
    assert n_clip == 8


def test_integrate_dual_mortar_polygon():
    # Unit square [0, 1] x [0, 1]
    px = np.array([0.0, 1.0, 1.0, 0.0])
    py = np.array([0.0, 0.0, 1.0, 1.0])
    slave_xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    master_xy = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    D_mat = np.zeros((4, 4), dtype=np.float64)
    M_mat = np.zeros((4, 4), dtype=np.float64)
    D_dual = np.zeros(4, dtype=np.float64)

    area = integrate_dual_mortar_polygon_numba(px, py, 4, slave_xy, master_xy, D_mat, M_mat, D_dual)
    assert abs(area - 1.0) < 1e-12
    assert abs(np.sum(D_dual) - 1.0) < 1e-12
    assert abs(np.sum(D_mat) - 1.0) < 1e-12
    assert abs(np.sum(M_mat) - 1.0) < 1e-12
    for i in range(4):
        assert abs(D_dual[i] - 0.25) < 1e-12


def test_tree_queries_and_self_contact():
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
        [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0],
    ], dtype=np.float64)

    faces = [
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (3, 4, 7, 6),
        (4, 5, 8, 7),
    ]

    tree = FlatBVHTree3D(faces, coords, skin=0.01)

    # Query intersection with box enclosing face 0
    hit = tree.query_intersection(np.array([0.1, 0.1, -0.05, 0.9, 0.9, 0.05]))
    assert 0 in hit

    # Self-contact: 4 adjacent faces sharing nodes -> 0 self-contact pairs
    self_pairs = tree.query_self_contact()
    assert len(self_pairs) == 0

    # Add 2 disconnected opposite faces (e.g. parallel plates at z=0 and z=0.01)
    faces2 = [(0, 1, 4, 3), (1, 2, 5, 4)]
    coords_master = np.copy(coords)
    coords_master[:, 2] += 0.005  # Within skin margin

    tree_slave = FlatBVHTree3D(faces2, coords, skin=0.02)
    tree_master = FlatBVHTree3D(faces2, coords_master, skin=0.02)

    inter_pairs = tree_slave.query_pairs_against(tree_master)
    assert len(inter_pairs) >= 2
