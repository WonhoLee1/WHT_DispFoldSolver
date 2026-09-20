"""
test_bvh_tree3d.py
===================
Comprehensive test suite for 3D Dynamic Bounding Volume Hierarchy (BVH) collision detection.

Tests:
1. test_morton_code_monotonicity_and_range
2. test_aabb_operations_and_skin_dilation
3. test_bvh_broadphase_vs_bruteforce_equivalence
4. test_self_contact_topological_filtering
5. test_dynamic_bvh_refit_consistency
"""

import numpy as np
import pytest

from dispsolver.contact.bvh_tree3d import (
    AABB3D,
    DynamicBVHTree3D,
    compute_morton_code_3d,
)


def test_morton_code_monotonicity_and_range():
    """Verify Morton code calculation produces correct bit-interleaved 30-bit integers."""
    bmin = np.array([0.0, 0.0, 0.0])
    bmax = np.array([10.0, 10.0, 10.0])

    code_origin = compute_morton_code_3d(0.0, 0.0, 0.0, bmin, bmax)
    code_max = compute_morton_code_3d(10.0, 10.0, 10.0, bmin, bmax)

    assert code_origin == 0
    # Maximum 30-bit integer is 2^30 - 1 = 1073741823 (0x3FFFFFFF)
    assert code_max == 0x3FFFFFFF

    # Monotonicity along diagonal
    prev_code = -1
    for t in np.linspace(0.0, 10.0, 20):
        c = compute_morton_code_3d(t, t, t, bmin, bmax)
        assert c >= prev_code
        prev_code = c


def test_aabb_operations_and_skin_dilation():
    """Verify AABB overlap, point containment, merging, and skin dilation."""
    pts1 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    box1 = AABB3D.from_points(pts1, skin=0.0)

    pts2 = np.array([[1.5, 1.5, 1.5], [2.5, 2.5, 2.5]])
    box2 = AABB3D.from_points(pts2, skin=0.0)

    # Initially disjoint
    assert not box1.overlaps(box2)

    # Dilating box1 with skin=0.6 makes them overlap (max of box1 becomes 1.6 > 1.5)
    box1_dilated = AABB3D.from_points(pts1, skin=0.6)
    assert box1_dilated.overlaps(box2)

    # Point containment
    assert box1.contains_point([0.5, 0.5, 0.5])
    assert not box1.contains_point([1.5, 0.5, 0.5])

    # Merge
    box_merged = box1.merge(box2)
    assert np.allclose(box_merged.min_pt, [0.0, 0.0, 0.0])
    assert np.allclose(box_merged.max_pt, [2.5, 2.5, 2.5])


def test_bvh_broadphase_vs_bruteforce_equivalence():
    """Verify that BVH dual-tree traversal yields 100% exact match with exhaustive O(N*M) test."""
    rng = np.random.RandomState(42)

    # Create 16 slave quads on a 4x4 grid at z=0.0
    slave_coords = []
    slave_faces = []
    s_nid = 1
    s_grid = {}
    for j, y in enumerate(np.linspace(0.0, 4.0, 5)):
        for i, x in enumerate(np.linspace(0.0, 4.0, 5)):
            slave_coords.append([x, y, 0.0])
            s_grid[(i, j)] = s_nid
            s_nid += 1

    for j in range(4):
        for i in range(4):
            slave_faces.append((
                s_grid[(i, j)], s_grid[(i + 1, j)],
                s_grid[(i + 1, j + 1)], s_grid[(i, j + 1)]
            ))

    # Create 9 master quads on a 3x3 grid at z=0.05 with skin=0.1
    master_coords = []
    master_faces = []
    m_nid = 1
    m_grid = {}
    for j, y in enumerate(np.linspace(1.0, 3.0, 4)):
        for i, x in enumerate(np.linspace(1.0, 3.0, 4)):
            master_coords.append([x, y, 0.05])
            m_grid[(i, j)] = m_nid
            m_nid += 1

    for j in range(3):
        for i in range(3):
            master_faces.append((
                m_grid[(i, j)], m_grid[(i + 1, j)],
                m_grid[(i + 1, j + 1)], m_grid[(i, j + 1)]
            ))

    slave_bvh = DynamicBVHTree3D(
        faces=slave_faces,
        coords=np.array(slave_coords, dtype=np.float64),
        nid_to_idx={nid: nid - 1 for nid in range(1, s_nid)},
        skin=0.0,
    )
    master_bvh = DynamicBVHTree3D(
        faces=master_faces,
        coords=np.array(master_coords, dtype=np.float64),
        nid_to_idx={nid: nid - 1 for nid in range(1, m_nid)},
        skin=0.1,  # Expands master z to [-0.05, 0.15], overlapping slave at z=0
    )

    # 1. BVH query
    bvh_pairs = sorted(slave_bvh.query_pairs_against(master_bvh))

    # 2. Brute-force exhaustive ground truth
    brute_force_pairs = []
    for s_idx, s_aabb in enumerate(slave_bvh.face_aabbs):
        for m_idx, m_aabb in enumerate(master_bvh.face_aabbs):
            if s_aabb.overlaps(m_aabb):
                brute_force_pairs.append((s_idx, m_idx))
    brute_force_pairs.sort()

    assert len(bvh_pairs) > 0, "Expected non-empty contact candidates"
    assert bvh_pairs == brute_force_pairs, (
        f"BVH pairs mismatch! BVH found {len(bvh_pairs)}, Brute Force found {len(brute_force_pairs)}"
    )


def test_self_contact_topological_filtering():
    """Verify that adjacent faces sharing nodes are filtered out in self-contact detection."""
    # A single strip of 4 quads along X: [0, 1], [1, 2], [2, 3], [3, 4]
    # Faces 0 and 1 share 2 nodes (edge neighbor) -> must NOT be reported as self-contact
    # When quad 3 is bent over quad 0 (in a folded loop), they overlap without sharing nodes -> must be reported!
    coords = np.array([
        # Face 0: (1, 2, 4, 3) at x in [0, 1]
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        # Face 1: (2, 5, 6, 4) at x in [1, 2]
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        # Face 2: (5, 7, 8, 6) at x in [2, 3]
        [3.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        # Face 3: (9, 10, 11, 12) placed directly on top of Face 0 at z = 0.05
        # (disjoint node IDs, representing a folded loop self-contact!)
        [0.0, 0.0, 0.05],
        [1.0, 0.0, 0.05],
        [1.0, 1.0, 0.05],
        [0.0, 1.0, 0.05],
    ], dtype=np.float64)

    nid_to_idx = {nid: nid - 1 for nid in range(1, 13)}

    faces = [
        (1, 2, 4, 3),      # Face 0
        (2, 5, 6, 4),      # Face 1 (shares nodes 2, 4 with Face 0)
        (5, 7, 8, 6),      # Face 2 (shares nodes 5, 6 with Face 1)
        (9, 10, 11, 12),   # Face 3 (self-contact loop touching Face 0)
    ]

    bvh = DynamicBVHTree3D(
        faces=faces,
        coords=coords,
        nid_to_idx=nid_to_idx,
        skin=0.1,
    )

    self_pairs = bvh.query_self_contact()

    # Face 0 and Face 1 share nodes -> must NOT be in pairs
    assert (0, 1) not in self_pairs
    # Face 1 and Face 2 share nodes -> must NOT be in pairs
    assert (1, 2) not in self_pairs

    # Face 0 and Face 3 do NOT share any nodes and overlap -> MUST be detected!
    assert (0, 3) in self_pairs


def test_dynamic_bvh_refit_consistency():
    """Verify that refit(u) updates bounding boxes correctly after large deformation."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    nid_to_idx = {nid: nid - 1 for nid in range(1, 5)}
    faces = [(1, 2, 3, 4)]

    bvh = DynamicBVHTree3D(faces=faces, coords=coords, nid_to_idx=nid_to_idx, skin=0.0)

    # Initial AABB
    assert np.allclose(bvh.root.aabb.min_pt, [0.0, 0.0, 0.0])
    assert np.allclose(bvh.root.aabb.max_pt, [1.0, 1.0, 0.0])

    # Apply rigid translation u = (+5.0, +5.0, +5.0) to all nodes
    u = np.full(12, 5.0, dtype=np.float64)
    bvh.refit(u)

    assert np.allclose(bvh.root.aabb.min_pt, [5.0, 5.0, 5.0])
    assert np.allclose(bvh.root.aabb.max_pt, [6.0, 6.0, 5.0])

    # Query intersection with new bounding box
    query_box = AABB3D(np.array([5.5, 5.5, 4.5]), np.array([5.8, 5.8, 5.5]))
    hit = bvh.query_intersection(query_box)
    assert len(hit) == 1 and hit[0] == 0

    # Old location should now return no hits
    old_query = AABB3D(np.array([0.2, 0.2, -0.1]), np.array([0.8, 0.8, 0.1]))
    assert len(bvh.query_intersection(old_query)) == 0
