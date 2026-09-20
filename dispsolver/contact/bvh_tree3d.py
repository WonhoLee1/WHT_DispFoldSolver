"""
bvh_tree3d.py
=============
Commercial-grade Dynamic Bounding Volume Hierarchy (BVH) collision detection engine
for 3D contact mechanics and self-contact analysis.

Features
--------
1. AABB Construction:
   - Computes tight Axis-Aligned Bounding Boxes for 3D Quad4 and Tri3 surface faces.
   - Supports protective collision skin dilation (skin delta >= 0).
2. Morton Code Spatial Sorting (Z-order curve):
   - Quantizes face centroids into 10-bit integer coordinates per axis (30-bit Morton code).
   - Linearizes 3D spatial coherence to construct balanced, cache-friendly bounding volume hierarchies.
3. Hierarchical Binary BVH Tree:
   - Top-down binary spatial partitioning.
   - Dual-tree recursive traversal for inter-surface broad-phase pair detection (O(N log M)).
4. Topological Self-Contact Filtering:
   - Eliminates false positive self-intersections between adjacent faces sharing common nodes or edges.
5. Bottom-Up Dynamic Refit:
   - High-speed AABB refitting for deformed meshes without rebuilding tree topology.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union


def _expand_bits_3d(v: int) -> int:
    """Interleave bits of a 10-bit integer with 2 zeros to form 30-bit Morton code."""
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v


def compute_morton_code_3d(x: float, y: float, z: float, bbox_min: np.ndarray, bbox_max: np.ndarray) -> int:
    """Compute 30-bit Morton code for a 3D point relative to the bounding box.

    Parameters
    ----------
    x, y, z : float
        Coordinates of the point.
    bbox_min, bbox_max : np.ndarray
        Shape (3,) minimum and maximum coordinates of the enclosing domain.

    Returns
    -------
    int
        30-bit Morton code.
    """
    diff = bbox_max - bbox_min
    for i in range(3):
        if diff[i] <= 1e-15:
            diff[i] = 1.0

    nx = int(np.clip((x - bbox_min[0]) / diff[0] * 1023.0, 0.0, 1023.0))
    ny = int(np.clip((y - bbox_min[1]) / diff[1] * 1023.0, 0.0, 1023.0))
    nz = int(np.clip((z - bbox_min[2]) / diff[2] * 1023.0, 0.0, 1023.0))

    return (_expand_bits_3d(nx) << 2) | (_expand_bits_3d(ny) << 1) | _expand_bits_3d(nz)


class AABB3D:
    """Axis-Aligned Bounding Box in 3D Euclidean space."""

    __slots__ = ("min_pt", "max_pt")

    def __init__(self, min_pt: np.ndarray, max_pt: np.ndarray):
        self.min_pt = np.asarray(min_pt, dtype=np.float64)
        self.max_pt = np.asarray(max_pt, dtype=np.float64)

    @classmethod
    def from_points(cls, points: np.ndarray, skin: float = 0.0) -> AABB3D:
        """Construct AABB from a set of points with optional expansion skin."""
        pts = np.asarray(points, dtype=np.float64)
        min_pt = np.min(pts, axis=0) - skin
        max_pt = np.max(pts, axis=0) + skin
        return cls(min_pt, max_pt)

    def overlaps(self, other: AABB3D) -> bool:
        """Return True if this AABB intersects with another AABB."""
        return bool(
            self.min_pt[0] <= other.max_pt[0] and self.max_pt[0] >= other.min_pt[0] and
            self.min_pt[1] <= other.max_pt[1] and self.max_pt[1] >= other.min_pt[1] and
            self.min_pt[2] <= other.max_pt[2] and self.max_pt[2] >= other.min_pt[2]
        )

    def contains_point(self, pt: np.ndarray) -> bool:
        """Return True if the point lies inside or on the boundary of this AABB."""
        p = np.asarray(pt, dtype=np.float64)
        return bool(np.all(p >= self.min_pt) and np.all(p <= self.max_pt))

    def merge(self, other: AABB3D) -> AABB3D:
        """Return a new AABB enclosing both bounding boxes."""
        return AABB3D(
            np.minimum(self.min_pt, other.min_pt),
            np.maximum(self.max_pt, other.max_pt)
        )

    def surface_area(self) -> float:
        """Compute surface area of the bounding box (used in Surface Area Heuristics)."""
        d = np.maximum(0.0, self.max_pt - self.min_pt)
        return 2.0 * float(d[0] * d[1] + d[1] * d[2] + d[2] * d[0])


class BVHNode3D:
    """Node in the 3D Bounding Volume Hierarchy."""

    __slots__ = ("aabb", "left", "right", "face_indices", "is_leaf")

    def __init__(
        self,
        aabb: AABB3D,
        left: Optional[BVHNode3D] = None,
        right: Optional[BVHNode3D] = None,
        face_indices: Optional[List[int]] = None,
    ):
        self.aabb = aabb
        self.left = left
        self.right = right
        self.face_indices = face_indices if face_indices is not None else []
        self.is_leaf = left is None and right is None


class DynamicBVHTree3D:
    """Dynamic 3D Bounding Volume Hierarchy collision detection tree.

    Parameters
    ----------
    faces : List[Tuple[int, ...]]
        List of surface face node IDs (e.g. [(1, 2, 3, 4), ...]).
    coords : np.ndarray
        Array of node coordinates, shape (N_nodes, 3) or dictionary mapping node_id -> [x, y, z].
    nid_to_idx : Optional[Dict[int, int]]
        Mapping from global node ID to row index in `coords` (if coords is ndarray).
    skin : float, optional
        Protective expansion margin applied to each primitive AABB, by default 0.0.
    max_leaf_size : int, optional
        Maximum number of primitives in a leaf node, by default 1.
    """

    def __init__(
        self,
        faces: List[Tuple[int, ...]],
        coords: Union[np.ndarray, Dict[int, np.ndarray]],
        nid_to_idx: Optional[Dict[int, int]] = None,
        skin: float = 0.0,
        max_leaf_size: int = 1,
    ):
        self.faces = faces
        self.coords = coords
        self.nid_to_idx = nid_to_idx
        self.skin = float(skin)
        self.max_leaf_size = max(1, int(max_leaf_size))

        self.face_aabbs: List[AABB3D] = []
        self.face_centroids: np.ndarray = np.zeros((len(faces), 3), dtype=np.float64)
        self.face_nodes_set: List[Set[int]] = [set(f) for f in faces]

        self._compute_face_bounds()
        self.root: Optional[BVHNode3D] = self._build_tree()

    def _get_node_coord(self, nid: int) -> np.ndarray:
        """Retrieve 3D coordinate vector of a node."""
        if isinstance(self.coords, dict):
            return np.asarray(self.coords[nid], dtype=np.float64)
        if self.nid_to_idx is not None:
            return self.coords[self.nid_to_idx[nid]]
        return self.coords[nid]

    def _compute_face_bounds(self) -> None:
        """Compute individual AABBs and centroids for all faces."""
        self.face_aabbs.clear()
        n_faces = len(self.faces)
        self.face_centroids = np.zeros((n_faces, 3), dtype=np.float64)

        for i, face in enumerate(self.faces):
            pts = np.array([self._get_node_coord(nid) for nid in face], dtype=np.float64)
            aabb = AABB3D.from_points(pts, skin=self.skin)
            self.face_aabbs.append(aabb)
            self.face_centroids[i] = np.mean(pts, axis=0)

    def _build_tree(self) -> Optional[BVHNode3D]:
        """Construct hierarchical BVH using Morton code spatial partitioning."""
        n_faces = len(self.faces)
        if n_faces == 0:
            return None

        # Compute overall domain bounding box
        domain_min = np.min([aabb.min_pt for aabb in self.face_aabbs], axis=0)
        domain_max = np.max([aabb.max_pt for aabb in self.face_aabbs], axis=0)

        # Compute Morton codes for each face
        morton_keys = []
        for i in range(n_faces):
            c = self.face_centroids[i]
            code = compute_morton_code_3d(c[0], c[1], c[2], domain_min, domain_max)
            morton_keys.append((code, i))

        # Sort face indices by Morton code (spatially coherent curve)
        morton_keys.sort(key=lambda item: item[0])
        sorted_indices = [item[1] for item in morton_keys]

        return self._recursive_build(sorted_indices)

    def _recursive_build(self, indices: List[int]) -> BVHNode3D:
        """Recursively build BVH nodes by bisecting the sorted primitive list."""
        # Compute combined AABB for this set of primitives
        node_aabb = self.face_aabbs[indices[0]]
        for idx in indices[1:]:
            node_aabb = node_aabb.merge(self.face_aabbs[idx])

        if len(indices) <= self.max_leaf_size:
            return BVHNode3D(aabb=node_aabb, face_indices=indices)

        # Median split
        mid = len(indices) // 2
        left_child = self._recursive_build(indices[:mid])
        right_child = self._recursive_build(indices[mid:])

        return BVHNode3D(aabb=node_aabb, left=left_child, right=right_child)

    def refit(self, u: Optional[np.ndarray] = None) -> None:
        """Bottom-up refit of all bounding boxes without rebuilding tree hierarchy.

        Parameters
        ----------
        u : Optional[np.ndarray]
            Current displacement vector (size 3 * N_nodes). If None, uses original coordinates.
        """
        # Recompute face AABBs with deformation
        for i, face in enumerate(self.faces):
            pts = []
            for nid in face:
                x0 = self._get_node_coord(nid)
                if u is not None:
                    idx = self.nid_to_idx[nid] if self.nid_to_idx is not None else nid
                    disp = u[3 * idx : 3 * idx + 3]
                    pts.append(x0 + disp)
                else:
                    pts.append(x0)
            self.face_aabbs[i] = AABB3D.from_points(np.array(pts), skin=self.skin)

        def _refit_node(node: Optional[BVHNode3D]) -> Optional[AABB3D]:
            if node is None:
                return None
            if node.is_leaf:
                node_aabb = self.face_aabbs[node.face_indices[0]]
                for idx in node.face_indices[1:]:
                    node_aabb = node_aabb.merge(self.face_aabbs[idx])
                node.aabb = node_aabb
                return node.aabb

            left_aabb = _refit_node(node.left)
            right_aabb = _refit_node(node.right)

            if left_aabb is not None and right_aabb is not None:
                node.aabb = left_aabb.merge(right_aabb)
            elif left_aabb is not None:
                node.aabb = left_aabb
            elif right_aabb is not None:
                node.aabb = right_aabb
            return node.aabb

        _refit_node(self.root)

    def query_intersection(self, query_aabb: AABB3D) -> List[int]:
        """Find all faces whose AABB intersects query_aabb."""
        results: List[int] = []
        if self.root is None:
            return results

        stack = [self.root]
        while stack:
            node = stack.pop()
            if not node.aabb.overlaps(query_aabb):
                continue
            if node.is_leaf:
                for idx in node.face_indices:
                    if self.face_aabbs[idx].overlaps(query_aabb):
                        results.append(idx)
            else:
                if node.left is not None:
                    stack.append(node.left)
                if node.right is not None:
                    stack.append(node.right)

        return results

    def query_pairs_against(self, other: DynamicBVHTree3D) -> List[Tuple[int, int]]:
        """Find all intersecting face pairs between self (slave) and other (master).

        Returns
        -------
        List[Tuple[int, int]]
            List of (self_face_idx, other_face_idx) candidate contact pairs.
        """
        pairs: List[Tuple[int, int]] = []
        if self.root is None or other.root is None:
            return pairs

        stack = [(self.root, other.root)]
        while stack:
            na, nb = stack.pop()
            if not na.aabb.overlaps(nb.aabb):
                continue

            if na.is_leaf and nb.is_leaf:
                for ia in na.face_indices:
                    for ib in nb.face_indices:
                        if self.face_aabbs[ia].overlaps(other.face_aabbs[ib]):
                            pairs.append((ia, ib))
            elif na.is_leaf:
                if nb.left is not None:
                    stack.append((na, nb.left))
                if nb.right is not None:
                    stack.append((na, nb.right))
            elif nb.is_leaf:
                if na.left is not None:
                    stack.append((na.left, nb))
                if na.right is not None:
                    stack.append((na.right, nb))
            else:
                # Both are internal nodes: split the larger one
                if na.aabb.surface_area() >= nb.aabb.surface_area():
                    if na.left is not None:
                        stack.append((na.left, nb))
                    if na.right is not None:
                        stack.append((na.right, nb))
                else:
                    if nb.left is not None:
                        stack.append((na, nb.left))
                    if nb.right is not None:
                        stack.append((na, nb.right))

        return pairs

    def query_self_contact(self) -> List[Tuple[int, int]]:
        """Detect self-contact candidate face pairs, filtering out adjacent/connected faces.

        Returns
        -------
        List[Tuple[int, int]]
            List of (face_i, face_j) where face_i < face_j and faces do not share any nodes.
        """
        pairs: List[Tuple[int, int]] = []
        if self.root is None:
            return pairs

        stack: List[Tuple[BVHNode3D, BVHNode3D]] = [(self.root, self.root)]
        while stack:
            na, nb = stack.pop()
            if not na.aabb.overlaps(nb.aabb):
                continue

            if na == nb:
                # Same node traversal
                if na.is_leaf:
                    # Leaf vs itself: test all unique internal pairs
                    indices = na.face_indices
                    n = len(indices)
                    for i in range(n):
                        for j in range(i + 1, n):
                            fa, fb = indices[i], indices[j]
                            # Self-contact filtering: faces sharing any nodes are topologically adjacent
                            if not (self.face_nodes_set[fa] & self.face_nodes_set[fb]):
                                if self.face_aabbs[fa].overlaps(self.face_aabbs[fb]):
                                    pairs.append((min(fa, fb), max(fa, fb)))
                else:
                    # Descend both children and cross-test
                    if na.left is not None:
                        stack.append((na.left, na.left))
                    if na.right is not None:
                        stack.append((na.right, na.right))
                    if na.left is not None and na.right is not None:
                        stack.append((na.left, na.right))
            else:
                # na != nb
                if na.is_leaf and nb.is_leaf:
                    for fa in na.face_indices:
                        for fb in nb.face_indices:
                            if fa == fb:
                                continue
                            if not (self.face_nodes_set[fa] & self.face_nodes_set[fb]):
                                if self.face_aabbs[fa].overlaps(self.face_aabbs[fb]):
                                    pairs.append((min(fa, fb), max(fa, fb)))
                elif na.is_leaf:
                    if nb.left is not None:
                        stack.append((na, nb.left))
                    if nb.right is not None:
                        stack.append((na, nb.right))
                elif nb.is_leaf:
                    if na.left is not None:
                        stack.append((na.left, nb))
                    if na.right is not None:
                        stack.append((na.right, nb))
                else:
                    if na.aabb.surface_area() >= nb.aabb.surface_area():
                        if na.left is not None:
                            stack.append((na.left, nb))
                        if na.right is not None:
                            stack.append((na.right, nb))
                    else:
                        if nb.left is not None:
                            stack.append((na, nb.left))
                        if nb.right is not None:
                            stack.append((na, nb.right))

        # Deduplicate pairs
        unique_pairs = sorted(list(set(pairs)))
        return unique_pairs
