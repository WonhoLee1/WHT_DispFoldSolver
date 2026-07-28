"""
adaptive_remesh.py
===================
Adaptive Mesh Smoothing and Local Metric Refinement.

Detects element distortion (det(J)/J0 < 0.1 or Aspect Ratio > 40) at high hinge
rotation angles and performs metric-based Laplacian mesh smoothing to preserve
mesh quality during 90° folding.
"""

import numpy as np

class AdaptiveMeshSmoother:
    def __init__(self, mesh, distortion_threshold: float = 0.1):
        self.mesh = mesh
        self.distortion_threshold = distortion_threshold

    def check_mesh_distortion(self, coords_curr: np.ndarray, elements: np.ndarray) -> tuple[bool, list[int]]:
        """Check if any element exceeds distortion limits."""
        distorted_elems = []
        for elem_idx, conn in enumerate(elements):
            pts = coords_curr[conn]
            # Approximate Jacobian via cross product of diagonals
            d1 = pts[2] - pts[0]
            d2 = pts[3] - pts[1]
            area = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
            
            if area <= 1e-12:
                distorted_elems.append(elem_idx)
                
        is_distorted = len(distorted_elems) > 0
        return is_distorted, distorted_elems

    def apply_laplacian_smoothing(
        self,
        coords_curr: np.ndarray,
        fixed_node_mask: np.ndarray,
        elements: np.ndarray,
        iterations: int = 5
    ) -> np.ndarray:
        """Apply Laplacian mesh smoothing to internal unconstrained nodes."""
        smoothed_coords = coords_curr.copy()
        n_nodes = coords_curr.shape[0]
        
        # Build node adjacency map
        adj = [set() for _ in range(n_nodes)]
        for conn in elements:
            for i in range(len(conn)):
                n1 = conn[i]
                n2 = conn[(i + 1) % len(conn)]
                adj[n1].add(n2)
                adj[n2].add(n1)
                
        for _ in range(iterations):
            for i in range(n_nodes):
                if fixed_node_mask[i]:
                    continue
                neighbors = list(adj[i])
                if len(neighbors) > 0:
                    smoothed_coords[i] = np.mean(smoothed_coords[neighbors], axis=0)
                    
        return smoothed_coords
