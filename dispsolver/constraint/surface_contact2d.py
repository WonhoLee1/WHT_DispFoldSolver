"""
surface_contact2d.py
====================
Penalty-based 2D Node-to-Analytical-Rigid-Line Contact Constraint.

Frictionless, hard / soft, small-sliding, node-to-surface penalty contact
between deformable 2D slave nodes and a flat analytical rigid line.
Pure SPD variational formulation compatible with Intel oneMKL PARDISO.
"""

from __future__ import annotations
from typing import Any, List, Tuple, Dict, Optional, Set
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


class SurfaceContactConstraint2D:
    """Penalty-based node-to-analytical-rigid-line frictionless contact in 2D.

    Parameters
    ----------
    slave_node_ids : list of int
        Node IDs of the deformable body's candidate contact surface.
    rigid_line_point : (2,) ndarray
        A point on the rigid line.
    rigid_line_normal : (2,) ndarray
        UNIT normal of the rigid line, pointing toward the deformable body
        (a slave node has penetrated once (x - point) . normal < 0).
    nid_to_idx : dict
        Mapping from global node ID to solver node index (0-indexed).
    coords : ndarray of shape (n_nodes, 2)
        Initial (reference) 2D coordinates of all nodes in the mesh.
    penalty_stiffness : float
        Penalty stiffness k_contact [force/length].
    strain_free_adjust : bool
        If True, adjust initial penetration to zero.
    name : str
        Constraint name.
    """

    def __init__(
        self,
        slave_node_ids: List[int],
        rigid_line_point: np.ndarray,
        rigid_line_normal: np.ndarray,
        nid_to_idx: Dict[int, int],
        coords: np.ndarray,
        penalty_stiffness: float = 1.0e8,
        strain_free_adjust: bool = False,
        stabilization_coefficient: float = 0.0,
        augmented_lagrange: bool = False,
        law: Optional[Any] = None,
        chatter_stabilization: bool = False,
        hysteresis_band: float = 0.0,
        friction_coefficient: float = 0.0,
        reference_node: Optional[Any] = None,
        ref_idx: Optional[int] = None,
        name: str = "Contact2D",
    ):
        self.name = name
        self.slave_node_ids = list(slave_node_ids)
        self.nid_to_idx = dict(nid_to_idx)
        self.point = np.asarray(rigid_line_point, dtype=np.float64)[:2]
        self.point_init = self.point.copy()
        n_raw = np.asarray(rigid_line_normal, dtype=np.float64)[:2]
        norm_n = np.linalg.norm(n_raw)
        self.normal = n_raw / norm_n if norm_n > 1e-12 else np.array([0.0, 1.0], dtype=np.float64)

        self.reference_node = reference_node
        self.ref_idx = ref_idx
        if self.ref_idx is None and reference_node is not None:
            if reference_node in self.nid_to_idx:
                self.ref_idx = self.nid_to_idx[reference_node]
            elif isinstance(reference_node, str) and reference_node.isdigit() and int(reference_node) in self.nid_to_idx:
                self.ref_idx = self.nid_to_idx[int(reference_node)]

        self.penalty_stiffness = float(penalty_stiffness)
        self.strain_free_adjust = bool(strain_free_adjust)
        self.stabilization_coefficient = float(stabilization_coefficient)
        self.augmented_lagrange = bool(augmented_lagrange)
        self.law = law
        self.chatter_stabilization = bool(chatter_stabilization)
        self.hysteresis_band = float(hysteresis_band)
        self.friction_coefficient = float(friction_coefficient)

        # Store initial coordinates
        self.ref_coords = {}
        for nid in self.slave_node_ids:
            if nid in self.nid_to_idx:
                idx = self.nid_to_idx[nid]
                self.ref_coords[nid] = coords[idx, :2].copy()

        # Strain-free initial adjustment if initial penetration exists
        self.initial_offsets = {nid: 0.0 for nid in self.slave_node_ids}
        if self.strain_free_adjust:
            for nid, X0 in self.ref_coords.items():
                g0 = float(np.dot(X0 - self.point, self.normal))
                if g0 < 0.0:
                    self.initial_offsets[nid] = -g0

        # Multiplier storage for Augmented Lagrangian
        self._lam = {nid: 0.0 for nid in self.slave_node_ids}

    def _get_current_plane_point(self, u: np.ndarray) -> np.ndarray:
        if self.ref_idx is not None and len(u) > 2 * self.ref_idx + 1:
            u_ref = u[2 * self.ref_idx : 2 * self.ref_idx + 2]
            return self.point_init + u_ref
        return self.point

    def get_active_set(self, u: np.ndarray) -> frozenset:
        """Return frozenset of penetrated slave node IDs for Severe Discontinuity Iteration tracking."""
        active = []
        curr_pt = self._get_current_plane_point(u)
        for nid, X0 in self.ref_coords.items():
            idx = self.nid_to_idx[nid]
            u_node = u[2 * idx : 2 * idx + 2]
            curr_pos = X0 + u_node
            gap = float(np.dot(curr_pos - curr_pt, self.normal)) + self.initial_offsets[nid]
            if gap < 0.0:
                active.append(nid)
        return frozenset(active)

    def assemble(self, u: np.ndarray) -> Tuple[csr_matrix, np.ndarray, Dict[str, Any]]:
        """Assemble contact stiffness and contact internal force vector for 2D system.

        Returns
        -------
        K_c : csr_matrix
            Sparse contact tangent stiffness (num_dofs, num_dofs).
        f_c : ndarray
            Contact force vector (num_dofs,).
        stats : dict
            Contact diagnostics (active_count, max_penetration, etc.).
        """
        n_dofs = len(u)
        n_nodes = n_dofs // 2

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        f_c = np.zeros(n_dofs, dtype=np.float64)

        curr_pt = self._get_current_plane_point(u)
        nx, ny = self.normal[0], self.normal[1]
        nn_T = np.array([
            [nx * nx, nx * ny],
            [ny * nx, ny * ny]
        ], dtype=np.float64)

        active_count = 0
        max_pen = 0.0
        total_normal_force = 0.0

        has_ref = (self.ref_idx is not None and (2 * self.ref_idx + 1 < n_dofs))

        for nid, X0 in self.ref_coords.items():
            idx = self.nid_to_idx[nid]
            u_node = u[2 * idx : 2 * idx + 2]
            curr_pos = X0 + u_node
            gap = float(np.dot(curr_pos - curr_pt, self.normal)) + self.initial_offsets[nid]

            # Contact is active when penetrating (gap < 0)
            if gap < 0.0:
                pen = -gap
                active_count += 1
                max_pen = max(max_pen, pen)

                k_pen = self.penalty_stiffness
                f_mag = k_pen * pen
                total_normal_force += f_mag

                # Contact force pushes outward in +normal direction on slave node
                # f_int = f_int_elem - f_contact, so opposing internal force is -f_contact
                f_c[2 * idx + 0] -= f_mag * nx
                f_c[2 * idx + 1] -= f_mag * ny

                # Newton's 3rd Law: equal and opposite force on the rigid reference node
                if has_ref:
                    f_c[2 * self.ref_idx + 0] += f_mag * nx
                    f_c[2 * self.ref_idx + 1] += f_mag * ny

                # Contact tangent stiffness K_node = k_pen * (n x n^T)
                for a in range(2):
                    for b in range(2):
                        val = k_pen * nn_T[a, b]
                        # Slave diagonal block
                        rows.append(2 * idx + a)
                        cols.append(2 * idx + b)
                        data.append(val)

                        if has_ref:
                            # Reference diagonal block
                            rows.append(2 * self.ref_idx + a)
                            cols.append(2 * self.ref_idx + b)
                            data.append(val)

                            # Coupling cross-terms: d(f_slave)/d(u_ref) and d(f_ref)/d(u_slave)
                            rows.append(2 * idx + a)
                            cols.append(2 * self.ref_idx + b)
                            data.append(-val)

                            rows.append(2 * self.ref_idx + a)
                            cols.append(2 * idx + b)
                            data.append(-val)

        stats = {
            "active_count": active_count,
            "max_penetration": max_pen,
            "total_normal_force": total_normal_force,
        }
        return (
            f_c,
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(data, dtype=np.float64)),
            stats,
        )

    def update_augmented_multipliers(self, u: np.ndarray, relaxation: float = 1.0) -> None:
        """Update Augmented Lagrangian multipliers based on current penetration."""
        if not self.augmented_lagrange:
            return
        for nid, X0 in self.ref_coords.items():
            idx = self.nid_to_idx[nid]
            u_node = u[2 * idx : 2 * idx + 2]
            curr_pos = X0 + u_node
            gap = float(np.dot(curr_pos - self.point, self.normal)) + self.initial_offsets[nid]
            if gap < 0.0:
                self._lam[nid] += relaxation * self.penalty_stiffness * (-gap)
            else:
                self._lam[nid] = 0.0
