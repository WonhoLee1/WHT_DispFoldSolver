"""
surface_tie.py
==============
Penalty-based Surface-to-Surface Tie Constraint.

Couples slave nodes (e.g., bottom surface of display) to master segments
(e.g., top surface of rigid/deformable plate) using a distributed penalty
formulation to eliminate stress concentrations and element distortion.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


class SurfaceTieConstraint:
    """Penalty-based Surface-to-Surface Tie Constraint.

    Automatically pairs each slave node to the nearest master line segment,
    calculates projection shape functions N1(xi), N2(xi), and assembles
    coupled 3-node (Slave, Master1, Master2) penalty forces and stiffness matrices.

    Parameters
    ----------
    slave_node_ids : list of int
        Node IDs of the slave surface.
    master_node_ids : list of int
        Node IDs of the master surface (ordered along segment line).
    nid_to_idx : dict
        Mapping from global node ID to solver DOF index (0-indexed).
    coords : ndarray of shape (n_nodes, 2)
        Initial coordinates of all nodes in the mesh.
    penalty_stiffness : float
        Penalty stiffness parameter k_tie [N/mm]. Default 1e6.
    name : str
        Constraint identifier.
    """

    def __init__(
        self,
        slave_node_ids: List[int],
        master_node_ids: List[int],
        nid_to_idx: Dict[int, int],
        coords: np.ndarray,
        penalty_stiffness: float = 1e6,
        position_tolerance: float = 0.5,
        name: str = "SURFACE_TIE",
    ):
        self.slave_node_ids = list(slave_node_ids)
        self.master_node_ids = list(master_node_ids)
        self.nid_to_idx = nid_to_idx
        self.coords = coords
        self.k_tie = penalty_stiffness
        self.position_tolerance = position_tolerance
        self.name = name

        # Map pairs: list of tuples (slave_nid, master1_nid, master2_nid, xi)
        self.pairs: List[Tuple[int, int, int, float]] = []
        self._build_tie_pairs()

    def _build_tie_pairs(self) -> None:
        """Find nearest master segment for each slave node based on initial geometry."""
        # Build master segments from consecutive master nodes
        # Sort master nodes by X coordinate to form valid line segments
        m_nids = sorted(self.master_node_ids, key=lambda nid: self.coords[self.nid_to_idx[nid]][0])

        for s_nid in self.slave_node_ids:
            s_idx = self.nid_to_idx[s_nid]
            xs_0 = self.coords[s_idx]

            best_pair = None
            min_dist = float("inf")

            for i in range(len(m_nids) - 1):
                m1_nid = m_nids[i]
                m2_nid = m_nids[i + 1]

                m1_idx = self.nid_to_idx[m1_nid]
                m2_idx = self.nid_to_idx[m2_nid]

                xm1_0 = self.coords[m1_idx]
                xm2_0 = self.coords[m2_idx]

                seg_vec = xm2_0 - xm1_0
                seg_len_sq = np.dot(seg_vec, seg_vec)
                if seg_len_sq < 1e-14:
                    continue

                # Projection parameter t in [0, 1]
                t = np.dot(xs_0 - xm1_0, seg_vec) / seg_len_sq

                # Clamp t to segment bounds [0, 1] for robust pairing
                t_clamped = np.clip(t, 0.0, 1.0)
                proj_pt = xm1_0 + t_clamped * seg_vec
                dist = np.linalg.norm(xs_0 - proj_pt)

                # Convert t_clamped in [0, 1] to xi in [-1, 1]
                xi = 2.0 * t_clamped - 1.0

                if dist < min_dist:
                    min_dist = dist
                    best_pair = (s_nid, m1_nid, m2_nid, float(xi))

            if best_pair is not None and min_dist <= self.position_tolerance:
                self.pairs.append(best_pair)

    @property
    def n_active(self) -> int:
        """Number of active tied node pairs."""
        return len(self.pairs)

    def apply_penalty(self, u: np.ndarray, f_int: np.ndarray, K_eff) -> float:
        """Assemble penalty internal forces and stiffness into global solver vectors.

        Parameters
        ----------
        u : ndarray of shape (n_dofs,)
            Current displacement vector.
        f_int : ndarray of shape (n_dofs,)
            Global internal force vector (modified in-place).
        K_eff : scipy sparse matrix or ndarray
            Global effective stiffness matrix (modified in-place).

        Returns
        -------
        total_energy : float
            Total strain energy stored in tie penalty springs.
        """
        total_energy = 0.0
        I2 = np.eye(2, dtype=np.float64)

        for s_nid, m1_nid, m2_nid, xi in self.pairs:
            s_idx = self.nid_to_idx[s_nid]
            m1_idx = self.nid_to_idx[m1_nid]
            m2_idx = self.nid_to_idx[m2_nid]

            # Current coordinates: x = X0 + u
            xs = self.coords[s_idx] + u[2 * s_idx : 2 * s_idx + 2]
            xm1 = self.coords[m1_idx] + u[2 * m1_idx : 2 * m1_idx + 2]
            xm2 = self.coords[m2_idx] + u[2 * m2_idx : 2 * m2_idx + 2]

            # Shape functions
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)

            # Gap vector g = xs - (N1*xm1 + N2*xm2)
            gap = xs - (N1 * xm1 + N2 * xm2)
            total_energy += 0.5 * self.k_tie * np.dot(gap, gap)

            # Local 6-DOF index array
            dofs = np.array([
                2 * s_idx, 2 * s_idx + 1,
                2 * m1_idx, 2 * m1_idx + 1,
                2 * m2_idx, 2 * m2_idx + 1,
            ], dtype=int)

            # Penalty force vector (6,)
            f_local = self.k_tie * np.concatenate([gap, -N1 * gap, -N2 * gap])
            np.add.at(f_int, dofs, f_local)

            # Tangent stiffness block (6, 6)
            K_local = self.k_tie * np.block([
                [I2, -N1 * I2, -N2 * I2],
                [-N1 * I2, (N1**2) * I2, (N1 * N2) * I2],
                [-N2 * I2, (N1 * N2) * I2, (2.0 * N2**2 / 2.0) * I2],
            ])
            # Correct N2^2 block
            K_local[4:6, 4:6] = self.k_tie * (N2**2) * I2

            # Assemble into K_eff
            if hasattr(K_eff, "toarray"):
                # Dense or SciPy sparse matrix
                for ii in range(6):
                    for jj in range(6):
                        K_eff[dofs[ii], dofs[jj]] += K_local[ii, jj]
            else:
                np.add.at(K_eff, (dofs[:, None], dofs[None, :]), K_local)

        return total_energy
