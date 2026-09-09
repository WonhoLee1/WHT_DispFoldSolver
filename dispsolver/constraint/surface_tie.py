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
        self.k_tie = float(penalty_stiffness)
        self.position_tolerance = position_tolerance
        self.name = name
        self._k_auto_scale = 1.0

        # Persistent per-slave-node Augmented Lagrange multipliers
        self._lam_dict: Dict[int, np.ndarray] = {s_nid: np.zeros(2, dtype=np.float64) for s_nid in self.slave_node_ids}

        # Topological master node ordering along segment line (fixed by initial geometry)
        self._master_nids: List[int] = sorted(self.master_node_ids, key=lambda nid: self.coords[self.nid_to_idx[nid]][0])
        self.master_segments: List[Tuple[int, int]] = [(self._master_nids[i], self._master_nids[i + 1]) for i in range(len(self._master_nids) - 1)]

        # Map pairs: list of tuples (slave_nid, master1_nid, master2_nid, xi)
        self.pairs: List[Tuple[int, int, int, float]] = []
        self._build_tie_pairs()

    @property
    def _lam(self) -> Optional[np.ndarray]:
        if not self.pairs:
            return None
        return np.array([self._lam_dict[pair[0]] for pair in self.pairs], dtype=np.float64)

    def _build_tie_pairs(self) -> None:
        """Find nearest master segment for each slave node based on initial geometry."""
        for s_nid in self.slave_node_ids:
            s_idx = self.nid_to_idx[s_nid]
            xs_0 = self.coords[s_idx]

            best_pair = None
            min_dist = float("inf")

            for m1_nid, m2_nid in self.master_segments:
                m1_idx = self.nid_to_idx[m1_nid]
                m2_idx = self.nid_to_idx[m2_nid]

                xm1_0 = self.coords[m1_idx]
                xm2_0 = self.coords[m2_idx]

                seg_vec = xm2_0 - xm1_0
                seg_len_sq = np.dot(seg_vec, seg_vec)
                if seg_len_sq < 1e-14:
                    continue

                t = np.dot(xs_0 - xm1_0, seg_vec) / seg_len_sq
                t_clamped = np.clip(t, 0.0, 1.0)
                proj_pt = xm1_0 + t_clamped * seg_vec
                dist = np.linalg.norm(xs_0 - proj_pt)

                xi = 2.0 * t_clamped - 1.0

                if dist < min_dist:
                    min_dist = dist
                    best_pair = (s_nid, m1_nid, m2_nid, float(xi))

            if best_pair is not None and min_dist <= self.position_tolerance:
                self.pairs.append(best_pair)
        self._initial_tied_slave_nids = [pair[0] for pair in self.pairs]

    def reproject_deformed(self, u: np.ndarray) -> None:
        """Dynamically re-neighbor and update projection parameter xi for all slave nodes using current deformed geometry."""
        target_slave_nids = getattr(self, "_initial_tied_slave_nids", self.slave_node_ids)
        if not target_slave_nids or not self.master_segments:
            return

        new_pairs = []
        for s_nid in target_slave_nids:
            s_idx = self.nid_to_idx[s_nid]
            xs = self.coords[s_idx] + u[2 * s_idx : 2 * s_idx + 2]

            best_pair = None
            min_dist = float("inf")

            for m1_nid, m2_nid in self.master_segments:
                m1_idx = self.nid_to_idx[m1_nid]
                m2_idx = self.nid_to_idx[m2_nid]

                xm1 = self.coords[m1_idx] + u[2 * m1_idx : 2 * m1_idx + 2]
                xm2 = self.coords[m2_idx] + u[2 * m2_idx : 2 * m2_idx + 2]

                seg_vec = xm2 - xm1
                seg_len_sq = np.dot(seg_vec, seg_vec)
                if seg_len_sq < 1e-14:
                    continue

                t = np.dot(xs - xm1, seg_vec) / seg_len_sq
                t_clamped = np.clip(t, 0.0, 1.0)
                proj_pt = xm1 + t_clamped * seg_vec
                dist = np.linalg.norm(xs - proj_pt)

                xi = 2.0 * t_clamped - 1.0

                if dist < min_dist:
                    min_dist = dist
                    best_pair = (s_nid, m1_nid, m2_nid, float(xi))

            if best_pair is not None:
                new_pairs.append(best_pair)

        self.pairs = new_pairs

    @property
    def n_active(self) -> int:
        return len(self.pairs)

    def auto_scale_k(self, E_ref: float = 4000.0, h_elem: float = 0.5, beta: float = 50.0) -> float:
        k_auto = beta * E_ref * h_elem
        self._k_auto_scale = float(k_auto)
        return float(k_auto)

    def update_augmented_lagrange(self, u: np.ndarray) -> float:
        """Update Augmented Lagrange multipliers based on current tie gap."""
        if not self.pairs:
            return 0.0
        max_gap = 0.0
        for s_nid, m1_nid, m2_nid, xi in self.pairs:
            s_idx = self.nid_to_idx[s_nid]
            m1_idx = self.nid_to_idx[m1_nid]
            m2_idx = self.nid_to_idx[m2_nid]
            xs = self.coords[s_idx] + u[2 * s_idx : 2 * s_idx + 2]
            xm1 = self.coords[m1_idx] + u[2 * m1_idx : 2 * m1_idx + 2]
            xm2 = self.coords[m2_idx] + u[2 * m2_idx : 2 * m2_idx + 2]
            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)
            gap = xs - (N1 * xm1 + N2 * xm2)
            self._lam_dict[s_nid] += self.k_tie * gap
            max_gap = max(max_gap, float(np.linalg.norm(gap)))
        return max_gap

    def reset_augmented_lagrange(self) -> None:
        for s_nid in self._lam_dict:
            self._lam_dict[s_nid].fill(0.0)

    def apply_penalty(self, u: np.ndarray, f_int: np.ndarray, K_eff) -> float:
        """Assemble penalty internal forces and stiffness into global solver vectors with rotational tangent coupling.

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

            xs = self.coords[s_idx] + u[2 * s_idx : 2 * s_idx + 2]
            xm1 = self.coords[m1_idx] + u[2 * m1_idx : 2 * m1_idx + 2]
            xm2 = self.coords[m2_idx] + u[2 * m2_idx : 2 * m2_idx + 2]

            N1 = 0.5 * (1.0 - xi)
            N2 = 0.5 * (1.0 + xi)

            gap = xs - (N1 * xm1 + N2 * xm2)
            gap_norm = float(np.linalg.norm(gap))
            total_energy += 0.5 * self.k_tie * (gap_norm ** 2)

            lam = self._lam_dict[s_nid]
            total_energy += float(np.dot(lam, gap))

            dofs = np.array([
                2 * s_idx, 2 * s_idx + 1,
                2 * m1_idx, 2 * m1_idx + 1,
                2 * m2_idx, 2 * m2_idx + 1,
            ], dtype=int)

            f_pen = self.k_tie * gap + lam
            f_local = np.concatenate([f_pen, -N1 * f_pen, -N2 * f_pen])
            np.add.at(f_int, dofs, f_local)

            # Standard 3-node kinematic block (6, 6)
            B_mat = np.block([[I2, -N1 * I2, -N2 * I2]])
            K_local = self.k_tie * (B_mat.T @ B_mat)

            # Rotational / Large Deformation Tangent Stiffness Coupling
            seg_vec = xm2 - xm1
            L2 = np.dot(seg_vec, seg_vec)
            if L2 > 1e-12:
                # Normal-tangent projection
                t_vec = seg_vec / np.sqrt(L2)
                P_perp = I2 - np.outer(t_vec, t_vec)
                # Geometric rotational coupling term
                G_mat = np.block([[np.zeros((2, 2)), -I2, I2]])
                K_rot = - (self.k_tie / np.sqrt(L2)) * (B_mat.T @ np.outer(t_vec, gap) @ G_mat)
                K_local = K_local + 0.5 * (K_rot + K_rot.T)

            # Assemble into K_eff
            if K_eff is not None:
                if hasattr(K_eff, "toarray"):
                    for ii in range(6):
                        for jj in range(6):
                            K_eff[dofs[ii], dofs[jj]] += K_local[ii, jj]
                else:
                    np.add.at(K_eff, (dofs[:, None], dofs[None, :]), K_local)

        return total_energy

