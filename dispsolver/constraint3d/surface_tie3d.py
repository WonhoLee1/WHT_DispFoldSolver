"""
surface_tie3d.py
================
Penalty-based 3D Surface-to-Surface Tie Constraint.

Couples 3D slave nodes (e.g., bottom surface of 3D display panel) to 3D master
surface faces (e.g., top face of 3D rigid support plate) using a 3D distributed
penalty formulation with Quad4/Tri3 projection to eliminate stress concentrations
and non-physical nodal pinning.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


def _project_point_to_quad4(
    p: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    q3: np.ndarray,
    q4: np.ndarray,
    max_iter: int = 10,
    tol: float = 1e-6
) -> Tuple[float, float, float, np.ndarray]:
    """Project 3D point p onto 3D Quad4 face defined by vertices (q1, q2, q3, q4).

    Returns
    -------
    xi : float
        Parametric xi coordinate in [-1, 1]
    eta : float
        Parametric eta coordinate in [-1, 1]
    dist : float
        Euclidean distance from p to projected surface point
    proj_pt : np.ndarray
        (3,) Projected point on Quad4 surface
    """
    # Initial guess xi = 0, eta = 0
    xi, eta = 0.0, 0.0

    for _ in range(max_iter):
        N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
        N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
        N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
        N4 = 0.25 * (1.0 - xi) * (1.0 + eta)

        x_surf = N1 * q1 + N2 * q2 + N3 * q3 + N4 * q4
        r = x_surf - p

        # Derivatives with respect to xi and eta
        dN1_dxi = -0.25 * (1.0 - eta)
        dN2_dxi =  0.25 * (1.0 - eta)
        dN3_dxi =  0.25 * (1.0 + eta)
        dN4_dxi = -0.25 * (1.0 + eta)

        dN1_deta = -0.25 * (1.0 - xi)
        dN2_deta = -0.25 * (1.0 + xi)
        dN3_deta =  0.25 * (1.0 + xi)
        dN4_deta =  0.25 * (1.0 - xi)

        dx_dxi = dN1_dxi * q1 + dN2_dxi * q2 + dN3_dxi * q3 + dN4_dxi * q4
        dx_deta = dN1_deta * q1 + dN2_deta * q2 + dN3_deta * q3 + dN4_deta * q4

        # Solve system: J * [dxi, deta]^T = -[r . dx_dxi, r . dx_deta]^T
        J11 = np.dot(dx_dxi, dx_dxi)
        J12 = np.dot(dx_dxi, dx_deta)
        J22 = np.dot(dx_deta, dx_deta)

        detJ = J11 * J22 - J12 * J12
        if abs(detJ) < 1e-14:
            break

        rhs1 = -np.dot(r, dx_dxi)
        rhs2 = -np.dot(r, dx_deta)

        dxi = (J22 * rhs1 - J12 * rhs2) / detJ
        deta = (-J12 * rhs1 + J11 * rhs2) / detJ

        xi = float(np.clip(xi + dxi, -2.0, 2.0))
        eta = float(np.clip(eta + deta, -2.0, 2.0))

        if abs(dxi) < tol and abs(deta) < tol:
            break

    # Clamp to quad boundaries [-1, 1]
    xi_clamped = float(np.clip(xi, -1.0, 1.0))
    eta_clamped = float(np.clip(eta, -1.0, 1.0))

    N1 = 0.25 * (1.0 - xi_clamped) * (1.0 - eta_clamped)
    N2 = 0.25 * (1.0 + xi_clamped) * (1.0 - eta_clamped)
    N3 = 0.25 * (1.0 + xi_clamped) * (1.0 + eta_clamped)
    N4 = 0.25 * (1.0 - xi_clamped) * (1.0 + eta_clamped)

    proj_pt = N1 * q1 + N2 * q2 + N3 * q3 + N4 * q4
    dist = float(np.linalg.norm(p - proj_pt))

    return xi_clamped, eta_clamped, dist, proj_pt


class SurfaceTieConstraint3D:
    """Penalty-based 3D Surface-to-Surface Tie Constraint.

    Automatically pairs each slave node to the nearest master 3D Quad face,
    calculates projection shape functions N1..N4(xi, eta), and assembles
    coupled 5-node (1 Slave, 4 Master Quad Vertices) penalty forces and stiffness matrices.

    Parameters
    ----------
    slave_node_ids : list of int
        Node IDs of the 3D slave surface.
    master_faces : list of tuple of 4 int
        List of Quad4 master face node IDs [(m1, m2, m3, m4), ...].
    nid_to_idx : dict
        Mapping from global node ID to solver node index (0-indexed).
    coords : ndarray of shape (n_nodes, 3)
        Initial 3D coordinates of all nodes in the mesh.
    penalty_stiffness : float
        Penalty stiffness parameter k_tie [N/mm]. Default 1e6.
    position_tolerance : float
        Maximum search radius [mm] for initial projection pairing.
    name : str
        Constraint identifier.
    """

    def __init__(
        self,
        slave_node_ids: List[int],
        master_faces: List[Tuple[int, int, int, int]],
        nid_to_idx: Dict[int, int],
        coords: np.ndarray,
        penalty_stiffness: float = 1e6,
        position_tolerance: float = 2.0,
        name: str = "SURFACE_TIE_3D",
    ):
        self.slave_node_ids = list(slave_node_ids)
        self.master_faces = list(master_faces)
        self.nid_to_idx = nid_to_idx
        self.coords = coords
        self.k_tie = float(penalty_stiffness)
        self.position_tolerance = float(position_tolerance)
        self.name = name

        # Persistent per-slave-node Augmented Lagrange multipliers (3D vector)
        self._lam_dict: Dict[int, np.ndarray] = {
            s_nid: np.zeros(3, dtype=np.float64) for s_nid in self.slave_node_ids
        }

        # Map pairs: list of tuples (slave_nid, (m1_nid, m2_nid, m3_nid, m4_nid), xi, eta)
        self.pairs: List[Tuple[int, Tuple[int, int, int, int], float, float]] = []
        self._build_tie_pairs()

    def _build_tie_pairs( me ) -> None:
        """Find nearest master face for each slave node based on initial 3D geometry."""
        for s_nid in me.slave_node_ids:
            s_idx = me.nid_to_idx[s_nid]
            xs_0 = me.coords[s_idx]

            best_pair = None
            min_dist = float("inf")

            for m_face in me.master_faces:
                m1_idx = me.nid_to_idx[m_face[0]]
                m2_idx = me.nid_to_idx[m_face[1]]
                m3_idx = me.nid_to_idx[m_face[2]]
                m4_idx = me.nid_to_idx[m_face[3]]

                q1 = me.coords[m1_idx]
                q2 = me.coords[m2_idx]
                q3 = me.coords[m3_idx]
                q4 = me.coords[m4_idx]

                xi, eta, dist, _ = _project_point_to_quad4(xs_0, q1, q2, q3, q4)

                if dist < min_dist:
                    min_dist = dist
                    best_pair = (s_nid, m_face, xi, eta)

            if best_pair is not None and min_dist <= me.position_tolerance:
                me.pairs.append(best_pair)
        me._initial_tied_slave_nids = [pair[0] for pair in me.pairs]

    def reproject_deformed(self, u: np.ndarray) -> None:
        """Dynamically update projection parameters (xi, eta) for all slave nodes using current 3D deformed geometry."""
        target_slave_nids = getattr(self, "_initial_tied_slave_nids", self.slave_node_ids)
        if not target_slave_nids or not self.master_faces:
            return

        new_pairs = []
        for s_nid in target_slave_nids:
            s_idx = self.nid_to_idx[s_nid]
            xs = self.coords[s_idx] + u[3 * s_idx : 3 * s_idx + 3]

            best_pair = None
            min_dist = float("inf")

            for m_face in self.master_faces:
                m1_idx = self.nid_to_idx[m_face[0]]
                m2_idx = self.nid_to_idx[m_face[1]]
                m3_idx = self.nid_to_idx[m_face[2]]
                m4_idx = self.nid_to_idx[m_face[3]]

                q1 = self.coords[m1_idx] + u[3 * m1_idx : 3 * m1_idx + 3]
                q2 = self.coords[m2_idx] + u[3 * m2_idx : 3 * m2_idx + 3]
                q3 = self.coords[m3_idx] + u[3 * m3_idx : 3 * m3_idx + 3]
                q4 = self.coords[m4_idx] + u[3 * m4_idx : 3 * m4_idx + 3]

                xi, eta, dist, _ = _project_point_to_quad4(xs, q1, q2, q3, q4)

                if dist < min_dist:
                    min_dist = dist
                    best_pair = (s_nid, m_face, xi, eta)

            if best_pair is not None:
                new_pairs.append(best_pair)

        self.pairs = new_pairs

    def assemble(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """Assemble 3D penalty residual forces and stiffness matrices for all tie pairs.

        Parameters
        ----------
        u : np.ndarray
            (3 * n_nodes,) Global 3D displacement vector

        Returns
        -------
        f_tie : np.ndarray
            (3 * n_nodes,) Global 3D internal tie penalty force vector
        K_tie_data : tuple or ndarray
            Triple (rows, cols, data) for COO sparse matrix assembly
        stats : dict
            Diagnostic metrics (max_gap, mean_gap)
        """
        n_dof = len(u)
        f_tie = np.zeros(n_dof, dtype=np.float64)

        rows, cols, data = [], [], []

        max_gap = 0.0
        sum_gap = 0.0

        for s_nid, m_face, xi, eta in self.pairs:
            s_idx = self.nid_to_idx[s_nid]
            m1_idx = self.nid_to_idx[m_face[0]]
            m2_idx = self.nid_to_idx[m_face[1]]
            m3_idx = self.nid_to_idx[m_face[2]]
            m4_idx = self.nid_to_idx[m_face[3]]

            # Current 3D positions
            xs = self.coords[s_idx] + u[3 * s_idx : 3 * s_idx + 3]
            xm1 = self.coords[m1_idx] + u[3 * m1_idx : 3 * m1_idx + 3]
            xm2 = self.coords[m2_idx] + u[3 * m2_idx : 3 * m2_idx + 3]
            xm3 = self.coords[m3_idx] + u[3 * m3_idx : 3 * m3_idx + 3]
            xm4 = self.coords[m4_idx] + u[3 * m4_idx : 3 * m4_idx + 3]

            # Quad4 shape functions
            N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
            N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
            N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
            N4 = 0.25 * (1.0 - xi) * (1.0 + eta)

            # Master projected position
            xm_proj = N1 * xm1 + N2 * xm2 + N3 * xm3 + N4 * xm4

            # Gap vector g = xs - xm_proj
            gap_vec = xs - xm_proj
            gap_norm = float(np.linalg.norm(gap_vec))

            if gap_norm > max_gap:
                max_gap = gap_norm
            sum_gap += gap_norm

            # Penalty Force: f_s = k_tie * gap_vec, f_ma = -N_a * k_tie * gap_vec
            f_slave = self.k_tie * gap_vec
            f_tie[3 * s_idx : 3 * s_idx + 3] += f_slave

            f_tie[3 * m1_idx : 3 * m1_idx + 3] -= N1 * f_slave
            f_tie[3 * m2_idx : 3 * m2_idx + 3] -= N2 * f_slave
            f_tie[3 * m3_idx : 3 * m3_idx + 3] -= N3 * f_slave
            f_tie[3 * m4_idx : 3 * m4_idx + 3] -= N4 * f_slave

            # Local 15-DOF Linear Transformation Matrix L (3 x 15)
            # L = [ I_3, -N1*I_3, -N2*I_3, -N3*I_3, -N4*I_3 ]
            node_indices = [s_idx, m1_idx, m2_idx, m3_idx, m4_idx]
            weights = [1.0, -N1, -N2, -N3, -N4]

            for a in range(5):
                idx_a = node_indices[a]
                w_a = weights[a]
                for b in range(5):
                    idx_b = node_indices[b]
                    w_b = weights[b]

                    k_ab = self.k_tie * w_a * w_b

                    for d in range(3):
                        rows.append(3 * idx_a + d)
                        cols.append(3 * idx_b + d)
                        data.append(k_ab)

        n_pairs = max(len(self.pairs), 1)
        stats = {
            "max_gap": max_gap,
            "mean_gap": sum_gap / n_pairs,
            "n_pairs": len(self.pairs)
        }

        return f_tie, (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(data, dtype=np.float64)), stats
