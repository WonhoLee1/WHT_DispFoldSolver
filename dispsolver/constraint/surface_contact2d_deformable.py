"""
surface_contact2d_deformable.py
===============================
2D Deformable-to-Deformable Node-to-Segment (NTS) Contact Constraint with:
- Continuous C0 Nodal-Averaged Normal Smoothing (eliminates facet corner chattering)
- Penetration-priority dynamic segment tracking for Finite Sliding support across multi-segment surfaces
- Incremental Coulomb friction with physical relative slip tracking (mm) and radial return:
    du = u - u_inc_start
    du_m = (1 - xi) * du_m1 + xi * du_m2
    d_s_rel = (du_s - du_m) . tangent
    s_trial = s_elastic_n + d_s_rel
    s_crit = (mu * f_mag) / k_t
    Stick: ft = k_t * s_trial, Ktt = k_t * (tangent (x) tangent)
    Slip : ft = mu * f_mag * sign(s_trial), Ktt = (mu * f_mag / |s_trial|) * (tangent (x) tangent)
- Exact consistent residual force assembly: f_vec = f_mag * n - f_t, f_c[node_a] += -weights[a] * f_vec
- Exact consistent tangent stiffness assembly: K_ab = weights[a] * weights[b] * (k_diag * n n^T + Ktt)
- Incremental friction update upon converged increments (commit_friction_state).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


class DeformableSurfaceContactConstraint2D:
    """2D Node-to-Segment Contact Constraint between two deformable bodies."""

    def __init__(
        self,
        slave_node_ids: List[int],
        master_segments: List[Tuple[int, int]],
        nid_to_idx: Dict[int, int],
        coords: np.ndarray,
        penalty_stiffness: float = 1.0e6,
        law: Optional[Any] = None,
        augmented_lagrange: bool = False,
        chatter_stabilization: bool = False,
        hysteresis_band: float = 0.0,
        friction_coefficient: float = 0.0,
        tangential_stiffness_ratio: float = 0.1,
        position_tolerance: float = 2.0,
        name: str = "DEFORMABLE_CONTACT_2D",
    ):
        self.name = name
        self.slave_node_ids = list(slave_node_ids)
        self.master_segments = list(master_segments)
        self.nid_to_idx = dict(nid_to_idx)
        self.coords = coords[:, :2].copy()
        self.k_contact = float(penalty_stiffness)
        self.law = law
        self.augmented_lagrange = bool(augmented_lagrange)
        self.chatter_stabilization = bool(chatter_stabilization)
        self.hysteresis_band = float(hysteresis_band)
        self.mu = float(friction_coefficient)
        self.k_t = self.k_contact * float(tangential_stiffness_ratio)
        self.position_tolerance = float(position_tolerance)

        # Precompute master segment reference normals and signs
        self._master_signs: Dict[Tuple[int, int], float] = {}
        self._init_master_normals()

        # Incremental friction tracking: accumulated elastic slip per slave node
        self._elastic_slip: Dict[int, float] = {nid: 0.0 for nid in self.slave_node_ids}
        self._u_inc_start: Optional[np.ndarray] = None
        self._gate_active: Dict[int, bool] = {nid: False for nid in self.slave_node_ids}
        self._lam: Dict[int, float] = {nid: 0.0 for nid in self.slave_node_ids}

    def _init_master_normals(self) -> None:
        """Determine outward normal sign for master segments relative to slave body."""
        for m1_nid, m2_nid in self.master_segments:
            if m1_nid not in self.nid_to_idx or m2_nid not in self.nid_to_idx:
                continue
            m1_idx = self.nid_to_idx[m1_nid]
            m2_idx = self.nid_to_idx[m2_nid]
            xm1 = self.coords[m1_idx]
            xm2 = self.coords[m2_idx]

            seg_vec = xm2 - xm1
            L_seg = float(np.linalg.norm(seg_vec))
            if L_seg < 1e-12:
                continue
            tangent = seg_vec / L_seg
            # Standard clockwise 90-degree outward normal for CCW boundary elements: [tangent[1], -tangent[0]]
            n_raw = np.array([tangent[1], -tangent[0]], dtype=np.float64)

            # Find nearest slave node in reference geometry to detect outward orientation
            best_dot = 0.0
            for s_nid in self.slave_node_ids:
                if s_nid not in self.nid_to_idx:
                    continue
                s_idx = self.nid_to_idx[s_nid]
                xs0 = self.coords[s_idx]
                dot_val = float(np.dot(xs0 - xm1, n_raw))
                if abs(dot_val) > abs(best_dot):
                    best_dot = dot_val

            sign = 1.0 if best_dot >= 0.0 else -1.0
            self._master_signs[(m1_nid, m2_nid)] = sign

    def _compute_nodal_normals(self, u: np.ndarray) -> Dict[int, np.ndarray]:
        """Compute area-weighted continuous nodal normals on master surface."""
        nodal_normals: Dict[int, np.ndarray] = {}
        for m1_nid, m2_nid in self.master_segments:
            if m1_nid not in self.nid_to_idx or m2_nid not in self.nid_to_idx:
                continue
            m1_idx = self.nid_to_idx[m1_nid]
            m2_idx = self.nid_to_idx[m2_nid]
            xm1 = self.coords[m1_idx] + u[2 * m1_idx : 2 * m1_idx + 2]
            xm2 = self.coords[m2_idx] + u[2 * m2_idx : 2 * m2_idx + 2]
            seg_vec = xm2 - xm1
            L_seg = float(np.linalg.norm(seg_vec))
            if L_seg < 1e-12:
                continue
            tangent = seg_vec / L_seg
            sign = self._master_signs.get((m1_nid, m2_nid), 1.0)
            n_seg = sign * np.array([tangent[1], -tangent[0]], dtype=np.float64)

            for nid in (m1_nid, m2_nid):
                if nid not in nodal_normals:
                    nodal_normals[nid] = np.zeros(2, dtype=np.float64)
                nodal_normals[nid] += n_seg * L_seg

        for nid in nodal_normals:
            norm_val = np.linalg.norm(nodal_normals[nid])
            if norm_val > 1e-12:
                nodal_normals[nid] /= norm_val

        return nodal_normals

    def _find_active_pair(
        self, u: np.ndarray, s_nid: int, nodal_normals: Optional[Dict[int, np.ndarray]] = None
    ) -> Optional[Tuple[int, int, float, np.ndarray, np.ndarray, float, float]]:
        """Find the best master segment for a slave node at current displacement u.

        Penetration-priority selection:
        - If any candidate segment shows penetration (gap < 0), pick the one with most negative gap.
        - Otherwise, pick the candidate with minimum positive gap (closest approach).

        Returns:
            (m1_nid, m2_nid, gap, normal, tangent, xi, L_seg) or None if outside tolerance.
        """
        s_idx = self.nid_to_idx[s_nid]
        xs = self.coords[s_idx] + u[2 * s_idx : 2 * s_idx + 2]

        if nodal_normals is None:
            nodal_normals = self._compute_nodal_normals(u)

        best_penetrating = None
        min_penetrating_gap = 0.0  # Most negative
        best_open = None
        min_open_gap = float("inf")

        for m1_nid, m2_nid in self.master_segments:
            if m1_nid not in self.nid_to_idx or m2_nid not in self.nid_to_idx:
                continue
            m1_idx = self.nid_to_idx[m1_nid]
            m2_idx = self.nid_to_idx[m2_nid]
            xm1 = self.coords[m1_idx] + u[2 * m1_idx : 2 * m1_idx + 2]
            xm2 = self.coords[m2_idx] + u[2 * m2_idx : 2 * m2_idx + 2]

            seg_vec = xm2 - xm1
            L_seg = float(np.linalg.norm(seg_vec))
            if L_seg < 1e-12:
                continue
            tangent_raw = seg_vec / L_seg

            xi = float(np.dot(xs - xm1, tangent_raw) / L_seg)
            # Accept projection within slight extension [-0.10, 1.10]
            if not (-0.10 <= xi <= 1.10):
                continue

            xi_clamped = np.clip(xi, 0.0, 1.0)
            xm_proj = (1.0 - xi_clamped) * xm1 + xi_clamped * xm2

            # Continuous C0 normal from nodal normals
            n1 = nodal_normals.get(m1_nid, None)
            n2 = nodal_normals.get(m2_nid, None)
            if n1 is not None and n2 is not None:
                n_interp = (1.0 - xi_clamped) * n1 + xi_clamped * n2
                norm_n = float(np.linalg.norm(n_interp))
                normal = n_interp / norm_n if norm_n > 1e-12 else n1
            else:
                sign = self._master_signs.get((m1_nid, m2_nid), 1.0)
                normal = sign * np.array([tangent_raw[1], -tangent_raw[0]], dtype=np.float64)

            # Continuous tangent orthogonal to smoothed normal, aligned with tangent_raw
            tangent = np.array([normal[1], -normal[0]], dtype=np.float64)
            if np.dot(tangent, tangent_raw) < 0.0:
                tangent = -tangent

            gap = float(np.dot(xs - xm_proj, normal))

            dist = float(np.linalg.norm(xs - xm_proj))
            if dist > self.position_tolerance:
                continue

            if gap < 0.0:
                if gap < min_penetrating_gap:
                    min_penetrating_gap = gap
                    best_penetrating = (m1_nid, m2_nid, gap, normal, tangent, xi_clamped, L_seg)
            else:
                if gap < min_open_gap:
                    min_open_gap = gap
                    best_open = (m1_nid, m2_nid, gap, normal, tangent, xi_clamped, L_seg)

        return best_penetrating if best_penetrating is not None else best_open

    def get_active_set(self, u: np.ndarray) -> frozenset:
        """Return set of currently active (penetrated) slave nodes for SDI tracking."""
        active = set()
        nodal_normals = self._compute_nodal_normals(u)
        for s_nid in self.slave_node_ids:
            res = self._find_active_pair(u, s_nid, nodal_normals=nodal_normals)
            if res is not None:
                _, _, gap, _, _, _, _ = res
                if gap < 0.0:
                    active.add(s_nid)
        return frozenset(active)

    def commit_friction_state(self, u: np.ndarray) -> None:
        """Commit friction slip state at converged increment boundary via radial return."""
        if self._u_inc_start is None:
            self._u_inc_start = u.copy()
            return

        du = u - self._u_inc_start
        nodal_normals = self._compute_nodal_normals(u)

        if self.mu > 0.0:
            for s_nid in self.slave_node_ids:
                res = self._find_active_pair(u, s_nid, nodal_normals=nodal_normals)
                if res is None:
                    self._elastic_slip[s_nid] = 0.0
                    continue
                m1_nid, m2_nid, gap, _, tangent, xi, _ = res
                if gap >= 0.0:
                    self._elastic_slip[s_nid] = 0.0
                    continue

                penetration = -gap
                f_mag = self.k_contact * penetration

                s_idx = self.nid_to_idx[s_nid]
                m1_idx = self.nid_to_idx[m1_nid]
                m2_idx = self.nid_to_idx[m2_nid]

                dus = du[2 * s_idx : 2 * s_idx + 2]
                dum1 = du[2 * m1_idx : 2 * m1_idx + 2]
                dum2 = du[2 * m2_idx : 2 * m2_idx + 2]
                dum = (1.0 - xi) * dum1 + xi * dum2

                d_s_rel = float(np.dot(dus - dum, tangent))
                s_trial = self._elastic_slip.get(s_nid, 0.0) + d_s_rel
                s_crit = (self.mu * f_mag / self.k_t) if self.k_t > 0.0 else 0.0

                if abs(s_trial) <= s_crit:
                    self._elastic_slip[s_nid] = s_trial
                else:
                    sgn = 1.0 if s_trial > 0.0 else -1.0
                    self._elastic_slip[s_nid] = s_crit * sgn

        # Advance increment start displacement
        self._u_inc_start = u.copy()

    def assemble(self, u: np.ndarray) -> Tuple[csr_matrix, np.ndarray, Dict[str, Any]]:
        """Assemble 2D contact tangent stiffness matrix K_c and residual force f_c.

        Sign convention (exact 3D Deformable parity):
        Residual equation: R = f_ext - f_int_elem - f_c = 0
        Physical contact force pushing slave away from master: F_phys_slave = f_mag * n - f_t
        f_int uses f_int = -F_phys, so:
            f_node_a = -weights[a] * (f_mag * n - f_t)
            K_ab = weights[a] * weights[b] * (k_diag * n n^T + Ktt)
        """
        n_dofs = len(u)
        if self._u_inc_start is None:
            self._u_inc_start = u.copy()

        du = u - self._u_inc_start
        nodal_normals = self._compute_nodal_normals(u)

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        f_c = np.zeros(n_dofs, dtype=np.float64)

        n_active = 0
        max_pen = 0.0
        total_fn = 0.0
        total_ft = 0.0

        for s_nid in self.slave_node_ids:
            res = self._find_active_pair(u, s_nid, nodal_normals=nodal_normals)
            if res is None:
                continue

            m1_nid, m2_nid, gap, normal, tangent, xi, _ = res
            penetration = -gap
            if penetration <= 0.0:
                continue

            n_active += 1
            if penetration > max_pen:
                max_pen = penetration

            # Normal compressive contact force magnitude: f_mag = k_contact * penetration
            f_mag = self.k_contact * penetration
            k_diag = self.k_contact
            total_fn += f_mag

            s_idx = self.nid_to_idx[s_nid]
            m1_idx = self.nid_to_idx[m1_nid]
            m2_idx = self.nid_to_idx[m2_nid]

            # Shape function weights: N1 for m1, N2 for m2
            N1 = 1.0 - xi
            N2 = xi
            weights = np.array([1.0, -N1, -N2], dtype=np.float64)
            node_indices = [s_idx, m1_idx, m2_idx]

            # Tangential friction
            f_t_2d = np.zeros(2, dtype=np.float64)
            Ktt = np.zeros((2, 2), dtype=np.float64)

            if self.mu > 0.0:
                dus = du[2 * s_idx : 2 * s_idx + 2]
                dum1 = du[2 * m1_idx : 2 * m1_idx + 2]
                dum2 = du[2 * m2_idx : 2 * m2_idx + 2]
                dum = N1 * dum1 + N2 * dum2

                d_s_rel = float(np.dot(dus - dum, tangent))
                s_trial = self._elastic_slip.get(s_nid, 0.0) + d_s_rel
                s_crit = (self.mu * f_mag / self.k_t) if self.k_t > 0.0 else 0.0

                T = np.outer(tangent, tangent)
                if abs(s_trial) <= s_crit:
                    # Stick state
                    ft = self.k_t * s_trial
                    Ktt = self.k_t * T
                else:
                    # Slip state: Coulomb limit traction with continuous secant tangent
                    sgn = 1.0 if s_trial > 0.0 else -1.0
                    ft = self.mu * f_mag * sgn
                    k_sec = (self.mu * f_mag) / max(abs(s_trial), s_crit)
                    Ktt = k_sec * T

                f_t_2d = ft * tangent
                total_ft += float(np.linalg.norm(f_t_2d))

            # Vector combination: f_vec = f_mag * normal - f_t_2d
            # Nodal force contribution: f_node_a = -weights[a] * f_vec
            f_vec = f_mag * normal - f_t_2d
            for a, idx_a in enumerate(node_indices):
                w_a = weights[a]
                f_c[2 * idx_a : 2 * idx_a + 2] += -w_a * f_vec

            # Tangent stiffness: K_local = k_diag * (n x n^T) + Ktt
            # K_ab = weights[a] * weights[b] * K_local
            K_local = k_diag * np.outer(normal, normal) + Ktt

            for a, idx_a in enumerate(node_indices):
                w_a = weights[a]
                dofs_a = np.array([2 * idx_a, 2 * idx_a + 1], dtype=np.int32)
                for b, idx_b in enumerate(node_indices):
                    w_b = weights[b]
                    dofs_b = np.array([2 * idx_b, 2 * idx_b + 1], dtype=np.int32)
                    K_block = (w_a * w_b) * K_local

                    rows.extend([dofs_a[0], dofs_a[0], dofs_a[1], dofs_a[1]])
                    cols.extend([dofs_b[0], dofs_b[1], dofs_b[0], dofs_b[1]])
                    data.extend([K_block[0, 0], K_block[0, 1], K_block[1, 0], K_block[1, 1]])

        if rows:
            K_c = csr_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs))
        else:
            K_c = csr_matrix((n_dofs, n_dofs), dtype=np.float64)

        stats = {
            "active_nodes": n_active,
            "max_penetration": max_pen,
            "total_normal_force": total_fn,
            "total_tangential_force": total_ft,
        }
        return K_c, f_c, stats
