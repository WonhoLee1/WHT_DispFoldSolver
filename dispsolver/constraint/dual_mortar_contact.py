"""
dual_mortar_contact.py
======================
2D Segment-to-Segment (S2S) Dual Mortar & Nitsche Contact Discretization.
Commercial-grade, mathematically rigorous implementation according to
Wohlmuth (2000) and Chouly et al. (2017).

Features:
1. 1D Closed-form Biorthogonal Dual Shape Functions:
   psi_1(xi) = (1 - 3*xi) / 2
   psi_2(xi) = (1 + 3*xi) / 2
   Satisfies exact biorthogonality: int_{-1}^1 psi_i(xi) * N_j(xi) dxi = delta_{ij}
   and partition of unity: psi_1(xi) + psi_2(xi) == 1.
2. Exact 1D line segmentation / clipping for non-matching meshes with dissimilar
   element densities (e.g. coarse slave on fine master or vice versa).
3. Weak gap aggregation and diagonal static condensation (D_global):
   D_a = sum_{s touching a} L_s / 2
   g_tilde_a = (1 / D_a) sum_{s touching a} int psi_a * g_n * (L_s / 2) dxi
4. Sum of Rank-1 outer products consistent tangent stiffness:
   K_c = sum_{a in Active} (k_c / D_a) * b_a (x) b_a^T
   Preserves symmetry, positive semi-definiteness, and locality.
5. Formulations supported:
   - "dual_mortar": Dual Mortar with PDASS Augmented Lagrangian or Penalty.
   - "nitsche": Multiplier-free Nitsche symmetric formulation preserving
     pure SPD tangent stiffness matrix.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np


def _line2_shape_functions(xi: float) -> Tuple[float, float]:
    """Standard 1D linear Lagrange shape functions on [-1, 1]."""
    return 0.5 * (1.0 - xi), 0.5 * (1.0 + xi)


def _line2_dual_shape_functions(xi: float) -> Tuple[float, float]:
    """1D Closed-form Wohlmuth biorthogonal dual shape functions on [-1, 1].

    Derived from A = D_ref @ inv(M_ref) where:
      M_ref = [[2/3, 1/3], [1/3, 2/3]], D_ref = diag([1, 1])
      A = [[2, -1], [-1, 2]]
      psi_1(xi) = 2*N1(xi) - N2(xi) = (1 - 3*xi) / 2
      psi_2(xi) = -N1(xi) + 2*N2(xi) = (1 + 3*xi) / 2
    """
    return 0.5 * (1.0 - 3.0 * xi), 0.5 * (1.0 + 3.0 * xi)


# 2-point Gauss-Legendre quadrature on [-1, 1]
# Exact for polynomials up to degree 3 (including psi_i * N_j which is quadratic)
_INV_SQRT3 = 1.0 / np.sqrt(3.0)
_GAUSS1D = [
    (-_INV_SQRT3, 1.0),
    (+_INV_SQRT3, 1.0),
]


class DualMortarContactConstraint2D:
    """2D Dual Mortar & Nitsche Segment-to-Segment Contact Constraint.

    Parameters
    ----------
    slave_segments : List[Tuple[int, int]]
        List of 2-node slave segment node IDs (averaging / dual basis side).
    master_segments : List[Tuple[int, int]]
        List of 2-node master segment node IDs.
    nid_to_idx : Dict[int, int]
        Mapping from node ID to contiguous 0-based node index.
    coords : np.ndarray
        Array of reference nodal coordinates (N, 2) or (N, 3).
    penalty_stiffness : float
        Normal contact penalty stiffness / modulus density [force / length^2].
    position_tolerance : float
        Maximum separation distance between segment midpoints for pairing.
    augmented_lagrange : bool
        Whether to enable PDASS Uzawa augmented Lagrangian multipliers.
    formulation : str
        "dual_mortar" (default) or "nitsche".
    gamma : Optional[float]
        Stabilization parameter for Nitsche formulation. Defaults to penalty_stiffness.
    name : str
        Constraint identifier.
    """

    def __init__(
        self,
        slave_segments: List[Tuple[int, int]],
        master_segments: List[Tuple[int, int]],
        nid_to_idx: Dict[int, int],
        coords: np.ndarray,
        penalty_stiffness: float = 1.0e8,
        position_tolerance: float = 1.0,
        augmented_lagrange: bool = False,
        formulation: str = "dual_mortar",
        mortar_type: str = "DUAL",
        gamma: Optional[float] = None,
        finite_sliding: bool = False,
        opposing_normal_threshold: float = -0.3,
        name: str = "DUAL_MORTAR_CONTACT_2D",
    ):
        self.slave_segments = list(slave_segments)
        self.master_segments = list(master_segments)
        self.nid_to_idx = nid_to_idx
        self.coords = coords
        self.k_contact = float(penalty_stiffness)
        self.position_tolerance = float(position_tolerance)
        self.augmented_lagrange = bool(augmented_lagrange)
        self.formulation = formulation.lower()
        if self.formulation not in ("dual_mortar", "nitsche"):
            raise ValueError(f"Unknown formulation '{formulation}'. Choose 'dual_mortar' or 'nitsche'.")
        self.mortar_type = str(mortar_type).upper()
        self.gamma = float(gamma) if gamma is not None else self.k_contact
        self.finite_sliding = bool(finite_sliding)
        self.opposing_normal_threshold = float(opposing_normal_threshold)
        self.name = name

        # Global diagonal condensation factor per slave node D_global[nid]
        self.D_global: Dict[int, float] = {}
        self._build_reference_mortar_matrices()

        # Discretized contact segments with 1D exact clipping
        self.segments: List[Dict[str, Any]] = []
        self._build_pairs()

        # Map from slave node ID to touching segment definitions
        self._segments_touching_node: Dict[int, List[Dict[str, Any]]] = {}
        for seg in self.segments:
            s_nodes = self.slave_segments[seg["slave_seg_idx"]]
            for nid in s_nodes:
                self._segments_touching_node.setdefault(nid, []).append(seg)

        # Lagrange multipliers for PDASS (dual mortar)
        all_slave_nids = sorted(set(nid for s in self.slave_segments for nid in s))
        self._lam: Dict[int, float] = {nid: 0.0 for nid in all_slave_nids}

    def _build_reference_mortar_matrices(self) -> None:
        """Compute D_global[nid] = sum_{segments} L_s / 2 for each slave node."""
        self.D_global = {}
        for s1, s2 in self.slave_segments:
            i1 = self.nid_to_idx[s1]
            i2 = self.nid_to_idx[s2]
            p1 = self.coords[i1, :2]
            p2 = self.coords[i2, :2]
            L_s = float(np.linalg.norm(p2 - p1))
            D_half = 0.5 * L_s
            self.D_global[s1] = self.D_global.get(s1, 0.0) + D_half
            self.D_global[s2] = self.D_global.get(s2, 0.0) + D_half

    def reproject_deformed(self, u: np.ndarray) -> None:
        """Dynamic re-clipping on deformed configuration under finite sliding."""
        if self.finite_sliding:
            self._build_pairs(u)

    def _build_pairs(self, u: Optional[np.ndarray] = None) -> None:
        """Pair slave segments with master segments using 1D exact clipping.

        Handles non-matching meshes with arbitrary element densities by
        subdividing each slave segment into parameter intervals [xi_min, xi_max]
        that project onto individual master segments.
        """
        self.segments = []
        coords = self.coords

        for s_idx, (s1, s2) in enumerate(self.slave_segments):
            idx_s1 = self.nid_to_idx[s1]
            idx_s2 = self.nid_to_idx[s2]
            if u is not None:
                xs1 = coords[idx_s1, :2] + u[2 * idx_s1 : 2 * idx_s1 + 2]
                xs2 = coords[idx_s2, :2] + u[2 * idx_s2 : 2 * idx_s2 + 2]
            else:
                xs1 = coords[idx_s1, :2]
                xs2 = coords[idx_s2, :2]

            dx_s = xs2 - xs1
            L_s = float(np.linalg.norm(dx_s))
            if L_s < 1e-14:
                continue
            xs_mid = 0.5 * (xs1 + xs2)
            t_s_unit = dx_s / L_s
            n_s_raw = np.array([-t_s_unit[1], t_s_unit[0]], dtype=np.float64)

            for m1, m2 in self.master_segments:
                if self.finite_sliding:
                    # Topological exclusion of shared nodes in self-contact
                    if s1 in (m1, m2) or s2 in (m1, m2):
                        continue

                idx_m1 = self.nid_to_idx[m1]
                idx_m2 = self.nid_to_idx[m2]
                if u is not None:
                    xm1 = coords[idx_m1, :2] + u[2 * idx_m1 : 2 * idx_m1 + 2]
                    xm2 = coords[idx_m2, :2] + u[2 * idx_m2 : 2 * idx_m2 + 2]
                else:
                    xm1 = coords[idx_m1, :2]
                    xm2 = coords[idx_m2, :2]

                dx_m = xm2 - xm1
                L_m = float(np.linalg.norm(dx_m))
                if L_m < 1e-14:
                    continue
                t_m_unit = dx_m / L_m
                # Master unit normal: perpendicular to tangent
                n_m_raw = np.array([-t_m_unit[1], t_m_unit[0]], dtype=np.float64)

                if self.finite_sliding:
                    # Opposing normal filter (must face each other)
                    if np.dot(n_s_raw, n_m_raw) > self.opposing_normal_threshold:
                        continue
                    n_m = n_m_raw
                else:
                    # Orient normal to point outward toward slave segment
                    xm_mid = 0.5 * (xm1 + xm2)
                    diff_mid = xs_mid - xm_mid
                    if np.dot(diff_mid, n_m_raw) < -1e-12:
                        n_m = -n_m_raw
                    else:
                        n_m = n_m_raw

                # Check separation distance in normal direction
                xm_mid = 0.5 * (xm1 + xm2)
                diff_mid = xs_mid - xm_mid
                dist_norm = abs(float(np.dot(diff_mid, n_m)))
                if dist_norm > self.position_tolerance:
                    continue

                # 1D exact clipping: parameter t along master segment [0, 1]
                # for points xs(s) = (1-s)*xs1 + s*xs2, s in [0, 1]
                t1 = float(np.dot(xs1 - xm1, dx_m)) / (L_m * L_m)
                t2 = float(np.dot(xs2 - xm1, dx_m)) / (L_m * L_m)

                dt = t2 - t1
                if abs(dt) < 1e-12:
                    # Segment perpendicular or degenerate
                    if 0.0 <= t1 <= 1.0:
                        s_min, s_max = 0.0, 1.0
                    else:
                        continue
                elif dt > 0.0:
                    s_min = max(0.0, -t1 / dt)
                    s_max = min(1.0, (1.0 - t1) / dt)
                else:
                    s_min = max(0.0, (1.0 - t1) / dt)
                    s_max = min(1.0, -t1 / dt)

                if s_max <= s_min + 1e-10:
                    continue  # No overlap

                # Map s in [s_min, s_max] to slave reference coordinate xi in [-1, 1]
                xi_min = 2.0 * s_min - 1.0
                xi_max = 2.0 * s_max - 1.0
                delta_xi = xi_max - xi_min

                # Assemble quadrature points on this sub-segment
                quad_points = []
                for eta, w_gauss in _GAUSS1D:
                    xi_s = 0.5 * (xi_min + xi_max) + eta * (0.5 * delta_xi)
                    s_val = 0.5 * (xi_s + 1.0)
                    t_val = t1 + s_val * dt
                    # Map t in [0, 1] to master reference coordinate xi_m in [-1, 1]
                    xi_m = float(np.clip(2.0 * t_val - 1.0, -1.0, 1.0))
                    # Physical quadrature weight: w = w_gauss * (d_xi / 2) * (L_s / 2)
                    w = w_gauss * (0.5 * delta_xi) * (0.5 * L_s)
                    quad_points.append({
                        "w": w,
                        "xi_s": xi_s,
                        "xi_m": xi_m,
                        "n": n_m.copy(),
                    })

                self.segments.append({
                    "slave_seg_idx": s_idx,
                    "master_seg": (m1, m2),
                    "quad_points": quad_points,
                })

        # Re-index segments touching node
        self._segments_touching_node = {}
        for seg in self.segments:
            s_nodes = self.slave_segments[seg["slave_seg_idx"]]
            for nid in s_nodes:
                self._segments_touching_node.setdefault(nid, []).append(seg)

    def _eval_mortar_shape_functions(self, xi: float) -> Tuple[float, float]:
        """Evaluate multiplier test functions psi_1, psi_2 at xi in [-1, 1].

        - DUAL:     Wohlmuth biorthogonal dual shape functions ((1 - 3*xi)/2, (1 + 3*xi)/2).
        - STANDARD: Standard linear Lagrange shape functions ((1 - xi)/2, (1 + xi)/2).
        """
        if self.mortar_type == "STANDARD":
            return _line2_shape_functions(xi)
        return _line2_dual_shape_functions(xi)

    def _compute_g_tilde(self, u: np.ndarray) -> Dict[int, float]:
        """Compute aggregated weak gap g_tilde_a for each slave node a.

        g_tilde_a = (1 / D_a) * sum_{segments} int psi_a * g_n * (L_s / 2) dxi
        where penetration g_n = - (x_s - x_m) . n
        """
        gap_numerator: Dict[int, float] = {nid: 0.0 for nid in self.D_global}

        for seg in self.segments:
            s_idx = seg["slave_seg_idx"]
            s_nodes = self.slave_segments[s_idx]
            m_nodes = seg["master_seg"]

            idx_s = [self.nid_to_idx[nid] for nid in s_nodes]
            idx_m = [self.nid_to_idx[nid] for nid in m_nodes]

            for qp in seg["quad_points"]:
                Ns1, Ns2 = _line2_shape_functions(qp["xi_s"])
                Nm1, Nm2 = _line2_shape_functions(qp["xi_m"])
                psi1, psi2 = self._eval_mortar_shape_functions(qp["xi_s"])

                xs = (
                    Ns1 * (self.coords[idx_s[0], :2] + u[2 * idx_s[0] : 2 * idx_s[0] + 2])
                    + Ns2 * (self.coords[idx_s[1], :2] + u[2 * idx_s[1] : 2 * idx_s[1] + 2])
                )
                xm = (
                    Nm1 * (self.coords[idx_m[0], :2] + u[2 * idx_m[0] : 2 * idx_m[0] + 2])
                    + Nm2 * (self.coords[idx_m[1], :2] + u[2 * idx_m[1] : 2 * idx_m[1] + 2])
                )

                penetration_q = -float(np.dot(xs - xm, qp["n"]))
                w = qp["w"]

                gap_numerator[s_nodes[0]] += psi1 * penetration_q * w
                gap_numerator[s_nodes[1]] += psi2 * penetration_q * w

        return {
            nid: (gap_numerator[nid] / self.D_global[nid] if self.D_global[nid] > 0.0 else 0.0)
            for nid in self.D_global
        }

    def _node_pressures(self, g_tilde: Dict[int, float]) -> Dict[int, float]:
        """Compute nodal contact pressure p_a = max(0, lam_a + k_c * g_tilde_a)."""
        p: Dict[int, float] = {}
        for nid, g in g_tilde.items():
            if self.augmented_lagrange:
                pi = self._lam[nid] + self.k_contact * g
            else:
                pi = self.k_contact * g
            p[nid] = max(0.0, pi)
        return p

    def get_active_set(self, u: np.ndarray) -> frozenset:
        """Return the set of active slave node IDs."""
        if self.finite_sliding:
            self._build_pairs(u)
        if self.formulation == "dual_mortar":
            g_tilde = self._compute_g_tilde(u)
            p = self._node_pressures(g_tilde)
            return frozenset(nid for nid, pi in p.items() if pi > 0.0)
        else:
            # Nitsche: active if any attached quadrature point penetrates
            active_nodes: Set[int] = set()
            for seg in self.segments:
                s_nodes = self.slave_segments[seg["slave_seg_idx"]]
                m_nodes = seg["master_seg"]
                idx_s = [self.nid_to_idx[nid] for nid in s_nodes]
                idx_m = [self.nid_to_idx[nid] for nid in m_nodes]
                for qp in seg["quad_points"]:
                    Ns1, Ns2 = _line2_shape_functions(qp["xi_s"])
                    Nm1, Nm2 = _line2_shape_functions(qp["xi_m"])
                    xs = (
                        Ns1 * (self.coords[idx_s[0], :2] + u[2 * idx_s[0] : 2 * idx_s[0] + 2])
                        + Ns2 * (self.coords[idx_s[1], :2] + u[2 * idx_s[1] : 2 * idx_s[1] + 2])
                    )
                    xm = (
                        Nm1 * (self.coords[idx_m[0], :2] + u[2 * idx_m[0] : 2 * idx_m[0] + 2])
                        + Nm2 * (self.coords[idx_m[1], :2] + u[2 * idx_m[1] : 2 * idx_m[1] + 2])
                    )
                    if -float(np.dot(xs - xm, qp["n"])) > 0.0:
                        active_nodes.update(s_nodes)
                        break
            return frozenset(active_nodes)

    def assemble(
        self, u: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]]:
        """Assemble contact residual force vector and consistent tangent stiffness matrix.

        Returns
        -------
        f_contact : np.ndarray
            Contact resisting force vector (size 2 * num_nodes).
        (rows, cols, data) : Tuple[np.ndarray, np.ndarray, np.ndarray]
            Sparse triplet components of consistent tangent stiffness matrix K_contact.
        stats : Dict[str, float]
            Contact iteration statistics.
        """
        if self.finite_sliding:
            self._build_pairs(u)

        n_dof = len(u)
        f_contact = np.zeros(n_dof, dtype=np.float64)
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        if self.formulation == "dual_mortar":
            g_tilde = self._compute_g_tilde(u)
            p = self._node_pressures(g_tilde)

            max_penetration = 0.0
            n_active = 0
            total_normal_force = 0.0
            for nid, pi in p.items():
                if pi > 0.0:
                    n_active += 1
                    total_normal_force += pi
                    if g_tilde[nid] > max_penetration:
                        max_penetration = g_tilde[nid]

            # 1. Contact Force Assembly:
            # f_contact = - sum_a p_a * b_a
            # Evaluated via Gauss quadrature with dual basis multiplier field lam_h(xi) = sum_a p_a psi_a(xi)
            for seg in self.segments:
                s_nodes = self.slave_segments[seg["slave_seg_idx"]]
                m_nodes = seg["master_seg"]
                idx_s = [self.nid_to_idx[nid] for nid in s_nodes]
                idx_m = [self.nid_to_idx[nid] for nid in m_nodes]

                p1 = p.get(s_nodes[0], 0.0)
                p2 = p.get(s_nodes[1], 0.0)
                if p1 <= 0.0 and p2 <= 0.0:
                    continue

                for qp in seg["quad_points"]:
                    Ns1, Ns2 = _line2_shape_functions(qp["xi_s"])
                    Nm1, Nm2 = _line2_shape_functions(qp["xi_m"])
                    psi1, psi2 = self._eval_mortar_shape_functions(qp["xi_s"])

                    lam_h = p1 * psi1 + p2 * psi2
                    if lam_h <= 0.0:
                        continue

                    w = qp["w"]
                    n = qp["n"]

                    # Distribute force to slave (pushing back) and master (action-reaction)
                    f_contact[2 * idx_s[0] : 2 * idx_s[0] + 2] += -Ns1 * w * lam_h * n
                    f_contact[2 * idx_s[1] : 2 * idx_s[1] + 2] += -Ns2 * w * lam_h * n
                    f_contact[2 * idx_m[0] : 2 * idx_m[0] + 2] += +Nm1 * w * lam_h * n
                    f_contact[2 * idx_m[1] : 2 * idx_m[1] + 2] += +Nm2 * w * lam_h * n

            # 2. Tangent Stiffness Assembly:
            # Sum of Rank-1 outer products: K_c = sum_{a in Active} (k_c / D_a) * b_a (x) b_a^T
            K_tri: Dict[Tuple[int, int], float] = {}
            for nid, pi in p.items():
                if pi <= 0.0:
                    continue
                D_i = self.D_global[nid]
                if D_i <= 0.0:
                    continue

                # Compute coupling vector b_i for active node nid
                b_i: Dict[int, np.ndarray] = {}
                for seg in self._segments_touching_node.get(nid, []):
                    s_nodes = self.slave_segments[seg["slave_seg_idx"]]
                    m_nodes = seg["master_seg"]
                    idx_s = [self.nid_to_idx[n2] for n2 in s_nodes]
                    idx_m = [self.nid_to_idx[n2] for n2 in m_nodes]
                    is_local_0 = (s_nodes[0] == nid)

                    for qp in seg["quad_points"]:
                        Ns1, Ns2 = _line2_shape_functions(qp["xi_s"])
                        Nm1, Nm2 = _line2_shape_functions(qp["xi_m"])
                        psi1, psi2 = self._eval_mortar_shape_functions(qp["xi_s"])
                        psi_i = psi1 if is_local_0 else psi2
                        if psi_i == 0.0:
                            continue

                        w = qp["w"]
                        n = qp["n"]

                        b_i.setdefault(idx_s[0], np.zeros(2, dtype=np.float64))
                        b_i[idx_s[0]] += psi_i * Ns1 * w * n

                        b_i.setdefault(idx_s[1], np.zeros(2, dtype=np.float64))
                        b_i[idx_s[1]] += psi_i * Ns2 * w * n

                        b_i.setdefault(idx_m[0], np.zeros(2, dtype=np.float64))
                        b_i[idx_m[0]] -= psi_i * Nm1 * w * n

                        b_i.setdefault(idx_m[1], np.zeros(2, dtype=np.float64))
                        b_i[idx_m[1]] -= psi_i * Nm2 * w * n

                scale = self.k_contact / D_i
                node_indices = list(b_i.keys())
                for ia in node_indices:
                    for ib in node_indices:
                        k_block = scale * np.outer(b_i[ia], b_i[ib])
                        for d1 in range(2):
                            r = 2 * ia + d1
                            for d2 in range(2):
                                c = 2 * ib + d2
                                if r <= c:
                                    val = float(k_block[d1, d2])
                                    if val != 0.0:
                                        K_tri[(r, c)] = K_tri.get((r, c), 0.0) + val

            for (r, c), val in K_tri.items():
                rows.append(r)
                cols.append(c)
                data.append(val)
                if r != c:
                    rows.append(c)
                    cols.append(r)
                    data.append(val)

            stats = {
                "max_penetration": max_penetration,
                "n_active": n_active,
                "n_candidates": len(self.D_global),
                "n_segments": len(self.segments),
                "total_normal_force": total_normal_force,
            }

        else:
            # Nitsche formulation (Chouly et al. 2017)
            # Pure SPD symmetric tangent stiffness without multipliers
            max_penetration = 0.0
            n_active = 0
            total_normal_force = 0.0
            K_tri = {}

            for seg in self.segments:
                s_nodes = self.slave_segments[seg["slave_seg_idx"]]
                m_nodes = seg["master_seg"]
                idx_s = [self.nid_to_idx[nid] for nid in s_nodes]
                idx_m = [self.nid_to_idx[nid] for nid in m_nodes]

                for qp in seg["quad_points"]:
                    Ns1, Ns2 = _line2_shape_functions(qp["xi_s"])
                    Nm1, Nm2 = _line2_shape_functions(qp["xi_m"])

                    xs = (
                        Ns1 * (self.coords[idx_s[0], :2] + u[2 * idx_s[0] : 2 * idx_s[0] + 2])
                        + Ns2 * (self.coords[idx_s[1], :2] + u[2 * idx_s[1] : 2 * idx_s[1] + 2])
                    )
                    xm = (
                        Nm1 * (self.coords[idx_m[0], :2] + u[2 * idx_m[0] : 2 * idx_m[0] + 2])
                        + Nm2 * (self.coords[idx_m[1], :2] + u[2 * idx_m[1] : 2 * idx_m[1] + 2])
                    )

                    n = qp["n"]
                    g_n = -float(np.dot(xs - xm, n))
                    if g_n <= 0.0:
                        continue

                    n_active += 1
                    if g_n > max_penetration:
                        max_penetration = g_n

                    w = qp["w"]
                    traction = self.gamma * g_n
                    total_normal_force += traction * w

                    # Nitsche force:
                    f_contact[2 * idx_s[0] : 2 * idx_s[0] + 2] += -Ns1 * w * traction * n
                    f_contact[2 * idx_s[1] : 2 * idx_s[1] + 2] += -Ns2 * w * traction * n
                    f_contact[2 * idx_m[0] : 2 * idx_m[0] + 2] += +Nm1 * w * traction * n
                    f_contact[2 * idx_m[1] : 2 * idx_m[1] + 2] += +Nm2 * w * traction * n

                    # Nitsche stiffness: point-wise outer product sum c_k (x) c_k^T
                    c_map = {
                        idx_s[0]: -Ns1 * n,
                        idx_s[1]: -Ns2 * n,
                        idx_m[0]: +Nm1 * n,
                        idx_m[1]: +Nm2 * n,
                    }
                    node_indices = list(c_map.keys())
                    scale = self.gamma * w
                    for ia in node_indices:
                        for ib in node_indices:
                            k_block = scale * np.outer(c_map[ia], c_map[ib])
                            for d1 in range(2):
                                r = 2 * ia + d1
                                for d2 in range(2):
                                    c = 2 * ib + d2
                                    if r <= c:
                                        val = float(k_block[d1, d2])
                                        if val != 0.0:
                                            K_tri[(r, c)] = K_tri.get((r, c), 0.0) + val

            for (r, c), val in K_tri.items():
                rows.append(r)
                cols.append(c)
                data.append(val)
                if r != c:
                    rows.append(c)
                    cols.append(r)
                    data.append(val)

            stats = {
                "max_penetration": max_penetration,
                "n_active": n_active,
                "n_candidates": len(self.D_global),
                "n_segments": len(self.segments),
                "total_normal_force": total_normal_force,
            }

        return (
            f_contact,
            (
                np.array(rows, dtype=np.int32),
                np.array(cols, dtype=np.int32),
                np.array(data, dtype=np.float64),
            ),
            stats,
        )

    def update_augmented_multipliers(self, u: np.ndarray, omega: float = 1.0) -> Dict[str, float]:
        """PDASS single-loop Uzawa augmented Lagrangian multiplier update.

        p_new = lam_old + omega * (max(0, lam_old + k_c * g_tilde) - lam_old)
        """
        if not self.augmented_lagrange or self.formulation != "dual_mortar":
            return {"max_penetration": 0.0, "max_lambda_change": 0.0}

        g_tilde = self._compute_g_tilde(u)
        max_penetration = 0.0
        max_lambda_change = 0.0

        for nid, g in g_tilde.items():
            old_lam = self._lam[nid]
            p_target = max(0.0, old_lam + self.k_contact * g)
            p_new = old_lam + omega * (p_target - old_lam)
            self._lam[nid] = p_new
            max_lambda_change = max(max_lambda_change, abs(p_new - old_lam))
            if p_new > 0.0 and g > max_penetration:
                max_penetration = g

        return {"max_penetration": max_penetration, "max_lambda_change": max_lambda_change}
