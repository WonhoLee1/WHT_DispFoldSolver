"""
dual_mortar_clipping3d.py
=========================
Stage B Sutherland-Hodgman Polygon Clipping and Dual Mortar integration
for non-matching (dissimilar mesh density) 3D contact surfaces.

Theoretical Foundations:
1. Sutherland-Hodgman Planar Polygon Clipping:
   - Projects master Quad4 face onto slave Quad4 plane.
   - Successively clips the candidate polygon against the 4 half-spaces defined by slave edges.
   - Produces convex intersection polygons (triangles, quads, pentagons, hexagons).
2. Triangular Sub-triangulation & Gauss Quadrature:
   - Decomposes intersection polygon into triangles via centroid fan triangulation.
   - Evaluates 3-point or 7-point symmetric Gauss quadrature per sub-triangle.
3. Wohlmuth Bi-Orthogonal Dual Basis (Psi_i):
   - Local diagonal metric condensation D_f = diag(M_f 1), A_f = D_f M_f^-1.
   - Exact satisfaction of the contact patch test on non-conforming, non-matching meshes.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from dispsolver.constraint3d.surface_contact3d_s2s import (
    _quad4_shape_functions,
    _quad4_tangents,
    _quad4_normal,
    _quad4_area_jacobian,
    _GAUSS2X2,
)


def _intersect_line_segments_2d(p1: np.ndarray, p2: np.ndarray, cp1: np.ndarray, cp2: np.ndarray) -> np.ndarray:
    """Compute 2D intersection point between segment (p1, p2) and infinite line (cp1, cp2)."""
    dc = cp2 - cp1
    dp = p2 - p1
    denom = dc[0] * dp[1] - dc[1] * dp[0]
    if abs(denom) < 1e-14:
        return 0.5 * (p1 + p2)
    t = ((p1[0] - cp1[0]) * dp[1] - (p1[1] - cp1[1]) * dp[0]) / denom
    return cp1 + t * dc


def clip_polygon_2d(subject_poly: List[np.ndarray], clip_poly: List[np.ndarray]) -> List[np.ndarray]:
    """Sutherland-Hodgman 2D convex polygon clipping.

    Parameters
    ----------
    subject_poly : list of (2,) np.ndarray
        Vertices of the polygon to be clipped (master face projected into (xi, eta) of slave).
    clip_poly : list of (2,) np.ndarray
        Convex boundary polygon (typically slave face square [-1, 1]^2).

    Returns
    -------
    list of (2,) np.ndarray
        Clipped intersection polygon vertices in counter-clockwise order.
    """
    output_list = list(subject_poly)

    for i in range(len(clip_poly)):
        cp1 = clip_poly[i]
        cp2 = clip_poly[(i + 1) % len(clip_poly)]
        input_list = output_list
        output_list = []
        if not input_list:
            break

        s = input_list[-1]
        edge_vec = cp2 - cp1
        # Outward normal to edge: (edge_vec[1], -edge_vec[0]) for CCW clip_poly
        # Inside test: (p - cp1) x edge_vec <= 0 (points to the left/inside of CCW edge)
        def _is_inside(pt: np.ndarray) -> bool:
            return (edge_vec[0] * (pt[1] - cp1[1]) - edge_vec[1] * (pt[0] - cp1[0])) >= -1e-12

        for p in input_list:
            if _is_inside(p):
                if not _is_inside(s):
                    inter = _intersect_line_segments_2d(s, p, cp1, cp2)
                    output_list.append(inter)
                output_list.append(p)
            elif _is_inside(s):
                inter = _intersect_line_segments_2d(s, p, cp1, cp2)
                output_list.append(inter)
            s = p

    return output_list


def triangulate_convex_polygon(poly_pts: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Fan triangulate convex 2D polygon about its centroid."""
    if len(poly_pts) < 3:
        return []
    if len(poly_pts) == 3:
        return [(poly_pts[0], poly_pts[1], poly_pts[2])]

    centroid = np.mean(poly_pts, axis=0)
    triangles = []
    n = len(poly_pts)
    for i in range(n):
        triangles.append((centroid, poly_pts[i], poly_pts[(i + 1) % n]))
    return triangles


# Standard 3-point Dunavant Gauss quadrature on reference triangle (u, v, w = 1 - u - v)
_TRI_GAUSS3 = [
    (1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0),
    (2.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0),
    (1.0 / 6.0, 2.0 / 3.0, 1.0 / 3.0),
]


class DualMortarClippedContactConstraint3D:
    """Stage B Dual-Mortar Surface-to-Surface Contact with Sutherland-Hodgman Polygon Clipping.

    Supports arbitrary non-matching (coarse-on-fine, non-conforming) meshes
    with exact patch test consistency and symmetric diagonal static condensation.
    """

    def __init__(
        self,
        slave_faces: List[Tuple[int, int, int, int]],
        master_faces: List[Tuple[int, int, int, int]],
        nid_to_idx: Dict[int, int],
        coords: np.ndarray,
        penalty_stiffness: float = 1.0e8,
        position_tolerance: float = 2.0,
        augmented_lagrange: bool = False,
        mortar_type: str = "DUAL",
        name: str = "DUAL_MORTAR_CLIPPED_3D",
    ):
        self.slave_faces = list(slave_faces)
        self.master_faces = list(master_faces)
        self.nid_to_idx = nid_to_idx
        self.coords = coords
        self.k_contact = float(penalty_stiffness)
        self.position_tolerance = float(position_tolerance)
        self.augmented_lagrange = bool(augmented_lagrange)
        self.mortar_type = str(mortar_type).upper()
        self.name = name

        # 1. Per-slave-face mortar basis transformation A_f and metric D_f
        #    - DUAL:     Wholmuth bi-orthogonal dual basis: A_f = diag(D_f) @ inv(M_f)
        #    - STANDARD: Standard primal shape functions:   A_f = I_{4x4}
        self._A_f: Dict[int, np.ndarray] = {}
        self.D_global: Dict[int, float] = {}
        self._build_mortar_basis()

        # 2. Polygon clipped segments
        self.segments: List[Dict[str, Any]] = []
        self._build_clipped_pairs()

        # 3. Segments touching slave node index
        self._segments_touching_node: Dict[int, List[Dict[str, Any]]] = {}
        for seg in self.segments:
            face = self.slave_faces[seg["slave_face_idx"]]
            for nid in face:
                self._segments_touching_node.setdefault(nid, []).append(seg)

        all_slave_nids = sorted(set(nid for face in self.slave_faces for nid in face))
        self._lam: Dict[int, float] = {nid: 0.0 for nid in all_slave_nids}

    def _build_mortar_basis(self) -> None:
        self.D_global = {}
        for f_idx, face in enumerate(self.slave_faces):
            idx = [self.nid_to_idx[nid] for nid in face]
            q = [self.coords[i] for i in idx]
            M_f = np.zeros((4, 4), dtype=np.float64)
            for xi, eta, w_gauss in _GAUSS2X2:
                N = np.array(_quad4_shape_functions(xi, eta))
                dA = _quad4_area_jacobian(q[0], q[1], q[2], q[3], xi, eta) * w_gauss
                M_f += np.outer(N, N) * dA
            D_f = M_f.sum(axis=1)
            if self.mortar_type == "STANDARD":
                # Standard Mortar: Lagrange multipliers interpolated with standard shape functions Psi_a = N_a
                A_f = np.eye(4, dtype=np.float64)
            else:
                # Wohlmuth (2000) Bi-orthogonal Dual Mortar: int Psi_a N_b dA = D_a delta_{ab}
                A_f = np.diag(D_f) @ np.linalg.inv(M_f)
            self._A_f[f_idx] = A_f
            for a, nid in enumerate(face):
                self.D_global[nid] = self.D_global.get(nid, 0.0) + D_f[a]

    def _build_clipped_pairs(self) -> None:
        from dispsolver.constraint3d.surface_tie3d import _project_point_to_quad4
        self.segments = []

        slave_box_2d = [
            np.array([-1.0, -1.0]),
            np.array([ 1.0, -1.0]),
            np.array([ 1.0,  1.0]),
            np.array([-1.0,  1.0]),
        ]

        for sf_idx, s_face in enumerate(self.slave_faces):
            s_idx = [self.nid_to_idx[nid] for nid in s_face]
            sq = [self.coords[i] for i in s_idx]
            s_centroid = 0.25 * (sq[0] + sq[1] + sq[2] + sq[3])

            for mf_idx, m_face in enumerate(self.master_faces):
                m_idx = [self.nid_to_idx[nid] for nid in m_face]
                mq = [self.coords[i] for i in m_idx]

                # Project 4 master vertices into slave face's (xi, eta) parameter space
                proj_master_2d: List[np.ndarray] = []
                for k in range(4):
                    xi_s, eta_s, dist, _ = _project_point_to_quad4(mq[k], *sq)
                    proj_master_2d.append(np.array([xi_s, eta_s]))

                # Sutherland-Hodgman clipping against slave [-1, 1]^2
                clipped_poly = clip_polygon_2d(proj_master_2d, slave_box_2d)
                if len(clipped_poly) < 3:
                    continue

                # Determine master face normal sign so that initial gap0 >= 0
                n_c_raw = _quad4_normal(mq[0], mq[1], mq[2], mq[3], 0.0, 0.0)
                m_centroid = 0.25 * (mq[0] + mq[1] + mq[2] + mq[3])
                gap0 = float(np.dot(s_centroid - m_centroid, n_c_raw))
                sign = 1.0 if gap0 >= 0.0 else -1.0

                # Triangulate intersection polygon
                sub_triangles = triangulate_convex_polygon(clipped_poly)
                quad_points = []

                for t0, t1, t2 in sub_triangles:
                    # 2D triangle area in (xi, eta) space
                    tri_area = 0.5 * abs((t1[0] - t0[0]) * (t2[1] - t0[1]) - (t1[1] - t0[1]) * (t2[0] - t0[0]))
                    if tri_area < 1e-12:
                        continue

                    for u_g, v_g, w_g in _TRI_GAUSS3:
                        w_bary = 1.0 - u_g - v_g
                        xi_s = w_bary * t0[0] + u_g * t1[0] + v_g * t2[0]
                        eta_s = w_bary * t0[1] + u_g * t1[1] + v_g * t2[1]

                        Ns = _quad4_shape_functions(xi_s, eta_s)
                        xs_qp = sum(Ns[k] * sq[k] for k in range(4))

                        # Project xs_qp to master face
                        xi_m, eta_m, dist, _ = _project_point_to_quad4(xs_qp, *mq)
                        area_jac = _quad4_area_jacobian(sq[0], sq[1], sq[2], sq[3], xi_s, eta_s)
                        w = tri_area * w_g * area_jac
                        n_qp = sign * _quad4_normal(mq[0], mq[1], mq[2], mq[3], xi_m, eta_m)

                        quad_points.append({
                            "w": w,
                            "xi_s": xi_s, "eta_s": eta_s,
                            "xi_m": xi_m, "eta_m": eta_m,
                            "n": n_qp,
                        })

                if quad_points:
                    self.segments.append({
                        "slave_face_idx": sf_idx,
                        "master_face": m_face,
                        "quad_points": quad_points,
                    })

    def bind_solver(self, solver: Any, dt: float = 1.0) -> None:
        """Small sliding: the contact plane rotates with the master surface
        under NLGEOM (Abaqus); without NLGEOM it stays in the reference
        configuration. This class used the reference normal in both cases,
        so a large rigid rotation of the contact pair was not objective."""
        self.mode = "small" if bool(getattr(solver, "nlgeom", True)) else "linear"

    def _qp_normal(self, u: np.ndarray, idx_m: List[int], qp: Dict[str, Any]) -> np.ndarray:
        """Contact normal at a quadrature point: the stored reference normal
        ("linear"), or the current master normal with the same orientation
        ("small"). The tangent keeps the reference-normal structure (the
        normal's rotation derivative is not included) -- the residual is
        exact, convergence under large rotation is not quadratic."""
        if getattr(self, "mode", "linear") != "small":
            return qp["n"]
        q = [self.coords[i] + u[3 * i: 3 * i + 3] for i in idx_m]
        n_raw = _quad4_normal(q[0], q[1], q[2], q[3], qp["xi_m"], qp["eta_m"])
        sgn = qp.get("_sgn")
        if sgn is None:
            q0 = [self.coords[i] for i in idx_m]
            sgn = 1.0 if float(np.dot(_quad4_normal(q0[0], q0[1], q0[2], q0[3], qp["xi_m"], qp["eta_m"]),
                                      qp["n"])) >= 0.0 else -1.0
            qp["_sgn"] = sgn
        return sgn * n_raw

    def _compute_g_tilde(self, u: np.ndarray) -> Dict[int, float]:
        gap_numerator: Dict[int, float] = {nid: 0.0 for nid in self.D_global}
        for seg in self.segments:
            f_idx = seg["slave_face_idx"]
            face = self.slave_faces[f_idx]
            m_face = seg["master_face"]
            A_f = self._A_f[f_idx]
            idx_s = [self.nid_to_idx[nid] for nid in face]
            idx_m = [self.nid_to_idx[nid] for nid in m_face]
            for qp in seg["quad_points"]:
                Ns = np.array(_quad4_shape_functions(qp["xi_s"], qp["eta_s"]))
                Nm = np.array(_quad4_shape_functions(qp["xi_m"], qp["eta_m"]))
                Psi = A_f @ Ns
                xs = sum(Ns[a] * (self.coords[idx_s[a]] + u[3 * idx_s[a]: 3 * idx_s[a] + 3]) for a in range(4))
                xm = sum(Nm[b] * (self.coords[idx_m[b]] + u[3 * idx_m[b]: 3 * idx_m[b] + 3]) for b in range(4))
                penetration_q = -float(np.dot(xs - xm, self._qp_normal(u, idx_m, qp)))
                for a in range(4):
                    gap_numerator[face[a]] += Psi[a] * penetration_q * qp["w"]

        return {
            nid: (gap_numerator[nid] / self.D_global[nid] if self.D_global[nid] != 0.0 else 0.0)
            for nid in self.D_global
        }

    def assemble(self, u: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]]:
        n_dof = len(u)
        f_contact = np.zeros(n_dof, dtype=np.float64)
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        g_tilde = self._compute_g_tilde(u)
        p = {nid: max(0.0, self._lam[nid] + self.k_contact * g) for nid, g in g_tilde.items()}

        max_penetration = 0.0
        n_active = 0
        total_normal_force = 0.0
        for nid, pi in p.items():
            if pi > 0.0:
                n_active += 1
                total_normal_force += pi
                if g_tilde[nid] > max_penetration:
                    max_penetration = g_tilde[nid]

        for seg in self.segments:
            f_idx = seg["slave_face_idx"]
            face = self.slave_faces[f_idx]
            m_face = seg["master_face"]
            A_f = self._A_f[f_idx]
            idx_s = [self.nid_to_idx[nid] for nid in face]
            idx_m = [self.nid_to_idx[nid] for nid in m_face]
            p_face = np.array([p[nid] for nid in face])
            if not np.any(p_face > 0.0):
                continue
            for qp in seg["quad_points"]:
                Ns = np.array(_quad4_shape_functions(qp["xi_s"], qp["eta_s"]))
                Nm = np.array(_quad4_shape_functions(qp["xi_m"], qp["eta_m"]))
                Psi = A_f @ Ns
                lam_h = float(np.dot(np.where(p_face > 0.0, p_face, 0.0), Psi))
                if lam_h == 0.0:
                    continue
                n_qp = self._qp_normal(u, idx_m, qp)
                w = qp["w"]
                for a in range(4):
                    f_contact[3 * idx_s[a]: 3 * idx_s[a] + 3] += -Ns[a] * w * lam_h * n_qp
                for b in range(4):
                    f_contact[3 * idx_m[b]: 3 * idx_m[b] + 3] += +Nm[b] * w * lam_h * n_qp

        for nid, pi in p.items():
            if pi <= 0.0:
                continue
            D_i = self.D_global[nid]
            if D_i == 0.0:
                continue
            b_i: Dict[int, np.ndarray] = {}
            for seg in self._segments_touching_node.get(nid, []):
                f_idx = seg["slave_face_idx"]
                face = self.slave_faces[f_idx]
                m_face = seg["master_face"]
                A_f = self._A_f[f_idx]
                idx_s = [self.nid_to_idx[n2] for n2 in face]
                idx_m = [self.nid_to_idx[n2] for n2 in m_face]
                a_local = face.index(nid)
                for qp in seg["quad_points"]:
                    Ns = np.array(_quad4_shape_functions(qp["xi_s"], qp["eta_s"]))
                    Nm = np.array(_quad4_shape_functions(qp["xi_m"], qp["eta_m"]))
                    Psi_i_qp = float(A_f[a_local, :] @ Ns)
                    if Psi_i_qp == 0.0:
                        continue
                    n_qp = self._qp_normal(u, idx_m, qp)
                    w = qp["w"]
                    for a in range(4):
                        b_i.setdefault(idx_s[a], np.zeros(3))
                        b_i[idx_s[a]] += Psi_i_qp * Ns[a] * w * n_qp
                    for b in range(4):
                        b_i.setdefault(idx_m[b], np.zeros(3))
                        b_i[idx_m[b]] -= Psi_i_qp * Nm[b] * w * n_qp

            scale = self.k_contact / D_i
            node_idxs = list(b_i.keys())
            for ia in node_idxs:
                for ib in node_idxs:
                    k_block = scale * np.outer(b_i[ia], b_i[ib])
                    for d1 in range(3):
                        for d2 in range(3):
                            val = float(k_block[d1, d2])
                            if val != 0.0:
                                rows.append(3 * ia + d1)
                                cols.append(3 * ib + d2)
                                data.append(val)

        stats = {
            "max_penetration": float(max_penetration),
            "n_active": float(n_active),
            "total_normal_force": float(total_normal_force),
            "n_clipped_segments": float(len(self.segments)),
        }
        return f_contact, (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(data, dtype=np.float64)), stats
