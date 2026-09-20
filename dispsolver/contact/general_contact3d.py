"""
general_contact3d.py
====================
Commercial-grade Abaqus-compatible General Contact (ContactStd) Engine for 3D FEA.
Supports ALL EXTERIOR automatic contact detection, part-exterior vs part-exterior contact,
and self-contact in large-deformation folding and bending.

Theoretical Foundations:
1. Dynamic Bounding Volume Hierarchy (BVH):
   - Broad-phase collision detection using Morton-code sorted 3D BVH (O(N log N)).
   - Bottom-up dynamic AABB refitting at each Newton iteration.
2. Topological Self-Contact Filtering:
   - Automatically eliminates false positive interactions between adjacent faces sharing nodes.
3. Multiplier-Free Symmetric Nitsche / Penalty Normal Contact:
   - Enforces non-penetration via symmetric dyadic tangent assembly:
     K_c = sum_{qp} gamma * w_qp * (c_qp (x) c_qp^T)
   - Preserves Positive Semi-Definiteness (SPD) and bit-level symmetry |K - K^T| = 0.0,
     ensuring high-speed PARDISO Cholesky (LLT) convergence without saddle-point instability.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set, Sequence, Union
import numpy as np

from dispsolver.contact.bvh_tree3d import DynamicBVHTree3D, AABB3D
from dispsolver.contact.contact_surface import auto_detect_exterior_3d, ContactSurface3D
from dispsolver.constraint3d.surface_tie3d import _project_point_to_quad4


# ---------------------------------------------------------------------------
# Bilinear Quad4 & Linear Tri3 Geometry Kernels
# ---------------------------------------------------------------------------

def _quad4_shape_functions(xi: float, eta: float) -> Tuple[float, float, float, float]:
    """Evaluate standard bilinear Quad4 shape functions on [-1, 1]^2."""
    N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
    return N1, N2, N3, N4


def _quad4_normal(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray, xi: float = 0.0, eta: float = 0.0) -> np.ndarray:
    """Compute outward unit normal vector of Quad4 face at (xi, eta)."""
    dN1_dxi = -0.25 * (1.0 - eta)
    dN2_dxi =  0.25 * (1.0 - eta)
    dN3_dxi =  0.25 * (1.0 + eta)
    dN4_dxi = -0.25 * (1.0 + eta)

    dN1_deta = -0.25 * (1.0 - xi)
    dN2_deta = -0.25 * (1.0 + xi)
    dN3_deta =  0.25 * (1.0 + xi)
    dN4_deta =  0.25 * (1.0 - xi)

    t_xi = dN1_dxi * q1 + dN2_dxi * q2 + dN3_dxi * q3 + dN4_dxi * q4
    t_eta = dN1_deta * q1 + dN2_deta * q2 + dN3_deta * q3 + dN4_deta * q4
    n = np.cross(t_xi, t_eta)
    norm = float(np.linalg.norm(n))
    if norm < 1e-14:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return n / norm


def _quad4_area_jacobian(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray, xi: float, eta: float) -> float:
    """Compute physical differential area dA = ||dx/dxi x dx/deta||."""
    dN1_dxi = -0.25 * (1.0 - eta)
    dN2_dxi =  0.25 * (1.0 - eta)
    dN3_dxi =  0.25 * (1.0 + eta)
    dN4_dxi = -0.25 * (1.0 + eta)

    dN1_deta = -0.25 * (1.0 - xi)
    dN2_deta = -0.25 * (1.0 + xi)
    dN3_deta =  0.25 * (1.0 + xi)
    dN4_deta =  0.25 * (1.0 - xi)

    t_xi = dN1_dxi * q1 + dN2_dxi * q2 + dN3_dxi * q3 + dN4_dxi * q4
    t_eta = dN1_deta * q1 + dN2_deta * q2 + dN3_deta * q3 + dN4_deta * q4
    return float(np.linalg.norm(np.cross(t_xi, t_eta)))


def _project_point_to_tri3(
    p: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
) -> Tuple[float, float, float, float, np.ndarray]:
    """Project 3D point p onto planar 3D Tri3 triangle (t1, t2, t3).

    Returns
    -------
    u, v, w : float
        Barycentric coordinates (p_proj = u*t1 + v*t2 + w*t3, u + v + w = 1).
    dist : float
        Distance from p to projected surface point.
    proj_pt : np.ndarray
        (3,) Projected point on triangle.
    """
    e1 = t2 - t1
    e2 = t3 - t1
    normal = np.cross(e1, e2)
    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-14:
        return 0.3333, 0.3333, 0.3334, 0.0, (t1 + t2 + t3) / 3.0
    unit_n = normal / norm_val

    # Distance to plane
    dist_plane = float(np.dot(p - t1, unit_n))
    proj_pt = p - dist_plane * unit_n

    # Barycentric coordinates via Cramer's rule
    v0 = e1
    v1 = e2
    v2 = proj_pt - t1
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-14:
        return 0.3333, 0.3333, 0.3334, float(np.linalg.norm(p - proj_pt)), proj_pt

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w

    dist = float(np.linalg.norm(p - proj_pt))
    return u, v, w, dist, proj_pt


def _tangent_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal (t1, t2) spanning the plane perpendicular to unit normal `n`."""
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    t1 = ref - float(np.dot(ref, n)) * n
    norm_t1 = float(np.linalg.norm(t1))
    if norm_t1 < 1e-14:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        t1 = ref - float(np.dot(ref, n)) * n
        norm_t1 = float(np.linalg.norm(t1))
    t1 = t1 / norm_t1
    t2 = np.cross(n, t1)
    norm_t2 = float(np.linalg.norm(t2))
    if norm_t2 > 1e-14:
        t2 = t2 / norm_t2
    return t1, t2


# 2x2 Gauss-Legendre quadrature on [-1, 1]^2
_INV_SQRT3 = 1.0 / np.sqrt(3.0)
_GAUSS2X2 = [
    (-_INV_SQRT3, -_INV_SQRT3, 1.0),
    (+_INV_SQRT3, -_INV_SQRT3, 1.0),
    (+_INV_SQRT3, +_INV_SQRT3, 1.0),
    (-_INV_SQRT3, +_INV_SQRT3, 1.0),
]


# ---------------------------------------------------------------------------
# GeneralContact3D Constraint Class
# ---------------------------------------------------------------------------

class GeneralContact3D:
    """Abaqus-compatible 3D General Contact Engine.

    Parameters
    ----------
    faces : List[Tuple[int, ...]]
        List of all participating exterior boundary face node tuples.
    coords : np.ndarray
        Reference nodal coordinates array of shape (N, 3).
    nid_to_idx : Dict[int, int]
        Mapping from global node ID to contiguous 0-based node index.
    use_all_exterior : bool, optional
        If True, enables automatic contact and self-contact across all exterior faces.
    included_pairs : Optional[List[Tuple[Any, Any]]], optional
        Explicit surface/part pairs to include. If None, all exterior pairs are eligible.
    excluded_pairs : Optional[List[Tuple[Any, Any]]], optional
        Explicit surface/part pairs to exclude.
    gamma : float, optional
        Normal contact penalty / Nitsche stabilization parameter. Defaults to 1.0e8.
    skin : float, optional
        AABB expansion skin for broad-phase candidate queries. Defaults to 0.05.
    law : Optional[Any], optional
        Pressure-overclosure law (HardLaw, LinearSoftLaw, ExponentialSoftLaw, TabularSoftLaw, AdaptivePowerSoftLaw).
    damping_coeff : Optional[float], optional
        Contact stabilization / damping coefficient.
    damping_clearance : Optional[float], optional
        Clearance distance over which damping ramps to zero.
    tangent_fraction : float, optional
        Tangential damping ratio relative to normal contact damping (Abaqus tangentFraction).
    friction_coefficient : float, optional
        Coulomb friction coefficient mu (Abaqus *FRICTION). Defaults to 0.0 (frictionless).
    tangential_stiffness_ratio : float, optional
        Tangential stick penalty stiffness ratio k_t / gamma. Defaults to 1.0.
    name : str, optional
        Identifier name for this contact interaction.
    """

    def __init__(
        self,
        faces: List[Tuple[int, ...]],
        coords: np.ndarray,
        nid_to_idx: Dict[int, int],
        use_all_exterior: bool = True,
        included_pairs: Optional[List[Tuple[Any, Any]]] = None,
        excluded_pairs: Optional[List[Tuple[Any, Any]]] = None,
        gamma: float = 1.0e8,
        skin: float = 0.05,
        law: Optional[Any] = None,
        damping_coeff: Optional[float] = None,
        damping_clearance: Optional[float] = None,
        tangent_fraction: float = 0.0,
        friction_coefficient: float = 0.0,
        tangential_stiffness_ratio: float = 1.0,
        name: str = "GeneralContact3D",
        face_to_elem: Optional[Dict[int, int]] = None,
        elem_to_pid: Optional[Dict[int, int]] = None,
        materials: Optional[Dict[int, Any]] = None,
        contact_enforcement: str = "NITSCHE",
        augmented_lagrange: bool = False,
        face_labels: Optional[Dict[int, Set[Any]]] = None,
        face_to_instance: Optional[Dict[int, Any]] = None,
        face_to_part: Optional[Dict[int, Any]] = None,
        face_to_instance_name: Optional[Dict[int, str]] = None,
        face_to_part_name: Optional[Dict[int, str]] = None,
    ):
        self.faces = faces
        self.coords = np.asarray(coords, dtype=np.float64)
        self.nid_to_idx = nid_to_idx
        self.use_all_exterior = bool(use_all_exterior)
        self.included_pairs = included_pairs
        self.excluded_pairs = excluded_pairs
        self.gamma = float(gamma)
        self.law = law
        self.damping_coeff = float(damping_coeff) if damping_coeff is not None else 0.0
        self.damping_clearance = float(damping_clearance) if damping_clearance is not None else 0.0
        self.tangent_fraction = float(tangent_fraction)
        self.mu = float(friction_coefficient)
        self.k_t = float(tangential_stiffness_ratio) * self.gamma
        self.name = str(name)
        self.face_to_elem = face_to_elem or {}
        self.elem_to_pid = elem_to_pid or {}
        self.materials = materials or {}
        self.contact_enforcement = str(contact_enforcement).upper()
        self.augmented_lagrange = bool(augmented_lagrange)
        self.face_labels = face_labels or {}
        self.face_to_instance = face_to_instance or {}
        self.face_to_part = face_to_part or {}
        self.face_to_instance_name = face_to_instance_name or {}
        self.face_to_part_name = face_to_part_name or {}

        # Precompute face Young's moduli if materials are available
        from dispsolver.constraint3d.contact_stiffness import _material_effective_E
        self._face_E: Dict[int, float] = {}
        for f_idx in range(len(self.faces)):
            eid = self.face_to_elem.get(f_idx, None)
            pid = self.elem_to_pid.get(eid, 0) if eid is not None else 0
            mat_obj = self.materials.get(pid, None)
            self._face_E[f_idx] = _material_effective_E(mat_obj, default_E=200000.0)

        # Reference modulus for scaling (median over faces, or 200000.0)
        if self._face_E:
            self.E_ref = float(np.median(list(self._face_E.values())))
        else:
            self.E_ref = 200000.0

        # Fast Numba Quad4 data structures
        self.use_numba = True
        self._all_quad4 = len(self.faces) > 0 and all(len(f) == 4 for f in self.faces)
        if self._all_quad4:
            self._faces_idx_arr = np.empty((len(self.faces), 4), dtype=np.int32)
            for fi, f in enumerate(self.faces):
                for ki in range(4):
                    self._faces_idx_arr[fi, ki] = self.nid_to_idx[f[ki]]
            self._face_E_arr = np.array([self._face_E[fi] for fi in range(len(self.faces))], dtype=np.float64)
        else:
            self._faces_idx_arr = None
            self._face_E_arr = None

        # Convected anchor tracking for Coulomb friction: (fa_idx, fb_idx, qp_idx) -> (xi_m, eta_m) or (u_t, v_t, w_t)
        self._anchors: Dict[Tuple[int, int, int], Tuple[float, ...]] = {}

        # Scale-invariant collision skin: broad-phase AABB margin must scale with mesh characteristic size
        if len(faces) > 0 and len(coords) > 0:
            edge_lens = []
            for face in faces[:min(50, len(faces))]:
                idx0 = self.nid_to_idx[face[0]]
                idx1 = self.nid_to_idx[face[1]]
                edge_lens.append(float(np.linalg.norm(self.coords[idx1] - self.coords[idx0])))
            L_char = float(np.median(edge_lens)) if edge_lens else 1.0
            auto_skin = 0.05 * L_char
            self.skin = float(skin) if (skin is not None and skin > auto_skin) else max(0.05, auto_skin)
        else:
            self.skin = float(skin) if skin is not None else 0.05

        # If damping_clearance wasn't explicitly given but damping_coeff was, default clearance to 2 * skin
        if self.damping_coeff > 0.0 and self.damping_clearance <= 0.0:
            self.damping_clearance = 2.0 * self.skin

        # Build initial BVH collision tree
        self.bvh = DynamicBVHTree3D(
            faces=self.faces,
            coords=self.coords,
            nid_to_idx=self.nid_to_idx,
            skin=max(self.skin, self.damping_clearance),
        )

        # Track active contact pairs and stats
        self.active_pairs: List[Tuple[int, int]] = []
        self.last_stats: Dict[str, float] = {}

    @classmethod
    def from_mesh(
        cls,
        mesh: Any,
        use_all_exterior: bool = True,
        gamma: float = 1.0e8,
        skin: float = 0.05,
        friction_coefficient: float = 0.0,
        tangential_stiffness_ratio: float = 1.0,
        name: str = "GeneralContact3D",
    ) -> GeneralContact3D:
        """Construct GeneralContact3D automatically by detecting all exterior faces of a Mesh3D."""
        exterior_surf = auto_detect_exterior_3d(mesh)
        nid_to_idx = mesh.node_id_to_index() if hasattr(mesh, "node_id_to_index") else mesh.nid_to_idx
        coords = mesh.nodes_array() if hasattr(mesh, "nodes_array") else mesh.coords
        return cls(
            faces=exterior_surf.faces,
            coords=coords,
            nid_to_idx=nid_to_idx,
            use_all_exterior=use_all_exterior,
            gamma=gamma,
            skin=skin,
            friction_coefficient=friction_coefficient,
            tangential_stiffness_ratio=tangential_stiffness_ratio,
            name=name,
        )

    def _is_all_exterior(self, pairs: Any) -> bool:
        if pairs is None:
            return True
        if isinstance(pairs, (list, tuple)) and len(pairs) == 1:
            p = pairs[0]
            if isinstance(p, (list, tuple)) and len(p) >= 1:
                if str(p[0]).upper() in ("ALL EXTERIOR", "ALL_EXTERIOR", "ALL"):
                    return True
        return False

    def _face_matches_region(self, f_idx: int, region: Any) -> bool:
        if region is None or str(region).upper() in ("ALL EXTERIOR", "ALL_EXTERIOR", "ALL"):
            return True

        # 1. Direct object matching (Part, Instance, Surface)
        if hasattr(self, "face_to_part") and self.face_to_part.get(f_idx) is region:
            return True
        if hasattr(self, "face_to_instance") and self.face_to_instance.get(f_idx) is region:
            return True

        # 2. Name-based matching for surfaces, parts, instances
        reg_name = getattr(region, "name", None)
        if reg_name is not None and isinstance(reg_name, str):
            if hasattr(self, "face_to_part_name") and self.face_to_part_name.get(f_idx) == reg_name:
                return True
            if hasattr(self, "face_to_instance_name") and self.face_to_instance_name.get(f_idx) == reg_name:
                return True
            if hasattr(self, "face_labels") and f_idx in self.face_labels and reg_name in self.face_labels[f_idx]:
                return True
            if reg_name.endswith("_EXTERIOR"):
                pname = reg_name[:-9]
                if hasattr(self, "face_to_part_name") and self.face_to_part_name.get(f_idx) == pname:
                    return True
                if hasattr(self, "face_to_instance_name") and self.face_to_instance_name.get(f_idx) == pname:
                    return True

        inst_name = getattr(region, "instance_name", None)
        if inst_name is not None and hasattr(self, "face_to_instance_name") and self.face_to_instance_name.get(f_idx) == inst_name:
            return True

        # 3. String labels (Instance, Part, Surface, Set names)
        if isinstance(region, str):
            if hasattr(self, "face_labels") and f_idx in self.face_labels and region in self.face_labels[f_idx]:
                return True
            if hasattr(self, "face_to_instance_name") and self.face_to_instance_name.get(f_idx) == region:
                return True
            if hasattr(self, "face_to_part_name") and self.face_to_part_name.get(f_idx) == region:
                return True
            if region.endswith("_EXTERIOR"):
                pname = region[:-9]
                if hasattr(self, "face_to_part_name") and self.face_to_part_name.get(f_idx) == pname:
                    return True
                if hasattr(self, "face_to_instance_name") and self.face_to_instance_name.get(f_idx) == pname:
                    return True

        # 4. Geometric face or node matching
        face = self.faces[f_idx]
        if hasattr(region, "node_ids"):
            if all(nid in region.node_ids for nid in face):
                return True
        if hasattr(region, "faces"):
            f_key = tuple(face)
            f_rev = tuple(reversed(face))
            if (f_key in region.faces) or (f_rev in region.faces) or (f_key in getattr(region, "face_set", ())):
                return True

        if isinstance(region, (set, list, tuple)):
            if region and isinstance(next(iter(region)), (tuple, list)):
                f_key = tuple(face)
                f_rev = tuple(reversed(face))
                return f_key in region or f_rev in region
            return all(nid in region for nid in face)
        return False

    def _is_pair_excluded(self, fa_idx: int, fb_idx: int) -> bool:
        if not self.excluded_pairs:
            return False
        for reg_a, reg_b in self.excluded_pairs:
            match_ab = self._face_matches_region(fa_idx, reg_a) and self._face_matches_region(fb_idx, reg_b)
            match_ba = self._face_matches_region(fa_idx, reg_b) and self._face_matches_region(fb_idx, reg_a)
            if match_ab or match_ba:
                return True
        return False

    def _is_pair_included(self, fa_idx: int, fb_idx: int) -> bool:
        if self.included_pairs is None or self._is_all_exterior(self.included_pairs):
            return True
        for reg_a, reg_b in self.included_pairs:
            match_ab = self._face_matches_region(fa_idx, reg_a) and self._face_matches_region(fb_idx, reg_b)
            match_ba = self._face_matches_region(fa_idx, reg_b) and self._face_matches_region(fb_idx, reg_a)
            if match_ab or match_ba:
                return True
        return False

    def _filter_candidate_pairs(self, candidate_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not self.excluded_pairs and (self.included_pairs is None or self._is_all_exterior(self.included_pairs)):
            return candidate_pairs
        filtered = []
        for fa_idx, fb_idx in candidate_pairs:
            if self._is_pair_excluded(fa_idx, fb_idx):
                continue
            if not self._is_pair_included(fa_idx, fb_idx):
                continue
            filtered.append((fa_idx, fb_idx))
        return filtered

    def reproject_deformed(self, u: np.ndarray) -> None:
        """Update BVH collision hierarchy with deformed coordinates."""
        self.bvh.refit(u)

    def get_active_set(self, u: np.ndarray) -> frozenset:
        """Return frozenset of participating node IDs currently in penetrating contact."""
        active_nodes: Set[int] = set()
        self.bvh.refit(u)
        candidate_pairs = self._filter_candidate_pairs(self.bvh.query_self_contact())

        for fa_idx, fb_idx in candidate_pairs:
            face_a = self.faces[fa_idx]
            face_b = self.faces[fb_idx]

            idx_a = [self.nid_to_idx[nid] for nid in face_a]
            idx_b = [self.nid_to_idx[nid] for nid in face_b]

            qa = [self.coords[i] + u[3 * i : 3 * i + 3] for i in idx_a]
            qb = [self.coords[j] + u[3 * j : 3 * j + 3] for j in idx_b]

            # Face normals
            na = _quad4_normal(qa[0], qa[1], qa[2], qa[3]) if len(qa) == 4 else np.cross(qa[1] - qa[0], qa[2] - qa[0])
            nb = _quad4_normal(qb[0], qb[1], qb[2], qb[3]) if len(qb) == 4 else np.cross(qb[1] - qb[0], qb[2] - qb[0])
            na = na / (np.linalg.norm(na) + 1e-14)
            nb = nb / (np.linalg.norm(nb) + 1e-14)

            # Check opposing orientation
            if np.dot(na, nb) > -0.3:
                continue

            # Check penetration
            if len(qa) == 4 and len(qb) == 4:
                for xi_s, eta_s, _ in _GAUSS2X2:
                    Ns = _quad4_shape_functions(xi_s, eta_s)
                    x_qp = sum(Ns[k] * qa[k] for k in range(4))
                    xi_m, eta_m, _, proj_pt = _project_point_to_quad4(x_qp, *qb)
                    if -1.05 <= xi_m <= 1.05 and -1.05 <= eta_m <= 1.05:
                        g_n = -float(np.dot(x_qp - proj_pt, nb))
                        if g_n > 0.0:
                            active_nodes.update(face_a)
                            active_nodes.update(face_b)
                            break

        return frozenset(active_nodes)

    def assemble(
        self, u: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]]:
        """Assemble 3D General Contact forces and symmetric tangent stiffness.

        Returns
        -------
        f_contact : np.ndarray
            Contact resisting force vector (size 3 * N_nodes).
        (rows, cols, data) : Tuple[np.ndarray, np.ndarray, np.ndarray]
            Sparse triplet components of strictly symmetric tangent stiffness matrix.
        stats : Dict[str, float]
            Iteration statistics (max_penetration, n_active, total_normal_force).
        """
        n_dof = len(u)
        f_contact = np.zeros(n_dof, dtype=np.float64)
        K_tri: Dict[Tuple[int, int], float] = {}

        # 1. Update BVH to current deformed configuration
        self.bvh.refit(u)

        # 2. Broad-phase query for candidate pairs
        candidate_pairs = self._filter_candidate_pairs(self.bvh.query_self_contact())

        if len(candidate_pairs) == 0:
            stats = {
                "max_penetration": 0.0,
                "n_active": 0.0,
                "total_normal_force": 0.0,
                "n_candidate_pairs": 0.0,
            }
            self.last_stats = stats
            empty_arr = np.empty(0, dtype=np.int32)
            empty_val = np.empty(0, dtype=np.float64)
            return f_contact, (empty_arr, empty_arr, empty_val), stats

        # Fast path: Numba JIT narrow-phase for Quad4 exterior meshes (Hard contact / default)
        if self.use_numba and self._all_quad4 and self.law is None and self._faces_idx_arr is not None:
            from dispsolver.contact.general_contact_numba import assemble_general_contact_quad4_numba
            cand_arr = np.array(candidate_pairs, dtype=np.int32)
            f_cont, r_buf, c_buf, v_buf, stats_arr = assemble_general_contact_quad4_numba(
                cand_arr,
                self._faces_idx_arr,
                self.coords,
                u,
                self._face_E_arr,
                self.E_ref,
                self.gamma,
                self.damping_coeff,
                self.damping_clearance,
                self.tangent_fraction,
                self.mu,
                self.k_t,
            )
            # Symmetrize off-diagonals in sparse buffer: r_buf, c_buf, v_buf contains upper/lower parts
            # But assemble_general_contact_quad4_numba assembles all 8x8 pairs directly
            stats = {
                "max_penetration": float(stats_arr[0]),
                "n_active": float(stats_arr[1]),
                "total_normal_force": float(stats_arr[2]),
                "n_candidate_pairs": float(len(candidate_pairs)),
            }
            self.last_stats = stats
            return f_cont, (r_buf, c_buf, v_buf), stats

        max_penetration = 0.0
        n_active = 0
        total_normal_force = 0.0

        # 3. Narrow-phase evaluation for each broad-phase candidate pair (Generic Python fallback)
        for fa_idx, fb_idx in candidate_pairs:
            face_a = self.faces[fa_idx]
            face_b = self.faces[fb_idx]

            idx_a = [self.nid_to_idx[nid] for nid in face_a]
            idx_b = [self.nid_to_idx[nid] for nid in face_b]

            qa = [self.coords[i] + u[3 * i : 3 * i + 3] for i in idx_a]
            qb = [self.coords[j] + u[3 * j : 3 * j + 3] for j in idx_b]

            len_a = len(qa)
            len_b = len(qb)

            # Compute current face unit normals
            if len_a == 4:
                na = _quad4_normal(qa[0], qa[1], qa[2], qa[3])
            else:
                v0, v1 = qa[1] - qa[0], qa[2] - qa[0]
                n_raw = np.cross(v0, v1)
                na = n_raw / (np.linalg.norm(n_raw) + 1e-14)

            if len_b == 4:
                nb = _quad4_normal(qb[0], qb[1], qb[2], qb[3])
            else:
                v0, v1 = qb[1] - qb[0], qb[2] - qb[0]
                n_raw = np.cross(v0, v1)
                nb = n_raw / (np.linalg.norm(n_raw) + 1e-14)

            # CAE Normal Alignment Filter: contacting faces must face toward each other (opposing normals)
            if np.dot(na, nb) > -0.3:
                continue

            # Process Quadrature Points on face_a projected onto face_b
            if len_a == 4:
                quad_points = []
                for xi_s, eta_s, w_gauss in _GAUSS2X2:
                    Ns = _quad4_shape_functions(xi_s, eta_s)
                    x_qp = sum(Ns[k] * qa[k] for k in range(4))
                    dA = _quad4_area_jacobian(qa[0], qa[1], qa[2], qa[3], xi_s, eta_s) * w_gauss
                    quad_points.append((Ns, x_qp, dA))
            else:
                # Tri3: Single centroid integration with triangle area
                tri_area = 0.5 * float(np.linalg.norm(np.cross(qa[1] - qa[0], qa[2] - qa[0])))
                quad_points = [
                    ((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), sum(qa) / 3.0, tri_area)
                ]

            for qp_idx, (N_a, x_qp, w) in enumerate(quad_points):
                # Project onto face_b
                if len_b == 4:
                    xi_m, eta_m, dist, proj_pt = _project_point_to_quad4(x_qp, *qb)
                    if not (-1.05 <= xi_m <= 1.05 and -1.05 <= eta_m <= 1.05):
                        continue
                    N_b = _quad4_shape_functions(xi_m, eta_m)
                    proj_coord = (xi_m, eta_m)
                else:
                    u_t, v_t, w_t, dist, proj_pt = _project_point_to_tri3(x_qp, *qb)
                    if not (-0.05 <= u_t <= 1.05 and -0.05 <= v_t <= 1.05 and -0.05 <= w_t <= 1.05):
                        continue
                    N_b = (u_t, v_t, w_t)
                    proj_coord = (u_t, v_t, w_t)

                # Penetration gap: master normal nb points outwards toward slave
                g_n = -float(np.dot(x_qp - proj_pt, nb))

                # 1. Normal contact pressure & tangent from Law (or linear penalty fallback)
                # Compute pair effective modulus scaling factor via harmonic mean
                E_a = self._face_E.get(fa_idx, self.E_ref)
                E_b = self._face_E.get(fb_idx, self.E_ref)
                if E_a > 0.0 and E_b > 0.0 and (E_a + E_b) > 0.0:
                    E_eff = 2.0 * E_a * E_b / (E_a + E_b)
                    mod_scale = E_eff / max(self.E_ref, 1e-12)
                else:
                    mod_scale = 1.0

                p_n = 0.0
                dp_dh = 0.0
                if self.law is not None:
                    p_val, dp_val = self.law.evaluate(g_n)
                    p_n = float(p_val) * mod_scale
                    dp_dh = float(dp_val) * mod_scale
                else:
                    if g_n > 0.0:
                        gamma_pair = self.gamma * mod_scale
                        p_n = gamma_pair * g_n
                        dp_dh = gamma_pair

                # 2. Contact Stabilization & Damping (*CONTACT DAMPING / *CONTACT CONTROLS, STABILIZE)
                p_stab = 0.0
                dp_stab = 0.0
                clearance = -g_n
                if self.damping_coeff > 0.0 and clearance > 0.0:
                    if self.damping_clearance <= 0.0 or clearance < self.damping_clearance:
                        ramp = (1.0 - clearance / self.damping_clearance) if self.damping_clearance > 0.0 else 1.0
                        mu_eff = self.damping_coeff * max(0.0, ramp) * mod_scale
                        dp_stab = mu_eff

                p_total = p_n + p_stab
                dp_total = dp_dh + dp_stab

                if p_total <= 0.0 and dp_total <= 0.0:
                    continue

                if g_n > 0.0 or p_n > 0.0:
                    n_active += 1
                    if g_n > max_penetration:
                        max_penetration = g_n

                total_normal_force += p_total * w

                # 3. Tangential Coulomb Friction (Stick-Slip Return Mapping)
                f_t = np.zeros(3, dtype=np.float64)
                K_tt = np.zeros((3, 3), dtype=np.float64)
                if self.mu > 0.0 and p_total > 0.0:
                    key = (fa_idx, fb_idx, qp_idx)
                    if key not in self._anchors:
                        self._anchors[key] = proj_coord

                    if len_b == 4:
                        xi_a, eta_a = self._anchors[key]
                        Na_m = _quad4_shape_functions(xi_a, eta_a)
                        x_anchor = sum(Na_m[k] * qb[k] for k in range(4))
                    else:
                        u_a, v_a, w_a = self._anchors[key]
                        x_anchor = u_a * qb[0] + v_a * qb[1] + w_a * qb[2]

                    d = x_qp - x_anchor
                    s_t = d - float(np.dot(d, nb)) * nb
                    t1, t2 = _tangent_basis(nb)
                    s1 = float(np.dot(s_t, t1))
                    s2 = float(np.dot(s_t, t2))
                    s_mag = float(np.hypot(s1, s2))
                    k_t_pair = self.k_t * mod_scale
                    s_crit = (self.mu * p_total / k_t_pair) if k_t_pair > 0.0 else 0.0
                    T = np.outer(t1, t1) + np.outer(t2, t2)

                    if s_mag <= s_crit or s_mag < 1e-14:
                        ft1, ft2 = k_t_pair * s1, k_t_pair * s2
                        K_tt = k_t_pair * T
                    else:
                        ft1 = self.mu * p_total * (s1 / s_mag)
                        ft2 = self.mu * p_total * (s2 / s_mag)
                        shat = (s1 / s_mag) * t1 + (s2 / s_mag) * t2
                        K_tt = (self.mu * p_total / s_mag) * (T - np.outer(shat, shat))

                    f_t = ft1 * t1 + ft2 * t2

                # Contact force: Slave receives -f_vec, Master receives +f_vec
                # f_vec = p_total * nb - f_t (f_t opposes slave relative sliding)
                f_vec = p_total * nb - f_t
                for a in range(len_a):
                    f_contact[3 * idx_a[a] : 3 * idx_a[a] + 3] += -N_a[a] * w * f_vec
                for b in range(len_b):
                    f_contact[3 * idx_b[b] : 3 * idx_b[b] + 3] += +N_b[b] * w * f_vec

                # Symmetric tangent stiffness:
                # Normal dyadic outer product + Tangential friction + Tangential damping
                c_map: Dict[int, np.ndarray] = {}
                w_map: Dict[int, float] = {}
                for a in range(len_a):
                    c_map[idx_a[a]] = -N_a[a] * nb
                    w_map[idx_a[a]] = -N_a[a]
                for b in range(len_b):
                    c_map[idx_b[b]] = +N_b[b] * nb
                    w_map[idx_b[b]] = +N_b[b]

                scale_norm = dp_total * w
                scale_tan_stab = (self.tangent_fraction * dp_stab * w) if (self.tangent_fraction > 0.0 and dp_stab > 0.0) else 0.0
                I_proj = np.eye(3, dtype=np.float64) - np.outer(nb, nb)

                node_indices = list(c_map.keys())
                for ia in node_indices:
                    for ib in node_indices:
                        wab = w_map[ia] * w_map[ib]
                        k_block = np.zeros((3, 3), dtype=np.float64)
                        if dp_total > 0.0:
                            k_block += scale_norm * np.outer(c_map[ia], c_map[ib])
                        if self.mu > 0.0:
                            k_block += (wab * w) * K_tt
                        if scale_tan_stab > 0.0:
                            k_block += (wab * scale_tan_stab) * I_proj

                        for d1 in range(3):
                            r = 3 * ia + d1
                            for d2 in range(3):
                                c = 3 * ib + d2
                                if r <= c:
                                    val = float(k_block[d1, d2])
                                    if val != 0.0:
                                        K_tri[(r, c)] = K_tri.get((r, c), 0.0) + val

        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        for (r, c), val in K_tri.items():
            rows.append(r)
            cols.append(c)
            data.append(val)
            if r != c:
                rows.append(c)
                cols.append(r)
                data.append(val)

        stats = {
            "max_penetration": float(max_penetration),
            "n_active": float(n_active),
            "total_normal_force": float(total_normal_force),
            "n_candidate_pairs": float(len(candidate_pairs)),
        }
        self.last_stats = stats

        return (
            f_contact,
            (
                np.array(rows, dtype=np.int32),
                np.array(cols, dtype=np.int32),
                np.array(data, dtype=np.float64),
            ),
            stats,
        )

    def get_friction_state(self, u: np.ndarray) -> frozenset:
        """Return frozenset of active (fa_idx, fb_idx, qp_idx) tuples currently slipping."""
        if self.mu <= 0.0:
            return frozenset()

        slipping = set()
        self.bvh.refit(u)
        candidate_pairs = self._filter_candidate_pairs(self.bvh.query_self_contact())

        for fa_idx, fb_idx in candidate_pairs:
            face_a = self.faces[fa_idx]
            face_b = self.faces[fb_idx]
            idx_a = [self.nid_to_idx[nid] for nid in face_a]
            idx_b = [self.nid_to_idx[nid] for nid in face_b]
            qa = [self.coords[i] + u[3 * i : 3 * i + 3] for i in idx_a]
            qb = [self.coords[j] + u[3 * j : 3 * j + 3] for j in idx_b]

            len_a = len(qa)
            len_b = len(qb)
            if len_a == 4:
                na = _quad4_normal(qa[0], qa[1], qa[2], qa[3])
            else:
                v0, v1 = qa[1] - qa[0], qa[2] - qa[0]
                n_raw = np.cross(v0, v1)
                na = n_raw / (np.linalg.norm(n_raw) + 1e-14)

            if len_b == 4:
                nb = _quad4_normal(qb[0], qb[1], qb[2], qb[3])
            else:
                v0, v1 = qb[1] - qb[0], qb[2] - qb[0]
                n_raw = np.cross(v0, v1)
                nb = n_raw / (np.linalg.norm(n_raw) + 1e-14)

            if np.dot(na, nb) > -0.3:
                continue

            if len_a == 4:
                quad_points = []
                for xi_s, eta_s, _ in _GAUSS2X2:
                    Ns = _quad4_shape_functions(xi_s, eta_s)
                    x_qp = sum(Ns[k] * qa[k] for k in range(4))
                    quad_points.append(x_qp)
            else:
                quad_points = [sum(qa) / 3.0]

            for qp_idx, x_qp in enumerate(quad_points):
                if len_b == 4:
                    xi_m, eta_m, _, proj_pt = _project_point_to_quad4(x_qp, *qb)
                    if not (-1.05 <= xi_m <= 1.05 and -1.05 <= eta_m <= 1.05):
                        continue
                else:
                    u_t, v_t, w_t, _, proj_pt = _project_point_to_tri3(x_qp, *qb)
                    if not (-0.05 <= u_t <= 1.05 and -0.05 <= v_t <= 1.05 and -0.05 <= w_t <= 1.05):
                        continue

                g_n = -float(np.dot(x_qp - proj_pt, nb))
                p_n = 0.0
                if self.law is not None:
                    p_val, _ = self.law.evaluate(g_n)
                    p_n = float(p_val)
                else:
                    if g_n > 0.0:
                        p_n = self.gamma * g_n

                if p_n <= 0.0:
                    continue

                key = (fa_idx, fb_idx, qp_idx)
                if key not in self._anchors:
                    continue

                if len_b == 4:
                    xi_a, eta_a = self._anchors[key]
                    Na_m = _quad4_shape_functions(xi_a, eta_a)
                    x_anchor = sum(Na_m[k] * qb[k] for k in range(4))
                else:
                    u_a, v_a, w_a = self._anchors[key]
                    x_anchor = u_a * qb[0] + v_a * qb[1] + w_a * qb[2]

                d = x_qp - x_anchor
                s_t = d - float(np.dot(d, nb)) * nb
                s_mag = float(np.linalg.norm(s_t))
                s_crit = (self.mu * p_n / self.k_t) if self.k_t > 0.0 else 0.0
                if s_mag > s_crit and s_crit > 0.0:
                    slipping.add(key)

        return frozenset(slipping)

    def commit_friction_state(self, u: np.ndarray) -> None:
        """Commit converged Coulomb friction slip and update convected anchors."""
        if self.mu <= 0.0:
            return

        self.bvh.refit(u)
        candidate_pairs = self._filter_candidate_pairs(self.bvh.query_self_contact())

        for fa_idx, fb_idx in candidate_pairs:
            face_a = self.faces[fa_idx]
            face_b = self.faces[fb_idx]
            idx_a = [self.nid_to_idx[nid] for nid in face_a]
            idx_b = [self.nid_to_idx[nid] for nid in face_b]
            qa = [self.coords[i] + u[3 * i : 3 * i + 3] for i in idx_a]
            qb = [self.coords[j] + u[3 * j : 3 * j + 3] for j in idx_b]

            len_a = len(qa)
            len_b = len(qb)
            if len_a == 4:
                na = _quad4_normal(qa[0], qa[1], qa[2], qa[3])
            else:
                v0, v1 = qa[1] - qa[0], qa[2] - qa[0]
                n_raw = np.cross(v0, v1)
                na = n_raw / (np.linalg.norm(n_raw) + 1e-14)

            if len_b == 4:
                nb = _quad4_normal(qb[0], qb[1], qb[2], qb[3])
            else:
                v0, v1 = qb[1] - qb[0], qb[2] - qb[0]
                n_raw = np.cross(v0, v1)
                nb = n_raw / (np.linalg.norm(n_raw) + 1e-14)

            if np.dot(na, nb) > -0.3:
                continue

            if len_a == 4:
                quad_points = []
                for xi_s, eta_s, _ in _GAUSS2X2:
                    Ns = _quad4_shape_functions(xi_s, eta_s)
                    x_qp = sum(Ns[k] * qa[k] for k in range(4))
                    quad_points.append(x_qp)
            else:
                quad_points = [sum(qa) / 3.0]

            for qp_idx, x_qp in enumerate(quad_points):
                if len_b == 4:
                    xi_m, eta_m, _, proj_pt = _project_point_to_quad4(x_qp, *qb)
                    if not (-1.05 <= xi_m <= 1.05 and -1.05 <= eta_m <= 1.05):
                        continue
                else:
                    u_t, v_t, w_t, _, proj_pt = _project_point_to_tri3(x_qp, *qb)
                    if not (-0.05 <= u_t <= 1.05 and -0.05 <= v_t <= 1.05 and -0.05 <= w_t <= 1.05):
                        continue

                g_n = -float(np.dot(x_qp - proj_pt, nb))
                p_n = 0.0
                if self.law is not None:
                    p_val, _ = self.law.evaluate(g_n)
                    p_n = float(p_val)
                else:
                    if g_n > 0.0:
                        p_n = self.gamma * g_n

                if p_n <= 0.0:
                    continue

                key = (fa_idx, fb_idx, qp_idx)
                if key not in self._anchors:
                    if len_b == 4:
                        self._anchors[key] = (xi_m, eta_m)
                    else:
                        self._anchors[key] = (u_t, v_t, w_t)
                    continue

                if len_b == 4:
                    xi_a, eta_a = self._anchors[key]
                    Na_m = _quad4_shape_functions(xi_a, eta_a)
                    x_anchor = sum(Na_m[k] * qb[k] for k in range(4))
                else:
                    u_a, v_a, w_a = self._anchors[key]
                    x_anchor = u_a * qb[0] + v_a * qb[1] + w_a * qb[2]

                d = x_qp - x_anchor
                s_t = d - float(np.dot(d, nb)) * nb
                s_mag = float(np.linalg.norm(s_t))
                s_crit = (self.mu * p_n / self.k_t) if self.k_t > 0.0 else 0.0
                if s_mag > s_crit and s_crit > 0.0:
                    # Radial return: move anchor to x_qp - s_crit * shat
                    shat = s_t / s_mag
                    new_anchor_pt = x_qp - s_crit * shat
                    if len_b == 4:
                        xi_new, eta_new, _, _ = _project_point_to_quad4(new_anchor_pt, *qb)
                        self._anchors[key] = (xi_new, eta_new)
                    else:
                        u_new, v_new, w_new, _, _ = _project_point_to_tri3(new_anchor_pt, *qb)
                        self._anchors[key] = (u_new, v_new, w_new)

    def update_augmented_multipliers(self, u: np.ndarray, omega: float = 1.0) -> None:
        """Compatibility hook for Augmented Lagrange or PDASS multipliers."""
        pass
