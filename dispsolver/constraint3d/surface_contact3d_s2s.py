"""
surface_contact3d_s2s.py
==========================
Surface-to-surface (segment-to-segment, dual-mortar) 3D contact
discretization -- Stage A (design doc
dev_log/contact_surface_to_surface_precise_design_20260915.md sec5.4):
matching-topology slave/master faces, ONE segment per slave face (no
AABB broad phase, no Sutherland-Hodgman clipping yet -- that is Stage B,
not built here). This is a NEW class, not a modification of
DeformableSurfaceContactConstraint3D -- the design doc sec2.6 flags the
constructor signature change (`slave_node_ids` -> `slave_faces`) as a
real, breaking API change, so the existing node-to-surface class is left
completely untouched (still valid, still Abaqus's own supported
"node-to-surface" option) and this is an additive alternative,
selected explicitly, not a silent replacement.

What this buys over the point-collocation `DeformableSurfaceContactConstraint3D`
(design doc sec0, quoting Abaqus's own `ctc_contactpairform_std.txt`):
enforces contact in an AVERAGED sense over each slave node's own local
patch (a dual/bi-orthogonal Ψ basis with LOCAL, per-node support -- see
sec2.1/2.2), instead of only at the individual slave node's own
projection point. Abaqus's own measured comparison on a dissimilar-mesh
interface: node-to-surface 13-31% max CPRESS error vs. surface-to-surface
~1%.

Mathematics (design doc, exact section references below):

sec2.1 -- per slave face f (4 nodes), computed ONCE at construction from
REFERENCE coordinates (frozen, matching this project's small-sliding
convention):

    M_f[i,j] = integral over f of  N_i(xi,eta) * N_j(xi,eta) * dA
    D_f[i]   = sum_j M_f[i,j]
    A_f      = diag(D_f) @ inv(M_f)          (4x4, once per face)
    Psi_f(xi,eta) = A_f @ N(xi,eta)           (dual/bi-orthogonal basis)

sec2.2 -- D_global[nid] = sum over every slave face owning nid of that
face's own D_f entry (plain FE-style scatter-add; D_global stays a
diagonal/per-node scalar, not a matrix, even though a shared node
receives contributions from multiple faces).

sec2.4 -- weak gap per global slave node, two-pass aggregation:

    gap_numerator[nid] = sum over every segment/quad-point touching nid
                          of Psi_i(qp) * penetration(qp) * w(qp)
    g_tilde[nid] = gap_numerator[nid] / D_global[nid]

`g_tilde[nid]` substitutes for the point-collocation classes' own
`penetration_i(u)` in the UNMODIFIED PDASS formulas
(dev_log/contact_pdass_precise_design_20260915.md): `p_i = lam_i +
k_contact*g_tilde[nid]`, active iff `p_i > 0`.

sec2.5 -- force distributes to a slave face's own 4 PRIMAL-weighted
nodes and every touched master face's own 4 primal-weighted nodes via
the interpolated multiplier field `lam_h(qp) = sum_a p[face[a]] *
Psi_a(qp)` (only active nodes contribute); the tangent is a SUM OF
RANK-1 OUTER PRODUCTS, one per active multiplier node, each sized only
by that node's own local dof patch -- the concrete payoff of D_global
being diagonal (local per-node static condensation, no dense coupled
solve, no saddle-point/indefinite system).

sec3 -- `get_active_set(u)` still returns a plain `frozenset` of slave
node ids (one well-defined scalar per node, aggregation happens before
any activation decision) -- `DynamicSolver3D._contact_active_set()` and
its SDI logic need ZERO changes to consume this class, exactly like the
point-collocation classes.

Scope, explicitly (design doc sec5.4): HARD/LINEAR penalty and
`augmented_lagrange` only (mirrors `DeformableSurfaceContactConstraint3D`'s
own initial scope); frictionless; small-sliding, frozen pairing (no
`reproject_deformed()`); Stage A only -- matching-topology meshes, single
segment per slave face. Stage B (genuinely non-matching, dissimilar-
density meshes via AABB broad phase + Sutherland-Hodgman clipping) is a
separate, larger follow-on (design doc sec1, sec5.4) and is NOT
implemented in this file.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from dispsolver.constraint3d.surface_tie3d import _project_point_to_quad4
from dispsolver.constraint3d.pressure_overclosure import HardLaw


def _quad4_shape_functions(xi: float, eta: float) -> Tuple[float, float, float, float]:
    N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
    N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
    N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
    N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
    return N1, N2, N3, N4


def _quad4_tangents(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray, xi: float, eta: float):
    """dx/dxi, dx/deta at (xi, eta) -- shared by both the unit normal
    (below) and the physical-area Jacobian (`_quad4_area_jacobian`),
    computed once and reused rather than duplicating the cross product."""
    dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi = -0.25 * (1.0 - eta), 0.25 * (1.0 - eta), 0.25 * (1.0 + eta), -0.25 * (1.0 + eta)
    dN1_deta, dN2_deta, dN3_deta, dN4_deta = -0.25 * (1.0 - xi), -0.25 * (1.0 + xi), 0.25 * (1.0 + xi), 0.25 * (1.0 - xi)
    t_xi = dN1_dxi * q1 + dN2_dxi * q2 + dN3_dxi * q3 + dN4_dxi * q4
    t_eta = dN1_deta * q1 + dN2_deta * q2 + dN3_deta * q3 + dN4_deta * q4
    return t_xi, t_eta


def _quad4_normal(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray, xi: float, eta: float) -> np.ndarray:
    """Unit normal at (xi, eta). Same convention as
    surface_contact3d_deformable.py's own helper of this name."""
    t_xi, t_eta = _quad4_tangents(q1, q2, q3, q4, xi, eta)
    n = np.cross(t_xi, t_eta)
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def _quad4_area_jacobian(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray, xi: float, eta: float) -> float:
    """|dx/dxi x dx/deta| -- the physical-area quadrature weight factor
    (design doc sec2.1's `dA`), NOT normalized (unlike `_quad4_normal`)."""
    t_xi, t_eta = _quad4_tangents(q1, q2, q3, q4, xi, eta)
    return float(np.linalg.norm(np.cross(t_xi, t_eta)))


# 2x2 Gauss-Legendre on [-1,1]^2 -- exact for the M_f/D_f integral on a
# PLANAR (undistorted-in-plane) Quad4, matching this project's own
# AGENTS.md sec4.18 bilinear-quad identity (a 2D bilinear quad's own
# detJ is linear in (xi,eta), so 2x2 Gauss is exact for it; the physical
# `dA` here has the SAME bilinear structure per face, planar case).
_INV_SQRT3 = 1.0 / np.sqrt(3.0)
_GAUSS2X2 = [
    (-_INV_SQRT3, -_INV_SQRT3, 1.0),
    (_INV_SQRT3, -_INV_SQRT3, 1.0),
    (_INV_SQRT3, _INV_SQRT3, 1.0),
    (-_INV_SQRT3, _INV_SQRT3, 1.0),
]


class SurfaceToSurfaceContactConstraint3D:
    """Surface-to-surface (dual-mortar) frictionless 3D contact, Stage A
    (matching-topology, single segment per slave face). See module
    docstring for the full derivation and design doc cross-references.

    Parameters
    ----------
    slave_faces : list of 4-tuples of int
        Quad4 slave face node IDs (the averaging/dual-basis side).
    master_faces : list of 4-tuples of int
        Quad4 master face node IDs.
    nid_to_idx, coords : as in DeformableSurfaceContactConstraint3D.
    penalty_stiffness : float
        Used to build the default HardLaw when `law` is not given.
        **Units, found empirically while cross-checking against
        DeformableSurfaceContactConstraint3D (tests/test_surface_to_
        surface_contact.py's `test_penalty_stiffness_units_differ_from_
        point_collocation_by_design`), stated here so it is not
        rediscovered the hard way**: this is a PRESSURE-like penalty
        MODULUS (force/length^3 -- force/area per unit weak-gap length),
        NOT the point-collocation classes' own per-node lumped spring
        constant (force/length). `g_tilde` (module docstring sec2.4)
        carries units of length, and the force-distribution formula
        (sec2.5) multiplies `p = lam + k_contact*g_tilde` by a physical
        AREA quadrature weight `w` to produce a force -- so `k_contact`
        here must already be a pressure/length density for that product
        to come out in force units. Passing the SAME numeric value used
        for `DeformableSurfaceContactConstraint3D.penalty_stiffness` on
        the same mesh will NOT produce a comparable total force (differs
        by roughly the interface's own representative nodal tributary
        area, confirmed ~3.8x on this project's own two-block regression
        fixture) -- do not reuse that class's own `k_ref` auto-derivation
        formula unchanged for this class.
    law : optional PressureOverclosureLaw
        Scope restriction (design doc sec2.1, mirrors the point-
        collocation classes' own AL restriction): must be a HardLaw (or
        the default) -- the weak-gap `g_tilde` plugs directly into the
        Alart-Curnier `lam + k_contact*g_tilde` formula, which assumes a
        linear law; a graduated soft/nonlinear law is not yet wired here
        (matches DeformableSurfaceContactConstraint3D's own current
        scope).
    position_tolerance : float
        Max slave-face-centroid-to-master-face distance for pairing.
    augmented_lagrange : bool
        PDASS-compatible single-loop augmented Lagrangian (design doc
        sec2.4's note: `g_tilde` substitutes for point penetration in
        the UNMODIFIED PDASS formulas -- `DynamicSolver3D.solve_step()`
        already calls `update_augmented_multipliers()` at its existing
        commit points for any constraint exposing that method and an
        `augmented_lagrange` attribute; no solver-side change needed).
    """

    def __init__(
        self,
        slave_faces: List[Tuple[int, int, int, int]],
        master_faces: List[Tuple[int, int, int, int]],
        nid_to_idx: Dict[int, int],
        coords: np.ndarray,
        penalty_stiffness: float = 1.0e8,
        law: Optional[Any] = None,
        position_tolerance: float = 2.0,
        augmented_lagrange: bool = False,
        name: str = "SURFACE_TO_SURFACE_CONTACT_3D",
    ):
        self.slave_faces = list(slave_faces)
        self.master_faces = list(master_faces)
        self.nid_to_idx = nid_to_idx
        self.coords = coords
        self.k_contact = float(penalty_stiffness)
        if law is not None and not isinstance(law, HardLaw):
            raise ValueError(
                f"SurfaceToSurfaceContactConstraint3D '{name}': only a HardLaw (or the "
                "default) is supported -- the weak-gap g_tilde plugs directly into the "
                "linear Alart-Curnier formula lam + k_contact*g_tilde (dev_log/contact_"
                f"surface_to_surface_precise_design_20260915.md sec2.4). Got law={type(law).__name__}."
            )
        self.law = law if law is not None else HardLaw(self.k_contact)
        self.position_tolerance = float(position_tolerance)
        self.augmented_lagrange = bool(augmented_lagrange)
        self.name = name

        # sec2.1: per-slave-face dual-basis matrix, built ONCE from
        # reference coordinates.
        self._A_f: Dict[int, np.ndarray] = {}
        # sec2.2: global per-node D, scatter-added across every slave
        # face that owns that node.
        self.D_global: Dict[int, float] = {}
        self._build_reference_mortar_matrices()

        # sec1 (Stage A only): one segment per slave face, frozen pairing
        # + frozen (xi_m, eta_m) per quad point at construction (small-
        # sliding, matching every sibling class's own convention).
        self.segments: List[Dict[str, Any]] = []
        self._build_pairs()

        # sec2.5: which segments touch a given global slave node id --
        # needed because an interior node can be shared by up to 4 slave
        # faces (hence up to 4 segments) in a regular mesh; the tangent's
        # rank-1 vector for that node sums contributions across ALL of
        # them (design doc sec2.5's "for seg in segments_of_face(f)",
        # generalized here to "for seg touching node i").
        self._segments_touching_node: Dict[int, List[Dict[str, Any]]] = {}
        for seg in self.segments:
            face = self.slave_faces[seg["slave_face_idx"]]
            for nid in face:
                self._segments_touching_node.setdefault(nid, []).append(seg)

        all_slave_nids = sorted(set(nid for face in self.slave_faces for nid in face))
        self._lam: Dict[int, float] = {nid: 0.0 for nid in all_slave_nids}

    # -- construction-time (reference-config) setup -----------------

    def _build_reference_mortar_matrices(self) -> None:
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
            A_f = np.diag(D_f) @ np.linalg.inv(M_f)
            self._A_f[f_idx] = A_f
            for a, nid in enumerate(face):
                self.D_global[nid] = self.D_global.get(nid, 0.0) + D_f[a]

    def _build_pairs(self) -> None:
        self.segments = []
        for f_idx, face in enumerate(self.slave_faces):
            idx = [self.nid_to_idx[nid] for nid in face]
            q = [self.coords[i] for i in idx]
            centroid = 0.25 * (q[0] + q[1] + q[2] + q[3])

            best = None
            min_dist = float("inf")
            for m_face in self.master_faces:
                m_idx = [self.nid_to_idx[nid] for nid in m_face]
                mq = [self.coords[i] for i in m_idx]
                xi_c, eta_c, dist, _ = _project_point_to_quad4(centroid, *mq)
                if dist < min_dist:
                    min_dist = dist
                    best = (m_face, mq, xi_c, eta_c)

            if best is None or min_dist > self.position_tolerance:
                continue  # unpaired slave face (Stage A: no clipping fallback)
            m_face, mq, xi_c, eta_c = best

            # Sign orientation from the CENTROID gap, matching every
            # sibling class's own frozen small-sliding convention
            # (gap0 >= 0 defines the positive side).
            n_c_raw = _quad4_normal(mq[0], mq[1], mq[2], mq[3], xi_c, eta_c)
            Nm_c = _quad4_shape_functions(xi_c, eta_c)
            xm_c = sum(Nm_c[k] * mq[k] for k in range(4))
            gap0 = float(np.dot(centroid - xm_c, n_c_raw))
            sign = 1.0 if gap0 >= 0.0 else -1.0

            quad_points = []
            for xi_s, eta_s, w_gauss in _GAUSS2X2:
                Ns = _quad4_shape_functions(xi_s, eta_s)
                xs_qp = sum(Ns[k] * q[k] for k in range(4))
                xi_m, eta_m, _, _ = _project_point_to_quad4(xs_qp, *mq)
                area_jac = _quad4_area_jacobian(q[0], q[1], q[2], q[3], xi_s, eta_s)
                w = w_gauss * area_jac
                n_qp = sign * _quad4_normal(mq[0], mq[1], mq[2], mq[3], xi_m, eta_m)
                quad_points.append({
                    "w": w, "xi_s": xi_s, "eta_s": eta_s,
                    "xi_m": xi_m, "eta_m": eta_m, "n": n_qp,
                })

            self.segments.append({
                "slave_face_idx": f_idx, "master_face": m_face, "quad_points": quad_points,
            })

    # -- per-solve-call evaluation ------------------------------------

    def _compute_g_tilde(self, u: np.ndarray) -> Dict[int, float]:
        """sec2.4's two-pass weak-gap aggregation. Recomputed fresh every
        call (NOT cached across calls) -- deliberately, since `u`
        changes every Newton iteration/trial."""
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
                penetration_q = -float(np.dot(xs - xm, qp["n"]))
                for a in range(4):
                    gap_numerator[face[a]] += Psi[a] * penetration_q * qp["w"]
        return {
            nid: (gap_numerator[nid] / self.D_global[nid] if self.D_global[nid] != 0.0 else 0.0)
            for nid in self.D_global
        }

    def _node_pressures(self, g_tilde: Dict[int, float]) -> Dict[int, float]:
        p: Dict[int, float] = {}
        for nid, g in g_tilde.items():
            if self.augmented_lagrange:
                pi = self._lam[nid] + self.k_contact * g
                p[nid] = pi if pi > 0.0 else 0.0
            else:
                pi, _ = self.law.evaluate(g)
                p[nid] = pi if pi > 0.0 else 0.0
        return p

    def get_active_set(self, u: np.ndarray) -> frozenset:
        """Same contract as every sibling contact class: a plain
        frozenset of slave node ids, no per-segment detail leaked --
        DynamicSolver3D._contact_active_set()/SDI need zero changes
        (design doc sec3)."""
        g_tilde = self._compute_g_tilde(u)
        p = self._node_pressures(g_tilde)
        return frozenset(nid for nid, pi in p.items() if pi > 0.0)

    def assemble(self, u: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]]:
        n_dof = len(u)
        f_contact = np.zeros(n_dof, dtype=np.float64)
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

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

        # Force (sec2.5): lam_h(qp) = sum_a p[face[a]]*Psi_a(qp), only
        # active nodes contribute; distributes to the slave face's own
        # PRIMAL N_a and the master face's own PRIMAL N_b (Psi is the
        # multiplier's test function, not the displacement interpolation
        # -- design doc sec2.5's explicit warning against reweighting by
        # Psi on the force side too).
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
                n_qp = qp["n"]
                w = qp["w"]
                for a in range(4):
                    f_contact[3 * idx_s[a]: 3 * idx_s[a] + 3] += -Ns[a] * w * lam_h * n_qp
                for b in range(4):
                    f_contact[3 * idx_m[b]: 3 * idx_m[b] + 3] += +Nm[b] * w * lam_h * n_qp

        # Tangent (sec2.5): ONE rank-1 outer product per ACTIVE
        # multiplier node, over that node's own local dof patch (its own
        # slave face(s)' nodes + every touched master node) -- the
        # concrete payoff of D_global being diagonal.
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
                    n_qp = qp["n"]
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
                            rows.append(3 * ia + d1)
                            cols.append(3 * ib + d2)
                            data.append(float(k_block[d1, d2]))

        stats = {
            "max_penetration": max_penetration,
            "n_active": n_active,
            "n_candidates": len(self.D_global),
            "n_segments": len(self.segments),
            "total_normal_force": total_normal_force,
        }
        return (
            f_contact,
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(data, dtype=np.float64)),
            stats,
        )

    def update_augmented_multipliers(self, u: np.ndarray, omega: float = 1.0) -> Dict[str, float]:
        """PDASS single-loop augmented Lagrangian (design doc sec2.4's
        note that g_tilde substitutes for point penetration in the
        UNMODIFIED formula) -- near-verbatim port of every sibling
        class's own method of this name. Called by
        DynamicSolver3D.solve_step() at its existing commit points (no
        solver-side change needed: dispatch is via
        `getattr(c, "augmented_lagrange", False)` + a plain method call,
        already generic)."""
        if not self.augmented_lagrange:
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
