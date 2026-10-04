"""
surface_contact3d_deformable.py
=================================
Deformable-vs-deformable 3D node-to-surface contact (design doc
dev_log/3d_contact_implementation_design_20260913.md sec9 Phase 2:
"Extend from body-vs.-rigid-plane to body-vs.-body").

Generalizes SurfaceContactConstraint3D's rigid-plane contact to a MOVING
master surface, reusing surface_tie3d.py's proven node-to-Quad4-face
small-sliding projection (`_project_point_to_quad4`) -- the pairing
(which face, which (xi, eta)) is frozen at construction, matching
SurfaceTieConstraint3D's default `freeze_projection=True` and this
project's own small-sliding scope decision (design doc sec3) -- but the
face's CURRENT (deformed) position and local normal are recomputed every
assemble() call, so the constraint tracks real deformation of the master
body, unlike the rigid-plane case where the master has no DOFs at all.

Tangent: by default (`consistent_tangent=True`, since 2026-10-01) the
normal part is the exact Jacobian of the pair force, including the
normal's rotation dn/du; this makes the block UNSYMMETRIC, which
DynamicSolver3D._solve_linear detects and factors with LU instead of
Cholesky. Friction keeps the earlier symmetric stick/slip block unless
`consistent_friction_tangent=True` (exact, FD-verified, but it made the
Hertz roller SLIDE step diverge while the symmetric block completes it).
`consistent_tangent=False` restores the earlier modified-Newton block
(normal held frozen).

One-sided complementarity (contact, not a bonded tie) through the SAME
PressureOverclosureLaw seam (dispsolver/constraint3d/pressure_overclosure.py)
rigid-plane contact and its nonlinear/soft laws already use -- HardLaw,
NonlinearPenaltyLaw, and the three soft laws all plug in unchanged.
Augmented Lagrangian (PDASS single-loop, dev_log/contact_pdass_precise_
design_20260915.md) IS supported, restricted to HARD contact only
(design doc sec B.4, enforced by `__init__`'s own HardLaw check) --
same restriction as the rigid-plane sibling, not a further limitation.
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


def _quad4_normal(q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, q4: np.ndarray, xi: float, eta: float) -> np.ndarray:
    """Unit normal of the Quad4 face at (xi, eta), from the cross product
    of the local tangent vectors dx/dxi, dx/deta. Its derivative is
    _quad4_normal_derivatives (used by the default consistent tangent)."""
    dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi = -0.25 * (1.0 - eta), 0.25 * (1.0 - eta), 0.25 * (1.0 + eta), -0.25 * (1.0 + eta)
    dN1_deta, dN2_deta, dN3_deta, dN4_deta = -0.25 * (1.0 - xi), -0.25 * (1.0 + xi), 0.25 * (1.0 + xi), 0.25 * (1.0 - xi)
    t_xi = dN1_dxi * q1 + dN2_dxi * q2 + dN3_dxi * q3 + dN4_dxi * q4
    t_eta = dN1_deta * q1 + dN2_deta * q2 + dN3_deta * q3 + dN4_deta * q4
    n = np.cross(t_xi, t_eta)
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def _skew(a: np.ndarray) -> np.ndarray:
    """[a]x such that [a]x @ v == np.cross(a, v)."""
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def _quad4_normal_derivatives(q: List[np.ndarray], xi: float, eta: float, n: np.ndarray) -> List[np.ndarray]:
    """dn/dq_b (3x3) for each of the face's 4 vertices, for the SIGNED unit
    normal `n` (= sign * m/|m|, m = t_xi x t_eta) at (xi, eta):
    dn = sign*(I - m^ m^T)/|m| dm = (I - n n^T)/|m| * sign*dm, and the sign is
    recovered from n.m. Zero blocks for a degenerate face (norm < 1e-14),
    matching _quad4_normal's fixed-fallback normal."""
    dN_dxi = (-0.25 * (1.0 - eta), 0.25 * (1.0 - eta), 0.25 * (1.0 + eta), -0.25 * (1.0 + eta))
    dN_deta = (-0.25 * (1.0 - xi), -0.25 * (1.0 + xi), 0.25 * (1.0 + xi), 0.25 * (1.0 - xi))
    t_xi = sum(dN_dxi[b] * q[b] for b in range(4))
    t_eta = sum(dN_deta[b] * q[b] for b in range(4))
    m = np.cross(t_xi, t_eta)
    norm = np.linalg.norm(m)
    if norm < 1e-14:
        return [np.zeros((3, 3)) for _ in range(4)]
    sign = 1.0 if np.dot(m, n) >= 0.0 else -1.0
    P = (np.eye(3) - np.outer(n, n)) * (sign / norm)
    S_eta, S_xi = _skew(t_eta), _skew(t_xi)
    return [P @ (-dN_dxi[b] * S_eta + dN_deta[b] * S_xi) for b in range(4)]


def _tangent_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal (t1, t2) spanning the plane perpendicular to `n`, by a
    single Gram-Schmidt step (design doc dev_log/contact_friction_precise_
    design_20260916.md sec4). Recomputed FRESH every call here (unlike
    the rigid-plane sibling's ONE-TIME basis) because the deformable
    master face's own normal genuinely rotates between increments."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = ref - np.dot(ref, n) * n
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


class DeformableSurfaceContactConstraint3D:
    """Frictionless, small-sliding, node-to-DEFORMABLE-surface penalty
    contact. See module docstring for the small-sliding/modified-Newton
    scope and the shared pressure-overclosure law seam.

    Parameters
    ----------
    slave_node_ids : list of int
    master_faces : list of 4-tuples of int
        Quad4 master face node IDs, same convention as
        SurfaceTieConstraint3D.
    nid_to_idx, coords : as in SurfaceContactConstraint3D.
    penalty_stiffness : float
        Used only to build the default HardLaw when `law` is not given
        (mirrors SurfaceContactConstraint3D's own constructor).
    law : optional PressureOverclosureLaw
    position_tolerance : float
        Max initial slave-to-face distance for pairing (mirrors
        SurfaceTieConstraint3D's own parameter).
    """

    def __init__(
        self,
        slave_node_ids: List[int],
        master_faces: List[Tuple[int, int, int, int]],
        nid_to_idx: Dict[int, int],
        coords: np.ndarray,
        penalty_stiffness: float = 1.0e8,
        law: Optional[Any] = None,
        position_tolerance: float = 2.0,
        augmented_lagrange: bool = False,
        chatter_stabilization: bool = False,
        hysteresis_band: float = 0.0,
        friction_coefficient: float = 0.0,
        tangential_stiffness_ratio: float = 0.1,
        name: str = "SURFACE_CONTACT_3D_DEFORMABLE",
        consistent_tangent: bool = True,
        consistent_friction_tangent: bool = False,
        master_interior: Optional[Dict[tuple, np.ndarray]] = None,
    ):
        self.master_interior = dict(master_interior or {})
        self.consistent_tangent = bool(consistent_tangent)
        self.consistent_friction_tangent = bool(consistent_friction_tangent)
        self.slave_node_ids = list(slave_node_ids)
        self.master_faces = list(master_faces)
        self.nid_to_idx = nid_to_idx
        self.coords = coords
        self.k_contact = float(penalty_stiffness)
        # Active-set CHATTERING stabilization (design doc
        # dev_log/contact_stabilization_precise_design_20260915.md sec2) --
        # near-verbatim port of SurfaceContactConstraint3D's own gate/scope
        # restriction. See that class's __init__ docstring comment for the
        # full rationale (why c_stab-style force damping is not enough,
        # and why the gate needs the raw unclamped HardLaw argument).
        if chatter_stabilization and law is not None and not isinstance(law, HardLaw):
            raise ValueError(
                f"DeformableSurfaceContactConstraint3D '{name}': chatter_stabilization=True "
                "requires a HardLaw (or the default) -- the hysteresis gate reads the raw, "
                "unclamped lam + k_contact*penetration argument (dev_log/contact_stabilization_"
                f"precise_design_20260915.md sec2.1). Got law={type(law).__name__}."
            )
        # Augmented Lagrangian restricted to HARD contact only (mirrors
        # SurfaceContactConstraint3D's own identical check -- Abaqus's own
        # documented restriction, dev_log/contact_abaqus_grade_design_
        # 20260915.md sec B.4). Without this guard, assemble()'s AL branch
        # silently ignores whatever `law` was passed (its `elif
        # self.augmented_lagrange` branch always wins over `self.law.
        # evaluate(...)`, dev_log/contact_deformable_soft_nonlinear_
        # 20260916.md) -- a caller combining AL with a soft/nonlinear law
        # would get plain-linear AL behavior with no error, exactly the
        # silent-wrong-answer class AGENTS.md sec4.8/4.16 already
        # catalogued; refuse it explicitly instead.
        if augmented_lagrange and law is not None and not isinstance(law, HardLaw):
            raise ValueError(
                f"DeformableSurfaceContactConstraint3D '{name}': augmented_lagrange=True requires "
                "a HardLaw (or the default) -- Abaqus restricts the augmented Lagrange method to "
                "hard pressure-overclosure relationships only (dev_log/contact_abaqus_grade_"
                f"design_20260915.md sec B.4). Got law={type(law).__name__}."
            )
        self.law = law if law is not None else HardLaw(self.k_contact)
        self.chatter_stabilization = bool(chatter_stabilization)
        self.hysteresis_band = float(hysteresis_band)
        self.position_tolerance = float(position_tolerance)
        self.name = name

        # pairs: (slave_nid, master_face, xi, eta, sign) -- `sign` orients
        # the raw cross-product normal so that the slave starts on the
        # POSITIVE side (gap0 >= 0), fixed at construction (frozen
        # small-sliding pairing, matching SurfaceTieConstraint3D's own
        # default freeze_projection=True).
        self.pairs: List[Tuple[int, Tuple[int, int, int, int], float, float, float]] = []
        self._build_pairs()

        # PDASS / single-loop augmented Lagrangian (design doc
        # dev_log/contact_pdass_precise_design_20260915.md sec3): a
        # near-verbatim port of SurfaceContactConstraint3D's own
        # augmented_lagrange/_lam machinery. self._lam is keyed by SLAVE
        # node id (one multiplier per pair, matching the rigid sibling's
        # per-node convention -- built from self.pairs, not
        # self.slave_node_ids, since a candidate slave outside
        # position_tolerance never got paired and has no gap to augment).
        self.augmented_lagrange = bool(augmented_lagrange)
        self._lam: Dict[int, float] = {p[0]: 0.0 for p in self.pairs}
        # Chattering gate state, keyed the same way as self._lam above
        # (by slave node id, built from self.pairs -- an unpaired
        # candidate has no gap to gate). Stage 1 (static band): no
        # rolling-window/escalation state yet, design doc sec5.4.
        self._gate_active: Dict[int, bool] = {p[0]: False for p in self.pairs}

        # Penalty-regularized Coulomb friction (design doc dev_log/
        # contact_friction_precise_design_20260916.md sec1-3,4,7.2).
        # mu=0.0 (default) is an exact byte-identical fallback to the
        # frictionless behavior above.
        self.mu = float(friction_coefficient)
        self.k_t = float(tangential_stiffness_ratio) * self.k_contact
        # Friction anchor stored as a CONVECTED (xi_anchor, eta_anchor)
        # material coordinate on the master face (sec4) -- NOT a fixed
        # global 3D point, which would misread rigid rotation of the
        # master face as spurious slip (the exact AGENTS.md sec4.14-class
        # defect this design proactively avoids). Initialized to the
        # pair's own frozen (xi, eta) -- i.e. zero initial slip, the
        # anchor starts exactly at the slave's own projection point.
        self._anchor_xi: Dict[int, float] = {p[0]: p[2] for p in self.pairs}
        self._anchor_eta: Dict[int, float] = {p[0]: p[3] for p in self.pairs}

    def _build_pairs(self) -> None:
        for s_nid in self.slave_node_ids:
            s_idx = self.nid_to_idx[s_nid]
            xs0 = self.coords[s_idx]

            best = None
            min_dist = float("inf")
            for m_face in self.master_faces:
                m_idx = [self.nid_to_idx[nid] for nid in m_face]
                q1, q2, q3, q4 = (self.coords[i] for i in m_idx)
                xi, eta, dist, _ = _project_point_to_quad4(xs0, q1, q2, q3, q4)
                if dist < min_dist:
                    min_dist = dist
                    best = (m_face, xi, eta, q1, q2, q3, q4)

            if best is None or min_dist > self.position_tolerance:
                continue
            m_face, xi, eta, q1, q2, q3, q4 = best
            n_raw = _quad4_normal(q1, q2, q3, q4, xi, eta)
            N1, N2, N3, N4 = _quad4_shape_functions(xi, eta)
            xm_proj0 = N1 * q1 + N2 * q2 + N3 * q3 + N4 * q4
            interior = self.master_interior.get(tuple(m_face))
            if interior is not None:
                # Outward from the master body, whatever the initial gap.
                sign = 1.0 if np.dot(n_raw, xm_proj0 - interior) >= 0.0 else -1.0
            else:
                # No owning element known (raw face tuples): legacy rule, the
                # normal points toward the slave. Wrong for an initially
                # overclosed pair -- the 1e-5 guard only covers flush meshes.
                gap0 = float(np.dot(xs0 - xm_proj0, n_raw))
                sign = 1.0 if (abs(gap0) < 1e-5 or gap0 >= 0.0) else -1.0
            self.pairs.append((s_nid, m_face, xi, eta, sign))

    def _current_gap_and_normal(self, u: np.ndarray, s_nid: int, m_face, xi: float, eta: float, sign: float):
        s_idx = self.nid_to_idx[s_nid]
        m_idx = [self.nid_to_idx[nid] for nid in m_face]
        xs = self.coords[s_idx] + u[3 * s_idx: 3 * s_idx + 3]
        q = [self.coords[i] + u[3 * i: 3 * i + 3] for i in m_idx]
        N1, N2, N3, N4 = _quad4_shape_functions(xi, eta)
        xm_proj = N1 * q[0] + N2 * q[1] + N3 * q[2] + N4 * q[3]
        n = sign * _quad4_normal(q[0], q[1], q[2], q[3], xi, eta)
        gap = float(np.dot(xs - xm_proj, n))
        return gap, n, (N1, N2, N3, N4), s_idx, m_idx

    def _friction_force_and_tangent(self, s_nid: int, xs: np.ndarray, q: List[np.ndarray],
                                     n: np.ndarray, p_i: float):
        """Pure computation (design doc sec1-2,4,7.2), no side effects --
        reused identically by assemble(), get_friction_state(), and
        commit_friction_state(). `q` are the master face's CURRENT
        (deformed) vertex coordinates (same 4 points `_current_gap_and_
        normal()` already computed for this pair). Returns (f_t_3d, Ktt
        (3x3), is_slip, s1, s2, s_mag, t1, t2)."""
        if self.mu <= 0.0 or p_i <= 0.0:
            t1, t2 = _tangent_basis(n)
            return np.zeros(3), np.zeros((3, 3)), False, 0.0, 0.0, 0.0, t1, t2
        t1, t2 = _tangent_basis(n)
        xi_a, eta_a = self._anchor_xi[s_nid], self._anchor_eta[s_nid]
        Na = _quad4_shape_functions(xi_a, eta_a)
        xm_anchor = Na[0] * q[0] + Na[1] * q[1] + Na[2] * q[2] + Na[3] * q[3]
        d = xs - xm_anchor
        s1 = float(np.dot(d, t1))
        s2 = float(np.dot(d, t2))
        s_mag = float(np.hypot(s1, s2))
        s_crit = (self.mu * p_i / self.k_t) if self.k_t > 0.0 else 0.0
        T = np.outer(t1, t1) + np.outer(t2, t2)
        if s_mag <= s_crit:
            ft1, ft2 = self.k_t * s1, self.k_t * s2
            Ktt = self.k_t * T
            is_slip = False
        else:
            ft1 = self.mu * p_i * (s1 / s_mag)
            ft2 = self.mu * p_i * (s2 / s_mag)
            shat = (s1 / s_mag) * t1 + (s2 / s_mag) * t2
            Ktt = (self.mu * p_i / s_mag) * (T - np.outer(shat, shat))
            is_slip = True
        f_t_3d = ft1 * t1 + ft2 * t2
        return f_t_3d, Ktt, is_slip, s1, s2, s_mag, t1, t2

    def _consistent_pair_stiffness(self, s_nid: int, xs: np.ndarray, q: List[np.ndarray],
                                   n: np.ndarray, xi: float, eta: float,
                                   f_mag: float, k_diag: float, Ktt: np.ndarray) -> np.ndarray:
        """15x15 Jacobian of this pair's contact force (slave + 4 master
        vertices, 3 DOFs each) as assemble() computes it, i.e.
        d(f_contact[a])/d(u_b) with f_contact[a] = -w_a * (f_mag*n - f_t).

        The normal part is exact: the normal's rotation dn/du (in both the
        gap and the force direction) and the (xs - xm).dn/du gap term, which
        the pre-2026-10-01 block dropped. The friction part is the same
        symmetric Ktt as before unless `consistent_friction_tangent`, which
        also differentiates the slip-branch mu*p(gap) coupling and the
        anchor's own shape-function weights; the friction force is written
        through T = I - n n^T, so neither depends on _tangent_basis's
        arbitrary t1/t2. FD-verified to ~1e-10 in the frictionless, stick
        and slip branches (tests/test_deformable_contact_consistent_
        tangent.py); the old block was 3-41% off there.
        """
        N = _quad4_shape_functions(xi, eta)
        w = np.array([1.0, -N[0], -N[1], -N[2], -N[3]])
        xm = N[0] * q[0] + N[1] * q[1] + N[2] * q[2] + N[3] * q[3]
        d_g = xs - xm
        Dn = [np.zeros((3, 3))] + _quad4_normal_derivatives(q, xi, eta, n)

        # d(penetration)/du_b = -(w_b n + Dn_b^T d_g)   (row vectors)
        dpen = [-(w[b] * n + Dn[b].T @ d_g) for b in range(5)]
        J = [k_diag * np.outer(n, dpen[b]) + f_mag * Dn[b] for b in range(5)]

        if self.mu > 0.0 and f_mag > 0.0 and not self.consistent_friction_tangent:
            # Symmetric friction block (the pre-2026-10-01 Ktt, projection
            # weights): the exact one below is FD-verified but made the
            # Hertz roller SLIDE step diverge at t=0.213 (active set 6->2->10
            # per iteration), while this one completes it.
            J = [J[b] - w[b] * Ktt for b in range(5)]
        elif self.mu > 0.0 and f_mag > 0.0:
            Na = _quad4_shape_functions(self._anchor_xi[s_nid], self._anchor_eta[s_nid])
            wa = np.array([1.0, -Na[0], -Na[1], -Na[2], -Na[3]])
            xa = Na[0] * q[0] + Na[1] * q[1] + Na[2] * q[2] + Na[3] * q[3]
            d_a = xs - xa
            T = np.eye(3) - np.outer(n, n)
            s_vec = T @ d_a
            s_mag = float(np.linalg.norm(s_vec))
            s_crit = (self.mu * f_mag / self.k_t) if self.k_t > 0.0 else 0.0
            n_da = float(np.dot(n, d_a))
            for b in range(5):
                ds = wa[b] * T - n_da * Dn[b] - np.outer(n, Dn[b].T @ d_a)
                if s_mag <= s_crit:
                    Jt = self.k_t * ds
                else:
                    shat = s_vec / s_mag
                    Jt = (self.mu * k_diag) * np.outer(shat, dpen[b]) \
                        + (self.mu * f_mag / s_mag) * (np.eye(3) - np.outer(shat, shat)) @ ds
                J[b] = J[b] - Jt

        return -np.kron(w[:, None], np.hstack(J))

    def get_active_set(self, u: np.ndarray) -> frozenset:
        """Same role as SurfaceContactConstraint3D.get_active_set -- feeds
        DynamicSolver3D's severe-discontinuity-iteration (SDI) handling."""
        frozen = getattr(self, "frozen_active_nodes", None)
        if self.chatter_stabilization:
            # Debounced label -- reads the persistent gate written by
            # update_gate_state(), same convention as
            # SurfaceContactConstraint3D's own sibling branch (design doc
            # sec2.2).
            return frozenset(nid for nid in self._gate_active if self._gate_active[nid])

        active = set()
        for s_nid, m_face, xi, eta, sign in self.pairs:
            gap, _, _, _, _ = self._current_gap_and_normal(u, s_nid, m_face, xi, eta, sign)
            if self.augmented_lagrange:
                p = self._lam[s_nid] + self.k_contact * (-gap)
            else:
                p, _ = self.law.evaluate(-gap)
            if p > 0.0:
                active.add(s_nid)
        # Frozen pairs stay in the set, but a pair that newly closes while a
        # freeze is on must still show up -- it is a real status change.
        if frozen is not None:
            active |= set(frozen)
        return frozenset(active)

    def update_gate_state(self, u: np.ndarray) -> None:
        """Active-set CHATTERING stabilization, Stage 1 -- near-verbatim
        port of SurfaceContactConstraint3D.update_gate_state() (design doc
        dev_log/contact_stabilization_precise_design_20260915.md sec2.2),
        substituting this class's own _current_gap_and_normal() for the
        rigid sibling's direct dot-product gap. Must be called by
        DynamicSolver3D.solve_step() on the accepted iterate, before
        _contact_active_set(), never from assemble()."""
        if not self.chatter_stabilization:
            return
        h = self.hysteresis_band
        for s_nid, m_face, xi, eta, sign in self.pairs:
            gap, _, _, _, _ = self._current_gap_and_normal(u, s_nid, m_face, xi, eta, sign)
            x_i = self._lam[s_nid] + self.k_contact * (-gap)
            g = self._gate_active[s_nid]
            if not g and x_i > h:
                g = True
            elif g and x_i < -h:
                g = False
            self._gate_active[s_nid] = g

    def _law_opening(self):
        """(h_open, slope) where the pressure-overclosure law starts carrying
        load: h = -c0 for every analytic law, the first table point for
        TabularSoftLaw; slope is the law's tangent just past that point."""
        if hasattr(self.law, "h_pts"):
            h_open = float(self.law.h_pts[0])
        else:
            h_open = -float(getattr(self.law, "c0", 0.0))
        eps = 1e-9 * max(1.0, abs(h_open))
        _, k_open = self.law.evaluate(h_open + eps)
        return h_open, (k_open if k_open > 0.0 else self.k_contact)

    def capture_state(self) -> dict:
        """Capture internal contact/friction states for rollback on increment cutback."""
        return {
            "_lam": self._lam.copy() if hasattr(self, "_lam") and self._lam is not None else None,
            "_gate_active": self._gate_active.copy() if hasattr(self, "_gate_active") and self._gate_active is not None else None,
            "_anchor_xi": self._anchor_xi.copy() if hasattr(self, "_anchor_xi") and self._anchor_xi is not None else None,
            "_anchor_eta": self._anchor_eta.copy() if hasattr(self, "_anchor_eta") and self._anchor_eta is not None else None,
        }

    def restore_state(self, state: dict) -> None:
        """Restore internal contact/friction states upon increment cutback."""
        if not isinstance(state, dict):
            return
        if state.get("_lam") is not None:
            self._lam = state["_lam"].copy()
        if state.get("_gate_active") is not None:
            self._gate_active = state["_gate_active"].copy()
        if state.get("_anchor_xi") is not None:
            self._anchor_xi = state["_anchor_xi"].copy()
        if state.get("_anchor_eta") is not None:
            self._anchor_eta = state["_anchor_eta"].copy()

    def assemble(self, u: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], Dict[str, float]]:
        n_dof = len(u)
        f_contact = np.zeros(n_dof, dtype=np.float64)
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []

        max_penetration = 0.0
        n_active = 0
        total_normal_force_mag = 0.0

        for s_nid, m_face, xi, eta, sign in self.pairs:
            gap, n, (N1, N2, N3, N4), s_idx, m_idx = self._current_gap_and_normal(u, s_nid, m_face, xi, eta, sign)
            penetration = -gap
            frozen = getattr(self, "frozen_active_nodes", None)
            if frozen is not None and s_nid in frozen:
                # SDI chatter freeze: keep this pair engaged even if it opens.
                # While the pressure-overclosure law gives a positive force it
                # is used AS IS -- the former `k_contact * penetration` here
                # silently replaced any soft/nonlinear law by a linear spring
                # for frozen pairs, a residual jump at fixed u that stalled
                # SOFT_EXPONENTIAL contact (2026-10-04). Past the law's opening
                # point it continues as a linear spring with the law's own
                # opening slope (bilateral, so the freeze removes the kink; a
                # unilateral variant stalled the Hertz roller INDENT step at
                # t=0.0486 vs 0.1702, measured 2026-09-30). For HardLaw this
                # is identical to the former expression.
                f_mag, k_diag = self.law.evaluate(penetration)
                if f_mag <= 0.0:
                    h_open, k_open = self._law_opening()
                    f_mag = k_open * (penetration - h_open)
                    k_diag = k_open
            elif self.chatter_stabilization:
                # Hysteresis-gated label (design doc sec2.2) -- identical
                # convention to SurfaceContactConstraint3D's own branch:
                # gate closed means zero force/stiffness; gate open keeps
                # k_diag=k_contact engaged across the whole dead band,
                # f_mag clamped >=0 (bounded discrepancy <= hysteresis_band,
                # sec2.3).
                if not self._gate_active[s_nid]:
                    continue
                f_mag = max(0.0, self._lam[s_nid] + self.k_contact * penetration)
                k_diag = self.k_contact
            elif self.augmented_lagrange:
                # Identical branch to SurfaceContactConstraint3D's own
                # augmented path: `lam` frozen for this call, `k_diag =
                # k_contact` (design doc sec1.3 -- the per-node
                # residual/tangent formula doesn't change, only WHEN
                # `self._lam` gets written, which is solve_step()'s job).
                f_mag = self._lam[s_nid] + self.k_contact * penetration
                if f_mag <= 0.0:
                    continue
                k_diag = self.k_contact
            else:
                f_mag, k_diag = self.law.evaluate(penetration)
                if f_mag <= 0.0:
                    continue

            n_active += 1
            if penetration > max_penetration:
                max_penetration = penetration
            total_normal_force_mag += f_mag
            # Friction reads p_i = f_mag exactly as computed above --
            # this class has no c_stab-style perturbation, so unlike the
            # rigid-plane sibling no separate capture point is needed
            # (design doc dev_log/contact_friction_precise_design_
            # 20260916.md sec1).
            q = [self.coords[i] + u[3 * i: 3 * i + 3] for i in m_idx]
            f_t_3d, Ktt, _is_slip, _s1, _s2, _s_mag, _t1, _t2 = self._friction_force_and_tangent(
                s_nid, self.coords[s_idx] + u[3 * s_idx: 3 * s_idx + 3], q, n, f_mag
            )

            # Weights w_a: +1 for the slave, -N_a for each master vertex --
            # exactly SurfaceTieConstraint3D's own `weights` pattern
            # (`[1, -N1, -N2, -N3, -N4]`), reused here for a ONE-SIDED law
            # instead of a two-sided bonded spring.
            node_indices = [s_idx] + m_idx
            weights = np.array([1.0, -N1, -N2, -N3, -N4], dtype=np.float64)

            # Sign convention (matches SurfaceContactConstraint3D's own
            # 2026-09-13 fix): f_int_global uses f_int = -F_physical, and
            # unlike a bonded tie's two-sided spring (whose own sign
            # cancels this automatically -- see pressure_overclosure.py's
            # HardLaw docstring history), a one-sided contact push needs
            # the explicit negation. Physical force on the slave is
            # f_mag*n (pushes it away from the master); on each master
            # vertex, the Newton's-third-law reaction distributed by
            # w_a = -N_a. `f_node_a = -w_a * f_mag * n` for every node in
            # the stencil (slave included, w_slave=+1) applies that
            # negation uniformly.
            #
            # Friction is DIFFERENT here, and this is a real fix, not a
            # restatement (dev_log/contact_friction_implementation_
            # 20260916.md; the design doc's own sec7.2 pseudocode, and
            # this file's first cut of it, both had this backwards and
            # failed a real two-cube sliding-block solve, root-caused by
            # finite difference): a STICK/SLIP tangential spring is
            # TWO-SIDED, exactly like a bonded tie (works symmetrically
            # for +s and -s), NOT one-sided like normal contact -- so
            # `f_t_3d`'s OWN sign already benefits from the tie-like
            # automatic cancellation the comment above says the normal
            # term does NOT get. Putting `f_t_3d` through the SAME
            # `-w_a*(...)` pattern as the normal term (i.e. `+f_t_3d`
            # inside f_vec, as an earlier version of this code did) is
            # therefore off by a sign for BOTH the slave and every master
            # vertex; subtracting it (`f_mag*n - f_t_3d`) reproduces the
            # correct per-node result derived directly from Newton's
            # third law: `f_int_slave = -f_mag*n + f_t_3d`, `f_int_
            # master_a = +N_a*f_mag*n - N_a*f_t_3d` -- verified by finite
            # difference against the ALSO-unchanged Ktt stiffness below
            # (which was already correctly signed; only the force needed
            # the fix, since Ktt's own sign already matched a `+w_a*f_t_3d`
            # per-node contribution, which is exactly what `-w_a*(f_mag*n
            # - f_t_3d)` produces).
            f_vec = f_mag * n - f_t_3d
            for a, idx_a in enumerate(node_indices):
                w_a = weights[a]
                f_contact[3 * idx_a: 3 * idx_a + 3] += -w_a * f_vec

            # Default: exact normal block (_consistent_pair_stiffness).
            # Legacy (consistent_tangent=False):
            # K[a][b] = w_a * w_b * (k_diag*outer(n,n) + Ktt) --
            # the SAME rank-1-per-pair structure
            # SurfaceContactConstraint3D's own slave-only 3x3 block uses
            # (`k_diag * normal[d1]*normal[d2]`), generalized to the
            # 5-node stencil the same way SurfaceTieConstraint3D's own
            # `_build_sparse_indices` generalizes its isotropic stiffness,
            # now carrying the tangential block (Ktt, zero when friction
            # is off) alongside the normal one. `n` is held fixed in this
            # legacy block (not differentiated); `t1, t2` inside Ktt are
            # likewise frozen for this call (recomputed fresh next call,
            # never differentiated within one call -- same convention).
            if self.consistent_tangent:
                K_pair = self._consistent_pair_stiffness(
                    s_nid, self.coords[s_idx] + u[3 * s_idx: 3 * s_idx + 3], q, n, xi, eta, f_mag, k_diag, Ktt
                )
                dofs = np.array([3 * i + d for i in node_indices for d in range(3)], dtype=np.int64)
                rows.extend(np.repeat(dofs, 15).tolist())
                cols.extend(np.tile(dofs, 15).tolist())
                data.extend(K_pair.ravel().tolist())
                continue
            K_local = k_diag * np.outer(n, n) + Ktt
            for a, idx_a in enumerate(node_indices):
                w_a = weights[a]
                for b, idx_b in enumerate(node_indices):
                    w_b = weights[b]
                    wab = w_a * w_b
                    for d1 in range(3):
                        for d2 in range(3):
                            rows.append(3 * idx_a + d1)
                            cols.append(3 * idx_b + d2)
                            data.append(wab * K_local[d1, d2])

        stats = {
            "max_penetration": max_penetration,
            "n_active": n_active,
            "n_candidates": len(self.slave_node_ids),
            "n_pairs": len(self.pairs),
            "total_normal_force": total_normal_force_mag,
        }

        return (
            f_contact,
            (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(data, dtype=np.float64)),
            stats,
        )

    def _normal_pressure(self, u: np.ndarray, s_nid: int, m_face, xi: float, eta: float, sign: float):
        """Recompute (xs, q, n, p_i) for one pair -- the SAME normal
        pressure assemble()'s active branch would use for this pair
        (design doc dev_log/contact_friction_precise_design_20260916.md
        sec1). Reused by get_friction_state()/commit_friction_state() so
        they agree with assemble() exactly. p_i=0.0 means inactive."""
        gap, n, _N, s_idx, m_idx = self._current_gap_and_normal(u, s_nid, m_face, xi, eta, sign)
        penetration = -gap
        xs = self.coords[s_idx] + u[3 * s_idx: 3 * s_idx + 3]
        q = [self.coords[i] + u[3 * i: 3 * i + 3] for i in m_idx]
        if self.chatter_stabilization:
            if not self._gate_active[s_nid]:
                return xs, q, n, 0.0
            p_i = max(0.0, self._lam[s_nid] + self.k_contact * penetration)
        elif self.augmented_lagrange:
            p_i = self._lam[s_nid] + self.k_contact * penetration
            if p_i <= 0.0:
                return xs, q, n, 0.0
        else:
            p_i, _ = self.law.evaluate(penetration)
            if p_i <= 0.0:
                return xs, q, n, 0.0
        return xs, q, n, p_i

    def get_friction_state(self, u: np.ndarray) -> frozenset:
        """Pure query (no side effects), mirrors get_active_set()'s own
        convention: the set of slave node ids CURRENTLY SLIPPING (design
        doc sec6). mu<=0.0 always returns frozenset()."""
        if self.mu <= 0.0:
            return frozenset()
        slipping = set()
        for s_nid, m_face, xi, eta, sign in self.pairs:
            xs, q, n, p_i = self._normal_pressure(u, s_nid, m_face, xi, eta, sign)
            if p_i <= 0.0:
                continue
            _f_t, _Ktt, is_slip, _s1, _s2, _s_mag, _t1, _t2 = self._friction_force_and_tangent(s_nid, xs, q, n, p_i)
            if is_slip:
                slipping.add(s_nid)
        return frozenset(slipping)

    def commit_friction_state(self, u: np.ndarray) -> None:
        """Called ONLY when a step has actually converged (design doc
        sec5 -- see SurfaceContactConstraint3D.commit_friction_state()'s
        own docstring for the full rationale, identical here). Advances
        each currently-slipping pair's `(xi_anchor, eta_anchor)` via a
        radial return (sec5.3): the NEW anchor is the (xi, eta)
        re-projection of `xs - s_crit*shat` onto the CURRENT master face,
        reusing `_project_point_to_quad4` verbatim -- the convected-
        anchor analogue of shrinking the anchor-to-slave vector down to
        exactly the elastic length. No-op if friction is disabled."""
        if self.mu <= 0.0:
            return
        for s_nid, m_face, xi, eta, sign in self.pairs:
            xs, q, n, p_i = self._normal_pressure(u, s_nid, m_face, xi, eta, sign)
            if p_i <= 0.0:
                continue
            _f_t, _Ktt, is_slip, s1, s2, s_mag, t1, t2 = self._friction_force_and_tangent(s_nid, xs, q, n, p_i)
            if not is_slip:
                continue  # STICK all increment -- anchor unchanged
            s_crit = (self.mu * p_i / self.k_t) if self.k_t > 0.0 else 0.0
            shat_3d = (s1 / s_mag) * t1 + (s2 / s_mag) * t2
            new_anchor_point = xs - s_crit * shat_3d
            xi_new, eta_new, _dist, _proj = _project_point_to_quad4(new_anchor_point, q[0], q[1], q[2], q[3])
            self._anchor_xi[s_nid] = xi_new
            self._anchor_eta[s_nid] = eta_new

    def update_augmented_multipliers(self, u: np.ndarray, omega: float = 1.0) -> Dict[str, float]:
        """Near-verbatim port of SurfaceContactConstraint3D's own method
        of the same name (design doc dev_log/contact_pdass_precise_design_20260915.md
        sec3, item4) -- same fixed-point map, `lam_new = lam_old +
        omega*(max(0, lam_old + k_contact*(-gap)) - lam_old)`, substituting
        this class's own `_current_gap_and_normal()` for the rigid
        sibling's direct dot-product gap. Called by DynamicSolver3D.solve_step()
        once per ACCEPTED Newton iterate (not from assemble(), which runs
        multiple times per iteration against trial states -- see
        solve_step()'s own docstring for why)."""
        if not self.augmented_lagrange:
            return {"max_penetration": 0.0, "max_lambda_change": 0.0}

        max_penetration = 0.0
        max_lambda_change = 0.0
        for s_nid, m_face, xi, eta, sign in self.pairs:
            gap, _, _, _, _ = self._current_gap_and_normal(u, s_nid, m_face, xi, eta, sign)
            old_lam = self._lam[s_nid]
            p_target = max(0.0, old_lam + self.k_contact * (-gap))
            p = old_lam + omega * (p_target - old_lam)
            self._lam[s_nid] = p
            max_lambda_change = max(max_lambda_change, abs(p - old_lam))
            if p > 0.0 and -gap > max_penetration:
                max_penetration = -gap
        return {"max_penetration": max_penetration, "max_lambda_change": max_lambda_change}
