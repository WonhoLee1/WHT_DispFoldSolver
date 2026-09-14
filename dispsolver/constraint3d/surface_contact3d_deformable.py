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

Tangent is a MODIFIED NEWTON approximation, matching this project's own
established precedent (AGENTS.md sec4.4): the local surface normal is
recomputed fresh every call for the FORCE/gap evaluation, but held FROZEN
(not differentiated) when building the tangent stiffness -- i.e. only
d(gap)/du through the shape-function-weighted position term is
differentiated, not d(normal)/du. This costs a couple of extra Newton
iterations near large relative-rotation increments (same tradeoff
AGENTS.md sec4.4 already documents for corotational-frame elements), not
full quadratic convergence -- acceptable because the same line-search/
SDI machinery that already covers modified-Newton element tangents
covers this too.

One-sided complementarity (contact, not a bonded tie) through the SAME
PressureOverclosureLaw seam (dispsolver/constraint3d/pressure_overclosure.py)
rigid-plane contact and its nonlinear/soft laws already use -- HardLaw,
NonlinearPenaltyLaw, and the three soft laws all plug in unchanged.
Augmented Lagrangian is NOT supported here yet (design doc sec B.4
restricts it to HARD contact on a single, well-defined normal direction;
generalizing the Uzawa multiplier update to a moving, shape-function-
weighted 5-node stencil is a real follow-on, not attempted in this pass).
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
    of the local tangent vectors dx/dxi, dx/deta. NOT the direction the
    tangent stiffness differentiates (see module docstring) -- only used
    to evaluate the current gap/force each call."""
    dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi = -0.25 * (1.0 - eta), 0.25 * (1.0 - eta), 0.25 * (1.0 + eta), -0.25 * (1.0 + eta)
    dN1_deta, dN2_deta, dN3_deta, dN4_deta = -0.25 * (1.0 - xi), -0.25 * (1.0 + xi), 0.25 * (1.0 + xi), 0.25 * (1.0 - xi)
    t_xi = dN1_dxi * q1 + dN2_dxi * q2 + dN3_dxi * q3 + dN4_dxi * q4
    t_eta = dN1_deta * q1 + dN2_deta * q2 + dN3_deta * q3 + dN4_deta * q4
    n = np.cross(t_xi, t_eta)
    norm = np.linalg.norm(n)
    if norm < 1e-14:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


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
        name: str = "SURFACE_CONTACT_3D_DEFORMABLE",
    ):
        self.slave_node_ids = list(slave_node_ids)
        self.master_faces = list(master_faces)
        self.nid_to_idx = nid_to_idx
        self.coords = coords
        self.k_contact = float(penalty_stiffness)
        self.law = law if law is not None else HardLaw(self.k_contact)
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
            gap0 = float(np.dot(xs0 - xm_proj0, n_raw))
            sign = 1.0 if gap0 >= 0.0 else -1.0
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

    def get_active_set(self, u: np.ndarray) -> frozenset:
        """Same role as SurfaceContactConstraint3D.get_active_set -- feeds
        DynamicSolver3D's severe-discontinuity-iteration (SDI) handling."""
        active = set()
        for s_nid, m_face, xi, eta, sign in self.pairs:
            gap, _, _, _, _ = self._current_gap_and_normal(u, s_nid, m_face, xi, eta, sign)
            if self.augmented_lagrange:
                p = self._lam[s_nid] + self.k_contact * (-gap)
            else:
                p, _ = self.law.evaluate(-gap)
            if p > 0.0:
                active.add(s_nid)
        return frozenset(active)

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
            if self.augmented_lagrange:
                # Identical branch to SurfaceContactConstraint3D's own
                # augmented path: `lam` frozen for this call, `k_diag =
                # k_contact` (design doc sec1.3 -- the per-node
                # residual/tangent formula doesn't change, only WHEN
                # `self._lam` gets written, which is solve_step()'s job).
                f_mag = self._lam[s_nid] + self.k_contact * penetration
                k_diag = self.k_contact
            else:
                f_mag, k_diag = self.law.evaluate(penetration)
            if f_mag <= 0.0:
                continue

            n_active += 1
            if penetration > max_penetration:
                max_penetration = penetration
            total_normal_force_mag += f_mag

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
            for a, idx_a in enumerate(node_indices):
                w_a = weights[a]
                f_contact[3 * idx_a: 3 * idx_a + 3] += -w_a * f_mag * n

            # Stiffness: K[a][b] = w_a * w_b * k_diag * outer(n, n) --
            # the SAME rank-1-per-pair structure
            # SurfaceContactConstraint3D's own slave-only 3x3 block uses
            # (`k_diag * normal[d1]*normal[d2]`), generalized to the
            # 5-node stencil the same way SurfaceTieConstraint3D's own
            # `_build_sparse_indices` generalizes its isotropic stiffness.
            # `n` is held fixed here (not differentiated) -- the modified-
            # Newton approximation this module's docstring documents.
            for a, idx_a in enumerate(node_indices):
                w_a = weights[a]
                for b, idx_b in enumerate(node_indices):
                    w_b = weights[b]
                    k_ab = k_diag * w_a * w_b
                    for d1 in range(3):
                        for d2 in range(3):
                            rows.append(3 * idx_a + d1)
                            cols.append(3 * idx_b + d2)
                            data.append(k_ab * n[d1] * n[d2])

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
