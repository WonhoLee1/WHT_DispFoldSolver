"""
rigid_segment_contact2d.py
==========================
Frictionless penalty contact of 2D slave nodes against a rigid straight SEGMENT
that rotates about a fixed pivot -- the top face of a rigidly driven plate.

Why it exists: in the ex12 fold the display rests on each plate over an untied
strip next to the plate's inner edge (`tie_attach_start`). Without contact there
the display passes through the plate (measured 2026-10-09: 1.35 mm inside a
0.63 mm thick display at 41 deg/side), which is what the fold was failing on.
`SurfaceContactConstraint2D` covers only an infinite, translating line; this one
covers a finite segment whose orientation follows the plate's prescribed angle.

Kinematics (plate frame = reference frame rotated by theta about the pivot):
    x'  = R(-theta) (x - p) + p                  (current node, pulled back)
    g   = (x' - p) . n0                          (n0 = reference outward normal)
    active if g < 0 and s_min <= (x' - p) . t0 <= s_max   (t0 = segment tangent)
Penalty energy 0.5 k <-g>^2 gives
    f_int += k g n,   K += k n n^T,   n = R(theta) n0   (current normal)
The plate is rigid and its motion prescribed, so n does not depend on the slave
displacement and there is no geometric stiffness term; the reaction on the
plate is not fed back (its motion is prescribed, exactly as for the kinematic tie).

Interface: the legacy DynamicSolver `penalty_constraints` protocol --
`apply_penalty(u, f_int, K_lil_or_None)` -- plus `set_rotation(theta)` and
`stats(u)` for monitoring.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


class RotatingRigidSegmentContact2D:
    """Unilateral penalty contact against a rigid segment rotating about `pivot`.

    Parameters
    ----------
    slave_node_ids : node ids that may touch the segment
    nid_to_idx : node id -> solver node index
    coords : (n_nodes, 2) reference coordinates
    pivot : (2,) rotation centre of the plate (fixed in space)
    tangent : (2,) unit direction of the segment in the reference configuration
    normal : (2,) unit outward normal of the plate face in the reference
        configuration (points from the plate toward the slave body)
    s_range : (s_min, s_max) extent of the segment, measured from `pivot` along
        `tangent` in the reference configuration
    penalty_stiffness : k [force/length]
    stiffness_gap_tol : the tangent (not the force) also carries k for a node that is open by
        less than this gap. The force stays exactly unilateral -- k*g only for g < 0, so the
        converged state is unchanged -- but a node that one Newton correction pushed just clear
        of the face still feels the face in the next one, instead of being free to swing back
        through it. Against a very compliant body (the ex12 display deflects ~40 mm per N
        across its free span) the plain active-set tangent alternates open/closed and diverges.
    """

    def __init__(self, slave_node_ids: Iterable[int], nid_to_idx: Dict[int, int], coords: np.ndarray,
                 pivot, tangent, normal, s_range, penalty_stiffness: float = 1.0e4,
                 name: str = "RigidSegmentContact2D", stiffness_gap_tol: float = 0.0,
                 softening_gap: float = 0.0, fillet_radius: float = 0.0):
        self.name = name
        self.nid_to_idx = dict(nid_to_idx)
        self.slave_node_ids = [int(n) for n in slave_node_ids if int(n) in self.nid_to_idx]
        self._idx = np.array([self.nid_to_idx[n] for n in self.slave_node_ids], dtype=np.int64)
        self._X = np.asarray(coords, dtype=np.float64)[self._idx, :2].copy()
        self.pivot = np.asarray(pivot, dtype=np.float64)[:2].copy()
        t0 = np.asarray(tangent, dtype=np.float64)[:2]
        n0 = np.asarray(normal, dtype=np.float64)[:2]
        self.t0 = t0 / np.linalg.norm(t0)
        self.n0 = n0 / np.linalg.norm(n0)
        if abs(float(self.t0 @ self.n0)) > 1e-9:
            raise ValueError(f"{name}: tangent and normal must be orthogonal")
        self.s_min, self.s_max = float(min(s_range)), float(max(s_range))
        self.penalty_stiffness = float(penalty_stiffness)
        self.stiffness_gap_tol = float(stiffness_gap_tol)
        self.softening_gap = float(softening_gap)   # > 0: C1 softened pressure-overclosure (see _pressure)
        # > 0: the face's inner end (s_min, the plate edge next to the hinge) is rounded with this
        # radius, as Abaqus' FILLET RADIUS on an analytical rigid surface. A sharp segment end
        # switches contact on/off as a node slides across it (ex12 2026-10-10: residual flipping
        # 100 N <-> 3e-3 N at the edge node); the fillet makes the normal continuous there.
        self.fillet_radius = float(fillet_radius)
        self.theta = 0.0
        self.n_active = 0
        # DOFs the contact acted on in the last apply_penalty (slave nodes in the contact zone);
        # DynamicSolver reads it to scope CONVERGENCE CHECK to the contact DOFs.
        self.last_contact_dofs = np.zeros(0, dtype=np.int64)

    # -- prescribed plate motion ---------------------------------------
    def set_rotation(self, theta: float) -> None:
        """Plate angle (radians, counter-clockwise positive) for the coming increment."""
        self.theta = float(theta)

    def carry_contacting_nodes(self, u: np.ndarray, theta_old: float, theta_new: float) -> int:
        """Carry contacting slave nodes with the rotating rigid plate as Newton predictor.

        Nodes in contact at theta_old are rotated about the plate pivot by d_theta = theta_new - theta_old,
        updating their displacements in `u` in-place. Returns the count of carried nodes.
        """
        d_theta = float(theta_new - theta_old)
        if abs(d_theta) < 1e-12:
            return 0

        # Temporarily evaluate contact geometry at committed theta_old
        saved_theta = self.theta
        self.theta = float(theta_old)
        try:
            if self.softening_gap > 0.0 or self.fillet_radius > 0.0:
                g, _, _, on = self._geometry(u)
                contacting = on & (g < 0.0)
            else:
                g, active = self._gaps(u)
                contacting = active & (g < 0.0)
        finally:
            self.theta = saved_theta

        if not np.any(contacting):
            return 0

        # Rigid rotation about pivot: r_new = R(d_theta) * r_old
        # u_new = u_old + (R(d_theta) - I) * (X + u_old - pivot)
        c, s = np.cos(d_theta), np.sin(d_theta)
        R_minus_I = np.array([[c - 1.0, -s],
                              [s, c - 1.0]], dtype=np.float64)

        for j in np.flatnonzero(contacting):
            i = int(self._idx[j])
            xi = self._X[j, 0] + u[2 * i] - self.pivot[0]
            yi = self._X[j, 1] + u[2 * i + 1] - self.pivot[1]
            du_x = R_minus_I[0, 0] * xi + R_minus_I[0, 1] * yi
            du_y = R_minus_I[1, 0] * xi + R_minus_I[1, 1] * yi
            u[2 * i] += du_x
            u[2 * i + 1] += du_y

        return int(np.count_nonzero(contacting))

    def _rot(self, ang: float) -> np.ndarray:
        c, s = np.cos(ang), np.sin(ang)
        return np.array([[c, -s], [s, c]])

    def _gaps(self, u: np.ndarray, gap_tol: float = 0.0):
        x = self._X + np.column_stack([u[2 * self._idx], u[2 * self._idx + 1]])
        d = (x - self.pivot) @ self._rot(-self.theta).T      # pulled back into the plate frame
        g = d @ self.n0
        s = d @ self.t0
        active = (g < gap_tol) & (s >= self.s_min) & (s <= self.s_max)
        return g, active

    def _pressure(self, g: np.ndarray):
        """Contact pressure p(g) >= 0 and dp/dg.

        Hard (softening_gap = 0): p = k <-g>, the plain penalty.
        Softened (softening_gap = g0 > 0), C1 quadratic onset over a small overclosure g0:
            p = 0                         g >= 0
            p = k g^2 / (2 g0)            -g0 < g < 0
            p = k (-g - g0/2)             g <= -g0
        p and dp/dg are zero at touch, so a surface that starts exactly touching (the display
        on the plate) carries no initial load; the onset ramp removes the open/closed jump.
        Force and tangent stay consistent everywhere, so a node near the face neither
        chatters open/closed (hard law) nor stalls (a tangent without force).
        """
        k = self.penalty_stiffness
        g0 = self.softening_gap
        p = np.zeros_like(g)
        dp = np.zeros_like(g)
        if g0 <= 0.0:
            pen = g < 0.0
            p[pen] = -k * g[pen]
            dp[pen] = -k
            return p, dp
        band = (g < 0.0) & (g > -g0)
        p[band] = k * g[band] ** 2 / (2.0 * g0)
        dp[band] = k * g[band] / g0
        pen = g <= -g0
        p[pen] = k * (-g[pen] - 0.5 * g0)
        dp[pen] = -k
        return p, dp

    def _geometry(self, u: np.ndarray):
        """Per slave node: gap g, unit normal (current frame), curvature 1/rho of the face at the
        contact point (0 on the flat part), and whether the node projects onto the face."""
        x = self._X + np.column_stack([u[2 * self._idx], u[2 * self._idx + 1]])
        d = (x - self.pivot) @ self._rot(-self.theta).T      # plate frame, origin at the pivot
        g = d @ self.n0
        s = d @ self.t0
        nloc = np.tile(self.n0, (len(g), 1))
        curv = np.zeros(len(g))
        r = self.fillet_radius
        on = (s >= self.s_min) & (s <= self.s_max)
        if r > 0.0:
            on = (s >= self.s_min + r) & (s <= self.s_max)
            c = (self.s_min + r) * self.t0 - r * self.n0          # fillet centre (plate frame)
            v = d - c
            dist = np.linalg.norm(v, axis=1)
            arc = (s < self.s_min + r) & ((v @ self.t0) <= 0.0) & ((v @ self.n0) >= 0.0) & (dist > 1e-12)
            g = np.where(arc, dist - r, g)
            nloc[arc] = v[arc] / dist[arc, None]
            curv[arc] = 1.0 / dist[arc]
            on = on | arc
        R = self._rot(self.theta)
        n = nloc @ R.T
        return g, n, curv, on

    def _record_contact_dofs(self, near: np.ndarray) -> None:
        idx = self._idx[np.asarray(near, dtype=bool)]
        self.last_contact_dofs = np.concatenate([2 * idx, 2 * idx + 1]) if idx.size else np.zeros(0, dtype=np.int64)

    # -- DynamicSolver penalty-constraint protocol ----------------------
    def apply_penalty(self, u: np.ndarray, f_int: np.ndarray, K=None) -> None:
        if self.softening_gap > 0.0 or self.fillet_radius > 0.0:
            g, n, curv, on = self._geometry(u)
            near = on & (g < 0.0)
            p, dp = self._pressure(g)
            self.n_active = int(np.count_nonzero(near & (p > 0.0)))
            self._record_contact_dofs(near)
            if not np.any(near):
                return
            I2 = np.eye(2)
            for j in np.flatnonzero(near):
                i = int(self._idx[j])
                nj = n[j]
                # energy W(g), dW/dg = -p:  f_int = -p n;  K = -dp n n^T - p dn/dx,  dn/dx = curv (I - n n^T)
                f_int[2 * i] -= p[j] * nj[0]
                f_int[2 * i + 1] -= p[j] * nj[1]
                if K is not None:
                    Kj = -dp[j] * np.outer(nj, nj) - p[j] * curv[j] * (I2 - np.outer(nj, nj))
                    for a in range(2):
                        for b in range(2):
                            K[2 * i + a, 2 * i + b] += Kj[a, b]
            return
        g, near = self._gaps(u, max(self.stiffness_gap_tol, 0.0))
        active = near & (g < 0.0)
        self.n_active = int(np.count_nonzero(active))
        self._record_contact_dofs(near)
        if not np.any(near):
            return
        n = self._rot(self.theta) @ self.n0
        k = self.penalty_stiffness
        nnT = k * np.outer(n, n)
        for j in np.flatnonzero(active):
            i = int(self._idx[j])
            f_int[2 * i] += k * g[j] * n[0]
            f_int[2 * i + 1] += k * g[j] * n[1]
        if K is not None:
            for j in np.flatnonzero(near):
                i = int(self._idx[j])
                for a in range(2):
                    for b in range(2):
                        K[2 * i + a, 2 * i + b] += nnT[a, b]

    def stats(self, u: np.ndarray) -> Dict[str, float]:
        if self.softening_gap > 0.0 or self.fillet_radius > 0.0:
            g, _, _, on = self._geometry(u)
            active = on & (g < 0.0)
        else:
            g, active = self._gaps(u)
        pen = float(-g[active].min()) if np.any(active) else 0.0
        return {"n_active": int(np.count_nonzero(active)), "max_penetration": pen}
