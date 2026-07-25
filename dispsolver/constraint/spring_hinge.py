"""
spring_hinge.py
===============
Penalty (spring) hinge drive — a Symmetric-Positive-Definite alternative to the
Lagrange-multiplier RBE2 hinge constraint.

Motivation
----------
`RBE2HingeConstraint` enforces a rigid rotation of the slave nodes about an
offset master point with Lagrange multipliers. That produces an *indefinite*
KKT saddle system whose tangent contains nonlinear rotation coupling terms
(∂g/∂θ, ∂²g/∂θ²·λ, master-translation cross terms); a missing/incorrect term
there makes the GLOBAL Newton tangent inconsistent (full Newton diverges).

Because the hinge rotation θ is *prescribed* (driven by an amplitude curve), the
slave target position is a KNOWN function of time:

    target_s(t) = x_m + R(θ(t))·(x_s − x_m)        (master fixed)
    u_target_s(t) = target_s − x_s = (R(θ)−I)·(x_s − x_m)

So instead of a hard constraint we attach each slave node to its (moving) target
with a stiff linear spring of penalty stiffness k:

    E   = ½ k Σ_s |u_s − u_target_s|²
    f_s = ∂E/∂u_s = k (u_s − u_target_s)            (internal restoring force)
    K   = ∂²E/∂u_s² = k I                            (diagonal — trivially consistent)

This adds NO extra DOFs and NO multipliers, keeps the system SPD (valid line
search merit, far better conditioning), and the tangent is a plain diagonal
k·I so it cannot be inconsistent. The flap is rigid only up to the spring
stretch ~f/k, which is negligible for k ≫ structural stiffness.

The driver sets `self.theta` to the current prescribed angle each step before
the solve (the master point is fixed geometry, read from the mesh at init).

Optional viscous damping (c) is left as a hook; it requires nodal velocities,
which the penalty interface does not currently receive, so it is inactive here.

References
----------
- Wriggers, P. (2006) "Computational Contact Mechanics", 2nd ed. — penalty
  regularization of kinematic constraints.
- Belytschko, Liu & Moran (2000) "Nonlinear Finite Elements", Ch. 6 — penalty
  method for constraints (SPD, consistent diagonal tangent).
"""

from __future__ import annotations

import numpy as np


class SpringHingeConstraint:
    """Penalty spring drive that pulls slave nodes onto a rigid-rotated target.

    Parameters
    ----------
    mesh      : the mesh (for node coordinates / index map).
    master_id : hinge centre node id (fixed geometric point about which the
                slaves rotate).
    slave_ids : nodes pulled onto the rigid-rotated target.
    k_spring  : penalty stiffness [force/length]. Choose ≫ structural stiffness
                (e.g. tens× the local element stiffness) so the flap is
                effectively rigid while the system stays well-conditioned.
    """

    def __init__(self, mesh, master_id: int, slave_ids: list[int], k_spring: float):
        self.mesh = mesh
        self.k = float(k_spring)
        coords = mesh.nodes_array()
        nid_to_idx = mesh.node_id_to_index()
        self.master_idx = nid_to_idx[master_id]
        x_m, y_m = coords[self.master_idx]
        # Per-slave dof indices and lever arms (x_s − x_m, y_s − y_m).
        self.slave_dofs = []   # (dof_x, dof_y, dx, dy)
        for sid in slave_ids:
            s_idx = nid_to_idx[sid]
            x_s, y_s = coords[s_idx]
            self.slave_dofs.append((s_idx * 2, s_idx * 2 + 1,
                                    float(x_s - x_m), float(y_s - y_m)))
        self.theta = 0.0       # current prescribed rotation, set by the driver
        self.n_slaves = len(self.slave_dofs)
        # n_active is summed into the solver's "contacts" display; a spring is
        # not a contact, so keep it 0 to avoid the misleading count.
        self.n_active = 0

    # --- this is a pure penalty constraint: no Lagrange DOFs/multipliers ---
    def n_multipliers(self) -> int:
        return 0

    def n_extra_primal(self) -> int:
        return 0

    def max_stretch(self, u: np.ndarray) -> float:
        """Max |u_s − u_target| over slaves = worst constraint violation [length].

        This is the spring stretch f/k; it must be ≪ the relevant length scale
        for the flap to behave as (effectively) rigid. Use it to check whether
        k_spring is large enough.
        """
        cost = np.cos(self.theta)
        sint = np.sin(self.theta)
        worst = 0.0
        for dof_x, dof_y, dx, dy in self.slave_dofs:
            ux_t = (cost - 1.0) * dx - sint * dy
            uy_t = sint * dx + (cost - 1.0) * dy
            ex = u[dof_x] - ux_t
            ey = u[dof_y] - uy_t
            worst = max(worst, (ex * ex + ey * ey) ** 0.5)
        return worst

    def apply_penalty(self, u: np.ndarray, f_int: np.ndarray, K_T) -> None:
        """Add spring restoring force (to f_int) and stiffness k·I (to K_T)."""
        k = self.k
        cost = np.cos(self.theta)
        sint = np.sin(self.theta)
        for dof_x, dof_y, dx, dy in self.slave_dofs:
            # Target displacement = (R(θ) − I)(x_s − x_m), master fixed.
            ux_t = (cost - 1.0) * dx - sint * dy
            uy_t = sint * dx + (cost - 1.0) * dy
            # Internal restoring force  f = k (u_s − u_target)
            f_int[dof_x] += k * (u[dof_x] - ux_t)
            f_int[dof_y] += k * (u[dof_y] - uy_t)
            # Diagonal stiffness  ∂f/∂u_s = k I  (consistent by construction)
            K_T[dof_x, dof_x] += k
            K_T[dof_y, dof_y] += k
