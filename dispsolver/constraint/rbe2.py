"""
rbe2.py
=======
Rigid Body Element (RBE2) constraint for plane strain 2D hinge.

Formulation
-----------
Couples m slave nodes to a master node (the hinge rotation centre) via
Lagrange multipliers.  The constraint equations enforce that each slave
node lies at a fixed offset from the master node, rotated by a shared
unknown angle θ:

    g_i(u, θ) = x_si(u_si) − x_m(u_m) − R(θ)·(X_si − X_m) = 0

where R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]].  The resulting KKT saddle-point
system has 2·m multiplier equations and 1 extra primal DOF (θ), giving a
theoretically rigid hinge connection.

The RBE2 name comes from MSC Nastran (Rigid Body Element type 2), where
a single "master" node defines the motion of any number of "slave" nodes.
Unlike MPC (multipoint constraint) approaches that linearise at each step,
this formulation retains the exact non-linear rotation, enabling large
hinge angles (0→90° in the folding simulation) without geometric drift.

Implementation notes
--------------------
- The rotation angle θ is stored in u_extra as an extra primal unknown.
- The tangent stiffness contribution includes geometric terms from the
  linearisation of R(θ) w.r.t. u and θ.
- Assemble() returns the constraint vector g and its Jacobian
  [∂g/∂u, ∂g/∂θ], which are assembled into the KKT system.

References
----------
- MSC Nastran 2021 "Rigid Body Elements" documentation.
- Felippa, C.A. (2004) "Introduction to Finite Element Methods",
  Ch. 28: Multi-Freedom Constraints.
"""

import numpy as np
from .base import BaseConstraint

class RBE2HingeConstraint(BaseConstraint):
    def __init__(self, mesh, master_id: int, slave_ids: list[int], extra_primal_offset: int):
        self.mesh = mesh
        self.master_id = master_id
        self.slave_ids = slave_ids
        
        self.coords = mesh.nodes_array()
        self.nid_to_idx = mesh.node_id_to_index()
        
        self.master_idx = self.nid_to_idx[self.master_id]
        self.slave_indices = [self.nid_to_idx[sid] for sid in self.slave_ids]
        
        self.extra_primal_offset = extra_primal_offset

    def n_multipliers(self) -> int:
        return 2 * len(self.slave_ids)

    def n_extra_primal(self) -> int:
        return 1  # 1 rotation theta

    def extra_geometric_stiffness(self, u_extra: np.ndarray, lam: np.ndarray) -> tuple:
        """Rotation-DOF geometric stiffness  K_θθ = Σ_i λ_i · ∂²g_i/∂θ².

        The rigid-rotation constraint g is nonlinear in θ (sin/cos), so the
        consistent KKT tangent needs the second-derivative term λ·∂²g/∂θ² on the
        (θ,θ) diagonal of the extra-primal block. The solver previously used only
        a 1e-12 regularization there, leaving the global tangent inconsistent as
        soon as θ≠0 — Newton then stalls with a constant displacement correction
        (no quadratic convergence). With:
            ∂g_x/∂θ = sinθ·dx + cosθ·dy ,  ∂²g_x/∂θ² = cosθ·dx − sinθ·dy
            ∂g_y/∂θ = −cosθ·dx + sinθ·dy , ∂²g_y/∂θ² = sinθ·dx + cosθ·dy
        and  J(θ,θ) = Σ λ·∂(∂g/∂θ)/∂θ.

        Parameters
        ----------
        u_extra : full extra-primal vector (this constraint reads its offset).
        lam     : multiplier slice for THIS constraint, length 2*n_slaves.
        """
        theta = float(u_extra[self.extra_primal_offset])
        cost = np.cos(theta)
        sint = np.sin(theta)
        x_m, y_m = self.coords[self.master_idx]
        k = 0.0
        for i, s_idx in enumerate(self.slave_indices):
            x_s, y_s = self.coords[s_idx]
            dx = x_s - x_m
            dy = y_s - y_m
            k += lam[2 * i]     * (cost * dx - sint * dy)
            k += lam[2 * i + 1] * (sint * dx + cost * dy)
        return self.extra_primal_offset, float(k)

    def assemble(self, u: np.ndarray, u_extra: np.ndarray):
        n_slaves = len(self.slave_indices)
        
        row_u, col_u, val_u = [], [], []
        row_ext, col_ext, val_ext = [], [], []
        g = np.zeros(2 * n_slaves, dtype=np.float64)
        
        x_m, y_m = self.coords[self.master_idx]
        dof_mx = self.master_idx * 2
        dof_my = self.master_idx * 2 + 1
        
        theta = float(u_extra[self.extra_primal_offset])
        cost = np.cos(theta)
        sint = np.sin(theta)
        
        ux_m = u[dof_mx]
        uy_m = u[dof_my]
        
        for i, s_idx in enumerate(self.slave_indices):
            x_s, y_s = self.coords[s_idx]
            dof_sx = s_idx * 2
            dof_sy = s_idx * 2 + 1
            
            dx = x_s - x_m
            dy = y_s - y_m
            
            eq_x = 2 * i
            eq_y = 2 * i + 1
            
            # Equation X: ux_s - ux_m - ((cost - 1)*dx - sint*dy) = 0
            row_u.extend([eq_x, eq_x])
            col_u.extend([dof_sx, dof_mx])
            val_u.extend([1.0, -1.0])
            
            row_ext.append(eq_x)
            col_ext.append(self.extra_primal_offset)
            val_ext.append(sint * dx + cost * dy) # d/dtheta of -((cost-1)dx - sint*dy)
            
            # Equation Y: uy_s - uy_m - (sint*dx + (cost - 1)*dy) = 0
            row_u.extend([eq_y, eq_y])
            col_u.extend([dof_sy, dof_my])
            val_u.extend([1.0, -1.0])
            
            row_ext.append(eq_y)
            col_ext.append(self.extra_primal_offset)
            val_ext.append(-cost * dx + sint * dy) # d/dtheta of -(sint*dx + (cost-1)dy)
            
            # Gap
            ux_s = u[dof_sx]
            uy_s = u[dof_sy]
            
            g[eq_x] = ux_s - ux_m - ((cost - 1.0) * dx - sint * dy)
            g[eq_y] = uy_s - uy_m - (sint * dx + (cost - 1.0) * dy)
            
        return (np.array(row_u, dtype=np.int32), np.array(col_u, dtype=np.int32), np.array(val_u, dtype=np.float64),
                np.array(row_ext, dtype=np.int32), np.array(col_ext, dtype=np.int32), np.array(val_ext, dtype=np.float64),
                g)
