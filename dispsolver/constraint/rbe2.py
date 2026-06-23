"""
rbe2.py
=======
Rigid Body Element (RBE2) constraint for plane strain 2D.
Couples a set of slave nodes to a master node, allowing only a rigid body rotation
about the Z axis.
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
