"""
tie.py
======
Tie constraint connecting a set of slave nodes to a set of master nodes.
Each slave node is tied to the nearest master node, enforcing identical displacements.
"""

import numpy as np
from scipy.spatial import KDTree
from .base import BaseConstraint

class TieConstraint(BaseConstraint):
    def __init__(self, mesh, master_set_name: str, slave_set_name: str, tol: float = 1e-6):
        self.mesh = mesh
        
        master_set = mesh.get_nodeset(master_set_name)
        slave_set = mesh.get_nodeset(slave_set_name)
        
        self.coords = mesh.nodes_array()
        self.nid_to_idx = mesh.node_id_to_index()
        
        master_indices = [self.nid_to_idx[nid] for nid in master_set.node_ids]
        slave_indices = [self.nid_to_idx[nid] for nid in slave_set.node_ids]
        
        # Find nearest master for each slave
        master_coords = self.coords[master_indices]
        tree = KDTree(master_coords)
        
        slave_coords = self.coords[slave_indices]
        dists, indices = tree.query(slave_coords)
        
        self.pairs = []
        for i, s_idx in enumerate(slave_indices):
            if dists[i] < tol:
                m_idx = master_indices[indices[i]]
                self.pairs.append((s_idx, m_idx))
            else:
                import warnings
                warnings.warn(f"Slave node {slave_set.node_ids[i]} has no master node within tolerance {tol}.")

    def n_multipliers(self) -> int:
        return 2 * len(self.pairs)

    def n_extra_primal(self) -> int:
        return 0
        
    def assemble(self, u: np.ndarray, u_extra: np.ndarray):
        n_pairs = len(self.pairs)
        
        row_u, col_u, val_u = [], [], []
        g = np.zeros(2 * n_pairs, dtype=np.float64)
        
        for i, (s_idx, m_idx) in enumerate(self.pairs):
            dof_sx = s_idx * 2
            dof_sy = s_idx * 2 + 1
            dof_mx = m_idx * 2
            dof_my = m_idx * 2 + 1
            
            eq_x = 2 * i
            eq_y = 2 * i + 1
            
            # Equation X: ux_s - ux_m = 0
            row_u.extend([eq_x, eq_x])
            col_u.extend([dof_sx, dof_mx])
            val_u.extend([1.0, -1.0])
            
            # Equation Y: uy_s - uy_m = 0
            row_u.extend([eq_y, eq_y])
            col_u.extend([dof_sy, dof_my])
            val_u.extend([1.0, -1.0])
            
            # Gap
            g[eq_x] = u[dof_sx] - u[dof_mx]
            g[eq_y] = u[dof_sy] - u[dof_my]
            
        return (np.array(row_u, dtype=np.int32), np.array(col_u, dtype=np.int32), np.array(val_u, dtype=np.float64),
                np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float64),
                g)
