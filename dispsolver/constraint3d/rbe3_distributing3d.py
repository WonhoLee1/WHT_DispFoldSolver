import numpy as np
from typing import List, Tuple

class RBE3DistributingConstraint3D:
    """Penalty-based RBE3 Distributing Coupling Constraint for 3D solver.
    
    Couples the translation of a Reference Point (RP) to the weighted average
    translation of a set of slave nodes. No rotational DOFs are coupled, as
    continuum solid nodes only have 3 translational DOFs.
    
    Energy: E = 0.5 * k * sum_{d=x,y,z} (u_{RP,d} - sum_i W_i u_{i,d})^2
    """
    def __init__(self, master_id: int, slave_ids: List[int], nid_to_idx: dict, num_dofs: int, penalty_stiffness: float = 1e7, name: str = "RBE3"):
        self.master_id = master_id
        self.slave_ids = slave_ids
        self.nid_to_idx = nid_to_idx
        self.num_dofs = num_dofs
        self.penalty = penalty_stiffness
        self.name = name
        
        self.master_idx = self.nid_to_idx[master_id]
        self.slave_idxs = [self.nid_to_idx[sid] for sid in slave_ids]
        
        # Default uniform weights for RBE3 translation
        self.N = len(self.slave_idxs)
        self.weights = np.ones(self.N, dtype=np.float64) / self.N
        
        # Pre-allocate stiffness patterns
        self._build_stiffness_pattern()
        
    def _build_stiffness_pattern(self):
        """Pre-compute the COO row and column indices for the penalty stiffness matrix."""
        rows = []
        cols = []
        data = []
        
        m_dofs = [self.master_idx * 3 + d for d in range(3)]
        s_dofs_list = [[s_idx * 3 + d for d in range(3)] for s_idx in self.slave_idxs]
        
        k = self.penalty
        
        for d in range(3):
            m_dof = m_dofs[d]
            
            # K_master,master
            rows.append(m_dof)
            cols.append(m_dof)
            data.append(k)
            
            for i, s_idx_i in enumerate(self.slave_idxs):
                s_dof_i = s_dofs_list[i][d]
                w_i = self.weights[i]
                
                # K_master,slave
                rows.append(m_dof)
                cols.append(s_dof_i)
                data.append(-k * w_i)
                
                # K_slave,master
                rows.append(s_dof_i)
                cols.append(m_dof)
                data.append(-k * w_i)
                
                # K_slave,slave
                for j, s_idx_j in enumerate(self.slave_idxs):
                    s_dof_j = s_dofs_list[j][d]
                    w_j = self.weights[j]
                    rows.append(s_dof_i)
                    cols.append(s_dof_j)
                    data.append(k * w_i * w_j)
                    
        self.rows = np.array(rows, dtype=np.int32)
        self.cols = np.array(cols, dtype=np.int32)
        self.data = np.array(data, dtype=np.float64)
        
    def assemble(self, u_vec: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray], bool]:
        """Assemble the penalty internal force vector and stiffness matrix."""
        f_c = np.zeros(self.num_dofs, dtype=np.float64)
        
        m_dofs = [self.master_idx * 3 + d for d in range(3)]
        s_dofs_list = [[s_idx * 3 + d for d in range(3)] for s_idx in self.slave_idxs]
        
        k = self.penalty
        
        for d in range(3):
            m_dof = m_dofs[d]
            u_rp = u_vec[m_dof]
            
            avg_u_slave = 0.0
            for i, s_idx in enumerate(self.slave_idxs):
                s_dof = s_dofs_list[i][d]
                avg_u_slave += self.weights[i] * u_vec[s_dof]
                
            gap = u_rp - avg_u_slave
            
            # f_RP = k * gap
            f_c[m_dof] += k * gap
            
            # f_i = -k * W_i * gap
            for i, s_idx in enumerate(self.slave_idxs):
                s_dof = s_dofs_list[i][d]
                f_c[s_dof] -= k * self.weights[i] * gap
                
        return f_c, (self.rows, self.cols, self.data), False
