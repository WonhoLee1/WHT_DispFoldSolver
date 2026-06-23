"""
penalty_hinge.py
================
Penalty-based hinge constraint for plane strain 2D.

Unlike the Lagrange Multiplier RBE2, this constraint works correctly for
coincident nodes (dx=dy=0) by directly penalizing the difference in
translational DOFs between master and slave nodes.

The hinge allows free relative rotation at the connection point while
constraining translational displacements to match. This is equivalent
to a pin joint.

Penalty Formulation
-------------------
For each slave node:
    g_x = u_sx - u_mx = 0
    g_y = u_sy - u_my = 0

Contribution to internal forces:
    f_penalty = k_pen * [g_x, g_y, ...]

Contribution to stiffness:
    K_penalty[sx,sx] += k_pen,  K_penalty[sx,mx] -= k_pen, etc.
"""

import numpy as np
from ..mesh import Mesh


class PenaltyHingeConstraint:
    """Penalty-based hinge (pin joint) constraint.

    Enforces translational DOF matching between master and slave nodes
    using a penalty stiffness. Unlike RBE2HingeConstraint, this does not
    introduce Lagrange multipliers or extra primal DOFs, and works
    correctly for coincident nodes.

    Parameters
    ----------
    mesh : Mesh
        The finite element mesh.
    master_id : int
        Node ID of the master (reference) node.
    slave_ids : list[int]
        Node IDs of the slave nodes to be constrained.
    k_penalty : float
        Penalty stiffness. Should be large enough to enforce the constraint
        but not so large as to cause ill-conditioning. Typical value:
        10~100x the material stiffness * characteristic length.
    """

    def __init__(self, mesh: Mesh, master_id: int, slave_ids: list[int],
                 k_penalty: float = 1e9):
        self.mesh = mesh
        self.master_id = master_id
        self.slave_ids = slave_ids
        self.k_penalty = float(k_penalty)

        self.nid_to_idx = mesh.node_id_to_index()
        self.master_idx = self.nid_to_idx[self.master_id]
        self.slave_indices = [self.nid_to_idx[sid] for sid in self.slave_ids]

    def n_multipliers(self) -> int:
        """No Lagrange multipliers — penalty method."""
        return 0

    def n_extra_primal(self) -> int:
        """No extra primal DOFs — penalty method."""
        return 0

    def apply_penalty(self, u: np.ndarray, f_int: np.ndarray, K_T) -> None:
        """Apply penalty contributions to internal force and tangent stiffness.

        Parameters
        ----------
        u : (n_dofs,) array
            Current displacement vector.
        f_int : (n_dofs,) array
            Internal force vector (modified in-place).
        K_T : sparse matrix (lil_matrix or csr_matrix)
            Tangent stiffness matrix (modified in-place).
        """
        k = self.k_penalty
        dof_mx = self.master_idx * 2
        dof_my = self.master_idx * 2 + 1

        for s_idx in self.slave_indices:
            dof_sx = s_idx * 2
            dof_sy = s_idx * 2 + 1

            # Gap in x and y
            gx = u[dof_sx] - u[dof_mx]
            gy = u[dof_sy] - u[dof_my]

            # Force contributions (f_penalty = k * gap)
            f_int[dof_sx] += k * gx
            f_int[dof_mx] -= k * gx
            f_int[dof_sy] += k * gy
            f_int[dof_my] -= k * gy

            # Stiffness contributions (symmetric)
            # K[sx, sx] += k, K[sx, mx] -= k, K[mx, sx] -= k, K[mx, mx] += k
            K_T[dof_sx, dof_sx] += k
            K_T[dof_sx, dof_mx] -= k
            K_T[dof_mx, dof_sx] -= k
            K_T[dof_mx, dof_mx] += k

            K_T[dof_sy, dof_sy] += k
            K_T[dof_sy, dof_my] -= k
            K_T[dof_my, dof_sy] -= k
            K_T[dof_my, dof_my] += k
