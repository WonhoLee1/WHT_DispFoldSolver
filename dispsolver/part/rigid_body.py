"""
rigid_body.py
=============
Rigid Body Part representation with dual-mode formulation support
(Kinematic Condensation vs. Penalty Constraint).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from ..constraint.rbe2_condensed import KinematicRBE2Constraint


class RigidBodyPart:
    """Rigid Body Part container managing a Master Reference Point (RP) and Slave nodes.

    Parameters
    ----------
    name : str
        Name identifier for the rigid part (e.g., "PLATE_LEFT").
    master_id : int
        Node ID of the master reference point.
    slave_ids : list of int
        Node IDs of all slave nodes attached to this rigid body.
    master_coord : ndarray of shape (2,)
        Initial coordinates of the master RP.
    slave_coords : ndarray of shape (N, 2)
        Initial coordinates of the N slave nodes.
    mode : str
        Formulation mode: "condensation" (default, kinematic SPD reduction)
        or "constraint" (Lagrange multiplier / Penalty formulation).
    penalty_stiffness : float
        Penalty stiffness if mode="constraint".
    """

    def __init__(
        self,
        name: str,
        master_id: int,
        slave_ids: List[int],
        master_coord: np.ndarray,
        slave_coords: np.ndarray,
        mode: str = "condensation",
        penalty_stiffness: float = 1e8,
        mesh=None,
    ):
        self.name = name
        self.master_id = master_id
        self.slave_ids = list(slave_ids)
        self.master_coord = np.asarray(master_coord, dtype=np.float64)
        self.slave_coords = np.asarray(slave_coords, dtype=np.float64)
        self.mode = mode.lower()
        self.penalty_stiffness = penalty_stiffness

        # Initial relative displacement vectors: d0_j = X_s_j - X_m
        self.d0 = self.slave_coords - self.master_coord[None, :]

        # Create underlying KinematicRBE2Constraint if mesh is provided
        self.mesh = mesh
        if self.mesh is not None:
            self.rbe2_constraint = KinematicRBE2Constraint(
                mesh=self.mesh,
                master_id=self.master_id,
                slave_ids=self.slave_ids,
            )
        else:
            self.rbe2_constraint = None

    @property
    def n_slaves(self) -> int:
        """Number of slave nodes."""
        return len(self.slave_ids)

    def get_slave_displacements(self, u_master: np.ndarray, theta: float) -> np.ndarray:
        """Compute exact kinematic slave node displacements for a given RP motion.

        u_slave_j = u_master + (R(theta) - I) @ d0_j

        Parameters
        ----------
        u_master : (2,) ndarray
            Displacement (ux, uy) of master RP.
        theta : float
            Rotation angle [rad] of master RP.

        Returns
        -------
        u_slaves : (2*N,) ndarray
            Flattened array of slave node displacements [ux1, uy1, ux2, uy2, ...].
        """
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        R_minus_I = R - np.eye(2, dtype=np.float64)

        u_slaves = np.zeros(2 * self.n_slaves, dtype=np.float64)
        for j in range(self.n_slaves):
            u_s_j = u_master + R_minus_I @ self.d0[j]
            u_slaves[2 * j : 2 * j + 2] = u_s_j

        return u_slaves
