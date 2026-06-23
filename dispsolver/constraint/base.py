"""
base.py
=======
Base constraint formulation for Lagrange Multipliers.
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseConstraint(ABC):
    """Abstract base class for kinematic constraints."""

    @abstractmethod
    def n_multipliers(self) -> int:
        """Return the number of Lagrange multiplier DOFs this constraint adds."""
        pass

    @abstractmethod
    def n_extra_primal(self) -> int:
        """Return the number of extra primal DOFs (e.g., rotation angles) this constraint adds."""
        pass

    @abstractmethod
    def assemble(self, u: np.ndarray, u_extra: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray,
        np.ndarray
    ]:
        """Assemble the constraint matrix C and gap g.
        The constraint equation is: C_u @ u + C_extra @ u_extra - g = 0

        Parameters
        ----------
        u : (n_dofs,) array
            Current global nodal displacement vector.
        u_extra : (n_total_extra_primal,) array
            Current global extra primal DOFs.

        Returns
        -------
        row_u : array of int
            Equation index (from 0 to n_multipliers-1).
        col_u : array of int
            Global nodal DOF index.
        val_u : array of float
            Values of C_u.
        row_extra : array of int
            Equation index.
        col_extra : array of int
            Global extra primal DOF index.
        val_extra : array of float
            Values of C_extra.
        g : (n_multipliers,) array of float
            Constraint gap.
        """
        pass
