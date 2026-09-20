"""
dynamic_shell.py
================
Linear and Nonlinear Static/Dynamic FE Solver for 3D Shell Elements (6 DOFs per node).
Compatible with MITC3+ (Tri3) and MITC4+ (Quad4) Shell Formulations.

Degrees of Freedom per node:
    dof 0: u_x  (global X translation)
    dof 1: u_y  (global Y translation)
    dof 2: u_z  (global Z translation)
    dof 3: theta_x (global X rotation)
    dof 4: theta_y (global Y rotation)
    dof 5: theta_z (global Z rotation)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from dispsolver.element_shell.mitc3_plus_numba import compute_mitc3_plus_stiffness
from dispsolver.element_shell.mitc4_plus_numba import compute_mitc4_plus_stiffness

try:
    from pypardiso import spsolve as pardiso_spsolve
    HAS_PARDISO = True
except ImportError:
    HAS_PARDISO = False


class DynamicSolverShell:
    """Finite Element Solver Engine for 3D Shell Structural Mechanics.

    Parameters
    ----------
    nodes : ndarray (n_nodes, 3)
        Global node coordinates.
    elements : ndarray (n_elems, 3 or 4) or list
        Element connectivity.
    elem_type : str
        "MITC3" / "MITC3_PLUS" or "MITC4" / "MITC4_PLUS".
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    thickness : float
        Shell thickness.
    density : float, optional
        Material mass density.
    """

    def __init__(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        elem_type: str = "MITC4",
        E: float = 2.0e11,
        nu: float = 0.3,
        thickness: float = 0.01,
        density: float = 7850.0
    ):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.elements = np.asarray(elements, dtype=np.int64)
        self.elem_type = elem_type.upper()
        self.E = float(E)
        self.nu = float(nu)
        self.thickness = float(thickness)
        self.density = float(density)

        self.num_nodes = self.nodes.shape[0]
        self.num_dofs = self.num_nodes * 6
        self.dofs_per_node = 6

        # Displacement vector (num_dofs,)
        self.u = np.zeros(self.num_dofs, dtype=np.float64)
        self.fixed_dofs: Dict[int, float] = {}

    def fix_dof(self, node_id: int, dof_local: int, value: float = 0.0):
        """Constrain a specific DOF at a given node.

        Parameters
        ----------
        node_id : int (0-based)
            Node index.
        dof_local : int (0..5)
            0: ux, 1: uy, 2: uz, 3: rotx, 4: roty, 5: rotz.
        value : float
            Prescribed displacement or rotation.
        """
        global_dof = node_id * 6 + dof_local
        self.fixed_dofs[global_dof] = float(value)

    def assemble_stiffness_matrix(self) -> sp.csc_matrix:
        """Assemble global sparse stiffness matrix K."""
        is_tri = (self.elem_type in ["MITC3", "MITC3_PLUS", "S3", "S3R"] or self.elements.shape[1] == 3)
        nodes_per_elem = 3 if is_tri else 4
        edofs = nodes_per_elem * 6

        n_elems = self.elements.shape[0]
        rows_list = []
        cols_list = []
        vals_list = []

        for e in range(n_elems):
            conn = self.elements[e]
            coords = self.nodes[conn]

            if is_tri:
                Ke = compute_mitc3_plus_stiffness(coords, self.E, self.nu, self.thickness)
            else:
                Ke = compute_mitc4_plus_stiffness(coords, self.E, self.nu, self.thickness)

            dof_indices = np.zeros(edofs, dtype=np.int64)
            for i, nid in enumerate(conn):
                for d in range(6):
                    dof_indices[i * 6 + d] = nid * 6 + d

            for r in range(edofs):
                gr = dof_indices[r]
                for c in range(edofs):
                    gc = dof_indices[c]
                    val = Ke[r, c]
                    if abs(val) > 1e-15:
                        rows_list.append(gr)
                        cols_list.append(gc)
                        vals_list.append(val)

        K_global = sp.coo_matrix(
            (vals_list, (rows_list, cols_list)),
            shape=(self.num_dofs, self.num_dofs),
            dtype=np.float64
        ).tocsc()

        return K_global

    def solve_static(self, f_ext: np.ndarray) -> np.ndarray:
        """Solve static linear equilibrium K u = f_ext with prescribed Dirichlet BCs.

        Parameters
        ----------
        f_ext : ndarray (num_dofs,)
            External force and moment vector.

        Returns
        -------
        u : ndarray (num_dofs,)
            Solved displacement and rotation vector.
        """
        K = self.assemble_stiffness_matrix()
        f = np.asarray(f_ext, dtype=np.float64).copy()

        # Identify free and fixed DOFs
        all_dofs = np.arange(self.num_dofs)
        fixed_indices = np.array(list(self.fixed_dofs.keys()), dtype=np.int64) if self.fixed_dofs else np.array([], dtype=np.int64)
        free_mask = np.ones(self.num_dofs, dtype=bool)
        if len(fixed_indices) > 0:
            free_mask[fixed_indices] = False
        free_indices = all_dofs[free_mask]

        # Apply inhomogeneous boundary conditions: f_free -= K_free_fixed @ u_fixed
        u_sol = np.zeros(self.num_dofs, dtype=np.float64)
        if len(fixed_indices) > 0:
            for fdof, fval in self.fixed_dofs.items():
                u_sol[fdof] = fval

            if np.any(u_sol[fixed_indices] != 0.0):
                K_free_fixed = K[free_indices, :][:, fixed_indices]
                f[free_indices] -= K_free_fixed.dot(u_sol[fixed_indices])

        # Submatrix solve for free DOFs
        K_free = K[free_indices, :][:, free_indices]
        f_free = f[free_indices]

        if HAS_PARDISO:
            try:
                u_free = pardiso_spsolve(K_free, f_free)
            except Exception:
                u_free = spsolve(K_free, f_free)
        else:
            u_free = spsolve(K_free, f_free)

        u_sol[free_indices] = u_free
        self.u = u_sol
        return u_sol
