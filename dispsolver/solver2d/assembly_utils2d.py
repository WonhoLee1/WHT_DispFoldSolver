"""
assembly_utils2d.py
===================
Data-Oriented Design (DOD) Topology and Sparse Assembly utilities for 2D Solid Elements.
Calculates element-to-global scatter maps for 2D plane strain / plane stress.
"""

import numpy as np
from numba import njit, prange


def build_global_topology_2d(num_dofs: int, elem_conn: np.ndarray):
    """
    Pre-computes (rows, cols) coordinate arrays for 2D sparse matrix assembly.
    
    Args:
        num_dofs: Total DOFs (n_nodes * 2).
        elem_conn: (n_elems, n_nodes) integer array of 0-based node indices.
        
    Returns:
        rows: Flat numpy array of row indices.
        cols: Flat numpy array of column indices.
    """
    n_elems = elem_conn.shape[0]
    n_nodes = elem_conn.shape[1] if n_elems > 0 else 4
    n_dofs = n_nodes * 2
    n_entries = n_elems * n_dofs * n_dofs

    rows = np.zeros(n_entries, dtype=np.int32)
    cols = np.zeros(n_entries, dtype=np.int32)

    if n_elems > 0:
        actual_entries = _populate_topology_2d(elem_conn, rows, cols)
        return rows[:actual_entries], cols[:actual_entries]

    return rows, cols


@njit(fastmath=True)
def _populate_topology_2d(elem_conn: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> int:
    n_elems = elem_conn.shape[0]
    n_nodes = elem_conn.shape[1]
    n_dofs = n_nodes * 2
    idx = 0
    dofs_e = np.zeros(n_dofs, dtype=np.int32)

    for e in range(n_elems):
        nodes_e = elem_conn[e]
        for i in range(n_nodes):
            n = nodes_e[i]
            if n >= 0:
                dofs_e[2 * i + 0] = 2 * n + 0
                dofs_e[2 * i + 1] = 2 * n + 1
            else:
                dofs_e[2 * i + 0] = -1
                dofs_e[2 * i + 1] = -1

        for i in range(n_dofs):
            r = dofs_e[i]
            if r < 0:
                continue
            for j in range(n_dofs):
                c = dofs_e[j]
                if c < 0:
                    continue
                rows[idx] = r
                cols[idx] = c
                idx += 1

    return idx


@njit(fastmath=True)
def scatter_f_int_2d(f_elems: np.ndarray, elem_conn: np.ndarray, f_int_global: np.ndarray):
    """
    Scatters 2D element internal force vectors (n_elems, n_nodes*2) into global force vector.
    """
    n_elems = elem_conn.shape[0]
    n_nodes = elem_conn.shape[1]
    for e in range(n_elems):
        nodes_e = elem_conn[e]
        for i in range(n_nodes):
            n = nodes_e[i]
            if n >= 0:
                f_int_global[2 * n + 0] += f_elems[e, 2 * i + 0]
                f_int_global[2 * n + 1] += f_elems[e, 2 * i + 1]
