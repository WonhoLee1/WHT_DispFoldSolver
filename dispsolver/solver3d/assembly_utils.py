"""
assembly_utils.py
=================
Data-Oriented Design (DOD) Topology and Sparse Assembly utilities.
Calculates element-to-global scatter maps for in-place CSR/COO matrix updates.
"""

import numpy as np
from numba import njit
from scipy.sparse import coo_matrix, csr_matrix

def build_global_topology(num_dofs: int, elem_conn: np.ndarray):
    """
    Pre-computes the (rows, cols) coordinate arrays for a fully sparse matrix.
    Since scipy's coo_matrix handles duplicate (row, col) entries by automatically
    summing them, we can build flat arrays of length (num_elems * 24 * 24).
    
    Args:
        num_dofs: Total degrees of freedom in the system.
        elem_conn: (n_elems, 8) integer array of 0-based node indices.
        
    Returns:
        rows: Flat numpy array of row indices.
        cols: Flat numpy array of column indices.
        scatter_map: Not strictly needed for COO, we just fill `data` array sequentially.
    """
    n_elems = elem_conn.shape[0]
    n_entries = n_elems * 24 * 24
    
    # Pre-allocate
    rows = np.zeros(n_entries, dtype=np.int32)
    cols = np.zeros(n_entries, dtype=np.int32)
    
    # Numba JIT compiling the population loop for instant execution
    _populate_topology(elem_conn, rows, cols)
    
    return rows, cols

@njit(fastmath=True)
def _populate_topology(elem_conn: np.ndarray, rows: np.ndarray, cols: np.ndarray):
    n_elems = elem_conn.shape[0]
    idx = 0
    dofs_e = np.zeros(24, dtype=np.int32)
    
    for e in range(n_elems):
        nodes_e = elem_conn[e]
        for i in range(8):
            n = nodes_e[i]
            dofs_e[3*i + 0] = 3 * n + 0
            dofs_e[3*i + 1] = 3 * n + 1
            dofs_e[3*i + 2] = 3 * n + 2
            
        for i in range(24):
            for j in range(24):
                rows[idx] = dofs_e[i]
                cols[idx] = dofs_e[j]
                idx += 1


@njit(fastmath=True)
def scatter_f_int_3d(f_elems: np.ndarray, elem_conn: np.ndarray, f_int_global: np.ndarray):
    """
    Scatters element internal force vectors (n_elems, 24) into global force vector (num_dofs,).
    Executed via Numba JIT with zero Python overhead.
    """
    n_elems = elem_conn.shape[0]
    for e in range(n_elems):
        nodes_e = elem_conn[e]
        for i in range(8):
            n = nodes_e[i]
            f_int_global[3 * n + 0] += f_elems[e, 3 * i + 0]
            f_int_global[3 * n + 1] += f_elems[e, 3 * i + 1]
            f_int_global[3 * n + 2] += f_elems[e, 3 * i + 2]


def compute_element_coloring(elem_conn: np.ndarray, num_nodes: int):
    """
    Compute element coloring such that elements sharing nodes have different colors.
    Enables race-condition-free multi-threaded parallel assembly.
    
    Args:
        elem_conn: (n_elems, 8) integer array of 0-based node indices.
        num_nodes: Total number of nodes.
        
    Returns:
        colors: List of 1D integer numpy arrays, each containing element indices of that color.
    """
    n_elems = elem_conn.shape[0]
    # Build node-to-element adjacency
    node_to_elems = [[] for _ in range(num_nodes)]
    for e in range(n_elems):
        for n in elem_conn[e]:
            node_to_elems[n].append(e)

    elem_colors = np.full(n_elems, -1, dtype=np.int32)
    
    for e in range(n_elems):
        # Find colors of neighboring elements
        neighbor_colors = set()
        for n in elem_conn[e]:
            for adj_e in node_to_elems[n]:
                if adj_e != e and elem_colors[adj_e] != -1:
                    neighbor_colors.add(elem_colors[adj_e])
        
        # Pick lowest available color
        c = 0
        while c in neighbor_colors:
            c += 1
        elem_colors[e] = c

    num_colors = elem_colors.max() + 1 if n_elems > 0 else 0
    color_groups = [np.where(elem_colors == c)[0].astype(np.int32) for c in range(num_colors)]
    return color_groups

