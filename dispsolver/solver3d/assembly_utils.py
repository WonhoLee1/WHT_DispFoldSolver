"""
assembly_utils.py
=================
Data-Oriented Design (DOD) Topology and Sparse Assembly utilities.
Calculates element-to-global scatter maps for in-place CSR/COO matrix updates.
"""

import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix, csr_matrix
from dispsolver._jit_cache import njit_cached


def build_global_topology(num_dofs: int, elem_conn: np.ndarray):
    """
    Pre-computes the (rows, cols) coordinate arrays for a fully sparse matrix.
    Since scipy's coo_matrix handles duplicate (row, col) entries by automatically
    summing them, we can build flat arrays of length (num_elems * n_dofs * n_dofs).
    
    Args:
        num_dofs: Total degrees of freedom in the system.
        elem_conn: (n_elems, n_nodes) integer array of 0-based node indices.
        
    Returns:
        rows: Flat numpy array of row indices.
        cols: Flat numpy array of column indices.
    """
    n_elems = elem_conn.shape[0]
    n_nodes = elem_conn.shape[1] if n_elems > 0 else 8
    n_dofs = n_nodes * 3
    n_entries = n_elems * n_dofs * n_dofs
    
    rows = np.zeros(n_entries, dtype=np.int32)
    cols = np.zeros(n_entries, dtype=np.int32)
    
    if n_elems > 0:
        actual_entries = _populate_topology(elem_conn, rows, cols)
        return rows[:actual_entries], cols[:actual_entries]
    
    return rows, cols


@njit(fastmath=True)
def _populate_topology(elem_conn: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> int:
    n_elems = elem_conn.shape[0]
    n_nodes = elem_conn.shape[1]
    n_dofs = n_nodes * 3
    idx = 0
    dofs_e = np.zeros(n_dofs, dtype=np.int32)
    
    for e in range(n_elems):
        nodes_e = elem_conn[e]
        for i in range(n_nodes):
            n = nodes_e[i]
            if n >= 0:
                dofs_e[3*i + 0] = 3 * n + 0
                dofs_e[3*i + 1] = 3 * n + 1
                dofs_e[3*i + 2] = 3 * n + 2
            else:
                dofs_e[3*i + 0] = -1
                dofs_e[3*i + 1] = -1
                dofs_e[3*i + 2] = -1
            
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
def scatter_f_int_3d(f_elems: np.ndarray, elem_conn: np.ndarray, f_int_global: np.ndarray):
    """
    Scatters element internal force vectors (n_elems, n_nodes*3) into global force vector (num_dofs,).
    Executed via Numba JIT with zero Python overhead.
    """
    n_elems = elem_conn.shape[0]
    n_nodes = elem_conn.shape[1]
    for e in range(n_elems):
        nodes_e = elem_conn[e]
        for i in range(n_nodes):
            n = nodes_e[i]
            if n >= 0:
                f_int_global[3 * n + 0] += f_elems[e, 3 * i + 0]
                f_int_global[3 * n + 1] += f_elems[e, 3 * i + 1]
                f_int_global[3 * n + 2] += f_elems[e, 3 * i + 2]


@njit(parallel=True, fastmath=True)
def scatter_k_csr_numba(k_elems: np.ndarray, data_offsets: np.ndarray, csr_data: np.ndarray):
    """
    In-place scatter of element stiffness matrices (n_elems, 24, 24) into pre-allocated CSR data array.
    Zero memory re-allocation, zero COO-to-CSR conversion.
    """
    csr_data.fill(0.0)
    n_elems = k_elems.shape[0]
    for e in prange(n_elems):
        for i in range(24):
            for j in range(24):
                off = data_offsets[e, i, j]
                csr_data[off] += k_elems[e, i, j]


@njit_cached
def build_coo_to_csr_mapping(rows: np.ndarray, cols: np.ndarray, indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Build a 1D mapping array of length len(rows) where mapping[k] is the index in csr.data
    corresponding to (rows[k], cols[k]).
    Uses binary search within the row's column index slice indices[indptr[r]:indptr[r+1]].
    """
    N = len(rows)
    mapping = np.empty(N, dtype=np.int32)
    for k in range(N):
        r = rows[k]
        c = cols[k]
        start = indptr[r]
        end = indptr[r + 1]
        low = start
        high = end - 1
        found = -1
        while low <= high:
            mid = (low + high) // 2
            mid_val = indices[mid]
            if mid_val == c:
                found = mid
                break
            elif mid_val < c:
                low = mid + 1
            else:
                high = mid - 1
        mapping[k] = found
    return mapping


@njit_cached(fastmath=True)
def scatter_k_csr_direct(data: np.ndarray, mapping: np.ndarray, csr_data: np.ndarray):
    """
    Accumulates flat element stiffness data into pre-allocated CSR data array using precomputed 1D mapping.
    Zero memory allocation, zero sorting/re-indexing.
    """
    csr_data.fill(0.0)
    for k in range(len(data)):
        m = mapping[k]
        if m >= 0:
            csr_data[m] += data[k]


@njit_cached(parallel=True, fastmath=True)
def apply_dirichlet_bcs_csr_fast(
    csr_data: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    is_fixed: np.ndarray,
    penalty: float
):
    """
    Applies Dirichlet boundary conditions directly in-place on CSR data matrix.
    Zeros out rows and columns of fixed DOFs to preserve matrix symmetry, sets
    diagonal elements to penalty, and provides 1.0 on diagonal for isolated DOFs.
    """
    n_rows = len(indptr) - 1
    for r in prange(n_rows):
        r_fixed = is_fixed[r]
        row_start = indptr[r]
        row_end = indptr[r + 1]
        
        diag_p = -1
        row_has_nonzero = False
        
        for p in range(row_start, row_end):
            c = indices[p]
            if c == r:
                diag_p = p
            if r_fixed or is_fixed[c]:
                if r == c:
                    csr_data[p] = penalty
                    row_has_nonzero = True
                else:
                    csr_data[p] = 0.0
            else:
                if csr_data[p] != 0.0:
                    row_has_nonzero = True
                    
        if not r_fixed and not row_has_nonzero and diag_p != -1:
            csr_data[diag_p] = 1.0


def compute_element_coloring(elem_conn: np.ndarray, num_nodes: int):
    """
    Compute element coloring such that elements sharing nodes have different colors.
    Enables race-condition-free multi-threaded parallel assembly.
    """
    n_elems = elem_conn.shape[0]
    node_to_elems = [[] for _ in range(num_nodes)]
    for e in range(n_elems):
        for n in elem_conn[e]:
            node_to_elems[n].append(e)

    elem_colors = np.full(n_elems, -1, dtype=np.int32)
    
    for e in range(n_elems):
        neighbor_colors = set()
        for n in elem_conn[e]:
            for adj_e in node_to_elems[n]:
                if adj_e != e and elem_colors[adj_e] != -1:
                    neighbor_colors.add(elem_colors[adj_e])
        
        c = 0
        while c in neighbor_colors:
            c += 1
        elem_colors[e] = c

    num_colors = elem_colors.max() + 1 if n_elems > 0 else 0
    color_groups = [np.where(elem_colors == c)[0].astype(np.int32) for c in range(num_colors)]
    return color_groups
