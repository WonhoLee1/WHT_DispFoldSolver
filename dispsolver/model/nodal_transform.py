"""
nodal_transform.py
==================
Abaqus *TRANSFORM Nodal Coordinate System Transformation Engine.

Computes exact 2D (2x2) and 3D (3x3) rotation matrices for nodes assigned to
local coordinate systems (*TRANSFORM, NSET=..., TYPE=R|C|S in Abaqus).

Enables exact enforcement of slanted/inclined boundary conditions, symmetry,
and local-DOF loading without penalty approximations or ill-conditioning.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import numpy as np


def _orthogonal_vector_3d(v: np.ndarray) -> np.ndarray:
    """Find a unit vector orthogonal to 3D unit vector v."""
    abs_v = np.abs(v)
    if abs_v[0] <= abs_v[1] and abs_v[0] <= abs_v[2]:
        v_other = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif abs_v[1] <= abs_v[2]:
        v_other = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        v_other = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v_orth = v_other - np.dot(v_other, v) * v
    norm = np.linalg.norm(v_orth)
    if norm < 1e-12:
        v_other = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v_orth = v_other - np.dot(v_other, v) * v
        norm = np.linalg.norm(v_orth)
    return v_orth / norm


def compute_rotation_matrix(spec: Dict[str, Any], x_node: np.ndarray, dim: int = 3) -> np.ndarray:
    """Compute the (dim, dim) rotation matrix R mapping local DOFs to global DOFs (u_global = R @ u_local).

    Parameters
    ----------
    spec : dict
        Transform specification dict from Model.transforms:
        {"name": str, "nset": str, "type": "R"|"C"|"S", "points": tuple of tuples}
    x_node : np.ndarray
        Global coordinates of the node (dim,)
    dim : int
        Spatial dimension (2 or 3)

    Returns
    -------
    R : np.ndarray
        (dim, dim) orthogonal rotation matrix (R @ R.T = I).
    """
    ttype = str(spec.get("type", "R")).upper()
    pts = spec.get("points", ())
    flat_pts: List[float] = []
    for row in pts:
        if isinstance(row, (list, tuple, np.ndarray)):
            flat_pts.extend([float(v) for v in row])
        else:
            flat_pts.append(float(row))

    if dim == 2:
        if ttype == "R":
            # Rectangular transform in 2D
            if len(flat_pts) >= 2:
                ax, ay = flat_pts[0], flat_pts[1]
                v1 = np.array([ax, ay], dtype=np.float64)
                norm1 = np.linalg.norm(v1)
                if norm1 < 1e-12:
                    return np.eye(2, dtype=np.float64)
                e1 = v1 / norm1
                e2 = np.array([-e1[1], e1[0]], dtype=np.float64)
                return np.column_stack([e1, e2])
            return np.eye(2, dtype=np.float64)

        elif ttype in ("C", "S"):
            # Cylindrical / Polar transform in 2D centered at Point A
            origin_a = np.zeros(2, dtype=np.float64)
            if len(flat_pts) >= 2:
                origin_a[0] = flat_pts[0]
                origin_a[1] = flat_pts[1]
            r_vec = np.asarray(x_node[:2], dtype=np.float64) - origin_a
            norm_r = np.linalg.norm(r_vec)
            if norm_r < 1e-12:
                return np.eye(2, dtype=np.float64)
            e1 = r_vec / norm_r
            e2 = np.array([-e1[1], e1[0]], dtype=np.float64)
            return np.column_stack([e1, e2])

        else:
            return np.eye(2, dtype=np.float64)

    else:
        # dim == 3
        if ttype == "R":
            # Rectangular transform in 3D
            if len(flat_pts) >= 2:
                ax, ay = flat_pts[0], flat_pts[1]
                az = flat_pts[2] if len(flat_pts) >= 3 else 0.0
                v1 = np.array([ax, ay, az], dtype=np.float64)
                norm1 = np.linalg.norm(v1)
                if norm1 < 1e-12:
                    return np.eye(3, dtype=np.float64)
                e1 = v1 / norm1

                if len(flat_pts) >= 6:
                    bx, by, bz = flat_pts[3], flat_pts[4], flat_pts[5]
                    b_vec = np.array([bx, by, bz], dtype=np.float64)
                    v2_tmp = b_vec - np.dot(b_vec, e1) * e1
                    norm2 = np.linalg.norm(v2_tmp)
                    if norm2 > 1e-12:
                        e2 = v2_tmp / norm2
                    else:
                        e2 = _orthogonal_vector_3d(e1)
                elif abs(e1[2]) < 1e-12:
                    # 2D in-plane vector specified in a 3D system: e1 is in XY plane, e2 is in-plane perp, e3 is Z-axis
                    e2 = np.array([-e1[1], e1[0], 0.0], dtype=np.float64)
                else:
                    e2 = _orthogonal_vector_3d(e1)

                e3 = np.cross(e1, e2)
                norm3 = np.linalg.norm(e3)
                if norm3 > 1e-12:
                    e3 = e3 / norm3
                return np.column_stack([e1, e2, e3])
            return np.eye(3, dtype=np.float64)

        elif ttype == "C":
            # Cylindrical transform in 3D: A to B defines z-axis (axial)
            pt_a = np.zeros(3, dtype=np.float64)
            pt_b = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if len(flat_pts) >= 3:
                pt_a[:3] = flat_pts[:3]
            if len(flat_pts) >= 6:
                pt_b[:3] = flat_pts[3:6]

            z_axis = pt_b - pt_a
            norm_z = np.linalg.norm(z_axis)
            if norm_z < 1e-12:
                z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                z_axis = z_axis / norm_z

            e3 = z_axis
            r_vec = np.asarray(x_node[:3], dtype=np.float64) - pt_a
            r_proj = r_vec - np.dot(r_vec, e3) * e3
            norm_r = np.linalg.norm(r_proj)

            if norm_r < 1e-12:
                e1 = _orthogonal_vector_3d(e3)
            else:
                e1 = r_proj / norm_r

            e2 = np.cross(e3, e1)
            e2 = e2 / np.linalg.norm(e2)
            return np.column_stack([e1, e2, e3])

        elif ttype == "S":
            # Spherical transform in 3D centered at Point A
            pt_a = np.zeros(3, dtype=np.float64)
            if len(flat_pts) >= 3:
                pt_a[:3] = flat_pts[:3]
            r_vec = np.asarray(x_node[:3], dtype=np.float64) - pt_a
            norm_r = np.linalg.norm(r_vec)
            if norm_r < 1e-12:
                return np.eye(3, dtype=np.float64)
            e1 = r_vec / norm_r
            e2 = _orthogonal_vector_3d(e1)
            e3 = np.cross(e1, e2)
            e3 = e3 / np.linalg.norm(e3)
            return np.column_stack([e1, e2, e3])

        else:
            return np.eye(dim, dtype=np.float64)


def build_nodal_transforms(sys: Any, model: Any) -> Tuple[np.ndarray, bool]:
    """Build nodal rotation matrices for all nodes in FlattenedSolverSystem.

    Parameters
    ----------
    sys : FlattenedSolverSystem
        Flat solver system
    model : Model
        CAE model containing model.transforms definitions

    Returns
    -------
    node_transforms : np.ndarray
        (N_nodes, dim, dim) array of nodal rotation matrices R_i
    has_transforms : bool
        True if at least one node has a non-identity rotation matrix.
    """
    n_nodes = sys.coords.shape[0]
    dim = sys.dim
    node_transforms = np.zeros((n_nodes, dim, dim), dtype=np.float64)
    for i in range(n_nodes):
        node_transforms[i] = np.eye(dim, dtype=np.float64)

    if not hasattr(model, "transforms") or not model.transforms:
        return node_transforms, False

    has_transforms = False

    for spec_name, spec in model.transforms.items():
        nset_name = spec.get("nset", "")
        if not nset_name:
            continue

        # Look up node indices for this nset name in sys.global_nsets
        node_indices = None
        if nset_name in sys.global_nsets:
            node_indices = sys.global_nsets[nset_name]
        else:
            # Case-insensitive / alias match
            for g_name, g_indices in sys.global_nsets.items():
                if g_name.upper() == nset_name.upper() or g_name.upper().endswith("." + nset_name.upper()):
                    node_indices = g_indices
                    break

        if node_indices is None:
            # Used to skip the transform silently (nodes stayed in the global frame).
            loc = spec.get("source_location")
            raise ValueError(f"transform {spec_name!r}" + (f" [{loc}]" if loc else "")
                             + f": node set {nset_name!r} is not defined")
        if len(node_indices) == 0:
            continue

        for idx in node_indices:
            if idx < 0 or idx >= n_nodes:
                continue
            x_c = sys.coords[idx]
            R = compute_rotation_matrix(spec, x_c, dim=dim)
            if not np.allclose(R, np.eye(dim)):
                node_transforms[idx] = R
                has_transforms = True

    return node_transforms, has_transforms


def transform_u_to_global(u_vec: np.ndarray, node_transforms: np.ndarray, dim: int) -> np.ndarray:
    """Transform displacement vector u_vec from local nodal DOFs to global physical DOFs."""
    n_nodes = len(node_transforms)
    u_global = u_vec.copy()
    I_dim = np.eye(dim, dtype=np.float64)
    for i in range(n_nodes):
        R = node_transforms[i]
        if not np.allclose(R, I_dim):
            u_global[i * dim: i * dim + dim] = R @ u_vec[i * dim: i * dim + dim]
    return u_global


def transform_u_to_local(u_global: np.ndarray, node_transforms: np.ndarray, dim: int) -> np.ndarray:
    """Transform displacement vector u_global from global physical DOFs to local nodal DOFs."""
    n_nodes = len(node_transforms)
    u_local = u_global.copy()
    I_dim = np.eye(dim, dtype=np.float64)
    for i in range(n_nodes):
        R = node_transforms[i]
        if not np.allclose(R, I_dim):
            u_local[i * dim: i * dim + dim] = R.T @ u_global[i * dim: i * dim + dim]
    return u_local


def apply_nodal_transforms_to_element_group(
    f_sub: np.ndarray,
    K_sub: np.ndarray,
    sub_conn: np.ndarray,
    node_transforms: np.ndarray,
    dim: int
) -> None:
    """Transform element internal forces f_sub and stiffness K_sub in-place from global DOFs to nodal local DOFs."""
    ne, n_dofs = f_sub.shape
    nodes_per_elem = sub_conn.shape[1]

    n_nodes = len(node_transforms)
    is_non_identity = np.zeros(n_nodes, dtype=bool)
    I_dim = np.eye(dim, dtype=np.float64)
    for i in range(n_nodes):
        R_i = node_transforms[i][:dim, :dim] if node_transforms[i].ndim == 2 else node_transforms[i]
        if R_i.shape == (dim, dim) and not np.allclose(R_i, I_dim):
            is_non_identity[i] = True

    if not np.any(is_non_identity):
        return

    for e in range(ne):
        conn_e = sub_conn[e]
        if not np.any(is_non_identity[conn_e]):
            continue

        # Build element block transformation matrix T_e
        T_e = np.eye(n_dofs, dtype=np.float64)
        n_block = min(nodes_per_elem, n_dofs // dim)
        for a in range(n_block):
            n_idx = conn_e[a]
            R_node = node_transforms[n_idx][:dim, :dim] if node_transforms[n_idx].ndim == 2 else node_transforms[n_idx]
            if R_node.shape == (dim, dim):
                T_e[a * dim: a * dim + dim, a * dim: a * dim + dim] = R_node

        # Transform f_sub
        f_sub[e] = T_e.T @ f_sub[e]

        # Transform K_sub
        K_orig = K_sub[e].reshape((n_dofs, n_dofs))
        K_transformed = T_e.T @ K_orig @ T_e
        if K_sub[e].ndim == 1:
            K_sub[e] = K_transformed.ravel()
        else:
            K_sub[e] = K_transformed

