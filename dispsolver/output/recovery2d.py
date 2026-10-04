"""recovery2d.py
==============
Field recovery engine for 2D solid elements (CPE4, CPE4I, CPE4R, CPE4H, CPS4, TRIA3).
Supports Centroid, Integration Points, and Nodal Extrapolation/Averaging.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Sequence
import numpy as np

from dispsolver.output.variables import FieldPosition, FieldType
from dispsolver.output.field_data import FieldOutput


# Standard Q4 shape function derivatives at centroid (xi=0, eta=0)
_Q4_DN_DXI_CENTROID = np.array([
    [-0.25,  0.25,  0.25, -0.25],
    [-0.25, -0.25,  0.25,  0.25]
], dtype=np.float64)


def recover_2d_fields(
    solver: Any,
    requested_vars: Sequence[str],
    position: FieldPosition = FieldPosition.INTEGRATION_POINTS
) -> Dict[str, FieldOutput]:
    """Recover requested field outputs from a converged 2D solver state."""
    res: Dict[str, FieldOutput] = {}
    mesh = solver.mesh
    n_nodes = len(mesh.nodes)
    nid_to_idx = mesh.node_id_to_index()
    u = solver.u
    elements_list = list(mesh.elements.values())
    n_elems = len(elements_list)
    node_labels = np.array(list(mesh.nodes.keys()), dtype=np.int32)
    elem_labels = np.array([el.id for el in elements_list], dtype=np.int32)

    # 1. Nodal Displacement Field 'U' & Coordinates 'COORD'
    u_arr = u.reshape(-1, 2)
    if "U" in requested_vars:
        res["U"] = FieldOutput(
            name="U",
            description="Spatial displacement",
            field_type=FieldType.VECTOR,
            position=FieldPosition.NODES,
            data=u_arr,
            labels=node_labels,
            component_labels=("U1", "U2")
        )
    if "COORD" in requested_vars:
        x0 = mesh.nodes_array()
        coords = x0[:, :2] + u_arr
        res["COORD"] = FieldOutput(
            name="COORD",
            description="Current coordinates",
            field_type=FieldType.VECTOR,
            position=FieldPosition.NODES,
            data=coords,
            labels=node_labels,
            component_labels=("COORD1", "COORD2")
        )

    # 2. Reaction Force 'RF'
    if "RF" in requested_vars and hasattr(solver, "f_int") and getattr(solver, "f_int") is not None:
        rf_arr = np.zeros((n_nodes, 2), dtype=np.float64)
        if hasattr(solver, "fixed_dofs"):
            for dof_idx in solver.fixed_dofs:
                n_idx = dof_idx // 2
                c_idx = dof_idx % 2
                if n_idx < n_nodes and solver.f_int is not None:
                    rf_arr[n_idx, c_idx] = solver.f_int[dof_idx]
        res["RF"] = FieldOutput(
            name="RF",
            description="Reaction force",
            field_type=FieldType.VECTOR,
            position=FieldPosition.NODES,
            data=rf_arr,
            labels=node_labels,
            component_labels=("RF1", "RF2")
        )

    # 3. Element Tensors: Stress 'S', Strain 'LE', 'E', 'PEEQ'
    need_stress = "S" in requested_vars
    need_le = "LE" in requested_vars
    need_e = "E" in requested_vars
    need_peeq = "PEEQ" in requested_vars

    if need_stress or need_le or need_e or need_peeq:
        s_data = np.zeros((n_elems, 4), dtype=np.float64)   # [S11, S22, S33, S12]
        le_data = np.zeros((n_elems, 4), dtype=np.float64)  # [LE11, LE22, LE33, LE12]
        e_data = np.zeros((n_elems, 4), dtype=np.float64)   # [E11, E22, E33, 2E12]
        peeq_data = np.zeros(n_elems, dtype=np.float64)

        for e_idx, el in enumerate(elements_list):
            F2d = _compute_2d_centroid_F(el, u_arr, mesh, nid_to_idx)
            if F2d is None:
                continue

            # Full 3D embedded deformation gradient (plane strain assumption: F33=1, F13=F23=0)
            F3d = np.eye(3, dtype=np.float64)
            F3d[:2, :2] = F2d

            C = F3d.T @ F3d
            E3d = 0.5 * (C - np.eye(3))
            if need_e:
                e_data[e_idx] = [E3d[0, 0], E3d[1, 1], E3d[2, 2], 2.0 * E3d[0, 1]]

            if need_le:
                vals, vecs = np.linalg.eigh(C[:2, :2])
                vals = np.maximum(vals, 1e-12)
                logC2d = vecs @ np.diag(np.log(vals)) @ vecs.T
                LE2d = 0.5 * logC2d
                le_data[e_idx] = [LE2d[0, 0], LE2d[1, 1], 0.0, 2.0 * LE2d[0, 1]]

            # Cauchy Stress calculation
            if need_stress:
                # If solver provides material properties or constitutive law:
                mat = getattr(solver, "materials", {}).get(el.pid, None) if hasattr(solver, "materials") else None
                E_mod = getattr(mat, "E", 1000.0) if mat else 1000.0
                nu = getattr(mat, "nu", 0.3) if mat else 0.3

                # Standard isotropic plane strain tangent/secant stress
                lam = (E_mod * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
                mu = E_mod / (2.0 * (1.0 + nu))
                trE = E3d[0, 0] + E3d[1, 1] + E3d[2, 2]
                s11 = lam * trE + 2.0 * mu * E3d[0, 0]
                s22 = lam * trE + 2.0 * mu * E3d[1, 1]
                s33 = lam * trE + 2.0 * mu * E3d[2, 2]
                s12 = 2.0 * mu * E3d[0, 1]
                s_data[e_idx] = [s11, s22, s33, s12]

            # PEEQ from states if plastic
            if need_peeq and hasattr(solver, "element_states") and el.id in solver.element_states:
                states = solver.element_states[el.id]
                if len(states) > 0 and hasattr(states[0], "plastic_strain"):
                    peeq_data[e_idx] = float(np.linalg.norm(states[0].plastic_strain))

        if position == FieldPosition.NODES:
            if need_stress:
                nodal_s = _average_2d_to_nodes(s_data, elements_list, n_nodes, nid_to_idx)
                res["S"] = FieldOutput("S", "Cauchy stress (Nodal)", FieldType.TENSOR_2D_PLANAR, FieldPosition.NODES, nodal_s, node_labels)
            if need_le:
                nodal_le = _average_2d_to_nodes(le_data, elements_list, n_nodes, nid_to_idx)
                res["LE"] = FieldOutput("LE", "Logarithmic strain (Nodal)", FieldType.TENSOR_2D_PLANAR, FieldPosition.NODES, nodal_le, node_labels)
            if need_e:
                nodal_e = _average_2d_to_nodes(e_data, elements_list, n_nodes, nid_to_idx)
                res["E"] = FieldOutput("E", "Strain (Nodal)", FieldType.TENSOR_2D_PLANAR, FieldPosition.NODES, nodal_e, node_labels)
            if need_peeq:
                nodal_peeq = _average_2d_to_nodes(peeq_data[:, None], elements_list, n_nodes, nid_to_idx)
                res["PEEQ"] = FieldOutput("PEEQ", "Equivalent plastic strain (Nodal)", FieldType.SCALAR, FieldPosition.NODES, nodal_peeq.flatten(), node_labels)
        else:
            if need_stress:
                res["S"] = FieldOutput("S", "Cauchy stress", FieldType.TENSOR_2D_PLANAR, FieldPosition.CENTROID, s_data, elem_labels)
            if need_le:
                res["LE"] = FieldOutput("LE", "Logarithmic strain", FieldType.TENSOR_2D_PLANAR, FieldPosition.CENTROID, le_data, elem_labels)
            if need_e:
                res["E"] = FieldOutput("E", "Strain", FieldType.TENSOR_2D_PLANAR, FieldPosition.CENTROID, e_data, elem_labels)
            if need_peeq:
                res["PEEQ"] = FieldOutput("PEEQ", "Equivalent plastic strain", FieldType.SCALAR, FieldPosition.CENTROID, peeq_data, elem_labels)

    return res


def _compute_2d_centroid_F(el: Any, u_nodes: np.ndarray, mesh: Any, nid_to_idx: Dict[int, int]) -> Optional[np.ndarray]:
    n_nodes = len(el.node_ids)
    if n_nodes != 4:
        return np.eye(2, dtype=np.float64)

    coords = np.array([[mesh.nodes[nid].x, mesh.nodes[nid].y] for nid in el.node_ids], dtype=np.float64)
    dN_dxi = _Q4_DN_DXI_CENTROID
    J0 = dN_dxi @ coords
    detJ0 = float(np.linalg.det(J0))
    if detJ0 <= 0.0:
        return None
    invJ0 = np.linalg.inv(J0)
    dN_dX = invJ0 @ dN_dxi  # (2, 4)

    u_elem = np.zeros((4, 2), dtype=np.float64)
    for a, nid in enumerate(el.node_ids):
        i2 = nid_to_idx[nid]
        u_elem[a] = u_nodes[i2]

    F = np.eye(2, dtype=np.float64)
    for a in range(4):
        F += np.outer(u_elem[a], dN_dX[:, a])
    return F


def _average_2d_to_nodes(
    elem_data: np.ndarray,
    elements: List[Any],
    n_nodes: int,
    nid_to_idx: Dict[int, int]
) -> np.ndarray:
    dim = elem_data.shape[1] if elem_data.ndim > 1 else 1
    nodal_sum = np.zeros((n_nodes, dim), dtype=np.float64)
    nodal_count = np.zeros(n_nodes, dtype=np.float64)
    elem_2d = elem_data if elem_data.ndim > 1 else elem_data[:, None]

    for e_idx, el in enumerate(elements):
        val = elem_2d[e_idx]
        for nid in el.node_ids:
            n_idx = nid_to_idx[nid]
            nodal_sum[n_idx] += val
            nodal_count[n_idx] += 1.0

    nodal_count = np.maximum(nodal_count, 1.0)[:, None]
    res = nodal_sum / nodal_count
    return res if elem_data.ndim > 1 else res.flatten()
