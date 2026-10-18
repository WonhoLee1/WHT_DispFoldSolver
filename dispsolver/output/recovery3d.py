"""recovery3d.py
==============
Field recovery engine for 3D elements (Cauchy stress S, Strain E/LE, PEEQ, SENER).
Supports Centroid, Integration Points, and Nodal Extrapolation/Averaging.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np

from dispsolver.material3d.numba_materials import material_dispatch_3d
from dispsolver.output.variables import FieldPosition, FieldType
from dispsolver.output.field_data import FieldOutput

_TET4_TYPES = {"C3D4", "C3D4_ANP", "ANP"}
_HEX8_TYPES = {"C3D8", "C3D8R", "C3D8_CR", "C3D8I", "C3D8H"}
_TET10_TYPES = {"C3D10", "C3D10M", "C3D10_MODIFIED"}
_WEDGE6_TYPES = {"C3D6", "C3D6_WEDGE", "WEDGE6"}
_HEX8_FBAR_TYPES = {"C3D8_FBAR"}

_TET4_DN_DXI = np.array([
    [-1.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0, 1.0],
], dtype=np.float64)


def _hex8_shape_derivs_centroid() -> np.ndarray:
    xi = eta = zeta = 0.0
    dN_dxi = np.array([
        -0.125 * (1.0 - eta) * (1.0 - zeta),
         0.125 * (1.0 - eta) * (1.0 - zeta),
         0.125 * (1.0 + eta) * (1.0 - zeta),
        -0.125 * (1.0 + eta) * (1.0 - zeta),
        -0.125 * (1.0 - eta) * (1.0 + zeta),
         0.125 * (1.0 - eta) * (1.0 + zeta),
         0.125 * (1.0 + eta) * (1.0 + zeta),
        -0.125 * (1.0 + eta) * (1.0 + zeta),
    ], dtype=np.float64)
    dN_deta = np.array([
        -0.125 * (1.0 - xi) * (1.0 - zeta),
        -0.125 * (1.0 + xi) * (1.0 - zeta),
         0.125 * (1.0 + xi) * (1.0 - zeta),
         0.125 * (1.0 - xi) * (1.0 - zeta),
        -0.125 * (1.0 - xi) * (1.0 + zeta),
        -0.125 * (1.0 + xi) * (1.0 + zeta),
         0.125 * (1.0 + xi) * (1.0 + zeta),
         0.125 * (1.0 - xi) * (1.0 + zeta),
    ], dtype=np.float64)
    dN_dzeta = np.array([
        -0.125 * (1.0 - xi) * (1.0 - eta),
        -0.125 * (1.0 + xi) * (1.0 - eta),
        -0.125 * (1.0 + xi) * (1.0 + eta),
        -0.125 * (1.0 - xi) * (1.0 + eta),
         0.125 * (1.0 - xi) * (1.0 - eta),
         0.125 * (1.0 + xi) * (1.0 - eta),
         0.125 * (1.0 + xi) * (1.0 + eta),
         0.125 * (1.0 - xi) * (1.0 + eta),
    ], dtype=np.float64)
    return np.vstack([dN_dxi, dN_deta, dN_dzeta])


_HEX8_DN_DXI_CENTROID = _hex8_shape_derivs_centroid()


def recover_3d_fields(
    solver: Any,
    requested_vars: Sequence[str],
    position: FieldPosition = FieldPosition.INTEGRATION_POINTS
) -> Dict[str, FieldOutput]:
    """Recover requested field outputs from a converged 3D solver state."""
    res: Dict[str, FieldOutput] = {}
    mesh = solver.mesh
    n_nodes = len(mesh.nodes)
    nid_to_idx = mesh.node_id_to_index()
    u = solver.u
    elements_list = list(mesh.elements.values())
    n_elems = len(elements_list)

    # 1. Nodal Displacement Field 'U'
    if "U" in requested_vars or "COORD" in requested_vars:
        u_arr = u.reshape(-1, 3)
        node_labels = np.array(list(mesh.nodes.keys()), dtype=np.int32)
        if "U" in requested_vars:
            res["U"] = FieldOutput(
                name="U",
                description="Spatial displacement",
                field_type=FieldType.VECTOR,
                position=FieldPosition.NODES,
                data=u_arr,
                labels=node_labels,
                component_labels=("U1", "U2", "U3")
            )
        if "COORD" in requested_vars:
            x0 = mesh.nodes_array()
            coords = x0 + u_arr
            res["COORD"] = FieldOutput(
                name="COORD",
                description="Current coordinates",
                field_type=FieldType.VECTOR,
                position=FieldPosition.NODES,
                data=coords,
                labels=node_labels,
                component_labels=("COORD1", "COORD2", "COORD3")
            )

    # 2. Reaction Force 'RF'
    if "RF" in requested_vars and hasattr(solver, "f_int") and getattr(solver, "f_int") is not None:
        # At equilibrium, RF = F_int - F_ext on constrained nodes
        rf_arr = np.zeros((n_nodes, 3), dtype=np.float64)
        if hasattr(solver, "fixed_dofs"):
            for dof_idx in solver.fixed_dofs:
                n_idx = dof_idx // 3
                c_idx = dof_idx % 3
                if n_idx < n_nodes and solver.f_int is not None:
                    rf_arr[n_idx, c_idx] = solver.f_int[dof_idx]
        node_labels = np.array(list(mesh.nodes.keys()), dtype=np.int32)
        res["RF"] = FieldOutput(
            name="RF",
            description="Reaction force",
            field_type=FieldType.VECTOR,
            position=FieldPosition.NODES,
            data=rf_arr,
            labels=node_labels,
            component_labels=("RF1", "RF2", "RF3")
        )

    # 3. Element Tensors & Invariants: Stress 'S', 'MISES', 'MAXP', 'MIDP', 'MINP', 'PRESS', 'LE', 'E', 'PEEQ'
    need_s = "S" in requested_vars
    need_mises = "MISES" in requested_vars
    need_maxp = "MAXP" in requested_vars
    need_midp = "MIDP" in requested_vars
    need_minp = "MINP" in requested_vars
    need_press = "PRESS" in requested_vars
    need_stress = need_s or need_mises or need_maxp or need_midp or need_minp or need_press

    need_le = "LE" in requested_vars
    need_e = "E" in requested_vars
    need_peeq = "PEEQ" in requested_vars

    if need_stress or need_le or need_e or need_peeq:
        s_data = np.zeros((n_elems, 6), dtype=np.float64)
        le_data = np.zeros((n_elems, 6), dtype=np.float64)
        e_data = np.zeros((n_elems, 6), dtype=np.float64)
        peeq_data = np.zeros(n_elems, dtype=np.float64)
        elem_labels = np.array([el.id for el in elements_list], dtype=np.int32)
        node_labels = np.array(list(mesh.nodes.keys()), dtype=np.int32)

        elem_mat_types = getattr(solver, "elem_mat_types", np.zeros(n_elems, dtype=np.int32))
        elem_props = getattr(solver, "elem_props", np.zeros((n_elems, 10), dtype=np.float64))

        for e_idx, el in enumerate(elements_list):
            F = _compute_element_centroid_F(el, u, mesh, nid_to_idx)
            if F is None:
                continue

            # Cauchy Stress via material dispatch
            if need_stress:
                sigma_voigt = _compute_cauchy_stress(F, int(elem_mat_types[e_idx]), elem_props[e_idx])
                if sigma_voigt is not None:
                    s_data[e_idx] = sigma_voigt

            # Green-Lagrange Strain E = 0.5 * (F^T F - I)
            C = F.T @ F
            E_mat = 0.5 * (C - np.eye(3))
            if need_e:
                e_data[e_idx] = [E_mat[0, 0], E_mat[1, 1], E_mat[2, 2], 2.0 * E_mat[0, 1], 2.0 * E_mat[0, 2], 2.0 * E_mat[1, 2]]

            # Logarithmic Hencky Strain LE = 0.5 * log(C)
            if need_le:
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-12)
                logC = vecs @ np.diag(np.log(vals)) @ vecs.T
                LE_mat = 0.5 * logC
                le_data[e_idx] = [LE_mat[0, 0], LE_mat[1, 1], LE_mat[2, 2], 2.0 * LE_mat[0, 1], 2.0 * LE_mat[0, 2], 2.0 * LE_mat[1, 2]]

            # PEEQ from element states if plastic
            if need_peeq and hasattr(solver, "element_states") and el.id in solver.element_states:
                states = solver.element_states[el.id]
                if len(states) > 0 and hasattr(states[0], "plastic_strain"):
                    peeq_data[e_idx] = float(np.linalg.norm(states[0].plastic_strain))

        # Compute derived stress invariants if needed
        if need_mises:
            s11, s22, s33 = s_data[:, 0], s_data[:, 1], s_data[:, 2]
            s12, s13, s23 = s_data[:, 3], s_data[:, 4], s_data[:, 5]
            mises_sq = 0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2) + 3.0 * (s12**2 + s13**2 + s23**2)
            mises_data = np.sqrt(np.maximum(mises_sq, 0.0))

        if need_maxp or need_midp or need_minp:
            maxp_data = np.zeros(n_elems, dtype=np.float64)
            midp_data = np.zeros(n_elems, dtype=np.float64)
            minp_data = np.zeros(n_elems, dtype=np.float64)
            for i in range(n_elems):
                s = s_data[i]
                tensor = np.array([
                    [s[0], s[3], s[4]],
                    [s[3], s[1], s[5]],
                    [s[4], s[5], s[2]]
                ], dtype=np.float64)
                eigvals = np.linalg.eigvalsh(tensor)
                maxp_data[i] = eigvals[2]
                midp_data[i] = eigvals[1]
                minp_data[i] = eigvals[0]

        if need_press:
            press_data = -(s_data[:, 0] + s_data[:, 1] + s_data[:, 2]) / 3.0

        # Check if nodal averaging requested
        if position == FieldPosition.NODES:
            # Extrapolate/Average to nodes
            if need_s:
                nodal_s = _average_element_to_nodes(s_data, elements_list, n_nodes, nid_to_idx)
                res["S"] = FieldOutput("S", "Cauchy stress (Nodal)", FieldType.TENSOR_3D_FULL, FieldPosition.NODES, nodal_s, node_labels)
            if need_mises:
                nodal_mises = _average_element_to_nodes(mises_data[:, None], elements_list, n_nodes, nid_to_idx)
                res["MISES"] = FieldOutput("MISES", "Von Mises equivalent stress (Nodal)", FieldType.SCALAR, FieldPosition.NODES, nodal_mises.flatten(), node_labels, component_labels=("MISES",))
            if need_maxp:
                nodal_maxp = _average_element_to_nodes(maxp_data[:, None], elements_list, n_nodes, nid_to_idx)
                res["MAXP"] = FieldOutput("MAXP", "Maximum principal stress (Nodal)", FieldType.SCALAR, FieldPosition.NODES, nodal_maxp.flatten(), node_labels, component_labels=("MAXP",))
            if need_midp:
                nodal_midp = _average_element_to_nodes(midp_data[:, None], elements_list, n_nodes, nid_to_idx)
                res["MIDP"] = FieldOutput("MIDP", "Intermediate principal stress (Nodal)", FieldType.SCALAR, FieldPosition.NODES, nodal_midp.flatten(), node_labels, component_labels=("MIDP",))
            if need_minp:
                nodal_minp = _average_element_to_nodes(minp_data[:, None], elements_list, n_nodes, nid_to_idx)
                res["MINP"] = FieldOutput("MINP", "Minimum principal stress (Nodal)", FieldType.SCALAR, FieldPosition.NODES, nodal_minp.flatten(), node_labels, component_labels=("MINP",))
            if need_press:
                nodal_press = _average_element_to_nodes(press_data[:, None], elements_list, n_nodes, nid_to_idx)
                res["PRESS"] = FieldOutput("PRESS", "Hydrostatic pressure (Nodal)", FieldType.SCALAR, FieldPosition.NODES, nodal_press.flatten(), node_labels, component_labels=("PRESS",))
            if need_le:
                nodal_le = _average_element_to_nodes(le_data, elements_list, n_nodes, nid_to_idx)
                res["LE"] = FieldOutput("LE", "Logarithmic strain (Nodal)", FieldType.TENSOR_3D_FULL, FieldPosition.NODES, nodal_le, node_labels)
            if need_e:
                nodal_e = _average_element_to_nodes(e_data, elements_list, n_nodes, nid_to_idx)
                res["E"] = FieldOutput("E", "Strain (Nodal)", FieldType.TENSOR_3D_FULL, FieldPosition.NODES, nodal_e, node_labels)
            if need_peeq:
                nodal_peeq = _average_element_to_nodes(peeq_data[:, None], elements_list, n_nodes, nid_to_idx)
                res["PEEQ"] = FieldOutput("PEEQ", "Equivalent plastic strain (Nodal)", FieldType.SCALAR, FieldPosition.NODES, nodal_peeq.flatten(), node_labels)
        else:
            # Centroid / Element output
            if need_s:
                res["S"] = FieldOutput("S", "Cauchy stress", FieldType.TENSOR_3D_FULL, FieldPosition.CENTROID, s_data, elem_labels)
            if need_mises:
                res["MISES"] = FieldOutput("MISES", "Von Mises equivalent stress", FieldType.SCALAR, FieldPosition.CENTROID, mises_data, elem_labels, component_labels=("MISES",))
            if need_maxp:
                res["MAXP"] = FieldOutput("MAXP", "Maximum principal stress", FieldType.SCALAR, FieldPosition.CENTROID, maxp_data, elem_labels, component_labels=("MAXP",))
            if need_midp:
                res["MIDP"] = FieldOutput("MIDP", "Intermediate principal stress", FieldType.SCALAR, FieldPosition.CENTROID, midp_data, elem_labels, component_labels=("MIDP",))
            if need_minp:
                res["MINP"] = FieldOutput("MINP", "Minimum principal stress", FieldType.SCALAR, FieldPosition.CENTROID, minp_data, elem_labels, component_labels=("MINP",))
            if need_press:
                res["PRESS"] = FieldOutput("PRESS", "Hydrostatic pressure", FieldType.SCALAR, FieldPosition.CENTROID, press_data, elem_labels, component_labels=("PRESS",))
            if need_le:
                res["LE"] = FieldOutput("LE", "Logarithmic strain", FieldType.TENSOR_3D_FULL, FieldPosition.CENTROID, le_data, elem_labels)
            if need_e:
                res["E"] = FieldOutput("E", "Strain", FieldType.TENSOR_3D_FULL, FieldPosition.CENTROID, e_data, elem_labels)
            if need_peeq:
                res["PEEQ"] = FieldOutput("PEEQ", "Equivalent plastic strain", FieldType.SCALAR, FieldPosition.CENTROID, peeq_data, elem_labels)

    # 4. Contact Variables: CPRESS, COPEN, CSHEAR, CSTATUS, CSLIP
    need_cpress = "CPRESS" in requested_vars
    need_copen = "COPEN" in requested_vars
    need_cshear = "CSHEAR" in requested_vars
    need_cstatus = "CSTATUS" in requested_vars
    need_cslip = "CSLIP" in requested_vars

    if need_cpress or need_copen or need_cshear or need_cstatus or need_cslip:
        node_labels = np.array(list(mesh.nodes.keys()), dtype=np.int32)
        cpress_data = np.zeros(n_nodes, dtype=np.float64)
        copen_data = np.zeros(n_nodes, dtype=np.float64)
        cstatus_data = np.zeros(n_nodes, dtype=np.float64)
        cshear_data = np.zeros((n_nodes, 2), dtype=np.float64)
        cslip_data = np.zeros((n_nodes, 2), dtype=np.float64)

        if hasattr(solver, "constraints") and solver.constraints:
            for c in solver.constraints:
                # 1. SurfaceContactConstraint3D (Node-to-Rigid-Plane)
                if hasattr(c, "slave_node_ids") and hasattr(c, "normal") and hasattr(c, "point") and hasattr(c, "_normal_pressure"):
                    for s_nid in c.slave_node_ids:
                        s_idx = c.nid_to_idx.get(s_nid, None)
                        if s_idx is None or s_idx >= n_nodes:
                            continue
                        xs, p_i = c._normal_pressure(u, s_nid)
                        gap_offset = c._gap_offset.get(s_nid, 0.0) if hasattr(c, "_gap_offset") else 0.0
                        gap = float(np.dot(xs - c.point, c.normal)) - gap_offset
                        copen_data[s_idx] = gap
                        cpress_data[s_idx] = p_i
                        if p_i <= 0.0:
                            cstatus_data[s_idx] = 0.0
                        else:
                            is_slip = False
                            if hasattr(c, "_friction_force_and_tangent") and getattr(c, "mu", 0.0) > 0.0:
                                f_t_3d, Ktt, is_slip, s1, s2, s_mag = c._friction_force_and_tangent(s_nid, xs, p_i)
                                t1 = getattr(c, "t1", np.array([1.0, 0.0, 0.0]))
                                t2 = getattr(c, "t2", np.array([0.0, 1.0, 0.0]))
                                cshear_data[s_idx, 0] = float(np.dot(f_t_3d, t1))
                                cshear_data[s_idx, 1] = float(np.dot(f_t_3d, t2))
                                cslip_data[s_idx, 0] = s1
                                cslip_data[s_idx, 1] = s2
                            cstatus_data[s_idx] = 2.0 if is_slip else 1.0

                # 2. DeformableSurfaceContactConstraint3D (Node-to-Segment deformable)
                elif hasattr(c, "_pairs") and hasattr(c, "_normal_pressure") and hasattr(c, "_current_gap_and_normal"):
                    for pair in c._pairs:
                        s_nid = pair[0]
                        m_face = pair[1]
                        xi, eta, sign = pair[2], pair[3], pair[4]
                        s_idx = c.nid_to_idx.get(s_nid, None)
                        if s_idx is None or s_idx >= n_nodes:
                            continue
                        xs, p_i = c._normal_pressure(u, s_nid, m_face, xi, eta, sign)
                        gap, n, N_shape, _, _ = c._current_gap_and_normal(u, s_nid, m_face, xi, eta, sign)
                        copen_data[s_idx] = gap
                        cpress_data[s_idx] = p_i
                        if p_i <= 0.0:
                            cstatus_data[s_idx] = 0.0
                        else:
                            is_slip = False
                            if getattr(c, "mu", 0.0) > 0.0:
                                m_idx = [c.nid_to_idx[nid] for nid in m_face]
                                q = [c.coords[i] + u[3 * i : 3 * i + 3] for i in m_idx]
                                f_t_3d, Ktt, is_slip, s1, s2, s_mag, t1, t2 = c._friction_force_and_tangent(s_nid, xs, q, n, p_i)
                                cshear_data[s_idx, 0] = float(np.dot(f_t_3d, t1))
                                cshear_data[s_idx, 1] = float(np.dot(f_t_3d, t2))
                                cslip_data[s_idx, 0] = s1
                                cslip_data[s_idx, 1] = s2
                            cstatus_data[s_idx] = 2.0 if is_slip else 1.0

                # 3. Generic fallback
                elif hasattr(c, "get_active_set"):
                    active = c.get_active_set(u)
                    for s_nid in active:
                        idx = nid_to_idx.get(s_nid, None)
                        if idx is not None and idx < n_nodes:
                            if cstatus_data[idx] == 0.0:
                                cstatus_data[idx] = 1.0

        if need_cpress:
            res["CPRESS"] = FieldOutput("CPRESS", "Contact pressure", FieldType.SCALAR, FieldPosition.NODES, cpress_data, node_labels, component_labels=("CPRESS",))
        if need_copen:
            res["COPEN"] = FieldOutput("COPEN", "Contact opening", FieldType.SCALAR, FieldPosition.NODES, copen_data, node_labels, component_labels=("COPEN",))
        if need_cstatus:
            res["CSTATUS"] = FieldOutput("CSTATUS", "Contact status (0=Open, 1=Stick, 2=Slip)", FieldType.SCALAR, FieldPosition.NODES, cstatus_data, node_labels, component_labels=("CSTATUS",))
        if need_cshear:
            res["CSHEAR"] = FieldOutput("CSHEAR", "Frictional shear stress", FieldType.VECTOR, FieldPosition.NODES, cshear_data, node_labels, component_labels=("CSHEAR1", "CSHEAR2"))
        if need_cslip:
            res["CSLIP"] = FieldOutput("CSLIP", "Contact slip", FieldType.VECTOR, FieldPosition.NODES, cslip_data, node_labels, component_labels=("CSLIP1", "CSLIP2"))


    return res


def _compute_element_centroid_F(el: Any, u: np.ndarray, mesh: Any, nid_to_idx: Dict[int, int]) -> Optional[np.ndarray]:
    """Compute deformation gradient F at element centroid."""
    et = el.elem_type.upper()
    n_nodes = len(el.node_ids)
    coords = np.array([[mesh.nodes[nid].x, mesh.nodes[nid].y, mesh.nodes[nid].z] for nid in el.node_ids], dtype=np.float64)

    if et in _HEX8_TYPES or et in _HEX8_FBAR_TYPES or (n_nodes == 8 and et.startswith("C3D")):
        dN_dxi = _HEX8_DN_DXI_CENTROID
    elif et in _TET4_TYPES or (n_nodes == 4 and et.startswith("C3D")):
        dN_dxi = _TET4_DN_DXI
    else:
        # Generic centroid approximation
        return np.eye(3, dtype=np.float64)

    J0 = dN_dxi @ coords
    detJ0 = float(np.linalg.det(J0))
    if detJ0 <= 0.0:
        return None
    invJ0 = np.linalg.inv(J0)
    dN_dX = invJ0 @ dN_dxi  # (3, n_nodes)

    u_elem = np.zeros((n_nodes, 3), dtype=np.float64)
    for a, nid in enumerate(el.node_ids):
        i3 = nid_to_idx[nid]
        u_elem[a] = u[3 * i3: 3 * i3 + 3]

    F = np.eye(3, dtype=np.float64)
    for a in range(n_nodes):
        F += np.outer(u_elem[a], dN_dX[:, a])

    return F


def _compute_cauchy_stress(F: np.ndarray, mat_type: int, props: np.ndarray) -> Optional[np.ndarray]:
    detF = float(np.linalg.det(F))
    if detF <= 0.0:
        return None
    C = F.T @ F
    E = 0.5 * (C - np.eye(3))
    E_voigt = np.array([E[0, 0], E[1, 1], E[2, 2], 2.0 * E[0, 1], 2.0 * E[0, 2], 2.0 * E[1, 2]])

    sdv_dummy = np.zeros(0, dtype=np.float64)
    S_voigt, _C_mat, _sdv_new, err = material_dispatch_3d(int(mat_type), props, sdv_dummy, E_voigt, F, detF, 1.0)
    if err != 0:
        return None

    S = np.array([
        [S_voigt[0], S_voigt[3], S_voigt[4]],
        [S_voigt[3], S_voigt[1], S_voigt[5]],
        [S_voigt[4], S_voigt[5], S_voigt[2]],
    ])
    sigma = (F @ S @ F.T) / detF
    return np.array([sigma[0, 0], sigma[1, 1], sigma[2, 2], sigma[0, 1], sigma[0, 2], sigma[1, 2]])


def _average_element_to_nodes(
    elem_data: np.ndarray,
    elements: List[Any],
    n_nodes: int,
    nid_to_idx: Dict[int, int]
) -> np.ndarray:
    """Averages element quantities to nodes using inverse connectivity weighting."""
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
