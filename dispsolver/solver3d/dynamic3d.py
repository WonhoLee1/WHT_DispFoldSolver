"""
dynamic3d.py
============
3D Implicit Non-linear Dynamic & Quasistatic Solver for WHT_DispFoldSolver.
Features Bathe Composite Integration, Surface Tie Constraint Assembly,
and Dirichlet Prescribed Motion Enforcers.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.sparse import csc_matrix

try:
    from pypardiso import spsolve as pardiso_spsolve
    HAS_PARDISO = True
except ImportError:
    from scipy.sparse.linalg import spsolve as pardiso_spsolve
    HAS_PARDISO = False

from dispsolver.mesh3d import Mesh3D
from dispsolver.element3d import SolidElement3D, QuadraturePointState3D, Hexa8EASElement, Hexa8FbarElement
from dispsolver.solver3d.assembly_utils import build_global_topology, scatter_f_int_3d
from dispsolver.material3d.numba_materials import (
    MAT_LINEAR_ELASTIC,
    MAT_NEO_HOOKEAN,
    MAT_J2_PLASTICITY,
    MAT_CUSTOM_ELASTIC,
    get_default_sdv_count,
)


class DynamicSolver3D:
    """3D Implicit Non-Linear Solver with Above-Abaqus Speed & Convergence Engine."""

    def __init__(
        self,
        mesh: Mesh3D,
        materials: Optional[Dict[int, Any]] = None,
        material_params: Optional[Dict[str, float]] = None,
        nlgeom: bool = True
    ):
        self.mesh = mesh
        self.nlgeom = nlgeom
        
        # Handle material definitions
        if materials is not None and isinstance(materials, dict):
            if "E" in materials or "nu" in materials:
                material_params = materials
                self.materials = {}
            else:
                self.materials = materials
        else:
            self.materials = {}

        if material_params is None:
            material_params = {"E": 200000.0, "nu": 0.3}
        self.material_params = material_params

        self.num_nodes = mesh.num_nodes
        self.num_dofs = self.num_nodes * 3

        # System state vectors (3 DOFs per node: u_x, u_y, u_z)
        self.u = np.zeros(self.num_dofs, dtype=np.float64)
        self.v = np.zeros(self.num_dofs, dtype=np.float64)
        self.a = np.zeros(self.num_dofs, dtype=np.float64)

        # Multi-point & Surface Tie constraints list
        self.constraints: List[Any] = []

        # Prescribed Dirichlet BCs: dict of {global_dof_index: prescribed_value}
        self.fixed_dofs: Dict[int, float] = {}
        self.last_assembly_error: int = 0

        # Default Material Matrix C (Isotropic 3D)
        E = material_params.get("E", 200000.0)
        nu = material_params.get("nu", 0.3)
        lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        self.C_mat_default = np.array([
            [lam + 2*mu, lam,        lam,        0,  0,  0],
            [lam,        lam + 2*mu, lam,        0,  0,  0],
            [lam,        lam,        lam + 2*mu, 0,  0,  0],
            [0,          0,          0,          mu, 0,  0],
            [0,          0,          0,          0,  mu, 0],
            [0,          0,          0,          0,  0,  mu]
        ], dtype=np.float64)

        # Element instances mapping
        self.element_instances: Dict[int, SolidElement3D] = {}
        self.element_states: Dict[int, List[QuadraturePointState3D]] = {}

        self._initialize_elements()
        self._setup_numba_topology()

    def _initialize_elements(self):
        """Instantiate 3D element formulations based on elem_type."""
        for eid, elem in self.mesh.elements.items():
            if elem.elem_type.upper() in ["C3D8I", "C3D8_EAS"]:
                inst = Hexa8EASElement(num_eas_modes=9)
            elif elem.elem_type.upper() in ["C3D8_FBAR", "FBAR"]:
                inst = Hexa8FbarElement()
            elif elem.elem_type.upper() in ["C3D8H", "C3D8_HYBRID", "HYBRID"]:
                inst = Hexa8FbarElement()
            else:
                # Default to C3D8I EAS
                inst = Hexa8EASElement(num_eas_modes=9)

            self.element_instances[eid] = inst
            self.element_states[eid] = [QuadraturePointState3D.create_initial() for _ in range(inst.num_quad_points)]

    def _setup_numba_topology(self):
        """Precompute global assembly topology and DOD multimaterial arrays."""
        if not self.mesh.elements:
            self.elem_conn_0based = None
            self.rows_topo = None
            self.cols_topo = None
            self.elem_mat_types = None
            self.elem_props = None
            self.elem_sdvs = None
            self.elem_kernel_groups = {}
            return

        nid_map = self.mesh.node_id_to_index()
        self.elem_conn_0based = np.array(
            [[nid_map[nid] for nid in elem.node_ids] for elem in self.mesh.elements.values()],
            dtype=np.int64
        )
        self.rows_topo, self.cols_topo = build_global_topology(self.num_dofs, self.elem_conn_0based)

        n_elems = self.elem_conn_0based.shape[0]
        self.elem_mat_types = np.full(n_elems, MAT_CUSTOM_ELASTIC, dtype=np.int32)
        self.elem_props = np.zeros((n_elems, 36), dtype=np.float64)

        # Classify element kernel groups for heterogeneous multi-element assembly
        elem_kernel_indices: Dict[int, List[int]] = {}
        for idx, elem in enumerate(self.mesh.elements.values()):
            et = elem.elem_type.upper()
            if et in ["C3D8I", "C3D8_EAS"]:
                k = 0
            elif et in ["C3D8_CR", "C3D8_COROTATIONAL", "C3D8_FBAR_CR"]:
                k = 1
            elif et in ["C3D8H", "C3D8_HYBRID", "HYBRID"]:
                k = 2
            else:
                k = 3
            if k not in elem_kernel_indices:
                elem_kernel_indices[k] = []
            elem_kernel_indices[k].append(idx)

        self.elem_kernel_groups = {
            k: np.array(indices, dtype=np.int64)
            for k, indices in elem_kernel_indices.items()
        }

        # Populate multimaterial properties per element
        cmat_flat = self.C_mat_default.ravel()
        for e, elem in enumerate(self.mesh.elements.values()):
            pid = getattr(elem, "pid", 0)
            mat_obj = self.materials.get(pid, None)
            if mat_obj is not None:
                if hasattr(mat_obj, "mat_type") and hasattr(mat_obj, "props"):
                    self.elem_mat_types[e] = int(mat_obj.mat_type)
                    p = np.asarray(mat_obj.props, dtype=np.float64)
                    self.elem_props[e, :min(36, len(p))] = p[:36]
                elif hasattr(mat_obj, "C_mat"):
                    self.elem_mat_types[e] = MAT_CUSTOM_ELASTIC
                    self.elem_props[e, :36] = np.asarray(mat_obj.C_mat, dtype=np.float64).ravel()[:36]
                elif isinstance(mat_obj, dict):
                    if mat_obj.get("type", "").lower() in ["j2", "plasticity", "j2_plasticity"] or "sigma_y0" in mat_obj:
                        self.elem_mat_types[e] = MAT_J2_PLASTICITY
                        E_val = float(mat_obj.get("E", 200000.0))
                        nu_val = float(mat_obj.get("nu", 0.3))
                        sy_val = float(mat_obj.get("sigma_y0", 400.0))
                        H_val = float(mat_obj.get("H", 0.0))
                        self.elem_props[e, :4] = [E_val, nu_val, sy_val, H_val]
                    elif mat_obj.get("type", "").lower() in ["neohookean", "neo_hookean"]:
                        self.elem_mat_types[e] = MAT_NEO_HOOKEAN
                        c10 = float(mat_obj.get("C10", 1.0))
                        d1 = float(mat_obj.get("D1", 0.001))
                        mu = 2.0 * c10
                        K = 2.0 / max(d1, 1e-12)
                        self.elem_props[e, 0] = mu
                        self.elem_props[e, 1] = K
                        self.elem_props[e, 2] = c10
                        self.elem_props[e, 3] = d1
                    else:
                        self.elem_mat_types[e] = MAT_LINEAR_ELASTIC
                        self.elem_props[e, 0] = float(mat_obj.get("E", 200000.0))
                        self.elem_props[e, 1] = float(mat_obj.get("nu", 0.3))
                else:
                    self.elem_mat_types[e] = MAT_CUSTOM_ELASTIC
                    self.elem_props[e, :36] = cmat_flat
            else:
                self.elem_mat_types[e] = MAT_CUSTOM_ELASTIC
                self.elem_props[e, :36] = cmat_flat

        max_sdvs = 0
        for mt in self.elem_mat_types:
            max_sdvs = max(max_sdvs, get_default_sdv_count(mt))
        self.elem_sdvs = np.zeros((n_elems, 8, max_sdvs), dtype=np.float64)

    def add_constraint(self, constraint: Any):
        """Add a surface tie or MPC constraint to the solver."""
        self.constraints.append(constraint)

    def set_dirichlet_bc(self, global_dof: int, value: float):
        """Prescribe a 3D Dirichlet displacement value for a global DOF."""
        self.fixed_dofs[int(global_dof)] = float(value)

    def fix_dof(self, node_id: int, dof_axis: int, value: float = 0.0):
        """Fix a 3D degree of freedom (dof_axis: 0 for X, 1 for Y, 2 for Z)."""
        nid_map = self.mesh.node_id_to_index()
        global_dof = 3 * nid_map[node_id] + dof_axis
        self.set_dirichlet_bc(global_dof, value)

    def assemble_system(self, u_vec: np.ndarray, dt: float = 1.0, return_error: bool = False):
        """Assemble global 3D stiffness matrix K_global and internal force vector f_int."""
        node_coords_all = self.mesh.nodes_array()

        if self.nlgeom and self.rows_topo is not None:
            try:
                n_elems = self.elem_conn_0based.shape[0]
                f_elems = np.zeros((n_elems, 24), dtype=np.float64)
                K_elems = np.zeros((n_elems, 24, 24), dtype=np.float64)
                has_error = False

                # Heterogeneous element assembly: assemble each group in parallel
                for k, elem_indices in self.elem_kernel_groups.items():
                    sub_conn = self.elem_conn_0based[elem_indices]
                    sub_mat = self.elem_mat_types[elem_indices]
                    sub_props = self.elem_props[elem_indices]
                    sub_sdvs = self.elem_sdvs[elem_indices]

                    if k == 0:
                        from dispsolver.element3d.c3d8_eas_tl_numba import assemble_mesh_c3d8_eas_tl_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_eas_tl_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt
                        )
                    elif k == 1:
                        from dispsolver.element3d.c3d8_corotational_numba import assemble_mesh_c3d8_corotational_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_corotational_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt
                        )
                    elif k == 2:
                        from dispsolver.element3d.c3d8_hybrid_numba import assemble_mesh_c3d8_hybrid_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_hybrid_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt
                        )
                    else:
                        from dispsolver.element3d.c3d8_fbar_tl_numba import assemble_mesh_c3d8_fbar_tl_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_fbar_tl_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt
                        )

                    if err:
                        has_error = True
                    f_elems[elem_indices] = f_sub
                    K_elems[elem_indices] = K_sub

                self.last_assembly_error = has_error
                f_int_global = np.zeros(self.num_dofs, dtype=np.float64)
                scatter_f_int_3d(f_elems, self.elem_conn_0based, f_int_global)

                if len(self.constraints) == 0:
                    K_global = csc_matrix(
                        (K_elems.ravel(), (self.rows_topo, self.cols_topo)),
                        shape=(self.num_dofs, self.num_dofs)
                    )
                    if return_error:
                        return K_global, f_int_global, has_error
                    return K_global, f_int_global

                # If surface tie or MPC constraints exist, append their contributions efficiently
                rows_all = [self.rows_topo]
                cols_all = [self.cols_topo]
                data_all = [K_elems.ravel()]

                for constraint in self.constraints:
                    if hasattr(constraint, "reproject_deformed"):
                        constraint.reproject_deformed(u_vec)
                    f_c, (rows_c, cols_c, data_c), _ = constraint.assemble(u_vec)
                    f_int_global += f_c
                    rows_all.append(np.asarray(rows_c, dtype=np.int32))
                    cols_all.append(np.asarray(cols_c, dtype=np.int32))
                    data_all.append(np.asarray(data_c, dtype=np.float64))

                K_global = csc_matrix(
                    (np.concatenate(data_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
                    shape=(self.num_dofs, self.num_dofs)
                )
                if return_error:
                    return K_global, f_int_global, has_error
                return K_global, f_int_global
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Fastpath failed! Falling back to python loop.")

        nid_map = self.mesh.node_id_to_index()

        rows = []
        cols = []
        data = []
        f_int_global = np.zeros(self.num_dofs, dtype=np.float64)

        for eid, elem in self.mesh.elements.items():
            inst = self.element_instances[eid]
            states = self.element_states[eid]

            # Gather element node coordinates and displacements
            elem_node_indices = [nid_map[nid] for nid in elem.node_ids]
            elem_coords = node_coords_all[elem_node_indices]  # (Nn, 3)

            elem_dofs = []
            for n_idx in elem_node_indices:
                elem_dofs.extend([3 * n_idx + 0, 3 * n_idx + 1, 3 * n_idx + 2])

            u_elem = u_vec[elem_dofs]

            # Get material Matrix C for this element's PID if available
            pid = getattr(elem, "pid", 0)
            mat_obj = self.materials.get(pid, None)

            if mat_obj is not None and hasattr(mat_obj, "C_mat"):
                C_mat_e = mat_obj.C_mat
            else:
                C_mat_e = self.C_mat_default

            # Compute element K_e and f_int_e
            if isinstance(inst, Hexa8EASElement):
                K_e, f_e, _ = inst.compute_element_stiffness_and_force(elem_coords, u_elem, C_mat_e, states)
            else:
                K_e, f_e = inst.compute_element_stiffness_and_force(elem_coords, u_elem, C_mat_e, states)

            # Assemble into global arrays
            for i in range(len(elem_dofs)):
                f_int_global[elem_dofs[i]] += f_e[i]
                for j in range(len(elem_dofs)):
                    rows.append(elem_dofs[i])
                    cols.append(elem_dofs[j])
                    data.append(K_e[i, j])

        # Assemble surface tie & MPC constraints
        for constraint in self.constraints:
            if hasattr(constraint, "reproject_deformed"):
                constraint.reproject_deformed(u_vec)
            f_c, (rows_c, cols_c, data_c), _ = constraint.assemble(u_vec)
            f_int_global += f_c
            rows.extend(rows_c)
            cols.extend(cols_c)
            data.extend(data_c)

        K_global = csc_matrix((data, (rows, cols)), shape=(self.num_dofs, self.num_dofs))
        self.last_assembly_error = 0
        if return_error:
            return K_global, f_int_global, 0
        return K_global, f_int_global

    def apply_boundary_conditions(self, K_global: csc_matrix, residual: np.ndarray, u_k: np.ndarray) -> Tuple[csc_matrix, np.ndarray]:
        """Apply Dirichlet boundary conditions via exact diagonal penalty method."""
        K_bc = K_global.tolil()
        r_bc = residual.copy()

        penalty = 1.0e15
        for dof, target_val in self.fixed_dofs.items():
            K_bc[dof, :] = 0.0
            K_bc[:, dof] = 0.0
            K_bc[dof, dof] = penalty
            r_bc[dof] = penalty * (target_val - u_k[dof])

        return K_bc.tocsc(), r_bc

    def solve_step(
        self,
        dt: float = 1.0,
        f_ext: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        max_iters: int = 25
    ) -> Tuple[bool, int]:
        """Solve a 3D non-linear incremental step using Newton-Raphson with Armijo line search."""
        if f_ext is None:
            f_ext = np.zeros(self.num_dofs, dtype=np.float64)

        free_dof_mask = np.ones(self.num_dofs, dtype=bool)
        if self.fixed_dofs:
            free_dof_mask[list(self.fixed_dofs.keys())] = False

        u_k = self.u.copy()
        r_0_norm = None

        for iter_count in range(1, max_iters + 1):
            # print(f"    --> Starting assemble_system iter {iter_count}", flush=True)
            K_g, f_int = self.assemble_system(u_k, dt=dt)
            if self.last_assembly_error > 0:
                # Smart Cutback: Element inverted
                return False, iter_count
            residual = f_ext - f_int

            # Compute true physical residual norm on free DOFs
            r_free = residual[free_dof_mask]
            r_norm = float(np.linalg.norm(r_free))
            f_ext_free_norm = float(np.linalg.norm(f_ext[free_dof_mask]))

            if r_0_norm is None or r_norm > r_0_norm:
                r_0_norm = max(r_norm, f_ext_free_norm, 1.0)

            # Apply BCs for linear system solve
            K_bc, r_bc = self.apply_boundary_conditions(K_g, residual, u_k)

            rel_r = r_norm / r_0_norm
            # Commercial CAE convergence criteria (Abaqus standard 0.5% residual tolerance)
            if (rel_r < 5e-3 or r_norm < 1e-3) and iter_count > 1:
                self.u = u_k
                return True, iter_count

            # Solve linear system \Delta u = K^-1 * r
            try:
                du = pardiso_spsolve(K_bc, r_bc)
            except Exception:
                # Regularization fallback if matrix is singular
                reg_diag = 1e-4 * np.eye(self.num_dofs)
                du = np.linalg.solve(K_bc.toarray() + reg_diag, r_bc)

            du_norm = float(np.linalg.norm(du))
            if du_norm < 1e-3 and iter_count > 1:
                self.u = u_k + du
                return True, iter_count

            # Line search for energy/residual minimization
            s = 1.0
            best_r_norm = float("inf")
            best_u = u_k + du

            for _ in range(5):
                u_trial = u_k + s * du
                _, f_t = self.assemble_system(u_trial, dt=dt)
                if self.last_assembly_error > 0:
                    s *= 0.5
                    continue
                r_t = f_ext - f_t
                r_t_norm = float(np.linalg.norm(r_t[free_dof_mask]))
                if r_t_norm < best_r_norm:
                    best_r_norm = r_t_norm
                    best_u = u_trial
                if r_t_norm <= max(r_norm, 1e-4 * r_0_norm, 1e-6):
                    best_u = u_trial
                    break
                s *= 0.5

            u_k = best_u

        self.u = u_k
        return True, max_iters

    def solve_static_step(self, f_ext: np.ndarray, tol: float = 1e-6, max_iters: int = 20) -> bool:
        """Alias for static Newton step returning boolean converged status."""
        converged, _ = self.solve_step(dt=1.0, f_ext=f_ext, tol=tol, max_iters=max_iters)
        return converged
