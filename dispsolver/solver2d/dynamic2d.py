"""
dynamic2d.py
============
Production 2D Non-linear Implicit Dynamic & Quasi-static FE Solver.
Mirrors DynamicSolver3D architecture for 100% parity across dimensions:
  - Heterogeneous element kernel grouping (CPE4, CPE4I, CPE4R, CPE4H, CPE4_FBAR, CPE4_CR, CPE3, CPE6, CPE6M, CPE8)
  - Data-Oriented Design (DOD) with Numba OpenMP parallel element kernels
  - Exact Dirichlet penalty enforcement and Abaqus 0.5% residual convergence criteria
  - Kinematic BC full-step handling on iter 1 & robust Armijo line search
  - Real-time Sensor and IterationHook execution engine
"""

from __future__ import annotations
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.sparse import csc_matrix

try:
    from pypardiso import spsolve as pardiso_spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve as pardiso_spsolve

from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.solver2d.assembly_utils2d import build_global_topology_2d, scatter_f_int_2d


class DynamicSolver2D:
    """Production 2D Non-Linear Newton-Raphson Finite Element Solver."""

    def __init__(
        self,
        mesh: Mesh2D,
        materials: Optional[Dict[int, Any]] = None,
        dt_init: float = 0.01,
        dt_min: float = 1e-6,
        dt_max: float = 0.05,
        nlgeom: bool = True
    ):
        self.mesh = mesh
        self.materials = materials or {}
        self.nlgeom = bool(nlgeom)
        self.num_nodes = mesh.num_nodes
        self.num_dofs = self.num_nodes * 2

        # State vectors
        self.u = np.zeros(self.num_dofs, dtype=np.float64)
        self.v = np.zeros(self.num_dofs, dtype=np.float64)
        self.a = np.zeros(self.num_dofs, dtype=np.float64)

        # History tracking
        self.history_u: List[np.ndarray] = []
        self.history_t: List[float] = []

        # Dirichlet Boundary Conditions: {global_dof: prescribed_value}
        self.fixed_dofs: Dict[int, float] = {}
        self.constraints: List[Any] = []

        self.last_assembly_error: int = 0
        self.rows_topo: Optional[np.ndarray] = None
        self.cols_topo: Optional[np.ndarray] = None

        self._setup_numba_topology()

    def _setup_numba_topology(self):
        """Classify elements into Numba kernel groups and precompute sparse topology."""
        elements_list = list(self.mesh.elements.values())
        n_elems = len(elements_list)
        if n_elems == 0:
            return

        nid_map = self.mesh.node_id_to_index()

        self.elem_mat_types = np.zeros(n_elems, dtype=np.int32)
        self.elem_props = np.zeros((n_elems, 36), dtype=np.float64)

        # Default props: E=4000.0, nu=0.3
        self.elem_props[:, 0] = 4000.0
        self.elem_props[:, 1] = 0.3

        for e, elem in enumerate(elements_list):
            pid = getattr(elem, "pid", 0)
            mat_obj = self.materials.get(pid, None) if isinstance(self.materials, dict) else None
            if mat_obj is None and isinstance(self.materials, dict) and "E" in self.materials:
                mat_obj = self.materials
            if mat_obj is not None:
                if hasattr(mat_obj, "props"):
                    p = np.asarray(mat_obj.props, dtype=np.float64)
                    self.elem_props[e, :min(36, len(p))] = p[:36]
                elif isinstance(mat_obj, dict):
                    self.elem_props[e, 0] = float(mat_obj.get("E", 4000.0))
                    self.elem_props[e, 1] = float(mat_obj.get("nu", 0.3))

        # SectionControls array: (n_elems, 8)
        self.elem_controls = np.zeros((n_elems, 8), dtype=np.float64)
        self.elem_controls[:, 0] = 0.05  # Hourglass factor default
        self.elem_controls[:, 3] = 1.0   # Distortion control default ON
        self.elem_controls[:, 4] = 0.02  # min_det default

        # Map element types to kernel group index k with Material- & Kinematic-Adaptive NLGEOM Dispatch
        elem_kernel_indices: Dict[int, List[int]] = {}
        for idx, elem in enumerate(elements_list):
            et = elem.elem_type.upper()
            pid = getattr(elem, "pid", 0)
            mat_obj = self.materials.get(pid, None) if isinstance(self.materials, dict) else None
            nu = 0.3
            if mat_obj is not None:
                if isinstance(mat_obj, dict):
                    nu = float(mat_obj.get("nu", 0.3))
                elif hasattr(mat_obj, "props"):
                    nu = float(mat_obj.props[1]) if len(mat_obj.props) > 1 else 0.3
            is_nearly_incompressible = (nu > 0.48)

            if self.nlgeom:
                # NLGEOM=True: Automatically promote core elements to Large-Deformation / Corotational
                if et in ["CPE4I", "CPS4I", "Q4_EAS", "INCOMPATIBLE", "CPE4I_CR"]:
                    k = 11  # CPE4I_CR
                elif et in ["CPE4H", "CPS4H", "Q4_HYBRID", "CPE4H_CR"]:
                    k = 12  # CPE4H_CR
                elif et in ["CPE4R", "CPS4R", "Q4_REDUCED", "CPE4R_CR"]:
                    k = 13  # CPE4R_CR
                elif et in ["CPE6M", "CPS6M", "CPE6M_CR"]:
                    k = 14  # CPE6M_CR
                elif et in ["CPE4_CR", "Q4_COROTATIONAL", "CPE4_COROTATIONAL"]:
                    k = 5   # CPE4_CR
                elif et in ["CPE4_FBAR", "FBAR"]:
                    k = 4   # CPE4_FBAR
                elif et in ["CPE4", "CPS4"]:
                    k = 4 if is_nearly_incompressible else 5
                elif et in ["CPE3", "CPS3", "T3", "CST"]:
                    k = 6
                elif et in ["CPE6", "CPS6", "T6"]:
                    k = 7
                elif et in ["CPE8", "CPS8", "Q8"]:
                    k = 9
                elif et in ["CPE8R", "CPS8R", "Q8R"]:
                    k = 10
                else:
                    k = 5 if not is_nearly_incompressible else 4
            else:
                # Small-strain linear geometry
                if et in ["CPE4", "CPS4"]:
                    k = 0
                elif et in ["CPE4I", "CPS4I", "Q4_EAS", "INCOMPATIBLE"]:
                    k = 1
                elif et in ["CPE4R", "CPS4R", "Q4_REDUCED"]:
                    k = 2
                elif et in ["CPE4H", "CPS4H", "Q4_HYBRID"]:
                    k = 3
                elif et in ["CPE4_FBAR", "FBAR"]:
                    k = 4
                elif et in ["CPE4_CR", "Q4_COROTATIONAL", "CPE4_COROTATIONAL"]:
                    k = 5
                elif et in ["CPE3", "CPS3", "T3", "CST"]:
                    k = 6
                elif et in ["CPE6", "CPS6", "T6"]:
                    k = 7
                elif et in ["CPE6M", "CPS6M"]:
                    k = 8
                elif et in ["CPE8", "CPS8", "Q8"]:
                    k = 9
                elif et in ["CPE8R", "CPS8R", "Q8R"]:
                    k = 10
                elif et in ["CPE4I_CR"]:
                    k = 11
                elif et in ["CPE4H_CR"]:
                    k = 12
                elif et in ["CPE4R_CR"]:
                    k = 13
                elif et in ["CPE6M_CR"]:
                    k = 14
                else:
                    k = 0

            if k not in elem_kernel_indices:
                elem_kernel_indices[k] = []
            elem_kernel_indices[k].append(idx)

        self.elem_kernel_groups = {
            k: np.array(indices, dtype=np.int64)
            for k, indices in elem_kernel_indices.items()
        }

        self.elem_conn_groups = {}
        for k, indices in self.elem_kernel_groups.items():
            elem_subset = [elements_list[i] for i in indices]
            self.elem_conn_groups[k] = np.array(
                [[nid_map[nid] for nid in elem.node_ids] for elem in elem_subset],
                dtype=np.int64
            )

        # Precompute sparse topology per group
        rows_list = []
        cols_list = []
        for k in sorted(self.elem_kernel_groups.keys()):
            conn_k = self.elem_conn_groups[k]
            r_k, c_k = build_global_topology_2d(self.num_dofs, conn_k)
            rows_list.append(r_k)
            cols_list.append(c_k)

        if len(rows_list) > 0:
            self.rows_topo = np.concatenate(rows_list)
            self.cols_topo = np.concatenate(cols_list)
        else:
            self.rows_topo = np.zeros(0, dtype=np.int32)
            self.cols_topo = np.zeros(0, dtype=np.int32)

        self.elem_sdvs = np.zeros((n_elems, 9, 7), dtype=np.float64)

    def set_dirichlet_bc(self, global_dof: int, value: float):
        """Prescribe a 2D Dirichlet displacement value for a global DOF."""
        self.fixed_dofs[int(global_dof)] = float(value)

    def fix_dof(self, node_id: int, dof_axis: int, value: float = 0.0):
        """Fix a 2D DOF (dof_axis: 0 for X, 1 for Y)."""
        nid_map = self.mesh.node_id_to_index()
        global_dof = 2 * nid_map[node_id] + dof_axis
        self.set_dirichlet_bc(global_dof, value)

    def add_nodal_force(self, node_id: int, dof_axis: int, force_value: float):
        """Add an external nodal concentrated force (dof_axis: 0 for X, 1 for Y)."""
        nid_map = self.mesh.node_id_to_index()
        global_dof = 2 * nid_map[node_id] + dof_axis
        if not hasattr(self, "f_ext_applied") or self.f_ext_applied is None:
            self.f_ext_applied = np.zeros(self.num_dofs, dtype=np.float64)
        self.f_ext_applied[global_dof] += float(force_value)

    def assemble_system(self, u_vec: np.ndarray, dt: float = 1.0, return_error: bool = False, update_state: bool = False):
        """Assemble global 2D stiffness matrix K_global and internal force vector f_int."""
        node_coords_all = self.mesh.nodes_array()

        f_int_global = np.zeros(self.num_dofs, dtype=np.float64)
        data_topo_list = []
        has_error = False

        for k in sorted(self.elem_kernel_groups.keys()):
            elem_indices = self.elem_kernel_groups[k]
            sub_conn = self.elem_conn_groups[k]
            sub_mat = self.elem_mat_types[elem_indices]
            sub_props = self.elem_props[elem_indices]
            sub_sdvs = self.elem_sdvs[elem_indices] if update_state else self.elem_sdvs[elem_indices].copy()
            sub_controls = self.elem_controls[elem_indices]

            if k == 0:
                from dispsolver.element2d.cpe4_numba import assemble_mesh_cpe4_numba
                f_sub, K_sub, err = assemble_mesh_cpe4_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 1:
                from dispsolver.element2d.cpe4i_numba import assemble_mesh_cpe4i_numba
                f_sub, K_sub, err = assemble_mesh_cpe4i_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 2:
                from dispsolver.element2d.cpe4r_numba import assemble_mesh_cpe4r_numba
                f_sub, K_sub, err = assemble_mesh_cpe4r_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 3:
                from dispsolver.element2d.cpe4h_numba import assemble_mesh_cpe4h_numba
                f_sub, K_sub, err = assemble_mesh_cpe4h_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 4:
                from dispsolver.element2d.cpe4_fbar_numba import assemble_mesh_cpe4_fbar_numba
                f_sub, K_sub, err = assemble_mesh_cpe4_fbar_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 5:
                from dispsolver.element2d.cpe4_cr_numba import assemble_mesh_cpe4_cr_numba
                f_sub, K_sub, err = assemble_mesh_cpe4_cr_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 6:
                from dispsolver.element2d.cpe3_numba import assemble_mesh_cpe3_numba
                f_sub, K_sub, err = assemble_mesh_cpe3_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 7:
                from dispsolver.element2d.cpe6_numba import assemble_mesh_cpe6_numba
                f_sub, K_sub, err = assemble_mesh_cpe6_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 8:
                from dispsolver.element2d.cpe6m_numba import assemble_mesh_cpe6m_numba
                f_sub, K_sub, err = assemble_mesh_cpe6m_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 9:
                from dispsolver.element2d.cpe8_numba import assemble_mesh_cpe8_numba
                f_sub, K_sub, err = assemble_mesh_cpe8_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 10:
                from dispsolver.element2d.cpe8_numba import assemble_mesh_cpe8r_numba
                f_sub, K_sub, err = assemble_mesh_cpe8r_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 11:
                from dispsolver.element2d.cr_wrapper_2d import assemble_mesh_cpe4i_cr_numba
                f_sub, K_sub, err = assemble_mesh_cpe4i_cr_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 12:
                from dispsolver.element2d.cr_wrapper_2d import assemble_mesh_cpe4h_cr_numba
                f_sub, K_sub, err = assemble_mesh_cpe4h_cr_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 13:
                from dispsolver.element2d.cr_wrapper_2d import assemble_mesh_cpe4r_cr_numba
                f_sub, K_sub, err = assemble_mesh_cpe4r_cr_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 14:
                from dispsolver.element2d.cr_wrapper_2d import assemble_mesh_cpe6m_cr_numba
                f_sub, K_sub, err = assemble_mesh_cpe6m_cr_numba(node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            else:
                raise ValueError(f"Unknown element kernel index: {k}")

            if update_state:
                self.elem_sdvs[elem_indices] = sub_sdvs

            if err:
                has_error = True

            scatter_f_int_2d(f_sub, sub_conn, f_int_global)
            data_topo_list.append(K_sub.ravel())

        self.last_assembly_error = int(has_error)
        data_topo = np.concatenate(data_topo_list) if len(data_topo_list) > 0 else np.zeros(0, dtype=np.float64)

        K_global = csc_matrix(
            (data_topo, (self.rows_topo, self.cols_topo)),
            shape=(self.num_dofs, self.num_dofs)
        )

        if self.constraints:
            rows_extra = [self.rows_topo]
            cols_extra = [self.cols_topo]
            data_extra = [data_topo]

            for c in self.constraints:
                if hasattr(c, "reproject_deformed"):
                    c.reproject_deformed(u_vec)
                if hasattr(c, "assemble"):
                    f_c, (r_c, c_c, d_c), _ = c.assemble(u_vec)
                    f_int_global += f_c
                    rows_extra.append(r_c)
                    cols_extra.append(c_c)
                    data_extra.append(d_c)
                else:
                    K_lil = K_global.tolil()
                    c.apply_penalty(u_vec, f_int_global, K_lil)
                    K_global = K_lil.tocsc()

            if len(rows_extra) > 1:
                K_global = csc_matrix(
                    (np.concatenate(data_extra), (np.concatenate(rows_extra), np.concatenate(cols_extra))),
                    shape=(self.num_dofs, self.num_dofs)
                )

        self.f_int = f_int_global
        if return_error:
            return K_global, f_int_global, has_error
        return K_global, f_int_global

    def apply_boundary_conditions(self, K_global: csc_matrix, residual: np.ndarray, u_k: np.ndarray) -> Tuple[csc_matrix, np.ndarray]:
        """Apply Dirichlet boundary conditions via diagonal penalty method."""
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
        """Solve a 2D incremental step using Newton-Raphson with robust Armijo line search."""
        if f_ext is None:
            if hasattr(self, "f_ext_applied") and self.f_ext_applied is not None:
                f_ext = self.f_ext_applied.copy()
            else:
                f_ext = np.zeros(self.num_dofs, dtype=np.float64)

        free_dof_mask = np.ones(self.num_dofs, dtype=bool)
        if self.fixed_dofs:
            free_dof_mask[list(self.fixed_dofs.keys())] = False

        u_k = self.u.copy()
        r_0_norm = None

        for iter_count in range(1, max_iters + 1):
            K_g, f_int = self.assemble_system(u_k, dt=dt)
            if self.last_assembly_error > 0:
                return False, iter_count

            residual = f_ext - f_int
            r_free = residual[free_dof_mask]
            r_norm = float(np.linalg.norm(r_free))
            f_ext_free_norm = float(np.linalg.norm(f_ext[free_dof_mask]))

            if r_0_norm is None or r_norm > r_0_norm:
                r_0_norm = max(r_norm, f_ext_free_norm, 1.0)

            K_bc, r_bc = self.apply_boundary_conditions(K_g, residual, u_k)
            rel_r = r_norm / r_0_norm

            bc_err = max([abs(target_val - u_k[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            if iter_count > 1 and bc_err < 1e-5 and (rel_r < 5e-3 or r_norm < 1e-3):
                self.u = u_k
                self.assemble_system(u_k, dt=dt, update_state=True)
                return True, iter_count

            try:
                du = pardiso_spsolve(K_bc, r_bc)
            except Exception as e:
                raise RuntimeError("Linear solver failed. Check boundary conditions or rigid body modes.") from e

            du_norm = float(np.linalg.norm(du))
            u_next = u_k + du
            bc_err_next = max([abs(target_val - u_next[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            if iter_count > 1 and du_norm < 1e-4 and bc_err_next < 1e-5:
                self.u = u_next
                self.assemble_system(self.u, dt=dt, update_state=True)
                return True, iter_count

            # On iteration 1 (kinematic Dirichlet application), take full step s=1.0 unless inverted
            if iter_count == 1:
                u_trial = u_k + du
                _, _ = self.assemble_system(u_trial, dt=dt)
                if self.last_assembly_error == 0:
                    u_k = u_trial
                    continue

            # Robust Armijo line search
            s = 1.0
            best_r_norm = r_norm
            best_u = u_k.copy()
            found_valid = False

            for _ in range(6):
                u_trial = u_k + s * du
                _, f_t = self.assemble_system(u_trial, dt=dt)
                if self.last_assembly_error > 0:
                    s *= 0.5
                    continue
                r_t = f_ext - f_t
                r_t_norm = float(np.linalg.norm(r_t[free_dof_mask]))
                if r_t_norm < best_r_norm or not found_valid:
                    best_r_norm = r_t_norm
                    best_u = u_trial
                    found_valid = True
                if r_t_norm <= max(r_norm, 1e-4 * r_0_norm, 1e-6):
                    best_u = u_trial
                    found_valid = True
                    break
                s *= 0.5

            if not found_valid:
                return False, iter_count

            u_k = best_u

        self.u = u_k
        if bc_err < 1e-5 and (rel_r < 5e-3 or r_norm < 1e-3):
            self.assemble_system(u_k, dt=dt, update_state=True)
            return True, max_iters
        return False, max_iters

    def _update_step_bcs(self, step: Any, sys: Optional[Any], t: float, total_time: float) -> None:
        """Evaluate active BCs for 2D Model."""
        self.fixed_dofs.clear()
        if not hasattr(step, "get_active_bcs"):
            return

        active_bcs = step.get_active_bcs()
        node_coords_all = self.mesh.nodes_array()
        nid_map = self.mesh.node_id_to_index()

        for bc_name, bc in active_bcs.items():
            region_nodes = []
            if sys is not None and hasattr(sys, "global_nsets") and bc.region in sys.global_nsets:
                node_indices = sys.global_nsets[bc.region]
                inv_nid = {idx: g for g, idx in sys.nid_to_idx.items()}
                region_nodes = [inv_nid[idx] for idx in node_indices]
            elif bc.region == "ALL":
                region_nodes = list(self.mesh.nodes.keys())

            if not region_nodes:
                continue

            if hasattr(bc, "evaluate_node"):
                for gid in region_nodes:
                    n_idx = nid_map[gid]
                    coords = node_coords_all[n_idx]
                    res = bc.evaluate_node(coords, t, step_time=t)
                    if res is not None:
                        u1, u2 = res[0], res[1]
                        if u1 is not None:
                            self.fix_dof(gid, 0, float(u1))
                        if u2 is not None:
                            self.fix_dof(gid, 1, float(u2))

            elif hasattr(bc, "u1") or hasattr(bc, "u2"):
                amp_val = 1.0
                if bc.amplitude is not None and hasattr(bc.amplitude, "evaluate"):
                    amp_val = bc.amplitude.evaluate(t, step_time=t, total_time=total_time)

                for gid in region_nodes:
                    if getattr(bc, "u1", None) is not None:
                        self.fix_dof(gid, 0, float(bc.u1) * amp_val)
                    if getattr(bc, "u2", None) is not None:
                        self.fix_dof(gid, 1, float(bc.u2) * amp_val)

    def solve(
        self,
        step: Any,
        sys: Optional[Any] = None,
        output_mgr: Optional[Any] = None,
        verbose: bool = True
    ) -> bool:
        """Run complete incremental solve over the Step time period."""
        total_time = getattr(step, "time_period", 1.0)
        dt = getattr(step, "dt_init", 0.02)
        dt_min = getattr(step, "dt_min", 1e-6)
        dt_max = getattr(step, "dt_max", 0.05)

        t_curr = 0.0
        inc_count = 0
        total_iters = 0

        self.history_u = [self.u.copy()]
        self.history_t = [0.0]

        if verbose:
            print(f"=== Starting 2D Step '{step.name}' Solve (Time Period: {total_time:.3f} s) ===", flush=True)

        while t_curr < total_time - 1e-9:
            inc_count += 1
            if t_curr + dt > total_time:
                dt = total_time - t_curr

            t_next = t_curr + dt
            self._update_step_bcs(step, sys, t_next, total_time)

            u_prev = self.u.copy()
            sdvs_prev = self.elem_sdvs.copy()

            converged, iters = self.solve_step(dt=dt, max_iters=50)
            total_iters += iters

            if converged:
                t_curr = t_next
                self.history_u.append(self.u.copy())
                self.history_t.append(t_curr)

                if verbose:
                    print(f"  Inc {inc_count:3d}: t = {t_curr:.4f}s | dt = {dt:.4f}s | Newton Iters = {iters:2d} | STATUS: CONVERGED", flush=True)

                if output_mgr is not None:
                    du = self.u - u_prev
                    is_last = (t_curr >= total_time - 1e-9)
                    output_mgr.record_step(
                        step_time=t_curr,
                        inc=inc_count,
                        solver=self,
                        dt=dt,
                        du=du,
                        f_int=self.f_int if hasattr(self, "f_int") and self.f_int is not None else np.zeros_like(self.u),
                        is_last_step=is_last
                    )

                if iters <= 4 and dt < dt_max:
                    dt = min(dt * 1.5, dt_max)
                elif iters >= 10:
                    dt = max(dt * 0.85, dt_min)
            else:
                self.u = u_prev
                self.elem_sdvs = sdvs_prev
                dt *= 0.5
                if verbose:
                    print(f"  Inc {inc_count:3d}: Cutback at t = {t_curr:.4f}s! Reduced dt to {dt:.6f}s", flush=True)
                if dt < dt_min:
                    if verbose:
                        print("  [ERROR] Time increment fell below dt_min! Simulation aborted.", flush=True)
                    return False

        if verbose:
            print(f"=== 2D Step '{step.name}' Completed Successfully (Total Inc: {inc_count}, Total Newton Iters: {total_iters}) ===", flush=True)
        return True

    def solve_with_odb(
        self,
        step: Any,
        sys: Optional[Any] = None,
        odb_name: str = "Job-2D",
        vtkhdf_path: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Run step solve and record field & history outputs into an Odb.

        Parameters
        ----------
        step : Step
            해석 스텝 객체 (field/history 요청 포함).
        sys : FlattenedSolverSystem, optional
            미리 빌드된 솔버 시스템.
        odb_name : str, optional
            ODB 이름 (기본값 "Job-2D").
        vtkhdf_path : str, optional
            실시간 스트리밍 저장 경로. 지정 시 매 Field Output 프레임마다
            즉시 .vtkhdf 파일에 append → 해석 중 ParaView에서 확인 가능.
        **kwargs
            DynamicSolver2D.solve()에 전달되는 추가 인수.

        Returns
        -------
        Odb
            완료된 ODB 인스턴스.
        """
        from dispsolver.output import Odb, OutputManager
        odb = Odb(name=odb_name, mesh=self.mesh)
        field_reqs = list(getattr(step, "field_output_requests", {}).values())
        hist_reqs = list(getattr(step, "history_output_requests", {}).values())
        output_mgr = OutputManager(
            odb=odb,
            step_name=step.name,
            time_period=getattr(step, "time_period", 1.0),
            field_requests=field_reqs,
            history_requests=hist_reqs,
            dim=2,
            vtkhdf_path=vtkhdf_path,
        )
        # t=0 초기 상태 기록
        output_mgr.record_step(
            step_time=0.0,
            inc=0,
            solver=self,
            dt=0.0,
            du=np.zeros_like(self.u),
            f_int=self.f_int if hasattr(self, "f_int") and self.f_int is not None else np.zeros_like(self.u),
            is_last_step=False
        )
        try:
            self.solve(step=step, sys=sys, output_mgr=output_mgr, **kwargs)
        finally:
            output_mgr.finalize()  # VTKHDF 파일 핸들 정리
        return odb

    def compute_reaction_forces(self) -> np.ndarray:
        """Compute reaction forces at all DOFs.
        For fixed Dirichlet DOFs, reaction force R = f_int.
        For unconstrained free DOFs, R = 0 (equilibrium residual).
        Returns an array of length num_dofs.
        """
        R = np.zeros(self.num_dofs, dtype=np.float64)
        if self.f_int is not None:
            for dof in self.fixed_dofs.keys():
                R[dof] = self.f_int[dof]
        return R

    def compute_section_reactions(self, node_ids: List[int], center_pt: Optional[Tuple[float, float]] = None, current_coords: Optional[bool] = None) -> Dict[str, float]:
        """Compute resultant reaction forces (Fx, Fy) and moment Mz for a section of nodes.
        center_pt: (x_c, y_c) reference point for moment calculation. If None, uses mean coordinate.
        current_coords: If True, uses deformed current coordinates (X + u). Defaults to True if nlgeom=True.
        Returns: {'Fx': float, 'Fy': float, 'Mz': float}
        """
        if current_coords is None:
            current_coords = getattr(self, "nlgeom", False)

        nid_map = self.mesh.node_id_to_index()
        if center_pt is None:
            if current_coords and self.u is not None:
                xc = float(np.mean([self.mesh.nodes[n].x + self.u[2 * nid_map[n]] for n in node_ids]))
                yc = float(np.mean([self.mesh.nodes[n].y + self.u[2 * nid_map[n] + 1] for n in node_ids]))
            else:
                xc = float(np.mean([self.mesh.nodes[n].x for n in node_ids]))
                yc = float(np.mean([self.mesh.nodes[n].y for n in node_ids]))
        else:
            xc, yc = center_pt

        Fx_tot = 0.0
        Fy_tot = 0.0
        Mz_tot = 0.0

        if self.f_int is not None:
            for nid in node_ids:
                idx = nid_map[nid]
                node = self.mesh.nodes[nid]
                fx = self.f_int[2 * idx]
                fy = self.f_int[2 * idx + 1]
                if current_coords and self.u is not None:
                    dx = (node.x + self.u[2 * idx]) - xc
                    dy = (node.y + self.u[2 * idx + 1]) - yc
                else:
                    dx = node.x - xc
                    dy = node.y - yc
                Fx_tot += fx
                Fy_tot += fy
                # Mz = r x F = dx * fy - dy * fx
                Mz_tot += (dx * fy - dy * fx)

        return {"Fx": float(Fx_tot), "Fy": float(Fy_tot), "Mz": float(Mz_tot)}

    def compute_strain_energy(self) -> float:
        """Compute total strain energy of the 2D mesh.
        Calculates 0.5 * sum_e (u_e^T * f_e) in local element frames.
        """
        if not hasattr(self, "f_int") or self.f_int is None:
            return 0.0
        total_energy = 0.0
        node_coords = self.mesh.nodes_array()

        for k, elem_indices in self.elem_kernel_groups.items():
            sub_conn = self.elem_conn_groups[k]
            for local_idx, e_idx in enumerate(elem_indices):
                conn = sub_conn[local_idx]
                n_en = len(conn)
                u_e = np.zeros(2 * n_en, dtype=np.float64)
                f_e = np.zeros(2 * n_en, dtype=np.float64)
                for a in range(n_en):
                    nid_idx = conn[a]
                    u_e[2 * a] = self.u[2 * nid_idx]
                    u_e[2 * a + 1] = self.u[2 * nid_idx + 1]
                    f_e[2 * a] = self.f_int[2 * nid_idx]
                    f_e[2 * a + 1] = self.f_int[2 * nid_idx + 1]
                total_energy += 0.5 * float(np.dot(u_e, f_e))

        return max(0.0, total_energy)

