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

from dispsolver.model.regions import describe as _describe
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

try:
    from pypardiso import spsolve as pardiso_spsolve
except ImportError:
    from scipy.sparse.linalg import spsolve as pardiso_spsolve

from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.solver2d.assembly_utils2d import build_global_topology_2d, scatter_f_int_2d
from dispsolver.solver3d.assembly_utils import (
    build_coo_to_csr_mapping,
    scatter_k_csr_direct,
    apply_dirichlet_bcs_csr_fast,
)



# Small-strain plane kernels co-rotated under NLGEOM (element2d/cr_generic_2d):
# 6 CPE3, 7 CPE6, 9 CPE8, 10 CPE8R, 18 T2D2.
from dispsolver.model.element_catalog import cr_wrapped_kernels as _cr_wrapped_kernels
_CR_WRAPPED_KERNELS_2D = _cr_wrapped_kernels(2)

class DynamicSolver2D:
    """Production 2D Non-Linear Newton-Raphson Finite Element Solver."""

    def __init__(
        self,
        mesh: Mesh2D,
        materials: Optional[Dict[int, Any]] = None,
        dt_init: float = 0.01,
        dt_min: float = 1e-6,
        dt_max: float = 0.05,
        nlgeom: bool = True,
        time_integrator: str = "quasistatic",
        alpha_hht: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        alpha_m: Optional[float] = None,
        alpha_f: Optional[float] = None,
        rho_inf: Optional[float] = None,
        density: Optional[float] = None,
        alpha_m_damp: float = 0.0,
        beta_k_damp: float = 0.0,
    ):
        self.mesh = mesh
        self.materials = materials or {}
        self.nlgeom = bool(nlgeom)
        self.density = float(density) if density is not None else 7.85e-9
        self.alpha_m_damp = float(alpha_m_damp)
        self.beta_k_damp = float(beta_k_damp)

        from dispsolver.solver.time_integrator_presets import resolve_time_integrator
        self.config_integrator = resolve_time_integrator(
            time_integrator=time_integrator,
            alpha_hht=alpha_hht,
            beta=beta,
            gamma=gamma,
            alpha_m=alpha_m,
            alpha_f=alpha_f,
            rho_inf=rho_inf,
        )
        self.time_integrator = self.config_integrator.name
        self.alpha_hht = self.config_integrator.alpha_hht
        self.beta_hht = self.config_integrator.beta
        self.gamma_hht = self.config_integrator.gamma
        self.alpha_m = self.config_integrator.alpha_m
        self.alpha_f = self.config_integrator.alpha_f

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
        self.f_int = np.zeros(self.num_dofs, dtype=np.float64)
        self.last_accept = None
        self.rows_topo: Optional[np.ndarray] = None
        self.cols_topo: Optional[np.ndarray] = None

        self.current_temperature: float = 20.0
        self.ref_temperature: float = 20.0
        self._f_thermal_cached: Optional[np.ndarray] = None
        self._thermal_dirty: bool = True
        self.node_transforms: Optional[np.ndarray] = None
        self.has_nodal_transforms: bool = False

        self._setup_numba_topology()


    def set_nodal_transforms(self, node_transforms: np.ndarray):
        """Set nodal rotation matrices (N_nodes, 2, 2) for Abaqus *TRANSFORM."""
        dim = 2
        arr = np.asarray(node_transforms, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[1] >= 2 and arr.shape[2] >= 2:
            self.node_transforms = arr[:, :2, :2].copy()
        else:
            self.node_transforms = arr
        n_nodes = len(self.node_transforms)
        I_dim = np.eye(dim, dtype=np.float64)
        self.has_nodal_transforms = any(not np.allclose(self.node_transforms[i], I_dim) for i in range(n_nodes))

    def get_global_displacements(self, u_vec: np.ndarray) -> np.ndarray:
        """Convert nodal local DOFs u_vec to global physical displacements u_global."""
        if not self.has_nodal_transforms or self.node_transforms is None:
            return u_vec.copy()
        from dispsolver.model.nodal_transform import transform_u_to_global
        return transform_u_to_global(u_vec, self.node_transforms, dim=2)



    def _setup_numba_topology(self):
        """Classify elements into Numba kernel groups and precompute sparse topology."""
        elements_list = list(self.mesh.elements.values())
        n_elems = len(elements_list)
        if n_elems == 0:
            return

        nid_map = self.mesh.node_id_to_index()

        self.elem_mat_types = np.zeros(n_elems, dtype=np.int32)
        self.elem_props = np.zeros((n_elems, 64), dtype=np.float64)

        # Default props: E=4000.0, nu=0.3
        self.elem_props[:, 0] = 4000.0
        self.elem_props[:, 1] = 0.3

        self.elem_density = np.zeros(n_elems, dtype=np.float64)
        default_rho = self.density if getattr(self, "density", None) is not None else 7.85e-9

        for e, elem in enumerate(elements_list):
            pid = getattr(elem, "pid", 0)
            mat_obj = self.materials.get(pid, None) if isinstance(self.materials, dict) else None
            if mat_obj is None and isinstance(self.materials, dict) and ("E" in self.materials or "density" in self.materials):
                mat_obj = self.materials
            
            rho_val = default_rho
            if mat_obj is not None:
                if hasattr(mat_obj, "to_numba_props"):
                    mat_obj = mat_obj.to_numba_props()
                if hasattr(mat_obj, "mat_type") and hasattr(mat_obj, "props"):
                    self.elem_mat_types[e] = int(mat_obj.mat_type)
                    p = np.asarray(mat_obj.props, dtype=np.float64)
                    self.elem_props[e, :min(64, len(p))] = p[:64]
                    if hasattr(mat_obj, "density"):
                        rho_val = float(getattr(mat_obj, "density"))
                elif isinstance(mat_obj, dict):
                    mtype = mat_obj.get("mat_type", mat_obj.get("type", 1))
                    E_val = float(mat_obj.get("E", 4000.0))
                    nu_val = float(mat_obj.get("nu", 0.3))
                    rho_val = float(mat_obj.get("density", mat_obj.get("rho", default_rho)))

                    if mtype in [12, "TABULAR_PLASTIC", "RADIOSS_LAW36"] or "tabular" in mat_obj or "tabular_plastic" in mat_obj:
                        self.elem_mat_types[e] = 12  # MAT_J2_PLASTICITY_TABULAR
                        table = mat_obj.get("tabular", mat_obj.get("tabular_plastic", []))
                        eps_fail = float(mat_obj.get("eps_fail", 1e9))
                        sigma_fail = float(mat_obj.get("sigma_fail", 0.0))
                        n_pts = min(len(table), 10)
                        self.elem_props[e, 0] = E_val
                        self.elem_props[e, 1] = nu_val
                        self.elem_props[e, 2] = float(n_pts)
                        self.elem_props[e, 3] = eps_fail
                        self.elem_props[e, 4] = sigma_fail
                        for p_idx in range(n_pts):
                            self.elem_props[e, 5 + 2 * p_idx] = float(table[p_idx][0])
                            self.elem_props[e, 6 + 2 * p_idx] = float(table[p_idx][1])
                    elif mtype in [7, "J2_PLASTICITY_NONLINEAR"] or "R_inf" in mat_obj or "b_exp" in mat_obj:
                        self.elem_mat_types[e] = 7  # MAT_J2_PLASTICITY_NONLINEAR
                        sy_val = float(mat_obj.get("yield_stress", mat_obj.get("sigma_y0", 400.0)))
                        H_val = float(mat_obj.get("hardening_modulus", mat_obj.get("H", 0.0)))
                        R_inf = float(mat_obj.get("R_inf", 100.0))
                        b_exp = float(mat_obj.get("b_exp", 10.0))
                        self.elem_props[e, :6] = [E_val, nu_val, sy_val, H_val, R_inf, b_exp]
                    elif mtype in [2, "J2_PLASTICITY", "PLASTIC"] or str(mtype).lower() in ["j2", "plasticity", "j2_plasticity"] or "sigma_y0" in mat_obj or "yield_stress" in mat_obj:
                        self.elem_mat_types[e] = 2  # MAT_J2_PLASTICITY
                        sy_val = float(mat_obj.get("yield_stress", mat_obj.get("sigma_y0", 400.0)))
                        H_val = float(mat_obj.get("hardening_modulus", mat_obj.get("H", 0.0)))
                        self.elem_props[e, :4] = [E_val, nu_val, sy_val, H_val]
                    else:
                        self.elem_mat_types[e] = 0  # MAT_LINEAR_ELASTIC (id 1 is Neo-Hookean)
                        self.elem_props[e, 0] = E_val
                        self.elem_props[e, 1] = nu_val
                elif hasattr(mat_obj, "props"):
                    p = np.asarray(mat_obj.props, dtype=np.float64)
                    self.elem_props[e, :min(64, len(p))] = p[:64]
                    if hasattr(mat_obj, "density"):
                        rho_val = float(getattr(mat_obj, "density"))
                else:
                    self.elem_mat_types[e] = 0
                    self.elem_props[e, 0] = 4000.0
                    self.elem_props[e, 1] = 0.3
            else:
                self.elem_mat_types[e] = 0
                self.elem_props[e, 0] = 4000.0
                self.elem_props[e, 1] = 0.3

        # SectionControls array: (n_elems, 8)
        self.elem_controls = np.zeros((n_elems, 8), dtype=np.float64)
        self.elem_controls[:, 0] = 0.05  # Hourglass factor default
        self.elem_controls[:, 3] = 1.0   # Distortion control default ON
        self.elem_controls[:, 4] = 0.02  # min_det default

        self._build_kernel_groups()

        self.elem_sdvs = np.zeros((n_elems, 9, 7), dtype=np.float64)

    def _build_kernel_groups(self) -> None:
        """Assign every element its kernel from its type and the CURRENT
        self.nlgeom, then build the per-kernel connectivity and sparsity
        pattern. Split out of _setup_numba_topology so solve() can re-run it
        when a step's NLGEOM differs -- the 2D solver used to fix the kernels
        at construction, so every deck run (built without a step) used the
        NLGEOM=YES kernels whatever the step said."""
        nid_map = self.mesh.node_id_to_index()
        elements_list = list(self.mesh.elements.values())
        # Map element types to kernel group index k with Material- & Kinematic-Adaptive NLGEOM Dispatch
        # Element type -> kernel: model/element_catalog.py decides (NLGEOM,
        # plane stress, axisymmetry, hybrid/reduced substitutions included).
        from dispsolver.model.element_catalog import ElementCatalog
        elem_kernel_indices: Dict[int, List[int]] = {}
        failures: Dict[str, str] = {}
        subst: Dict[str, str] = {}
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
            linear_elastic = int(self.elem_mat_types[idx]) == 0
            try:
                k, notes = ElementCatalog.resolve(et, 2, bool(self.nlgeom), nu=nu, linear_elastic=linear_elastic)
            except NotImplementedError as exc:
                failures.setdefault(et, str(exc))
                continue
            if k is None:
                continue
            desc = ElementCatalog.lookup(et, 2)
            if desc is not None and desc.plane_stress and not (linear_elastic and abs(float(self.elem_props[idx, 1])) < 1e-15):
                # No 2D kernel implements plane stress (sigma_33 = 0); the
                # plane-strain kernels give the same answer only for a linear
                # elastic material with nu = 0.
                failures.setdefault(et, f"element type {et}: plane stress is not implemented in the 2D solver "
                                        f"(it would run as plane strain); only linear elastic nu=0 material is equivalent")
                continue
            if notes:
                subst[et] = "; ".join(notes)
            elem_kernel_indices.setdefault((k, len(elem.node_ids)), []).append(idx)

        if failures:
            raise NotImplementedError("; ".join(failures.values()))
        new = {et: why for et, why in subst.items() if et not in getattr(self, "element_substitutions", {})}
        self.element_substitutions = subst
        import warnings as _warnings
        for et, why in new.items():
            _warnings.warn(f"element type {et}: {why}")

        self.elem_kernel_groups = {
            g_key: np.array(indices, dtype=np.int64)
            for g_key, indices in elem_kernel_indices.items()
        }

        from dispsolver.model.element_catalog import KERNELS as _KERNELS
        self.elem_conn_groups = {}
        for g_key, indices in self.elem_kernel_groups.items():
            k_val, n_nodes_val = g_key
            n_use = min(n_nodes_val, _KERNELS[2][k_val].nodes)   # kernel consumes its own node count
            self.elem_conn_groups[g_key] = np.array(
                [[nid_map[nid] for nid in elements_list[i].node_ids[:n_use]] for i in indices],
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
            # Precompute static CSR sparsity template and 1D direct scatter mapping
            diags = np.arange(self.num_dofs, dtype=np.int32)
            rows_all = np.concatenate([self.rows_topo, diags])
            cols_all = np.concatenate([self.cols_topo, diags])
            data_dummy = np.zeros(len(rows_all), dtype=np.float64)
            coo_dummy = coo_matrix((data_dummy, (rows_all, cols_all)), shape=(self.num_dofs, self.num_dofs))
            self.csr_template = coo_dummy.tocsr()
            self.csr_indptr = self.csr_template.indptr
            self.csr_indices = self.csr_template.indices
            self.csr_data = np.zeros_like(self.csr_template.data)
            self.topo_to_csr_map = build_coo_to_csr_mapping(
                self.rows_topo, self.cols_topo, self.csr_indptr, self.csr_indices
            )
        else:
            self.rows_topo = np.zeros(0, dtype=np.int32)
            self.cols_topo = np.zeros(0, dtype=np.int32)
            self.csr_template = None
            self.csr_indptr = None
            self.csr_indices = None
            self.csr_data = None
            self.topo_to_csr_map = None


    def add_constraint(self, constraint: Any):
        """Add a constraint or surface tie object to the 2D solver."""
        # The assembly loop only knows constraints that assemble themselves.
        # Model-level *TIE / *RIGID BODY / *MPC records have no assemble():
        # the 2D loop skipped them (bodies silently untied), the 3D loop
        # crashed with an AttributeError deep in Newton.
        if not callable(getattr(constraint, "assemble", None)):
            raise NotImplementedError(
                f"constraint {getattr(constraint, 'name', '?')!r} of type {type(constraint).__name__} has no "
                f"solver implementation in this solver (no assemble())")
        self.constraints.append(constraint)

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

    def set_temperature(self, temp: float, ref_temp: Optional[float] = None) -> None:
        """Prescribe a global predefined temperature field (Abaqus *TEMPERATURE parity)."""
        self.current_temperature = float(temp)
        if ref_temp is not None:
            self.ref_temperature = float(ref_temp)
        self._thermal_dirty = True

    def compute_thermal_load_vector(self) -> np.ndarray:
        """Compute equivalent global thermal load vector for 2D plane strain elements.
        f_th = int B^T C eps_th dV.
        """
        if not self._thermal_dirty and self._f_thermal_cached is not None:
            return self._f_thermal_cached

        f_th = np.zeros(self.num_dofs, dtype=np.float64)
        if not self.mesh.elements:
            self._f_thermal_cached = f_th
            self._thermal_dirty = False
            return f_th

        from dispsolver.material.solver_thermal_adapter import compute_element_thermal_force_2d
        node_coords = self.mesh.nodes_array()
        elements_list = list(self.mesh.elements.values())
        nid_map = self.mesh.node_id_to_index()

        for e, elem in enumerate(elements_list):
            pid = getattr(elem, "pid", 0)
            mat_obj = self.materials.get(pid, None) if hasattr(self, "materials") and isinstance(self.materials, dict) else None
            alpha = 0.0
            t0 = self.ref_temperature
            E_val = 4000.0
            nu_val = 0.3

            if mat_obj is not None:
                if hasattr(mat_obj, "to_numba_props"):
                    m_props = mat_obj.to_numba_props()
                elif isinstance(mat_obj, dict):
                    m_props = mat_obj
                else:
                    m_props = {}
                alpha = float(m_props.get("alpha", 0.0))
                t0 = float(m_props.get("T0", self.ref_temperature))
                E_val = float(m_props.get("E", E_val))
                nu_val = float(m_props.get("nu", nu_val))
            elif hasattr(self, "material_params"):
                alpha = float(self.material_params.get("alpha", 0.0))
                t0 = float(self.material_params.get("T0", self.ref_temperature))
                E_val = float(self.material_params.get("E", E_val))
                nu_val = float(self.material_params.get("nu", nu_val))

            delta_T = self.current_temperature - t0
            if abs(alpha * delta_T) > 1e-20:
                conn_e = [nid_map[nid] for nid in elem.node_ids]
                if len(conn_e) == 4:
                    coords_e = node_coords[conn_e]
                    f_th_e = np.zeros(8, dtype=np.float64)
                    compute_element_thermal_force_2d(coords_e, alpha, delta_T, E_val, nu_val, f_th_e)
                    for a in range(4):
                        nid_idx = conn_e[a]
                        f_th[2 * nid_idx + 0] += f_th_e[2 * a + 0]
                        f_th[2 * nid_idx + 1] += f_th_e[2 * a + 1]

        self._f_thermal_cached = f_th
        self._thermal_dirty = False
        return f_th

    def assemble_system(self, u_vec: np.ndarray, dt: float = 1.0, return_error: bool = False, update_state: bool = False):
        """Assemble global 2D stiffness matrix K_global and internal force vector f_int."""
        node_coords_all = self.mesh.nodes_array()

        if self.has_nodal_transforms and self.node_transforms is not None:
            from dispsolver.model.nodal_transform import transform_u_to_global
            u_eval = transform_u_to_global(u_vec, self.node_transforms, dim=2)
        else:
            u_eval = u_vec

        f_int_global = np.zeros(self.num_dofs, dtype=np.float64)
        data_topo_list = []
        has_error = False

        for g_key in sorted(self.elem_kernel_groups.keys()):
            k = g_key[0] if isinstance(g_key, tuple) else g_key
            elem_indices = self.elem_kernel_groups[g_key]
            sub_conn = self.elem_conn_groups[g_key]
            sub_mat = self.elem_mat_types[elem_indices]
            sub_props = self.elem_props[elem_indices]
            sub_sdvs = self.elem_sdvs[elem_indices] if update_state else self.elem_sdvs[elem_indices].copy()
            sub_controls = self.elem_controls[elem_indices]

            # NLGEOM: small-strain plane kernels run in co-rotated element
            # frames (element2d/cr_generic_2d); they used to run unchanged.
            cr_cs = None
            if self.nlgeom and k in _CR_WRAPPED_KERNELS_2D:
                from dispsolver.element2d.cr_generic_2d import corotational_local_mesh_2d
                _cr_saved = (node_coords_all, sub_conn, u_eval)
                node_coords_all, sub_conn, u_eval, cr_cs = corotational_local_mesh_2d(
                    np.ascontiguousarray(node_coords_all[:, :2], dtype=np.float64),
                    np.ascontiguousarray(sub_conn, dtype=np.int64),
                    np.ascontiguousarray(u_eval, dtype=np.float64))

            if k == 0:
                from dispsolver.element2d.cpe4_numba import assemble_mesh_cpe4_numba
                f_sub, K_sub, err = assemble_mesh_cpe4_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 1:
                from dispsolver.element2d.cpe4i_numba import assemble_mesh_cpe4i_numba
                f_sub, K_sub, err = assemble_mesh_cpe4i_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 2:
                from dispsolver.element2d.cpe4r_numba import assemble_mesh_cpe4r_numba
                f_sub, K_sub, err = assemble_mesh_cpe4r_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 3:
                from dispsolver.element2d.cpe4h_numba import assemble_mesh_cpe4h_numba
                f_sub, K_sub, err = assemble_mesh_cpe4h_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 4:
                from dispsolver.element2d.cpe4_fbar_numba import assemble_mesh_cpe4_fbar_numba
                f_sub, K_sub, err = assemble_mesh_cpe4_fbar_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 5:
                from dispsolver.element2d.cpe4_cr_numba import assemble_mesh_cpe4_cr_numba
                f_sub, K_sub, err = assemble_mesh_cpe4_cr_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 6:
                from dispsolver.element2d.cpe3_numba import assemble_mesh_cpe3_numba
                f_sub, K_sub, err = assemble_mesh_cpe3_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 7:
                from dispsolver.element2d.cpe6_numba import assemble_mesh_cpe6_numba
                f_sub, K_sub, err = assemble_mesh_cpe6_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 8:
                from dispsolver.element2d.cpe6m_numba import assemble_mesh_cpe6m_numba
                f_sub, K_sub, err = assemble_mesh_cpe6m_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 9:
                from dispsolver.element2d.cpe8_numba import assemble_mesh_cpe8_numba
                f_sub, K_sub, err = assemble_mesh_cpe8_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 10:
                from dispsolver.element2d.cpe8_numba import assemble_mesh_cpe8r_numba
                f_sub, K_sub, err = assemble_mesh_cpe8r_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 11:
                from dispsolver.element2d.cr_wrapper_2d import assemble_mesh_cpe4i_cr_numba
                f_sub, K_sub, err = assemble_mesh_cpe4i_cr_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 12:
                from dispsolver.element2d.cr_wrapper_2d import assemble_mesh_cpe4h_cr_numba
                f_sub, K_sub, err = assemble_mesh_cpe4h_cr_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 13:
                from dispsolver.element2d.cr_wrapper_2d import assemble_mesh_cpe4r_cr_numba
                f_sub, K_sub, err = assemble_mesh_cpe4r_cr_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 14:
                from dispsolver.element2d.cr_wrapper_2d import assemble_mesh_cpe6m_cr_numba
                f_sub, K_sub, err = assemble_mesh_cpe6m_cr_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 15:
                from dispsolver.element2d.cpe8_numba import assemble_mesh_cax8_numba
                f_sub, K_sub, err = assemble_mesh_cax8_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 16:
                from dispsolver.element2d.cpe8_numba import assemble_mesh_cax8r_numba
                f_sub, K_sub, err = assemble_mesh_cax8r_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 17:
                from dispsolver.element2d.cpe4_numba import assemble_mesh_cax4_numba
                f_sub, K_sub, err = assemble_mesh_cax4_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            elif k == 18:
                from dispsolver.element2d.t2d2_numba import assemble_mesh_t2d2_numba
                f_sub, K_sub, err = assemble_mesh_t2d2_numba(node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls)
            else:
                raise ValueError(f"Unknown element kernel index: {k}")

            if update_state:
                self.elem_sdvs[elem_indices] = sub_sdvs

            if err:
                has_error = True

            if cr_cs is not None:
                from dispsolver.element2d.cr_generic_2d import corotational_to_global_2d
                f_sub = np.ascontiguousarray(f_sub, dtype=np.float64)
                K_sub = np.ascontiguousarray(K_sub, dtype=np.float64)
                corotational_to_global_2d(f_sub, K_sub, cr_cs)
                node_coords_all, sub_conn, u_eval = _cr_saved

            if self.has_nodal_transforms and self.node_transforms is not None:
                from dispsolver.model.nodal_transform import apply_nodal_transforms_to_element_group
                apply_nodal_transforms_to_element_group(f_sub, K_sub, sub_conn, self.node_transforms, dim=2)

            scatter_f_int_2d(f_sub, sub_conn, f_int_global)
            data_topo_list.append(K_sub.ravel())

        # Incorporate equivalent thermal load vector: f_int = f_int^mech - f_th
        f_th = self.compute_thermal_load_vector()
        if f_th is not None:
            if self.has_nodal_transforms and self.node_transforms is not None:
                from dispsolver.model.nodal_transform import transform_u_to_local
                f_th = transform_u_to_local(f_th, self.node_transforms, dim=2)
            f_int_global -= f_th

        self.last_assembly_error = int(has_error)
        data_topo = np.concatenate(data_topo_list) if len(data_topo_list) > 0 else np.zeros(0, dtype=np.float64)

        if not self.constraints:
            if getattr(self, "topo_to_csr_map", None) is not None:
                scatter_k_csr_direct(data_topo, self.topo_to_csr_map, self.csr_data)
                K_global = csr_matrix(
                    (self.csr_data.copy(), self.csr_indices, self.csr_indptr),
                    shape=(self.num_dofs, self.num_dofs)
                )
            else:
                K_global = csr_matrix(
                    (data_topo, (self.rows_topo, self.cols_topo)),
                    shape=(self.num_dofs, self.num_dofs)
                )
        else:
            rows_extra = [self.rows_topo]
            cols_extra = [self.cols_topo]
            data_extra = [data_topo]

            for c in self.constraints:
                if hasattr(c, "reproject_deformed"):
                    c.reproject_deformed(u_vec)
                if hasattr(c, "assemble"):
                    res = c.assemble(u_vec)
                    if len(res) == 3 and isinstance(res[1], tuple):
                        f_c, (r_c, c_c, d_c), _ = res
                        f_int_global += f_c
                        if len(r_c) > 0:
                            rows_extra.append(r_c)
                            cols_extra.append(c_c)
                            data_extra.append(d_c)
                    elif len(res) == 3:
                        K_c, f_c, _ = res
                        f_int_global += f_c
                        coo_c = K_c.tocoo() if hasattr(K_c, "tocoo") else coo_matrix(K_c)
                        if len(coo_c.row) > 0:
                            rows_extra.append(coo_c.row)
                            cols_extra.append(coo_c.col)
                            data_extra.append(coo_c.data)
                elif hasattr(c, "apply_penalty"):
                    K_lil = csc_matrix((data_topo, (self.rows_topo, self.cols_topo)), shape=(self.num_dofs, self.num_dofs)).tolil()
                    c.apply_penalty(u_vec, f_int_global, K_lil)
                    K_global = K_lil.tocsr()

            if "K_global" not in locals():
                K_global = csr_matrix(
                    (np.concatenate(data_extra), (np.concatenate(rows_extra), np.concatenate(cols_extra))),
                    shape=(self.num_dofs, self.num_dofs)
                )

        self.f_int = f_int_global
        if return_error:
            return K_global, f_int_global, has_error
        return K_global, f_int_global

    def apply_boundary_conditions(self, K_global: Any, residual: np.ndarray, u_k: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """Apply Dirichlet boundary conditions via exact 1.0 diagonal reduction.
        Fast in-place CSR path eliminates COO conversion and array re-allocation.
        """
        r_bc = residual.copy()
        if len(self.fixed_dofs) == 0:
            if not isinstance(K_global, csr_matrix):
                K_global = K_global.tocsr()
            return K_global, r_bc

        penalty = 1.0
        fixed = np.fromiter(self.fixed_dofs.keys(), dtype=np.int64, count=len(self.fixed_dofs))
        is_fixed = np.zeros(K_global.shape[0], dtype=bool)
        is_fixed[fixed] = True
        tgt = np.fromiter(self.fixed_dofs.values(), dtype=np.float64, count=len(self.fixed_dofs))

        if isinstance(K_global, csr_matrix):
            K_bc_data = K_global.data.copy()
            apply_dirichlet_bcs_csr_fast(
                K_bc_data, K_global.indptr, K_global.indices, is_fixed, penalty
            )
            K_bc = csr_matrix((K_bc_data, K_global.indices, K_global.indptr), shape=K_global.shape)
            r_bc[fixed] = penalty * (tgt - u_k[fixed])
        else:
            # Fallback for non-CSR format
            from scipy.sparse import coo_matrix
            coo = K_global.tocoo()
            keep = ~(is_fixed[coo.row] | is_fixed[coo.col])
            rows = np.concatenate([coo.row[keep], fixed])
            cols = np.concatenate([coo.col[keep], fixed])
            data = np.concatenate([coo.data[keep], np.full(fixed.shape, penalty)])
            K_bc = coo_matrix((data, (rows, cols)), shape=K_global.shape).tocsr()
            r_bc[fixed] = penalty * (tgt - u_k[fixed])

        row_nnz = K_bc.indptr[1:] - K_bc.indptr[:-1]
        empty_rows = np.where(row_nnz == 0)[0]
        if len(empty_rows) > 0:
            from scipy.sparse import coo_matrix
            add_fixed = empty_rows
            add_data = np.ones(len(empty_rows), dtype=np.float64)
            coo_bc = K_bc.tocoo()
            rows_aug = np.concatenate([coo_bc.row, add_fixed])
            cols_aug = np.concatenate([coo_bc.col, add_fixed])
            data_aug = np.concatenate([coo_bc.data, add_data])
            K_bc = coo_matrix((data_aug, (rows_aug, cols_aug)), shape=K_global.shape).tocsr()
            for er in empty_rows:
                if er in self.fixed_dofs:
                    r_bc[er] = 1.0 * (self.fixed_dofs[er] - u_k[er])
                else:
                    r_bc[er] = 0.0

        return K_bc, r_bc

    def _record_acceptance(self, branch: str, f_acc: np.ndarray,
                           f_ext: np.ndarray, iters: int) -> None:
        """Read-only calibration probe: true free-DOF residual norm at the
        accepted state. Never influences acceptance (AGENTS.md section 4.7:
        the AND-criterion rewrite caused dt-collapse, so thresholds stay
        untouched -- this only measures what they accept). Ported from
        DynamicSolver3D._record_acceptance (dev_log §12)."""
        mask = np.ones(self.num_dofs, dtype=bool)
        if self.fixed_dofs:
            mask[list(self.fixed_dofs.keys())] = False
        r = float(np.linalg.norm((f_ext - np.asarray(f_acc))[mask]))
        self.last_accept = {"branch": branch, "r_norm": r, "iters": iters}

    def solve_step(
        self,
        dt: float = 1.0,
        f_ext: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        max_iters: int = 25,
        use_anderson_accel: bool = False,
        anderson_m: int = 4,
        anderson_beta: float = 1.0,
        augment_relaxation: float = 1.0,
    ) -> Tuple[bool, int]:
        """Solve a 2D incremental step using Newton-Raphson with robust Armijo line search
        and optional Anderson Acceleration (AA-Newton)."""
        if not self.config_integrator.is_quasistatic and not self.config_integrator.is_emcs:
            u_n = self.u.copy()
            v_n = self.v.copy()
            a_n = self.a.copy()
            u_trial = u_n.copy()

            if f_ext is None:
                if hasattr(self, "f_ext_applied") and self.f_ext_applied is not None:
                    f_ext = self.f_ext_applied.copy()
                else:
                    f_ext = np.zeros(self.num_dofs, dtype=np.float64)

            # Assemble element mass matrices M_global (2D)
            M_elems_list = []
            default_rho = self.density if getattr(self, "density", None) is not None else 7.85e-9
            for g_key in sorted(self.elem_kernel_groups.keys()):
                elem_indices = self.elem_kernel_groups[g_key]
                sub_conn = self.elem_conn_groups[g_key]
                n_sub = len(sub_conn)
                n_nodes_e = sub_conn.shape[1] if n_sub > 0 else 4
                n_dofs_e = n_nodes_e * 2
                M_sub = np.zeros((n_sub, n_dofs_e, n_dofs_e), dtype=np.float64)
                for local_idx, e_idx in enumerate(elem_indices):
                    rho_e = self.elem_density[e_idx] if hasattr(self, "elem_density") else default_rho
                    M_sub[local_idx] = np.eye(n_dofs_e, dtype=np.float64) * (rho_e * (1.0 / n_nodes_e))
                M_elems_list.append(M_sub.ravel())

            M_data = np.concatenate(M_elems_list) if M_elems_list else np.zeros(0, dtype=np.float64)
            M_global = csc_matrix((M_data, (self.rows_topo, self.cols_topo)), shape=(self.num_dofs, self.num_dofs))
            self.M_global = M_global

            K_n, f_int_n = self.assemble_system(u_n, dt=dt, update_state=False)

            beta_hht = self.beta_hht
            gamma_hht = self.gamma_hht
            alpha_hht = self.alpha_hht

            coeff_m = 1.0 / (beta_hht * dt * dt)

            if use_anderson_accel:
                from dispsolver.solver.anderson_acceleration import AndersonAccelerator
                anderson = AndersonAccelerator(m=anderson_m, beta=anderson_beta)
            else:
                anderson = None

            converged = False
            for n_iter in range(1, max_iters + 1):
                a_trial = coeff_m * (u_trial - u_n) - (1.0 / (beta_hht * dt)) * v_n - (0.5 / beta_hht - 1.0) * a_n
                v_trial = v_n + dt * ((1.0 - gamma_hht) * a_n + gamma_hht * a_trial)

                K_tan, f_int_trial = self.assemble_system(u_trial, dt=dt, update_state=False)

                f_int_hht = (1.0 + alpha_hht) * f_int_trial - alpha_hht * f_int_n
                f_inertial = M_global @ a_trial

                residual = f_ext - f_inertial - f_int_hht

                free_mask = np.ones(self.num_dofs, dtype=bool)
                if self.fixed_dofs:
                    free_mask[list(self.fixed_dofs.keys())] = False

                res_norm = float(np.linalg.norm(residual[free_mask]))
                if n_iter == 1:
                    res_init = max(res_norm, 1e-12)

                rel_res = res_norm / res_init
                if res_norm < tol or rel_res < tol:
                    converged = True
                    break

                K_eff = (1.0 + alpha_hht) * K_tan + coeff_m * M_global
                K_bc, r_bc = self.apply_boundary_conditions(K_eff, residual, u_trial)

                try:
                    du = pardiso_spsolve(K_bc, r_bc)
                except Exception as e:
                    raise RuntimeError("Linear solver failed in 2D dynamic step.") from e

                du_norm = float(np.linalg.norm(du[free_mask]))
                if du_norm < 1e-7:
                    u_trial += du
                    converged = True
                    break

                if anderson is not None:
                    u_trial = anderson.step(u_trial, du)
                else:
                    u_trial += du

            if converged:
                self.u[:] = u_trial
                self.a[:] = coeff_m * (self.u - u_n) - (1.0 / (beta_hht * dt)) * v_n - (0.5 / beta_hht - 1.0) * a_n
                self.v[:] = v_n + dt * ((1.0 - gamma_hht) * a_n + gamma_hht * self.a)
                self.assemble_system(self.u, dt=dt, update_state=True)

            return converged, n_iter

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

        accelerator = None
        if use_anderson_accel:
            from dispsolver.solver.anderson_acceleration import AndersonAccelerator
            accelerator = AndersonAccelerator(m=anderson_m, beta=anderson_beta)

        for iter_count in range(1, max_iters + 1):
            K_g, f_int = self.assemble_system(u_k, dt=dt)
            if self.last_assembly_error > 0:
                return False, iter_count

            residual = f_ext - f_int
            r_free = residual[free_dof_mask]
            r_norm = float(np.linalg.norm(r_free))
            f_ext_free_norm = float(np.linalg.norm(f_ext[free_dof_mask]))

            f_int_norm = float(np.linalg.norm(f_int))
            if r_0_norm is None or r_norm > r_0_norm:
                r_0_norm = max(r_norm, f_ext_free_norm, f_int_norm, 1.0)

            rel_r = r_norm / r_0_norm
            rel_f = r_norm / max(f_int_norm, 1.0)
            r_tol = max(tol, 0.05 if len(self.constraints) > 0 else 0.01)
            f_tol = 0.05 if len(self.constraints) > 0 else 0.01

            bc_err = max([abs(target_val - u_k[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            if iter_count > 1 and bc_err < 1e-5 and (rel_r < r_tol or rel_f < f_tol or r_norm < 1e-3):
                self.u = u_k
                _, f_acc = self.assemble_system(u_k, dt=dt, update_state=True)
                self._record_acceptance("residual", f_acc, f_ext, iter_count)
                for c in self.constraints:
                    if hasattr(c, "commit_friction_state"):
                        c.commit_friction_state(u_k)
                return True, iter_count

            K_bc, r_bc = self.apply_boundary_conditions(K_g, residual, u_k)

            try:
                du = pardiso_spsolve(K_bc, r_bc)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError("Linear solver failed. Check boundary conditions or rigid body modes.") from e

            du_norm = float(np.linalg.norm(du))
            u_next = u_k + du
            bc_err_next = max([abs(target_val - u_next[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            if iter_count > 1 and du_norm < 1e-4 and bc_err_next < 1e-5:
                self.u = u_next
                _, f_acc = self.assemble_system(self.u, dt=dt, update_state=True)
                self._record_acceptance("correction", f_acc, f_ext, iter_count)
                for c in self.constraints:
                    if hasattr(c, "commit_friction_state"):
                        c.commit_friction_state(self.u)
                return True, iter_count

            # On iteration 1 (kinematic Dirichlet application), take full step s=1.0 unless element inverts
            if iter_count == 1:
                u_trial = u_k + du
                _, _ = self.assemble_system(u_trial, dt=dt)
                if self.last_assembly_error == 0:
                    if accelerator is not None:
                        accelerator.record(u_k, du)
                    u_k = u_trial
                    for c in self.constraints:
                        if getattr(c, "augmented_lagrange", False) and hasattr(c, "update_augmented_multipliers"):
                            c.update_augmented_multipliers(u_k, omega=augment_relaxation)
                    continue

            # If Anderson Acceleration is active, attempt accelerated trial step first
            aa_accepted = False
            if accelerator is not None and iter_count > 1:
                u_aa = accelerator.step(u_k, du, mask=free_dof_mask)
                for dof, target_val in self.fixed_dofs.items():
                    u_aa[dof] = target_val

                _, f_aa = self.assemble_system(u_aa, dt=dt)
                if self.last_assembly_error == 0:
                    r_aa = f_ext - f_aa
                    r_aa_norm = float(np.linalg.norm(r_aa[free_dof_mask]))
                    if r_aa_norm < r_norm or r_aa_norm <= max(1.2 * r_norm, 1e-4 * r_0_norm, 1e-5):
                        u_k = u_aa
                        aa_accepted = True
                        for c in self.constraints:
                            if getattr(c, "augmented_lagrange", False) and hasattr(c, "update_augmented_multipliers"):
                                c.update_augmented_multipliers(u_k, omega=augment_relaxation)
                        continue

                accelerator.pop_last()

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
            if accelerator is not None and not aa_accepted:
                accelerator.record(u_k - s * du, s * du)

            for c in self.constraints:
                if getattr(c, "augmented_lagrange", False) and hasattr(c, "update_augmented_multipliers"):
                    c.update_augmented_multipliers(u_k, omega=augment_relaxation)

        self.u = u_k
        if bc_err < 1e-5 and (rel_r < r_tol or rel_f < f_tol or r_norm < 1e-3):
            _, f_acc = self.assemble_system(u_k, dt=dt, update_state=True)
            self._record_acceptance("fallthrough", f_acc, f_ext, max_iters)
            for c in self.constraints:
                if hasattr(c, "commit_friction_state"):
                    c.commit_friction_state(u_k)
            return True, max_iters
        return False, max_iters

    def solve_augmented_step(
        self,
        dt: float = 1.0,
        f_ext: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        max_iters: int = 25,
        max_augment_iters: int = 10,
        augment_tol: float = 1e-6,
        augment_relaxation: float = 1.0,
    ) -> Tuple[bool, int, int]:
        """Multi-cycle Augmented Lagrangian driver for constraints with augmented_lagrange=True."""
        augmented_constraints = [c for c in self.constraints if getattr(c, "augmented_lagrange", False)]

        for augment_iter in range(1, max_augment_iters + 1):
            converged, inner_iters = self.solve_step(
                dt=dt, f_ext=f_ext, tol=tol, max_iters=max_iters,
                augment_relaxation=augment_relaxation,
            )
            if not converged:
                return False, augment_iter, inner_iters

            if not augmented_constraints:
                return True, augment_iter, inner_iters

            max_pen = 0.0
            for c in augmented_constraints:
                _, _, stats = c.assemble(self.u)
                max_pen = max(max_pen, stats.get("max_penetration", 0.0))

            if max_pen < augment_tol:
                return True, augment_iter, inner_iters

        return False, max_augment_iters, inner_iters

    def _update_step_bcs(self, step: Any, sys: Optional[Any], t: float, total_time: float) -> None:
        """Evaluate active BCs for 2D Model."""
        self.fixed_dofs.clear()
        if not hasattr(step, "get_active_bcs"):
            return

        active_bcs = step.get_active_bcs()
        node_coords_all = self.mesh.nodes_array()
        nid_map = self.mesh.node_id_to_index()

        for bc_name, bc in active_bcs.items():
            region_nodes = self._resolve_region_nodes(bc.region, sys, step, _describe(bc, f"BC {bc_name}"))

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
                amp_val = None
                if bc.amplitude is not None and hasattr(bc.amplitude, "evaluate"):
                    amp_val = bc.amplitude.evaluate(t, step_time=t, total_time=total_time)
                elif bc.amplitude is not None and isinstance(bc.amplitude, str) and hasattr(step, "parent_model") and hasattr(step.parent_model, "amplitudes"):
                    amp_obj = step.parent_model.amplitudes.get(bc.amplitude, None)
                    if amp_obj is None:
                        raise ValueError(f"{_describe(bc, 'BC ' + bc_name)}: amplitude {bc.amplitude!r} is not defined")
                    amp_val = amp_obj.evaluate(t, step_time=t, total_time=total_time)

                is_ramp = False
                ramp_scale = 1.0
                if amp_val is None:
                    proc = str(getattr(step, "procedure", "STATIC")).upper()
                    if proc in ("STATIC", "QUASISTATIC"):
                        ramp_scale = min(max(t / max(total_time, 1e-12), 0.0), 1.0)
                        is_ramp = True
                    else:
                        amp_val = 1.0

                u_step_start = getattr(self, "_u_step_start", None)
                for gid in region_nodes:
                    n_idx = nid_map.get(gid, None)
                    for d, val in enumerate([getattr(bc, "u1", None), getattr(bc, "u2", None)]):
                        if val is not None:
                            target_val = float(val)
                            if is_ramp:
                                u0 = float(u_step_start[2 * n_idx + d]) if (u_step_start is not None and n_idx is not None) else 0.0
                                u_prescribed = u0 + (target_val - u0) * ramp_scale
                            else:
                                u_prescribed = target_val * amp_val
                            self.fix_dof(gid, d, u_prescribed)

    def _static_ramp(self, vec: np.ndarray, load: Any, step: Any, t: float) -> np.ndarray:
        """Abaqus default amplitude: ramp over a STATIC step, step otherwise."""
        if getattr(load, "amplitude", None) is None:
            proc = str(getattr(step, "procedure", "STATIC")).upper()
            if proc in ("STATIC", "QUASISTATIC"):
                period = float(getattr(step, "time_period", 1.0))
                return vec * min(max(t / max(period, 1e-12), 0.0), 1.0)
        return vec

    def _resolve_region_nodes(self, region: Any, sys: Optional[Any], step: Any, what: str) -> List[int]:
        """Global node ids of a BC/load region; raises if it does not resolve."""
        if sys is not None and hasattr(sys, "global_nsets"):
            from dispsolver.model.regions import resolve_node_region, node_ids_of
            return node_ids_of(sys, resolve_node_region(sys, region, step, what=what))
        if isinstance(region, str) and region.upper() == "ALL":
            return list(self.mesh.nodes.keys())
        for attr in ("node_sets", "sets"):
            for k, v in (getattr(self.mesh, attr, {}) or {}).items():
                if str(k).upper() == str(region).upper():
                    return list(v.node_ids)
        if region in self.mesh.nodes:
            return [region]
        if isinstance(region, str) and region.isdigit() and int(region) in self.mesh.nodes:
            return [int(region)]
        raise ValueError(f"{what}: node region {region!r} does not resolve")

    def _compute_step_external_loads(self, step: Any, sys: Optional[Any], t: float, total_time: float) -> np.ndarray:
        """Compute external load vector f_ext from active loads in step at time t."""
        f_ext = np.zeros(self.num_dofs, dtype=np.float64)
        if not hasattr(step, "get_active_loads"):
            return f_ext

        active_loads = step.get_active_loads()
        nid_map = self.mesh.node_id_to_index()

        for load_name, load in active_loads.items():
            if hasattr(load, "get_acceleration_vector") or type(load).__name__ == "BodyForce" or hasattr(load, "b1"):
                # Used to fall through every branch and be ignored.
                raise NotImplementedError(f"load {load_name}: body/gravity loads are not implemented "
                                          f"in the 2D solver")
            if hasattr(load, "get_force_vector"):
                f_vec = np.asarray(load.get_force_vector(t=t, step_time=t, model=getattr(step, "parent_model", None)),
                                   dtype=np.float64)
                # Abaqus STATIC default amplitude is a ramp (3D already did this;
                # 2D applied the full load from the first increment).
                f_vec = self._static_ramp(f_vec, load, step, t)
                for gid in self._resolve_region_nodes(load.region, sys, step, _describe(load, f"load {load_name}")):
                    n_idx = nid_map[gid]
                    f_ext[2 * n_idx + 0] += f_vec[0]
                    f_ext[2 * n_idx + 1] += f_vec[1]

            elif (hasattr(load, "magnitude") or type(load).__name__ == "Pressure") and hasattr(load, "region"):
                from dispsolver.solver.traction import SurfaceTraction
                # get_magnitude applies the load's amplitude whether it is an
                # object or a *AMPLITUDE name (the name used to be ignored).
                if hasattr(load, "get_magnitude"):
                    p_mag = float(load.get_magnitude(t=t, step_time=t, model=getattr(step, "parent_model", None)))
                else:
                    p_mag = float(getattr(load, "magnitude", 0.0))
                p_mag = float(self._static_ramp(np.array([p_mag]), load, step, t)[0])

                load_name = getattr(load, "name", "")
                face_name = getattr(load, "face", getattr(load, "face_name", "P1"))
                if not face_name or face_name == "P1":
                    for f_tag in ["P1", "P2", "P3", "P4", "S1", "S2", "S3", "S4"]:
                        if f_tag in load_name or f_tag in str(getattr(load, "region", "")):
                            face_name = f_tag
                            break

                st = SurfaceTraction(elset_name=load.region, face=face_name, pressure=p_mag)
                f_press, _ = st.assemble_forces_and_stiffness(self.mesh, self.u, p_mag)
                f_ext += f_press

        return f_ext

    def solve(
        self,
        step: Any,
        sys: Optional[Any] = None,
        output_mgr: Optional[Any] = None,
        verbose: bool = True
    ) -> bool:
        """Run complete incremental solve over the Step time period."""
        # Effective NLGEOM for this step (same rule as DynamicSolver3D.solve:
        # once on, it stays on for the rest of the analysis).
        want = bool(getattr(step, "nlgeom", self.nlgeom)) or bool(getattr(self, "_nlgeom_engaged", False))
        if want != self.nlgeom:
            self.nlgeom = want
            self._build_kernel_groups()
        self._nlgeom_engaged = bool(getattr(self, "_nlgeom_engaged", False) or self.nlgeom)

        total_time = getattr(step, "time_period", 1.0)
        dt = getattr(step, "dt_init", 0.02)
        dt_min = getattr(step, "dt_min", 1e-6)
        dt_max = getattr(step, "dt_max", 0.05)

        t_curr = 0.0
        inc_count = 0
        total_iters = 0
        # per-solve() counters, read by execute_inp
        self.last_solve_stats = {"iterations": 0, "increments": 0, "max_increment_iterations": 0}

        self.history_u = [self.u.copy()]
        self.history_t = [0.0]
        self._u_step_start = self.u.copy()

        if verbose:
            print(f"=== Starting 2D Step '{step.name}' Solve (Time Period: {total_time:.3f} s) ===", flush=True)

        # Setup active constraints and interactions for this step
        if hasattr(step, "interactions"):
            for iname, entry in step.interactions.items():
                if hasattr(entry, "is_active") and entry.is_active():
                    inter = entry.entity
                    if hasattr(inter, "build_runtime_constraint"):
                        existing_names = [getattr(c, "name", "") for c in self.constraints]
                        if inter.name not in existing_names:
                            nid_to_idx = self.mesh.node_id_to_index()
                            coords = self.mesh.nodes_array() if hasattr(self.mesh, "nodes_array") else (sys.coords if sys is not None else None)
                            rc = inter.build_runtime_constraint(
                                mesh=self.mesh,
                                nid_to_idx=nid_to_idx,
                                coords=coords,
                                model=getattr(step, "parent_model", None),
                                sys=sys
                            )
                            self.add_constraint(rc)

        while t_curr < total_time - 1e-9:
            inc_count += 1
            if t_curr + dt > total_time:
                dt = total_time - t_curr

            t_next = t_curr + dt
            self._update_step_bcs(step, sys, t_next, total_time)
            f_ext = self._compute_step_external_loads(step, sys, t_next, total_time)

            u_prev = self.u.copy()
            sdvs_prev = self.elem_sdvs.copy()

            converged, iters = self.solve_step(dt=dt, f_ext=f_ext, max_iters=50)
            total_iters += iters
            if converged:
                self.last_solve_stats["increments"] += 1
                self.last_solve_stats["max_increment_iterations"] = max(self.last_solve_stats["max_increment_iterations"], int(iters))
            self.last_solve_stats["iterations"] = total_iters

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

    def compute_total_energy(self) -> Tuple[float, float, float]:
        """Compute Total Mechanical Energy (E_total, E_kin, E_strain) for 2D mesh."""
        E_strain = self.compute_strain_energy()
        E_kin = 0.0
        if hasattr(self, "v") and self.v is not None:
            E_kin = 0.5 * float(np.dot(self.v, self.v) * self.density)
        return E_strain + E_kin, E_kin, E_strain

    def get_state(self) -> Dict[str, Any]:
        """Return full state vector dictionary (u, v, a, time_integrator)."""
        return {
            "u": self.u.copy(),
            "v": self.v.copy(),
            "a": self.a.copy(),
            "time_integrator": self.time_integrator,
        }


