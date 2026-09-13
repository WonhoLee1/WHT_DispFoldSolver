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

try:
    from dispsolver.solver.pardiso_manager import PardisoNonlinearSolver, PARDISO_AVAILABLE as _PARDISO_MGR_AVAILABLE
except Exception:
    PardisoNonlinearSolver = None
    _PARDISO_MGR_AVAILABLE = False

from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d.assembly_utils import build_global_topology, scatter_f_int_3d
from dispsolver.material3d.numba_materials import (
    MAT_LINEAR_ELASTIC,
    MAT_NEO_HOOKEAN,
    MAT_J2_PLASTICITY,
    MAT_VISCOELASTIC_PRONY,
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

        # Phase-reuse Pardiso wrapper (mirrors dispsolver/solver/dynamic.py's
        # 2D PardisoNonlinearSolver usage, AGENTS.md sec4.11): the raw
        # pypardiso.spsolve() path below re-runs a FULL analysis+numerical
        # factorization (Pardiso phase 12) on every single Newton
        # iteration, since it only skips re-factorization when the matrix
        # is byte-identical -- never true once residual-driven values
        # change. For a contact problem specifically, the active-set can
        # also change the sparsity PATTERN between iterations, and MKL's
        # internal analysis structures for the old pattern are only
        # released by PardisoNonlinearSolver's own free_memory() call
        # (pardiso_manager.py's _solve_internal), not by Python's GC --
        # native MKL memory is invisible to gc.collect(). Long multi-
        # thousand-iteration 3D runs (fold examples, contact benchmarks)
        # using the naive path were the suspected root cause of an
        # observed multi-GB, still-growing python process after ~1 day
        # of wall time (2026-09-13). This wrapper is optional -- solves
        # fall back to plain pardiso_spsolve/scipy spsolve if pypardiso
        # or pardiso_manager aren't importable.
        self.pardiso_solver = (
            PardisoNonlinearSolver(mtype="auto", phase_reuse=True)
            if (_PARDISO_MGR_AVAILABLE and PardisoNonlinearSolver is not None)
            else None
        )

        self.f_int = None
        self._setup_numba_topology()


    def _solve_linear(self, K_bc, r_bc):
        """Solve K_bc @ du = r_bc, preferring the phase-reuse Pardiso
        wrapper (see __init__) over the raw pypardiso.spsolve()/scipy
        fallback so repeated Newton-iteration solves don't force a full
        MKL re-analysis (and its associated native-memory churn) every
        call. has_lagrange_multipliers=False: this solver's constraints
        (RBE3, surface tie, contact) are all penalty/condensation-based,
        scattered directly into K_bc's own DOF rows/cols -- no separate
        KKT dual rows exist in the 3D assembly path.
        """
        if self.pardiso_solver is not None:
            return self.pardiso_solver.solve(K_bc, r_bc, has_lagrange_multipliers=False)
        return pardiso_spsolve(K_bc, r_bc)

    def close(self):
        """Release the Pardiso solver's native MKL memory (factorization
        buffers + internal analysis structures). Not required for
        short scripts (process exit reclaims everything the OS tracks
        regardless), but call this explicitly at the end of any
        long-running, many-thousand-iteration solve loop run in a
        process that will keep doing other work afterward (e.g. a
        benchmark sweep instantiating many DynamicSolver3D objects in
        one interpreter session) -- see the __init__ docstring note on
        self.pardiso_solver for why this doesn't happen automatically."""
        if self.pardiso_solver is not None:
            self.pardiso_solver.clear()

    def _setup_numba_topology(self):
        """Precompute global assembly topology and DOD multimaterial arrays."""
        if not self.mesh.elements:
            self.elem_conn_0based = None
            self.rows_topo = None
            self.cols_topo = None
            self.elem_mat_types = None
            self.elem_props = None
            self.elem_sdvs = None
            self.elem_stress_init = None
            self.elem_kernel_groups = {}
            return

        nid_map = self.mesh.node_id_to_index()
        elements_list = list(self.mesh.elements.values())
        try:
            self.elem_conn_0based = np.array(
                [[nid_map[nid] for nid in elem.node_ids] for elem in elements_list],
                dtype=np.int64
            )
        except Exception:
            self.elem_conn_0based = None

        n_elems = len(elements_list)
        self.elem_mat_types = np.full(n_elems, MAT_CUSTOM_ELASTIC, dtype=np.int32)
        self.elem_props = np.zeros((n_elems, 36), dtype=np.float64)

        # Populate multimaterial properties per element
        cmat_flat = self.C_mat_default.ravel()
        for e, elem in enumerate(elements_list):
            pid = getattr(elem, "pid", 0)
            mat_obj = self.materials.get(pid, None) if hasattr(self, "materials") and isinstance(self.materials, dict) else None
            if mat_obj is not None:
                if hasattr(mat_obj, "mat_type") and hasattr(mat_obj, "props"):
                    self.elem_mat_types[e] = int(mat_obj.mat_type)
                    p = np.asarray(mat_obj.props, dtype=np.float64)
                    self.elem_props[e, :min(36, len(p))] = p[:36]
                elif hasattr(mat_obj, "C_mat"):
                    self.elem_mat_types[e] = MAT_CUSTOM_ELASTIC
                    self.elem_props[e, :36] = np.asarray(mat_obj.C_mat, dtype=np.float64).ravel()[:36]
                elif isinstance(mat_obj, dict):
                    mtype = mat_obj.get("mat_type", mat_obj.get("type", ""))
                    if mtype == MAT_J2_PLASTICITY or str(mtype).lower() in ["j2", "plasticity", "j2_plasticity"] or "sigma_y0" in mat_obj or "yield_stress" in mat_obj:
                        self.elem_mat_types[e] = MAT_J2_PLASTICITY
                        E_val = float(mat_obj.get("E", 200000.0))
                        nu_val = float(mat_obj.get("nu", 0.3))
                        sy_val = float(mat_obj.get("yield_stress", mat_obj.get("sigma_y0", 400.0)))
                        H_val = float(mat_obj.get("hardening_modulus", mat_obj.get("H", 0.0)))
                        self.elem_props[e, :4] = [E_val, nu_val, sy_val, H_val]
                    elif mtype == MAT_NEO_HOOKEAN or str(mtype).lower() in ["neohookean", "neo_hookean"] or "c10" in mat_obj or "C10" in mat_obj:
                        self.elem_mat_types[e] = MAT_NEO_HOOKEAN
                        c10 = float(mat_obj.get("c10", mat_obj.get("C10", 0.1)))
                        d1 = float(mat_obj.get("d1", mat_obj.get("D1", 0.01)))
                        mu = 2.0 * c10
                        K = 2.0 / max(d1, 1e-12)
                        self.elem_props[e, 0] = mu
                        self.elem_props[e, 1] = K
                        self.elem_props[e, 2] = c10
                        self.elem_props[e, 3] = d1
                    elif mtype == MAT_VISCOELASTIC_PRONY or str(mtype).lower() in ["viscoelastic", "prony", "visco"]:
                        self.elem_mat_types[e] = MAT_VISCOELASTIC_PRONY
                        self.elem_props[e, 0] = float(mat_obj.get("E", 1000.0))
                        self.elem_props[e, 1] = float(mat_obj.get("nu", 0.45))
                        self.elem_props[e, 2] = float(mat_obj.get("g1", 0.2))
                        self.elem_props[e, 3] = float(mat_obj.get("tau1", 1.0))
                    elif isinstance(mtype, int):
                        self.elem_mat_types[e] = mtype
                        if "props" in mat_obj:
                            p = np.asarray(mat_obj["props"], dtype=np.float64)
                            self.elem_props[e, :min(36, len(p))] = p[:36]
                        else:
                            self.elem_props[e, 0] = float(mat_obj.get("E", 200000.0))
                            self.elem_props[e, 1] = float(mat_obj.get("nu", 0.3))
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

        max_sdvs = 1
        for mt in self.elem_mat_types:
            max_sdvs = max(max_sdvs, get_default_sdv_count(mt))
        self.elem_sdvs = np.zeros((n_elems, 8, max_sdvs), dtype=np.float64)
        self.elem_stress_init = np.zeros((n_elems, 8, 6), dtype=np.float64)

        # Allocate Abaqus-compatible SectionControls array (n_elems, 8)
        self.elem_controls = np.zeros((n_elems, 8), dtype=np.float64)
        self.elem_controls[:, 0] = 1.0   # distortion_control default ON
        self.elem_controls[:, 1] = 0.1   # length_ratio default 0.1
        self.elem_controls[:, 2] = 0.0   # viscous_damping default 0.0
        self.elem_controls[:, 3] = 1.0   # anti_inversion_barrier default ON
        self.elem_controls[:, 4] = 0.02  # min_det_f default 0.02
        
        # Classify element kernel groups with Material- & Kinematic-Adaptive NLGEOM Dispatch
        elem_kernel_indices: Dict[int, List[int]] = {}
        for idx, elem in enumerate(elements_list):
            et = elem.elem_type.upper()
            pid = getattr(elem, "pid", 0)
            mat_obj = self.materials.get(pid, None) if hasattr(self, "materials") and isinstance(self.materials, dict) else None
            nu = 0.3
            if mat_obj is not None:
                if isinstance(mat_obj, dict):
                    nu = float(mat_obj.get("nu", 0.3))
                elif hasattr(mat_obj, "props"):
                    nu = float(mat_obj.props[1]) if len(mat_obj.props) > 1 else 0.3
            is_nearly_incompressible = (nu > 0.48)

            if self.nlgeom:
                # NLGEOM=True: Automatically promote core elements to Large-Deformation / Corotational
                if et in ["C3D8I", "C3D8_EAS", "C3D8I_CR"]:
                    k = 11  # C3D8I_CR (EAS-9 + Corotational)
                elif et in ["C3D8H", "C3D8_HYBRID", "HYBRID", "C3D8H_CR"]:
                    k = 2   # C3D8H (u-P Mixed Hybrid + Corotational)
                elif et in ["C3D8R", "C3D8_REDUCED", "C3D8R_CR"]:
                    k = 13  # C3D8R_CR (Reduced Integration + Corotational)
                elif et in ["C3D8_CR", "C3D8_COROTATIONAL", "C3D8_FBAR_CR"]:
                    k = 1   # C3D8_CR
                elif et in ["C3D8_FBAR"]:
                    k = 3   # C3D8_FBAR
                elif et in ["C3D8"]:
                    k = 1   # C3D8_CR (Hughes B-bar + Corotational)
                elif et in ["C3D10M", "C3D10_MODIFIED"]:
                    k = 4
                elif et in ["C3D10"]:
                    k = 5
                elif et in ["C3D4"]:
                    k = 6
                elif et in ["C3D6", "C3D6_WEDGE", "WEDGE6"]:
                    k = 7
                elif et in ["C3D4_ANP", "ANP"]:
                    k = 9
                else:
                    k = 1 if not is_nearly_incompressible else 3
            else:
                # Small-strain linear geometry
                if et in ["C3D8I", "C3D8_EAS"]:
                    k = 0
                elif et in ["C3D8_CR", "C3D8_COROTATIONAL", "C3D8_FBAR_CR"]:
                    k = 1
                elif et in ["C3D8H", "C3D8_HYBRID", "HYBRID"]:
                    k = 2
                elif et in ["C3D8_FBAR"]:
                    k = 3
                elif et in ["C3D8"]:
                    k = 10
                elif et in ["C3D10M", "C3D10_MODIFIED"]:
                    k = 4
                elif et in ["C3D10"]:
                    k = 5
                elif et in ["C3D4"]:
                    k = 6
                elif et in ["C3D6", "C3D6_WEDGE", "WEDGE6"]:
                    k = 7
                elif et in ["C3D8R", "C3D8_REDUCED"]:
                    k = 8
                elif et in ["C3D4_ANP", "ANP"]:
                    k = 9
                elif et in ["C3D8I_CR"]:
                    k = 11
                elif et in ["C3D8H_CR"]:
                    k = 2
                elif et in ["C3D8R_CR"]:
                    k = 13
                else:
                    raise NotImplementedError(
                        f"Element type '{et}' is not yet supported in the Numba fast-path."
                    )
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

        rows_list = []
        cols_list = []
        for k in sorted(self.elem_kernel_groups.keys()):
            conn_k = self.elem_conn_groups[k]
            r_k, c_k = build_global_topology(self.num_dofs, conn_k)
            rows_list.append(r_k)
            cols_list.append(c_k)

        if len(rows_list) > 0:
            self.rows_topo = np.concatenate(rows_list)
            self.cols_topo = np.concatenate(cols_list)
        else:
            self.rows_topo = np.zeros(0, dtype=np.int32)
            self.cols_topo = np.zeros(0, dtype=np.int32)

        # Check if mesh or sections provide custom controls
        if hasattr(self.mesh, "elem_controls") and self.mesh.elem_controls is not None:
            self.elem_controls[:] = self.mesh.elem_controls

    def set_element_controls(
        self,
        elem_indices: Any,
        controls: Any
    ):
        """Set Abaqus-compatible SectionControls for specific 3D solid elements.

        Parameters:
            elem_indices: int, slice, list of ints, or numpy array of element indices (0-based).
            controls: SectionControls instance or (8,) numpy float array.
        """
        if hasattr(controls, "to_control_array"):
            ctrl_arr = controls.to_control_array()
        elif isinstance(controls, np.ndarray):
            ctrl_arr = controls
        else:
            raise ValueError(f"Unsupported controls type: {type(controls)}")

        self.elem_controls[elem_indices] = ctrl_arr

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

    def add_nodal_force(self, node_id: int, dof_axis: int, force_value: float):
        """Add an external nodal concentrated force (dof_axis: 0 for X, 1 for Y, 2 for Z)."""
        nid_map = self.mesh.node_id_to_index()
        global_dof = 3 * nid_map[node_id] + dof_axis
        if not hasattr(self, "f_ext_applied") or self.f_ext_applied is None:
            self.f_ext_applied = np.zeros(self.num_dofs, dtype=np.float64)
        self.f_ext_applied[global_dof] += float(force_value)

    def assemble_system(self, u_vec: np.ndarray, dt: float = 1.0, return_error: bool = False, update_state: bool = False):
        """Assemble global 3D stiffness matrix K_global and internal force vector f_int."""
        node_coords_all = self.mesh.nodes_array()

        if self.nlgeom and self.rows_topo is not None:
            try:
                f_int_global = np.zeros(self.num_dofs, dtype=np.float64)
                data_topo_list = []
                has_error = False

                # Heterogeneous element assembly: assemble each group in parallel
                for k in sorted(self.elem_kernel_groups.keys()):
                    elem_indices = self.elem_kernel_groups[k]
                    sub_conn = self.elem_conn_groups[k]
                    sub_mat = self.elem_mat_types[elem_indices]
                    sub_props = self.elem_props[elem_indices]
                    
                    if update_state:
                        sub_sdvs = self.elem_sdvs[elem_indices]
                    else:
                        sub_sdvs = self.elem_sdvs[elem_indices].copy()
                        
                    sub_stress_init = self.elem_stress_init[elem_indices]
                    sub_controls = self.elem_controls[elem_indices]

                    if k == 0:
                        from dispsolver.element3d.c3d8_eas_tl_numba import assemble_mesh_c3d8_eas_tl_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_eas_tl_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_stress_init=sub_stress_init
                        )
                    elif k == 1:
                        from dispsolver.element3d.c3d8_corotational_numba import assemble_mesh_c3d8_corotational_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_corotational_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 2:
                        from dispsolver.element3d.c3d8_hybrid_numba import assemble_mesh_c3d8_hybrid_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_hybrid_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 3:
                        from dispsolver.element3d.c3d8_fbar_tl_numba import assemble_mesh_c3d8_fbar_tl_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_fbar_tl_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_stress_init=sub_stress_init
                        )
                    elif k == 4:
                        from dispsolver.element3d.c3d10m_numba import assemble_mesh_c3d10m_numba
                        f_sub, K_sub, err = assemble_mesh_c3d10m_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 5:
                        from dispsolver.element3d.c3d10_numba import assemble_mesh_c3d10_numba
                        f_sub, K_sub = assemble_mesh_c3d10_numba(
                            node_coords_all, sub_conn, u_vec, self.C_mat_default
                        )
                        err = 0
                    elif k == 6:
                        from dispsolver.element3d.c3d4_numba import assemble_mesh_c3d4_numba
                        f_sub, K_sub = assemble_mesh_c3d4_numba(
                            node_coords_all, sub_conn, u_vec, self.C_mat_default
                        )
                        err = 0
                    elif k == 7:
                        from dispsolver.element3d.c3d6_numba import assemble_mesh_c3d6_numba
                        f_sub, K_sub, err = assemble_mesh_c3d6_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 8:
                        from dispsolver.element3d.c3d8r_numba import assemble_mesh_c3d8r_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8r_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 9:
                        from dispsolver.element3d.c3d4_anp_numba import assemble_mesh_c3d4_anp_numba
                        f_sub, K_sub, err = assemble_mesh_c3d4_anp_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 10:
                        from dispsolver.element3d.c3d8_numba import assemble_mesh_c3d8_numba
                        f_sub, K_sub = assemble_mesh_c3d8_numba(
                            node_coords_all, sub_conn, u_vec, self.C_mat_default
                        )
                        err = 0
                    elif k == 11:
                        from dispsolver.element3d.cr_wrapper_3d import assemble_mesh_c3d8i_cr_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8i_cr_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 13:
                        from dispsolver.element3d.cr_wrapper_3d import assemble_mesh_c3d8r_cr_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8r_cr_numba(
                            node_coords_all, sub_conn, u_vec, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    else:
                        raise ValueError(f"Unknown element kernel index: {k}")

                    if update_state:
                        # NumPy advanced indexing creates a copy. We must explicitly write the modified sub_sdvs back.
                        self.elem_sdvs[elem_indices] = sub_sdvs

                    if err:
                        has_error = True

                    scatter_f_int_3d(f_sub, sub_conn, f_int_global)
                    data_topo_list.append(K_sub.ravel())

                self.last_assembly_error = has_error
                data_topo = np.concatenate(data_topo_list) if len(data_topo_list) > 0 else np.zeros(0, dtype=np.float64)

                self.f_int = f_int_global
                if len(self.constraints) == 0:
                    K_global = csc_matrix(
                        (data_topo, (self.rows_topo, self.cols_topo)),
                        shape=(self.num_dofs, self.num_dofs)
                    )
                    if return_error:
                        return K_global, f_int_global, has_error
                    return K_global, f_int_global

                # If surface tie or MPC constraints exist, append their contributions efficiently
                rows_all = [self.rows_topo]
                cols_all = [self.cols_topo]
                data_all = [data_topo]

                for constraint in self.constraints:
                    if hasattr(constraint, "reproject_deformed"):
                        constraint.reproject_deformed(u_vec)
                    f_c, (rows_c, cols_c, data_c), _ = constraint.assemble(u_vec)
                    f_int_global += f_c
                    rows_all.append(np.asarray(rows_c, dtype=np.int32))
                    cols_all.append(np.asarray(cols_c, dtype=np.int32))
                    data_all.append(np.asarray(data_c, dtype=np.float64))

                self.f_int = f_int_global
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
                print("CRITICAL: Numba assembly fastpath failed! Aborting to prevent silent fallback to incorrect physics.")
                raise e
        else:
            raise NotImplementedError("Linear geometry or uninitialized topology is not supported. The Python fallback loop has been removed.")

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

    def _contact_active_set(self, u: np.ndarray) -> frozenset:
        """Union of get_active_set(u) across every constraint that exposes
        one (currently only SurfaceContactConstraint3D). Empty (and always
        equal to itself) when there are no contact-type constraints, so
        the severe-discontinuity-iteration (SDI) handling in solve_step()
        below is a complete no-op for every non-contact 3D solve -- see
        that method's docstring."""
        if not self.constraints:
            return frozenset()
        active = None
        for c in self.constraints:
            if hasattr(c, "get_active_set"):
                s = c.get_active_set(u)
                active = s if active is None else (active | s)
        return active if active is not None else frozenset()

    def solve_step(
        self,
        dt: float = 1.0,
        f_ext: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        max_iters: int = 25,
        max_sdi_iters: int = 200,
    ) -> Tuple[bool, int]:
        """Solve a 3D non-linear incremental step using Newton-Raphson with
        Armijo line search.

        Severe discontinuity iterations (SDI): when a contact constraint's
        active set (which nodes are currently penetrating) changes between
        consecutive iterates, the residual is genuinely discontinuous in
        that variable, not just large -- the same physical mechanism
        exists in Abaqus/Standard
        (https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/gss/ch11s03.html),
        which exempts such an iteration from the normal equilibrium
        convergence check rather than requiring it to show residual
        decrease, and only starts counting toward normal convergence once
        an iteration completes with an UNCHANGED active set.
        dev_log/hertz_contact_benchmark_20260913.md found this codebase's
        plain backtracking line search (below) has no such exemption: it
        always accepts SOME step each iteration (only genuine element
        inversion on every one of its 6 halvings triggers a real
        rejection -- residual growth alone does not), so an activation
        event's residual spike just keeps getting "accepted" and chases a
        moving target until max_iters runs out and the step is reported
        DIVERGED, independent of load-step size.

        max_iters now counts only EQUILIBRIUM iterations (active set
        unchanged from the previous iterate); max_sdi_iters is a separate
        budget for iterations where the active set is still changing, so
        an active-set transient cannot silently eat into (or exhaust) the
        normal convergence budget, and normal convergence is never
        evaluated against a residual baseline still in the middle of an
        activation event. With no contact-type constraints (the common
        case), _contact_active_set() always returns frozenset() for every
        iterate, so every iteration is trivially "equilibrium" and this
        reduces exactly to the original loop.
        """
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

        prev_active_set = self._contact_active_set(u_k)
        equil_iters = 0
        raw_iter = 0
        max_raw_iters = max_iters + max_sdi_iters

        while equil_iters < max_iters and raw_iter < max_raw_iters:
            raw_iter += 1
            iter_count = raw_iter
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

            # Severe-discontinuity-iteration (SDI) check -- see solve_step's
            # own docstring. Bookkeeping happens unconditionally here (not
            # at the bottom of the loop) so it isn't skipped by the
            # iteration-1 `continue` a few lines below.
            current_active_set = self._contact_active_set(u_k)
            is_sdi = current_active_set != prev_active_set
            if not is_sdi:
                equil_iters += 1
            if getattr(self, "debug_sdi", False):
                print(f"[SDI] raw_iter={raw_iter} equil_iters={equil_iters} is_sdi={is_sdi} "
                      f"n_active={len(current_active_set)} r_norm={r_norm:.4g} rel_r={r_norm/r_0_norm:.4g}", flush=True)
            prev_active_set = current_active_set

            rel_r = r_norm / r_0_norm
            # Commercial CAE convergence criteria (Abaqus standard 0.5% residual tolerance)
            bc_err = max([abs(target_val - u_k[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            if iter_count > 1 and not is_sdi and bc_err < 1e-5 and (rel_r < 5e-3 or r_norm < 1e-3):
                self.u = u_k
                # Commit state variables (SDV) for the converged increment
                self.assemble_system(u_k, dt=dt, update_state=True)
                return True, iter_count

            # Solve linear system \Delta u = K^-1 * r
            try:
                du = self._solve_linear(K_bc, r_bc)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError("Pardiso linear solver failed (matrix is likely singular). Check boundary conditions or rigid body modes.") from e

            du_norm = float(np.linalg.norm(du))
            u_next = u_k + du
            bc_err_next = max([abs(target_val - u_next[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            if iter_count > 1 and not is_sdi and du_norm < 1e-4 and bc_err_next < 1e-5:
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

            # Robust Armijo line search for energy/residual minimization.
            #
            # FIXED 2026-09-13 (dev_log/hertz_contact_benchmark_20260913.md,
            # found while verifying the SDI fix above): this loop used to
            # set `found_valid = True` unconditionally on its very FIRST
            # trial (`r_t_norm < best_r_norm or not found_valid`, and
            # `not found_valid` is always True on trial 0) -- so best_u
            # was seeded to the FULL step (s=1.0) regardless of whether it
            # actually improved the residual, and the `if not found_valid:
            # return False` cutback a few lines below could then only ever
            # fire if literally every one of the 6 halvings caused element
            # inversion. A full step that made the residual WORSE, with
            # none of the 5 remaining (smaller) halvings improving on it
            # either, was silently accepted anyway. This is not contact-
            # specific -- it is a base Newton-loop defect that happened to
            # rarely trigger because most of this solver's problems have a
            # full Newton step that naturally decreases residual; a
            # multi-node-simultaneous-contact system exposed it directly
            # (measured: residual crept from 1.112 to 1.143 over 50
            # "accepted" full-step iterations, never once rejected).
            # Fixed: found_valid is now only set True on a GENUINE
            # improvement over the running best (seeded to this
            # iteration's own r_norm, not an arbitrary first guess) or on
            # meeting the acceptance threshold -- so if none of the 6
            # halvings actually helps, found_valid correctly stays False
            # and the increment is genuinely cut back, exactly as a real
            # Armijo line search requires.
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
                if r_t_norm < best_r_norm:
                    best_r_norm = r_t_norm
                    best_u = u_trial
                    found_valid = True
                if r_t_norm <= max(r_norm, 1e-4 * r_0_norm, 1e-6):
                    best_u = u_trial
                    found_valid = True
                    break
                s *= 0.5

            if not found_valid:
                # All trial steps inverted or failed: cutback increment
                return False, iter_count

            u_k = best_u

        # Exhausted the equilibrium-iteration budget (max_iters, counting
        # only non-SDI iterations) or the total raw budget
        # (max_iters+max_sdi_iters) without hitting either convergence
        # branch above -- do NOT report success unconditionally (this used
        # to `return True, max_iters` regardless of residual size, silently
        # masking genuine non-convergence as a converged, physically wrong
        # answer -- see dev_log/solve_step_false_convergence_20260913.md).
        # Re-check the SAME criteria the loop itself uses, against the
        # last computed rel_r/r_norm/bc_err/is_sdi (last iteration's locals
        # are still in scope here), and only accept if they're actually
        # met AND the active set was stable on that last iteration.
        self.u = u_k
        if not is_sdi and bc_err < 1e-5 and (rel_r < 5e-3 or r_norm < 1e-3):
            self.assemble_system(u_k, dt=dt, update_state=True)
            return True, raw_iter
        return False, raw_iter

    def solve_step_augmented(
        self,
        dt: float = 1.0,
        f_ext: Optional[np.ndarray] = None,
        max_iters: int = 25,
        max_sdi_iters: int = 200,
        max_augment_iters: int = 15,
        augment_tol: float = 1e-4,
        augment_relaxation: float = 1.0,
    ) -> Tuple[bool, int, int]:
        """Augmented-Lagrangian outer loop around solve_step() (design doc
        sec4, Phase 2: "reuses the same penalty assembly, wrapped in an
        outer augmentation loop ... converge with penalty, check
        penetration against a tolerance, 'augment' the effective contact
        pressure and re-solve, repeat" -- ctc_contactconstraints_std.txt).

        **FIXED, 2026-09-13** -- see SurfaceContactConstraint3D.assemble()'s
        f_node sign-fix comment for the root cause (contact force was
        added into f_int_global with the opposite sign of the internal-
        force convention every element and the surface tie already use)
        and SurfaceContactConstraint3D.__init__'s updated docstring for the
        before/after measurements. Was: geometric divergence (~1.42x per
        cycle) on both a single-element sanity case and the graded Hertz
        benchmark, at every under-relaxation tested (0.1-1.0). Now:
        max_penetration contracts geometrically (~0.77x per cycle on the
        single-element case) toward zero, matching standard Uzawa theory.

        Only meaningful when at least one of self.constraints has
        augmented_lagrange=True (SurfaceContactConstraint3D); with none,
        this degenerates to exactly one call to solve_step() (the inner
        loop) since update_augmented_multipliers() is a no-op for a
        plain-penalty constraint and max_penetration reports 0
        immediately, satisfying augment_tol on the first outer cycle.

        Returns (converged, n_augment_cycles, last_inner_iters). The
        u/state are committed by solve_step() itself as usual; this
        method's only additional side effect is updating each augmented
        constraint's persistent self._lam across outer cycles (undone
        implicitly on failure only in the sense that a failed inner solve
        leaves lam at its PREVIOUS cycle's value -- lam is deliberately
        NOT rolled back on inner failure, matching the design doc's
        description of augmentation as operating on an already-converged
        equilibrium, never on a failed one).
        """
        augmented_constraints = [c for c in self.constraints if getattr(c, "augmented_lagrange", False)]

        for augment_iter in range(1, max_augment_iters + 1):
            converged, inner_iters = self.solve_step(dt=dt, f_ext=f_ext, max_iters=max_iters, max_sdi_iters=max_sdi_iters)
            if not converged:
                return False, augment_iter, inner_iters

            if not augmented_constraints:
                return True, augment_iter, inner_iters

            max_pen = 0.0
            for c in augmented_constraints:
                stats = c.update_augmented_multipliers(self.u, omega=augment_relaxation)
                max_pen = max(max_pen, stats["max_penetration"])

            if max_pen < augment_tol:
                return True, augment_iter, inner_iters

        return False, max_augment_iters, inner_iters

    def solve_static_step(self, f_ext: np.ndarray, tol: float = 1e-6, max_iters: int = 20) -> bool:
        """Alias for static Newton step returning boolean converged status."""
        converged, _ = self.solve_step(dt=1.0, f_ext=f_ext, tol=tol, max_iters=max_iters)
        return converged

    def _update_step_bcs(self, step: Any, sys: Optional[Any], t: float, total_time: float) -> None:
        """Evaluate and apply all active BCs and Amplitudes for the given step at time t."""
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
                        u3 = res[2] if len(res) > 2 else None
                        if u1 is not None:
                            self.fix_dof(gid, 0, float(u1))
                        if u2 is not None:
                            self.fix_dof(gid, 1, float(u2))
                        if u3 is not None:
                            self.fix_dof(gid, 2, float(u3))

            elif hasattr(bc, "u1") or hasattr(bc, "u2") or hasattr(bc, "u3"):
                amp_val = 1.0
                if bc.amplitude is not None:
                    amp_obj = None
                    if isinstance(bc.amplitude, str) and hasattr(step, "parent_model") and hasattr(step.parent_model, "amplitudes"):
                        amp_obj = step.parent_model.amplitudes.get(bc.amplitude, None)
                    elif hasattr(bc.amplitude, "evaluate"):
                        amp_obj = bc.amplitude

                    if amp_obj is not None:
                        amp_val = amp_obj.evaluate(t, step_time=t, total_time=total_time)

                for gid in region_nodes:
                    if bc.u1 is not None:
                        self.fix_dof(gid, 0, float(bc.u1) * amp_val)
                    if bc.u2 is not None:
                        self.fix_dof(gid, 1, float(bc.u2) * amp_val)
                    if bc.u3 is not None:
                        self.fix_dof(gid, 2, float(bc.u3) * amp_val)

    def _compute_step_external_loads(self, step: Any, sys: Optional[Any], t: float, total_time: float) -> np.ndarray:
        """Compute external load vector f_ext from active loads in step at time t."""
        f_ext = np.zeros(self.num_dofs, dtype=np.float64)
        if not hasattr(step, "get_active_loads"):
            return f_ext

        active_loads = step.get_active_loads()
        nid_map = self.mesh.node_id_to_index()

        for load_name, load in active_loads.items():
            if hasattr(load, "get_force_vector"):
                region_nodes = []
                if sys is not None and hasattr(sys, "global_nsets") and load.region in sys.global_nsets:
                    node_indices = sys.global_nsets[load.region]
                    inv_nid = {idx: g for g, idx in sys.nid_to_idx.items()}
                    region_nodes = [inv_nid[idx] for idx in node_indices]
                elif load.region == "ALL":
                    region_nodes = list(self.mesh.nodes.keys())

                if region_nodes:
                    f_vec = load.get_force_vector(t=t, step_time=t, model=getattr(step, "parent_model", None))
                    for gid in region_nodes:
                        n_idx = nid_map[gid]
                        f_ext[3 * n_idx + 0] += f_vec[0]
                        f_ext[3 * n_idx + 1] += f_vec[1]
                        f_ext[3 * n_idx + 2] += f_vec[2]

            elif hasattr(load, "get_acceleration_vector"):
                g_vec = load.get_acceleration_vector(t=t, step_time=t, model=getattr(step, "parent_model", None))
                if hasattr(self, "elem_conn_0based") and self.elem_conn_0based is not None:
                    for e in range(self.elem_conn_0based.shape[0]):
                        conn_e = self.elem_conn_0based[e]
                        f_node = g_vec / 8.0
                        for n_idx in conn_e:
                            f_ext[3 * n_idx + 0] += f_node[0]
                            f_ext[3 * n_idx + 1] += f_node[1]
                            f_ext[3 * n_idx + 2] += f_node[2]
            else:
                raise NotImplementedError(f"Load type {type(load).__name__} integration is not yet supported in 3D solver.")

        return f_ext

    def _solve_step_with_hooks(
        self,
        dt: float,
        t_next: float,
        sys: Optional[Any],
        sensor_mgr: Any,
        hooks: List[Any],
        f_ext: Optional[np.ndarray] = None,
        max_iters: int = 50
    ) -> Tuple[bool, int]:
        """Newton-Raphson iteration loop with Sensor evaluation and IterationHooks execution."""
        from dispsolver.model.sensor import ControlAction

        if f_ext is None:
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

            if sensor_mgr and hooks:
                sensor_vals = sensor_mgr.evaluate_all(self, sys)
                for hook in hooks:
                    act = hook.execute(iter_count, t_next, self, sensor_vals)
                    if act == ControlAction.ABORT_STEP:
                        return False, iter_count

            K_bc, r_bc = self.apply_boundary_conditions(K_g, residual, u_k)
            rel_r = r_norm / r_0_norm

            bc_err = max([abs(target_val - u_k[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            if iter_count > 1 and bc_err < 1e-5 and (rel_r < 5e-3 or r_norm < 1e-3):
                self.u = u_k
                self.assemble_system(u_k, dt=dt, update_state=True)
                return True, iter_count

            try:
                du = self._solve_linear(K_bc, r_bc)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError("Pardiso linear solver failed (matrix is likely singular). Check boundary conditions or rigid body modes.") from e

            du_free = du[free_dof_mask]
            du_free_norm = float(np.linalg.norm(du_free))
            du_norm = float(np.linalg.norm(du))
            u_next = u_k + du
            bc_err_next = max([abs(target_val - u_next[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            u_free_norm = float(np.linalg.norm(u_k[free_dof_mask]))
            rel_du = du_free_norm / max(u_free_norm, 1.0)

            if iter_count > 1 and bc_err_next < 1e-5 and (rel_du < 1e-3 or du_free_norm < 1e-3 or du_norm < 1e-4):
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

            # Robust Armijo line search for energy/residual minimization.
            #
            # FIXED 2026-09-13 (dev_log/hertz_contact_benchmark_20260913.md,
            # found while verifying the SDI fix above): this loop used to
            # set `found_valid = True` unconditionally on its very FIRST
            # trial (`r_t_norm < best_r_norm or not found_valid`, and
            # `not found_valid` is always True on trial 0) -- so best_u
            # was seeded to the FULL step (s=1.0) regardless of whether it
            # actually improved the residual, and the `if not found_valid:
            # return False` cutback a few lines below could then only ever
            # fire if literally every one of the 6 halvings caused element
            # inversion. A full step that made the residual WORSE, with
            # none of the 5 remaining (smaller) halvings improving on it
            # either, was silently accepted anyway. This is not contact-
            # specific -- it is a base Newton-loop defect that happened to
            # rarely trigger because most of this solver's problems have a
            # full Newton step that naturally decreases residual; a
            # multi-node-simultaneous-contact system exposed it directly
            # (measured: residual crept from 1.112 to 1.143 over 50
            # "accepted" full-step iterations, never once rejected).
            # Fixed: found_valid is now only set True on a GENUINE
            # improvement over the running best (seeded to this
            # iteration's own r_norm, not an arbitrary first guess) or on
            # meeting the acceptance threshold -- so if none of the 6
            # halvings actually helps, found_valid correctly stays False
            # and the increment is genuinely cut back, exactly as a real
            # Armijo line search requires.
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
                if r_t_norm < best_r_norm:
                    best_r_norm = r_t_norm
                    best_u = u_trial
                    found_valid = True
                if r_t_norm <= max(r_norm, 1e-4 * r_0_norm, 1e-6):
                    best_u = u_trial
                    found_valid = True
                    break
                s *= 0.5

            if not found_valid:
                # All trial steps inverted or failed: cutback increment
                return False, iter_count

            u_k = best_u

        # Same false-convergence fallthrough as solve_step() above -- fixed
        # the same way (dev_log/solve_step_false_convergence_20260913.md).
        self.u = u_k
        if bc_err < 1e-5 and (rel_r < 5e-3 or r_norm < 1e-3):
            self.assemble_system(u_k, dt=dt, update_state=True)
            return True, max_iters
        return False, max_iters

    def solve(
        self,
        step: Any,
        sys: Optional[Any] = None,
        time_period: Optional[float] = None,
        sensors: Optional[List[Any]] = None,
        iteration_hooks: Optional[List[Any]] = None,
        output_mgr: Optional[Any] = None,
        verbose: bool = True
    ) -> bool:
        """Fully encapsulated Abaqus-like step solve engine."""
        from dispsolver.solver.dt_controller import AdaptiveDtController
        from dispsolver.model.sensor import SensorManager, Sensor, IterationHook

        T_total = time_period if time_period is not None else getattr(step, "time_period", 1.0)
        dt_init = getattr(step, "dt_init", 0.02)
        dt_min = getattr(step, "dt_min", 1e-5)
        dt_max = getattr(step, "dt_max", 0.05)
        target_iters = getattr(step, "target_iters", 8)

        dt_ctrl = AdaptiveDtController(dt_init=dt_init, dt_min=dt_min, dt_max=dt_max, target_iters=target_iters)

        sensor_mgr = SensorManager(sensors if sensors else [])
        if hasattr(step, "sensors"):
            for s in step.sensors.values():
                sensor_mgr.add_sensor(s)

        hooks = list(iteration_hooks) if iteration_hooks else []
        if hasattr(step, "iteration_hooks"):
            hooks.extend(list(step.iteration_hooks.values()))

        t = 0.0
        step_inc = 0
        total_iters = 0

        if not hasattr(self, "history_u") or self.history_u is None:
            self.history_u = []
        self.history_u.clear()
        self.history_u.append(self.u.copy())

        if not hasattr(self, "history_t") or self.history_t is None:
            self.history_t = []
        self.history_t.clear()
        self.history_t.append(0.0)

        if verbose:
            print(f"=== Starting Step '{step.name}' Solve (Time Period: {T_total:.3f} s) ===", flush=True)

        while t < T_total - 1e-12:
            dt = min(dt_ctrl.dt, T_total - t)
            t_next = t + dt
            step_inc += 1

            # Backup state in case cutback is needed
            u_prev = self.u.copy()
            sdvs_prev = self.elem_sdvs.copy() if self.elem_sdvs is not None else None

            self._update_step_bcs(step, sys, t_next, T_total)
            f_ext = self._compute_step_external_loads(step, sys, t_next, T_total)

            converged, iters = self._solve_step_with_hooks(
                dt=dt,
                t_next=t_next,
                sys=sys,
                sensor_mgr=sensor_mgr,
                hooks=hooks,
                f_ext=f_ext,
                max_iters=50
            )

            total_iters += iters

            if converged:
                t = t_next
                self.history_u.append(self.u.copy())
                self.history_t.append(t)
                if verbose:
                    print(f"  Inc {step_inc:3d}: t = {t:6.4f}s | dt = {dt:6.4f}s | Newton Iters = {iters:2d} | STATUS: CONVERGED", flush=True)
                dt_ctrl.notify_success(iters)

                if output_mgr is not None:
                    du = self.u - u_prev
                    is_last = (t >= T_total - 1e-12)
                    output_mgr.record_step(
                        step_time=t,
                        inc=step_inc,
                        solver=self,
                        dt=dt,
                        du=du,
                        f_int=self.f_int if hasattr(self, "f_int") and self.f_int is not None else np.zeros_like(self.u),
                        f_ext=f_ext,
                        is_last_step=is_last
                    )
            else:
                # Rollback displacement and SDV state on cutback
                self.u = u_prev
                if sdvs_prev is not None:
                    self.elem_sdvs = sdvs_prev.copy()
                if verbose:
                    print(f"  Inc {step_inc:3d}: t = {t:6.4f}s | dt = {dt:6.4f}s | Newton Iters = {iters:2d} | STATUS: CUTBACK", flush=True)
                ok = dt_ctrl.notify_cutback()
                if not ok:
                    if verbose:
                        print(f"!!! STEP ABORTED: dt reduced below dt_min ({dt_min}) !!!", flush=True)
                    return False

        if verbose:
            print(f"=== Step '{step.name}' Completed Successfully (Total Inc: {step_inc}, Total Newton Iters: {total_iters}) ===", flush=True)

        # Geostatic reset logic
        if getattr(step, "procedure", "").upper() == "GEOSTATIC":
            if verbose:
                print(f"    * GEOSTATIC PROCEDURE: Zeroing displacement and advancing internal stress baseline.", flush=True)
            self.u.fill(0.0)

        return True

    def solve_with_odb(
        self,
        step: Any,
        sys: Optional[Any] = None,
        odb_name: str = "Job-3D",
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
            ODB 이름 (기본값 "Job-3D").
        vtkhdf_path : str, optional
            실시간 스트리밍 저장 경로. 지정 시 매 Field Output 프레임마다
            즉시 .vtkhdf 파일에 append → 해석 중 ParaView에서 확인 가능.
        **kwargs
            DynamicSolver3D.solve()에 전달되는 추가 인수.

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
            dim=3,
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

    def compute_section_reactions(self, node_ids: List[int], center_pt: Optional[Tuple[float, float, float]] = None, current_coords: Optional[bool] = None) -> Dict[str, float]:
        """Compute resultant reaction forces (Fx, Fy, Fz) and moments (Mx, My, Mz) for a section of 3D nodes.
        center_pt: (x_c, y_c, z_c) reference point for moment calculation. If None, uses mean coordinate.
        current_coords: If True, uses deformed current coordinates (X + u). Defaults to True if nlgeom=True.
        Returns: {'Fx': float, 'Fy': float, 'Fz': float, 'Mx': float, 'My': float, 'Mz': float}
        """
        if current_coords is None:
            current_coords = getattr(self, "nlgeom", False)

        nid_map = self.mesh.node_id_to_index()
        if center_pt is None:
            if current_coords and self.u is not None:
                xc = float(np.mean([self.mesh.nodes[n].x + self.u[3 * nid_map[n]] for n in node_ids]))
                yc = float(np.mean([self.mesh.nodes[n].y + self.u[3 * nid_map[n] + 1] for n in node_ids]))
                zc = float(np.mean([self.mesh.nodes[n].z + self.u[3 * nid_map[n] + 2] for n in node_ids]))
            else:
                xc = float(np.mean([self.mesh.nodes[n].x for n in node_ids]))
                yc = float(np.mean([self.mesh.nodes[n].y for n in node_ids]))
                zc = float(np.mean([self.mesh.nodes[n].z for n in node_ids]))
        else:
            xc, yc, zc = center_pt

        Fx_tot, Fy_tot, Fz_tot = 0.0, 0.0, 0.0
        Mx_tot, My_tot, Mz_tot = 0.0, 0.0, 0.0

        if self.f_int is not None:
            for nid in node_ids:
                idx = nid_map[nid]
                node = self.mesh.nodes[nid]
                fx = self.f_int[3 * idx]
                fy = self.f_int[3 * idx + 1]
                fz = self.f_int[3 * idx + 2]
                if current_coords and self.u is not None:
                    dx = (node.x + self.u[3 * idx]) - xc
                    dy = (node.y + self.u[3 * idx + 1]) - yc
                    dz = (node.z + self.u[3 * idx + 2]) - zc
                else:
                    dx = node.x - xc
                    dy = node.y - yc
                    dz = node.z - zc
                Fx_tot += fx
                Fy_tot += fy
                Fz_tot += fz
                # M = r x F
                Mx_tot += (dy * fz - dz * fy)
                My_tot += (dz * fx - dx * fz)
                Mz_tot += (dx * fy - dy * fx)

        return {
            "Fx": float(Fx_tot), "Fy": float(Fy_tot), "Fz": float(Fz_tot),
            "Mx": float(Mx_tot), "My": float(My_tot), "Mz": float(Mz_tot)
        }

    def compute_strain_energy(self) -> float:
        """Compute total strain energy of the 3D mesh.
        Calculates 0.5 * sum_e (u_e^T * f_e) across all elements.
        """
        total_energy = 0.0
        for k, elem_indices in self.elem_kernel_groups.items():
            sub_conn = self.elem_conn_groups[k]
            for local_idx, e_idx in enumerate(elem_indices):
                conn = sub_conn[local_idx]
                n_en = len(conn)
                u_e = np.zeros(3 * n_en, dtype=np.float64)
                f_e = np.zeros(3 * n_en, dtype=np.float64)
                for a in range(n_en):
                    nid_idx = conn[a]
                    u_e[3 * a] = self.u[3 * nid_idx]
                    u_e[3 * a + 1] = self.u[3 * nid_idx + 1]
                    u_e[3 * a + 2] = self.u[3 * nid_idx + 2]
                    f_e[3 * a] = self.f_int[3 * nid_idx]
                    f_e[3 * a + 1] = self.f_int[3 * nid_idx + 1]
                    f_e[3 * a + 2] = self.f_int[3 * nid_idx + 2]
                total_energy += 0.5 * float(np.dot(u_e, f_e))

        return max(0.0, total_energy)

