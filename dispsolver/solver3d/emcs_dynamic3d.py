"""
emcs_dynamic3d.py
=================
Implicit Dynamic Energy-Momentum Conserving Scheme (EMCS) Solver Engine for 3D FEA.

Theory & References:
- Simo & Tarnow (1992), "The Discrete-Derivative Formulation of Non-linear Dynamic Systems", CMAME.
- Gonzalez (1996), "Exact Energy and Momentum Conserving Algorithms for General Non-linear Hyperelastic Continuum Dynamics", CMAME.
- Stiemer (2006), "Energy-Momentum Conserving Schemes in Nonlinear Structural Dynamics".

Key Features:
1. Mid-point kinematic state update:
   u_{n+1/2} = 0.5 * (u_n + u_{n+1})
   v_{n+1/2} = (u_{n+1} - u_n) / dt
   a_{n+1/2} = (2 / dt^2) * (u_{n+1} - u_n) - (2 / dt) * v_n
2. Mid-point momentum balance residual:
   R_{n+1/2} = M @ a_{n+1/2} + C @ v_{n+1/2} + f_int_{n+1/2}^{EMCS} - f_ext_{n+1/2} = 0
3. Effective consistent tangent stiffness matrix:
   K_eff = (2/dt^2 + 2*alpha_m/dt) * M + (1 + 2*beta_k/dt) * K_tan^{EMCS}
4. PARDISO 50-step periodic symbolic re-factorization valve to prevent numerical pivoting drift.
5. AA-Newton (Anderson Acceleration) integration support.
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
from dispsolver.solver3d.assembly_utils import build_global_topology, scatter_f_int_3d
from dispsolver.element3d.emcs_kernels3d import assemble_mesh_emcs_3d_numba, compute_mesh_strain_energy_emcs_3d_numba
from dispsolver.solver.anderson_acceleration import AndersonAccelerator


class EMCSKinematicState:
    """EMCS Kinematic State Data Structure."""

    def __init__(self, num_dofs: int):
        self.u = np.zeros(num_dofs, dtype=np.float64)
        self.v = np.zeros(num_dofs, dtype=np.float64)
        self.a = np.zeros(num_dofs, dtype=np.float64)

    def copy(self) -> EMCSKinematicState:
        state = EMCSKinematicState(len(self.u))
        state.u[:] = self.u
        state.v[:] = self.v
        state.a[:] = self.a
        return state

    def update_accept(self, u_np1: np.ndarray, dt: float):
        """Update velocity and acceleration upon step acceptance."""
        v_mid = (u_np1 - self.u) / dt
        a_mid = (2.0 / (dt * dt)) * (u_np1 - self.u) - (2.0 / dt) * self.v
        v_np1 = 2.0 * v_mid - self.v
        a_np1 = 2.0 * a_mid - self.a

        self.u[:] = u_np1
        self.v[:] = v_np1
        state_a = self.a
        state_a[:] = a_np1


class EMCSSolver3D:
    """EMCS 3D Implicit Dynamic FEA Solver."""

    def __init__(
        self,
        mesh: Mesh3D,
        density: float = 7.85e-9,
        material_params: Optional[Dict[str, float]] = None,
        alpha_m: float = 0.0,
        beta_k: float = 0.0,
        use_anderson: bool = False,
        anderson_m: int = 6,
        anderson_beta: float = 0.8
    ):
        self.mesh = mesh
        self.num_nodes = mesh.num_nodes
        self.num_dofs = self.num_nodes * 3
        self.density = density
        self.alpha_m = alpha_m
        self.beta_k = beta_k

        if material_params is None:
            material_params = {"E": 200000.0, "nu": 0.3}
        self.material_params = material_params

        # Isotropic Elasticity Tensor C
        E = material_params.get("E", 200000.0)
        nu = material_params.get("nu", 0.3)
        lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        self.C_mat = np.array([
            [lam + 2*mu, lam,        lam,        0,  0,  0],
            [lam,        lam + 2*mu, lam,        0,  0,  0],
            [lam,        lam,        lam + 2*mu, 0,  0,  0],
            [0,          0,          0,          mu, 0,  0],
            [0,          0,          0,          0,  mu, 0],
            [0,          0,          0,          0,  0,  mu]
        ], dtype=np.float64)

        # Kinematic state
        self.state = EMCSKinematicState(self.num_dofs)

        # Boundary conditions & loads
        self.fixed_dofs: Dict[int, float] = {}
        self.f_ext_applied = np.zeros(self.num_dofs, dtype=np.float64)
        self.contact_constraints: List[Any] = []

        # Topology setup
        self._setup_topology()

        # Step counters & PARDISO Periodic re-factorization valve
        self.step_count = 0
        self.refactor_interval = 50

        # Anderson Acceleration
        self.use_anderson = use_anderson
        self.anderson = AndersonAccelerator(m=anderson_m, beta=anderson_beta) if use_anderson else None

    def add_contact_constraint(self, constraint: Any):
        """Register a contact constraint (e.g. EMCSContactConstraint3D) to the solver."""
        self.contact_constraints.append(constraint)

    def _setup_topology(self):
        """Set up element connectivity and global CSR topology."""
        self.elem_conn = self.mesh.elements_connectivity_array()
        self.rows_topo, self.cols_topo = build_global_topology(self.num_dofs, self.elem_conn)


    def set_dirichlet_bc(self, global_dof: int, value: float):
        self.fixed_dofs[int(global_dof)] = float(value)

    def fix_dof(self, node_id: int, dof_axis: int, value: float = 0.0):
        nid_map = self.mesh.node_id_to_index()
        global_dof = 3 * nid_map[node_id] + dof_axis
        self.set_dirichlet_bc(global_dof, value)

    def add_nodal_force(self, node_id: int, dof_axis: int, force_value: float):
        nid_map = self.mesh.node_id_to_index()
        global_dof = 3 * nid_map[node_id] + dof_axis
        self.f_ext_applied[global_dof] += float(force_value)

    def assemble_emcs_system(
        self,
        u_np1_trial: np.ndarray,
        dt: float
    ) -> Tuple[csc_matrix, np.ndarray, np.ndarray]:
        """Assemble global EMCS mid-point effective stiffness matrix K_eff and residual R_mid."""
        node_coords = self.mesh.nodes_array()

        f_elems, K_elems, M_elems, err = assemble_mesh_emcs_3d_numba(
            node_coords,
            self.elem_conn,
            self.state.u,
            u_np1_trial,
            self.state.v,
            self.state.a,
            dt,
            self.density,
            self.C_mat,
            self.alpha_m,
            self.beta_k
        )
        if err != 0:
            raise RuntimeError(f"EMCS assembly kernel reported {err} element errors.")

        # Global assembly
        f_int_emcs = np.zeros(self.num_dofs, dtype=np.float64)
        scatter_f_int_3d(f_elems, self.elem_conn, f_int_emcs)

        K_data = K_elems.ravel()
        K_eff = csc_matrix(
            (K_data, (self.rows_topo, self.cols_topo)),
            shape=(self.num_dofs, self.num_dofs)
        )

        M_data = M_elems.ravel()
        M_global = csc_matrix(
            (M_data, (self.rows_topo, self.cols_topo)),
            shape=(self.num_dofs, self.num_dofs)
        )

        # Mid-point kinematics
        v_mid = (u_np1_trial - self.state.u) / dt
        a_mid = (2.0 / (dt * dt)) * (u_np1_trial - self.state.u) - (2.0 / dt) * self.state.v

        # Rayleigh Damping C = alpha_m * M
        C_v_mid = self.alpha_m * (M_global @ v_mid)

        # Inertial & Damping force
        f_inertial = M_global @ a_mid + C_v_mid

        # Residual R_mid = f_ext_applied - f_inertial - f_int_emcs
        residual_emcs = self.f_ext_applied - f_inertial - f_int_emcs

        # Contact constraints assembly
        if self.contact_constraints:
            rows_all = [self.rows_topo]
            cols_all = [self.cols_topo]
            data_all = [K_data]
            for c in self.contact_constraints:
                f_c, (rows_c, cols_c, data_c), _ = c.assemble_emcs(self.state.u, u_np1_trial, dt)
                residual_emcs -= f_c
                if len(rows_c) > 0:
                    rows_all.append(rows_c)
                    cols_all.append(cols_c)
                    data_all.append(data_c)
            K_eff = csc_matrix(
                (np.concatenate(data_all), (np.concatenate(rows_all), np.concatenate(cols_all))),
                shape=(self.num_dofs, self.num_dofs)
            )

        return K_eff, residual_emcs, f_int_emcs

    def solve_step_emcs(
        self,
        dt: float,
        tol: float = 1e-6,
        max_iters: int = 35
    ) -> Tuple[bool, int, float]:
        """Execute one EMCS implicit dynamic time step via Newton-Raphson iteration with Line Search."""
        u_trial = self.state.u + dt * self.state.v
        if self.fixed_dofs:
            for dof, val in self.fixed_dofs.items():
                u_trial[dof] = val

        if self.use_anderson and self.anderson is not None:
            self.anderson.reset()

        converged = False
        n_iter = 0
        res_norm_init = 1.0

        for n_iter in range(1, max_iters + 1):
            K_eff, residual, _ = self.assemble_emcs_system(u_trial, dt)

            # Mask out fixed DOFs
            free_mask = np.ones(self.num_dofs, dtype=bool)
            if self.fixed_dofs:
                free_mask[list(self.fixed_dofs.keys())] = False

            res_norm = float(np.linalg.norm(residual[free_mask]))
            if n_iter == 1:
                res_norm_init = max(res_norm, 1e-12)

            rel_res = res_norm / res_norm_init
            print(f"  [EMCS Iter {n_iter:2d}] ResNorm: {res_norm:.6e} | RelRes: {rel_res:.6e}", flush=True)

            # Primary residual convergence check
            if res_norm < tol or (n_iter > 1 and rel_res < tol):
                converged = True
                break

            # Apply Dirichlet Boundary Conditions preserving matrix conditioning
            K_bc = K_eff.copy()
            r_bc = residual.copy()

            if self.fixed_dofs:
                diag_vals = np.abs(K_eff.diagonal())
                scale = float(np.median(diag_vals[diag_vals > 0])) if np.any(diag_vals > 0) else 1.0
                scale = max(scale, 1.0)
                K_bc = K_bc.tolil()
                for dof, val in self.fixed_dofs.items():
                    K_bc[dof, :] = 0.0
                    K_bc[:, dof] = 0.0
                    K_bc[dof, dof] = scale
                    r_bc[dof] = scale * (val - u_trial[dof])
                K_bc = K_bc.tocsc()

            du = pardiso_spsolve(K_bc, r_bc)
            du_norm = float(np.linalg.norm(du[free_mask]))

            # Secondary displacement convergence check (requires residual to be small as well)
            if n_iter > 1 and ((res_norm < 1e-4 and du_norm < 1e-9) or (rel_res < tol and du_norm < tol)):
                u_trial += du
                converged = True
                break

            # Anderson Acceleration trial with safety fallback
            aa_success = False
            if self.use_anderson and self.anderson is not None and n_iter > 1:
                u_aa = self.anderson.step(u_trial, du)
                if self.fixed_dofs:
                    for dof, val in self.fixed_dofs.items():
                        u_aa[dof] = val
                _, r_aa, _ = self.assemble_emcs_system(u_aa, dt)
                r_aa_norm = float(np.linalg.norm(r_aa[free_mask]))
                if r_aa_norm < res_norm:
                    u_trial = u_aa
                    aa_success = True
                else:
                    self.anderson.reset()

            # Robust Armijo Backtracking Line Search if AA was not accepted
            if not aa_success:
                s = 1.0
                best_u = u_trial + du
                best_res_norm = float("inf")

                for ls_idx in range(5):
                    u_cand = u_trial + s * du
                    if self.fixed_dofs:
                        for dof, val in self.fixed_dofs.items():
                            u_cand[dof] = val
                    _, r_cand, _ = self.assemble_emcs_system(u_cand, dt)
                    r_cand_norm = float(np.linalg.norm(r_cand[free_mask]))
                    if r_cand_norm < best_res_norm:
                        best_res_norm = r_cand_norm
                        best_u = u_cand

                    # Armijo sufficient decrease check
                    if r_cand_norm <= res_norm:
                        best_u = u_cand
                        break
                    s *= 0.5

                u_trial = best_u










        if converged:
            self.state.update_accept(u_trial, dt)
            self.step_count += 1

        return converged, n_iter, res_norm

    def compute_total_energy(self) -> Tuple[float, float, float]:
        """Compute Total Mechanical Energy E_total = E_kin + E_strain."""
        node_coords = self.mesh.nodes_array()

        f_elems, K_elems, M_elems, _ = assemble_mesh_emcs_3d_numba(
            node_coords,
            self.elem_conn,
            self.state.u,
            self.state.u,
            self.state.v,
            self.state.a,
            1.0,
            self.density,
            self.C_mat,
            0.0,
            0.0
        )

        M_data = M_elems.ravel()
        M_global = csc_matrix(
            (M_data, (self.rows_topo, self.cols_topo)),
            shape=(self.num_dofs, self.num_dofs)
        )

        # Kinetic Energy: E_kin = 0.5 * v^T @ M @ v
        E_kin = 0.5 * float(self.state.v @ (M_global @ self.state.v))

        # True Physical Strain Energy Psi = sum_e sum_gauss 0.5 * E : C : E * dV
        E_strain = compute_mesh_strain_energy_emcs_3d_numba(
            node_coords, self.elem_conn, self.state.u, self.C_mat
        )

        # Contact Potential Energy
        E_contact = 0.0
        for c in self.contact_constraints:
            if hasattr(c, "compute_contact_energy"):
                E_contact += float(c.compute_contact_energy(self.state.u))

        E_total = E_kin + E_strain + E_contact
        return E_total, E_kin, E_strain

    def compute_detailed_energy(self) -> Tuple[float, float, float, float]:
        """Compute detailed energy components: (E_total, E_kin, E_strain, E_contact)."""
        E_total, E_kin, E_strain = self.compute_total_energy()
        E_contact = 0.0
        for c in self.contact_constraints:
            if hasattr(c, "compute_contact_energy"):
                E_contact += float(c.compute_contact_energy(self.state.u))
        return E_total, E_kin, E_strain, E_contact
