"""
dynamic.py
==========
Implicit dynamic FEM solver using Newmark integration + Newton-Raphson.

Formulation
-----------
- Total Lagrangian kinematics (reference-configuration strain measures)
- Q4 quadrilateral with B-bar SRI (volumetric locking mitigation)
- 2 DOF/node (ux, uy) — plane strain
- Newmark time integration (beta=0.25, gamma=0.5 default)
- Newton-Raphson equilibrium iteration with energy-norm convergence
- Supports Lagrange Multiplier constraints via scipy.sparse

Supports both:
  - JAX-based hyperelastic materials (MaterialModel subclass — no state)
  - Numpy-based inelastic materials (J2Plasticity, ViscoelasticMaterial — with state)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple
import time
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

try:
    from pypardiso import spsolve as pardiso_spsolve
    _PARDISO_AVAILABLE = True
except ImportError:
    _PARDISO_AVAILABLE = False


def _solve_linear_system(J, b, n_refine: int = 2, tol: float = 1e-11):
    """Solve J x = b with a multithreaded direct solver + iterative refinement.

    Multithreaded PARDISO can lose a few digits on the indefinite saddle-point
    KKT system created by Lagrange-multiplier constraints (thread-dependent
    pivot rounding). Iterative refinement restores full accuracy while keeping
    every core busy: after the first solve, the residual r = b - J x is solved
    again and added back. This is the same technique commercial direct solvers
    (e.g. Abaqus) use, so multi-core no longer degrades Newton convergence.
    """
    if _PARDISO_AVAILABLE:
        solve = lambda rhs: pardiso_spsolve(J, rhs)
    else:
        solve = lambda rhs: spla.spsolve(J, rhs)

    x = solve(b)
    b_norm = np.linalg.norm(b) + 1e-30
    for _ in range(n_refine):
        r = b - J @ x
        if np.linalg.norm(r) <= tol * b_norm:
            break
        x = x + solve(r)
    return x

from ..element import q4
from ..mesh import Mesh
from ..material.base import MaterialModel
from ..constraint.base import BaseConstraint


# ------------------------------------------------------------------
# Material adapter
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Material adapter
# ------------------------------------------------------------------

def _flat_to_tensor_3d(flat: np.ndarray, M: int) -> np.ndarray:
    """Unflatten a 1D state array of shape (6*(M+1),) into (M+1, 3, 3) tensor."""
    tensor = np.zeros((M + 1, 3, 3), dtype=np.float64)
    for i in range(M + 1):
        idx = i * 6
        v = flat[idx:idx+6]
        tensor[i, 0, 0] = v[0]
        tensor[i, 1, 1] = v[1]
        tensor[i, 2, 2] = v[2]
        tensor[i, 0, 1] = tensor[i, 1, 0] = v[3]
        tensor[i, 0, 2] = tensor[i, 2, 0] = v[4]
        tensor[i, 1, 2] = tensor[i, 2, 1] = v[5]
    return tensor


def _tensor_3d_to_flat(tensor: np.ndarray) -> np.ndarray:
    """Flatten a (M+1, 3, 3) tensor state array into (6*(M+1),) 1D array."""
    M_plus_1 = tensor.shape[0]
    flat = np.zeros(6 * M_plus_1, dtype=np.float64)
    for i in range(M_plus_1):
        T = tensor[i]
        idx = i * 6
        flat[idx]     = T[0, 0]
        flat[idx + 1] = T[1, 1]
        flat[idx + 2] = T[2, 2]
        flat[idx + 3] = T[0, 1]
        flat[idx + 4] = T[0, 2]
        flat[idx + 5] = T[1, 2]
    return flat


class MaterialAdapter:
    def __init__(self, material, params: Optional[Dict] = None):
        self.material = material
        self.params = params if params is not None else {}
        self.has_state = hasattr(material, "n_internal_vars")

    @property
    def n_internal_vars(self) -> int:
        if self.has_state:
            return int(self.material.n_internal_vars)
        return 0

    def __call__(
        self, F: np.ndarray, state_gp: Optional[np.ndarray] = None, dt: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        from ..material.viscoelastic import ViscoelasticMaterial
        
        if isinstance(self.material, ViscoelasticMaterial):
            dt_val = dt if dt is not None else 1e-14
            M = self.material.M
            if state_gp is None:
                h_prev = self.material.initial_internal_vars()
            else:
                h_prev = _flat_to_tensor_3d(state_gp, M)
            
            S, h_new = self.material.pk2_voigt(F, self.params, h_prev, dt_val)
            C = self.material.tangent_voigt(F, self.params, dt_val)
            state_new = _tensor_3d_to_flat(h_new)
            return S, C, state_new
            
        elif hasattr(self.material, "pk2_tensor_full"):
            S, C, state_new = self.material.pk2_voigt(F, self.params, state_gp)
            return S, C, state_new
            
        else:
            S = np.asarray(self.material.pk2_voigt(F, self.params))
            C = np.asarray(self.material.tangent_voigt(F, self.params))
            return S, C, None



# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _compute_F(coords: np.ndarray, u_elem: np.ndarray, xi: float, eta: float) -> np.ndarray:
    _, _, invJ = q4.jacobian(xi, eta, coords)
    dN_dxi, dN_deta = q4.shape_derivatives(xi, eta)

    grad_u = np.zeros((2, 2), dtype=np.float64)
    for a in range(4):
        dN_dx = invJ[0, 0] * dN_dxi[a] + invJ[0, 1] * dN_deta[a]
        dN_dy = invJ[1, 0] * dN_dxi[a] + invJ[1, 1] * dN_deta[a]
        grad_u[0, 0] += u_elem[2 * a] * dN_dx
        grad_u[0, 1] += u_elem[2 * a] * dN_dy
        grad_u[1, 0] += u_elem[2 * a + 1] * dN_dx
        grad_u[1, 1] += u_elem[2 * a + 1] * dN_dy

    return np.eye(2, dtype=np.float64) + grad_u

_GP2 = q4._GP2
_W2 = q4._W2

def _element_contributions(
    coords: np.ndarray, u_elem: np.ndarray, state_elem: np.ndarray, material: MaterialAdapter,
    dt: Optional[float] = None, thickness: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, invJ0 = q4.jacobian(0.0, 0.0, coords)
    B0 = q4.B_matrix(0.0, 0.0, invJ0)

    n_gp = len(_GP2)
    n_vars = material.n_internal_vars
    state_new = np.zeros((n_gp, n_vars), dtype=np.float64) if n_vars > 0 else None

    f_int = np.zeros(8, dtype=np.float64)
    K_e = np.zeros((8, 8), dtype=np.float64)

    for gp in range(n_gp):
        xi, eta = _GP2[gp]
        _, detJ, invJ = q4.jacobian(xi, eta, coords)

        Bb = q4.B_bar_matrix(xi, eta, invJ, B0, invJ0)
        F = _compute_F(coords, u_elem, xi, eta)
        if material.has_state:
            F_eval = F
        else:
            F0 = _compute_F(coords, u_elem, 0.0, 0.0)
            J = np.linalg.det(F)
            J0 = np.linalg.det(F0)
            ratio = J0 / np.where(np.abs(J) > 1e-30, J, 1e-30 * np.sign(J + 1e-35))
            if ratio < 0.0:
                F_eval = F * np.nan
            else:
                F_eval = F * np.sqrt(ratio)

        state_gp = state_elem[gp] if state_elem is not None else None
        S_v, C_v, sg_new = material(F_eval, state_gp, dt)

        w = detJ * _W2[gp] * thickness
        f_int += Bb.T @ S_v * w
        
        # Geometric stiffness
        dN_dxi, dN_deta = q4.shape_derivatives(xi, eta)
        dN_dX = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dY = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        grad_N = np.stack([dN_dX, dN_dY], axis=1)
        
        S_tensor = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
        gamma = grad_N @ S_tensor @ grad_N.T
        
        K_geo = np.zeros((8, 8))
        K_geo[0::2, 0::2] = gamma
        K_geo[1::2, 1::2] = gamma
        
        K_e += (Bb.T @ C_v @ Bb + K_geo) * w

        if n_vars > 0 and sg_new is not None:
            state_new[gp] = sg_new

    return f_int, K_e, state_new

def _lumped_mass_matrix(elem_coords: np.ndarray, conn: np.ndarray, rho: float, n_dofs: int,
                        thickness: Optional[np.ndarray] = None) -> np.ndarray:
    diag = np.zeros(n_dofs, dtype=np.float64)
    n_elem = len(elem_coords)
    for e in range(n_elem):
        t_e = 1.0 if thickness is None else float(thickness[e])
        M_e = q4.compute_M_lumped(elem_coords[e], rho) * t_e
        nids = conn[e]
        for a in range(4):
            dof_base = int(nids[a]) * 2
            row_base = 2 * a
            for i in range(2):
                diag[dof_base + i] += M_e[row_base + i, row_base + i]
    return diag


# ------------------------------------------------------------------
# Solver
# ------------------------------------------------------------------

class DynamicSolver:
    def __init__(
        self,
        mesh: Mesh,
        material: object,
        rho: float,
        material_params: Optional[Dict] = None,
        constraints: Optional[List[BaseConstraint]] = None,
        penalty_constraints: Optional[List] = None,
        beta: float = 0.25,
        gamma: float = 0.5,
        max_iter: int = 20,
        tol: float = 1e-8,
        verbose: bool = False,
        max_step: Optional[float] = None,
        element_type: str = "Q4",
        fast_assembly: bool = True,
        section_thickness=1.0,
    ):
        self.mesh = mesh
        self.fast_assembly = fast_assembly
        self.section_thickness = section_thickness
        if isinstance(material, dict):
            params_dict = material_params if isinstance(material_params, dict) else {}
            self.materials = {
                pid: MaterialAdapter(mat, params_dict.get(pid, {}))
                for pid, mat in material.items()
            }
            self.material = next(iter(self.materials.values()))
        else:
            adapter = MaterialAdapter(material, material_params)
            self.material = adapter
            self.materials = {0: adapter}

        self.rho = float(rho)
        self.beta = beta
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.max_step = max_step
        # element_type may be a single string (uniform) or a {pid: str} dict
        # (per-material) so e.g. viscoelastic layers use the Q1P0 hybrid while
        # plastic layers use standard Q4 B-bar.
        if isinstance(element_type, dict):
            self.element_type_by_pid = dict(element_type)
            self.element_type = "MIXED"
        else:
            self.element_type_by_pid = None
            self.element_type = element_type
        self.constraints = constraints if constraints is not None else []
        self.penalty_constraints = penalty_constraints if penalty_constraints is not None else []

        # --- mesh data
        conn, nid_to_idx, sorted_nids, elem_ids = mesh.connectivity_array()
        self.conn = conn
        self.elem_ids = elem_ids
        self.sorted_nids = np.array(sorted_nids)
        self.n_nodes = len(sorted_nids)
        self.n_dofs = self.n_nodes * 2
        self.n_elem = len(conn)
        
        # --- constraint setup
        self.n_extra = sum(c.n_extra_primal() for c in self.constraints)
        self.n_lambdas = sum(c.n_multipliers() for c in self.constraints)
        self.n_total = self.n_dofs + self.n_extra + self.n_lambdas

        self.elem_coords = mesh.nodes_array()[conn]

        # --- section (out-of-plane) thickness per element.
        # section_thickness may be a scalar (uniform) or a {pid: t} dict so each
        # stacked layer can have its own out-of-plane depth. The thickness scales
        # the element internal force, stiffness, and mass (plane-strain virtual
        # work integral). Constraints/contact are not auto-scaled.
        self._elem_thickness = np.ones(self.n_elem, dtype=np.float64)
        if isinstance(section_thickness, dict):
            for e in range(self.n_elem):
                elem = mesh.elements[elem_ids[e]]
                pid = elem.pid if elem.pid is not None else 0
                self._elem_thickness[e] = float(section_thickness.get(pid, 1.0))
        else:
            self._elem_thickness[:] = float(section_thickness)

        # --- mechanical state
        self.u = np.zeros(self.n_dofs, dtype=np.float64)
        self.v = np.zeros(self.n_dofs, dtype=np.float64)
        self.a = np.zeros(self.n_dofs, dtype=np.float64)
        
        self.u_extra = np.zeros(self.n_extra, dtype=np.float64)
        self.v_extra = np.zeros(self.n_extra, dtype=np.float64)
        self.a_extra = np.zeros(self.n_extra, dtype=np.float64)
        
        self.lam = np.zeros(self.n_lambdas, dtype=np.float64)

        # --- precompute dof indices for assembly
        self.dof_indices = np.zeros((self.n_elem, 8), dtype=np.int32)
        self.dof_indices[:, 0::2] = self.conn * 2
        self.dof_indices[:, 1::2] = self.conn * 2 + 1
        
        self.K_rows = np.repeat(self.dof_indices, 8, axis=1).flatten()
        self.K_cols = np.tile(self.dof_indices, (1, 8)).flatten()

        # --- pid grouping (always built for dict materials)
        self._pid_elem_indices: Dict[int, np.ndarray] = {}
        self._pid_K_rows: Dict[int, np.ndarray] = {}
        self._pid_K_cols: Dict[int, np.ndarray] = {}
        if isinstance(material, dict):
            pid_buckets: Dict[int, list] = {}
            for e in range(self.n_elem):
                eid = self.elem_ids[e]
                elem = self.mesh.elements[eid]
                pid = elem.pid if elem.pid is not None else 0
                pid_buckets.setdefault(pid, []).append(e)
            for pid, idxs in pid_buckets.items():
                arr = np.array(idxs, dtype=np.int32)
                dof_g = self.dof_indices[arr]
                self._pid_elem_indices[pid] = arr
                self._pid_K_rows[pid] = np.repeat(dof_g, 8, axis=1).flatten()
                self._pid_K_cols[pid] = np.tile(dof_g, (1, 8)).flatten()

        # --- check if material is JAX-compatible for vectorization
        self.use_jax_vmap = False
        self.use_jax_grouped_vmap = False
        self.use_multi_material_batch = False
        from ..material.base import MaterialModel
        from ..material.plastic import J2Plasticity
        try:
            import jax
            from .dynamic_jax import build_element_contributions_jax, build_hybrid_element_contributions_jax

            if not isinstance(material, dict):
                # Single material path
                if isinstance(self.material.material, MaterialModel):
                    self.use_jax_vmap = True
                    if self.element_type == "Q4_UP":
                        elem_fn = build_hybrid_element_contributions_jax(self.material.params)
                    else:
                        elem_fn = build_element_contributions_jax(self.material.material, self.material.params)
                    self.vmapped_elem_contribs = jax.jit(jax.vmap(elem_fn))
            else:
                all_jax = all(isinstance(m.material, MaterialModel) for m in self.materials.values())
                if all_jax and self.element_type != "Q4_UP":
                    # All JAX hyperelastic → grouped vmap
                    self.use_jax_grouped_vmap = True
                    params_dict = material_params if isinstance(material_params, dict) else {}
                    self.vmapped_elem_contribs_by_pid = {
                        pid: jax.jit(jax.vmap(
                            build_element_contributions_jax(mat.material, params_dict.get(pid, {}))
                        ))
                        for pid, mat in self.materials.items()
                    }
                elif any(isinstance(m.material, J2Plasticity) for m in self.materials.values()):
                    # Mixed: some J2 (batch), others sequential
                    self.use_multi_material_batch = True
        except ImportError:
            pass
        self.M = _lumped_mass_matrix(self.elem_coords, conn, self.rho, self.n_dofs, self._elem_thickness)

        max_vars = max(m.n_internal_vars for m in self.materials.values())
        n_gp = len(_GP2)
        if max_vars > 0:
            self.state = np.zeros((self.n_elem, n_gp, max_vars), dtype=np.float64)
            for e in range(self.n_elem):
                eid = self.elem_ids[e]
                elem = self.mesh.elements[eid]
                pid = elem.pid if elem.pid is not None else 0
                mat_adapter = self.materials.get(pid, self.material)
                if hasattr(mat_adapter.material, "initial_internal_vars"):
                    init = mat_adapter.material.initial_internal_vars()
                    if isinstance(init, np.ndarray) and init.ndim > 1:
                        init_flat = _tensor_3d_to_flat(init)
                    else:
                        init_flat = init
                    n_vars_curr = len(init_flat)
                    for gp in range(n_gp):
                        self.state[e, gp, :n_vars_curr] = init_flat
        else:
            self.state = None

        # --- Precompute reference geometry (B-bar, weights, dN/dX) for all elements × GPs
        self._B_bar_all, self._weights_all, self._dN_dX_all = self._precompute_reference_geometry()

        # --- Detect single-material J2 batch path
        self.use_j2_batch = False
        if not isinstance(material, dict) and not self.use_jax_vmap:
            from ..material.plastic import J2Plasticity
            if isinstance(self.material.material, J2Plasticity):
                self.use_j2_batch = True

        # --- fast_assembly=False forces the proven sequential element loop
        #     (batched NumPy tangents can be subtly inconsistent and hurt
        #      Newton convergence on extreme-deformation steps). JAX vmap
        #      paths use exact autodiff tangents, so they remain enabled.
        if not self.fast_assembly:
            self.use_j2_batch = False
            self.use_multi_material_batch = False

        # A uniform Q4_UP mesh is assembled entirely in the sequential path.
        # A MIXED mesh keeps the batch path (J2 groups stay vectorized) and
        # routes only the hybrid pid-groups to the sequential hybrid kernel.
        if self.element_type == "Q4_UP":
            self.use_j2_batch = False
            self.use_multi_material_batch = False
            self.use_jax_grouped_vmap = False
        elif self.element_type_by_pid is not None:
            # Per-pid element types → use the multi-material grouped assembler
            self.use_jax_vmap = False
            self.use_jax_grouped_vmap = False
            self.use_j2_batch = False
            self.use_multi_material_batch = True

        self.f_ext = np.zeros(self.n_dofs, dtype=np.float64)
        self.bc_dofs = np.array([], dtype=np.int32)
        self.bc_vals = np.array([], dtype=np.float64)
        self._bc_base_vals = np.array([], dtype=np.float64)
        self._bc_amplitudes = None      # None, single Amplitude, or per-dof list
        self.time = 0.0                 # accumulated analysis time

    def set_prescribed_dofs(
        self,
        bc_dofs: np.ndarray,
        bc_vals: Optional[np.ndarray] = None,
        amplitudes=None,
    ) -> None:
        """Prescribe DOF values, optionally time-dependent via Amplitude curves.

        amplitudes : None | Amplitude | list
            - None: constant values (bc_vals held over time).
            - a single Amplitude: applied to every dof; value(t) = bc_vals * a(t).
            - a list (len == n_dofs): per-dof amplitude; entries may be None
              (that dof stays constant) or an Amplitude.
        The effective value at time t is  base_value * amplitude(t).
        """
        self.bc_dofs = np.asarray(bc_dofs, dtype=np.int32).ravel()
        if bc_vals is None:
            self._bc_base_vals = np.zeros(len(self.bc_dofs), dtype=np.float64)
        else:
            self._bc_base_vals = np.asarray(bc_vals, dtype=np.float64).ravel()
            assert len(self._bc_base_vals) == len(self.bc_dofs)
        self._bc_amplitudes = amplitudes
        self.bc_vals = self._eval_bc_vals(self.time)

    def _eval_bc_vals(self, t: float) -> np.ndarray:
        """Effective prescribed values at time t (base * amplitude)."""
        base = self._bc_base_vals
        amp = self._bc_amplitudes
        if amp is None:
            return base.copy()
        if isinstance(amp, (list, tuple, np.ndarray)):
            out = base.copy()
            for i, a in enumerate(amp):
                if a is not None:
                    out[i] = base[i] * a(t)
            return out
        # single Amplitude applied to all dofs
        return base * amp(t)

    def apply_load(self, dofs: Sequence[int], values: Sequence[float]) -> None:
        for d, v in zip(dofs, values):
            self.f_ext[d] = v

    def reaction_forces(self, u: Optional[np.ndarray] = None) -> np.ndarray:
        if u is None:
            u = self.u
            u_ext = self.u_extra
            lam = self.lam
        else:
            u_ext = np.zeros(self.n_extra)
            lam = np.zeros(self.n_lambdas)
        
        f_int, _, _ = self._assemble(u)
        
        # Constraint forces: C_u^T * lam
        f_c = np.zeros(self.n_dofs)
        lam_offset = 0
        for c in self.constraints:
            n_lam = c.n_multipliers()
            if n_lam > 0:
                row_u, col_u, val_u, _, _, _, _ = c.assemble(u, u_ext)
                for i in range(len(val_u)):
                    eq_idx = row_u[i] + lam_offset
                    dof_idx = col_u[i]
                    f_c[dof_idx] += val_u[i] * lam[eq_idx]
            lam_offset += n_lam
            
        a = self.a.copy() if u is self.u else np.zeros(self.n_dofs)
        return f_int + f_c + self.M * a - self.f_ext

    def _compute_R_total(
        self,
        u_k: np.ndarray,
        u_ext_k: np.ndarray,
        lam_k: np.ndarray,
        u_n: np.ndarray,
        v_n: np.ndarray,
        a_n: np.ndarray,
        u_ext_n: np.ndarray,
        v_ext_n: np.ndarray,
        a_ext_n: np.ndarray,
        dt: float,
        inv_beta_dt2: float,
        beta: float,
    ) -> np.ndarray:
        f_int, _, _ = self._assemble(u_k, dt)
        
        a_k = (u_k - u_n - dt * v_n) * inv_beta_dt2 - (1.0 - 2.0 * beta) / (2.0 * beta) * a_n
        a_ext_k = (u_ext_k - u_ext_n - dt * v_ext_n) * inv_beta_dt2 - (1.0 - 2.0 * beta) / (2.0 * beta) * a_ext_n

        R_u = self.f_ext - self.M * a_k - f_int
        R_ext = np.zeros(self.n_extra)
        R_lam = np.zeros(self.n_lambdas)
        
        lam_offset = 0
        for c in self.constraints:
            n_lam = c.n_multipliers()
            if n_lam > 0:
                r_u, c_u, v_u, r_ext, c_ext, v_ext, g = c.assemble(u_k, u_ext_k)
                
                for i in range(len(v_u)):
                    eq_idx = r_u[i] + lam_offset
                    dof_idx = c_u[i]
                    R_u[dof_idx] -= v_u[i] * lam_k[eq_idx]
                    
                for i in range(len(v_ext)):
                    eq_idx = r_ext[i] + lam_offset
                    ext_idx = c_ext[i]
                    R_ext[ext_idx] -= v_ext[i] * lam_k[eq_idx]
                    
                for i in range(n_lam):
                    R_lam[lam_offset + i] = -g[i]
                    
            lam_offset += n_lam
            
        R_total = np.concatenate([R_u, R_ext, R_lam])
        
        if len(self.bc_dofs) > 0:
            for i, idx in enumerate(self.bc_dofs):
                val = self.bc_vals[i]
                if idx < self.n_dofs:
                    R_total[idx] = val - u_k[idx]
                else:
                    ext_idx = idx - self.n_dofs
                    R_total[idx] = val - u_ext_k[ext_idx]
                
        return R_total

    def solve_step(self, dt: float) -> int:
        t_start = time.time()
        beta, gamma, dt2 = self.beta, self.gamma, dt * dt

        # Evaluate time-dependent prescribed values at the end of the increment
        # (Abaqus-style amplitude ramp). On cutback dt shrinks and the target
        # time is recomputed from the (unadvanced) current time.
        if self._bc_amplitudes is not None:
            self.bc_vals = self._eval_bc_vals(self.time + dt)

        u_n = self.u.copy()
        v_n = self.v.copy()
        a_n = self.a.copy()
        u_ext_n = self.u_extra.copy()
        v_ext_n = self.v_extra.copy()
        a_ext_n = self.a_extra.copy()

        u_k = u_n + dt * v_n + 0.5 * dt2 * a_n
        u_ext_k = u_ext_n + dt * v_ext_n + 0.5 * dt2 * a_ext_n
        lam_k = self.lam.copy()

        has_bc = len(self.bc_dofs) > 0
        inv_beta_dt2 = 1.0 / (beta * dt2)

        if getattr(self, 'verbose', False):
            print(f"", flush=True)
            print(f"  Nonlinear Iteration Summary (Newton-Raphson)", flush=True)
            print(f"  ==========================================================================================================================", flush=True)
            print(f"  Time increment dt: {dt:.3e}", flush=True)
            print(f"  Convergence Tol  : {self.tol:.1e} (Disp Ratio)     Max Iterations: {self.max_iter}", flush=True)
            print(f"  --------------------------------------------------------------------------------------------------------------------------", flush=True)
            print(f"  Iter   Max Res.Force  (Node/DOF)    Max Disp.Corr  (Node/DOF)    Max Disp.Incr  Disp.Ratio  Energy.Err  LS.alpha  Contacts", flush=True)
            print(f"  --------------------------------------------------------------------------------------------------------------------------", flush=True)

        # --- debug check
        if np.any(np.isnan(u_k) | np.isinf(u_k)):
            print(f"DEBUG: u_k has NaN at start of solve_step! (dt={dt:.3e})", flush=True)
            print(f"  u_n has NaN: {np.any(np.isnan(u_n))}", flush=True)
            print(f"  v_n has NaN: {np.any(np.isnan(v_n))}", flush=True)
            print(f"  a_n has NaN: {np.any(np.isnan(a_n))}", flush=True)

        res_norm_0 = None
        converged = False
        for n_iter in range(self.max_iter):
            # --- debug check
            if np.any(np.isnan(u_k)):
                print(f"DEBUG: u_k has NaN before assembly at iter {n_iter+1}!", flush=True)
                
            f_int, K_T, state_new = self._assemble(u_k, dt)
            
            # --- debug check
            if np.any(np.isnan(f_int)) or (K_T is not None and np.any(np.isnan(K_T.data))):
                print(f"DEBUG: NaN detected in assembly at iter {n_iter+1}!", flush=True)
                print(f"  f_int has NaN: {np.any(np.isnan(f_int))}", flush=True)
                if K_T is not None:
                    if hasattr(K_T, "data"):
                        print(f"  K_T.data has NaN: {np.any(np.isnan(K_T.data))}", flush=True)
                    else:
                        print(f"  K_T has NaN (non-coo/csr): {np.any(np.isnan(K_T.toarray()))}", flush=True)
                if self.state is not None:
                    print(f"  self.state has NaN: {np.any(np.isnan(self.state))}", flush=True)
                
                # Check displacement scales
                max_u_idx = np.argmax(np.abs(u_k))
                max_u_node = self.sorted_nids[max_u_idx // 2]
                max_u_dof = 'UX' if max_u_idx % 2 == 0 else 'UY'
                print(f"  Max displacement magnitude in u_k: {u_k[max_u_idx]:.5e} at Node {max_u_node}({max_u_dof})", flush=True)

            a_k = (u_k - u_n - dt * v_n) * inv_beta_dt2 - (1.0 - 2.0 * beta) / (2.0 * beta) * a_n
            a_ext_k = (u_ext_k - u_ext_n - dt * v_ext_n) * inv_beta_dt2 - (1.0 - 2.0 * beta) / (2.0 * beta) * a_ext_n

            R_u = self.f_ext - self.M * a_k - f_int
            R_ext = np.zeros(self.n_extra)
            R_lam = np.zeros(self.n_lambdas)
            
            # Constraints
            C_row_u, C_col_u, C_val_u = [], [], []
            C_row_ext, C_col_ext, C_val_ext = [], [], []
            
            lam_offset = 0
            for c in self.constraints:
                n_lam = c.n_multipliers()
                if n_lam > 0:
                    r_u, c_u, v_u, r_ext, c_ext, v_ext, g = c.assemble(u_k, u_ext_k)
                    
                    for i in range(len(v_u)):
                        eq_idx = r_u[i] + lam_offset
                        dof_idx = c_u[i]
                        R_u[dof_idx] -= v_u[i] * lam_k[eq_idx]
                        C_row_u.append(eq_idx)
                        C_col_u.append(dof_idx)
                        C_val_u.append(v_u[i])
                        
                    for i in range(len(v_ext)):
                        eq_idx = r_ext[i] + lam_offset
                        ext_idx = c_ext[i]
                        R_ext[ext_idx] -= v_ext[i] * lam_k[eq_idx]
                        C_row_ext.append(eq_idx)
                        C_col_ext.append(ext_idx)
                        C_val_ext.append(v_ext[i])
                        
                    for i in range(n_lam):
                        R_lam[lam_offset + i] = -g[i]
                        
                lam_offset += n_lam
            
            # Combine Residual
            R_total = np.concatenate([R_u, R_ext, R_lam])

            # Combine Stiffness
            K_eff = K_T.copy()
            K_eff.setdiag(K_eff.diagonal() + inv_beta_dt2 * self.M)
            
            # Convert to COO for global assembly
            K_eff_coo = K_eff.tocoo()
            global_row = list(K_eff_coo.row)
            global_col = list(K_eff_coo.col)
            global_val = list(K_eff_coo.data)
            
            # Assemble C_u^T and C_u
            lam_start_idx = self.n_dofs + self.n_extra
            for r, c, v in zip(C_row_u, C_col_u, C_val_u):
                global_row.append(r + lam_start_idx)
                global_col.append(c)
                global_val.append(v)
                
                global_row.append(c)
                global_col.append(r + lam_start_idx)
                global_val.append(v)
                
            # Assemble C_ext^T and C_ext
            ext_start_idx = self.n_dofs
            for r, c, v in zip(C_row_ext, C_col_ext, C_val_ext):
                global_row.append(r + lam_start_idx)
                global_col.append(c + ext_start_idx)
                global_val.append(v)
                
                global_row.append(c + ext_start_idx)
                global_col.append(r + lam_start_idx)
                global_val.append(v)

            # Extra primal inertia (0 for now, but add small regularization to avoid singular diagonal)
            for i in range(self.n_extra):
                global_row.append(ext_start_idx + i)
                global_col.append(ext_start_idx + i)
                global_val.append(1e-12)  # small stabilization
                
            # Lambda stabilization (optional, avoid saddle point issues, usually spsolve handles it)
            # but sometimes zeros on diagonal cause UMFPACK to complain.
            # We will rely on spsolve unless it fails.
            
            J = sps.coo_matrix((global_val, (global_row, global_col)), shape=(self.n_total, self.n_total)).tocsr()

            if has_bc:
                for i, idx in enumerate(self.bc_dofs):
                    val = self.bc_vals[i]
                    # Zero out row idx
                    start_ptr = J.indptr[idx]
                    end_ptr = J.indptr[idx+1]
                    for ptr in range(start_ptr, end_ptr):
                        col = J.indices[ptr]
                        if col == idx:
                            J.data[ptr] = 1.0
                        else:
                            J.data[ptr] = 0.0
                    if idx < self.n_dofs:
                        R_total[idx] = val - u_k[idx]
                    else:
                        ext_idx = idx - self.n_dofs
                        R_total[idx] = val - u_ext_k[ext_idx]

            # Solve (multithreaded direct + iterative refinement)
            du_all = _solve_linear_system(J, R_total)
            
            # Check if search direction has NaN or Inf
            if np.any(np.isnan(du_all) | np.isinf(du_all)):
                if getattr(self, 'verbose', False):
                    t_elapsed = time.time() - t_start
                    print(f"  ---------------------------------------------------------------------------------------------------------------------", flush=True)
                    print(f"  *** ERROR: Search direction contains NaN/Inf at iteration {n_iter+1}. (Time: {t_elapsed:.2f} s)\n", flush=True)
                return -(n_iter + 1)
            
            du = du_all[:self.n_dofs]
            du_ext = du_all[self.n_dofs:self.n_dofs+self.n_extra]
            dlam = du_all[self.n_dofs+self.n_extra:]

            # --- Backtracking Line Search (NaN/Inf-prevention) ---
            # Full Newton steps are kept (quadratic convergence on the
            # indefinite KKT system, whose residual norm is not a valid merit
            # function); alpha is only damped when a trial step produces a
            # non-finite residual (geometric blow-up / element inversion).
            alpha = 1.0
            alpha_min = 0.1
            R_norm_current = np.linalg.norm(R_total)

            while alpha > alpha_min + 1e-5:
                u_temp = u_k + alpha * du
                u_ext_temp = u_ext_k + alpha * du_ext
                lam_temp = lam_k + alpha * dlam

                R_temp = self._compute_R_total(
                    u_temp, u_ext_temp, lam_temp,
                    u_n, v_n, a_n,
                    u_ext_n, v_ext_n, a_ext_n,
                    dt, inv_beta_dt2, self.beta
                )
                R_norm_temp = np.linalg.norm(R_temp)

                if not np.isnan(R_norm_temp) and not np.isinf(R_norm_temp):
                    break
                alpha *= 0.5

            # If line search failed to find a non-NaN/non-inf residual
            if np.isnan(R_norm_temp) or np.isinf(R_norm_temp):
                if getattr(self, 'verbose', False):
                    t_elapsed = time.time() - t_start
                    print(f"  ---------------------------------------------------------------------------------------------------------------------", flush=True)
                    print(f"  *** ERROR: Line search failed to resolve NaN/Inf residual at iteration {n_iter+1}. (Time: {t_elapsed:.2f} s)\n", flush=True)
                return -(n_iter + 1)

            du *= alpha
            du_ext *= alpha
            dlam *= alpha
            du_all *= alpha
            
            res_norm = R_norm_current
            if res_norm_0 is None:
                res_norm_0 = res_norm + 1e-16
            res_ratio = res_norm / res_norm_0
            du_norm = np.linalg.norm(du)
            if self.max_step is not None and du_norm > self.max_step:
                scale = self.max_step / du_norm
                du *= scale
                du_ext *= scale
                dlam *= scale
                du_all *= scale
                du_norm = self.max_step
                
            u_norm = np.linalg.norm(u_k)
            rel_change = du_norm / (u_norm + 1e-12)
            
            if getattr(self, 'verbose', False):
                # 1. Max active force residual
                active_dofs = np.ones(self.n_dofs, dtype=bool)
                if len(self.bc_dofs) > 0:
                    node_bc_dofs = self.bc_dofs[self.bc_dofs < self.n_dofs]
                    active_dofs[node_bc_dofs] = False
                
                if np.any(active_dofs):
                    abs_R_active = np.abs(R_u) * active_dofs
                    max_R_val = np.max(abs_R_active)
                    max_R_dof = np.argmax(abs_R_active)
                    max_R_node_id = self.sorted_nids[max_R_dof // 2]
                    max_R_dof_name = 'UX' if max_R_dof % 2 == 0 else 'UY'
                    max_R_str = f"N{max_R_node_id}({max_R_dof_name})"
                else:
                    max_R_val = 0.0
                    max_R_str = "N/A"

                # 2. Max displacement correction
                if self.n_dofs > 0:
                    abs_du = np.abs(du)
                    max_du_val = np.max(abs_du)
                    max_du_dof = np.argmax(abs_du)
                    max_du_node_id = self.sorted_nids[max_du_dof // 2]
                    max_du_dof_name = 'UX' if max_du_dof % 2 == 0 else 'UY'
                    max_du_str = f"N{max_du_node_id}({max_du_dof_name})"
                else:
                    max_du_val = 0.0
                    max_du_str = "N/A"

                # 3. Max displacement increment
                delta_u = u_k - u_n
                max_disp_incr = np.max(np.abs(delta_u)) if self.n_dofs > 0 else 0.0

                energy_err = np.abs(np.dot(R_total, du_all))
                n_contacts = 0
                for pc in self.penalty_constraints:
                    if hasattr(pc, "n_active"):
                        n_contacts += getattr(pc, "n_active", 0)
                
                print(
                    f"  {n_iter+1:4d}   "
                    f"{max_R_val:11.4e}  {max_R_str:<13s} "
                    f"{max_du_val:11.4e}  {max_du_str:<13s} "
                    f"{max_disp_incr:11.4e}   "
                    f"{rel_change:10.2e}  "
                    f"{energy_err:10.2e}  "
                    f"{alpha:8.2f}  "
                    f"{n_contacts:8d}",
                    flush=True
                )

            if np.isnan(du_norm) or np.isinf(du_norm) or (res_norm_0 is not None and res_ratio > 1e15):
                if getattr(self, 'verbose', False):
                    t_elapsed = time.time() - t_start
                    print(f"  ---------------------------------------------------------------------------------------------------------------------", flush=True)
                    print(f"  *** ERROR: Divergence detected at iteration {n_iter+1}. (Time: {t_elapsed:.2f} s)\n", flush=True)
                return -(self.max_iter)

            energy_err = np.abs(np.dot(R_total, du_all))
            if (rel_change < self.tol or energy_err < 1e-15) and n_iter > 0:
                converged = True
                self.time += dt          # advance analysis time on success only
                if state_new is not None:
                    self.state = state_new
                if getattr(self, 'verbose', False):
                    t_elapsed = time.time() - t_start
                    print(f"  ---------------------------------------------------------------------------------------------------------------------", flush=True)
                    print(f"  => Increment Converged. (Time: {t_elapsed:.2f} s)\n", flush=True)
                break

            u_k += du
            u_ext_k += du_ext
            lam_k += dlam

        self.u = u_k.copy()
        # Guard against unphysical acceleration / velocity spikes due to tiny dt cutbacks
        self.a = np.clip(a_k.copy(), -1.0e4, 1.0e4)
        self.v = np.clip(v_n + dt * ((1.0 - gamma) * a_n + gamma * a_k), -1.0e3, 1.0e3)
        
        self.u_extra = u_ext_k.copy()
        self.a_extra = np.clip(a_ext_k.copy(), -1.0e4, 1.0e4)
        self.v_extra = np.clip(v_ext_n + dt * ((1.0 - gamma) * a_ext_n + gamma * a_ext_k), -1.0e3, 1.0e3)
        
        self.lam = lam_k.copy()

        if not converged:
            if getattr(self, 'verbose', False):
                t_elapsed = time.time() - t_start
                print(f"  ---------------------------------------------------------------------------------------------------------------------", flush=True)
                print(f"  *** ERROR: Failed to converge after {self.max_iter} iterations. (Time: {t_elapsed:.2f} s)\n", flush=True)
            return -(self.max_iter)

        return n_iter

    def solve(self, num_steps: int, dt: float, callback: Optional[Callable] = None) -> List[np.ndarray]:
        history = []
        for step in range(num_steps):
            n_iter = self.solve_step(dt)
            if n_iter < 0:
                import warnings
                warnings.warn(f"Step {step}: NR did not converge ({-n_iter} iters)")
            history.append(self.u.copy())
            if callback:
                callback(step, self)
        return history

    def save_state(self) -> dict:
        """Save current solver state for adaptive time stepping rollback.

        Returns
        -------
        state_dict : dict
            Snapshot of all mutable solver state variables.
        """
        d = {
            'u': self.u.copy(),
            'v': self.v.copy(),
            'a': self.a.copy(),
            'u_extra': self.u_extra.copy(),
            'v_extra': self.v_extra.copy(),
            'a_extra': self.a_extra.copy(),
            'lam': self.lam.copy(),
            'time': self.time,
        }
        if self.state is not None:
            d['state'] = self.state.copy()
        return d

    def restore_state(self, state_dict: dict) -> None:
        """Restore solver state from a saved snapshot.

        Parameters
        ----------
        state_dict : dict
            Previously saved state from save_state().
        """
        self.u = state_dict['u'].copy()
        self.v = state_dict['v'].copy()
        self.a = state_dict['a'].copy()
        self.u_extra = state_dict['u_extra'].copy()
        self.v_extra = state_dict['v_extra'].copy()
        self.a_extra = state_dict['a_extra'].copy()
        self.lam = state_dict['lam'].copy()
        if 'time' in state_dict:
            self.time = state_dict['time']
        if 'state' in state_dict and self.state is not None:
            self.state = state_dict['state'].copy()

    def _pid_element_type(self, pid: int) -> str:
        """Element type for a material group (per-pid override or global)."""
        if self.element_type_by_pid is None:
            return self.element_type
        return self.element_type_by_pid.get(pid, "Q4")

    def _assemble_multi_material_batch(self, u: np.ndarray, dt=None):
        """Mixed-strategy assembly for multi-material meshes.

        J2Plasticity groups  → fully vectorized batch (pk2_voigt_batch).
        All other groups     → sequential per-element loop.
        """
        from ..material.plastic import J2Plasticity

        n_gp = len(_GP2)
        f_int = np.zeros(self.n_dofs, dtype=np.float64)
        max_vars = max(m.n_internal_vars for m in self.materials.values())
        state_new = (np.zeros_like(self.state) if self.state is not None else None)

        all_K_rows, all_K_cols, all_K_vals = [], [], []

        from ..material.linear_viscoelastic import LinearViscoelastic

        for pid, elem_indices in self._pid_elem_indices.items():
            mat_adapter = self.materials[pid]

            if (self._pid_element_type(pid) == "Q4_UP"
                    and isinstance(mat_adapter.material, LinearViscoelastic)):
                # ---- Q1P0 hybrid (mean-dilatation) for linear viscoelasticity ----
                from ..element.q4_visco_hybrid import compute_visco_hybrid_contributions
                n_vars = mat_adapter.n_internal_vars
                temp = getattr(self, 'temperature', 20.0)
                dt_h = dt if dt is not None else 1.0
                for e in elem_indices:
                    coords = self.elem_coords[e]
                    nids = self.conn[e]
                    u_elem = u[self.dof_indices[e]]
                    state_elem = (self.state[e, :, :n_vars]
                                  if self.state is not None else None)
                    f_e, K_e, se_new = compute_visco_hybrid_contributions(
                        coords, u_elem, state_elem, mat_adapter.material, dt_h, temp,
                        self._elem_thickness[e],
                    )
                    np.add.at(f_int, self.dof_indices[e], f_e)
                    if state_new is not None and se_new is not None:
                        state_new[e, :, :n_vars] = se_new
                    dof_g = self.dof_indices[e]
                    all_K_rows.append(np.repeat(dof_g, 8))
                    all_K_cols.append(np.tile(dof_g, 8))
                    all_K_vals.append(K_e.flatten())
                continue

            if isinstance(mat_adapter.material, J2Plasticity):
                # ---- Batch path ----
                B_bar_g   = self._B_bar_all[elem_indices]    # (Ng, n_gp, 3, 8)
                weights_g = self._weights_all[elem_indices]  # (Ng, n_gp)
                dN_dX_g   = self._dN_dX_all[elem_indices]   # (Ng, n_gp, 2, 4)

                u_elems = u[self.dof_indices[elem_indices]]
                ux = u_elems[:, 0::2]
                uy = u_elems[:, 1::2]
                grad_ux = np.einsum('ea,egja->egj', ux, dN_dX_g)
                grad_uy = np.einsum('ea,egja->egj', uy, dN_dX_g)

                Ng = len(elem_indices)
                F_all = np.empty((Ng, n_gp, 2, 2), dtype=np.float64)
                F_all[:, :, 0, 0] = 1.0 + grad_ux[:, :, 0]
                F_all[:, :, 0, 1] =       grad_ux[:, :, 1]
                F_all[:, :, 1, 0] =       grad_uy[:, :, 0]
                F_all[:, :, 1, 1] = 1.0 + grad_uy[:, :, 1]

                N_tot      = Ng * n_gp
                F_flat     = F_all.reshape(N_tot, 2, 2)
                n_vars     = mat_adapter.n_internal_vars
                state_flat = self.state[elem_indices].reshape(N_tot, max_vars)[:, :n_vars]

                S_flat, C_flat, sn_flat = mat_adapter.material.pk2_voigt_batch(F_flat, {}, state_flat)

                S_all = S_flat.reshape(Ng, n_gp, 3)
                C_all = C_flat.reshape(Ng, n_gp, 3, 3)

                if state_new is not None:
                    state_new[elem_indices, :, :n_vars] = sn_flat.reshape(Ng, n_gp, n_vars)

                f_e = np.einsum('egab,ega,eg->eb', B_bar_g, S_all, weights_g)
                np.add.at(f_int, self.dof_indices[elem_indices].flatten(), f_e.flatten())

                K_mat = np.einsum('egki,egkl,eglj,eg->egij', B_bar_g, C_all, B_bar_g, weights_g)

                S_tensor = np.zeros((Ng, n_gp, 2, 2), dtype=np.float64)
                S_tensor[:, :, 0, 0] = S_all[:, :, 0]
                S_tensor[:, :, 1, 1] = S_all[:, :, 1]
                S_tensor[:, :, 0, 1] = S_tensor[:, :, 1, 0] = S_all[:, :, 2]
                gamma = np.einsum('egka,egkl,eglb->egab', dN_dX_g, S_tensor, dN_dX_g)
                gamma_w = gamma * weights_g[:, :, None, None]
                K_geo = np.zeros((Ng, n_gp, 8, 8), dtype=np.float64)
                K_geo[:, :, 0::2, 0::2] = gamma_w
                K_geo[:, :, 1::2, 1::2] = gamma_w

                K_e_all = (K_mat + K_geo).sum(axis=1)  # (Ng, 8, 8)
                all_K_rows.append(self._pid_K_rows[pid])
                all_K_cols.append(self._pid_K_cols[pid])
                all_K_vals.append(K_e_all.flatten())

            elif hasattr(mat_adapter.material, 'pk2_tangent_voigt_batch'):
                # ---- Batch path for ViscoelasticMaterial ----
                B_bar_g   = self._B_bar_all[elem_indices]    # (Ng, n_gp, 3, 8)
                weights_g = self._weights_all[elem_indices]  # (Ng, n_gp)
                dN_dX_g   = self._dN_dX_all[elem_indices]   # (Ng, n_gp, 2, 4)

                u_elems = u[self.dof_indices[elem_indices]]
                ux = u_elems[:, 0::2]
                uy = u_elems[:, 1::2]
                grad_ux = np.einsum('ea,egja->egj', ux, dN_dX_g)
                grad_uy = np.einsum('ea,egja->egj', uy, dN_dX_g)

                Ng = len(elem_indices)
                F_all = np.empty((Ng, n_gp, 2, 2), dtype=np.float64)
                F_all[:, :, 0, 0] = 1.0 + grad_ux[:, :, 0]
                F_all[:, :, 0, 1] =       grad_ux[:, :, 1]
                F_all[:, :, 1, 0] =       grad_uy[:, :, 0]
                F_all[:, :, 1, 1] = 1.0 + grad_uy[:, :, 1]

                N_tot      = Ng * n_gp
                F_flat     = F_all.reshape(N_tot, 2, 2)
                n_vars     = mat_adapter.n_internal_vars
                state_flat = self.state[elem_indices].reshape(N_tot, max_vars)[:, :n_vars]

                S_flat, C_flat, sn_flat = mat_adapter.material.pk2_tangent_voigt_batch(
                    F_flat, mat_adapter.params, state_flat, dt if dt is not None else 1.0
                )

                S_all = S_flat.reshape(Ng, n_gp, 3)
                C_all = C_flat.reshape(Ng, n_gp, 3, 3)

                if state_new is not None:
                    state_new[elem_indices, :, :n_vars] = sn_flat.reshape(Ng, n_gp, n_vars)

                f_e = np.einsum('egab,ega,eg->eb', B_bar_g, S_all, weights_g)
                np.add.at(f_int, self.dof_indices[elem_indices].flatten(), f_e.flatten())

                K_mat = np.einsum('egki,egkl,eglj,eg->egij', B_bar_g, C_all, B_bar_g, weights_g)

                S_tensor = np.zeros((Ng, n_gp, 2, 2), dtype=np.float64)
                S_tensor[:, :, 0, 0] = S_all[:, :, 0]
                S_tensor[:, :, 1, 1] = S_all[:, :, 1]
                S_tensor[:, :, 0, 1] = S_tensor[:, :, 1, 0] = S_all[:, :, 2]
                gamma = np.einsum('egka,egkl,eglb->egab', dN_dX_g, S_tensor, dN_dX_g)
                gamma_w = gamma * weights_g[:, :, None, None]
                K_geo = np.zeros((Ng, n_gp, 8, 8), dtype=np.float64)
                K_geo[:, :, 0::2, 0::2] = gamma_w
                K_geo[:, :, 1::2, 1::2] = gamma_w

                K_e_all = (K_mat + K_geo).sum(axis=1)  # (Ng, 8, 8)
                all_K_rows.append(self._pid_K_rows[pid])
                all_K_cols.append(self._pid_K_cols[pid])
                all_K_vals.append(K_e_all.flatten())

            else:
                # ---- Sequential fallback for any remaining material types ----
                K_seq = sps.lil_matrix((self.n_dofs, self.n_dofs), dtype=np.float64)
                for e in elem_indices:
                    coords  = self.elem_coords[e]
                    nids    = self.conn[e]
                    u_elem  = np.zeros(8, dtype=np.float64)
                    for a in range(4):
                        dof = int(nids[a]) * 2
                        u_elem[2*a]   = u[dof]
                        u_elem[2*a+1] = u[dof+1]

                    n_vars     = mat_adapter.n_internal_vars
                    state_elem = self.state[e, :, :n_vars] if self.state is not None else None
                    f_e, K_e, se_new = _element_contributions(
                        coords, u_elem, state_elem, mat_adapter, dt, self._elem_thickness[e])

                    for a in range(4):
                        dof_a = int(nids[a]) * 2
                        for i in range(2):
                            f_int[dof_a+i] += f_e[2*a+i]
                            for b in range(4):
                                dof_b = int(nids[b]) * 2
                                for j in range(2):
                                    K_seq[dof_a+i, dof_b+j] += K_e[2*a+i, 2*b+j]

                    if state_new is not None and se_new is not None:
                        state_new[e, :, :n_vars] = se_new

                K_seq_coo = K_seq.tocoo()
                all_K_rows.append(K_seq_coo.row)
                all_K_cols.append(K_seq_coo.col)
                all_K_vals.append(K_seq_coo.data)

        K_T = sps.coo_matrix(
            (np.concatenate(all_K_vals),
             (np.concatenate(all_K_rows), np.concatenate(all_K_cols))),
            shape=(self.n_dofs, self.n_dofs)
        ).tocsr()

        return f_int, K_T, state_new

    def _precompute_reference_geometry(self):
        """Precompute B-bar, integration weights, and dN/dX for all elements × GPs."""
        n_gp = len(_GP2)
        B_bar_all   = np.zeros((self.n_elem, n_gp, 3, 8), dtype=np.float64)
        weights_all = np.zeros((self.n_elem, n_gp),        dtype=np.float64)
        dN_dX_all   = np.zeros((self.n_elem, n_gp, 2, 4), dtype=np.float64)

        for e in range(self.n_elem):
            coords = self.elem_coords[e]
            _, _, invJ0 = q4.jacobian(0.0, 0.0, coords)
            B0 = q4.B_matrix(0.0, 0.0, invJ0)
            for gp in range(n_gp):
                xi, eta = _GP2[gp]
                _, detJ, invJ = q4.jacobian(xi, eta, coords)
                B_bar_all[e, gp] = q4.B_bar_matrix(xi, eta, invJ, B0, invJ0)
                weights_all[e, gp] = detJ * _W2[gp] * self._elem_thickness[e]
                dN_dxi, dN_deta = q4.shape_derivatives(xi, eta)
                dN_dX_all[e, gp, 0] = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
                dN_dX_all[e, gp, 1] = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta

        return B_bar_all, weights_all, dN_dX_all

    def _assemble_j2_batch(self, u: np.ndarray, dt=None):
        """Fully vectorized assembly for single-material J2Plasticity (no Python GP/element loop)."""
        n_gp = len(_GP2)
        f_int = np.zeros(self.n_dofs, dtype=np.float64)

        # Deformation gradients for all elements × GPs: (N_elem, n_gp, 2, 2)
        u_elems = u[self.dof_indices]                        # (N_elem, 8)
        ux = u_elems[:, 0::2]                               # (N_elem, 4)
        uy = u_elems[:, 1::2]                               # (N_elem, 4)
        # grad_u[e,g,i,j] = sum_a u_i[e,a] * dN_dX[e,g,j,a]
        grad_ux = np.einsum('ea,egja->egj', ux, self._dN_dX_all)  # (N_elem, n_gp, 2)
        grad_uy = np.einsum('ea,egja->egj', uy, self._dN_dX_all)
        F_all = np.empty((self.n_elem, n_gp, 2, 2), dtype=np.float64)
        F_all[:, :, 0, 0] = 1.0 + grad_ux[:, :, 0]
        F_all[:, :, 0, 1] =       grad_ux[:, :, 1]
        F_all[:, :, 1, 0] =       grad_uy[:, :, 0]
        F_all[:, :, 1, 1] = 1.0 + grad_uy[:, :, 1]

        # Batch material call: (N_elem*n_gp, 2, 2) -> S, C, state
        N_tot = self.n_elem * n_gp
        F_flat     = F_all.reshape(N_tot, 2, 2)
        state_flat = self.state.reshape(N_tot, self.state.shape[2])[:, :5]

        S_flat, C_flat, state_new_flat = self.material.material.pk2_voigt_batch(F_flat, {}, state_flat)

        S_all = S_flat.reshape(self.n_elem, n_gp, 3)
        C_all = C_flat.reshape(self.n_elem, n_gp, 3, 3)
        state_new = np.zeros_like(self.state)
        state_new[:, :, :5] = state_new_flat.reshape(self.n_elem, n_gp, 5)

        # Internal forces: f_e[e,b] = sum_g sum_a B_bar[e,g,a,b] * S[e,g,a] * w[e,g]
        f_e = np.einsum('egab,ega,eg->eb', self._B_bar_all, S_all, self._weights_all)
        np.add.at(f_int, self.dof_indices.flatten(), f_e.flatten())

        # Material stiffness: K_mat[e,g,i,j] = sum_{k,l} B_bar[e,g,k,i]*C[e,g,k,l]*B_bar[e,g,l,j]*w
        K_mat_eg = np.einsum('egki,egkl,eglj,eg->egij',
                              self._B_bar_all, C_all, self._B_bar_all, self._weights_all)

        # Geometric stiffness: gamma[e,g,a,b] = sum_{kl} dN_dX[e,g,k,a]*S_kl[e,g]*dN_dX[e,g,l,b]
        S_tensor = np.zeros((self.n_elem, n_gp, 2, 2), dtype=np.float64)
        S_tensor[:, :, 0, 0] = S_all[:, :, 0]
        S_tensor[:, :, 1, 1] = S_all[:, :, 1]
        S_tensor[:, :, 0, 1] = S_tensor[:, :, 1, 0] = S_all[:, :, 2]
        gamma = np.einsum('egka,egkl,eglb->egab', self._dN_dX_all, S_tensor, self._dN_dX_all)
        gamma_w = gamma * self._weights_all[:, :, None, None]

        K_geo_eg = np.zeros((self.n_elem, n_gp, 8, 8), dtype=np.float64)
        K_geo_eg[:, :, 0::2, 0::2] = gamma_w
        K_geo_eg[:, :, 1::2, 1::2] = gamma_w

        K_e_all = (K_mat_eg + K_geo_eg).sum(axis=1)         # (N_elem, 8, 8)
        K_T = sps.coo_matrix(
            (K_e_all.flatten(), (self.K_rows, self.K_cols)),
            shape=(self.n_dofs, self.n_dofs)
        ).tocsr()

        return f_int, K_T, state_new

    def _assemble(self, u: np.ndarray, dt: Optional[float] = None) -> Tuple[np.ndarray, sps.csr_matrix, Optional[np.ndarray]]:
        f_int = np.zeros(self.n_dofs, dtype=np.float64)
        n_gp = len(_GP2)
        max_vars = max(m.n_internal_vars for m in self.materials.values())

        if self.use_jax_vmap:
            # --- Single-material Vectorized JAX Assembly ---
            u_elems = u[self.dof_indices]
            f_es, K_es = self.vmapped_elem_contribs(self.elem_coords, u_elems)
            f_es = np.asarray(f_es) * self._elem_thickness[:, None]
            K_es = np.asarray(K_es) * self._elem_thickness[:, None, None]

            np.add.at(f_int, self.dof_indices.flatten(), f_es.flatten())

            K_T = sps.coo_matrix(
                (K_es.flatten(), (self.K_rows, self.K_cols)),
                shape=(self.n_dofs, self.n_dofs)
            ).tocsr()

            state_new = None

        elif self.use_j2_batch:
            return self._assemble_j2_batch(u, dt)

        elif self.use_multi_material_batch:
            return self._assemble_multi_material_batch(u, dt)

        elif self.use_jax_grouped_vmap:
            # --- Multi-material Grouped Vectorized JAX Assembly ---
            all_K_rows, all_K_cols, all_K_vals = [], [], []

            for pid, elem_indices in self._pid_elem_indices.items():
                coords_g = self.elem_coords[elem_indices]
                u_elems_g = u[self.dof_indices[elem_indices]]

                t_g = self._elem_thickness[elem_indices]
                f_es, K_es = self.vmapped_elem_contribs_by_pid[pid](coords_g, u_elems_g)
                f_es = np.asarray(f_es) * t_g[:, None]
                K_es = np.asarray(K_es) * t_g[:, None, None]

                np.add.at(f_int, self.dof_indices[elem_indices].flatten(), f_es.flatten())

                all_K_rows.append(self._pid_K_rows[pid])
                all_K_cols.append(self._pid_K_cols[pid])
                all_K_vals.append(K_es.flatten())

            K_T = sps.coo_matrix(
                (np.concatenate(all_K_vals),
                 (np.concatenate(all_K_rows), np.concatenate(all_K_cols))),
                shape=(self.n_dofs, self.n_dofs)
            ).tocsr()

            state_new = None

        else:
            # --- Sequential Fallback Assembly ---
            K_T = sps.lil_matrix((self.n_dofs, self.n_dofs), dtype=np.float64)
            state_new = np.zeros((self.n_elem, n_gp, max_vars), dtype=np.float64) if max_vars > 0 else None

            for e in range(self.n_elem):
                coords = self.elem_coords[e]
                nids = self.conn[e]

                u_elem = np.zeros(8, dtype=np.float64)
                for a in range(4):
                    dof = int(nids[a]) * 2
                    u_elem[2 * a] = u[dof]
                    u_elem[2 * a + 1] = u[dof + 1]

                eid = self.elem_ids[e]
                elem = self.mesh.elements[eid]
                pid = elem.pid if elem.pid is not None else 0
                mat_adapter = self.materials.get(pid, self.material)

                if self._pid_element_type(pid) == "Q4_UP":
                    from ..material.linear_viscoelastic import LinearViscoelastic
                    if isinstance(mat_adapter.material, LinearViscoelastic):
                        # Stress-based hybrid (mean-dilatation) for linear viscoelasticity
                        from ..element.q4_visco_hybrid import compute_visco_hybrid_contributions
                        n_vars = mat_adapter.n_internal_vars
                        state_elem = (self.state[e][:, :n_vars]
                                      if self.state is not None else None)
                        temp = getattr(self, 'temperature', 20.0)
                        f_e, K_e, se_new = compute_visco_hybrid_contributions(
                            coords, u_elem, state_elem, mat_adapter.material,
                            dt if dt is not None else 1.0, temp, self._elem_thickness[e],
                        )
                    else:
                        from ..element.q4_up_jax import compute_hybrid_element_contributions
                        f_e_jax, K_e_jax = compute_hybrid_element_contributions(coords, u_elem, mat_adapter.params)
                        f_e = np.asarray(f_e_jax) * self._elem_thickness[e]
                        K_e = np.asarray(K_e_jax) * self._elem_thickness[e]
                        se_new = None
                else:
                    state_elem = self.state[e][:, :mat_adapter.n_internal_vars] if self.state is not None else None
                    f_e, K_e, se_new = _element_contributions(
                        coords, u_elem, state_elem, mat_adapter, dt, self._elem_thickness[e])

                for a in range(4):
                    dof_a = int(nids[a]) * 2
                    for i in range(2):
                        f_int[dof_a + i] += f_e[2 * a + i]
                        for b in range(4):
                            dof_b = int(nids[b]) * 2
                            for j in range(2):
                                K_T[dof_a + i, dof_b + j] += K_e[2 * a + i, 2 * b + j]

                if max_vars > 0 and se_new is not None and state_new is not None:
                    state_new[e, :, :mat_adapter.n_internal_vars] = se_new
            K_T = K_T.tocsr()

        # --- Penalty constraints ---
        if self.penalty_constraints:
            # Convert to lil for efficient element-wise modification
            K_T = K_T.tolil()
            for pc in self.penalty_constraints:
                pc.apply_penalty(u, f_int, K_T)
            K_T = K_T.tocsr()

        return f_int, K_T, state_new
