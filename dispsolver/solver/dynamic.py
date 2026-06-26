"""
dynamic.py
==========
Implicit FEM dynamic solver — Newmark-β / HHT-α time integration
with Newton-Raphson equilibrium iteration.

====================  ==========  ===========  ==================================
Quantity              Symbol      Value        Reference
====================  ==========  ===========  ==================================
Solver type           —           Implicit     —
Time integration      β, γ        β=0.25       Newmark (1959); Hughes (1987,
                      (Newmark)   γ=0.5        Ch. 9) — trapezoidal rule,
                                               unconditional stability for
                                               linear undamped systems
Numerical damping     α (HHT)     α ∈ [-0.3,0] Hilber-Hughes-Taylor (1977);
via spectral radius   β=(1-α)²/4               Comp. Meth. 16(1), 1-16
                      γ=½-α
Spatial discretisat.  Bilinear    Q4            Hughes (1987) Ch. 4
                      quadrilateral
Volumetric locking    B-bar SRI   mean-         Hughes (1980) IJNME 15,
mitigation                       dilatation    1413-1430
Material nonlinearity  Newton-    Full Newton   —
                       Raphson    per iter.
Constitutive updates   J2 radial  Spectral      Simo (1992) CMAME 99(1),
                       return map  return map    61-112
Constraints            Lagrange   —             —
                       Multipliers
Self-contact           JAX        —             Wriggers (2006)
                       penalty                  "Computational Contact
                                                 Mechanics", Springer
Stiffness reg-         LM damping —             Levenberg (1944), Marquardt
ularization            (optional)                (1963); Deuflhard (2004)
====================  ==========  ===========  ==================================

Formulation
-----------
*Total Lagrangian kinematics* — all strain measures are referred to the
reference configuration.  The deformation gradient F = I + du/dX, 2nd Piola-
Kirchhoff stress S, and Green-Lagrange strain E are computed from the
reference geometry.  The solver does NOT update element coordinates.

*Newmark family* — the acceleration a_{n+1} and velocity v_{n+1} at step
n+1 are expressed in terms of the unknown displacement u_{n+1}:

    v_{n+1} = v_n + Δt·[(1−γ)·a_n + γ·a_{n+1}]
    u_{n+1} = u_n + Δt·v_n + Δt²·[(½−β)·a_n + β·a_{n+1}]

For β=¼, γ=½ (default, "trapezoidal rule"): 2nd-order accurate, no numerical
damping, unconditional stability for linear undamped systems.

*HHT-α method* (a.k.a. α-method, Hilber-Hughes-Taylor 1977):  generalises
Newmark by weighting the internal force at an intermediate time level:

    M·a_{n+1} + (1+α)·f_int(u_{n+1}) − α·f_int(u_n) = f_ext(t_{n+1})
    α ∈ [−⅓, 0]   ⇒   β = (1−α)²/4,  γ = ½−α

α=0 recovers the trapezoidal rule.  α<0 introduces high-frequency dissipation
(spurious ringing damping) while preserving 2nd-order accuracy and unconditional
stability for linear systems.  The solver provides named presets:
  "transient"   α=+0.00  — no damping, exact energy conservation
  "moderate-1"  α=−0.05  — light HF damping
  "moderate-2"  α=−0.15  — stronger damping, good for folding problems
  "quasistatic" static   — inertia removed, rate kept via dt for viscoelasticity

*Newton-Raphson linearisation* — the residual R(u) = f_M(u) + f_int(u) − f_ext
is linearised about the current iterate u_k:

    K_eff · du = −R_k   where   K_eff = ∂R/∂u|_{u_k}

For the HHT-α method K_eff = (1+α)·K_T + (1/β/Δt²)·M when the inertial term
includes the HHT-α weighting.  Solve via PARDISO direct solver (saddle-point
KKT system from LM constraints) with iterative refinement.

*LM damping* (Levenberg-Marquardt regularisation) — when the tangent
stiffness becomes nearly singular (buckling / bifurcation), the solver
can optionally add a fraction λ·|diag(K_eff)| to the diagonal of the
system matrix before factoring.  This shifts the near-zero eigenvalues
by λ·|diag|, regularising the search direction while biasing it toward
steepest descent.  λ is controlled externally via the module variable
_LM_ACTIVE (see ex03_optimized_v3.py for the standard workflow).

Project-specific implementation notes (display folding)
-------------------------------------------------------
- Multi-material mesh:  PET (J2Plasticity, layers 0,2,4,6) uses
  Q4_EAS (Simo-Rifai EAS-4) elements to avoid bending locking through
  the thin (0.05mm/3 ≈ 0.017mm) sub-layers.
  PSA (LinearViscoelastic, layers 1,3,5) uses Q4_UP hybrid elements
  (Q1P0 mean-dilatation) for near-incompressibility (ν=0.49).
- Assembly strategy:  the `_assemble_multi_material_batch` method
  dispatches each pid-group to its optimal path: JAX vmap for Q4_EAS+J2,
  JAX vmap for Q4_UP+visco, or batch NumPy for pure J2.  When JAX vmap
  returns NaN for any element (excessive element distortion), it falls
  back to per-element sequential assembly.
- Constraints: RBE2HingeConstraint (rigid-body rotation of flap + hinge)
  uses Lagrange multipliers.  PenaltyContactConstraint (self-contact)
  uses a JAX-differentiated penalty potential with spatial hash search.
- PARDISO multithreaded solver with iterative refinement: the KKT system
  from Lagrange multipliers is a saddle-point matrix (indefinite, zero
  diagonal blocks).  PARDISO can lose precision on small pivots under
  heavy threading  →  iterative refinement (2-4 passes) recovers accuracy.
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


def _equilibrate(A: sps.csr_matrix, b: np.ndarray):
    """Diagonal equilibration (A * x = b) → (Ã * y = b̅),  x = D⁻¹·y.

    For KKT saddle-point systems the mechanical block (K_eff ≈ 10⁵) and the
    constraint block (C ≈ 1) differ by 5 orders of magnitude.  Equilibration
    rescales every row/column so |Ã_ii| ≈ 1, giving PARDISO a uniformly-scaled
    system whose pivot search is no longer biased by this unit mismatch.

    Lagrange-multiplier rows with |J_ii| < 1e-30 are assigned the median scale
    of the mechanical rows, balancing the C-block entries without making the
    zero-diagonal rows vanish in the scaled system.
    """
    diag = A.diagonal().copy()
    abs_diag = np.abs(diag)
    # Median of non‑zero mechanical diagonals
    mech_mask = abs_diag > 1e-30
    if np.any(mech_mask):
        med = float(np.median(abs_diag[mech_mask]))
        med_sqrt = np.sqrt(med)
    else:
        med_sqrt = 1.0
    scale = np.where(mech_mask, 1.0 / np.sqrt(abs_diag), 1.0 / med_sqrt)

    # Symmetric scaling:  Ã_ij = A_ij · s_i · s_j
    As = A.copy()
    for row in range(A.shape[0]):
        s, e = As.indptr[row], As.indptr[row + 1]
        As.data[s:e] *= scale[row]
        As.data[s:e] *= scale[As.indices[s:e]]
    bs = scale * b
    return As, bs, scale


def _solve_linear_system(J, b, n_refine: int = 4, tol: float = 1e-14):
    """Solve J x = b with diagonal equilibration + tuned PARDISO (KKT-aware).

    1. Equilibrate  →  J_s · y = b_s   (all |diag| ≈ 1)
    2. Factor/solve with ``_KKT_SOLVER`` (symmetric indefinite, iparm tuned).
    3. Iterative refinement on the EQUILIBRATED system (which is better
       conditioned, so refinement converges in fewer passes).
    4. Unscale  →  x = scale · y

    Parameters
    ----------
    n_refine : int
        Maximum iterative refinement passes (default 4).
    tol : float
        Stop when ‖r‖ ≤ tol · ‖b_s‖  (default 1e-14).
    """
    if _PARDISO_AVAILABLE:
        # ---- 1. Equilibrate KKT system ----
        Js, bs, scale = _equilibrate(J, b)

        # ---- 2. Solve scaled system via PARDISO ----
        # pardiso_spsolve uses mtype=11 (unsymmetric) by default, which is
        # suboptimal for the symmetric KKT but works robustly even with the
        # 2×2 pivot blocks (PARDISO's internal pivot search handles this).
        # Equilibration reduces κ(J) from ~10⁶ to <10², so the unsymmetric
        # solver no longer loses digits on the cross-block scaling.
        y = pardiso_spsolve(Js, bs)

        # ---- 3. Iterative refinement on SCALED system ----
        b_norm = np.linalg.norm(bs) + 1e-30
        for _ in range(n_refine):
            r = bs - Js @ y
            r_norm = np.linalg.norm(r)
            if r_norm <= tol * b_norm or r_norm < 1e-30:
                break
            dy = pardiso_spsolve(Js, r)
            y = y + dy

        # ---- 4. Unscale solution ----
        x = scale * y
        return x
    else:
        # Scipy fallback (no equilibration — SuperLU handles the units mismatch
        # via its own row/column scaling internally).
        x = spla.spsolve(J, b)
        b_norm = np.linalg.norm(b) + 1e-30
        for _ in range(n_refine):
            r = b - J @ x
            if np.linalg.norm(r) <= tol * b_norm:
                break
            x = x + spla.spsolve(J, r)
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

# ------------------------------------------------------------------
# Time-integration mode presets
# ------------------------------------------------------------------
# Named integration "characters" so users pick behaviour without hand-tuning
# the HHT-α / Newmark coefficients (which must stay mutually consistent:
# β = (1-α)²/4, γ = 1/2 - α).
#
# The spectral radius ρ(∞) for α<0 asymptotically approaches zero for
# high frequencies (ω → ∞): ρ(∞) = 1+α  (Hughes 1987, §9.4.3).
#   α =  0   →  ρ(∞)=1   trapezoidal rule (no damping, 2nd-order accurate)
#   α = -0.05 →  ρ(∞)=0.95  light HF damping
#   α = -0.15 →  ρ(∞)=0.85  strong HF damping, good for folding simulation
#   static    →  inertia removed; rate kept via dt for viscoelasticity
#
# The folding example uses "moderate-2" (α=-0.15) because the near-singular
# tangent stiffness at the bifurcation point (hinge compression zone) would
# trigger spurious high-frequency oscillations in the Newmark trapezoidal rule.
# HHT-α damping suppresses these without adding artificial bulk viscosity.
#
# Reference:
#   Hilber, H.M., Hughes, T.J.R. & Taylor, R.L. (1977). Improved numerical
#   dissipation for time integration algorithms in structural dynamics.
#   Earthquake Engineering & Structural Dynamics, 5(3), 283-292.
def _hht(alpha: float) -> dict:
    return dict(static_mode=False, alpha=alpha,
                beta=(1.0 - alpha) ** 2 / 4.0, gamma=0.5 - alpha)


INTEGRATION_MODES = {
    "transient":   _hht(0.0),
    "moderate-1":  _hht(-0.05),
    "moderate-2":  _hht(-0.15),
    "quasistatic": dict(static_mode=True, alpha=0.0, beta=0.25, gamma=0.5),
}


class DynamicSolver:
    """Implicit dynamic FEM solver with multi-material, multi-constraint support.

    This class manages the full simulation state: mesh connectivity, element
    topology, material models (per-pid), internal variables (state per GP),
    constraint manifold (Lagrange + penalty), and the Newmark/HHT-α time
    integrator.

    Key design decisions
    --------------------
    *State architecture*: displacement u, velocity v, acceleration a are stored
    as flat 1D numpy arrays (2 DOF/node, plane strain).  The extra primal DOFs
    (hinge rotations) are stored in u_extra, and Lagrange multipliers in lam.

    *Per-pid element routing*: each material group (pid) can use a different
    element formulation: Q4_EAS (Simo-Rifai enhanced strain for PET bending),
    Q4_UP (Q1P0 hybrid mean-dilatation for PSA near-incompressibility), or
    standard Q4 B-bar.  The `_assemble_multi_material_batch` dispatch routes
    each pid to the correct element kernel during assembly.

    *Assembly acceleration*: JAX vmap is used for Q4_EAS+J2Plasticity and
    Q4_UP+LinearViscoelastic pid-groups when all elements in the group have
    the same formuation.  If JAX returns NaN for any element (excessive
    deformation), a per-element sequential fallback is triggered.

    *Line search*: a simple NaN/Inf-guarding backtracking line search is used
    (no Armijo condition) because the KKT residual from LM constraints is
    not a valid merit function for the indefinite saddle-point system.

    *Time-stepping convention*: solver.time advances ON CONVERGENCE only
    (solver.u reflects the converged state at solver.time).  On cutback,
    the caller restores a saved state and retries with smaller dt.  This
    is the standard Abaqus-style controlled increment scheme.
    """
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
        atol: float = 1e-9,
        rtol: float = 1e-4,
        verbose: bool = False,
        max_step: Optional[float] = None,
        element_type: str = "Q4",
        fast_assembly: bool = True,
        section_thickness=1.0,
        static_mode: bool = False,
        alpha: float = 0.0,
        mode: Optional[str] = None,
    ):
        # Named integration preset overrides the individual β/γ/α/static_mode
        # arguments with a mutually-consistent set (see INTEGRATION_MODES).
        if mode is not None:
            if mode not in INTEGRATION_MODES:
                raise ValueError(
                    f"unknown integration mode {mode!r}; "
                    f"choose from {sorted(INTEGRATION_MODES)}"
                )
            preset = INTEGRATION_MODES[mode]
            beta = preset["beta"]
            gamma = preset["gamma"]
            alpha = preset["alpha"]
            static_mode = preset["static_mode"]
        self.mode = mode
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
        self.atol = atol   # absolute displacement-correction floor (P2)
        self.rtol = rtol   # relative residual-force convergence tolerance (P2)
        self.verbose = verbose
        self.max_step = max_step
        self.static_mode = static_mode
        self.alpha = alpha
        self.f_int_n = None  # for HHT-α alpha method
        
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

            # --- JAX EAS+J2 vmap batch setup (per-pid groups) ---
            # Pre-build jax.jit(jax.vmap(...)) for each Q4_EAS + J2Plasticity pid group
            self._eas_jax_vmap_by_pid: Dict[int, object] = {}
            try:
                import jax
                from ..element.q4_eas_jax import compute_eas_j2_contributions_jax
                from ..material.plastic_jax import pk2_voigt_jax as _pk2_jax
                for pid, mat_adapter in self.materials.items():
                    if (self._pid_element_type(pid) == "Q4_EAS"
                            and isinstance(mat_adapter.material, J2Plasticity)):
                        mat = mat_adapter.material
                        lam = float(mat.lam)
                        mu = float(mat.mu)
                        sigma_y0 = float(mat.sigma_y0)
                        H = float(mat.H)
                        _single = lambda coords, u_e, a, s: compute_eas_j2_contributions_jax(
                            coords, u_e, a, s, lam, mu, sigma_y0, H, 1.0,
                        )
                        self._eas_jax_vmap_by_pid[pid] = jax.jit(jax.vmap(_single))
            except ImportError:
                pass

            # --- JAX finite-strain visco vmap batch setup (per-pid groups) ---
            # Pre-build jax.jit(jax.vmap(...)) for each Q4_UP + LinearViscoelastic pid group.
            # Uses q4_visco_hybrid_fs_jax (Green-Lagrange + F-bar + autodiff tangent).
            self._visco_fs_jax_vmap_by_pid: Dict[int, object] = {}
            try:
                import jax
                import jax.numpy as jnp
                from ..element.q4_visco_hybrid_fs_jax import (
                    compute_single as _visco_fs_single,
                )
                from ..material.linear_viscoelastic import LinearViscoelastic
                for pid, mat_adapter in self.materials.items():
                    if (self._pid_element_type(pid) == "Q4_UP"
                            and isinstance(mat_adapter.material, LinearViscoelastic)):
                        visco_mat = mat_adapter.material
                        K_bulk = float(visco_mat.K)
                        G0 = float(visco_mat.G0)
                        g_i = jnp.asarray(visco_mat.g_i)
                        tau_i = jnp.asarray(visco_mat.tau_i)
                        # thickness=1.0 inside element; scaled externally during assembly
                        _single_visco = lambda coords, u_e, s, dt: _visco_fs_single(
                            coords, u_e, s, K_bulk, g_i, tau_i, G0, dt, 1.0,
                        )
                        self._visco_fs_jax_vmap_by_pid[pid] = jax.jit(jax.vmap(
                            _single_visco,
                            in_axes=(0, 0, 0, None),
                        ))
            except ImportError:
                pass

        # --- EAS internal parameters (4 per element); only Q4_EAS elements use
        #     them, others stay zero. Persisted/rolled back like material state.
        self.eas_alpha = np.zeros((self.n_elem, 4), dtype=np.float64)

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
        """Full KKT residual at trial state (u_k, u_ext_k, lam_k).

        Used by the line search in solve_step to evaluate whether a trial
        Newton step produces a finite residual.  Computes internal forces
        f_int via _assemble, then assembles the full residual vector
        [R_u, R_ext, R_lam] including Lagrange multiplier forces.

        For HHT-α (self.alpha != 0), the internal force is weighted:
            R_u = f_ext − M·a_k − (1+α)·f_int(u_k) + α·f_int(u_n)

        BC residuals are set as: R[idx] = bc_val − u_k[idx] (penalty
        substitution, not LM — bc DOFs are row-eliminated from K_eff).

        Returns
        -------
        R_total : (n_dofs + n_extra + n_lambdas,) ndarray
            Full residual vector.  Norm used by the line search to check
            NaN/Inf.  Not used for Armijo (KKT residual is not a valid
            merit function for saddle-point systems).
        """
        f_int, _, _ = self._assemble(u_k, dt)
        
        a_k = (u_k - u_n - dt * v_n) * inv_beta_dt2 - (1.0 - 2.0 * beta) / (2.0 * beta) * a_n
        a_ext_k = (u_ext_k - u_ext_n - dt * v_ext_n) * inv_beta_dt2 - (1.0 - 2.0 * beta) / (2.0 * beta) * a_ext_n

        if self.static_mode:
            R_u = self.f_ext - f_int
        elif self.alpha != 0.0 and self.f_int_n is not None:
            R_u = self.f_ext - self.M * a_k - (1.0 + self.alpha) * f_int + self.alpha * self.f_int_n
        else:
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
        """Solve one time increment via Newton-Raphson.

        This is the core non-linear solver loop.  It takes the mechanical state
        at the beginning of the increment (self.u, self.v, etc.) and computes
        the state at self.time + dt.

        Algorithm
        ---------
        1. Predictor: u_k = u_n + Δt·v_n + ½·Δt²·a_n  (Newmark predictor)
        2. Evaluate time-dependent BCs at the target time (Abaqus-style).
        3. Iterate (Newton-Raphson):
           a. Assemble f_int(u_k) and K_T(u_k) via _assemble.
           b. Compute residual R = f_ext − f_int − M·a_k  (+ LM forces).
           c. Build effective stiffness K_eff = (1+α)·K_T + (1/β/Δt²)·M.
           d. Construct KKT saddle-point system including C_lam^T/0 blocks.
           e. Apply BCs via row elimination (penalty substitution).
           f. Solve K_eff·du = R via PARDISO with iterative refinement.
           g. Backtracking line search: if trial step produces NaN/Inf in
              residual, halve α until finite residual found.
           h. Update: u_k += α·du,  lam_k += α·dlam.
           i. Check convergence: rel_change = ‖du‖/(‖u_k‖+ε) < tol.
        4. On convergence: advance self.time, update state variables
           (u, v, a, state, lam).  On failure (max iter or NaN/Inf):
           return negative iteration count for caller cutback.

        Convergence criterion
        ---------------------
        The solver uses a displacement-based convergence check:
            ‖Δu‖ / (‖u_k‖ + ε) < tol   (default: tol=1e-3)
        OR the energy error:  |R·du| < 1e-15  (tight tolerance suitable
        for poorly-scaled KKT saddle-point systems).

        The displacement ratio is reliable because:
        - u_k is well-scaled (mm displacement on mm-sized panel)
        - It does NOT depend on the residual norm, which can behave
          erratically on the indefinite KKT system (R contains both
          force equilibrium and LM constraint equations).

        For the saddle-point KKT matrix [K  C^T; C  0], the residual
        norm ‖R‖ is NOT a valid merit function: a step that reduces
        ‖R‖ can actually be worse for equilibrium (Greenstadt 1967,
        Dennis & Schnabel 1996 §6.4).  We therefore use ‖Δu‖/‖u‖.

        Line search (NaN-only backtracking)
        ------------------------------------
        Only guards against divergence (element inversion → NaN in
        residual).  Does NOT enforce Armijo/Wolfe conditions because:
        - The KKT residual is not monotonic (the LM blocks break
          the quadratic convergence guarantee of Newton for SPD systems)
        - A damped Newton step (α<1) on a saddle-point system can
          cause the KKT residual to INCREASE while the mechanical
          residual decreases — an Armijo check would falsely reject it.
        This is consistent with the strategy used in commercial FE codes
        for contact problems (Abaqus uses the same NaN-only line search
        for its default Newton solver).

        Returns
        -------
        n_iter : int
            Positive = converged in n_iter+1 iterations.
            Negative = failed: -(n_iter+1), where n_iter is the last
            successful iteration before failure.
            Negative = -max_iter indicates divergence (norm blow-up).
        """
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

        # HHT-α: compute f_int_n on first call (initial converged internal force)
        if self.alpha != 0.0 and self.f_int_n is None:
            self.f_int_n, _, _ = self._assemble(u_n, dt)

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

            if self.static_mode:
                R_u = self.f_ext - f_int
            elif self.alpha != 0.0 and self.f_int_n is not None:
                R_u = self.f_ext - self.M * a_k - (1.0 + self.alpha) * f_int + self.alpha * self.f_int_n
            else:
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
            if self.static_mode:
                K_eff = K_T.copy()  # no inertial contribution
            elif self.alpha != 0.0:
                K_eff = (1.0 + self.alpha) * K_T
                K_eff.setdiag(K_eff.diagonal() + inv_beta_dt2 * self.M)
            else:
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
            # Extra-primal (rotation θ) diagonal block. The CONSISTENT tangent
            # requires the constraint geometric stiffness K_θθ = Σ λ·∂²g/∂θ²
            # (the rigid rotation is nonlinear in θ). Without it the global
            # tangent is inconsistent once θ≠0 and Newton stalls. A tiny 1e-12
            # is kept on top to stabilize the otherwise-zero diagonal at θ=0.
            k_extra_geo = np.zeros(self.n_extra, dtype=np.float64)
            lam_off = 0
            for c in self.constraints:
                n_lam_c = c.n_multipliers()
                if n_lam_c > 0 and hasattr(c, "extra_geometric_stiffness"):
                    off, kval = c.extra_geometric_stiffness(
                        u_ext_k, lam_k[lam_off:lam_off + n_lam_c]
                    )
                    k_extra_geo[off] += kval
                lam_off += n_lam_c
            for i in range(self.n_extra):
                global_row.append(ext_start_idx + i)
                global_col.append(ext_start_idx + i)
                global_val.append(k_extra_geo[i] + 1e-12)
                
            # Lambda stabilization (optional, avoid saddle point issues, usually spsolve handles it)
            # but sometimes zeros on diagonal cause UMFPACK to complain.
            # We will rely on spsolve unless it fails.
            
            J = sps.coo_matrix((global_val, (global_row, global_col)), shape=(self.n_total, self.n_total)).tocsr()

            # --- Eigenvalue regularization (Tikhonov / ridge) ---
            # Adds ε·I to shift near-zero eigenvalues of the tangent stiffness
            # toward positive, improving conditioning without altering the
            # Newton direction for well-conditioned modes.  ε is chosen small
            # enough to preserve accuracy (~machine-ε × mean|diag(K)|).
            _diag_mean = np.abs(J.diagonal()[:self.n_dofs]).mean()
            _eps_reg = max(1e-12, 1e-8 * _diag_mean)
            J = J + _eps_reg * sps.eye(self.n_total, format='csr')

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

            # --- Backtracking Line Search (residual-aware + under-relaxation) ---
            # Full Newton steps (α=1.0) are always accepted first, preserving
            # quadratic convergence in the attraction basin.  If the full step
            # causes the residual to EXPLODE (>10× R_current), the step is
            # likely pointing into a bifurcation branch that diverges — the
            # tangent stiffness has lost positive-definiteness.  Subsequent
            # damped steps (α=0.5, 0.25, …) let the solver creep across the
            # unstable zone without blowing up.  Once past it, α=1.0 resumes.
            #
            # NOTE: a previous unconditional under-relaxation cap (alpha_max=0.7)
            # forced EVERY accepted step — including the full Newton step — down
            # to 0.7, which destroys quadratic convergence: with a consistent
            # tangent the residual then contracts only ~linearly and trivial
            # near-zero increments stall to max_iter (observed as constant du and
            # LS.alpha pinned at 0.70). Bifurcation regularization is already
            # provided by the LM diagonal shift, so the cap is both redundant and
            # harmful. The full Newton step (alpha=1.0) is now allowed; the loop
            # below only backtracks to escape NaN/Inf (geometric blow-up).
            alpha = 1.0
            alpha_min = 0.1
            alpha_max = 1.0        # no under-relaxation cap (full Newton allowed)
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

                if np.isnan(R_norm_temp) or np.isinf(R_norm_temp):
                    alpha *= 0.5
                    continue

                # Accept this α if:
                #   (a) it is the full Newton step (subject to alpha_max cap), OR
                #   (b) residual has dropped / not exploded, OR
                #   (c) we are already deep in backtracking
                if alpha == 1.0 or R_norm_temp < R_norm_current * 10.0 or alpha < 0.25:
                    break
                alpha *= 0.5

            # Enforce under-relaxation cap
            alpha = min(alpha, alpha_max)

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
            # ---- Residual-based convergence criterion ----
            # For KKT saddle-point systems the displacement ratio du_norm / u_norm
            # can plateau at ~0.006 even when the Newton direction is accurate
            # (the KKT system is solved to machine precision but the saddle-point
            # conditioning makes |du| ~ |u|·√(κ) unavoidable — see `_solve_linear_system`).
            # If the KKT residual has dropped by 10 orders relative to the first
            # Newton iteration, the direction is good enough to accept.
            # Relative residual-force convergence (P2): the KKT residual norm
            # has dropped by ≥ -log10(rtol) orders from the first Newton
            # iteration, i.e. the configuration is in equilibrium to rtol. This
            # is the standard force criterion (Abaqus default ~5e-3); the prior
            # 1e-9 threshold was unreachable on the equilibrated KKT (residual
            # floors at the linear-solver precision ~1e-6 absolute) and never
            # fired, so steps with a near-zero displacement ratio stalled to
            # max_iter even when already converged.
            residual_converged = (
                res_norm_0 is not None and res_ratio < self.rtol
            )
            # Absolute convergence (P2): when the Newton correction itself is
            # negligibly small, the increment has converged regardless of the
            # *relative* disp ratio. This cures the near-zero-deformation stall
            # of a flat (C2) amplitude start and saddle-point noise, where
            # du_norm/u_norm is dominated by numerical noise (0/0) and never
            # reaches `tol` even though |du| ~ 1e-11.
            abs_converged = du_norm < self.atol
            if (rel_change < self.tol or energy_err < 1e-15
                    or residual_converged or abs_converged) and n_iter > 0:
                converged = True
                self.time += dt          # advance analysis time on success only
                if state_new is not None:
                    self.state = state_new
                if self.alpha != 0.0:
                    self.f_int_n = f_int.copy()  # store for HHT-α next step
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

    # ------------------------------------------------------------------
    # Riks Arc-Length Continuation
    # ------------------------------------------------------------------
    def solve_step_riks(
        self,
        dt: float,
        ds: float,
        psi: float = 1.0,
        max_corr: int = 10,
        tol: float = 1e-6,
    ) -> int:
        """Solve one time increment via Crisfield modified Riks arc-length.

        The standard Newton-Raphson (``solve_step``) fails when the tangent
        stiffness loses positive-definiteness near limit points (snap-through
        or snap-back).  The Riks method overcomes this by treating the load
        factor λ as an additional unknown and adding an arc-length constraint
        that ties the displacement increment Δu and load increment Δλ:

            ‖Δu‖² + ψ² · Δλ² · ‖f_ext‖² = Δs²

        The augmented KKT system has size (n_total + 1):

            ┌ K_eff   C_u^T   −f_ext ┐ ┌ Δu   ┐   ┌ R_u   ┐
            │ C_u     0        0      │ │ Δlam  │ = │ R_lam │
            └ f_ext^T 0        0      ┘ └ Δλ   ┘   └   0   ┘

        The last row is the linearised arc-length constraint (Crisfield 1981).
        ψ = 1.0 gives a spherical constraint; ψ = 0.0 gives pure load control.

        Parameters
        ----------
        dt : float
            Time increment.
        ds : float
            Prescribed arc-length in generalised displacement space.
        psi : float
            Scaling factor for the load-parameter contribution.
        max_corr : int
            Maximum number of corrector iterations.
        tol : float
            Relative tolerance on the arc-length constraint.

        Returns
        -------
        n_iter : int
            Positive = converged in n_iter corrections.
            Negative = failed: -(n_corr+1).
        """
        t_start = time.time()
        beta, gamma, dt2 = self.beta, self.gamma, dt * dt
        inv_beta_dt2 = 1.0 / (beta * dt2)

        # --- save converged state (for rollback on failure) ---
        u_n = self.u.copy()
        v_n = self.v.copy()
        a_n = self.a.copy()
        u_ext_n = self.u_extra.copy()
        v_ext_n = self.v_extra.copy()
        a_ext_n = self.a_extra.copy()
        lam_k = self.lam.copy()
        bc_vals_orig = self.bc_vals.copy() if self.bc_vals is not None else None

        # Newmark predictor
        u_k = u_n + dt * v_n + 0.5 * dt2 * a_n
        u_ext_k = u_ext_n + dt * v_ext_n + 0.5 * dt2 * a_ext_n

        # Current load factor (1.0 = full gravity)
        lambda_k = 1.0

        norm_f_ext = np.linalg.norm(self.f_ext)

        if getattr(self, 'verbose', False):
            print(f"", flush=True)
            print(f"  Riks Arc-Length Step: ds={ds:.4e}  psi={psi:.2f}", flush=True)
            print(f"  {'='*120}", flush=True)

        # =====================================================================
        # Predictor: tangent direction from K_eff · du_t = f_ext
        # =====================================================================
        f_int, K_T, state_new = self._assemble(u_k, dt)

        if self.static_mode:
            K_eff = K_T.copy()
        elif self.alpha != 0.0:
            K_eff = (1.0 + self.alpha) * K_T
            K_eff.setdiag(K_eff.diagonal() + inv_beta_dt2 * self.M)
        else:
            K_eff = K_T.copy()
            K_eff.setdiag(K_eff.diagonal() + inv_beta_dt2 * self.M)

        # Eigenvalue regularisation (consistent with solve_step)
        _diag_mean = np.abs(K_eff.diagonal()[: self.n_dofs]).mean()
        _eps_reg = max(1e-12, 1e-8 * _diag_mean)
        K_eff = K_eff + _eps_reg * sps.eye(K_eff.shape[0], format='csr')

        du_t = _solve_linear_system(K_eff, self.f_ext)

        if np.any(np.isnan(du_t)) or np.any(np.isinf(du_t)):
            if getattr(self, 'verbose', False):
                print(f"  *** Riks predictor: NaN/Inf in tangent direction.  "
                      f"Falling back to solve_step.\n", flush=True)
            # restore state & fall back
            self.u = u_n; self.v = v_n; self.a = a_n
            self.u_extra = u_ext_n; self.v_extra = v_ext_n; self.a_extra = a_ext_n
            self.lam = lam_k
            if bc_vals_orig is not None:
                self.bc_vals = bc_vals_orig
            return self.solve_step(dt)

        norm_du_t = np.linalg.norm(du_t)
        denom = np.sqrt(norm_du_t**2 + psi**2 * norm_f_ext**2)
        if denom < 1e-30:
            if getattr(self, 'verbose', False):
                print(f"  *** Riks: zero tangent norm.  Falling back to solve_step.\n", flush=True)
            self.u = u_n; self.v = v_n; self.a = a_n
            self.u_extra = u_ext_n; self.v_extra = v_ext_n; self.a_extra = a_ext_n
            self.lam = lam_k
            if bc_vals_orig is not None:
                self.bc_vals = bc_vals_orig
            return self.solve_step(dt)

        scale = ds / denom
        du_pred = scale * du_t
        dlambda_pred = scale

        if getattr(self, 'verbose', False):
            print(f"  Predictor: ||du_t||={norm_du_t:.4e}  "
                  f"scale={scale:.4e}  dlam={dlambda_pred:.4e}", flush=True)

        # =====================================================================
        # Corrector iterations
        # =====================================================================
        u_trial = u_k + du_pred
        u_ext_trial = u_ext_k.copy()
        lambda_trial = lambda_k + dlambda_pred

        converged = False
        n_corr = 0

        for n_corr in range(max_corr):
            # Scale prescribed BCs by the load factor
            if bc_vals_orig is not None:
                self.bc_vals = bc_vals_orig * lambda_trial

            # --- Assemble at trial state ---
            f_int_trial, K_T_trial, state_new_trial = self._assemble(u_trial, dt)

            # Effective stiffness
            if self.static_mode:
                K_eff_t = K_T_trial.copy()
            elif self.alpha != 0.0:
                K_eff_t = (1.0 + self.alpha) * K_T_trial
                K_eff_t.setdiag(K_eff_t.diagonal() + inv_beta_dt2 * self.M)
            else:
                K_eff_t = K_T_trial.copy()
                K_eff_t.setdiag(K_eff_t.diagonal() + inv_beta_dt2 * self.M)

            # Regularisation
            _dm = np.abs(K_eff_t.diagonal()[: self.n_dofs]).mean()
            _ep = max(1e-12, 1e-8 * _dm)
            K_eff_t = K_eff_t + _ep * sps.eye(self.n_total, format='csr')

            # --- Mechanical residual ---
            a_trial = ((u_trial - u_n - dt * v_n) * inv_beta_dt2
                       - (1.0 - 2.0 * beta) / (2.0 * beta) * a_n)
            a_ext_trial = ((u_ext_trial - u_ext_n - dt * v_ext_n) * inv_beta_dt2
                           - (1.0 - 2.0 * beta) / (2.0 * beta) * a_ext_n)

            if self.static_mode:
                R_u = lambda_trial * self.f_ext - f_int_trial
            elif self.alpha != 0.0 and self.f_int_n is not None:
                R_u = (lambda_trial * self.f_ext - self.M * a_trial
                       - (1.0 + self.alpha) * f_int_trial
                       + self.alpha * self.f_int_n)
            else:
                R_u = lambda_trial * self.f_ext - self.M * a_trial - f_int_trial

            # --- Constraint residuals + Jacobian ---
            R_ext = np.zeros(self.n_extra)
            R_lam = np.zeros(self.n_lambdas)
            C_row_u, C_col_u, C_val_u = [], [], []
            C_row_ext, C_col_ext, C_val_ext = [], [], []

            lam_offset = 0
            for c in self.constraints:
                n_lam_c = c.n_multipliers()
                if n_lam_c > 0:
                    r_u, c_u, v_u, r_ext, c_ext, v_ext, g = c.assemble(
                        u_trial, u_ext_trial
                    )
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
                    for i in range(n_lam_c):
                        R_lam[lam_offset + i] = -g[i]
                lam_offset += n_lam_c

            R_total = np.concatenate([R_u, R_ext, R_lam])

            # --- Build KKT matrix (same sparsity as solve_step) ---
            K_eff_coo = K_eff_t.tocoo()
            g_row = list(K_eff_coo.row)
            g_col = list(K_eff_coo.col)
            g_val = list(K_eff_coo.data)

            lam_start = self.n_dofs + self.n_extra
            for r, c, v in zip(C_row_u, C_col_u, C_val_u):
                g_row.append(r + lam_start); g_col.append(c);         g_val.append(v)
                g_row.append(c);             g_col.append(r + lam_start); g_val.append(v)
            ext_start = self.n_dofs
            for r, c, v in zip(C_row_ext, C_col_ext, C_val_ext):
                g_row.append(r + lam_start); g_col.append(c + ext_start); g_val.append(v)
                g_row.append(c + ext_start); g_col.append(r + lam_start); g_val.append(v)
            for i in range(self.n_extra):
                g_row.append(ext_start + i); g_col.append(ext_start + i); g_val.append(1e-12)

            KKT = sps.coo_matrix(
                (g_val, (g_row, g_col)), shape=(self.n_total, self.n_total)
            ).tocsr()

            # --- Augment with load-factor DOF (λ) ---
            n_aug = self.n_total + 1
            a_row = list(g_row)
            a_col = list(g_col)
            a_val = list(g_val)

            # Column n_total:  −f_ext  (coupling: ∂R/∂λ = −f_ext)
            f_ext_len = min(len(self.f_ext), self.n_dofs)
            for i in range(f_ext_len):
                if self.f_ext[i] != 0.0:
                    a_row.append(i)
                    a_col.append(self.n_total)
                    a_val.append(-self.f_ext[i])

            # Row n_total:  f_ext^T  (linearised arc-length constraint)
            for i in range(f_ext_len):
                if self.f_ext[i] != 0.0:
                    a_row.append(self.n_total)
                    a_col.append(i)
                    a_val.append(self.f_ext[i])

            # Diagonal stabilisation for λ DOF
            a_row.append(self.n_total)
            a_col.append(self.n_total)
            a_val.append(0.0)

            KKT_aug = sps.coo_matrix(
                (a_val, (a_row, a_col)), shape=(n_aug, n_aug)
            ).tocsr()

            # Eigenvalue regularisation on augmented system
            _dm_a = np.abs(KKT_aug.diagonal()[: self.n_dofs]).mean()
            _ep_a = max(1e-12, 1e-8 * _dm_a)
            KKT_aug = KKT_aug + _ep_a * sps.eye(n_aug, format='csr')

            # Augmented residual
            R_aug = np.zeros(n_aug)
            R_aug[: self.n_total] = R_total
            R_aug[self.n_total] = 0.0  # constraint residual (linearised)

            # BC elimination on augmented system
            has_bc = len(self.bc_dofs) > 0
            if has_bc:
                for i, idx in enumerate(self.bc_dofs):
                    val = self.bc_vals[i]
                    sp = KKT_aug.indptr[idx]
                    ep = KKT_aug.indptr[idx + 1]
                    for ptr in range(sp, ep):
                        col = KKT_aug.indices[ptr]
                        KKT_aug.data[ptr] = 1.0 if col == idx else 0.0
                    if idx < self.n_dofs:
                        R_aug[idx] = val * lambda_trial - u_trial[idx]
                    else:
                        ext_idx = idx - self.n_dofs
                        R_aug[idx] = val * lambda_trial - u_ext_trial[ext_idx]

            # --- Solve augmented system ---
            delta_all = _solve_linear_system(KKT_aug, R_aug)

            if np.any(np.isnan(delta_all)) or np.any(np.isinf(delta_all)):
                if getattr(self, 'verbose', False):
                    print(f"  *** Corrector {n_corr+1}: NaN/Inf in augmented solve.\n",
                          flush=True)
                break

            delta_u     = delta_all[: self.n_dofs]
            delta_u_ext = delta_all[self.n_dofs : self.n_dofs + self.n_extra]
            delta_lam   = delta_all[self.n_dofs + self.n_extra : self.n_total]
            delta_lambda = delta_all[self.n_total]  # load-factor increment

            # --- Update trial state ---
            u_trial     += delta_u
            u_ext_trial += delta_u_ext
            lambda_trial += delta_lambda

            # --- Arc-length constraint check ---
            du_norm   = np.linalg.norm(u_trial - u_k)
            dlambda   = lambda_trial - lambda_k
            constraint = du_norm**2 + psi**2 * dlambda**2 * norm_f_ext**2
            c_err = abs(constraint - ds**2) / (ds**2 + 1e-30)

            res_norm = np.linalg.norm(R_aug)

            if getattr(self, 'verbose', False):
                print(
                    f"  Corr {n_corr+1:2d}: ||du||={du_norm:11.4e}  "
                    f"dlam={dlambda:+11.4e}  lam={lambda_trial:11.6f}  "
                    f"c_err={c_err:9.2e}  ||R||={res_norm:11.4e}",
                    flush=True,
                )

            # Convergence: both residual small AND arc-length constraint satisfied
            if c_err < tol and res_norm < tol * 100:
                converged = True
                break

        # =====================================================================
        # Finalise
        # =====================================================================
        # Restore original BCs
        if bc_vals_orig is not None:
            self.bc_vals = bc_vals_orig

        if converged:
            # Advance time and store state
            self.u = u_trial.copy()
            self.u_extra = u_ext_trial.copy()

            a_k = ((u_trial - u_n - dt * v_n) * inv_beta_dt2
                   - (1.0 - 2.0 * beta) / (2.0 * beta) * a_n)
            a_ext_k = ((u_ext_trial - u_ext_n - dt * v_ext_n) * inv_beta_dt2
                       - (1.0 - 2.0 * beta) / (2.0 * beta) * a_ext_n)

            self.a = np.clip(a_k, -1.0e4, 1.0e4)
            self.v = np.clip(v_n + dt * ((1.0 - gamma) * a_n + gamma * a_k),
                             -1.0e3, 1.0e3)
            self.a_extra = np.clip(a_ext_k, -1.0e4, 1.0e4)
            self.v_extra = np.clip(
                v_ext_n + dt * ((1.0 - gamma) * a_ext_n + gamma * a_ext_k),
                -1.0e3, 1.0e3,
            )
            self.lam = lam_k.copy()
            self.time += dt

            if state_new_trial is not None:
                self.state = state_new_trial

            if getattr(self, 'verbose', False):
                te = time.time() - t_start
                print(
                    f"  => Riks Step Converged ({n_corr+1} corrections).  "
                    f"lambda={lambda_trial:.6f}  (Time: {te:.2f} s)\n",
                    flush=True,
                )
            return n_corr + 1
        else:
            # Rollback to last converged state
            self.u = u_n.copy();  self.v = v_n.copy();  self.a = a_n.copy()
            self.u_extra = u_ext_n.copy()
            self.v_extra = v_ext_n.copy()
            self.a_extra = a_ext_n.copy()
            self.lam = lam_k.copy()

            if getattr(self, 'verbose', False):
                te = time.time() - t_start
                print(
                    f"  *** Riks step FAILED after {n_corr+1} corrections.  "
                    f"(Time: {te:.2f} s)\n",
                    flush=True,
                )
            return -(n_corr + 1)

    def solve(self, num_steps: int, dt: float, callback: Optional[Callable] = None) -> List[np.ndarray]:
        """Run the full time-stepping loop.

        Parameters
        ----------
        num_steps : int
            Number of constant-dt increments.
        dt : float
            Time increment per step.
        callback : callable or None
            Optional function(step, solver) called after each increment.
            Used for live visualisation or intermediate output.

        Returns
        -------
        history : list of ndarray
            One displacement vector per step (for post-processing).
            Each entry has shape (n_dofs,) with the converged u.

        Notes
        -----
        If a Newton step fails (n_iter < 0), a warning is emitted but the
        solver continues with the unconverged state.  For production use,
        override the callback (or subclass) to handle cutback:

            def my_callback(step, solver):
                if last_n_iter < 0:
                    dt_cut = dt * 0.5
                    solver.restore_state(checkpoint)
                    solver.solve_step(dt_cut)
        """
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
            'eas_alpha': self.eas_alpha.copy(),
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
        if 'eas_alpha' in state_dict:
            self.eas_alpha = state_dict['eas_alpha'].copy()
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
                import jax
                import jax.numpy as jnp
                n_vars = mat_adapter.n_internal_vars
                dt_h = dt if dt is not None else 1.0

                if pid in self._visco_fs_jax_vmap_by_pid:
                    _vmap_fn = self._visco_fs_jax_vmap_by_pid[pid]
                    Ng = len(elem_indices)
                    coords_b = jnp.asarray(self.elem_coords[elem_indices])
                    u_b = jnp.asarray(u[self.dof_indices[elem_indices]])
                    state_b = jnp.asarray(
                        self.state[elem_indices, :, :n_vars]
                    ) if self.state is not None else jnp.zeros((Ng, 4, n_vars))
                    t_b = jnp.asarray(self._elem_thickness[elem_indices])

                    f_es, K_es, se_all = _vmap_fn(coords_b, u_b, state_b, dt_h)
                    f_es_np = np.asarray(f_es)
                    K_es_np = np.asarray(K_es)

                    f_es_scaled = f_es_np * np.asarray(t_b)[:, None]
                    K_es_scaled = K_es_np * np.asarray(t_b)[:, None, None]

                    np.add.at(f_int, self.dof_indices[elem_indices].flatten(), f_es_scaled.flatten())

                    if state_new is not None:
                        state_new[elem_indices, :, :n_vars] = np.asarray(se_all)

                    all_K_rows.append(self._pid_K_rows[pid])
                    all_K_cols.append(self._pid_K_cols[pid])
                    all_K_vals.append(K_es_scaled.reshape(-1))
                else:
                    from ..element.q4_visco_hybrid_fs_jax import (
                        compute_single as _visco_fs,
                    )
                    visco_mat = mat_adapter.material
                    coords_b = jnp.asarray(self.elem_coords[elem_indices])
                    u_b = jnp.asarray(u[self.dof_indices[elem_indices]])
                    state_b = jnp.asarray(
                        self.state[elem_indices, :, :n_vars]
                    ) if self.state is not None else jnp.zeros((len(elem_indices), 4, n_vars))
                    t_b = jnp.asarray(self._elem_thickness[elem_indices])

                    _vmap_visco = jax.vmap(
                        _visco_fs,
                        in_axes=(0, 0, 0, None, None, None, None, None, 0),
                    )
                    f_es, K_es, se_all = _vmap_visco(
                        coords_b, u_b, state_b,
                        float(visco_mat.K), jnp.asarray(visco_mat.g_i),
                        jnp.asarray(visco_mat.tau_i), float(visco_mat.G0), dt_h,
                        t_b,
                    )
                    f_es = np.asarray(f_es)
                    K_es = np.asarray(K_es)
                    se_all = np.asarray(se_all)

                    np.add.at(f_int, self.dof_indices[elem_indices].flatten(), f_es.flatten())

                    if state_new is not None:
                        state_new[elem_indices, :, :n_vars] = se_all

                    all_K_rows.append(self._pid_K_rows[pid])
                    all_K_cols.append(self._pid_K_cols[pid])
                    all_K_vals.append(K_es.reshape(-1))
                continue

            if self._pid_element_type(pid) == "Q4_VISCO_SIMO":
                # ---- Complete finite-strain Simo viscoelasticity (Flory split,
                #      pluggable hyperelastic base) — q4_visco_simo_fs_jax. ----
                import jax
                import jax.numpy as jnp
                from functools import partial as _partial
                from ..material.viscoelastic import ViscoelasticMaterial as _VEM
                from ..element.q4_visco_simo_fs_jax import compute_single as _visco_simo
                vmat = mat_adapter.material
                if not isinstance(vmat, _VEM):
                    raise TypeError("Q4_VISCO_SIMO requires a ViscoelasticMaterial")
                n_vars = mat_adapter.n_internal_vars   # = 6*(M+1)
                dt_h = dt if dt is not None else 1.0
                base, bparams, kappa = vmat.simo_fs_args(mat_adapter.params)

                coords_b = jnp.asarray(self.elem_coords[elem_indices])
                u_b = jnp.asarray(u[self.dof_indices[elem_indices]])
                state_b = jnp.asarray(
                    self.state[elem_indices, :, :n_vars]
                ) if self.state is not None else jnp.zeros((len(elem_indices), 4, n_vars))
                t_b = jnp.asarray(self._elem_thickness[elem_indices])

                _fn = _partial(_visco_simo, base=base)   # bind static base out of vmap
                _vmap = jax.vmap(_fn, in_axes=(0, 0, 0, None, None, None, None, None, None, 0))
                f_es, K_es, se_all = _vmap(
                    coords_b, u_b, state_b, float(kappa), jnp.asarray(bparams),
                    jnp.asarray(vmat.g_i), jnp.asarray(vmat.tau_i), float(vmat.g_inf),
                    dt_h, t_b,
                )
                f_es = np.asarray(f_es); K_es = np.asarray(K_es); se_all = np.asarray(se_all)
                np.add.at(f_int, self.dof_indices[elem_indices].flatten(), f_es.flatten())
                if state_new is not None:
                    state_new[elem_indices, :, :n_vars] = se_all
                all_K_rows.append(self._pid_K_rows[pid])
                all_K_cols.append(self._pid_K_cols[pid])
                all_K_vals.append(K_es.reshape(-1))
                continue

            if (self._pid_element_type(pid) == "Q4_EAS"
                    and isinstance(mat_adapter.material, J2Plasticity)):
                n_vars = mat_adapter.n_internal_vars
                if pid in self._eas_jax_vmap_by_pid:
                    import jax
                    import jax.numpy as jnp
                    _vmap_fn = self._eas_jax_vmap_by_pid[pid]
                    Ng = len(elem_indices)
                    coords_b = jnp.asarray(self.elem_coords[elem_indices])
                    u_b = jnp.asarray(u[self.dof_indices[elem_indices]])
                    alpha_b = jnp.asarray(self.eas_alpha[elem_indices])
                    state_b = jnp.asarray(
                        self.state[elem_indices, :, :n_vars]
                    ) if self.state is not None else jnp.zeros((Ng, 4, n_vars))
                    t_b = jnp.asarray(self._elem_thickness[elem_indices])

                    f_es, K_es, alpha_all, se_all = _vmap_fn(coords_b, u_b, alpha_b, state_b)
                    f_es_np = np.asarray(f_es)
                    K_es_np = np.asarray(K_es)

                    nan_mask = ~np.all(np.isfinite(f_es_np), axis=(1,)) | ~np.all(np.isfinite(K_es_np), axis=(1, 2))
                    if np.any(nan_mask):
                        nan_idx = np.where(nan_mask)[0]
                        print(f"DEBUG: EAS JAX vmap NaN in {len(nan_idx)}/{Ng} elems (pid={pid}), falling back to sequential")
                        good_mask = ~nan_mask
                        if np.any(good_mask):
                            good_f = f_es_np[good_mask] * np.asarray(t_b[good_mask])[:, None]
                            good_K = K_es_np[good_mask] * np.asarray(t_b[good_mask])[:, None, None]
                            good_dofs = self.dof_indices[elem_indices[good_mask]]
                            np.add.at(f_int, good_dofs.flatten(), good_f.flatten())
                            all_K_rows.append(good_dofs.reshape(-1))
                            all_K_cols.append(np.repeat(good_dofs, 8, axis=1).reshape(-1))
                            all_K_vals.append(good_K.reshape(-1))
                        from ..element.q4_eas import compute_eas_j2_contributions
                        for idx in nan_idx:
                            e = elem_indices[idx]
                            coords = self.elem_coords[e]
                            u_elem = u[self.dof_indices[e]]
                            state_elem = (self.state[e, :, :n_vars]
                                          if self.state is not None else None)
                            f_e, K_e, alpha_new, se_new = compute_eas_j2_contributions(
                                coords, u_elem, self.eas_alpha[e], state_elem,
                                mat_adapter.material, mat_adapter.params,
                                self._elem_thickness[e],
                            )
                            self.eas_alpha[e] = alpha_new
                            np.add.at(f_int, self.dof_indices[e], f_e)
                            if state_new is not None and se_new is not None:
                                state_new[e, :, :n_vars] = se_new
                            dof_g = self.dof_indices[e]
                            all_K_rows.append(np.repeat(dof_g, 8))
                            all_K_cols.append(np.tile(dof_g, 8))
                            all_K_vals.append(K_e.flatten())
                    else:
                        f_es_scaled = f_es_np * np.asarray(t_b)[:, None]
                        K_es_scaled = K_es_np * np.asarray(t_b)[:, None, None]
                        self.eas_alpha[elem_indices] = np.asarray(alpha_all)
                        np.add.at(f_int, self.dof_indices[elem_indices].flatten(), f_es_scaled.flatten())
                        if state_new is not None:
                            state_new[elem_indices, :, :n_vars] = np.asarray(se_all)
                        all_K_rows.append(self._pid_K_rows[pid])
                        all_K_cols.append(self._pid_K_cols[pid])
                        all_K_vals.append(K_es_scaled.reshape(-1))
                else:
                    from ..element.q4_eas import compute_eas_j2_contributions
                    for e in elem_indices:
                        coords = self.elem_coords[e]
                        u_elem = u[self.dof_indices[e]]
                        state_elem = (self.state[e, :, :n_vars]
                                      if self.state is not None else None)
                        f_e, K_e, alpha_new, se_new = compute_eas_j2_contributions(
                            coords, u_elem, self.eas_alpha[e], state_elem,
                            mat_adapter.material, mat_adapter.params,
                            self._elem_thickness[e],
                        )
                        self.eas_alpha[e] = alpha_new
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
        """Precompute B-bar, integration weights, and dN/dX for all elements × GPs.

        Uses JAX jit + vmap for vectorised element-loop over all integration points.
        """
        import jax
        import jax.numpy as jnp

        n_gp = len(_GP2)
        GP2 = jnp.asarray(_GP2, dtype=jnp.float64)
        W2 = jnp.asarray(_W2, dtype=jnp.float64)
        elem_coords_j = jnp.asarray(self.elem_coords, dtype=jnp.float64)
        elem_thickness_j = jnp.asarray(self._elem_thickness, dtype=jnp.float64)

        @jax.jit
        def _precompute_single_element(coords, thickness):
            def _sd(xi, eta):
                dN_dxi = 0.25 * jnp.array([
                    -(1.0 - eta),  (1.0 - eta),
                     (1.0 + eta), -(1.0 + eta)])
                dN_deta = 0.25 * jnp.array([
                    -(1.0 - xi), -(1.0 + xi),
                     (1.0 + xi),  (1.0 - xi)])
                return dN_dxi, dN_deta

            def _jac(xi, eta):
                dN_dxi, dN_deta = _sd(xi, eta)
                J = jnp.array([
                    [jnp.dot(dN_dxi,  coords[:, 0]),
                     jnp.dot(dN_dxi,  coords[:, 1])],
                    [jnp.dot(dN_deta, coords[:, 0]),
                     jnp.dot(dN_deta, coords[:, 1])],
                ])
                return J, jnp.linalg.det(J), jnp.linalg.inv(J)

            def _B_mat(xi, eta, invJ):
                dN_dxi, dN_deta = _sd(xi, eta)
                dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
                dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
                B = jnp.zeros((3, 8))
                for i in range(4):
                    col = 2 * i
                    B = B.at[0, col].set(dN_dx[i])
                    B = B.at[1, col + 1].set(dN_dy[i])
                    B = B.at[2, col].set(dN_dy[i])
                    B = B.at[2, col + 1].set(dN_dx[i])
                return B

            P_vol = 0.5 * jnp.array([
                [1, 1, 0],
                [1, 1, 0],
                [0, 0, 0],
            ], dtype=jnp.float64)

            _, _, invJ0 = _jac(0.0, 0.0)
            B0_vol = P_vol @ _B_mat(0.0, 0.0, invJ0)

            B_bar_gp = jnp.empty((n_gp, 3, 8), dtype=jnp.float64)
            weights_gp = jnp.empty((n_gp,), dtype=jnp.float64)
            dN_dX_gp = jnp.empty((n_gp, 2, 4), dtype=jnp.float64)

            for gp in range(n_gp):
                xi, eta = GP2[gp]
                _, detJ, invJ = _jac(xi, eta)
                B_std = _B_mat(xi, eta, invJ)
                B_bar_gp = B_bar_gp.at[gp].set(B_std - P_vol @ B_std + B0_vol)
                weights_gp = weights_gp.at[gp].set(detJ * W2[gp] * thickness)
                dN_dxi, dN_deta = _sd(xi, eta)
                dN_dX_gp = dN_dX_gp.at[gp, 0].set(invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta)
                dN_dX_gp = dN_dX_gp.at[gp, 1].set(invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta)

            return B_bar_gp, weights_gp, dN_dX_gp

        _vmap_precompute = jax.jit(jax.vmap(_precompute_single_element))
        B_bar_all, weights_all, dN_dX_all = _vmap_precompute(elem_coords_j, elem_thickness_j)

        return (
            np.asarray(B_bar_all),
            np.asarray(weights_all),
            np.asarray(dN_dX_all),
        )

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
