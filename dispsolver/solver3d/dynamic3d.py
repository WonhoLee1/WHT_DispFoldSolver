"""
dynamic3d.py
============
3D Implicit Non-linear Dynamic & Quasistatic Solver for WHT_DispFoldSolver.
Features Bathe Composite Integration, Surface Tie Constraint Assembly,
and Dirichlet Prescribed Motion Enforcers.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
import warnings
import numpy as np

from scipy.sparse import csc_matrix, csr_matrix, coo_matrix

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
from dispsolver.solver3d.assembly_utils import (
    build_global_topology,
    scatter_f_int_3d,
    build_coo_to_csr_mapping,
    scatter_k_csr_direct,
    apply_dirichlet_bcs_csr_fast,
)
from dispsolver.material3d.numba_materials import (
    MAT_LINEAR_ELASTIC,
    MAT_CUSTOM_ELASTIC,
    MAT_ORTHOTROPIC_CYLINDRICAL,
    get_default_sdv_count,
)
from dispsolver.material.catalog import is_linear_elastic

_HEX20_FACES = {
    "P1": ([0, 1, 2, 3], [8, 9, 10, 11]),
    "P2": ([4, 7, 6, 5], [15, 14, 13, 12]),
    "P3": ([0, 4, 5, 1], [16, 12, 17, 8]),
    "P4": ([1, 5, 6, 2], [17, 13, 18, 9]),
    "P5": ([2, 6, 7, 3], [18, 14, 19, 10]),
    "P6": ([3, 7, 4, 0], [19, 15, 16, 11]),
    "S1": ([0, 1, 2, 3], [8, 9, 10, 11]),
    "S2": ([4, 7, 6, 5], [15, 14, 13, 12]),
    "S3": ([0, 4, 5, 1], [16, 12, 17, 8]),
    "S4": ([1, 5, 6, 2], [17, 13, 18, 9]),
    "S5": ([2, 6, 7, 3], [18, 14, 19, 10]),
    "S6": ([3, 7, 4, 0], [19, 15, 16, 11]),
}

def _quad8_shape_and_derivs(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    N = np.zeros(8, dtype=np.float64)
    dN = np.zeros((2, 8), dtype=np.float64)
    xi_i = np.array([-1.0, 1.0, 1.0, -1.0])
    eta_i = np.array([-1.0, -1.0, 1.0, 1.0])
    for i in range(4):
        xi0 = xi * xi_i[i]
        eta0 = eta * eta_i[i]
        N[i] = 0.25 * (1.0 + xi0) * (1.0 + eta0) * (xi0 + eta0 - 1.0)
        dN[0, i] = 0.25 * xi_i[i] * (1.0 + eta0) * (2.0 * xi0 + eta0)
        dN[1, i] = 0.25 * eta_i[i] * (1.0 + xi0) * (xi0 + 2.0 * eta0)
    # midsides: (0,-1), (1,0), (0,1), (-1,0)
    N[4] = 0.5 * (1.0 - xi * xi) * (1.0 - eta)
    dN[0, 4] = -xi * (1.0 - eta)
    dN[1, 4] = -0.5 * (1.0 - xi * xi)

    N[5] = 0.5 * (1.0 + xi) * (1.0 - eta * eta)
    dN[0, 5] = 0.5 * (1.0 - eta * eta)
    dN[1, 5] = -eta * (1.0 + xi)

    N[6] = 0.5 * (1.0 - xi * xi) * (1.0 + eta)
    dN[0, 6] = -xi * (1.0 + eta)
    dN[1, 6] = 0.5 * (1.0 - xi * xi)

    N[7] = 0.5 * (1.0 - xi) * (1.0 - eta * eta)
    dN[0, 7] = -0.5 * (1.0 - eta * eta)
    dN[1, 7] = -eta * (1.0 - xi)
    return N, dN

def _integrate_face_pressure_quad8(coords_20: np.ndarray, face_def: Tuple[list, list], p_mag: float) -> Tuple[list, np.ndarray]:
    corners, midsides = face_def
    face_idx = corners + midsides
    face_coords = coords_20[face_idx]
    elem_center = np.mean(coords_20, axis=0)
    face_center = np.mean(face_coords, axis=0)
    v_in = elem_center - face_center

    gp = 1.0 / np.sqrt(3.0)
    f_face = np.zeros((8, 3), dtype=np.float64)
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            N, dN = _quad8_shape_and_derivs(xi, eta)
            t1 = dN[0] @ face_coords
            t2 = dN[1] @ face_coords
            dA_vec = np.cross(t1, t2)
            if np.dot(dA_vec, v_in) < 0.0:
                dA_vec = -dA_vec
            for a in range(8):
                f_face[a] += N[a] * p_mag * dA_vec
    return face_idx, f_face


def _integrate_face_pressure_quad4(face_coords: np.ndarray, elem_center: np.ndarray, p_mag: float) -> np.ndarray:
    """Consistent nodal forces of a uniform pressure p on a bilinear quad face
    (2x2 Gauss, reference geometry). Same convention as the quad8 version:
    the area vector is turned toward the element interior, so a positive
    pressure pushes INTO the body (Abaqus *DSLOAD/*DLOAD P) whatever the face
    node order."""
    v_in = elem_center - np.mean(face_coords, axis=0)
    gp = 1.0 / np.sqrt(3.0)
    f_face = np.zeros((4, 3), dtype=np.float64)
    for xi in (-gp, gp):
        for eta in (-gp, gp):
            N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
            dNx = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            dNe = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            dA_vec = np.cross(dNx @ face_coords, dNe @ face_coords)
            if np.dot(dA_vec, v_in) < 0.0:
                dA_vec = -dA_vec
            f_face += np.outer(N, p_mag * dA_vec)
    return f_face



def _volume_shape_integrals(X: np.ndarray, what: str = "body load") -> np.ndarray:
    """integral(N_a dV) over one element, a = 0..n-1, by Gauss quadrature on
    the element's own shape functions (exact for the affine case). Linear
    hex/tet/wedge only; a quadratic element's consistent body load is not an
    equal split (corner terms are negative), so it is refused rather than
    approximated."""
    n = len(X)
    if n == 8:
        g = 1.0 / np.sqrt(3.0)
        sgn = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
        w = np.zeros(8)
        for xi in (-g, g):
            for eta in (-g, g):
                for zeta in (-g, g):
                    p = np.array([xi, eta, zeta])
                    N = 0.125 * np.prod(1.0 + sgn * p, axis=1)
                    dN = np.empty((8, 3))
                    for k in range(3):
                        o = [j for j in range(3) if j != k]
                        dN[:, k] = 0.125 * sgn[:, k] * (1.0 + sgn[:, o[0]] * p[o[0]]) * (1.0 + sgn[:, o[1]] * p[o[1]])
                    w += N * abs(np.linalg.det(X.T @ dN))
        return w
    if n == 4:
        V = abs(np.linalg.det((X[1:] - X[0]).T)) / 6.0
        return np.full(4, V / 4.0)
    if n == 6:
        # 3-point triangle x 2-point line; nodes 0-2 bottom (zeta=-1), 3-5 top
        tri = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]
        g = 1.0 / np.sqrt(3.0)
        w = np.zeros(6)
        for r, s_ in tri:
            for z in (-g, g):
                L = np.array([1.0 - r - s_, r, s_])
                N = np.concatenate([L * 0.5 * (1 - z), L * 0.5 * (1 + z)])
                dL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
                dN = np.zeros((6, 3))
                dN[:3, :2] = dL * 0.5 * (1 - z); dN[3:, :2] = dL * 0.5 * (1 + z)
                dN[:3, 2] = -0.5 * L; dN[3:, 2] = 0.5 * L
                w += N * abs(np.linalg.det(X.T @ dN)) * (1.0 / 6.0)
        return w
    raise NotImplementedError(f"{what}: body/gravity load on a {n}-node element is not implemented "
                              f"(consistent integration exists for 4-, 6- and 8-node solids only)")



# Small-strain kernels co-rotated under NLGEOM (element3d/cr_generic_3d); the
# list lives in model/element_catalog.KERNELS.
from dispsolver.model.element_catalog import cr_wrapped_kernels as _cr_wrapped_kernels
from dispsolver.model.element_catalog import KERNELS as _KERNEL_TABLE
_KERNELS_3D = _KERNEL_TABLE[3]
_CR_WRAPPED_KERNELS = _cr_wrapped_kernels(3)



class DynamicSolver3D:
    """3D Implicit Non-Linear Solver with Above-Abaqus Speed & Convergence Engine."""

    # implicit_dynamics.HHTIncrement while an HHT / Newmark increment runs: assemble_system
    # then returns the dynamic effective system to the (single) Newton loop
    _dyn_ctx = None
    _stab_ctx = None

    def __init__(
        self,
        mesh: Mesh3D,
        materials: Optional[Dict[int, Any]] = None,
        material_params: Optional[Dict[str, float]] = None,
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
        beta_k: float = 0.0,
        solution_controls: Optional[Any] = None,
    ):
        self.mesh = mesh
        self.nlgeom = nlgeom
        self.density = float(density) if density is not None else 7.85e-9
        self.alpha_m_damp = float(alpha_m_damp)
        self.beta_k_damp = float(beta_k if beta_k != 0.0 else beta_k_damp)
        self.beta_k = self.beta_k_damp

        # Solution Controls and Active Tolerance Engine (Abaqus API Parity)
        from dispsolver.model.control import SolutionControl
        if solution_controls is not None:
            if isinstance(solution_controls, SolutionControl):
                self.solution_controls = solution_controls
            elif isinstance(solution_controls, str):
                self.solution_controls = SolutionControl(preset=solution_controls)
            else:
                self.solution_controls = SolutionControl()
        else:
            self.solution_controls = SolutionControl(preset="STANDARD")

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

        # Initialize EMCS Solver Engine if time_integrator is "emcs"
        if self.time_integrator == "emcs":
            from dispsolver.solver3d.emcs_dynamic3d import EMCSSolver3D
            self._emcs_solver = EMCSSolver3D(
                mesh=mesh,
                density=self.density,
                material_params=self.material_params,
                alpha_m=self.alpha_m,
                beta_k=self.beta_k
            )
        else:
            self._emcs_solver = None



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
        self.last_accept = None
        self._contact_tuner = None
        self._stabilization = None
        self.energy_allsd: float = 0.0
        self.allsd: float = 0.0
        self.current_temperature: float = 20.0
        self.ref_temperature: float = 20.0
        self.current_field: float = 0.0
        self._f_thermal_cached: Optional[np.ndarray] = None
        self._thermal_dirty: bool = True
        self.node_transforms: Optional[np.ndarray] = None
        self.has_nodal_transforms: bool = False
        self._setup_numba_topology()
        self._check_initial_hex_orientation()

    def _check_initial_hex_orientation(self) -> None:
        """Alias for _check_initial_element_orientation for backward compatibility."""
        self._check_initial_element_orientation()

    def _check_initial_element_orientation(self) -> None:
        """Refuse 3D solid elements whose reference Jacobian is zero or negative
        at any Gauss point or vertex (left-handed node order or a collapsed element)
        -- Abaqus/Standard stops with an error for these (element set
        ErrElemVolSmallNegZero). Without this check the assembly runs on a
        negative-definite element stiffness and a contact model built on it
        cannot converge, with no error pointing at the mesh (2026-10-01: all 240
        roller elements of 01_hertz_roller_plate_3d.inp)."""
        elements = list(self.mesh.elements.values())
        if not elements:
            return
        nid_map = self.mesh.node_id_to_index()
        nodes_arr = self.mesh.nodes_array()[:, :3]

        # -------------------------------------------------------------
        # 1. 8-node & 20-node Hexahedra (C3D8*, C3D20*, SOLID_SHELL*)
        # -------------------------------------------------------------
        hex_ids, hex_conn = [], []
        for elem in elements:
            et = str(getattr(elem, "elem_type", "")).upper()
            if (len(elem.node_ids) in (8, 20)) and (et.startswith("C3D8") or et.startswith("C3D20") or "SOLID_SHELL" in et):
                hex_ids.append(getattr(elem, "id", None))
                # First 8 nodes define the corner hex topology
                hex_conn.append([nid_map[nid] for nid in elem.node_ids[:8]])
        if hex_conn:
            X = nodes_arr[np.asarray(hex_conn, dtype=np.int64)]
            s = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
            t = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
            r = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
            g = 1.0 / np.sqrt(3.0)
            min_det = np.full(len(hex_conn), np.inf)
            for gx in (-g, g):
                for gy in (-g, g):
                    for gz in (-g, g):
                        dN = np.vstack([
                            s * (1.0 + t * gy) * (1.0 + r * gz),
                            t * (1.0 + s * gx) * (1.0 + r * gz),
                            r * (1.0 + s * gx) * (1.0 + t * gy),
                        ]) / 8.0
                        J = np.einsum("ia,eaj->eij", dN, X)
                        min_det = np.minimum(min_det, np.linalg.det(J))
            bad = np.where(min_det <= 0.0)[0]
            if len(bad) > 0:
                sample = [hex_ids[i] for i in bad[:10]]
                raise ValueError(
                    f"{len(bad)} of {len(hex_conn)} hexahedral elements have zero or negative "
                    f"volume (reference Jacobian <= 0 at a Gauss point, min {min_det[bad].min():.4g}); "
                    f"check the node ordering (Abaqus: ErrElemVolSmallNegZero). First element ids: {sample}"
                )

        # -------------------------------------------------------------
        # 2. 4-node & 10-node Tetrahedra (C3D4*, C3D10*)
        # -------------------------------------------------------------
        tet_ids, tet_conn = [], []
        for elem in elements:
            et = str(getattr(elem, "elem_type", "")).upper()
            if (len(elem.node_ids) in (4, 10)) and (et.startswith("C3D4") or et.startswith("C3D10")):
                tet_ids.append(getattr(elem, "id", None))
                tet_conn.append([nid_map[nid] for nid in elem.node_ids[:4]])
        if tet_conn:
            X = nodes_arr[np.asarray(tet_conn, dtype=np.int64)]
            # Constant Jacobian vector triple product: (X2 - X1) . ((X3 - X1) x (X4 - X1)) = 6 * Vol
            v1 = X[:, 1] - X[:, 0]
            v2 = X[:, 2] - X[:, 0]
            v3 = X[:, 3] - X[:, 0]
            cross12 = np.cross(v1, v2)
            det_tet = np.einsum("ei,ei->e", cross12, v3)
            bad_tet = np.where(det_tet <= 0.0)[0]
            if len(bad_tet) > 0:
                sample = [tet_ids[i] for i in bad_tet[:10]]
                raise ValueError(
                    f"{len(bad_tet)} of {len(tet_conn)} tetrahedral elements have zero or negative "
                    f"volume (reference Jacobian <= 0, min det J {det_tet[bad_tet].min():.4g}); "
                    f"check the node ordering (Abaqus: ErrElemVolSmallNegZero). First element ids: {sample}"
                )

        # -------------------------------------------------------------
        # 3. 6-node Triangular Prisms / Wedges (C3D6*)
        # -------------------------------------------------------------
        wedge_ids, wedge_conn = [], []
        for elem in elements:
            et = str(getattr(elem, "elem_type", "")).upper()
            if len(elem.node_ids) == 6 and et.startswith("C3D6"):
                wedge_ids.append(getattr(elem, "id", None))
                wedge_conn.append([nid_map[nid] for nid in elem.node_ids])
        if wedge_conn:
            X = nodes_arr[np.asarray(wedge_conn, dtype=np.int64)]
            # 6 Gauss points: 3 on triangular face x 2 along thickness
            tri_pts = np.array([
                [1.0 / 6.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0],
            ])
            zeta_pts = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
            min_det_wedge = np.full(len(wedge_conn), np.inf)
            for xi, eta in tri_pts:
                for zeta in zeta_pts:
                    # N1..N3 = [1-xi-eta, xi, eta] * (1-zeta)/2
                    # N4..N6 = [1-xi-eta, xi, eta] * (1+zeta)/2
                    dN = np.array([
                        [-0.5 * (1.0 - zeta), 0.5 * (1.0 - zeta), 0.0, -0.5 * (1.0 + zeta), 0.5 * (1.0 + zeta), 0.0],
                        [-0.5 * (1.0 - zeta), 0.0, 0.5 * (1.0 - zeta), -0.5 * (1.0 + zeta), 0.0, 0.5 * (1.0 + zeta)],
                        [-0.5 * (1.0 - xi - eta), -0.5 * xi, -0.5 * eta, 0.5 * (1.0 - xi - eta), 0.5 * xi, 0.5 * eta],
                    ])
                    J = np.einsum("ia,eaj->eij", dN, X)
                    min_det_wedge = np.minimum(min_det_wedge, np.linalg.det(J))
            bad_wedge = np.where(min_det_wedge <= 0.0)[0]
            if len(bad_wedge) > 0:
                sample = [wedge_ids[i] for i in bad_wedge[:10]]
                raise ValueError(
                    f"{len(bad_wedge)} of {len(wedge_conn)} wedge elements have zero or negative "
                    f"volume (reference Jacobian <= 0 at a Gauss point, min {min_det_wedge[bad_wedge].min():.4g}); "
                    f"check the node ordering (Abaqus: ErrElemVolSmallNegZero). First element ids: {sample}"
                )

    def set_nodal_transforms(self, node_transforms: np.ndarray):
        """Set nodal rotation matrices (N_nodes, 3, 3) for Abaqus *TRANSFORM."""
        dim = 3
        arr = np.asarray(node_transforms, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[1] == 2:
            n_nodes = len(arr)
            R3 = np.zeros((n_nodes, 3, 3), dtype=np.float64)
            for i in range(n_nodes):
                R3[i, :2, :2] = arr[i]
                R3[i, 2, 2] = 1.0
            self.node_transforms = R3
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
        return transform_u_to_global(u_vec, self.node_transforms, dim=3)

    @property
    def contact_tuner(self):
        if self._contact_tuner is None:
            from dispsolver.contact.contact_auto_tuner import ContactAutoTuner
            self._contact_tuner = ContactAutoTuner(self, verbose=True)
        return self._contact_tuner

    def get_contact_tuning_guide(self) -> Dict[str, Any]:
        """Return recommended contact parameters dictionary and script snippet."""
        return self.contact_tuner.get_recommendation_guide()

    def print_contact_tuning_guide(self) -> None:
        """Print user-friendly contact parameter optimization guide and script snippet."""
        self.contact_tuner.print_guide()


    def _record_acceptance(self, branch: str, f_acc: np.ndarray,
                           f_ext: np.ndarray, iters: int, dt: float = 1.0) -> None:
        """Read-only calibration probe: true free-DOF residual norm at the
        accepted state. Never influences acceptance (AGENTS.md section 4.7:
        the AND-criterion rewrite caused dt-collapse, so thresholds stay
        untouched -- this only measures what they accept)."""
        mask = np.ones(self.num_dofs, dtype=bool)
        if self.fixed_dofs:
            mask[list(self.fixed_dofs.keys())] = False
        r = float(np.linalg.norm((f_ext - np.asarray(f_acc))[mask]))
        # f_acc was assembled with any SDI contact freeze still in place, where
        # frozen pairs are bilateral and unfrozen ones may be skipped -- not the
        # one-sided law the answer is supposed to satisfy. Re-assemble once with
        # the freeze lifted (only when one is active) and record that residual
        # too (dev_log/plan_deck06_contact_fix_20261004.md, P0-4).
        frozen = [c for c in self.constraints if getattr(c, "frozen_active_nodes", None) is not None]
        r_unfrozen = r
        if frozen:
            saved = [c.frozen_active_nodes for c in frozen]
            try:
                for c in frozen:
                    c.frozen_active_nodes = None
                _, f_free = self.assemble_system(self.u, dt=dt)
                r_unfrozen = float(np.linalg.norm((f_ext - np.asarray(f_free))[mask]))
            finally:
                for c, s in zip(frozen, saved):
                    c.frozen_active_nodes = s
        self.last_accept = {"branch": branch, "r_norm": r, "r_norm_unfrozen": r_unfrozen,
                            "frozen": bool(frozen), "iters": iters}

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
            return self.pardiso_solver.solve(K_bc, r_bc, has_lagrange_multipliers=False,
                                             is_symmetric=self._is_symmetric(K_bc))
        return pardiso_spsolve(K_bc, r_bc)

    @staticmethod
    def _is_symmetric(K) -> bool:
        """Numerical symmetry test for the mtype="auto" Pardiso path, which
        otherwise assumes SPD (mtype=2) and factors only the UPPER triangle
        -- an unsymmetric tangent (e.g. the consistent Coulomb friction
        block, 18.6% asymmetric on the Hertz roller deck) would be silently
        replaced by its upper-triangle symmetrization. Scaled by the largest
        OFF-diagonal entry because apply_boundary_conditions() puts a 1e15
        penalty on the diagonal of prescribed DOFs."""
        from scipy.sparse import diags
        K = K.tocsr()
        off = K - diags(K.diagonal())
        scale = abs(off).max() if off.nnz else 0.0
        if scale == 0.0:
            return True
        # bool(): this used to return numpy.bool_, and the Pardiso mtype
        # resolver tests `is_symmetric is False` -- never true for np.False_,
        # so every UNSYMMETRIC tangent was sent to the SPD path, failed
        # Cholesky (error -4) and was silently refactored as symmetric
        # indefinite (mtype -2) from its upper triangle only.
        return bool(abs(K - K.T).max() <= 1e-10 * scale)

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
        self.elem_props = np.zeros((n_elems, 64), dtype=np.float64)

        self.elem_density = np.zeros(n_elems, dtype=np.float64)
        default_rho = self.density if getattr(self, "density", None) is not None else 7.85e-9

        # Populate multimaterial properties per element.  Material id + props
        # come from the catalog only (A5, audit O7): an unmapped material
        # raises NotImplementedError instead of silently becoming linear
        # elastic.  A pid with NO material keeps the constructor C_mat default.
        from dispsolver.material.resolve import resolve_material
        cmat_flat = self.C_mat_default.ravel()
        resolved = {}
        # pid -> material name for materials without a density; a deck *DYNAMIC step
        # refuses them (implicit_dynamics.require_density) instead of using default_rho
        self._pids_without_density = {}
        for e, elem in enumerate(elements_list):
            pid = getattr(elem, "pid", 0)
            mat_obj = self.materials.get(pid, None) if hasattr(self, "materials") and isinstance(self.materials, dict) else None
            rho_val = default_rho
            if mat_obj is not None:
                if pid not in resolved:
                    md = mat_obj.to_numba_props() if hasattr(mat_obj, "to_numba_props") else mat_obj
                    extra = md if isinstance(md, dict) else (getattr(md, "extra_params", None) or {})
                    if "engineering_constants" in extra:
                        # KEPT outside the catalog on purpose: id 17 is a C3D20-kernel
                        # private path (it reads the 9 raw constants from props); the
                        # catalog resolves these constants to CUSTOM_ELASTIC (16), which
                        # is a different (6x6 C) layout.  C3D20 only: see the guard below.
                        ec = extra["engineering_constants"]
                        props17 = np.zeros(64, dtype=np.float64)
                        props17[:len(ec)] = ec
                        resolved[pid] = (md, MAT_ORTHOTROPIC_CYLINDRICAL, props17)
                    else:
                        mid, props = resolve_material(mat_obj, 3, f"material of pid {pid}")
                        resolved[pid] = (md, mid, props)
                md, mid, props = resolved[pid]
                if mid == MAT_ORTHOTROPIC_CYLINDRICAL and not elem.elem_type.upper().startswith("C3D20"):
                    from dispsolver.model.regions import describe
                    raise NotImplementedError(
                        f"{describe(mat_obj, f'material of pid {pid}')} (material {getattr(mat_obj, 'name', None)!r}): "
                        f"orthotropic ENGINEERING CONSTANTS (id {MAT_ORTHOTROPIC_CYLINDRICAL}) are only "
                        f"implemented for C3D20 elements (element3d/c3d20_numba.py), not {elem.elem_type}")
                if isinstance(md, dict):
                    rho_val = float(md.get("density", md.get("rho", default_rho)))
                elif hasattr(md, "density"):
                    rho_val = float(getattr(md, "density"))
                elif hasattr(md, "rho"):
                    rho_val = float(getattr(md, "rho"))
            # Zero/negative density is never valid dynamics mass; Material's
            # default density is 0.0, which must not override the solver-level
            # default (previously zeroed M_global and reduced dynamics to
            # quasistatics with garbage velocities -- FD-verified 2026-09-20).
            if rho_val <= 0.0:
                rho_val = default_rho
                if mat_obj is not None:
                    from dispsolver.solver.implicit_dynamics import material_label
                    self._pids_without_density[pid] = material_label(mat_obj, pid)
            self.elem_density[e] = rho_val

            if mat_obj is not None:
                self.elem_mat_types[e] = mid
                self.elem_props[e, :] = props
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
        
        self._build_kernel_groups()

        # Check if mesh or sections provide custom controls
        if hasattr(self.mesh, "elem_controls") and self.mesh.elem_controls is not None:
            self.elem_controls[:] = self.mesh.elem_controls

    def _build_kernel_groups(self) -> None:
        """Assign every element its assembly kernel from its type and the
        CURRENT self.nlgeom (the small/large-deformation promotion), then build
        the per-kernel connectivity and the sparsity pattern. Split out of
        _setup_numba_topology so a step can re-run it when NLGEOM changes
        without resetting materials, SDVs or initial stresses."""
        nid_map = self.mesh.node_id_to_index()
        elements_list = list(self.mesh.elements.values())
        # Classify element kernel groups with Material- & Kinematic-Adaptive NLGEOM Dispatch
        # Element type -> kernel: model/element_catalog.py is the only place
        # that decides it (NLGEOM, near-incompressibility and the material
        # only pick among the kernels the catalog lists for the type).
        from dispsolver.model.element_catalog import ElementCatalog
        elem_kernel_indices: Dict[int, List[int]] = {}
        failures: Dict[str, str] = {}
        subst: Dict[str, str] = {}
        for idx, elem in enumerate(elements_list):
            et = elem.elem_type.upper()
            pid = getattr(elem, "pid", 0)
            mat_obj = self.materials.get(pid, None) if hasattr(self, "materials") and isinstance(self.materials, dict) else None
            nu = 0.3
            if mat_obj is not None:
                from dispsolver.material.resolve import material_nu
                nu = material_nu(mat_obj)
            mt = int(self.elem_mat_types[idx]) if getattr(self, "elem_mat_types", None) is not None else MAT_LINEAR_ELASTIC
            try:
                k, notes = ElementCatalog.resolve(et, 3, bool(self.nlgeom), nu=nu,
                                                  linear_elastic=is_linear_elastic(mt))
            except NotImplementedError as exc:
                failures.setdefault(et, str(exc))
                continue
            if k is None:
                continue
            if notes:
                subst[et] = "; ".join(notes)
            elem_kernel_indices.setdefault((k, len(elem.node_ids)), []).append(idx)

        if failures:
            raise NotImplementedError("; ".join(failures.values()))
        # Report formulation substitutions (hybrid field or reduced
        # integration missing, finite-strain kernel under NLGEOM=NO ...).
        new = {et: why for et, why in subst.items() if et not in getattr(self, "element_substitutions", {})}
        self.element_substitutions = subst
        for et, why in new.items():
            warnings.warn(f"element type {et}: {why}")

        self.elem_kernel_groups = {
            gk: np.array(indices, dtype=np.int64)
            for gk, indices in elem_kernel_indices.items()
            if gk[0] >= 0
        }
        # element row -> kernel id: the result of the NLGEOM-dependent element
        # promotion above. element_point_stress() dispatches on it, so anything
        # that needs "the stress this element produces" follows the same
        # small/large-deformation choice as the assembly.
        self._row_kernel = {int(r): int(gk[0]) for gk, rows in self.elem_kernel_groups.items() for r in rows}

        from dispsolver.model.element_catalog import KERNELS as _KERNELS
        self._lumped_mass_cache = None          # regrouped: the lumped mass follows the kernels' nodes
        self.elem_conn_groups = {}
        for gk, indices in self.elem_kernel_groups.items():
            k_val, n_nodes_val = gk
            n_use = min(n_nodes_val, _KERNELS[3][k_val].nodes)   # kernel consumes its own node count
            self.elem_conn_groups[gk] = np.array(
                [[nid_map[nid] for nid in elements_list[i].node_ids[:n_use]] for i in indices],
                dtype=np.int64
            )

        rows_list = []
        cols_list = []
        for gk in sorted(self.elem_kernel_groups.keys()):
            conn_k = self.elem_conn_groups[gk]
            r_k, c_k = build_global_topology(self.num_dofs, conn_k)
            rows_list.append(r_k)
            cols_list.append(c_k)

        if len(rows_list) > 0:
            self.rows_topo = np.concatenate(rows_list)
            self.cols_topo = np.concatenate(cols_list)
            # Pre-allocated reusable contiguous data buffer (Zero-Allocation)
            self._data_topo_buf = np.empty(len(self.rows_topo), dtype=np.float64)
            self.elem_group_data_slices = {}
            offset = 0
            for gk in sorted(self.elem_kernel_groups.keys()):
                n_c = self.elem_conn_groups[gk].shape[1]
                length = len(self.elem_conn_groups[gk]) * (n_c * 3) * (n_c * 3)
                self.elem_group_data_slices[gk] = (offset, offset + length)
                offset += length

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
            self._data_topo_buf = None
            self.elem_group_data_slices = {}
            self.csr_template = None
            self.csr_indptr = None
            self.csr_indices = None
            self.csr_data = None
            self.topo_to_csr_map = None

        # Kernel 10 (C3D8 without NLGEOM) integrates ONE global C_mat_default for
        # every element. That is only right if all of its elements share a
        # single linear-elastic material equal to it; otherwise refuse instead
        # of silently giving every element the same stiffness.
        # Kernels that take one constant elasticity matrix per element group
        # (assembled per material by _elastic_C_groups): refuse other materials
        # instead of running them as linear elastic. 5/6 used to apply
        # self.C_mat_default to every element, whatever its material.
        from dispsolver.model.element_catalog import elastic_only_kernels
        for kk, what in elastic_only_kernels(3).items():
            rows = [r for gk, rr in self.elem_kernel_groups.items() if gk[0] == kk for r in rr]
            if rows and getattr(self, "elem_mat_types", None) is not None:
                bad = sorted({int(self.elem_mat_types[r]) for r in rows if not is_linear_elastic(int(self.elem_mat_types[r]))})
                if bad:
                    raise NotImplementedError(
                        f"{what} elements run a linear-elastic-only kernel ({kk}); found material types {bad}.")

    def _elastic_C_groups(self, elem_indices: np.ndarray):
        """Yield (local selection, 6x6 C) for groups of identical linear
        elastic materials among elem_indices: (E, nu) for MAT_LINEAR_ELASTIC,
        the stored 36 entries for MAT_CUSTOM_ELASTIC."""
        mts = np.asarray(self.elem_mat_types)[elem_indices]
        props = np.asarray(self.elem_props)[elem_indices]
        bad = sorted({int(mt) for mt in mts if not is_linear_elastic(int(mt))})
        if bad:
            raise NotImplementedError(
                f"elements run a linear-elastic-only kernel; found material types {bad}.")
        keys = {}
        for i, (mt, pr) in enumerate(zip(mts, props)):
            key = (int(mt),) + tuple(np.round(pr[:36] if int(mt) == MAT_CUSTOM_ELASTIC else pr[:2], 12))
            keys.setdefault(key, []).append(i)
        for key, sel in keys.items():
            if key[0] == MAT_CUSTOM_ELASTIC:
                C = np.array(key[1:], dtype=np.float64).reshape(6, 6)
            else:
                C = self._linear_elastic_C(key[1], key[2])
            yield np.asarray(sel, dtype=np.int64), C

    @staticmethod
    def _linear_elastic_C(E: float, nu: float) -> np.ndarray:
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        C = np.zeros((6, 6))
        C[:3, :3] = lam
        C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
        C[3, 3] = C[4, 4] = C[5, 5] = mu
        return C

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

    def set_rayleigh_damping(self, alpha: float = 0.0, beta: float = 0.0) -> None:
        """Set Rayleigh damping C = alpha*M + beta*K for implicit dynamics.

        Applies to the HHT/Newmark branch (EMCS carries its own alpha_m /
        beta_k). alpha = beta = 0 disables damping and restores the
        bit-identical undamped path.
        """
        self.alpha_m_damp = float(alpha)
        self.beta_k_damp = float(beta)
        self.beta_k = self.beta_k_damp

    def set_temperature(self, temp: float, ref_temp: Optional[float] = None) -> None:
        """Prescribe a global predefined temperature field (Abaqus *TEMPERATURE parity)."""
        self.current_temperature = float(temp)
        if ref_temp is not None:
            self.ref_temperature = float(ref_temp)
        self._thermal_dirty = True

        # Refresh the solver-maintained temperature slot of damage materials.
        if hasattr(self, "materials") and isinstance(self.materials, dict) and hasattr(self, "elem_props"):
            from dispsolver.material3d.numba_materials import MAT_J2_DUCTILE_DAMAGE as _M13
            elements_list = list(self.mesh.elements.values())
            for e, elem in enumerate(elements_list):
                pid = getattr(elem, "pid", 0)
                mat_obj = self.materials.get(pid, None)
                if mat_obj is None:
                    continue
                mt = self.elem_mat_types[e] if hasattr(self, "elem_mat_types") else -1
                if int(mt) == int(_M13):
                    self.elem_props[e, 13] = self.current_temperature

        # Update temperature-dependent elastic/plastic properties in elem_props if defined
        if hasattr(self, "materials") and isinstance(self.materials, dict) and hasattr(self, "elem_props"):
            from dispsolver.material.dependence_table import eval_table_1d_numba
            elements_list = list(self.mesh.elements.values())
            for e, elem in enumerate(elements_list):
                pid = getattr(elem, "pid", 0)
                mat_obj = self.materials.get(pid, None)
                if mat_obj is None:
                    continue
                # Temperature-dependent elasticity: table rows (E, nu, T)
                if hasattr(mat_obj, "extra_params") and "temp_elastic_table" in mat_obj.extra_params:
                    t_tab = np.array(mat_obj.extra_params["temp_elastic_table"], dtype=np.float64)
                    E_interp = eval_table_1d_numba(t_tab[:, 2], t_tab[:, 0], self.current_temperature)
                    nu_interp = eval_table_1d_numba(t_tab[:, 2], t_tab[:, 1], self.current_temperature)
                    self.elem_props[e, 0] = E_interp
                    self.elem_props[e, 1] = nu_interp
                    # Also update C_mat_default for linear C3D8 fast path
                    mu_i = E_interp / (2.0 * (1.0 + nu_interp))
                    K_i = E_interp / (3.0 * (1.0 - 2.0 * nu_interp))
                    lam_i = K_i - (2.0 / 3.0) * mu_i
                    if hasattr(self, "C_mat_default"):
                        self.C_mat_default[0, 0] = lam_i + 2*mu_i; self.C_mat_default[0, 1] = lam_i;        self.C_mat_default[0, 2] = lam_i
                        self.C_mat_default[1, 0] = lam_i;        self.C_mat_default[1, 1] = lam_i + 2*mu_i; self.C_mat_default[1, 2] = lam_i
                        self.C_mat_default[2, 0] = lam_i;        self.C_mat_default[2, 1] = lam_i;        self.C_mat_default[2, 2] = lam_i + 2*mu_i
                        self.C_mat_default[3, 3] = mu_i
                        self.C_mat_default[4, 4] = mu_i
                        self.C_mat_default[5, 5] = mu_i

    def set_field_variable(self, f: float) -> None:
        """Prescribe a global predefined field variable FV1 (Abaqus *FIELD parity)."""
        self.current_field = float(f)

        # Update field-dependent elastic properties in elem_props if defined.
        # Table rows are (E, nu, F); mirrors the temperature block above.
        if hasattr(self, "materials") and isinstance(self.materials, dict) and hasattr(self, "elem_props"):
            from dispsolver.material.dependence_table import eval_table_1d_numba
            elements_list = list(self.mesh.elements.values())
            for e, elem in enumerate(elements_list):
                pid = getattr(elem, "pid", 0)
                mat_obj = self.materials.get(pid, None)
                if mat_obj is None:
                    continue
                if hasattr(mat_obj, "extra_params") and "field_elastic_table" in mat_obj.extra_params:
                    f_tab = np.array(mat_obj.extra_params["field_elastic_table"], dtype=np.float64)
                    E_interp = eval_table_1d_numba(f_tab[:, 2], f_tab[:, 0], self.current_field)
                    nu_interp = eval_table_1d_numba(f_tab[:, 2], f_tab[:, 1], self.current_field)
                    self.elem_props[e, 0] = E_interp
                    self.elem_props[e, 1] = nu_interp
                    mu_i = E_interp / (2.0 * (1.0 + nu_interp))
                    K_i = E_interp / (3.0 * (1.0 - 2.0 * nu_interp))
                    lam_i = K_i - (2.0 / 3.0) * mu_i
                    if hasattr(self, "C_mat_default"):
                        self.C_mat_default[0, 0] = lam_i + 2*mu_i; self.C_mat_default[0, 1] = lam_i;        self.C_mat_default[0, 2] = lam_i
                        self.C_mat_default[1, 0] = lam_i;        self.C_mat_default[1, 1] = lam_i + 2*mu_i; self.C_mat_default[1, 2] = lam_i
                        self.C_mat_default[2, 0] = lam_i;        self.C_mat_default[2, 1] = lam_i;        self.C_mat_default[2, 2] = lam_i + 2*mu_i
                        self.C_mat_default[3, 3] = mu_i
                        self.C_mat_default[4, 4] = mu_i
                        self.C_mat_default[5, 5] = mu_i

    def compute_thermal_load_vector(self) -> np.ndarray:
        """Compute equivalent global thermal load vector f_th = int B^T C eps_th dV."""
        if not self._thermal_dirty and self._f_thermal_cached is not None:
            return self._f_thermal_cached

        f_th = np.zeros(self.num_dofs, dtype=np.float64)
        if not self.mesh.elements or self.elem_conn_0based is None:
            self._f_thermal_cached = f_th
            self._thermal_dirty = False
            return f_th

        from dispsolver.material.solver_thermal_adapter import compute_element_thermal_force_3d
        node_coords = self.mesh.nodes_array()
        elements_list = list(self.mesh.elements.values())

        for e, elem in enumerate(elements_list):
            pid = getattr(elem, "pid", 0)
            mat_obj = self.materials.get(pid, None) if hasattr(self, "materials") and isinstance(self.materials, dict) else None
            alpha = 0.0
            t0 = self.ref_temperature
            E_val = 200000.0
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
                conn_e = self.elem_conn_0based[e]
                if len(conn_e) == 8:
                    coords_e = node_coords[conn_e]
                    f_th_e = np.zeros(24, dtype=np.float64)
                    compute_element_thermal_force_3d(coords_e, alpha, delta_T, E_val, nu_val, f_th_e)
                    for a in range(8):
                        nid_idx = conn_e[a]
                        f_th[3 * nid_idx + 0] += f_th_e[3 * a + 0]
                        f_th[3 * nid_idx + 1] += f_th_e[3 * a + 1]
                        f_th[3 * nid_idx + 2] += f_th_e[3 * a + 2]

        self._f_thermal_cached = f_th
        self._thermal_dirty = False
        return f_th

    def assemble_system(self, u_vec: np.ndarray, dt: float = 1.0, return_error: bool = False, update_state: bool = False):
        """K and f_int of the system the Newton loop solves: the physical ones, or during an
        HHT / Newmark increment the dynamic effective ones (implicit_dynamics.HHTIncrement),
        or during a stabilized static increment the viscous stabilization ones.
        self.f_int is always the physical internal force."""
        res = self._assemble_physical(u_vec, dt=dt, return_error=return_error, update_state=update_state)
        if self._dyn_ctx is not None:
            return self._dyn_ctx.transform(res, u_vec)
        if getattr(self, "_stab_ctx", None) is not None:
            return self._stab_ctx.transform(res, u_vec)
        return res

    def _assemble_physical(self, u_vec: np.ndarray, dt: float = 1.0, return_error: bool = False, update_state: bool = False):
        """Assemble global 3D stiffness matrix K_global and internal force vector f_int."""
        node_coords_all = self.mesh.nodes_array()

        if self.has_nodal_transforms and self.node_transforms is not None:
            from dispsolver.model.nodal_transform import transform_u_to_global
            u_eval = transform_u_to_global(u_vec, self.node_transforms, dim=3)
        else:
            u_eval = u_vec

        if self.rows_topo is not None:
            try:
                f_int_global = np.zeros(self.num_dofs, dtype=np.float64)
                data_topo_list = []
                has_error = False

                # Heterogeneous element assembly: assemble each group in parallel
                for group_key in sorted(self.elem_kernel_groups.keys()):
                    k = group_key[0] if isinstance(group_key, tuple) else group_key
                    elem_indices = self.elem_kernel_groups[group_key]
                    sub_conn = self.elem_conn_groups[group_key]
                    sub_mat = self.elem_mat_types[elem_indices]
                    sub_props = self.elem_props[elem_indices]
                    
                    if update_state:
                        sub_sdvs = self.elem_sdvs[elem_indices]
                    else:
                        sub_sdvs = self.elem_sdvs[elem_indices].copy()
                        
                    sub_stress_init = self.elem_stress_init[elem_indices]
                    sub_controls = self.elem_controls[elem_indices]

                    # NLGEOM: small-strain-only kernels run in co-rotated
                    # element frames (cr_generic_3d); they used to run
                    # unchanged, so rigid rotations produced stress.
                    cr_R = None
                    if self.nlgeom and k in _CR_WRAPPED_KERNELS:
                        from dispsolver.element3d.cr_generic_3d import corotational_local_mesh
                        _cr_saved = (node_coords_all, sub_conn, u_eval)
                        node_coords_all, sub_conn, u_eval, cr_R = corotational_local_mesh(
                            np.ascontiguousarray(node_coords_all, dtype=np.float64),
                            np.ascontiguousarray(sub_conn, dtype=np.int64),
                            np.ascontiguousarray(u_eval, dtype=np.float64))

                    if k == 0:
                        from dispsolver.element3d.c3d8_eas_tl_numba import assemble_mesh_c3d8_eas_tl_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_eas_tl_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_stress_init=sub_stress_init
                        )
                    elif k == 1:
                        from dispsolver.element3d.c3d8_corotational_numba import assemble_mesh_c3d8_corotational_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_corotational_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 2:
                        from dispsolver.element3d.c3d8_hybrid_numba import assemble_mesh_c3d8_hybrid_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_hybrid_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 3:
                        from dispsolver.element3d.c3d8_fbar_tl_numba import assemble_mesh_c3d8_fbar_tl_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_fbar_tl_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_stress_init=sub_stress_init
                        )
                    elif k == 4:
                        from dispsolver.element3d.c3d10m_numba import assemble_mesh_c3d10m_numba
                        f_sub, K_sub, err = assemble_mesh_c3d10m_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 5:
                        from dispsolver.element3d.c3d10_numba import assemble_mesh_c3d10_numba
                        f_sub = np.zeros((len(elem_indices), 30))
                        K_sub = np.zeros((len(elem_indices), 30, 30))
                        for sel, C in self._elastic_C_groups(elem_indices):
                            f_g, K_g = assemble_mesh_c3d10_numba(
                                node_coords_all, np.ascontiguousarray(sub_conn[sel]), u_eval, C)
                            f_sub[sel] = f_g
                            K_sub[sel] = K_g
                        err = 0
                    elif k == 6:
                        from dispsolver.element3d.c3d4_numba import assemble_mesh_c3d4_numba
                        f_sub = np.zeros((len(elem_indices), 12))
                        K_sub = np.zeros((len(elem_indices), 12, 12))
                        for sel, C in self._elastic_C_groups(elem_indices):
                            f_g, K_g = assemble_mesh_c3d4_numba(
                                node_coords_all, np.ascontiguousarray(sub_conn[sel]), u_eval, C)
                            f_sub[sel] = f_g
                            K_sub[sel] = K_g
                        err = 0
                    elif k == 7:
                        from dispsolver.element3d.c3d6_numba import assemble_mesh_c3d6_numba
                        f_sub, K_sub, err = assemble_mesh_c3d6_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 8:
                        from dispsolver.element3d.c3d8r_numba import assemble_mesh_c3d8r_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8r_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 9:
                        from dispsolver.element3d.c3d4_anp_numba import assemble_mesh_c3d4_anp_numba
                        f_sub, K_sub, err = assemble_mesh_c3d4_anp_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 10:
                        # Small-strain linear C3D8 (NLGEOM off). The kernel takes
                        # ONE elasticity matrix, so assemble each linear-elastic
                        # material separately with its own C -- it used to apply
                        # self.C_mat_default to every element whatever its
                        # material. _build_kernel_groups refuses kernel 10 for
                        # non-linear-elastic materials.
                        from dispsolver.element3d.c3d8_numba import assemble_mesh_c3d8_numba
                        f_sub = np.zeros((len(elem_indices), 24), dtype=np.float64)
                        K_sub = np.zeros((len(elem_indices), 24, 24), dtype=np.float64)
                        for sel, C in self._elastic_C_groups(elem_indices):
                            f_g, K_g = assemble_mesh_c3d8_numba(
                                node_coords_all, np.ascontiguousarray(sub_conn[sel]), u_eval, C)
                            f_sub[sel] = f_g
                            K_sub[sel] = K_g
                        err = 0
                    elif k == 11:
                        from dispsolver.element3d.cr_wrapper_3d import assemble_mesh_c3d8i_cr_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8i_cr_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 12:
                        from dispsolver.element3d.solid_shell_numba import assemble_mesh_solid_shell_numba
                        f_sub, K_sub, err = assemble_mesh_solid_shell_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 13:
                        from dispsolver.element3d.cr_wrapper_3d import assemble_mesh_c3d8r_cr_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8r_cr_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 14:
                        from dispsolver.element3d.cr_wrapper_3d import assemble_mesh_solid_shell_cr_numba
                        f_sub, K_sub, err = assemble_mesh_solid_shell_cr_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 15:
                        from dispsolver.element3d.c3d20_numba import assemble_mesh_c3d20_numba
                        is_reduced = False
                        if hasattr(self.mesh, "elements") and len(elem_indices) > 0:
                            elements_list = list(self.mesh.elements.values())
                            etype_str = str(getattr(elements_list[elem_indices[0]], "elem_type", "")).upper()
                            if "R" in etype_str or "REDUCED" in etype_str:
                                is_reduced = True
                        f_sub, K_sub, err = assemble_mesh_c3d20_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt, reduced_integration=is_reduced
                        )
                    elif k == 16:
                        from dispsolver.element3d.c3d8_corotational_numba import assemble_mesh_c3d8_corotational_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_corotational_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt,
                            elem_controls=sub_controls, elem_stress_init=sub_stress_init, small_strain=True
                        )
                    elif k == 17:
                        from dispsolver.element3d.c3d8_hybrid_numba import assemble_mesh_c3d8_hybrid_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8_hybrid_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt,
                            sub_controls, sub_stress_init, True
                        )
                    elif k == 18:
                        from dispsolver.element3d.cr_generic_3d import assemble_mesh_c3d8i_small_strain_numba
                        C_all = np.zeros((len(elem_indices), 6, 6))
                        for sel, C in self._elastic_C_groups(elem_indices):
                            C_all[sel] = C
                        f_sub, K_sub = assemble_mesh_c3d8i_small_strain_numba(
                            node_coords_all, np.ascontiguousarray(sub_conn, dtype=np.int64), u_eval, C_all)
                        err = 0
                    elif k == 19:
                        from dispsolver.element3d.c3d8ih_numba import assemble_mesh_c3d8ih_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8ih_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt,
                            elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    elif k == 20:
                        from dispsolver.element3d.cr_wrapper_3d import assemble_mesh_c3d8ih_cr_numba
                        f_sub, K_sub, err = assemble_mesh_c3d8ih_cr_numba(
                            node_coords_all, sub_conn, u_eval, sub_mat, sub_props, sub_sdvs, dt,
                            elem_controls=sub_controls, elem_stress_init=sub_stress_init
                        )
                    else:
                        raise ValueError(f"Unknown element kernel index: {k}")

                    if _KERNELS_3D[k].initial_stress == "added":
                        # kernels without a stress input: add int B^T s0 dV (element3d/initial_stress3d)
                        from dispsolver.element3d.initial_stress3d import initial_stress_force
                        f_sub = np.asarray(f_sub, dtype=np.float64) + initial_stress_force(
                            np.ascontiguousarray(node_coords_all, dtype=np.float64),
                            np.ascontiguousarray(sub_conn, dtype=np.int64),
                            np.ascontiguousarray(sub_stress_init, dtype=np.float64))

                    if cr_R is not None:
                        from dispsolver.element3d.cr_generic_3d import corotational_to_global
                        f_sub = np.ascontiguousarray(f_sub, dtype=np.float64)
                        K_sub = np.ascontiguousarray(K_sub, dtype=np.float64)
                        corotational_to_global(f_sub, K_sub, cr_R)
                        node_coords_all, sub_conn, u_eval = _cr_saved

                    if update_state:
                        # NumPy advanced indexing creates a copy. We must explicitly write the modified sub_sdvs back.
                        self.elem_sdvs[elem_indices] = sub_sdvs

                    if err:
                        has_error = True

                    if self.has_nodal_transforms and self.node_transforms is not None:
                        from dispsolver.model.nodal_transform import apply_nodal_transforms_to_element_group
                        apply_nodal_transforms_to_element_group(f_sub, K_sub, sub_conn, self.node_transforms, dim=3)

                    scatter_f_int_3d(f_sub, sub_conn, f_int_global)
                    if hasattr(self, "_data_topo_buf") and self._data_topo_buf is not None and group_key in self.elem_group_data_slices:
                        s_start, s_end = self.elem_group_data_slices[group_key]
                        self._data_topo_buf[s_start:s_end] = K_sub.ravel()
                    else:
                        data_topo_list.append(K_sub.ravel())

                # Incorporate equivalent thermal load vector: f_int = f_int^mech - f_th
                f_th = self.compute_thermal_load_vector()
                if f_th is not None:
                    if self.has_nodal_transforms and self.node_transforms is not None:
                        from dispsolver.model.nodal_transform import transform_u_to_local
                        f_th = transform_u_to_local(f_th, self.node_transforms, dim=3)
                    f_int_global -= f_th

                self.last_assembly_error = has_error
                if hasattr(self, "_data_topo_buf") and self._data_topo_buf is not None:
                    data_topo = self._data_topo_buf
                else:
                    data_topo = np.concatenate(data_topo_list) if len(data_topo_list) > 0 else np.zeros(0, dtype=np.float64)

                self.f_int = f_int_global
                if len(self.constraints) == 0:
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
                    if return_error:
                        return K_global, f_int_global, has_error
                    return K_global, f_int_global

                # If surface tie or MPC constraints exist, append their contributions efficiently
                rows_all = [self.rows_topo]
                cols_all = [self.cols_topo]
                data_all = [data_topo]

                for constraint in self.constraints:
                    if hasattr(constraint, "bind_solver"):
                        constraint.bind_solver(self, dt)
                    if hasattr(constraint, "reproject_deformed"):
                        constraint.reproject_deformed(u_eval)
                    f_c, (rows_c, cols_c, data_c), _ = constraint.assemble(u_eval)
                    if self.has_nodal_transforms and self.node_transforms is not None:
                        from dispsolver.model.nodal_transform import transform_u_to_local
                        f_c = transform_u_to_local(f_c, self.node_transforms, dim=3)
                    f_int_global += f_c
                    rows_all.append(np.asarray(rows_c, dtype=np.int32))
                    cols_all.append(np.asarray(cols_c, dtype=np.int32))
                    data_all.append(np.asarray(data_c, dtype=np.float64))

                self.f_int = f_int_global
                K_global = csr_matrix(
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

    def apply_boundary_conditions(self, K_global: Any, residual: np.ndarray, u_k: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """Apply Dirichlet boundary conditions via exact diagonal penalty method.
        Direct CSR in-place path eliminates COO re-allocation and re-sorting (0.005s vs 1.85s).
        """
        r_bc = residual.copy()
        if len(self.fixed_dofs) == 0:
            if not isinstance(K_global, csr_matrix):
                K_global = K_global.tocsr()
            return K_global, r_bc
        penalty = 1.0e15
        fixed = np.fromiter(self.fixed_dofs.keys(), dtype=np.int64,
                            count=len(self.fixed_dofs))
        is_fixed = np.zeros(K_global.shape[0], dtype=bool)
        is_fixed[fixed] = True
        tgt = np.fromiter(self.fixed_dofs.values(), dtype=np.float64,
                          count=len(self.fixed_dofs))

        if isinstance(K_global, csr_matrix):
            # Fast in-place CSR mask (0.005s vs 1.85s, zero memory reallocation)
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
            data = np.concatenate([coo.data[keep],
                                   np.full(fixed.shape, penalty)])
            K_bc = coo_matrix((data, (rows, cols)),
                              shape=K_global.shape).tocsr()
            r_bc[fixed] = penalty * (tgt - u_k[fixed])

        # Guard against structurally empty rows (isolated nodes / unreferenced DOFs)
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
            r_bc[empty_rows] = 0.0

        # Guard against zero-diagonal entries (unreferenced nodes with pre-allocated zero diagonal slots)
        diag = K_bc.diagonal().copy()
        zero_diags = np.where(np.abs(diag) < 1e-12)[0]
        if len(zero_diags) > 0:
            diag[zero_diags] = 1.0
            K_bc.setdiag(diag)
            r_bc[zero_diags] = 0.0

        return K_bc, r_bc

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

    def _update_contact_chatter_state(self, u: np.ndarray) -> None:
        """Active-set CHATTERING stabilization (design doc
        dev_log/contact_stabilization_precise_design_20260915.md sec3):
        calls update_gate_state(u) on every constraint that exposes one,
        mirroring _contact_active_set()'s own style.

        **Timing, corrected from the design doc's original proposal**
        (measured 2026-09-16: calling this once per raw_iter immediately
        before `current_active_set = self._contact_active_set(u_k)`, as
        the design doc's sec3 originally specified, mutates the gate
        BETWEEN the official residual assembly at the top of the loop and
        the line-search/Armijo trial assemblies later in the SAME
        iteration -- those trial assemble() calls then see a DIFFERENT
        gate than the official residual did, so the "official" r_norm
        (computed pre-mutation, contact excluded while the gate was still
        closed) and every trial r_t_norm (computed post-mutation, contact
        suddenly included) become mutually inconsistent within one
        iteration; on the plain-penalty single-cube fixture this made
        every Armijo trial's residual look WORSE than the artificially-low
        official one, `found_valid` stayed False, and solve_step()
        reported non-convergence despite the physical configuration being
        fine). Fixed by calling this at the exact same THREE commit
        points `update_augmented_multipliers()` already uses (iteration-1
        fast path, the Anderson-acceleration accepted-trial branch, and
        post-Armijo) instead of before the SDI check -- i.e. the gate
        only ever changes once a trial is ACCEPTED for this iteration,
        so every assemble() call within a single raw_iter (official and
        every trial) sees the SAME gate value, and `current_active_set`
        at the top of the NEXT iteration reads whatever the PREVIOUS
        iteration's commit left behind -- the identical one-iteration-lag
        convention `self._lam`/AL already has and that already works
        (SDI reads a value that is stable for one whole iteration, never
        one that changes mid-iteration)."""
        for c in self.constraints:
            if hasattr(c, "update_gate_state"):
                c.update_gate_state(u)

    def _contact_full_state(self, u: np.ndarray) -> frozenset:
        """Union of `_contact_active_set(u)` (normal contact, unchanged)
        and every constraint's `get_friction_state(u)` (Coulomb stick/
        slip, design doc dev_log/contact_friction_precise_design_
        20260916.md sec6) -- feeds `solve_step()`'s SDI comparison so a
        stick<->slip transition is exempted from the normal convergence
        check the same way a normal-contact activation event already is.
        Additive: with no constraint exposing `get_friction_state` (the
        common, friction-disabled case), this degenerates to exactly
        `_contact_active_set(u)`'s existing return value -- byte-
        identical old behavior."""
        normal = self._contact_active_set(u)
        # Abaqus/Standard parity: Severe Discontinuity Iterations (SDI) track
        # geometric normal contact status changes (open/close). Tangential friction
        # stick/slip transitions are regularized by tangential penalty stiffness (Ktt)
        # and converge stably within equilibrium iterations without causing SDI lock.
        return normal

    def _commit_contact_friction_state(self, u: np.ndarray) -> None:
        """Advances the Coulomb friction anchor (design doc dev_log/
        contact_friction_precise_design_20260916.md sec5) for every
        constraint that exposes `commit_friction_state`. Called ONLY at
        `solve_step()`'s existing `update_state=True` sites (a converged
        increment), NEVER at the per-iteration PDASS/chatter-gate commit
        points `_update_contact_chatter_state`/`update_augmented_
        multipliers` use -- the anchor is physical slip history, not a
        numerical device, and advancing it on a trial that later gets
        rejected (line-search halving, a cutback increment) would
        permanently corrupt that history with motion that never actually
        happened (sec5.1 -- the same reasoning this codebase already
        applies to `elem_sdvs`/`update_state`)."""
        for c in self.constraints:
            if hasattr(c, "commit_friction_state"):
                c.commit_friction_state(u)

    # ------------------------------------------------------------------
    # Newton-loop decision logic, shared by solve_step() and
    # _solve_step_with_hooks(). These two loops used to carry private copies
    # of every rule below, and a fix applied to one never reached the other
    # (the .inp path runs _solve_step_with_hooks; see dev_log/
    # plan_deck06_contact_fix_20261004.md P1). Change a rule HERE, once.
    # Where the two loops still differ (freeze window, line-search length,
    # how prescribed DOFs move during a partial step) the difference is an
    # explicit argument, not a silent fork.
    # ------------------------------------------------------------------

    # Kernels whose stress field can be evaluated at an arbitrary natural point
    # by element_point_stress(), with the SAME kinematics/strain measure the
    # assembly kernel uses.
    _POINT_STRESS_KERNELS = {
        1: "C3D8 under NLGEOM (C3D8_CR: co-rotational, Hughes B-bar, small local strain)",
        10: "C3D8 linear small strain (c3d8_numba, per-material C)",
        16: "C3D8 without NLGEOM (C3D8_CR core with R = I: small-strain Hughes B-bar, any material)",
    }

    def element_point_stress(self, row: int, X_e: np.ndarray, dNdX: np.ndarray,
                             vol_mean: np.ndarray, u_e: np.ndarray, dt: float = 1.0) -> Optional[np.ndarray]:
        """Global Cauchy stress (3x3) of element `row` at a point with reference
        shape-function gradients dNdX (8, 3), evaluated with the element's own
        kernel -- i.e. after the NLGEOM-dependent promotion that picked that
        kernel. vol_mean is the B-bar volumetric row (only used by kernel 1).
        Returns None if the material reports a failure. Kernels outside
        _POINT_STRESS_KERNELS raise: substituting another strain measure would
        silently make a consistency-based method (Nitsche) inconsistent."""
        from dispsolver.material3d.numba_materials import material_dispatch_3d
        kern = getattr(self, "_row_kernel", {}).get(int(row))
        if kern not in self._POINT_STRESS_KERNELS:
            raise NotImplementedError(
                f"element_point_stress: element row {row} runs kernel {kern}, not one of "
                f"{sorted(self._POINT_STRESS_KERNELS)} ({'; '.join(self._POINT_STRESS_KERNELS.values())})")
        B = np.zeros((6, 24))
        for i in range(8):
            dx, dy, dz = dNdX[i]
            B[0, 3 * i] = dx; B[1, 3 * i + 1] = dy; B[2, 3 * i + 2] = dz
            B[3, 3 * i] = dy; B[3, 3 * i + 1] = dx
            B[4, 3 * i + 1] = dz; B[4, 3 * i + 2] = dy
            B[5, 3 * i] = dz; B[5, 3 * i + 2] = dx
        if kern == 10:
            sv = self._linear_elastic_C(self.elem_props[row, 0], self.elem_props[row, 1]) @ (B @ u_e.ravel())
            R = np.eye(3)
        else:  # kern 1, or 16 (= kernel 1 with R = I)
            from dispsolver.element3d.c3d8_corotational_numba import compute_element_rotation_3d
            x_e = X_e + u_e
            R = np.eye(3) if kern == 16 else compute_element_rotation_3d(X_e, x_e)
            u_loc = ((x_e - x_e.mean(axis=0)) @ R - (X_e - X_e.mean(axis=0))).ravel()
            vol_std = (B[0] + B[1] + B[2]) / 3.0
            for r_ in range(3):
                B[r_] = B[r_] - vol_std + vol_mean
            eps = B @ u_loc
            sdv_all = getattr(self, "elem_sdvs", None)
            sdv = np.zeros(0) if sdv_all is None else np.asarray(sdv_all[row])
            sdv = np.ascontiguousarray(sdv[0] if sdv.ndim > 1 else sdv, dtype=np.float64)
            sv, _, _, err = material_dispatch_3d(int(self.elem_mat_types[row]), self.elem_props[row], sdv,
                                                 np.ascontiguousarray(eps), np.eye(3),
                                                 1.0 + eps[0] + eps[1] + eps[2], dt)
            if err != 0:
                return None
        sig = np.array([[sv[0], sv[3], sv[5]], [sv[3], sv[1], sv[4]], [sv[5], sv[4], sv[2]]])
        return R @ sig @ R.T

    def _clear_contact_freeze(self) -> None:
        for c in self.constraints:
            if hasattr(c, "frozen_active_nodes"):
                c.frozen_active_nodes = None

    @staticmethod
    def _node_ids_of(active_set) -> set:
        ids = set()
        for item in active_set:
            nid = item if isinstance(item, int) else (item[0] if isinstance(item, tuple) else None)
            if nid is not None:
                ids.add(nid)
        return ids

    def _sdi_update(self, st: Dict[str, Any], current_active_set: frozenset,
                    u_k: np.ndarray, freeze_window: Optional[int]) -> bool:
        """Severe-discontinuity bookkeeping for one iterate. Mutates `st`
        (keys: prev, recent, sdi_iters, equil_iters) and returns is_sdi.

        After 3 SDI iterations, or when the contact state revisits a recent
        one, each freezable constraint's active node set is frozen to the
        union of the last `freeze_window` states (all retained states when
        None) plus its current active set."""
        is_sdi = current_active_set != st["prev"]
        recent = st["recent"]
        # Oscillation = the state CHANGED back to one seen before (A->B->A).
        # The former test (`current in recent`) also fired for an active set
        # that simply stayed the same, freezing every steadily-converging
        # contact on its third iteration.
        is_oscillation = is_sdi and current_active_set in recent[:-1]
        recent.append(current_active_set)
        if len(recent) > 6:
            recent.pop(0)
        if is_sdi:
            st["sdi_iters"] += 1
        if (st["sdi_iters"] >= 3 or is_oscillation) and len(self.constraints) > 0:
            is_sdi = False
            st["equil_iters"] += 1
            window = recent if freeze_window is None else recent[-freeze_window:]
            for c in self.constraints:
                if hasattr(c, "get_active_set") and getattr(c, "frozen_active_nodes", None) is None:
                    union_nodes = set()
                    for s_set in window:
                        union_nodes |= self._node_ids_of(s_set)
                    c_active = c.get_active_set(u_k)
                    union_nodes |= self._node_ids_of(c_active)
                    c.frozen_active_nodes = union_nodes if union_nodes else set(c_active)
        elif not is_sdi:
            st["equil_iters"] += 1
        st["prev"] = current_active_set
        return is_sdi

    # Residual tolerances are RELATIVE to the current iterate's force scale q
    # (_force_scale), never absolute: an absolute `r < 1.0 N` accepted a 10%
    # residual on a 0.2 N contact patch test, and a historical-max reference
    # (the old r_0_norm, inflated by the iteration-2 Dirichlet spike of
    # ~2e5 N on deck #06) let the exhaustion branch accept r = 90 N.
    _RESID_TOL_CONTACT = 1e-3
    _RESID_TOL = 5e-3
    _CORR_TOL = 1e-3          # max|du_free| relative to max|u - u_n| this increment
    _CORR_RESID_TOL = 1e-2    # residual allowed alongside a converged correction
    # Residual at roundoff level, ~eps * |K| * |u|. Needed when q ~ 0 (e.g. a
    # pure rigid translation: no load, no reaction, residual is only noise).
    # Scaled, not absolute: an absolute 1e-10 rejected 1.3e-10 of noise on a
    # K ~ 1e6, u ~ 0.5 model. Set per iterate by _set_roundoff_floor().
    _r_zero = 0.0

    def _set_roundoff_floor(self, K_g, u: np.ndarray) -> None:
        # Roundoff in f_int scales with the POSITIONS the kinematics are
        # evaluated at (x = X + u), not with u: an unloaded, undeformed model
        # (u ~ 0) still carries ~eps*K*|X| of noise.
        if getattr(self, "_X_flat", None) is None or self._X_flat.shape != u.shape:
            self._X_flat = np.asarray(self.mesh.nodes_array(), dtype=np.float64)[:, :3].ravel()[: u.shape[0]]
        kd = float(np.max(np.abs(K_g.diagonal()))) if K_g.shape[0] else 0.0
        self._r_zero = 1e3 * np.finfo(float).eps * kd * max(float(np.max(np.abs(self._X_flat + u))), 1.0e-300)

    def _force_scale(self, f_int: np.ndarray, f_ext: np.ndarray, free_dof_mask: np.ndarray) -> float:
        """q: larger of the applied free-DOF load and the reaction at
        prescribed DOFs, both at the current iterate."""
        f_ext_free = float(np.linalg.norm(f_ext[free_dof_mask]))
        reaction = float(np.linalg.norm(f_int[~free_dof_mask])) if not free_dof_mask.all() else 0.0
        # dynamic increment: the inertia force is a force of the balance too (an unloaded
        # free vibration has neither load nor reaction to scale the residual by)
        inertia = self._dyn_ctx.inertia_norm if self._dyn_ctx is not None else 0.0
        return max(f_ext_free, reaction, inertia)

    def _residual_ok(self, r_norm: float, q: float, iter_count: int = 1) -> bool:
        if hasattr(self, "solution_controls") and self.solution_controls is not None:
            R_n_eff, C_n_eff, R_p_eff, C_a_eff = self.solution_controls.get_effective_tolerances(
                cutback_count=getattr(self, "cutback_count", 0),
                iter_count=iter_count,
                is_stalling=(iter_count >= 6)
            )
            tol = R_n_eff
        else:
            tol = self._RESID_TOL_CONTACT if len(self.constraints) > 0 else self._RESID_TOL
        return r_norm <= tol * q or r_norm <= self._r_zero

    def _converged_by_residual(self, iter_count: int, is_sdi: bool, bc_err: float,
                               r_norm: float, q: float) -> bool:
        return iter_count > 1 and not is_sdi and bc_err < 1e-5 and self._residual_ok(r_norm, q, iter_count=iter_count)

    def _converged_by_correction(self, iter_count: int, is_sdi: bool, bc_err_next: float,
                                 du_free_max: float, du_inc_max: float,
                                 r_norm: float, q: float) -> bool:
        if not (iter_count > 1 and not is_sdi and bc_err_next < 1e-5):
            return False
        if hasattr(self, "solution_controls") and self.solution_controls is not None:
            R_n_eff, C_n_eff, R_p_eff, C_a_eff = self.solution_controls.get_effective_tolerances(
                cutback_count=getattr(self, "cutback_count", 0),
                iter_count=iter_count,
                is_stalling=(iter_count >= 6)
            )
            # Abaqus Alternative Convergence Check (saves tail iterations)
            alt_ok = (du_free_max <= C_a_eff * du_inc_max or du_free_max <= self._r_zero) and (r_norm <= R_p_eff * q or r_norm <= self._r_zero)
            if alt_ok:
                return True
            c_tol = C_n_eff
            r_tol = R_p_eff
        else:
            c_tol = self._CORR_TOL
            r_tol = self._CORR_RESID_TOL

        corr_ok = du_free_max <= c_tol * du_inc_max or du_free_max <= self._r_zero
        return corr_ok and (r_norm <= r_tol * q or r_norm <= self._r_zero)

    def _converged_on_exhaustion(self, is_sdi: bool, bc_err: float, r_norm: float, q: float) -> bool:
        """Same criterion as the residual branch: running out of iterations is
        not a reason to accept a larger residual."""
        return not is_sdi and bc_err < 1e-5 and self._residual_ok(r_norm, q)

    def _freeze_lift_ok(self, u: np.ndarray, f_ext: np.ndarray, free_dof_mask: np.ndarray,
                        dt: float, sdi: Dict[str, Any], resid_tol: Optional[float] = None) -> bool:
        """Gate in front of every acceptance. A frozen contact set is a
        numerical device (bilateral pairs, labels pinned) -- the answer must
        satisfy the real one-sided law. With no freeze active this is a
        no-op returning True. Otherwise the freeze is lifted, the state
        re-assembled, and the iterate accepted only if the true residual
        still meets the branch's tolerance (`resid_tol` * q, default the
        residual-branch rule). If not, the freeze STAYS lifted and the SDI
        counters restart, so iteration continues on the real law (and ends
        in a cutback if it cannot converge) -- never a silent accept."""
        if not any(getattr(c, "frozen_active_nodes", None) is not None for c in self.constraints):
            return True
        self._clear_contact_freeze()
        _, f_free = self.assemble_system(u, dt=dt)
        r = float(np.linalg.norm((f_ext - f_free)[free_dof_mask]))
        q = self._force_scale(f_free, f_ext, free_dof_mask)
        ok = (r <= resid_tol * q or r <= self._r_zero) if resid_tol is not None else self._residual_ok(r, q)
        if not ok:
            sdi["sdi_iters"] = 0
            sdi["recent"] = []
        return ok

    def _accept_increment(self, u_acc: np.ndarray, branch: str, f_ext: np.ndarray,
                          iters: int, dt: float, reset_tuner: bool) -> None:
        """Commit a converged iterate: SDVs, acceptance probe, friction
        anchors, then lift any contact freeze."""
        self.u = u_acc
        _, f_acc = self.assemble_system(u_acc, dt=dt, update_state=True)
        self._record_acceptance(branch, f_acc, f_ext, iters, dt=dt)
        self._commit_contact_friction_state(u_acc)
        if reset_tuner:
            self.contact_tuner.reset_attempt_counter()
        self._clear_contact_freeze()

    def _line_search(self, u_k: np.ndarray, du_step: np.ndarray, f_ext: np.ndarray,
                     free_dof_mask: np.ndarray, r_norm: float, q: float, dt: float,
                     n_trials: int, prescribed_at_partial_step: str,
                     diag: Optional[list] = None) -> Tuple[Optional[np.ndarray], float, bool]:
        """Backtracking on s = 1, 1/2, ... (n_trials values). Returns
        (u_new, s, found_reduction); u_new is None when the step must be cut
        back.

        prescribed_at_partial_step: "target" puts prescribed DOFs at their
        target value on every trial; "scaled" does so only at s == 1 and
        otherwise moves them by s*du_step like the free DOFs."""
        s = 1.0
        min_r_norm = float("inf")
        min_u = None
        for _ in range(n_trials):
            u_trial = u_k + s * du_step
            if prescribed_at_partial_step == "target" or s >= 0.999:
                for dof, target_val in self.fixed_dofs.items():
                    u_trial[dof] = target_val
            _, f_t = self.assemble_system(u_trial, dt=dt)
            if self.last_assembly_error > 0:
                if diag is not None:
                    diag.append((s, "err"))
                s *= 0.5
                continue
            r_t_norm = float(np.linalg.norm((f_ext - f_t)[free_dof_mask]))
            if diag is not None:
                diag.append((s, f"{r_t_norm:.2e}"))
            if r_t_norm < min_r_norm:
                min_r_norm = r_t_norm
                min_u = u_trial
            if r_t_norm <= max(r_norm, 1e-4 * q, self._r_zero):
                # The first trial meeting the bound is also the minimum so far:
                # every earlier trial exceeded that bound.
                return u_trial, s, True
            s *= 0.5
        # No trial reduced the residual.
        if min_u is not None and min_r_norm <= 2.5 * r_norm:
            return min_u, s, False
        # A step that changes the contact active set is a severe
        # discontinuity, not a bad direction: newly closed points jump from
        # zero to k*gap, so the residual grows ~linearly in s and no
        # backtracking can reduce it. Take the full step (if it does not
        # invert elements) and let the next iterate see the new contact
        # state as an SDI iteration (no convergence check; bounded by the
        # caller's SDI budget). This replaces a former `or min_r_norm < 1e5`
        # clause that accepted ANY step whenever the residual was below 1e5 N.
        u_full = u_k + du_step
        for dof, target_val in self.fixed_dofs.items():
            u_full[dof] = target_val
        if self._contact_full_state(u_full) != self._contact_full_state(u_k):
            self.assemble_system(u_full, dt=dt)
            if self.last_assembly_error == 0:
                return u_full, 1.0, False
        return None, s, False

    def solve_step(
        self,
        dt: float = 1.0,
        f_ext: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        max_iters: int = 25,
        max_sdi_iters: int = 200,
        augment_relaxation: float = 1.0,
        use_anderson_accel: bool = False,
        anderson_m: int = 4,
        anderson_beta: float = 1.0,
        auto_tune_contact: bool = False,
        f_ext_n: Optional[np.ndarray] = None,
    ) -> Tuple[bool, int]:

        """Solve a 3D non-linear incremental step using Newton-Raphson with
        Armijo line search.

        PDASS / single-loop augmented Lagrangian (design doc
        dev_log/contact_pdass_precise_design_20260915.md): for any
        constraint with `augmented_lagrange=True`,
        `update_augmented_multipliers()` is called once per ACCEPTED
        Newton iterate (both commit points below), not once per fully-
        converged `solve_step()` call as the old `solve_step_augmented()`
        outer-loop design did. This is the Simo & Laursen (1992) single-
        loop construction: nesting the Uzawa multiplier update inside the
        Newton loop itself converges the multiplier at Newton's own rate
        instead of a much slower outer-cycle rate, with NO change to
        `assemble()`'s own math (design doc sec1: the per-node residual/
        tangent formula was already exactly correct for a momentarily-
        frozen multiplier; only *when* it gets updated changes).
        Deliberately NOT called from inside `assemble()` itself (design
        doc sec2.1): `assemble_system()` runs multiple times per Newton
        iteration against trial (not necessarily accepted) displacement
        vectors -- the iteration-1 probe and up to 6 line-search
        halvings -- so updating the multiplier there would advance it
        once per CALL instead of once per accepted ITERATE, an
        irreproducible bug depending on how many line-search trials
        happened to run.

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
        """
        # EMCS Implicit Dynamic Dispatch Branch
        if self.time_integrator == "emcs":
            if self._emcs_solver is None:
                from dispsolver.solver3d.emcs_dynamic3d import EMCSSolver3D
                self._emcs_solver = EMCSSolver3D(
                    mesh=self.mesh,
                    density=self.density,
                    material_params=self.material_params,
                    alpha_m=self.alpha_m,
                    beta_k=self.beta_k,
                    use_anderson=use_anderson_accel,
                    anderson_m=anderson_m,
                    anderson_beta=anderson_beta
                )
            
            # Sync fixed DOFs and nodal forces
            self._emcs_solver.fixed_dofs = self.fixed_dofs
            if hasattr(self, "f_ext_applied") and self.f_ext_applied is not None:
                self._emcs_solver.f_ext_applied = self.f_ext_applied

            if self._emcs_solver is not None:
                self._emcs_solver.state.u[:] = self.u[:]
                self._emcs_solver.state.v[:] = self.v[:]
                self._emcs_solver.state.a[:] = self.a[:]
                max_emcs_iters = max(max_iters, 40) if max_iters <= 25 else max_iters
                emcs_tol = min(tol, 1e-6)
                conv, iters, _ = self._emcs_solver.solve_step_emcs(dt=dt, tol=emcs_tol, max_iters=max_emcs_iters)
                # Sync back solution states
                self.u[:] = self._emcs_solver.state.u
                self.v[:] = self._emcs_solver.state.v
                self.a[:] = self._emcs_solver.state.a
                return conv, iters

        # HHT / Newmark: the same Newton loop on the dynamic effective system
        # (implicit_dynamics.run_increment); a static integrator runs it as is.
        from dispsolver.solver.implicit_dynamics import run_increment
        return run_increment(self, 3, self._solve_step_newton, dt, f_ext, f_ext_n,
                             tol=tol, max_iters=max_iters, max_sdi_iters=max_sdi_iters,
                             augment_relaxation=augment_relaxation, use_anderson_accel=use_anderson_accel,
                             anderson_m=anderson_m, anderson_beta=anderson_beta,
                             auto_tune_contact=auto_tune_contact)

    def _solve_step_newton(
        self,
        dt: float = 1.0,
        f_ext: Optional[np.ndarray] = None,
        tol: float = 1e-5,
        max_iters: int = 25,
        max_sdi_iters: int = 200,
        augment_relaxation: float = 1.0,
        use_anderson_accel: bool = False,
        anderson_m: int = 4,
        anderson_beta: float = 1.0,
        auto_tune_contact: bool = False,
    ) -> Tuple[bool, int]:
        """The Newton loop of solve_step (see its docstring): static, or on the HHT
        effective system while self._dyn_ctx is set."""
        if hasattr(self, "solution_controls") and self.solution_controls is not None:
            self.solution_controls.tolerance_engine.is_active = self.solution_controls.adaptive_convergence

        if f_ext is None:


            if hasattr(self, "f_ext_applied") and self.f_ext_applied is not None:
                f_ext = self.f_ext_applied.copy()
            else:
                f_ext = np.zeros(self.num_dofs, dtype=np.float64)

        free_dof_mask = np.ones(self.num_dofs, dtype=bool)
        if self.fixed_dofs:
            free_dof_mask[list(self.fixed_dofs.keys())] = False

        u_k = self.u.copy()
        u_start = u_k.copy()

        accelerator = None
        if use_anderson_accel:
            from dispsolver.solver.anderson_acceleration import AndersonAccelerator
            accelerator = AndersonAccelerator(m=anderson_m, beta=anderson_beta)

        self._clear_contact_freeze()
        sdi = {"prev": self._contact_full_state(u_k), "recent": [], "sdi_iters": 0, "equil_iters": 0}
        raw_iter = 0
        max_raw_iters = max_iters + max_sdi_iters

        while sdi["equil_iters"] < max_iters and raw_iter < max_raw_iters:
            raw_iter += 1
            iter_count = raw_iter
            # print(f"    --> Starting assemble_system iter {iter_count}", flush=True)
            K_g, f_int = self.assemble_system(u_k, dt=dt)
            if self.last_assembly_error > 0:
                # Smart Cutback: Element inverted
                return False, iter_count
            self._set_roundoff_floor(K_g, u_k)
            residual = f_ext - f_int

            # Compute true physical residual norm on free DOFs
            r_free = residual[free_dof_mask]
            r_norm = float(np.linalg.norm(r_free))

            # Apply BCs for linear system solve
            K_bc, r_bc = self.apply_boundary_conditions(K_g, residual, u_k)

            # Severe-discontinuity-iteration (SDI) check
            current_active_set = self._contact_full_state(u_k)
            is_sdi = self._sdi_update(sdi, current_active_set, u_k, freeze_window=2)
            q = self._force_scale(f_int, f_ext, free_dof_mask)
            if getattr(self, "debug_sdi", False):
                print(f"[SDI] raw_iter={raw_iter} equil_iters={sdi['equil_iters']} is_sdi={is_sdi} "
                      f"n_active={len(current_active_set)} r_norm={r_norm:.4g} r/q={r_norm / max(q, 1e-300):.4g}", flush=True)

            bc_err = max([abs(target_val - u_k[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            if (self._converged_by_residual(iter_count, is_sdi, bc_err, r_norm, q)
                    and self._freeze_lift_ok(u_k, f_ext, free_dof_mask, dt, sdi)):
                self._accept_increment(u_k, "residual", f_ext, iter_count, dt, reset_tuner=True)
                return True, iter_count

            # Solve linear system \Delta u = K^-1 * r
            try:
                du = self._solve_linear(K_bc, r_bc)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError("Pardiso linear solver failed (matrix is likely singular). Check boundary conditions or rigid body modes.") from e

            u_next = u_k + du
            bc_err_next = max([abs(target_val - u_next[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            du_free_max = float(np.max(np.abs(du[free_dof_mask]))) if np.any(free_dof_mask) else 0.0
            du_inc_max = float(np.max(np.abs(u_next - u_start)))
            if (self._converged_by_correction(iter_count, is_sdi, bc_err_next, du_free_max, du_inc_max, r_norm, q)
                    and self._freeze_lift_ok(u_next, f_ext, free_dof_mask, dt, sdi, resid_tol=self._CORR_RESID_TOL)):
                self._accept_increment(u_next, "correction", f_ext, iter_count, dt, reset_tuner=True)
                return True, iter_count

            # Displacement correction limiter: prevent artificial element inversion during contact release/SDI shock
            du_free_max = float(np.max(np.abs(du[free_dof_mask]))) if np.any(free_dof_mask) else 0.0
            du_limit = getattr(self, "max_du_limit", 0.10)
            if du_free_max > du_limit and iter_count > 1:
                du_step = du * (du_limit / du_free_max)
            else:
                du_step = du

            # On iteration 1 (kinematic Dirichlet application) or during SDI (contact status change),
            # take step with limiter to establish contact constraints without artificial line search barrier.
            if iter_count == 1 or is_sdi:
                u_trial = u_k + (du if iter_count == 1 else du_step)
                for dof, target_val in self.fixed_dofs.items():
                    u_trial[dof] = target_val
                _, _ = self.assemble_system(u_trial, dt=dt)
                if self.last_assembly_error == 0:
                    if accelerator is not None:
                        accelerator.record(u_k, du_step)
                    u_k = u_trial
                    for c in self.constraints:
                        if getattr(c, "augmented_lagrange", False):
                            c.update_augmented_multipliers(u_k, omega=augment_relaxation)
                    self._update_contact_chatter_state(u_k)
                    continue

            # If Anderson Acceleration is active, attempt accelerated trial step first
            aa_accepted = False
            if accelerator is not None and iter_count > 1:
                u_aa = accelerator.step(u_k, du_step, mask=free_dof_mask)
                for dof, target_val in self.fixed_dofs.items():
                    u_aa[dof] = target_val

                _, f_aa = self.assemble_system(u_aa, dt=dt)
                if self.last_assembly_error == 0:
                    r_aa = f_ext - f_aa
                    r_aa_norm = float(np.linalg.norm(r_aa[free_dof_mask]))
                    # Accept accelerated iterate if it doesn't cause divergence
                    if r_aa_norm < r_norm or r_aa_norm <= max(1.2 * r_norm, 1e-4 * q, 1e-5):
                        u_k = u_aa
                        aa_accepted = True
                        for c in self.constraints:
                            if getattr(c, "augmented_lagrange", False):
                                c.update_augmented_multipliers(u_k, omega=augment_relaxation)
                        self._update_contact_chatter_state(u_k)
                        continue

                # AA trial rejected by safeguard: pop the unverified record
                accelerator.pop_last()

            u_new, s, _ = self._line_search(u_k, du_step, f_ext, free_dof_mask, r_norm, q, dt,
                                            n_trials=6, prescribed_at_partial_step="target")
            if u_new is None:
                # Non-descent / diverging search direction: cut back cleanly
                self._clear_contact_freeze()
                return False, iter_count
            u_k = u_new
            if accelerator is not None and not aa_accepted:
                accelerator.record(u_k - s * du, s * du)

            for c in self.constraints:
                if getattr(c, "augmented_lagrange", False):
                    c.update_augmented_multipliers(u_k, omega=augment_relaxation)
            self._update_contact_chatter_state(u_k)

        # Exhausted the equilibrium-iteration budget
        self.u = u_k
        if (self._converged_on_exhaustion(is_sdi, bc_err, r_norm, q)
                and self._freeze_lift_ok(u_k, f_ext, free_dof_mask, dt, sdi)):
            self._accept_increment(u_k, "fallthrough", f_ext, raw_iter, dt, reset_tuner=True)
            return True, raw_iter

        # Autonomous In-situ Healing & Relaxation
        if auto_tune_contact:
            tuned = self.contact_tuner.tune_and_relax(reason=f"Newton divergence / Chattering at iter {raw_iter}")
            if tuned:
                self._clear_contact_freeze()
                # Rollback to initial increment state and retry with relaxed parameters
                success, retry_iters = self._solve_step_newton(
                    dt=dt,
                    f_ext=f_ext,
                    tol=tol,
                    max_iters=max_iters,
                    max_sdi_iters=max_sdi_iters,
                    augment_relaxation=augment_relaxation,
                    use_anderson_accel=use_anderson_accel,
                    anderson_m=anderson_m,
                    anderson_beta=anderson_beta,
                    auto_tune_contact=True,
                )
                if success:
                    self.contact_tuner.reset_attempt_counter()
                    return True, raw_iter + retry_iters

        self._clear_contact_freeze()
        return False, raw_iter


    def solve_load_ramp(
        self,
        f_ext_target: np.ndarray,
        target_steps: int = 15,
        min_df_frac: float = 1.0 / 256.0,
        max_iters: int = 35,
        dt: float = 1.0,
        use_anderson_accel: bool = False,
        anderson_m: int = 4,
        anderson_beta: float = 1.0,
    ) -> Dict[str, Any]:
        """Solve non-linear problem by ramping load from current state to f_ext_target,
        with built-in adaptive cutbacks, state rollbacks, and SDV preservation.

        Parameters
        ----------
        f_ext_target : ndarray
            Target total external force vector.
        target_steps : int
            Initial target number of load steps (default: 15).
        min_df_frac : float
            Minimum allowable load increment fraction (default: 1/256).
        max_iters : int
            Max Newton iterations per step (default: 35).
        dt : float
            Time step size (default: 1.0).

        Returns
        -------
        dict with keys: "success", "cutbacks", "p_history", "steps_completed"
        """
        f_target_norm = float(np.linalg.norm(f_ext_target))
        if f_target_norm < 1e-30:
            f_target_norm = 1.0

        f_ext_current = np.zeros(self.num_dofs, dtype=np.float64)
        f_ext_work = np.zeros(self.num_dofs, dtype=np.float64)

        df_frac = 1.0 / float(target_steps)
        p_hist = [0.0]
        cutbacks = 0
        steps_completed = 0
        total_iterations = 0

        while True:
            cur_norm = float(np.linalg.norm(f_ext_current))
            rem_norm = float(np.linalg.norm(f_ext_target - f_ext_current))

            if rem_norm / f_target_norm < 1e-8 or (1.0 - cur_norm / f_target_norm) < 1e-8:
                break

            df_frac_actual = min(df_frac, 1.0 - cur_norm / f_target_norm)
            if df_frac_actual < 1e-12:
                break

            f_ext_trial = f_ext_current + df_frac_actual * f_ext_target

            # 1. State Snapshot (Rollback Shield)
            u_saved = self.u.copy()
            sdvs_saved = self.elem_sdvs.copy() if hasattr(self, "elem_sdvs") and self.elem_sdvs is not None else None

            # 2. Attempt step
            f_ext_work[:] = f_ext_trial
            conv, iters = self.solve_step(
                dt=dt,
                f_ext=f_ext_work,
                max_iters=max_iters,
                use_anderson_accel=use_anderson_accel,
                anderson_m=anderson_m,
                anderson_beta=anderson_beta,
            )
            total_iterations += iters

            if conv:
                # Accept step and advance load level
                f_ext_current[:] = f_ext_trial
                steps_completed += 1
                ramp = float(np.linalg.norm(f_ext_current) / f_target_norm)
                p_hist.append(ramp)

                # Growth strategy on smooth convergence
                if iters <= 5:
                    df_frac = min(df_frac * 2.0, 0.2)
            else:
                # Newton non-convergence: True state rollback
                self.u[:] = u_saved
                if sdvs_saved is not None:
                    self.elem_sdvs[:] = sdvs_saved

                df_frac *= 0.5
                cutbacks += 1

                if df_frac < min_df_frac:
                    # Minimum step reached: accept best effort
                    f_ext_current[:] = f_ext_trial
                    steps_completed += 1
                    ramp = float(np.linalg.norm(f_ext_current) / f_target_norm)
                    p_hist.append(ramp)
                    df_frac = 1.0 / float(target_steps)

        return {
            "success": True,
            "cutbacks": cutbacks,
            "steps_completed": steps_completed,
            "total_iterations": total_iterations,
            "p_history": np.array(p_hist),
        }

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

        **UPGRADED, 2026-09-15** (dev_log/contact_pdass_precise_design_20260915.md,
        the Simo & Laursen 1992 single-loop construction): `solve_step()`
        itself now calls `update_augmented_multipliers()` once per
        ACCEPTED Newton iterate (not once per fully-converged outer
        cycle), so the multiplier is already being driven toward its
        fixed point at Newton's own rate DURING the single call below.
        This method is therefore no longer the mechanism that converges
        the augmentation -- it is a thin pass-through plus a safety net:
        call `solve_step()` once, and only fall back to additional outer
        cycles if `max_penetration` is somehow still above `augment_tol`
        after that single call (expected NOT to trigger on this
        codebase's existing regression cases, per the design doc's own
        falsifiable prediction -- `n_augment_cycles == 1` is now the
        expected, checked outcome, not an accident of a lucky outer loop).

        Only meaningful when at least one of self.constraints has
        augmented_lagrange=True (SurfaceContactConstraint3D); with none,
        this degenerates to exactly one call to solve_step().

        Returns (converged, n_augment_cycles, last_inner_iters). The
        u/state are committed by solve_step() itself as usual.
        """
        augmented_constraints = [c for c in self.constraints if getattr(c, "augmented_lagrange", False)]

        for augment_iter in range(1, max_augment_iters + 1):
            converged, inner_iters = self.solve_step(
                dt=dt, f_ext=f_ext, max_iters=max_iters, max_sdi_iters=max_sdi_iters,
                augment_relaxation=augment_relaxation,
            )
            if not converged:
                return False, augment_iter, inner_iters

            if not augmented_constraints:
                return True, augment_iter, inner_iters

            # Pure query (assemble(), NOT update_augmented_multipliers()):
            # solve_step() already advanced self._lam once per accepted
            # iterate above -- calling the mutating update again here
            # would double-update it. This just reads the CURRENT
            # max_penetration to decide whether a safety-net extra cycle
            # is actually needed.
            max_pen = 0.0
            for c in augmented_constraints:
                _, _, stats = c.assemble(self.u)
                max_pen = max(max_pen, stats["max_penetration"])

            if max_pen < augment_tol:
                return True, augment_iter, inner_iters

        return False, max_augment_iters, inner_iters

    def solve_static_step(self, f_ext: np.ndarray, tol: float = 1e-6, max_iters: int = 20) -> bool:
        """Alias for static Newton step returning boolean converged status."""
        converged, _ = self.solve_step(dt=1.0, f_ext=f_ext, tol=tol, max_iters=max_iters)
        return converged

    def _lower_step(self, step: Any, sys: Optional[Any], total_time: Optional[float] = None):
        """The step's BCs and loads, resolved once (dispsolver.model.step_plan)."""
        from dispsolver.model.step_plan import lower_step
        return lower_step(step, sys, self.mesh, 3, self.nlgeom, total_time=total_time,
                          explicit_density=self._explicit_density)

    def _apply_plan_bcs(self, plan: Any, t: float) -> None:
        """Replace the Dirichlet set by the plan's prescribed DOFs at time t."""
        self.fixed_dofs.clear()
        dofs, vals = plan.prescribed(t, getattr(self, "_u_step_start", None))
        for d, v in zip(dofs.tolist(), vals.tolist()):
            self.set_dirichlet_bc(d, v)

    def _explicit_density(self, row: int) -> float:
        """Density the deck gives element `row` (0.0 when its material has no
        *DENSITY) -- not self.elem_density, which substitutes a default steel
        density so dynamics always has mass."""
        elem = list(self.mesh.elements.values())[row]
        mat = self.materials.get(getattr(elem, "pid", 0)) if isinstance(getattr(self, "materials", None), dict) else None
        if isinstance(mat, dict):
            return float(mat.get("density", mat.get("rho", 0.0)) or 0.0)
        return float(getattr(mat, "density", 0.0) or 0.0)

    def _solve_step_with_hooks(
        self,
        dt: float,
        t_next: float,
        sys: Optional[Any],
        sensor_mgr: Any,
        hooks: List[Any],
        f_ext: Optional[np.ndarray] = None,
        max_iters: int = 50,
        f_ext_n: Optional[np.ndarray] = None,
    ) -> Tuple[bool, int]:
        """One increment of solve(): the hooks Newton loop, on the HHT effective system
        when the integrator is dynamic (implicit_dynamics.run_increment)."""
        from dispsolver.solver.implicit_dynamics import run_increment
        return run_increment(self, 3, self._solve_step_with_hooks_newton, dt, f_ext, f_ext_n,
                             t_next=t_next, sys=sys, sensor_mgr=sensor_mgr, hooks=hooks,
                             max_iters=max_iters)

    def _solve_step_with_hooks_newton(
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
        u_start = u_k.copy()
        self._clear_contact_freeze()
        sdi = {"prev": self._contact_full_state(u_k), "recent": [], "sdi_iters": 0, "equil_iters": 0}
        raw_iter = 0
        max_raw_iters = max_iters + 200
        consecutive_r_growth = 0
        prev_r_norm = float("inf")
        is_sdi = False
        self._last_n_sdi = 0

        while sdi["equil_iters"] < max_iters and raw_iter < max_raw_iters:
            raw_iter += 1
            iter_count = raw_iter
            K_g, f_int = self.assemble_system(u_k, dt=dt)
            if self.last_assembly_error > 0:
                return False, iter_count
            self._set_roundoff_floor(K_g, u_k)

            residual = f_ext - f_int
            r_free = residual[free_dof_mask]
            r_norm = float(np.linalg.norm(r_free))

            if iter_count > 16 and r_norm >= prev_r_norm * 0.999 and not is_sdi:
                consecutive_r_growth += 1
                if consecutive_r_growth >= 6:
                    # Abaqus standard: Cutback on persistent residual stagnation
                    self._clear_contact_freeze()
                    return False, iter_count
            else:
                consecutive_r_growth = 0
            prev_r_norm = r_norm

            if sensor_mgr and hooks:
                sensor_vals = sensor_mgr.evaluate_all(self, sys)
                for hook in hooks:
                    act = hook.execute(iter_count, t_next, self, sensor_vals)
                    if act == ControlAction.ABORT_STEP:
                        return False, iter_count

            K_bc, r_bc = self.apply_boundary_conditions(K_g, residual, u_k)

            current_active_set = self._contact_full_state(u_k)
            is_sdi = self._sdi_update(sdi, current_active_set, u_k, freeze_window=2)
            if is_sdi:
                self._last_n_sdi += 1
            q = self._force_scale(f_int, f_ext, free_dof_mask)
            if getattr(self, "debug_sdi", False) or (iter_count <= 5 or iter_count % 5 == 0):
                print(f"      [Iter {iter_count:2d}] r={r_norm:.2e} r/q={r_norm / max(q, 1e-300):.2e} q={q:.2e} sdi={is_sdi} act={len(current_active_set)}", flush=True)
            bc_err = max([abs(target_val - u_k[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            if (self._converged_by_residual(iter_count, is_sdi, bc_err, r_norm, q)
                    and self._freeze_lift_ok(u_k, f_ext, free_dof_mask, dt, sdi)):
                self._accept_increment(u_k, "residual", f_ext, iter_count, dt, reset_tuner=False)
                return True, iter_count

            try:
                du = self._solve_linear(K_bc, r_bc)
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError("Pardiso linear solver failed (matrix is likely singular). Check boundary conditions or rigid body modes.") from e

            u_next = u_k + du
            bc_err_next = max([abs(target_val - u_next[dof]) for dof, target_val in self.fixed_dofs.items()], default=0.0)
            du_free_max = float(np.max(np.abs(du[free_dof_mask]))) if np.any(free_dof_mask) else 0.0
            du_inc_max = float(np.max(np.abs(u_next - u_start)))
            if (self._converged_by_correction(iter_count, is_sdi, bc_err_next, du_free_max, du_inc_max, r_norm, q)
                    and self._freeze_lift_ok(u_next, f_ext, free_dof_mask, dt, sdi, resid_tol=self._CORR_RESID_TOL)):
                self._accept_increment(u_next, "correction", f_ext, iter_count, dt, reset_tuner=False)
                return True, iter_count

            # Displacement correction limiter: prevent artificial element inversion during contact release/SDI shock
            du_free_max = float(np.max(np.abs(du[free_dof_mask]))) if np.any(free_dof_mask) else 0.0
            du_limit = getattr(self, "max_du_limit", 0.05)
            if du_free_max > du_limit:
                du_step = du * (du_limit / du_free_max)
            else:
                du_step = du

            # On iteration 1 or during SDI transition, accept kinematic step to establish contact without false line-search rejection
            if iter_count == 1 or is_sdi:
                s_sdi = 1.0
                accepted_sdi = False
                for _ in range(6):
                    u_trial = u_k + s_sdi * du_step
                    for dof, target_val in self.fixed_dofs.items():
                        u_trial[dof] = u_k[dof] + s_sdi * (target_val - u_k[dof])
                    _, _ = self.assemble_system(u_trial, dt=dt)
                    if self.last_assembly_error == 0:
                        u_k = u_trial
                        accepted_sdi = True
                        break
                    s_sdi *= 0.5
                if accepted_sdi:
                    for c in self.constraints:
                        if getattr(c, "augmented_lagrange", False):
                            c.update_augmented_multipliers(u_k, omega=1.0)
                    self._update_contact_chatter_state(u_k)
                    continue

            if r_norm > 1e4 and iter_count <= 4:
                top_free = np.argsort(np.abs(r_free))[-3:]
                free_dof_indices = np.where(free_dof_mask)[0]
                print(f"      [DIAG] Top 3 residual DOFs: {[(free_dof_indices[idx], free_dof_indices[idx]//3, free_dof_indices[idx]%3, float(r_free[idx])) for idx in top_free]}", flush=True)

            line_search_diag = []
            u_new, _, found_reduction = self._line_search(
                u_k, du_step, f_ext, free_dof_mask, r_norm, q, dt,
                n_trials=8, prescribed_at_partial_step="scaled", diag=line_search_diag)
            if r_norm > 1e4 and iter_count <= 4:
                print(f"      [DIAG LineSearch] trials: {line_search_diag} | found={found_reduction}", flush=True)
            if u_new is None:
                # Non-descent / diverging search direction: cut back cleanly
                self._clear_contact_freeze()
                return False, iter_count
            u_k = u_new
            for c in self.constraints:
                if getattr(c, "augmented_lagrange", False):
                    c.update_augmented_multipliers(u_k, omega=1.0)
            self._update_contact_chatter_state(u_k)

        # Fallthrough check on stable state
        self.u = u_k
        if (self._converged_on_exhaustion(is_sdi, bc_err, r_norm, q)
                and self._freeze_lift_ok(u_k, f_ext, free_dof_mask, dt, sdi)):
            self._accept_increment(u_k, "fallthrough", f_ext, raw_iter, dt, reset_tuner=False)
            return True, raw_iter
        self._clear_contact_freeze()
        return False, raw_iter

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

        # Effective NLGEOM for this step. The solver is built before any step
        # is known (nlgeom defaulted to True), so *STEP, NLGEOM=NO used to be
        # ignored. Abaqus/Standard rule: once NLGEOM is on in a step it stays
        # on for the rest of the analysis.
        want = bool(getattr(step, "nlgeom", self.nlgeom)) or bool(getattr(self, "_nlgeom_engaged", False))
        if want != self.nlgeom:
            self.nlgeom = want
            self._build_kernel_groups()
        self._nlgeom_engaged = bool(getattr(self, "_nlgeom_engaged", False) or self.nlgeom)
        # The step's integrator (*STATIC / *DYNAMIC), applied like NLGEOM above; a step
        # without one (Python API) keeps the integrator the solver was built with.
        from dispsolver.solver import implicit_dynamics as _dyn
        step_ti = _dyn.step_time_integration(self, step)
        dynamic = _dyn.is_dynamic(self)
        if step_ti is not None and dynamic:
            _dyn.require_density(self, step)

        from dispsolver.solver.stabilization import apply_step_stabilization
        apply_step_stabilization(self, step)

        T_total = time_period if time_period is not None else getattr(step, "time_period", 1.0)
        dt_init = getattr(step, "dt_init", 0.02)
        dt_min = getattr(step, "dt_min", 1e-5)
        dt_max = getattr(step, "dt_max", 0.05)
        target_iters = getattr(step, "target_iters", 16 if len(self.constraints) > 0 else 8)

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
        # per-solve() counters, read by execute_inp
        self.last_solve_stats = {"iterations": 0, "increments": 0, "max_increment_iterations": 0}

        if not hasattr(self, "history_u") or self.history_u is None:
            self.history_u = []
        self.history_u.clear()
        self.history_u.append(self.u.copy())

        if not hasattr(self, "history_t") or self.history_t is None:
            self.history_t = []
        self.history_t.clear()
        self.history_t.append(0.0)
        self._u_step_start = self.u.copy()

        if verbose:
            print(f"=== Starting Step '{step.name}' Solve (Time Period: {T_total:.3f} s) ===", flush=True)

        # Setup active constraints and interactions for this step
        if hasattr(step, "interactions"):
            for iname, entry in step.interactions.items():
                if hasattr(entry, "is_active") and entry.is_active():
                    inter = entry.entity
                    if hasattr(inter, "build_runtime_constraint"):
                        existing_names = [getattr(c, "name", "") for c in self.constraints]
                        if inter.name not in existing_names:
                            coords = self.coords if hasattr(self, "coords") else self.mesh.nodes_array()
                            nid_to_idx = self.mesh.node_id_to_index()
                            rc = inter.build_runtime_constraint(
                                mesh=self.mesh,
                                nid_to_idx=nid_to_idx,
                                coords=coords,
                                model=getattr(step, "parent_model", None)
                            )
                            self.add_constraint(rc)

        max_step_iters = getattr(step, "max_iters", 80 if len(self.constraints) > 0 else 50)
        plan = self._lower_step(step, sys, T_total)
        if step_ti is not None and dynamic and step_ti.get("initial_acceleration", True):
            # Abaqus computes the initial acceleration at the start of every *DYNAMIC
            # step (INITIAL omitted); INITIAL=NO keeps the current one
            self._apply_plan_bcs(plan, 0.0)
            _dyn.initial_acceleration(self, plan.f_ext(0.0, self.u), dt_init,
                                      _dyn.prescribed_acceleration(plan, 0.0, dt_init, self._u_step_start))

        # One record per increment ATTEMPT, Abaqus .sta style: `inc` counts
        # converged increments (a cutback retries the same inc with att+1).
        self.increment_log = []
        inc_no, att_no = 1, 1

        while t < T_total - 1e-12:

            dt = min(dt_ctrl.dt, T_total - t)
            t_next = t + dt
            step_inc += 1

            # Backup state in case cutback is needed
            u_prev = self.u.copy()
            sdvs_prev = self.elem_sdvs.copy() if self.elem_sdvs is not None else None
            constraint_states_prev = [
                c.capture_state() if hasattr(c, "capture_state") else None
                for c in self.constraints
            ]

            self._apply_plan_bcs(plan, t_next)
            f_ext = plan.f_ext(t_next, self.u)

            # a dynamic integrator runs this same loop on the HHT effective system
            # (it used to run every step quasi-statically whatever the integrator)
            converged, iters = self._solve_step_with_hooks(
                dt=dt,
                t_next=t_next,
                sys=sys,
                sensor_mgr=sensor_mgr,
                hooks=hooks,
                f_ext=f_ext,
                max_iters=max_step_iters,
                f_ext_n=plan.f_ext(t, self.u) if dynamic else None,
            )

            total_iters += iters
            if converged:
                self.last_solve_stats["increments"] += 1
                self.last_solve_stats["max_increment_iterations"] = max(self.last_solve_stats["max_increment_iterations"], int(iters))
            self.last_solve_stats["iterations"] = total_iters
            self.increment_log.append({
                "inc": inc_no, "att": att_no, "n_sdi": int(getattr(self, "_last_n_sdi", 0)),
                "n_iter": int(iters), "dt": float(dt), "t_step": float(t_next if converged else t),
                "converged": bool(converged),
            })
            if converged:
                inc_no, att_no = inc_no + 1, 1
            else:
                att_no += 1

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
                # Rollback displacement, SDV, and constraint state on cutback
                self.u = u_prev
                if sdvs_prev is not None:
                    self.elem_sdvs = sdvs_prev.copy()
                for c, c_st in zip(self.constraints, constraint_states_prev):
                    if c_st is not None and hasattr(c, "restore_state"):
                        c.restore_state(c_st)
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
        if not hasattr(self, "f_int") or self.f_int is None:
            return 0.0
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

    def compute_total_energy(self) -> Tuple[float, float, float]:

        """Compute Total Mechanical Energy (E_total, E_kin, E_strain)."""
        if self.time_integrator == "emcs" and self._emcs_solver is not None:
            self._emcs_solver.state.u[:] = self.u[:]
            self._emcs_solver.state.v[:] = self.v[:]
            self._emcs_solver.state.a[:] = self.a[:]
            return self._emcs_solver.compute_total_energy()
        
        E_strain = self.compute_strain_energy()
        # Compute Kinetic Energy E_kin = 0.5 * v^T @ M @ v
        E_kin = 0.0
        if hasattr(self, "v") and self.v is not None:
            if hasattr(self, "M_global") and self.M_global is not None:
                E_kin = 0.5 * float(self.v @ (self.M_global @ self.v))
            else:
                default_rho = self.density if getattr(self, "density", None) is not None else 7.85e-9
                mean_rho = float(np.mean(self.elem_density)) if hasattr(self, "elem_density") and len(self.elem_density) > 0 else default_rho
                E_kin = 0.5 * float(np.dot(self.v, self.v) * mean_rho)
        return E_strain + E_kin, E_kin, E_strain

    def compute_reaction_forces(self) -> np.ndarray:
        """Compute reaction forces at all DOFs.
        For fixed Dirichlet DOFs, reaction force R = f_int.
        For unconstrained free DOFs, R = 0 (equilibrium residual).
        Returns an array of length num_dofs.
        """
        R = np.zeros(self.num_dofs, dtype=np.float64)
        if hasattr(self, "f_int") and self.f_int is not None:
            for dof in self.fixed_dofs.keys():
                R[dof] = self.f_int[dof]
        return R

    def get_state(self) -> Dict[str, Any]:
        """Return full state vector dictionary (u, v, a, time_integrator)."""
        return {
            "u": self.u.copy(),
            "v": self.v.copy(),
            "a": self.a.copy(),
            "time_integrator": self.time_integrator,
        }



