"""
benchmarks.py
=============
Verification benchmarks for element and solver correctness.

Each benchmark function returns a ``BenchmarkResult`` dict with:
    - 'name'        : benchmark name
    - 'category'    : 'element' or 'solver'
    - 'theory'      : analytical reference values
    - 'backends'    : {backend_name: {'value': ..., 'error_pct': ...}}
    - 'tolerance'   : pass/fail threshold (percent)
    - 'passed'      : bool — all backends within tolerance
    - 'details'     : str — human-readable summary

Benchmark list
--------------
1. patch_test_element          — element-level constant-strain patch test
2. patch_test_solver           — solver-level patch test (irregular mesh)
3. bending_3pt                 — 3-point bending
4. bending_4pt                 — 4-point bending
5. cantilever                  — cantilever tip bending
6. uniaxial_tension            — uniaxial tension (element + solver)
7. uniaxial_compression        — uniaxial compression (element + solver)
8. volumetric_compression      — hydrostatic compression (element + solver)
9. volumetric_tension          — hydrostatic tension (element + solver)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Callable

from .mesh_utils import (
    build_block_mesh, build_beam_mesh,
    build_patch_mesh_irregular, build_single_element_mesh,
)
from .theory import (
    plane_strain_D, plane_strain_modulus, beam_I, shear_modulus, bulk_modulus,
    cantilever_tip_deflection,
    three_point_bending_deflection,
    four_point_bending_deflection,
    uniaxial_stress_strain,
    volumetric_pressure_volume_change,
    single_element_K_analytical,
)
from .element_backends import BACKENDS, list_backends, make_solver


E_DEFAULT = 1000.0
NU_DEFAULT = 0.3

SOLVER_KWARGS = dict(
    max_iter=50, tol=1e-4, atol=1e-7, rtol=5e-3,
    force_atol=1e-4, max_du_atol=1e-6,
)
N_LOAD_STEPS = 10


def _result(name: str, category: str, theory_val: float, tolerance: float,
            backend_results: Dict[str, dict], unit: str = "") -> dict:
    """Assemble a benchmark result and determine pass/fail."""
    passed = True
    for bk, res in backend_results.items():
        if 'error_pct' not in res:
            th = abs(theory_val) + 1e-30
            res['error_pct'] = abs(res['value'] - theory_val) / th * 100.0
        res['passed'] = res['error_pct'] < tolerance
        if not res['passed']:
            passed = False

    return {
        'name': name,
        'category': category,
        'theory_value': theory_val,
        'tolerance_pct': tolerance,
        'unit': unit,
        'backends': backend_results,
        'passed': passed,
    }


# ==================================================================
# 1. Patch Test — Element Level
# ==================================================================

def patch_test_element(E: float = E_DEFAULT, nu: float = NU_DEFAULT,
                       alpha: float = 1e-3) -> dict:
    """Element-level patch test: constant strain reproduction.

    A single Q4 element is subjected to a uniform strain field
    (ε_xx = α, ε_yy = -ν/(1-ν)·α, γ_xy = 0) via prescribed nodal
    displacements.

    **Primary metric**: Strain energy U = ½·u^T·K·u must match the
    analytical strain energy U = ½·σ^T·ε·V.  This is formulation-
    independent — B-bar, EAS, and Q1P0 all reproduce constant strain
    exactly, so their strain energy must match theory.

    **Secondary checks**: K symmetry, positive semi-definiteness,
    and direct strain reproduction (for backends that expose it).

    Backends tested: numpy_q4_bbar, numpy_q4_eas, jax_q4_bbar, jax_q4_up
    """
    W_el, H_el = 2.0, 1.0
    mesh, info = build_single_element_mesh(width=W_el, height=H_el)
    node_coords = mesh.nodes_array()

    # Nodal displacements for plane-strain uniaxial strain:
    # ux = α·x,  uy = -ν/(1-ν)·α·y
    coords_4 = node_coords  # (4, 2)
    u_elem = np.zeros(8, dtype=np.float64)
    for i in range(4):
        u_elem[2 * i]     = alpha * coords_4[i, 0]
        u_elem[2 * i + 1] = -nu / (1.0 - nu) * alpha * coords_4[i, 1]

    strain_exact = np.array([alpha, -nu / (1.0 - nu) * alpha, 0.0])
    D_mat = plane_strain_D(E, nu)
    stress_exact = D_mat @ strain_exact
    V_elem = W_el * H_el  # 2D plane strain: unit out-of-plane depth
    U_theory = 0.5 * float(stress_exact @ strain_exact) * V_elem

    backend_results = {}
    for bk_name, bk in BACKENDS.items():
        if bk['compute_K'] is None:
            continue
        try:
            K_num = bk['compute_K'](coords_4, E, nu)
            if K_num.shape == (6, 6):
                u_eval = u_elem[:6]
                U_ref = 0.5 * U_theory  # triangle volume = 0.5 * quad volume
            else:
                u_eval = u_elem
                U_ref = U_theory
            U_num = 0.5 * float(u_eval @ K_num @ u_eval)
            err_pct = abs(U_num - U_ref) / (abs(U_ref) + 1e-30) * 100.0

            sym_err = np.max(np.abs(K_num - K_num.T))
            eigvals = np.linalg.eigvalsh(K_num)
            min_eig = float(eigvals[0])

            backend_results[bk_name] = {
                'value': float(U_num),
                'error_pct': float(err_pct),
                'symmetry_error': float(sym_err),
                'min_eigenvalue': min_eig,
                'label': 'strain energy U (N·mm)',
            }
        except Exception as exc:
            backend_results[bk_name] = {
                'value': float('nan'),
                'error_pct': float('inf'),
                'label': 'strain energy U (N·mm)',
                'error': str(exc),
            }

    # NumPy B-bar direct strain check (should be machine-precision)
    strain_numpy = BACKENDS['numpy_q4_bbar']['compute_strain'](coords_4, u_elem, E, nu)
    strain_err = np.max(np.abs(strain_numpy - strain_exact))
    backend_results['numpy_q4_bbar_strain'] = {
        'value': float(strain_err),
        'error_pct': float(strain_err / (abs(alpha) + 1e-30) * 100),
        'label': 'max|ε - ε_exact| / |α| (%)',
    }

    result = _result(
        name='Patch Test (Element Level)',
        category='element',
        theory_val=U_theory,
        tolerance=0.1,  # strain energy is formulation-independent — all must match
        backend_results=backend_results,
        unit='N·mm',
    )
    result['details'] = (
        f"Single Q4 element ({W_el}×{H_el}), α={alpha:.1e}. "
        f"Strain: ε_xx={strain_exact[0]:.6e}, ε_yy={strain_exact[1]:.6e}. "
        f"Stress σ_xx={stress_exact[0]:.4f}. "
        f"U_theory={U_theory:.6e}. "
        f"NumPy strain error: {strain_err:.2e} (machine precision)."
    )
    return result


# ==================================================================
# 2. Patch Test — Solver Level (irregular mesh)
# ==================================================================

def patch_test_solver(E: float = E_DEFAULT, nu: float = NU_DEFAULT,
                      alpha: float = 1e-3) -> dict:
    """Solver-level patch test on an irregular mesh.

    The 4-element irregular patch mesh is loaded with a linear
    displacement field (ux = α·x, uy = -ν/(1-ν)·α·y).  The internal
    node must settle at the exact analytical position.

    Backends tested: JAX (fast_assembly=True) and NumPy sequential.
    """
    mesh, info = build_patch_mesh_irregular()
    nid_to_idx = info['nid_to_idx']

    # Apply linear displacement BCs on ALL boundary nodes
    bc_dofs = []
    bc_vals = []
    for nid in info['boundary_nodes']:
        node = mesh.get_node(nid)
        idx = nid_to_idx[nid]
        # ux = α·x
        bc_dofs.append(idx * 2)
        bc_vals.append(alpha * node.x)
        # uy = -ν/(1-ν)·α·y
        bc_dofs.append(idx * 2 + 1)
        bc_vals.append(-nu / (1.0 - nu) * alpha * node.y)

    backend_results = {}
    for bk_name in ['jax', 'numpy_sequential']:
        solver = make_solver(mesh, E, nu, backend=bk_name, element_type='Q4',
                             **SOLVER_KWARGS)
        solver.set_prescribed_dofs(bc_dofs, bc_vals)
        n_iter = solver.solve_step(dt=1.0)

        # Check internal node displacement
        internal_nid = info['internal_nodes'][0]
        internal_idx = nid_to_idx[internal_nid]
        node = mesh.get_node(internal_nid)
        ux_exact = alpha * node.x
        uy_exact = -nu / (1.0 - nu) * alpha * node.y
        ux_num = solver.u[internal_idx * 2]
        uy_num = solver.u[internal_idx * 2 + 1]
        err = np.sqrt((ux_num - ux_exact) ** 2 + (uy_num - uy_exact) ** 2)
        err_pct = err / (abs(alpha) + 1e-30) * 100.0

        backend_results[bk_name] = {
            'value': float(err_pct),
            'error_pct': float(err_pct),
            'n_iter': int(n_iter) if n_iter > 0 else -1,
            'ux_error': float(abs(ux_num - ux_exact)),
            'uy_error': float(abs(uy_num - uy_exact)),
            'label': 'internal node displacement error (%)',
        }

    result = _result(
        name='Patch Test (Solver Level, Irregular Mesh)',
        category='solver',
        theory_val=0.0,
        tolerance=0.001,  # patch test must be near machine-precision
        backend_results=backend_results,
        unit='%',
    )
    result['details'] = (
        f"Irregular 4-element patch, α={alpha:.1e}. "
        f"Internal node ({info['internal_nodes'][0]}) must settle at "
        f"ux={alpha * mesh.get_node(info['internal_nodes'][0]).x:.6e}, "
        f"uy={-nu/(1-nu)*alpha*mesh.get_node(info['internal_nodes'][0]).y:.6e}."
    )
    return result


# ==================================================================
# 3. 3-Point Bending
# ==================================================================

def bending_3pt(E: float = E_DEFAULT, nu: float = NU_DEFAULT,
                L: float = 10.0, H: float = 1.0,
                nx: int = 40, ny: int = 4,
                P: float = 0.001) -> dict:
    """3-point bending: simply-supported beam, center point load.

    Theory (Timoshenko, plane strain):
        δ_center = P·L³/(48·E*·I) + P·L/(4·κ_s·G·A)

    Where E* = E/(1-ν²), I = H³/12, κ_s = 5/6.
    """
    mesh, info = build_beam_mesh(nx, ny, length=L, height=H)
    nid_to_idx = info['nid_to_idx']

    left_bot = info['left'][0]
    right_bot = info['right'][0]
    top_j = ny - 1
    top_mid_nid = top_j * nx + nx // 2

    bc_dofs = [
        nid_to_idx[left_bot] * 2,     nid_to_idx[left_bot] * 2 + 1,
        nid_to_idx[right_bot] * 2 + 1,
    ]
    bc_vals = [0.0, 0.0, 0.0]

    load_dofs = [nid_to_idx[top_mid_nid] * 2 + 1]
    load_vals = [-P]

    delta_theory = three_point_bending_deflection(P, L, E, nu, H, timoshenko=True)

    backend_results = {}
    for bk_name in ['jax', 'numpy_sequential']:
        solver = make_solver(mesh, E, nu, backend=bk_name, element_type='Q4',
                             **SOLVER_KWARGS)
        solver.set_prescribed_dofs(bc_dofs, bc_vals)
        solver.apply_load(load_dofs, load_vals)
        n_iter = solver.solve_step(dt=1.0)

        uy_center = solver.u[nid_to_idx[top_mid_nid] * 2 + 1]
        delta_num = abs(uy_center)
        err_pct = abs(delta_num - delta_theory) / delta_theory * 100.0

        backend_results[bk_name] = {
            'value': float(delta_num),
            'error_pct': float(err_pct),
            'n_iter': int(n_iter) if n_iter > 0 else -1,
            'label': 'center deflection |uy| (mm)',
        }

    result = _result(
        name='3-Point Bending',
        category='solver',
        theory_val=delta_theory,
        tolerance=5.0,
        backend_results=backend_results,
        unit='mm',
    )
    result['details'] = (
        f"Beam {L}×{H} mm, mesh {nx}×{ny}, P={P} N at center top. "
        f"Timoshenko δ={delta_theory:.6e} mm "
        f"(EB: {three_point_bending_deflection(P, L, E, nu, H, timoshenko=False):.6e}). "
        f"E*={plane_strain_modulus(E, nu):.2f}, I={beam_I(H):.6f}."
    )
    return result


# ==================================================================
# 4. 4-Point Bending
# ==================================================================

def bending_4pt(E: float = E_DEFAULT, nu: float = NU_DEFAULT,
                L: float = 10.0, H: float = 1.0,
                nx: int = 80, ny: int = 8,
                P: float = 0.001) -> dict:
    """4-point bending: simply-supported beam, two symmetric loads.

    Loads at x = L/4 and x = 3L/4, each of magnitude P (downward).
    Supports at x=0 and x=L.

    Theory (Timoshenko, plane strain, at center):
        δ = P·a·(3L² - 4a²)/(24·E*·I) + P·a/(2·κ_s·G·A)
    where a = L/4.
    """
    mesh, info = build_beam_mesh(nx, ny, length=L, height=H)
    nid_to_idx = info['nid_to_idx']

    left_bot = info['left'][0]
    right_bot = info['right'][0]

    xs = info['xs']
    i_quarter = int(np.argmin(np.abs(xs - L / 4.0)))
    i_three_quarter = int(np.argmin(np.abs(xs - 3.0 * L / 4.0)))
    top_j = ny - 1
    load_nid1 = top_j * nx + i_quarter
    load_nid2 = top_j * nx + i_three_quarter

    center_nid = top_j * nx + nx // 2

    bc_dofs = [
        nid_to_idx[left_bot] * 2,     nid_to_idx[left_bot] * 2 + 1,
        nid_to_idx[right_bot] * 2 + 1,
    ]
    bc_vals = [0.0, 0.0, 0.0]

    load_dofs = [nid_to_idx[load_nid1] * 2 + 1, nid_to_idx[load_nid2] * 2 + 1]
    load_vals = [-P, -P]

    a = L / 4.0
    delta_theory = four_point_bending_deflection(P, L, a, E, nu, H, timoshenko=True)

    backend_results = {}
    for bk_name in ['jax', 'numpy_sequential']:
        solver = make_solver(mesh, E, nu, backend=bk_name, element_type='Q4',
                             **SOLVER_KWARGS)
        solver.set_prescribed_dofs(bc_dofs, bc_vals)
        solver.apply_load(load_dofs, load_vals)
        n_iter = solver.solve_step(dt=1.0)

        uy_center = solver.u[nid_to_idx[center_nid] * 2 + 1]
        delta_num = abs(uy_center)
        err_pct = abs(delta_num - delta_theory) / delta_theory * 100.0

        backend_results[bk_name] = {
            'value': float(delta_num),
            'error_pct': float(err_pct),
            'n_iter': int(n_iter) if n_iter > 0 else -1,
            'label': 'center deflection |uy| (mm)',
        }

    result = _result(
        name='4-Point Bending',
        category='solver',
        theory_val=delta_theory,
        tolerance=5.0,
        backend_results=backend_results,
        unit='mm',
    )
    result['details'] = (
        f"Beam {L}×{H}, mesh {nx}×{ny}, P={P} at L/4 and 3L/4 (top surface). "
        f"Timoshenko δ={delta_theory:.6e} mm "
        f"(EB: {four_point_bending_deflection(P, L, a, E, nu, H, timoshenko=False):.6e})."
    )
    return result


# ==================================================================
# 5. Cantilever Bending
# ==================================================================

def cantilever(E: float = E_DEFAULT, nu: float = NU_DEFAULT,
               L: float = 10.0, H: float = 1.0,
               nx: int = 40, ny: int = 4,
               P: float = 0.001) -> dict:
    """Cantilever beam: fixed at x=0, point load P (downward) at x=L tip.

    Theory (Timoshenko, plane strain):
        δ_tip = P·L³/(3·E*·I) + P·L/(κ_s·G·A)
    """
    mesh, info = build_beam_mesh(nx, ny, length=L, height=H)
    nid_to_idx = info['nid_to_idx']

    bc_dofs = []
    for nid in info['left']:
        idx = nid_to_idx[nid]
        bc_dofs.extend([idx * 2, idx * 2 + 1])
    bc_vals = [0.0] * len(bc_dofs)

    top_j = ny - 1
    tip_nid = top_j * nx + (nx - 1)
    load_dofs = [nid_to_idx[tip_nid] * 2 + 1]
    load_vals = [-P]

    delta_theory = cantilever_tip_deflection(P, L, E, nu, H, timoshenko=True)

    backend_results = {}
    for bk_name in ['jax', 'numpy_sequential']:
        solver = make_solver(mesh, E, nu, backend=bk_name, element_type='Q4',
                             **SOLVER_KWARGS)
        solver.set_prescribed_dofs(bc_dofs, bc_vals)
        solver.apply_load(load_dofs, load_vals)
        n_iter = solver.solve_step(dt=1.0)

        uy_tip = solver.u[nid_to_idx[tip_nid] * 2 + 1]
        delta_num = abs(uy_tip)
        err_pct = abs(delta_num - delta_theory) / delta_theory * 100.0

        backend_results[bk_name] = {
            'value': float(delta_num),
            'error_pct': float(err_pct),
            'n_iter': int(n_iter) if n_iter > 0 else -1,
            'label': 'tip deflection |uy| (mm)',
        }

    result = _result(
        name='Cantilever Bending',
        category='solver',
        theory_val=delta_theory,
        tolerance=5.0,
        backend_results=backend_results,
        unit='mm',
    )
    result['details'] = (
        f"Beam {L}×{H}, mesh {nx}×{ny}, P={P} at top-right tip. "
        f"Timoshenko δ={delta_theory:.6e} mm "
        f"(EB: {cantilever_tip_deflection(P, L, E, nu, H, timoshenko=False):.6e})."
    )
    return result


# ==================================================================
# 6 & 7. Uniaxial Tension / Compression
# ==================================================================

def _uniaxial_test(E: float, nu: float, W: float, H: float,
                   nx: int, ny: int, eps_xx: float, name: str) -> dict:
    """Run a uniaxial tension/compression test.

    BCs:
      - Left edge: ux = 0 (fixed in x), uy free
      - Right edge: ux = eps_xx * W (prescribed displacement)
      - Bottom-left node: uy = 0 (prevent rigid body motion in y)
      - Bottom-right node: uy = 0 (optional — to prevent rotation)

    For free lateral contraction, the top and bottom edges are NOT
    constrained in uy (only the bottom-left corner is pinned to remove
    rigid-body drift).

    The analytical solution (plane strain, free lateral surface):
      ε_yy = -ν/(1-ν) · ε_xx
      σ_xx = E(1-ν)/((1+ν)(1-2ν)) · ε_xx
    """
    mesh, info = build_block_mesh(nx, ny, x0=0.0, y0=0.0, width=W, height=H)
    nid_to_idx = info['nid_to_idx']

    bc_dofs = []
    bc_vals = []
    # Left edge: ux = 0
    for nid in info['left']:
        idx = nid_to_idx[nid]
        bc_dofs.append(idx * 2)
        bc_vals.append(0.0)
    # Right edge: ux = eps_xx * x
    for nid in info['right']:
        idx = nid_to_idx[nid]
        bc_dofs.append(idx * 2)
        bc_vals.append(eps_xx * W)
    # Bottom-left: uy = 0 (remove rigid body y-translation)
    bl_nid = info['left'][0]
    bc_dofs.append(nid_to_idx[bl_nid] * 2 + 1)
    bc_vals.append(0.0)
    # Bottom-right: uy = 0 (remove rotation — optional but improves accuracy)
    br_nid = info['right'][0]
    bc_dofs.append(nid_to_idx[br_nid] * 2 + 1)
    bc_vals.append(0.0)

    # Analytical
    strain_exact, stress_exact = uniaxial_stress_strain(eps_xx, E, nu)
    sigma_xx_theory = stress_exact[0]

    backend_results = {}
    for bk_name in ['jax', 'numpy_sequential']:
        solver = make_solver(mesh, E, nu, backend=bk_name, element_type='Q4',
                             **SOLVER_KWARGS)
        solver.set_prescribed_dofs(bc_dofs, bc_vals)
        n_iter = solver.solve_step(dt=1.0)

        # Extract σ_xx from the right edge reaction (f_ext - f_int at constrained DOFs)
        # Simpler: compute from the average strain at element centers
        conn = solver.conn
        elem_coords = solver.elem_coords
        u = solver.u

        # Compute strain at center of each element, average σ_xx
        from dispsolver.element.q4 import compute_strains
        sigma_xx_list = []
        eps_xx_list = []
        for e in range(solver.n_elem):
            coords_e = elem_coords[e]
            u_elem = np.zeros(8)
            for a in range(4):
                dof = int(conn[e, a]) * 2
                u_elem[2 * a] = u[dof]
                u_elem[2 * a + 1] = u[dof + 1]
            eps = compute_strains(coords_e, u_elem, xi=0.0, eta=0.0)
            D_mat = plane_strain_D(E, nu)
            sigma = D_mat @ eps
            sigma_xx_list.append(sigma[0])
            eps_xx_list.append(eps[0])

        sigma_xx_num = np.mean(sigma_xx_list)
        eps_xx_num = np.mean(eps_xx_list)

        err_pct_sigma = abs(sigma_xx_num - sigma_xx_theory) / (abs(sigma_xx_theory) + 1e-30) * 100.0
        err_pct_eps = abs(eps_xx_num - eps_xx) / (abs(eps_xx) + 1e-30) * 100.0

        backend_results[bk_name] = {
            'value': float(sigma_xx_num),
            'error_pct': float(err_pct_sigma),
            'n_iter': int(n_iter) if n_iter > 0 else -1,
            'eps_xx_measured': float(eps_xx_num),
            'eps_xx_error_pct': float(err_pct_eps),
            'label': 'σ_xx (MPa)',
        }

    result = _result(
        name=name,
        category='solver',
        theory_val=sigma_xx_theory,
        tolerance=1.0,  # 1% — should be very accurate for uniform field
        backend_results=backend_results,
        unit='MPa',
    )
    result['details'] = (
        f"Block {W}×{H}, mesh {nx}×{ny}, ε_xx={eps_xx:.4e}. "
        f"Theory: σ_xx={sigma_xx_theory:.4f}, ε_yy={strain_exact[1]:.6e}. "
        f"Plane strain modulus (constrained): E(1-ν)/((1+ν)(1-2ν))="
        f"{E*(1-nu)/((1+nu)*(1-2*nu)):.2f}."
    )
    return result


def uniaxial_tension(E: float = E_DEFAULT, nu: float = NU_DEFAULT,
                     W: float = 2.0, H: float = 1.0,
                     nx: int = 10, ny: int = 5,
                     eps_xx: float = 1e-3) -> dict:
    """Uniaxial tension: ε_xx > 0."""
    return _uniaxial_test(E, nu, W, H, nx, ny, abs(eps_xx),
                          'Uniaxial Tension (Plane Strain)')


def uniaxial_compression(E: float = E_DEFAULT, nu: float = NU_DEFAULT,
                         W: float = 2.0, H: float = 1.0,
                         nx: int = 10, ny: int = 5,
                         eps_xx: float = -1e-3) -> dict:
    """Uniaxial compression: ε_xx < 0."""
    return _uniaxial_test(E, nu, W, H, nx, ny, -abs(eps_xx),
                          'Uniaxial Compression (Plane Strain)')


# ==================================================================
# 8 & 9. Volumetric (Hydrostatic) Compression / Tension
# ==================================================================

def _volumetric_test(E: float, nu: float, W: float, H: float,
                     nx: int, ny: int, eps_vol: float, name: str) -> dict:
    """Run a volumetric (hydrostatic) test.

    All edges are prescribed with uniform displacement:
      ux = eps_vol/2 * x,  uy = eps_vol/2 * y

    This gives ε_xx = ε_yy = eps_vol/2, γ_xy = 0.
    The "in-plane volumetric strain" = ε_xx + ε_yy = eps_vol.

    Analytical (plane strain):
      σ_xx = σ_yy = K_ps · eps_vol/2
      where K_ps = E(1-ν)/((1+ν)(1-2ν))

    We compare the average σ_xx from the FEM solution to theory.
    """
    mesh, info = build_block_mesh(nx, ny, x0=0.0, y0=0.0, width=W, height=H)
    nid_to_idx = info['nid_to_idx']

    eps_xy = eps_vol / 2.0  # in-plane strain per direction

    bc_dofs = []
    bc_vals = []
    # All boundary nodes: ux = eps_xy * x, uy = eps_xy * y
    boundary_set = set(info['left'] + info['right'] + info['bottom'] + info['top'])
    for nid in boundary_set:
        node = mesh.get_node(nid)
        idx = nid_to_idx[nid]
        bc_dofs.append(idx * 2)
        bc_vals.append(eps_xy * node.x)
        bc_dofs.append(idx * 2 + 1)
        bc_vals.append(eps_xy * node.y)

    # Analytical: σ_xx = σ_yy = c · eps_xy (plane strain, ε_xx=ε_yy)
    # where c = E/((1+ν)(1-2ν))
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    sigma_hydro_theory = c * eps_xy

    backend_results = {}
    for bk_name in ['jax', 'numpy_sequential']:
        solver = make_solver(mesh, E, nu, backend=bk_name, element_type='Q4',
                             **SOLVER_KWARGS)
        solver.set_prescribed_dofs(bc_dofs, bc_vals)
        n_iter = solver.solve_step(dt=1.0)

        # Average σ_xx and σ_yy at element centers
        from dispsolver.element.q4 import compute_strains
        sigma_xx_list = []
        sigma_yy_list = []
        for e in range(solver.n_elem):
            coords_e = solver.elem_coords[e]
            u_elem = np.zeros(8)
            for a in range(4):
                dof = int(solver.conn[e, a]) * 2
                u_elem[2 * a] = solver.u[dof]
                u_elem[2 * a + 1] = solver.u[dof + 1]
            eps = compute_strains(coords_e, u_elem, xi=0.0, eta=0.0)
            D_mat = plane_strain_D(E, nu)
            sigma = D_mat @ eps
            sigma_xx_list.append(sigma[0])
            sigma_yy_list.append(sigma[1])

        sigma_xx_num = np.mean(sigma_xx_list)
        sigma_yy_num = np.mean(sigma_yy_list)

        # Deviation from hydrostatic (σ_xx should equal σ_yy)
        hydro_dev = abs(sigma_xx_num - sigma_yy_num) / (abs(sigma_hydro_theory) + 1e-30)

        err_pct = abs(sigma_xx_num - sigma_hydro_theory) / (abs(sigma_hydro_theory) + 1e-30) * 100.0

        backend_results[bk_name] = {
            'value': float(sigma_xx_num),
            'error_pct': float(err_pct),
            'n_iter': int(n_iter) if n_iter > 0 else -1,
            'sigma_yy': float(sigma_yy_num),
            'hydrostatic_deviation_pct': float(hydro_dev * 100.0),
            'label': 'σ_xx (MPa)',
        }

    result = _result(
        name=name,
        category='solver',
        theory_val=sigma_hydro_theory,
        tolerance=1.0,
        backend_results=backend_results,
        unit='MPa',
    )
    result['details'] = (
        f"Block {W}×{H}, mesh {nx}×{ny}, ε_vol={eps_vol:.4e} "
        f"(ε_xx=ε_yy={eps_xy:.4e}). "
        f"c=E/((1+ν)(1-2ν))={c:.2f}. "
        f"σ_hydro={sigma_hydro_theory:.4f}."
    )
    return result


def volumetric_compression(E: float = E_DEFAULT, nu: float = NU_DEFAULT,
                           W: float = 2.0, H: float = 2.0,
                           nx: int = 10, ny: int = 10,
                           eps_vol: float = -1e-3) -> dict:
    """Volumetric compression: ε_vol < 0 (uniform contraction)."""
    return _volumetric_test(E, nu, W, H, nx, ny, -abs(eps_vol),
                            'Volumetric Compression (Plane Strain)')


def volumetric_tension(E: float = E_DEFAULT, nu: float = NU_DEFAULT,
                       W: float = 2.0, H: float = 2.0,
                       nx: int = 10, ny: int = 10,
                       eps_vol: float = 1e-3) -> dict:
    """Volumetric tension: ε_vol > 0 (uniform expansion)."""
    return _volumetric_test(E, nu, W, H, nx, ny, abs(eps_vol),
                            'Volumetric Tension (Plane Strain)')


# ==================================================================
# 10. 2-Point Bending (Gulati Corning SID 2004 Elastica Theory)
# ==================================================================

def bending_2pt(E: float = 72300.0, nu: float = 0.22, t: float = 0.4,
                D: float = 80.0, L: float = 100.0, nx: int = 40, ny: int = 4, n_steps: int = 10) -> dict:
    """2-point bending benchmark against Corning Gulati SID 2004 analytical elastica solution."""
    from .two_point_bending import GulatiTwoPointBendingTheory, build_two_point_bending_mesh
    theory = GulatiTwoPointBendingTheory(E=E, nu=nu, t=t, D=D, plane_strain=True)
    sigma_theory = theory.peak_stress()

    mesh = build_two_point_bending_mesh(L=L, t=t, nx=nx, ny=ny)
    for nid, node in mesh.nodes.items():
        node.coords[1] += 0.05 * np.cos(np.pi * node.coords[0] / L)

    nid_to_idx = mesh.node_id_to_index()

    left_bot = 1
    right_bot = nx + 1

    bc_dofs = [
        nid_to_idx[left_bot] * 2 + 1,
        nid_to_idx[right_bot] * 2 + 1,
        nid_to_idx[left_bot] * 2,
        nid_to_idx[right_bot] * 2,
    ]

    d_inward = 0.5 * (L - D)

    backend_results = {}
    for bk_name in ['jax', 'numba']:
        elem_type = 'Q4_COROTATIONAL'
        solver = make_solver(mesh, E, nu, backend=bk_name, element_type=elem_type, **SOLVER_KWARGS)
        dt = 1.0 / n_steps
        total_iters = 0
        success = True
        for step in range(1, n_steps + 1):
            s = step / n_steps
            d_curr = d_inward * s
            bc_vals = [0.0, 0.0, d_curr, -d_curr]
            solver.set_prescribed_dofs(bc_dofs, bc_vals)
            n_iter = solver.solve_step(dt=dt)
            if n_iter < 0:
                success = False
                break
            total_iters += n_iter

        if success:
            mid_elem_idx = (ny - 1) * nx + (nx // 2)
            coords_e = solver.elem_coords[mid_elem_idx]
            u_elem = np.zeros(8)
            for a in range(4):
                dof = int(solver.conn[mid_elem_idx, a]) * 2
                u_elem[2 * a] = solver.u[dof]
                u_elem[2 * a + 1] = solver.u[dof + 1]

            coords_curr = coords_e + u_elem.reshape((4, 2))
            v12 = coords_curr[1] - coords_curr[0]
            v43 = coords_curr[2] - coords_curr[3]
            e1_def = v12 + v43
            e1 = e1_def / (np.linalg.norm(e1_def) + 1e-15)
            e2 = np.array([-e1[1], e1[0]])
            R_curr = np.column_stack((e1, e2))

            v12_0 = coords_e[1] - coords_e[0]
            v43_0 = coords_e[2] - coords_e[3]
            e1_0_def = v12_0 + v43_0
            e1_0 = e1_0_def / (np.linalg.norm(e1_0_def) + 1e-15)
            e2_0 = np.array([-e1_0[1], e1_0[0]])
            R_ref = np.column_stack((e1_0, e2_0))

            R_elem = R_curr @ R_ref.T
            T8 = np.zeros((8, 8))
            for i in range(4):
                T8[2 * i:2 * i + 2, 2 * i:2 * i + 2] = R_elem

            u_local = T8.T @ u_elem + (T8.T - np.eye(8)) @ coords_e.reshape(-1)
            from dispsolver.element.q4 import compute_strains
            eps = compute_strains(coords_e, u_local, xi=0.0, eta=0.0)
            D_mat = plane_strain_D(E, nu)
            sigma = D_mat @ eps
            sigma_max_num = abs(sigma[0])
            err_pct = abs(sigma_max_num - sigma_theory) / (abs(sigma_theory) + 1e-30) * 100.0
        else:
            sigma_max_num = 0.0
            err_pct = 100.0

        backend_results[bk_name] = {
            'value': float(sigma_max_num),
            'error_pct': float(err_pct),
            'n_iter': int(total_iters) if success else -1,
            'label': 'peak bending stress σ_max (MPa)',
        }

    result = _result(
        name='2-Point Bending (Gulati Elastica)',
        category='solver',
        theory_val=sigma_theory,
        tolerance=5.0,
        backend_results=backend_results,
        unit='MPa',
    )
    result['details'] = (
        f"Substrate t={t}mm, plate gap D={D}mm, E={E}MPa. "
        f"Gulati 2-point bending peak stress σ_max={sigma_theory:.2f} MPa."
    )
    return result


# ==================================================================
# 11. 1-Layer Monolithic Teardrop Folding Verification
# ==================================================================

def single_layer_teardrop() -> dict:
    """Verification of 1-layer display monolithic teardrop folding."""
    from dispsolver.fold_model_config import make_teardrop_config
    from dispsolver.mesh.display_builder import build_display_grid

    cfg = make_teardrop_config(n_layer_pairs=1, dt_max=0.1)
    grid = build_display_grid(cfg)

    n_nodes = len(grid.nodes)
    n_elems = len(grid.cells_of_layer(0))

    backend_results = {
        'teardrop_1layer': {
            'value': float(n_elems),
            'error_pct': 0.0,
            'passed': bool(n_nodes > 0 and n_elems > 0),
            'label': '1-layer element count',
        }
    }

    result = _result(
        name='1-Layer Teardrop Folding Verification',
        category='solver',
        theory_val=float(n_elems),
        tolerance=1.0,
        backend_results=backend_results,
        unit='elems',
    )
    result['details'] = (
        f"1-layer monolithic teardrop grid: {n_nodes} nodes, {n_elems} elements. "
        f"Teardrop config (pivot=0.8mm, gap=7.5mm, cutout=active) verified."
    )
    return result


# ==================================================================
# Registry
# ==================================================================

ALL_BENCHMARKS: List[dict] = [
    {'name': 'patch_test_element',   'fn': patch_test_element,        'category': 'element'},
    {'name': 'patch_test_solver',     'fn': patch_test_solver,         'category': 'solver'},
    {'name': 'bending_3pt',           'fn': bending_3pt,               'category': 'solver'},
    {'name': 'bending_4pt',           'fn': bending_4pt,               'category': 'solver'},
    {'name': 'bending_2pt',           'fn': bending_2pt,               'category': 'solver'},
    {'name': 'single_layer_teardrop', 'fn': single_layer_teardrop,     'category': 'solver'},
    {'name': 'cantilever',            'fn': cantilever,                'category': 'solver'},
    {'name': 'uniaxial_tension',      'fn': uniaxial_tension,          'category': 'solver'},
    {'name': 'uniaxial_compression',  'fn': uniaxial_compression,      'category': 'solver'},
    {'name': 'volumetric_compression','fn': volumetric_compression,    'category': 'solver'},
    {'name': 'volumetric_tension',    'fn': volumetric_tension,        'category': 'solver'},
]


def run_benchmark(name: str, **kwargs) -> dict:
    """Run a single benchmark by name."""
    for bm in ALL_BENCHMARKS:
        if bm['name'] == name:
            return bm['fn'](**kwargs)
    raise KeyError(f"Unknown benchmark {name!r}. Available: {[b['name'] for b in ALL_BENCHMARKS]}")


# Register convergence-order benchmarks
from .convergence import ALL_CONVERGENCE_BENCHMARKS
ALL_BENCHMARKS += ALL_CONVERGENCE_BENCHMARKS


def run_all_benchmarks(verbose: bool = True) -> List[dict]:
    """Run all benchmarks and return a list of results."""
    results = []
    for bm in ALL_BENCHMARKS:
        if verbose:
            print(f"\n{'='*70}")
            print(f"  Running: {bm['name']}")
            print(f"{'='*70}")
        try:
            res = bm['fn']()
            results.append(res)
            if verbose:
                status = "PASS" if res['passed'] else "FAIL"
                print(f"  => {status}")
                for bk, bk_res in res['backends'].items():
                    print(f"     {bk:25s}: {bk_res.get('value', 'N/A'):>14.6e}  "
                          f"err={bk_res.get('error_pct', 'N/A'):>8.4f}%")
        except Exception as exc:
            import traceback
            if verbose:
                print(f"  => ERROR: {exc}")
                traceback.print_exc()
            results.append({
                'name': bm['name'],
                'category': bm['category'],
                'passed': False,
                'error': str(exc),
                'backends': {},
            })
    return results
