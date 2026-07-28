"""
convergence.py
==============
Convergence-order studies: measure how discretization error scales with
mesh refinement (element size h) and load-step refinement (1/n_steps).

Three benchmarks:
  1. convergence_cantilever_mesh_q4bbar    — small-strain sanity (Timoshenko)
  2. convergence_elastica_mesh_corotational     — large-rotation mesh refinement
  3. convergence_elastica_loadstep_corotational — large-rotation step refinement

All reusing the existing ``make_solver`` / ``build_beam_mesh`` machinery.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from .theory import (
    plane_strain_modulus, beam_I, cantilever_tip_deflection,
    elastica_pure_moment_tip_state,
)
from .mesh_utils import build_beam_mesh
from .element_backends import make_solver


E_DEFAULT = 1000.0
NU_DEFAULT = 0.3

SOLVER_KWARGS = dict(
    max_iter=50, tol=1e-4, atol=1e-7, rtol=5e-3,
    force_atol=1e-4, max_du_atol=1e-6,
)


# ------------------------------------------------------------------
# Convergence order fitting: error ~ C * h^p  =>  log(error) = p*log(h) + log(C)
# ------------------------------------------------------------------

def fit_convergence_order(h_values: np.ndarray,
                          error_values: np.ndarray) -> dict:
    """Fit p and C via log-log linear regression.

    Parameters
    ----------
    h_values : (N,) array of element sizes (or 1/n_steps)
    error_values : (N,) array of corresponding discretization errors

    Returns
    -------
    dict with keys: order, log_C, r_squared, h_values, error_values
    """
    mask = (np.asarray(h_values) > 0) & (np.asarray(error_values) > 0)
    h = np.asarray(h_values, dtype=np.float64)[mask]
    e = np.asarray(error_values, dtype=np.float64)[mask]

    if len(h) < 2:
        return {
            'order': float('nan'), 'log_C': float('nan'),
            'r_squared': float('nan'),
            'h_values': list(h_values), 'error_values': list(error_values),
        }

    log_h = np.log(h)
    log_e = np.log(e)
    coeffs = np.polyfit(log_h, log_e, deg=1)
    p, log_C = coeffs[0], coeffs[1]

    log_e_pred = np.polyval(coeffs, log_h)
    ss_res = np.sum((log_e - log_e_pred) ** 2)
    ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return {
        'order': float(p), 'log_C': float(log_C), 'r_squared': float(r2),
        'h_values': list(h_values), 'error_values': list(error_values),
    }


# ------------------------------------------------------------------
# Work-equivalent nodal forces: pure end moment on a rectangular beam
# ------------------------------------------------------------------

def _consistent_axial_nodal_forces(M: float, H: float, ny: int,
                                   y_nodes: np.ndarray) -> np.ndarray:
    """Work-equivalent axial nodal forces for a pure end moment M.

    Integrates the linear bending stress t(y) = -M·y/I against each
    node's linear shape function.  Returns a self-equilibrated force
    vector (sum = 0).

    Parameters
    ----------
    M : float — applied moment (sign: +CCW = tip-up)
    H : float — beam height
    ny : int — number of node-rows (right-edge nodes = ny)
    y_nodes : (ny,) — y-coordinates of tip-section nodes

    Returns
    -------
    f : (ny,) — axial force per node
    """
    I_val = beam_I(H)
    n_nodes = len(y_nodes)  # = ny (right-edge has ny node rows)
    n_elems = n_nodes - 1   # = ny - 1 elements through thickness
    f = np.zeros(n_nodes)

    for j in range(n_elems):
        y0 = y_nodes[j]
        y1 = y_nodes[j + 1]
        h_e = y1 - y0
        coeff = -M * h_e / I_val
        f[j] += coeff * (y0 / 2.0 + h_e / 6.0)
        f[j + 1] += coeff * (y0 / 2.0 + h_e / 3.0)

    return f


def _tip_axial_direction(u_tip: np.ndarray, coords_tip: np.ndarray,
                         ny: int) -> Tuple[float, float]:
    """Deformed beam-axis direction at the tip.

    The cross-section vector (bottom→top) is rotated -90° to obtain
    the axial direction (perpendicular to the cross-section).

    Returns (dx, dy) unit vector along the deformed beam axis.
    """
    x_def = coords_tip[:, 0] + u_tip[:, 0]
    y_def = coords_tip[:, 1] + u_tip[:, 1]
    cs_dx = x_def[ny - 1] - x_def[0]
    cs_dy = y_def[ny - 1] - y_def[0]
    cs_norm = np.sqrt(cs_dx ** 2 + cs_dy ** 2)

    if cs_norm > 1e-15:
        ax_dx = cs_dy / cs_norm
        ax_dy = -cs_dx / cs_norm
    else:
        ax_dx, ax_dy = 1.0, 0.0
    return float(ax_dx), float(ax_dy)


# ------------------------------------------------------------------
# Solver driver: cantilever under follower pure end moment
# ------------------------------------------------------------------

def _run_pure_moment_case(
    nx: int, ny: int,
    L: float, H: float,
    E: float, nu: float,
    theta_target_deg: float,
    n_steps: int,
    element_type: str = 'Q4_COROTATIONAL',
) -> dict:
    """Run a cantilever under a follower pure end moment.

    Left end fixed.  Moment is applied as a self-equilibrated follower
    force couple at the right tip, ramped linearly over n_steps.

    Returns detailed result dict including tip error, iteration
    count, and a flag recording whether the solver actually used the
    corotational kernel.
    """
    mesh, info = build_beam_mesh(nx, ny, length=L, height=H)
    nid_to_idx = info['nid_to_idx']

    # Left-end fixed BCs
    bc_dofs = []
    bc_vals = []
    for nid in info['left']:
        idx = nid_to_idx[nid]
        bc_dofs.extend([idx * 2, idx * 2 + 1])
        bc_vals.extend([0.0, 0.0])

    backend = 'jax'
    solver = make_solver(mesh, E, nu, backend=backend,
                         element_type=element_type, **SOLVER_KWARGS)
    solver.sta_status = False

    use_mm = getattr(solver, 'use_multi_material_batch', None)
    et_by_pid = getattr(solver, 'element_type_by_pid', None)

    # Tip-section data (reference config)
    tip_nids = info['right']
    tip_indices = [nid_to_idx[nid] for nid in tip_nids]
    coords_tip = np.array([[mesh.get_node(nid).x, mesh.get_node(nid).y]
                           for nid in tip_nids])
    y_tip = coords_tip[:, 1]

    theta_target = np.deg2rad(theta_target_deg)
    E_star = plane_strain_modulus(E, nu)
    I_beam = beam_I(H)
    M_target = theta_target * E_star * I_beam / L
    M_step = M_target / max(n_steps, 1)

    solver.set_prescribed_dofs(bc_dofs, bc_vals)

    total_iter = 0
    for step_i in range(1, n_steps + 1):
        M_current = step_i * M_step

        u_tip_current = np.array([
            [solver.u[idx * 2], solver.u[idx * 2 + 1]]
            for idx in tip_indices
        ])
        ax_dx, ax_dy = _tip_axial_direction(u_tip_current, coords_tip, ny)
        f_axial = _consistent_axial_nodal_forces(M_current, H, ny, y_tip)

        load_dofs = []
        load_vals = []
        for j, nid in enumerate(tip_nids):
            idx = nid_to_idx[nid]
            f_mag = f_axial[j]
            load_dofs.append(idx * 2)
            load_vals.append(f_mag * ax_dx)
            load_dofs.append(idx * 2 + 1)
            load_vals.append(f_mag * ax_dy)

        solver.apply_load(load_dofs, load_vals)
        n_iter = solver.solve_step(dt=1.0)
        if n_iter < 0:
            # Retry with two sub-steps (dt=0.5) if 1-step load increment was too large
            n_iter1 = solver.solve_step(dt=0.5)
            if n_iter1 < 0:
                break
            n_iter2 = solver.solve_step(dt=0.5)
            if n_iter2 < 0:
                break
            n_iter = n_iter1 + n_iter2
        total_iter += int(n_iter)

    # Final tip position (mid-section node)
    u_tip_final = np.array([
        [solver.u[idx * 2], solver.u[idx * 2 + 1]]
        for idx in tip_indices
    ])
    x_def = coords_tip[:, 0] + u_tip_final[:, 0]
    y_def = coords_tip[:, 1] + u_tip_final[:, 1]
    mid_idx = ny // 2
    x_tip_num = float(x_def[mid_idx])
    y_tip_num = float(y_def[mid_idx])

    # Exact solution for the total applied moment
    exact = elastica_pure_moment_tip_state(M_target, L, E_star, I_beam)
    x_exact = float(exact['x_tip'])
    y_exact = float(exact['y_tip'])

    tip_error = np.sqrt((x_tip_num - x_exact) ** 2 +
                        (y_tip_num - y_exact) ** 2) / L

    return {
        'nx': nx, 'ny': ny, 'n_steps': n_steps,
        'L': L, 'H': H,
        'theta_target_deg': theta_target_deg,
        'theta_reached': float(exact['theta']),
        'M_applied': float(M_target),
        'n_iter_total': total_iter,
        'x_tip_num': x_tip_num, 'y_tip_num': y_tip_num,
        'x_tip_exact': x_exact, 'y_tip_exact': y_exact,
        'tip_error': tip_error,
        'use_multi_material_batch': use_mm,
        'element_type_by_pid': et_by_pid,
    }


# ------------------------------------------------------------------
# Convergence benchmark helpers
# ------------------------------------------------------------------

def _convergence_result(name: str, description: str,
                        refinement_label: str,
                        h_values: List[float],
                        error_values: List[float],
                        expected_order: float,
                        order_tol: float = 0.4,
                        r2_min: float = 0.95,
                        details: str = "") -> dict:
    """Assemble a convergence benchmark result dict.

    Pass requires BOTH fitted slope within tolerance AND r² ≥ r2_min.
    """
    fit = fit_convergence_order(np.array(h_values), np.array(error_values))
    order = fit['order']
    r2 = fit['r_squared']

    order_ok = not np.isnan(order) and abs(order - expected_order) <= order_tol
    r2_ok = not np.isnan(r2) and r2 >= r2_min
    passed = order_ok and r2_ok

    return {
        'name': name,
        'category': 'solver',
        'theory_value': expected_order,
        'tolerance_pct': order_tol,
        'unit': '',
        'backends': {
            'order_fit': {
                'value': order,
                'error_pct': 100.0 * abs(order - expected_order) / max(abs(expected_order), 1e-30),
                'passed': passed,
                'fitted_order': order,
                'r_squared': r2,
                'expected_order': expected_order,
                'order_tol': order_tol,
                'r2_min': r2_min,
                'order_within_tol': order_ok,
                'r2_above_min': r2_ok,
                'label': refinement_label,
            }
        },
        'passed': passed,
        'details': details,
        'refinement_levels': list(zip(h_values, error_values)),
        'fit': fit,
    }


# ------------------------------------------------------------------
# Benchmark 1: mesh refinement, small-strain, Q4 B-bar (Timoshenko)
# ------------------------------------------------------------------

def convergence_cantilever_mesh_q4bbar(
    E: float = E_DEFAULT,
    nu: float = NU_DEFAULT,
    L: float = 10.0,
    H: float = 1.0,
    P: float = 0.001,
    ny_values: Tuple[int, ...] = (4, 6, 8, 12),
    elements_per_height: int = 10,
    expected_order: float = 4.0,
    order_tol: float = 1.0,
    r2_min: float = 0.95,
) -> dict:
    """Mesh refinement convergence for Q4 B-bar cantilever.

    Small-strain sanity check: validates the order-fitting machinery
    against a case already trusted from the existing verification suite.

    Uses the finest mesh level's FEM result as the reference (overkill
    approach) instead of Timoshenko theory, because the beam-theory
    model error dominates at fine meshes and would mask the discretization
    convergence order.
    """
    the_ny_values = sorted(set(ny_values)) if isinstance(ny_values, (tuple, list)) else (ny_values,)

    # Step 1: run all meshes, store tip displacement for each
    deltas = {}
    h_map = {}
    for ny in the_ny_values:
        nx = max(ny * elements_per_height, 4)
        h_elem = H / ny

        mesh, info = build_beam_mesh(nx, ny, length=L, height=H)
        nid_to_idx = info['nid_to_idx']

        bc_dofs = []
        bc_vals = []
        for nid in info['left']:
            idx = nid_to_idx[nid]
            bc_dofs.extend([idx * 2, idx * 2 + 1])
            bc_vals.extend([0.0, 0.0])

        top_j = ny - 1
        tip_nid = top_j * nx + (nx - 1)
        load_dofs = [nid_to_idx[tip_nid] * 2 + 1]
        load_vals = [-P]

        solver = make_solver(mesh, E, nu, backend='jax',
                             element_type='Q4', **SOLVER_KWARGS)
        solver.sta_status = False
        solver.set_prescribed_dofs(bc_dofs, bc_vals)
        solver.apply_load(load_dofs, load_vals)
        solver.solve_step(dt=1.0)

        deltas[ny] = abs(solver.u[nid_to_idx[tip_nid] * 2 + 1])
        h_map[ny] = h_elem

    # Step 2: use finest mesh as overkill reference
    ref_ny = the_ny_values[-1]
    ref_delta = deltas[ref_ny]

    h_values = []
    error_values = []
    details_lines = []
    for ny in the_ny_values[:-1]:  # exclude reference level from fit
        h_values.append(h_map[ny])
        err_rel = abs(deltas[ny] - ref_delta) / abs(ref_delta)
        error_values.append(err_rel)
        details_lines.append(
            f"ny={ny:2d} nx={max(ny * elements_per_height, 4):3d}: "
            f"δ={deltas[ny]:.6e} err(vs ny={ref_ny})={err_rel:.2e}"
        )

    return _convergence_result(
        name='Convergence Cantilever Mesh (Q4 B-bar)',
        description='Mesh refinement: cantilever Q4 B-bar, overkill-FEM ref',
        refinement_label='fitted order (mesh)',
        h_values=h_values, error_values=error_values,
        expected_order=expected_order, order_tol=order_tol, r2_min=r2_min,
        details='\n'.join(details_lines),
    )


# ------------------------------------------------------------------
# Benchmark 2: mesh refinement, large-rotation, corotational
# ------------------------------------------------------------------

def convergence_elastica_mesh_corotational(
    E: float = E_DEFAULT,
    nu: float = NU_DEFAULT,
    L: float = 20.0,
    H: float = 1.0,
    theta_target_deg: float = 30.0,
    ny_values: Tuple[int, ...] = (4, 8, 12, 16),
    elements_per_height: int = 10,
    n_steps: int = 20,
    expected_order: float = 2.7,
    order_tol: float = 0.5,
    r2_min: float = 0.95,
) -> dict:
    """Mesh refinement convergence for corotational Q4 under pure moment.

    Large-rotation elastica: constant curvature beam under follower end
    moment.  Sweeps ny at fixed n_steps (fine enough that load-step
    error does not dominate the mesh error).
    """
    h_values = []
    error_values = []
    details_lines = []
    coro_ok = True
    the_ny_values = ny_values if isinstance(ny_values, (tuple, list)) else (ny_values,)

    for ny in the_ny_values:
        nx = max(ny * elements_per_height, 4)
        h_elem = H / ny

        result = _run_pure_moment_case(
            nx, ny, L, H, E, nu, theta_target_deg, n_steps,
            element_type='Q4_COROTATIONAL',
        )

        h_values.append(h_elem)
        error_values.append(result['tip_error'])

        if not result['use_multi_material_batch']:
            coro_ok = False

        details_lines.append(
            f"ny={ny:2d} nx={nx:3d}: tip_err={result['tip_error']:.2e} "
            f"iters={result['n_iter_total']}"
        )

    extra = ""
    if not coro_ok:
        extra = (
            " [WARNING: solver.use_multi_material_batch is False — "
            "corotational kernel may NOT be active!]"
        )

    return _convergence_result(
        name='Convergence Elastica Mesh (Corotational)',
        description='Mesh refinement: pure end moment, Q4_COROTATIONAL, jax',
        refinement_label='fitted order (mesh)',
        h_values=h_values, error_values=error_values,
        expected_order=expected_order, order_tol=order_tol, r2_min=r2_min,
        details='\n'.join(details_lines) + extra,
    )


# ------------------------------------------------------------------
# Benchmark 3: load-step refinement, large-rotation, corotational
# ------------------------------------------------------------------

def convergence_elastica_loadstep_corotational(
    E: float = E_DEFAULT,
    nu: float = NU_DEFAULT,
    L: float = 20.0,
    H: float = 1.0,
    theta_target_deg: float = 30.0,
    nx: int = 80,
    ny: int = 8,
    n_steps_values: Tuple[int, ...] = (2, 4, 8, 16),
    expected_order: float = 2.4,
    order_tol: float = 0.5,
    r2_min: float = 0.85,
) -> dict:
    """Load-step refinement convergence for corotational Q4 under pure moment.

    Sweeps n_steps at fixed mesh resolution.  Uses the finest n_steps
    FEM result as the reference (overkill approach) to cancel the fixed-
    mesh discretization error and isolate the load-discretization error.
    """
    the_n_steps = sorted(set(n_steps_values)) if isinstance(n_steps_values, (tuple, list)) else (n_steps_values,)

    # Step 1: run all load-step levels, store raw tip position
    tip_positions = {}
    iter_counts = {}
    for ns in the_n_steps:
        result = _run_pure_moment_case(
            nx, ny, L, H, E, nu, theta_target_deg, ns,
            element_type='Q4_COROTATIONAL',
        )
        tip_positions[ns] = (result['x_tip_num'], result['y_tip_num'])
        iter_counts[ns] = result['n_iter_total']

    # Step 2: use finest n_steps as overkill reference
    ref_ns = the_n_steps[-1]
    x_ref, y_ref = tip_positions[ref_ns]

    h_values = []
    error_values = []
    details_lines = []
    for ns in the_n_steps[:-1]:  # exclude reference level from fit
        h_step = 1.0 / ns
        x_ns, y_ns = tip_positions[ns]
        err = np.sqrt((x_ns - x_ref) ** 2 + (y_ns - y_ref) ** 2) / L
        h_values.append(h_step)
        error_values.append(err)
        details_lines.append(
            f"n_steps={ns:2d}: tip_err(vs ref={ref_ns})={err:.2e} "
            f"iters={iter_counts[ns]}"
        )

    return _convergence_result(
        name='Convergence Elastica Load-Step (Corotational)',
        description='Load-step ref.: pure end moment, Q4_COROTATIONAL, jax',
        refinement_label='fitted order (load-step)',
        h_values=h_values, error_values=error_values,
        expected_order=expected_order, order_tol=order_tol, r2_min=r2_min,
        details='\n'.join(details_lines),
    )


# ------------------------------------------------------------------
# Registry — imported and appended to ALL_BENCHMARKS in benchmarks.py
# ------------------------------------------------------------------

ALL_CONVERGENCE_BENCHMARKS: List[dict] = [
    {
        'name': 'convergence_cantilever_mesh_q4bbar',
        'fn': convergence_cantilever_mesh_q4bbar,
        'category': 'solver',
    },
    {
        'name': 'convergence_elastica_mesh_corotational',
        'fn': convergence_elastica_mesh_corotational,
        'category': 'solver',
    },
    {
        'name': 'convergence_elastica_loadstep_corotational',
        'fn': convergence_elastica_loadstep_corotational,
        'category': 'solver',
    },
]
