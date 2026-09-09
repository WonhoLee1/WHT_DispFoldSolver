"""
test_two_point_bending.py
=========================
Verification tests for Thin Glass Two-Point Bending based on Corning literature:
Suresh T. Gulati et al., "45.2: Two Point Bending of Thin Glass Substrate",
SID 2004 Digest, pp. 1-2.

Tests:
1. Analytical Jacobi elliptic integral constant and peak stress formula.
2. Exact closed-form elastica loop profile and aspect ratio.
3. FEA large-rotation pure bending of thin glass substrate (E = 72.3 GPa, nu = 0.22, t = 0.4 mm).
4. Element formulation comparison (Q4_COROTATIONAL_SRI vs Q4_COROTATIONAL).
"""

from __future__ import annotations

import numpy as np
import pytest

from verification.two_point_bending import (
    GulatiTwoPointBendingTheory,
    build_two_point_bending_mesh,
)
from dispsolver.material import J2Plasticity
from dispsolver.solver import DynamicSolver


# --------------------------------------------------------------------------
# 1. Analytical Theory Tests
# --------------------------------------------------------------------------

def test_gulati_integral_constant():
    """Verify that the Jacobi elliptic integral constant matches Corning paper (1.198)."""
    theory = GulatiTwoPointBendingTheory(E=72300.0, nu=0.22, t=0.4, D=20.0)
    expected_k = 1.1981402347355916
    assert np.isclose(theory.K_GULATI, expected_k, rtol=1e-6)
    assert np.isclose(round(theory.K_GULATI, 3), 1.198)


def test_gulati_peak_stress_analytical():
    """Verify peak bending stress sigma_max = 1.19814 * [E' * t / (D - t)] for multiple thicknesses."""
    E = 72300.0
    nu = 0.22
    E_eff_plane_strain = E / (1.0 - nu ** 2)

    test_cases = [
        (0.4, 20.0, 1800.0, 1900.0),
        (0.1, 10.0, 900.0, 950.0),
        (0.7, 30.0, 2100.0, 2200.0),
    ]

    for t, D, s_min, s_max in test_cases:
        theory = GulatiTwoPointBendingTheory(E=E, nu=nu, t=t, D=D, plane_strain=True)
        s_peak = theory.peak_stress()
        d_eff = D - t
        expected = theory.K_GULATI * (E_eff_plane_strain * t / d_eff)
        assert np.isclose(s_peak, expected, rtol=1e-9)
        assert s_min <= s_peak <= s_max

        s_psi_10 = theory.peak_stress(psi_deg=10.0)
        assert s_psi_10 < s_peak
        assert np.isclose(s_psi_10, s_peak * np.sqrt(np.cos(np.radians(10.0))), rtol=1e-6)


def test_elastica_profile_geometry():
    """Verify the closed-form elliptic elastica profile dimensions and aspect ratio."""
    D = 20.0
    t = 0.4
    theory = GulatiTwoPointBendingTheory(E=72300.0, nu=0.22, t=t, D=D)
    D_eff = theory.D_eff

    xs, ys = theory.exact_profile(n_points=200)

    assert np.isclose(ys[0], -0.5 * D_eff, atol=1e-3)
    assert np.isclose(ys[-1], 0.5 * D_eff, atol=1e-3)

    apex_idx = len(xs) // 2
    assert np.isclose(ys[apex_idx], 0.0, atol=0.1)

    x_apex_expected = D_eff / theory.K_GULATI
    assert np.isclose(np.max(xs), x_apex_expected, rtol=1e-3)

    ratio = np.max(xs) / (0.5 * D_eff)
    assert np.isclose(ratio, 2.0 / theory.K_GULATI, rtol=1e-3)

    L_loop = theory.loop_arc_length()
    assert 40.0 <= L_loop <= 45.0


# --------------------------------------------------------------------------
# 2. Large-Rotation FEA Pure Bending Tests
# --------------------------------------------------------------------------

def test_thin_glass_large_rotation_pure_bending_sri():
    """Test large-rotation pure circular bending of thin glass using Q4_COROTATIONAL_SRI.

    Solves a beam bent up to 90 degrees total angle.
    Success criteria:
    - Converges 100% without element inversion or cutbacks.
    - Midpoint vertical deflection matches analytical large-deformation elastica theory within < 1.0%.
    """
    L = 20.0
    t = 0.4
    nx = 40
    ny = 2
    mesh = build_two_point_bending_mesh(L, t, nx=nx, ny=ny, pid=1)

    theta_target = np.radians(90.0)
    E = 72300.0
    nu = 0.22
    mat = J2Plasticity(E=E, nu=nu, sigma_y0=1e9, H=0.0)

    solver = DynamicSolver(
        mesh, {1: mat}, rho=2.5e-9,
        mode="quasistatic",
        element_type="Q4_COROTATIONAL_SRI",
        tol=1e-3, rtol=1e-4, max_iter=20,
        verbose=False
    )

    left_nids = [nid for nid, node in mesh.nodes.items() if np.isclose(node.x, -L / 2.0)]
    right_nids = [nid for nid, node in mesh.nodes.items() if np.isclose(node.x, L / 2.0)]

    bc_dofs = []
    for nid in left_nids:
        idx = solver.nid_to_idx[nid]
        bc_dofs.extend([2 * idx, 2 * idx + 1])
    for nid in right_nids:
        idx = solver.nid_to_idx[nid]
        bc_dofs.extend([2 * idx, 2 * idx + 1])
    bc_dofs = np.array(bc_dofs, dtype=np.int32)

    n_steps = 20
    for step in range(1, n_steps + 1):
        th = theta_target * (step / n_steps)
        sinc = np.sin(th) / th if th > 1e-6 else 1.0
        cosc = (1.0 - np.cos(th)) / th if th > 1e-6 else 0.0
        dx_tip = L * (sinc - 1.0)
        dy_tip = L * cosc

        R_rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

        bc_vals = [0.0] * (2 * len(left_nids))
        for nid in right_nids:
            node = mesh.nodes[nid]
            dxy = np.array([0.0, node.y])
            new_xy = np.array([L / 2.0 + dx_tip, dy_tip]) + R_rot @ dxy
            bc_vals.extend([new_xy[0] - node.x, new_xy[1] - node.y])

        solver.set_prescribed_dofs(bc_dofs, bc_vals=np.array(bc_vals, dtype=np.float64))
        res = solver.solve_step(1.0 / n_steps)
        assert res >= 0, f"Step {step} failed to converge: res={res}"

    u = solver.u[:2 * solver.n_nodes].reshape(-1, 2)
    mid_nids = [nid for nid, node in mesh.nodes.items() if np.isclose(node.x, 0.0) and np.isclose(node.y, 0.0)]
    mid_uy = u[solver.nid_to_idx[mid_nids[0]], 1]

    R_final = L / theta_target
    mid_uy_th = R_final * (1.0 - np.cos(theta_target / 2.0))
    err_pct = abs(mid_uy - mid_uy_th) / mid_uy_th * 100.0

    print(f"\n[Q4_COROTATIONAL_SRI 90-deg] Midpoint UY: FEA={mid_uy:.4f} mm, Theory={mid_uy_th:.4f} mm, Error={err_pct:.3f}%")
    assert err_pct < 1.0, f"Midpoint deflection error {err_pct:.3f}% exceeds 1.0% limit"


def test_element_formulation_comparison():
    """Compare Q4_COROTATIONAL_SRI vs Q4_COROTATIONAL on large-rotation thin glass bending."""
    L = 10.0
    t = 0.4
    theta_target = np.radians(45.0)

    elements_to_test = ["Q4_COROTATIONAL_SRI", "Q4_COROTATIONAL"]
    results = {}

    for elem_type in elements_to_test:
        mesh = build_two_point_bending_mesh(L, t, nx=20, ny=2, pid=1)
        mat = J2Plasticity(E=72300.0, nu=0.22, sigma_y0=1e9, H=0.0)

        solver = DynamicSolver(
            mesh, {1: mat}, rho=2.5e-9,
            mode="quasistatic",
            element_type=elem_type,
            tol=1e-3, rtol=1e-4, max_iter=20,
            verbose=False
        )

        left_nids = [nid for nid, node in mesh.nodes.items() if np.isclose(node.x, -L / 2.0)]
        right_nids = [nid for nid, node in mesh.nodes.items() if np.isclose(node.x, L / 2.0)]

        bc_dofs = []
        for nid in left_nids:
            idx = solver.nid_to_idx[nid]
            bc_dofs.extend([2 * idx, 2 * idx + 1])
        for nid in right_nids:
            idx = solver.nid_to_idx[nid]
            bc_dofs.extend([2 * idx, 2 * idx + 1])
        bc_dofs = np.array(bc_dofs, dtype=np.int32)

        n_steps = 10
        converged = True
        total_iters = 0
        for step in range(1, n_steps + 1):
            th = theta_target * (step / n_steps)
            sinc = np.sin(th) / th if th > 1e-6 else 1.0
            cosc = (1.0 - np.cos(th)) / th if th > 1e-6 else 0.0
            dx_tip = L * (sinc - 1.0)
            dy_tip = L * cosc
            R_rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

            bc_vals = [0.0] * (2 * len(left_nids))
            for nid in right_nids:
                node = mesh.nodes[nid]
                dxy = np.array([0.0, node.y])
                new_xy = np.array([L / 2.0 + dx_tip, dy_tip]) + R_rot @ dxy
                bc_vals.extend([new_xy[0] - node.x, new_xy[1] - node.y])

            solver.set_prescribed_dofs(bc_dofs, bc_vals=np.array(bc_vals, dtype=np.float64))
            res = solver.solve_step(1.0 / n_steps)
            if res < 0:
                converged = False
                break
            total_iters += res

        assert converged, f"Element {elem_type} failed to converge"
        u = solver.u[:2 * solver.n_nodes].reshape(-1, 2)
        mid_nids = [nid for nid, node in mesh.nodes.items() if np.isclose(node.x, 0.0) and np.isclose(node.y, 0.0)]
        mid_uy = u[solver.nid_to_idx[mid_nids[0]], 1]
        R_final = L / theta_target
        mid_uy_th = R_final * (1.0 - np.cos(theta_target / 2.0))
        err_pct = abs(mid_uy - mid_uy_th) / mid_uy_th * 100.0
        results[elem_type] = {
            "error_pct": err_pct,
            "total_iters": total_iters,
            "avg_iters": total_iters / n_steps,
        }

    print("\n--- Element Formulation Comparison (45-deg Bending) ---")
    for elem, data in results.items():
        print(f"{elem:22s}: Error = {data['error_pct']:.3f}%, Avg Iters = {data['avg_iters']:.1f}")
        assert data["error_pct"] < 2.0
