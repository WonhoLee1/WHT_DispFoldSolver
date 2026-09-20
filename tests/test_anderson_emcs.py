"""
test_anderson_emcs.py
=====================
Unit tests for Anderson-Accelerated Newton (AA-Newton) and Energy-Momentum
Conserving Scheme (EMCS).
"""

import numpy as np
import pytest
from scipy.sparse import diags, csc_matrix

from dispsolver.solver.anderson_acceleration import AndersonAccelerator
from dispsolver.solver3d.time_integrator_emcs import EMCSTimeIntegrator
from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D
from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D


def test_anderson_accelerator_fixed_point():
    """Verify Anderson accelerator accelerates linear/affine fixed point mapping."""
    # Mapping g(x) = M * x + b with spectral radius < 1
    np.random.seed(42)
    N = 8
    A = np.random.randn(N, N) * 0.2
    # Ensure spectral radius < 1
    M_mat = A / (np.linalg.norm(A, 2) * 1.5)
    b_vec = np.ones(N)
    x_exact = np.linalg.solve(np.eye(N) - M_mat, b_vec)

    # Standard fixed point iteration
    x_std = np.zeros(N)
    for _ in range(15):
        x_std = M_mat @ x_std + b_vec
    err_std = float(np.linalg.norm(x_std - x_exact))

    # Anderson Accelerated fixed point
    accel = AndersonAccelerator(m=4, beta=1.0)
    x_aa = np.zeros(N)
    for k in range(15):
        gx = M_mat @ x_aa + b_vec
        fx = gx - x_aa
        x_aa = accel.step(x_aa, fx)
    err_aa = float(np.linalg.norm(x_aa - x_exact))

    assert err_aa < err_std
    assert err_aa < 1e-8


def test_anderson_accelerator_masking():
    """Verify Anderson accelerator strictly preserves masked/fixed DOFs."""
    accel = AndersonAccelerator(m=3, beta=1.0)
    x = np.array([1.0, 2.0, 0.5, 0.0])
    f = np.array([0.0, 0.1, -0.2, 0.0]) # DOFs 0 and 3 have 0 correction
    mask = np.array([False, True, True, False]) # Only DOFs 1, 2 free

    accel.record(x, f)
    x2 = np.array([1.0, 2.1, 0.3, 0.0])
    f2 = np.array([0.0, 0.05, -0.1, 0.0])

    x_next = accel.step(x2, f2, mask=mask)
    # Fixed DOFs must remain unchanged
    assert np.isclose(x_next[0], 1.0)
    assert np.isclose(x_next[3], 0.0)


def test_dynamicsolver3d_anderson_flag():
    """Verify DynamicSolver3D solve_step with use_anderson_accel=True."""
    mesh = Mesh3D()
    coords = [
        (1, 0.0, 0.0, 0.0), (2, 1.0, 0.0, 0.0),
        (3, 1.0, 1.0, 0.0), (4, 0.0, 1.0, 0.0),
        (5, 0.0, 0.0, 1.0), (6, 1.0, 0.0, 1.0),
        (7, 1.0, 1.0, 1.0), (8, 0.0, 1.0, 1.0),
    ]
    for nid, x, y, z in coords:
        mesh.add_node(nid, x, y, z)
    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], elem_type="C3D8")

    solver = DynamicSolver3D(mesh, material_params={"E": 1000.0, "nu": 0.3}, nlgeom=True)
    # Fix bottom
    for n in [1, 2, 5, 6]:
        solver.fix_dof(n, 0, 0.0)
        solver.fix_dof(n, 1, 0.0)
        solver.fix_dof(n, 2, 0.0)
    # Move top
    for n in [3, 4, 7, 8]:
        solver.fix_dof(n, 1, 0.1)

    conv, iters = solver.solve_step(
        dt=1.0,
        max_iters=25,
        use_anderson_accel=True,
        anderson_m=4
    )
    assert conv is True
    assert iters >= 2
    solver.close()


def test_dynamicsolver2d_anderson_flag():
    """Verify DynamicSolver2D solve_step with use_anderson_accel=True."""
    mesh = Mesh2D()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(3, 1.0, 1.0)
    mesh.add_node(4, 0.0, 1.0)
    mesh.add_element(1, [1, 2, 3, 4], elem_type="CPE4")

    solver = DynamicSolver2D(mesh, materials={"E": 1000.0, "nu": 0.3}, nlgeom=True)
    solver.fix_dof(1, 0, 0.0)
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(2, 0, 0.0)
    solver.fix_dof(2, 1, 0.0)
    solver.fix_dof(3, 1, 0.1)
    solver.fix_dof(4, 1, 0.1)

    conv, iters = solver.solve_step(
        dt=1.0,
        max_iters=25,
        use_anderson_accel=True,
        anderson_m=4
    )
    assert conv is True
    assert iters >= 2


def test_emcs_energy_conservation():
    """Verify EMCS conserves total mechanical energy to machine precision."""
    integrator = EMCSTimeIntegrator(scheme="emcs")
    M = diags([1e6, 1e6, 1e6, 2.0, 2.0, 2.0], format="csc")
    coords_0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    k_spring = 100.0
    L0 = 1.0

    def strain_energy(u):
        x1 = coords_0[1] + u[3:6]
        x0 = coords_0[0] + u[0:3]
        L = float(np.linalg.norm(x1 - x0))
        return 0.5 * k_spring * (L - L0) ** 2

    def internal_force(u):
        x1 = coords_0[1] + u[3:6]
        x0 = coords_0[0] + u[0:3]
        d = x1 - x0
        L = float(np.linalg.norm(d))
        n = d / max(L, 1e-12)
        f1 = k_spring * (L - L0) * n
        f_int = np.zeros(6, dtype=np.float64)
        f_int[0:3] = -f1
        f_int[3:6] = f1
        return f_int

    def tangent_stiffness(u):
        x1 = coords_0[1] + u[3:6]
        x0 = coords_0[0] + u[0:3]
        d = x1 - x0
        L = float(np.linalg.norm(d))
        n = d / max(L, 1e-12)
        K_b = k_spring * np.outer(n, n) + k_spring * (1.0 - L0 / max(L, 1e-12)) * (np.eye(3) - np.outer(n, n))
        K = np.zeros((6, 6))
        K[0:3, 0:3] = K_b
        K[0:3, 3:6] = -K_b
        K[3:6, 0:3] = -K_b
        K[3:6, 3:6] = K_b
        return csc_matrix(K)

    u_curr = np.zeros(6)
    u_curr[3] = 0.2
    v_curr = np.zeros(6)
    v_curr[4] = 1.5
    a_curr = np.zeros(6)

    dt = 0.05
    e0 = integrator.compute_total_energy(u_curr, v_curr, M, strain_energy_fn=strain_energy)

    for _ in range(50):
        u_n = u_curr.copy()
        v_n = v_curr.copy()
        a_n = a_curr.copy()
        pi_n = strain_energy(u_n)

        u_k, _, _ = integrator.predict(u_n, v_n, a_n, dt)
        u_k[0:3] = 0.0

        for _ in range(20):
            u_mid = 0.5 * (u_k + u_n)
            f_int = internal_force(u_mid)
            K_int = tangent_stiffness(u_mid)
            K_dyn, R_dyn = integrator.compute_effective_system(
                u_k, u_n, v_n, a_n, dt, K_int, f_int, M,
                strain_energy_fn=strain_energy, pi_n=pi_n
            )
            K_dyn_lil = K_dyn.tolil()
            for d in range(3):
                K_dyn_lil[d, :] = 0.0
                K_dyn_lil[:, d] = 0.0
                K_dyn_lil[d, d] = 1e15
                R_dyn[d] = 0.0
            K_dyn = K_dyn_lil.tocsc()

            if np.linalg.norm(R_dyn[3:6]) < 1e-12:
                break
            from scipy.sparse.linalg import spsolve
            du = spsolve(K_dyn, R_dyn)
            u_k += du
            u_k[0:3] = 0.0

        v_curr, a_curr = integrator.update_state(u_k, u_n, v_n, a_n, dt)
        u_curr = u_k

    e_final = integrator.compute_total_energy(u_curr, v_curr, M, strain_energy_fn=strain_energy)
    rel_error = abs(e_final - e0) / e0

    # Must be conserved to machine precision (< 1e-13)
    assert rel_error < 1e-13
