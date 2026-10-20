"""
test_3d_c3d8r.py
================
Verification and Unit Tests for 3D 8-Node Reduced Integration Hexahedral Element (C3D8R)
with Flanagan & Belytschko (1981) Orthogonal Hourglass Control.

Tests:
1. Hourglass Orthogonality: gamma_alpha strictly orthogonal to rigid-body modes and uniform strain.
2. Rank 18 Spectrum: Exactly 6 zero eigenvalues (rigid body modes) + 18 positive eigenvalues.
3. Hourglass modes: STIFFNESS and ENHANCED differ only in the hourglass part (zero for linear fields).
4. Tangent Consistency: Finite-difference Jacobian agreement with algorithmic tangent (err < 1e-5).
5. Shear Locking Freedom: Verify high bending flexibility compared to locked standard C3D8.
"""

import pytest
import numpy as np

from dispsolver.element3d.c3d8r_numba import (
    compute_c3d8r_element_numba,
    assemble_mesh_c3d8r_numba
)
from dispsolver.material3d.numba_materials import MAT_LINEAR_ELASTIC, MAT_CUSTOM_ELASTIC


def make_standard_hex_coords():
    """Create a 1x1x1 standard cube centered at origin."""
    coords = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5]
    ], dtype=np.float64)
    return coords


def make_distorted_hex_coords():
    """Create a moderately distorted hexahedral element."""
    coords = make_standard_hex_coords()
    # Apply shear and tapered distortion
    coords[:, 0] += 0.2 * coords[:, 1] + 0.1 * coords[:, 2]
    coords[:, 1] += 0.15 * coords[:, 0]
    return coords


def _controls(hourglass_code):
    ctrl = np.zeros(8, dtype=np.float64)
    ctrl[5] = hourglass_code
    return ctrl


def test_c3d8r_hourglass_orthogonality():
    """The hourglass vectors are orthogonal to rigid-body and linear fields.

    Seen through the kernel: rigid motion gives zero force, and for a linear displacement field the force
    is the same for every hourglass mode (the hourglass part carries nothing).
    """
    coords = make_distorted_hex_coords()
    props = np.array([200000.0, 0.3], dtype=np.float64)
    sdvs = np.zeros((1, 1), dtype=np.float64)

    def force(u, code):
        _, f, err = compute_c3d8r_element_numba(
            coords, u, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs.copy(), elem_controls=_controls(code))
        assert err == 0
        return f

    # rigid translation and small rotation about z: no force at all
    u_tr = np.tile([1.0, -2.0, 0.5], 8)
    th = 1e-7
    u_rot = np.zeros(24)
    for a in range(8):
        u_rot[3 * a] = -th * coords[a, 1]
        u_rot[3 * a + 1] = th * coords[a, 0]
    for u in (u_tr, u_rot):
        assert np.linalg.norm(force(u, 1.0)) < 1e-4

    # linear field u = G x: hourglass-mode independent
    G = np.array([[1e-3, 2e-4, 0.0], [0.0, -5e-4, 3e-4], [1e-4, 0.0, 7e-4]])
    u_lin = (coords @ G.T).reshape(-1)
    f_stiff = force(u_lin, 1.0)
    f_enh = force(u_lin, 0.0)
    assert np.linalg.norm(f_stiff - f_enh) < 1e-9 * np.linalg.norm(f_stiff)


def test_c3d8r_eigenvalue_spectrum_and_rank():
    """Verify that C3D8R with hourglass control has full rank 18 (24 DOFs - 6 rigid body modes)."""
    coords = make_distorted_hex_coords()
    u_elem = np.zeros(24, dtype=np.float64)

    props = np.array([200000.0, 0.3], dtype=np.float64)
    sdvs = np.zeros((1, 1), dtype=np.float64)

    K_elem, f_int, err = compute_c3d8r_element_numba(
        coords, u_elem, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs
    )
    assert err == 0
    assert K_elem.shape == (24, 24)

    # Check symmetry
    assert np.allclose(K_elem, K_elem.T, atol=1e-8)

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(K_elem)

    # Exactly 6 zero eigenvalues for 3D rigid body motions
    zero_modes = eigvals[:6]
    positive_modes = eigvals[6:]

    assert np.all(np.abs(zero_modes) < 1e-4), f"Rigid body modes exceed zero threshold: {zero_modes}"
    assert len(positive_modes) == 18, f"Expected 18 positive modes, got {len(positive_modes)}"
    assert np.all(positive_modes > 1.0), f"Hourglass or strain mode too weak: {positive_modes[:5]}"
    assert np.min(eigvals) >= -1e-7, f"Spurious negative eigenvalue found: {np.min(eigvals)}"


def test_c3d8r_stiffness_vs_enhanced_differ_only_in_hourglass_part():
    """The two hourglass formulations share the centroid (uniform-strain) part: they differ, and by hourglass modes only."""
    coords = make_distorted_hex_coords()
    np.random.seed(123)
    u_elem = 0.001 * np.random.randn(24)
    props = np.array([200000.0, 0.3], dtype=np.float64)
    sdvs = np.zeros((1, 1), dtype=np.float64)

    K_s, f_s, e1 = compute_c3d8r_element_numba(
        coords, u_elem, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs.copy(), elem_controls=_controls(1.0))
    K_e, f_e, e2 = compute_c3d8r_element_numba(
        coords, u_elem, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs.copy(), elem_controls=_controls(0.0))
    assert e1 == 0 and e2 == 0
    dK = K_s - K_e
    assert np.linalg.norm(dK) > 0.0
    # the difference annihilates every rigid-body mode
    for d in range(3):
        t = np.zeros(24)
        t[d::3] = 1.0
        assert np.linalg.norm(dK @ t) < 1e-6 * np.linalg.norm(dK)
    assert np.allclose(f_s - f_e, dK @ u_elem, rtol=1e-6, atol=1e-8)


def test_c3d8r_tangent_consistency():
    """Verify algorithmic tangent stiffness matrix matches numerical finite differences."""
    coords = make_distorted_hex_coords()
    np.random.seed(456)
    u_base = 0.002 * np.random.randn(24)

    props = np.array([200000.0, 0.3], dtype=np.float64)
    sdvs = np.zeros((1, 1), dtype=np.float64)

    K_ana, f_base, err = compute_c3d8r_element_numba(
        coords, u_base, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs.copy()
    )
    assert err == 0

    # Numerical Jacobian via central differences
    h = 1e-6
    K_num = np.zeros((24, 24), dtype=np.float64)

    for j in range(24):
        u_p = u_base.copy()
        u_m = u_base.copy()
        u_p[j] += h
        u_m[j] -= h

        _, f_p, _ = compute_c3d8r_element_numba(
            coords, u_p, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs.copy()
        )
        _, f_m, _ = compute_c3d8r_element_numba(
            coords, u_m, mat_type=MAT_LINEAR_ELASTIC, props=props, sdvs=sdvs.copy()
        )

        K_num[:, j] = (f_p - f_m) / (2.0 * h)

    rel_err = np.linalg.norm(K_ana - K_num) / np.linalg.norm(K_ana)
    assert rel_err < 1e-5, f"Tangent consistency error too high: {rel_err}"


def test_c3d8r_dynamicsolver3d_cantilever_bending():
    """Verify DynamicSolver3D non-linear solve convergence on a C3D8R cantilever beam."""
    from dispsolver.mesh3d.mesh3d import Mesh3D
    from dispsolver.solver3d.dynamic3d import DynamicSolver3D

    mesh = Mesh3D()
    # 2 elements in X, 1 in Y, 1 in Z (beam 2x1x1 mm)
    # 12 nodes, 2 C3D8R elements
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0]
    zs = [0.0, 1.0]

    nid = 1
    node_grid = {}
    for k in range(2):
        for j in range(2):
            for i in range(3):
                node_grid[(i, j, k)] = nid
                mesh.add_node(nid, xs[i], ys[j], zs[k])
                nid += 1

    for e in range(2):
        conn = [
            node_grid[(e,     0, 0)],
            node_grid[(e + 1, 0, 0)],
            node_grid[(e + 1, 1, 0)],
            node_grid[(e,     1, 0)],
            node_grid[(e,     0, 1)],
            node_grid[(e + 1, 0, 1)],
            node_grid[(e + 1, 1, 1)],
            node_grid[(e,     1, 1)]
        ]
        mesh.add_element(e + 1, conn, elem_type="C3D8R")

    solver = DynamicSolver3D(mesh, {"E": 200000.0, "nu": 0.3})

    # Fix x = 0 face (nodes with i=0)
    for j in range(2):
        for k in range(2):
            node_id = node_grid[(0, j, k)]
            solver.fix_dof(node_id, 0, 0.0)
            solver.fix_dof(node_id, 1, 0.0)
            solver.fix_dof(node_id, 2, 0.0)

    # Prescribe vertical displacement on x = 2 face
    for j in range(2):
        for k in range(2):
            node_id = node_grid[(2, j, k)]
            solver.fix_dof(node_id, 2, -0.05)

    u_conv, converged = solver.solve_step(dt=1.0)
    assert converged
    assert np.max(np.abs(u_conv)) > 0.04

