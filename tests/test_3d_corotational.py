import numpy as np
import pytest
from dispsolver.mesh3d import Mesh3D
from dispsolver.solver3d import DynamicSolver3D
from dispsolver.element3d.c3d8_corotational_numba import (
    compute_element_rotation_3d,
    assemble_mesh_c3d8_corotational_numba
)

def test_rotation_extraction_invariance():
    """Verify that arbitrary 3D rotation and translation produce zero local deformation."""
    X = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float64)

    # 45 deg around X, 60 deg around Y, 30 deg around Z
    rx, ry, rz = np.deg2rad(45), np.deg2rad(60), np.deg2rad(30)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R_true = Rz @ Ry @ Rx

    x_rot = (X @ R_true.T) + np.array([15.0, -8.0, 3.5])
    R_calc = compute_element_rotation_3d(X, x_rot)

    np.testing.assert_allclose(R_calc, R_true, atol=1e-13)
    np.testing.assert_allclose(R_calc @ R_calc.T, np.eye(3), atol=1e-14)


def test_rigid_body_rotation_zero_stress():
    """Verify that rotating an element 90 degrees produces zero internal force."""
    X = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float64)

    theta = np.deg2rad(90.0)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    x_curr = X @ R.T

    u_global = (x_curr - X).ravel()
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32)
    mat_types = np.array([0], dtype=np.int32)
    props = np.zeros((1, 36), dtype=np.float64)
    props[0, 0] = 200000.0  # E
    props[0, 1] = 0.3       # nu
    sdvs = np.zeros((1, 8, 0), dtype=np.float64)

    f_elems, K_elems, has_error = assemble_mesh_c3d8_corotational_numba(
        X, conn, u_global, mat_types, props, sdvs, 1.0
    )

    assert has_error == 0
    # Internal force must be machine zero
    np.testing.assert_allclose(f_elems[0], 0.0, atol=1e-8)


def test_corotational_multimaterial_assembly():
    """Verify corotational assembly handles multiple material types."""
    mesh = Mesh3D()
    # 2 elements along X
    for i, (x, y, z) in enumerate([
        (0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1),
        (2,0,0),(2,1,0),(2,0,1),(2,1,1)
    ]):
        mesh.add_node(i+1, x, y, z)

    mesh.add_element(1, [1,2,3,4,5,6,7,8], "C3D8_CR", pid=0)
    mesh.add_element(2, [2,9,10,3,6,11,12,7], "C3D8_CR", pid=1)

    materials = {
        0: {'type': 'j2_plasticity', 'E': 4000.0, 'nu': 0.35, 'yield_stress': 80.0, 'hardening_modulus': 200.0},
        1: {'type': 'neo_hookean', 'C10': 0.1, 'D1': 0.01}
    }

    solver = DynamicSolver3D(mesh, materials=materials)
    u_zeros = np.zeros(solver.num_dofs, dtype=np.float64)
    K_g, f_int = solver.assemble_system(u_zeros, dt=1.0)

    assert solver.last_assembly_error == 0
    assert K_g.shape == (solver.num_dofs, solver.num_dofs)
    np.testing.assert_allclose(f_int, 0.0, atol=1e-10)
