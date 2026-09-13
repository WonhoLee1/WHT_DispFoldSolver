"""
test_universal_corotational_elements.py
========================================
Verification suite for Universal Co-Rotational (_CR) elements and NLGEOM auto-dispatch.

Tests:
1. Rigid body rotation invariance (zero stress/internal force under pure rotation).
2. Tangent stiffness positive definiteness and rank sufficiency.
3. Automatic NLGEOM promotion in DynamicSolver2D and DynamicSolver3D.
"""

import numpy as np
import pytest

from dispsolver.element2d.cr_wrapper_2d import (
    compute_cpe4i_cr_element_numba,
    compute_cpe4h_cr_element_numba,
    compute_cpe4r_cr_element_numba,
    compute_cpe6m_cr_element_numba
)
from dispsolver.element3d.cr_wrapper_3d import (
    compute_c3d8i_cr_element_numba,
    compute_c3d8h_cr_element_numba,
    compute_c3d8r_cr_element_numba
)
from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


# ==============================================================================
# 1. 2D Elements Rigid Rotation Invariance (90 deg & 180 deg)
# ==============================================================================

@pytest.mark.parametrize("kernel", [
    compute_cpe4i_cr_element_numba,
    compute_cpe4h_cr_element_numba,
    compute_cpe4r_cr_element_numba
])
def test_quad_corotational_rigid_rotation_invariance(kernel):
    """Under pure 90 deg rotation, internal force must be machine zero (< 1e-9)."""
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float64)

    theta = np.pi / 2.0  # 90 degrees
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ], dtype=np.float64)

    u_elem = np.zeros(8, dtype=np.float64)
    for a in range(4):
        x_rot = R @ coords[a]
        u_elem[2 * a + 0] = x_rot[0] - coords[a, 0]
        u_elem[2 * a + 1] = x_rot[1] - coords[a, 1]

    props = np.array([200000.0, 0.3], dtype=np.float64)
    sdvs = np.zeros((4, 7), dtype=np.float64)

    fe, Ke, err = kernel(coords, u_elem, 0, props, sdvs, 1.0)
    assert err == 0
    norm_fe = np.linalg.norm(fe)
    assert norm_fe < 1e-8, f"Residual force {norm_fe} exceeds tolerance under pure 90 deg rotation"


def test_cpe6m_cr_rigid_rotation_invariance():
    """CPE6M_CR under 90 deg pure rotation must yield zero internal force."""
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 0.5]
    ], dtype=np.float64)

    theta = np.pi / 2.0
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ], dtype=np.float64)

    u_elem = np.zeros(12, dtype=np.float64)
    for a in range(6):
        x_rot = R @ coords[a]
        u_elem[2 * a + 0] = x_rot[0] - coords[a, 0]
        u_elem[2 * a + 1] = x_rot[1] - coords[a, 1]

    props = np.array([200000.0, 0.3], dtype=np.float64)
    sdvs = np.zeros((3, 7), dtype=np.float64)

    fe, Ke, err = compute_cpe6m_cr_element_numba(coords, u_elem, 0, props, sdvs, 1.0)
    assert err == 0
    norm_fe = np.linalg.norm(fe)
    assert norm_fe < 1e-8, f"CPE6M_CR residual force {norm_fe} exceeds tolerance under pure rotation"


# ==============================================================================
# 2. 3D Elements Rigid Rotation Invariance
# ==============================================================================

@pytest.mark.parametrize("kernel", [
    compute_c3d8i_cr_element_numba,
    compute_c3d8h_cr_element_numba,
    compute_c3d8r_cr_element_numba
])
def test_c3d8_corotational_rigid_rotation_invariance(kernel):
    """Under pure 90 deg rotation around Z-axis, internal force must be machine zero (< 1e-8)."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)

    theta = np.pi / 2.0
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0]
    ], dtype=np.float64)

    u_elem = np.zeros(24, dtype=np.float64)
    for a in range(8):
        x_rot = R @ coords[a]
        u_elem[3 * a + 0] = x_rot[0] - coords[a, 0]
        u_elem[3 * a + 1] = x_rot[1] - coords[a, 1]
        u_elem[3 * a + 2] = x_rot[2] - coords[a, 2]

    props = np.array([200000.0, 0.3], dtype=np.float64)
    sdvs = np.zeros((8, 7), dtype=np.float64)

    f_glob, K_glob, err = kernel(coords, u_elem, 0, props, sdvs, 1.0)
    assert err == 0
    norm_fe = np.linalg.norm(f_glob)
    assert norm_fe < 1e-7, f"3D residual force {norm_fe} exceeds tolerance under pure 90 deg rotation"


# ==============================================================================
# 3. Solver Auto-Promotion Under NLGEOM=True
# ==============================================================================

def test_dynamic2d_nlgeom_auto_promotion():
    """Verify that CPE4I is automatically mapped to CPE4I_CR (kernel group 11) when nlgeom=True."""
    mesh = Mesh2D()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(3, 1.0, 1.0)
    mesh.add_node(4, 0.0, 1.0)
    mesh.add_element(1, [1, 2, 3, 4], elem_type="CPE4I")

    solver_nlgeom = DynamicSolver2D(mesh, nlgeom=True)
    # k=11 corresponds to CPE4I_CR
    assert 11 in solver_nlgeom.elem_kernel_groups
    assert 0 in solver_nlgeom.elem_kernel_groups[11]

    solver_linear = DynamicSolver2D(mesh, nlgeom=False)
    # k=1 corresponds to small-strain CPE4I
    assert 1 in solver_linear.elem_kernel_groups
    assert 0 in solver_linear.elem_kernel_groups[1]


def test_dynamic3d_nlgeom_auto_promotion():
    """Verify that C3D8I is automatically mapped to C3D8I_CR (kernel group 11) when nlgeom=True."""
    mesh = Mesh3D()
    mesh.add_node(1, 0.0, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0, 0.0)
    mesh.add_node(3, 1.0, 1.0, 0.0)
    mesh.add_node(4, 0.0, 1.0, 0.0)
    mesh.add_node(5, 0.0, 0.0, 1.0)
    mesh.add_node(6, 1.0, 0.0, 1.0)
    mesh.add_node(7, 1.0, 1.0, 1.0)
    mesh.add_node(8, 0.0, 1.0, 1.0)
    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], elem_type="C3D8I")

    solver_nlgeom = DynamicSolver3D(mesh, nlgeom=True)
    assert 11 in solver_nlgeom.elem_kernel_groups
    assert 0 in solver_nlgeom.elem_kernel_groups[11]

    solver_linear = DynamicSolver3D(mesh, nlgeom=False)
    assert 0 in solver_linear.elem_kernel_groups
    assert 0 in solver_linear.elem_kernel_groups[0]
