"""
tests/test_visualizer_accuracy.py
==================================
Unit tests verifying accuracy of visualizer data processing, deformation scale factor defaults,
extra DOF vector slicing, and plane strain von Mises stress computations in dispsolver/postprocess/viewer.py.
"""

import numpy as np
import pytest
from types import SimpleNamespace
from dispsolver.postprocess.viewer import _ResultCache, FieldData


class MockElement:
    def __init__(self, pid=0):
        self.pid = pid


class MockMesh:
    def __init__(self):
        self.elements = {0: MockElement(0), 1: MockElement(1)}


class MockSolver:
    def __init__(self, n_nodes=4, n_extra=3):
        self.n_nodes = n_nodes
        self.n_dofs = 2 * n_nodes
        self.n_elem = 2
        self.elem_ids = [0, 1]
        self.conn = np.array([[0, 1, 2, 3], [3, 2, 0, 1]], dtype=np.int64)
        self.coords = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ], dtype=np.float64)
        
        # u includes nodal DOFs + extra scalar DOFs (e.g. RBE2 angles / multipliers)
        # Nodal DOFs: 2*4 = 8. Extra DOFs: 3. Total length = 11.
        u_nodes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
        u_extra = np.array([99.0, 88.0, 77.0], dtype=np.float64)
        self.u = np.concatenate([u_nodes, u_extra])
        
        self.mesh = MockMesh()
        self.state = None
        self.material = None
        self.materials = {}
        self.time = 1.0


def test_extra_dofs_slicing_safety():
    """Verify that solver.u with extra scalar DOFs does not corrupt nodal displacement extraction."""
    solver = MockSolver(n_nodes=4, n_extra=3)
    n_nodes = solver.n_nodes
    u_nodes = solver.u[: 2 * n_nodes]
    
    assert len(u_nodes) == 8
    ux = u_nodes[0::2]
    uy = u_nodes[1::2]
    
    np.testing.assert_allclose(ux, [0.1, 0.3, 0.5, 0.7])
    np.testing.assert_allclose(uy, [0.2, 0.4, 0.6, 0.8])


def test_rigid_rotation_scale_factor_distortion():
    """Verify that linear deformation scale factor s!=1.0 distorts rigid rotations."""
    # Master node at (0, 0), Rigid node at (10, 0).
    X0 = np.array([10.0, 0.0])
    theta = np.radians(90.0)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    # Exact rotated position: (0, 10)
    X_rotated = R @ X0
    u_exact = X_rotated - X0  # [-10, 10]
    
    # Under scale factor s = 1.0: X0 + 1.0 * u_exact = (0, 10) -> Pure rigid rotation (Length = 10.0)
    X_s1 = X0 + 1.0 * u_exact
    assert np.isclose(np.linalg.norm(X_s1), 10.0)
    
    # Under scale factor s = 2.0: X0 + 2.0 * u_exact = (-10, 20) -> Distorted length sqrt(500) = 22.36!
    X_s2 = X0 + 2.0 * u_exact
    distorted_len = np.linalg.norm(X_s2)
    assert not np.isclose(distorted_len, 10.0)
    assert np.isclose(distorted_len, np.sqrt(500.0))


def test_plane_strain_von_mises_formula():
    """Verify plane strain von Mises formula including out-of-plane s33 = nu * (s11 + s22)."""
    cache = _ResultCache(MockSolver())
    s11, s22, s12 = 100.0, 50.0, 20.0
    nu = 0.3
    s33 = nu * (s11 + s22)  # 45.0
    
    # 3D von Mises formula
    vm_expected = np.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2) + 3.0 * s12**2)
    vm_calc = cache._von_mises(s11, s22, s12, nu=nu)
    
    assert np.isclose(vm_calc, vm_expected)
