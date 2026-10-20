"""The co-rotational frame of a plane element is the rotation of its (least-squares) deformation gradient.

For a homogeneous motion x = R U X (R a rotation, U a symmetric stretch) the frame must be exactly R whatever the
node layout. The old definition (the best-fit angle of the node cloud, 2D Procrustes) equals R only when the
nodes are distributed isotropically; for a triangle it differs by O(strain), which is what the CPE3/CPE6
NLGEOM plasticity test measured (3.3e-4 on the stress).
"""
import numpy as np
import pytest

from dispsolver.element2d.cr_generic_2d import corotational_local_mesh_2d

_TRI = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_SKEW_QUAD = np.array([[0.0, 0.0], [2.0, 0.3], [2.4, 1.1], [0.2, 0.9]])


def _frame_angle(X, F):
    x = X @ F.T
    u = (x - X).ravel()
    conn = np.arange(len(X), dtype=np.int64).reshape(1, -1)
    _, _, _, cs = corotational_local_mesh_2d(np.ascontiguousarray(X), conn, np.ascontiguousarray(u))
    return float(np.arctan2(cs[0, 1], cs[0, 0]))


@pytest.mark.parametrize("X", [_TRI, _SQUARE, _SKEW_QUAD], ids=["triangle", "square", "skewed quad"])
@pytest.mark.parametrize("theta_deg", [0.0, 25.0, -70.0, 130.0])
@pytest.mark.parametrize("U", [np.eye(2), np.diag([1.05, 1.0]), np.array([[1.08, 0.03], [0.03, 0.97]])],
                         ids=["rigid", "uniaxial stretch", "stretch + shear"])
def test_frame_is_the_polar_rotation_of_a_homogeneous_motion(X, theta_deg, U):
    t = np.radians(theta_deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    assert _frame_angle(X, R @ U) == pytest.approx(t, abs=1e-12)


def test_two_node_truss_keeps_its_axis_rotation():
    X = np.array([[0.0, 0.0], [2.0, 0.0]])
    t = np.radians(33.0)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    assert _frame_angle(X, R @ np.diag([1.1, 1.0])) == pytest.approx(t, abs=1e-12)
