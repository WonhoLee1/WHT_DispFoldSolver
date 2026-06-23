import numpy as np
import pytest
from dispsolver.mesh import Mesh
from dispsolver.constraint.rbe2 import RBE2HingeConstraint

def test_rbe2_kinematics():
    mesh = Mesh()
    mesh.add_node(0, 0.0, 0.0)
    mesh.add_node(1, 1.0, 0.0)
    mesh.add_node(2, 0.0, 1.0)
    
    constraint = RBE2HingeConstraint(mesh, master_id=0, slave_ids=[1, 2], extra_primal_offset=0)
    
    assert constraint.n_extra_primal() == 1
    assert constraint.n_multipliers() == 4
    
    u = np.zeros(6)
    # Give master node a small displacement
    u[0] = 0.1  # ux_m
    u[1] = 0.2  # uy_m
    
    # Give slave 1 the exact displacement corresponding to theta=0.1
    # x_s = 1.0, y_s = 0.0
    # ux_s = ux_m + (cos(0.1)-1)*dx - sin(0.1)*dy
    # uy_s = uy_m + sin(0.1)*dx + (cos(0.1)-1)*dy
    theta = 0.1
    cost = np.cos(theta)
    sint = np.sin(theta)
    
    dx1, dy1 = 1.0, 0.0
    u[2] = 0.1 + (cost - 1.0) * dx1 - sint * dy1
    u[3] = 0.2 + sint * dx1 + (cost - 1.0) * dy1
    
    dx2, dy2 = 0.0, 1.0
    u[4] = 0.1 + (cost - 1.0) * dx2 - sint * dy2
    u[5] = 0.2 + sint * dx2 + (cost - 1.0) * dy2
    
    u_ext = np.array([theta])
    
    r_u, c_u, v_u, r_ext, c_ext, v_ext, g = constraint.assemble(u, u_ext)
    
    # Gap should be zero
    assert np.allclose(g, 0.0, atol=1e-12)
    
    # Check extra primal derivatives (d(gap)/dtheta)
    # g_x = ux_s - ux_m - ((cost - 1)*dx - sint*dy)
    # d(g_x)/dtheta = sint*dx + cost*dy
    # For slave 1 (dx=1, dy=0), expected val_ext is sint(0.1)
    # Equation 0 is x for slave 1
    idx = np.where(r_ext == 0)[0][0]
    assert np.isclose(v_ext[idx], sint)
