import numpy as np
import pytest
from dispsolver.mesh import Mesh
from dispsolver.constraint.tie import TieConstraint

def test_tie_constraint():
    mesh = Mesh()
    mesh.add_node(0, 0.0, 0.0)
    mesh.add_node(1, 0.0, 0.0)  # Same coordinate
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(3, 1.0, 0.0)  # Same coordinate
    
    mesh.add_nodeset("master", {0, 2})
    mesh.add_nodeset("slave", {1, 3})
    
    constraint = TieConstraint(mesh, "master", "slave", tol=1e-3)
    
    assert constraint.n_extra_primal() == 0
    assert constraint.n_multipliers() == 4
    
    u = np.zeros(8)
    u_ext = np.zeros(0)
    
    # Apply displacement to master nodes
    u[0] = 0.5; u[1] = 0.1  # node 0
    u[4] = 0.6; u[5] = -0.2 # node 2
    
    # Gap should be equal to the master displacements since slave is 0
    _, _, _, _, _, _, g = constraint.assemble(u, u_ext)
    
    assert np.isclose(g[0], -0.5) # Node 1 (slave) - Node 0 (master) = 0.0 - 0.5
    assert np.isclose(g[1], -0.1)
    assert np.isclose(g[2], -0.6) # Node 3 - Node 2
    assert np.isclose(g[3], 0.2)

    # Set slave exactly equal to master
    u[2] = 0.5; u[3] = 0.1
    u[6] = 0.6; u[7] = -0.2
    
    _, _, _, _, _, _, g = constraint.assemble(u, u_ext)
    assert np.allclose(g, 0.0)
