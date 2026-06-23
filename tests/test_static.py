import numpy as np
import pytest
from dispsolver.mesh import Mesh
from dispsolver.material import NeoHookean
from dispsolver.solver import DynamicSolver

def cooks_membrane_mesh(nx=4, ny=4) -> Mesh:
    """Generate Cook's membrane mesh.
    Bottom left (0,0), bottom right (48, 44)
    Top left (0, 44), top right (48, 60)
    """
    mesh = Mesh()
    
    # Coordinates
    x_left = 0.0; y_left_bot = 0.0; y_left_top = 44.0
    x_right = 48.0; y_right_bot = 44.0; y_right_top = 60.0
    
    for j in range(ny + 1):
        eta = j / ny
        for i in range(nx + 1):
            xi = i / nx
            
            x_b = x_left + xi * (x_right - x_left)
            y_b = y_left_bot + xi * (y_right_bot - y_left_bot)
            
            x_t = x_left + xi * (x_right - x_left)
            y_t = y_left_top + xi * (y_right_top - y_left_top)
            
            x = x_b + eta * (x_t - x_b)
            y = y_b + eta * (y_t - y_b)
            
            nid = j * (nx + 1) + i
            mesh.add_node(nid, x, y)
            
    # Elements
    eid = 0
    for j in range(ny):
        for i in range(nx):
            n1 = j * (nx + 1) + i
            n2 = j * (nx + 1) + i + 1
            n3 = (j + 1) * (nx + 1) + i + 1
            n4 = (j + 1) * (nx + 1) + i
            mesh.add_element(eid, [n1, n2, n3, n4], "QUAD4")
            eid += 1
            
    # Node sets
    left_nodes = [j * (nx + 1) for j in range(ny + 1)]
    right_nodes = [j * (nx + 1) + nx for j in range(ny + 1)]
    
    mesh.add_nodeset("left", set(left_nodes))
    mesh.add_nodeset("right", set(right_nodes))
    
    return mesh

def test_cooks_membrane_static():
    """Solve Cook's membrane with Neo-Hookean material and check tip deflection."""
    mesh = cooks_membrane_mesh(nx=4, ny=4)
    
    # typical properties for Cook's membrane: E ~ 250, nu ~ 0.3
    # mu = E / 2(1+nu) = 250 / 2.6 = 96.15
    # lambda = E nu / ((1+nu)(1-2nu)) = 250 * 0.3 / (1.3 * 0.4) = 144.23
    mat = NeoHookean()
    solver = DynamicSolver(mesh, mat, rho=1e-6, material_params={'mu': 96.15, 'lambda': 144.23},
                           max_iter=20, tol=1e-8)
    
    left = mesh.get_nodeset("left")
    bc_dofs = []
    nid_to_idx = mesh.node_id_to_index()
    for nid in left.node_ids:
        bc_dofs.extend([nid_to_idx[nid]*2, nid_to_idx[nid]*2+1])
        
    solver.set_prescribed_dofs(bc_dofs)
    
    # Total shear load at right edge = 100
    right = mesh.get_nodeset("right")
    load_per_node = 100.0 / len(right.node_ids)
    load_dofs = []
    load_vals = []
    for nid in right.node_ids:
        load_dofs.append(nid_to_idx[nid]*2+1)  # uy
        load_vals.append(load_per_node)
        
    solver.apply_load(load_dofs, load_vals)
    
    # Solve 1 step with large dt to act quasi-statically
    n_iter = solver.solve_step(dt=1.0)
    assert n_iter > 0
    
    # Top right node deflection
    top_right_nid = 5 * 5 - 1  # 24
    top_right_idx = nid_to_idx[top_right_nid]
    uy = solver.u[top_right_idx * 2 + 1]
    
    # Expected uy is ~6.33 for NeoHookean plane strain with E=250, nu=0.3, F=100
    assert uy > 5.0 and uy < 7.0, f"Tip deflection {uy} out of expected range"
