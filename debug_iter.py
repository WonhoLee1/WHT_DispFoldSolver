import numpy as np
from dispsolver.solver3d import DynamicSolver3D
from dispsolver.mesh3d import Mesh3D

mesh = Mesh3D()
mesh.add_node(1, 0, 0, 0); mesh.add_node(2, 1, 0, 0); mesh.add_node(3, 1, 1, 0); mesh.add_node(4, 0, 1, 0)
mesh.add_node(5, 0, 0, 1); mesh.add_node(6, 1, 0, 1); mesh.add_node(7, 1, 1, 1); mesh.add_node(8, 0, 1, 1)
mesh.add_element(1, [1,2,3,4,5,6,7,8], "C3D8_FBAR")

solver = DynamicSolver3D(mesh)
for i in range(1, 9):
    solver.fix_dof(i, 0, 0.1) # move x
    solver.fix_dof(i, 1, 0.0)
    solver.fix_dof(i, 2, 0.0)

conv, it = solver.solve_step()
print(f"Converged: {conv}, Iters: {it}")
