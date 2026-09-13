import numpy as np
from dispsolver.mesh import Mesh
from dispsolver.material import ArrudaBoyce, ViscoelasticMaterial
from dispsolver.solver import DynamicSolver

mesh = Mesh()
mesh.add_node(0, 0.0, 0.0)
mesh.add_node(1, 1.0, 0.0)
mesh.add_node(2, 1.0, 1.0)
mesh.add_node(3, 0.0, 1.0)
mesh.add_element(0, [0, 1, 2, 3], "CPE4RH")

base_mat = ArrudaBoyce()
visco_mat = ViscoelasticMaterial(base_mat, g_i=[0.20], tau_i=[3.33])
params = {"mu": 0.015614, "lambda_m": 3.0, "K": 8.3333}

solver = DynamicSolver(
    mesh, visco_mat, rho=1000.0, material_params=params,
    element_type="CPE4RH", verbose=True, tol=1e-8,
)

bc_dofs = []
bc_vals = []
for nid in [0, 3]:
    bc_dofs.extend([nid * 2, nid * 2 + 1])
    bc_vals.extend([0.0, 0.0])
for nid in [1, 2]:
    bc_dofs.extend([nid * 2, nid * 2 + 1])
    bc_vals.extend([0.05, 0.0])  # 5% stretch in x

solver.set_prescribed_dofs(bc_dofs, bc_vals)

conv = solver.solve_step(dt=0.5)
print("solve_step return code:", conv)
print("max|u| =", np.max(np.abs(solver.u)))
print("u:", solver.u)
print("finite:", np.all(np.isfinite(solver.u)))
print("state finite:", solver.state is None or np.all(np.isfinite(solver.state)))
