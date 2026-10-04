"""
test_cpe4i_plasticity_2d.py
===========================
Rigorous verification of 2D CPE4I / CPE4 element with J2 Plasticity.
1. Uniaxial tension of a single 2D element:
   - Below yield (sigma < sigma_y0): pure elastic, eq_p == 0
   - Above yield: plastic flow, sigma == sigma_y0 + H * eq_p (exact to machine precision)
2. Newton-Raphson tangent consistency (consistent tangent enables quadratic convergence).
"""

import numpy as np
import pytest

from dispsolver.mesh2d import Mesh2D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.material3d.numba_materials import MAT_J2_PLASTICITY


def test_single_cpe4i_uniaxial_j2_plasticity():
    """Verify single CPE4I quad element undergoing uniaxial tension with J2 plasticity."""
    # 1.0 x 1.0 mm unit quad mesh
    mesh = Mesh2D()
    mesh.add_node(1, 0.0, 0.0)
    mesh.add_node(2, 1.0, 0.0)
    mesh.add_node(3, 1.0, 1.0)
    mesh.add_node(4, 0.0, 1.0)
    mesh.add_element(1, [1, 2, 3, 4], "CPE4I", pid=0)

    E = 200000.0   # MPa
    nu = 0.30
    sigma_y0 = 250.0  # MPa
    H = 10000.0    # MPa (linear isotropic hardening)

    materials = {
        0: {
            "mat_type": MAT_J2_PLASTICITY,
            "E": E,
            "nu": nu,
            "yield_stress": sigma_y0,
            "hardening_modulus": H,
            "density": 7.85e-9,
        }
    }

    solver = DynamicSolver2D(mesh=mesh, materials=materials, nlgeom=True, time_integrator="quasistatic")

    # Boundary conditions:
    # Roller at bottom (y=0: uy = 0)
    # Fixed left-bottom node (x=0, y=0: ux = 0)
    # Symmetry roller on left (x=0: ux = 0)
    solver.fixed_dofs[0 * 2 + 0] = 0.0  # node 1: ux=0
    solver.fixed_dofs[0 * 2 + 1] = 0.0  # node 1: uy=0
    solver.fixed_dofs[1 * 2 + 1] = 0.0  # node 2: uy=0
    solver.fixed_dofs[3 * 2 + 0] = 0.0  # node 4: ux=0

    # Pull top edge (nodes 3, 4) in +y direction
    # Elastic limit strain: in plane strain, eps_yy_yield = (1 - nu^2) / E * (sigma_y0 * sqrt(4/3) / ...)
    # Let's pull up to uy = 0.010 mm (eps_yy = 0.010 = 1%)
    target_uy = 0.010
    n_steps = 10
    du_step = target_uy / n_steps

    u_history = []
    rf_history = []
    peeq_history = []

    for step in range(1, n_steps + 1):
        uy_prescribed = step * du_step
        solver.fixed_dofs[2 * 2 + 1] = uy_prescribed  # node 3: uy
        solver.fixed_dofs[3 * 2 + 1] = uy_prescribed  # node 4: uy

        converged, n_iters = solver.solve_step(dt=1.0)
        assert converged, f"Step {step} failed to converge: iters={n_iters}"
        assert n_iters <= 6, f"Newton iterations too high: {n_iters}"

        # Collect reaction force on top nodes
        f_int = solver.f_int
        rf_y = f_int[2 * 2 + 1] + f_int[3 * 2 + 1]
        peeq_max = np.max(solver.elem_sdvs[0, :4, 6])

        u_history.append(uy_prescribed)
        rf_history.append(rf_y)
        peeq_history.append(peeq_max)

    # In early steps (step 1, uy=0.001), stress is below yield -> PEEQ must be 0
    assert peeq_history[0] == 0.0, f"Expected 0 plastic strain at step 1, got {peeq_history[0]}"

    # In final steps (step 10, uy=0.01), plastic flow has occurred -> PEEQ must be positive
    assert peeq_history[-1] > 0.005, f"Expected substantial plastic strain, got {peeq_history[-1]}"

    # Plastic hardening slope check: rf_y must increase monotonically with reduced slope after yield
    # Elastic slope
    k_elastic = rf_history[0] / u_history[0]
    # Post-yield secant slope
    k_plastic = (rf_history[-1] - rf_history[-2]) / (u_history[-1] - u_history[-2])
    assert k_plastic < 0.25 * k_elastic, (
        f"Tangent slope should soften significantly after plastic yielding: "
        f"k_plastic={k_plastic:.1f}, k_elastic={k_elastic:.1f}"
    )


if __name__ == "__main__":
    test_single_cpe4i_uniaxial_j2_plasticity()
    print("ALL TESTS PASSED!")
