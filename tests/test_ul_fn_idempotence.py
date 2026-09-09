"""
tests/test_ul_fn_idempotence.py
=================================
T3 (dev_log/plan_abaqus_element_consolidation_20260908.md's opus review,
finding F5 / fix): `DynamicSolver._assemble(u, dt)` must be idempotent --
calling it twice with the SAME `u`, without any Newton-convergence commit
in between, must return the SAME residual and must NOT mutate
`self._ul_F_n`.

Before the fix, `self._ul_F_n[elem_indices] = F_n_new` was written on
EVERY assembly call (i.e. every Newton iteration), not just on
convergence -- so a second `_assemble` call with the same `u` would
evaluate `F = F_inc @ (F_inc @ F_n_conv)` instead of `F_inc @ F_n_conv`,
giving a DIFFERENT (wrong) residual the second time. This was masked in
production because the viscoelastic UL path converges in ~1 Newton
iteration per step; it would become active the moment any UL element
needs multiple iterations per step.
"""

import numpy as np

from dispsolver.mesh import Mesh
from dispsolver.material import ArrudaBoyce, ViscoelasticMaterial
from dispsolver.solver import DynamicSolver


def _build_solver():
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
        element_type="CPE4RH", mode="quasistatic",
    )
    return solver


def test_ul_f_n_not_mutated_by_repeated_assembly():
    solver = _build_solver()
    solver.ul_mode = True
    # Give the element a non-identity "last converged" F_n directly,
    # bypassing a real solve (isolates the assembly-time mutation bug).
    F_n_case = np.array([[1.10, 0.02], [0.0, 0.95]])
    solver._ul_F_n[:] = np.tile(F_n_case, (solver.n_elem, 4, 1, 1))
    solver._ul_u_ref = np.zeros(solver.n_dofs)

    u = np.zeros(solver.n_dofs)
    u[2] = 0.05  # some nonzero incremental displacement, node 1 UX

    f_n_before = solver._ul_F_n.copy()
    f_int_1, K_1, _ = solver._assemble(u, dt=0.5)
    f_n_after_call1 = solver._ul_F_n.copy()
    f_int_2, K_2, _ = solver._assemble(u, dt=0.5)
    f_n_after_call2 = solver._ul_F_n.copy()

    assert np.array_equal(f_n_before, f_n_after_call1), (
        "self._ul_F_n was mutated by a single _assemble() call -- it "
        "must only change at the commit-time resync (solve_step "
        "convergence), never during assembly/Newton iteration."
    )
    assert np.array_equal(f_n_after_call1, f_n_after_call2), (
        "self._ul_F_n changed between two identical _assemble() calls."
    )
    np.testing.assert_allclose(
        np.asarray(f_int_1.todense() if hasattr(f_int_1, "todense") else f_int_1),
        np.asarray(f_int_2.todense() if hasattr(f_int_2, "todense") else f_int_2),
        rtol=0, atol=0,
        err_msg="_assemble(u, dt) is not idempotent for the same u -- "
                "repeated calls without a convergence commit must give "
                "bitwise-identical residuals (F5 regression).",
    )


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
