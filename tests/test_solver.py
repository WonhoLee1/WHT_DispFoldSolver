"""
test_solver.py
==============
Tests for DynamicSolver (Implicit Newmark + Newton-Raphson).

Covers:
  - NR convergence for hyperelastic and stateful materials
  - Internal force / reaction equilibrium in simple deformations
  - State propagation for path-dependent materials
"""

from __future__ import annotations

import numpy as np
import pytest

from dispsolver.mesh import Mesh
from dispsolver.material import NeoHookean, J2Plasticity
from dispsolver.solver import DynamicSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unit_square_mesh() -> Mesh:
    """1 Q4 element covering [0,1] x [0,1]."""
    mesh = Mesh()
    mesh.add_node(0, 0.0, 0.0)
    mesh.add_node(1, 1.0, 0.0)
    mesh.add_node(2, 1.0, 1.0)
    mesh.add_node(3, 0.0, 1.0)
    mesh.add_element(0, [0, 1, 2, 3], "QUAD4")
    mesh.add_nodeset("left", {0, 3})
    mesh.add_nodeset("right", {1, 2})
    mesh.add_nodeset("bottom", {0, 1})
    return mesh


def make_bc_dofs(mesh) -> np.ndarray:
    """Return BC list: left edge (ux=0, uy=0) + bottom (uy=0)."""
    left = mesh.get_nodeset("left")
    bottom = mesh.get_nodeset("bottom")
    nid_to_idx = mesh.node_id_to_index()
    bc = []
    for nid in left.node_ids:
        base = nid_to_idx[nid] * 2
        bc.extend([base, base + 1])
    for nid in bottom.node_ids:
        if nid not in left.node_ids:
            base = nid_to_idx[nid] * 2
            bc.append(base + 1)
    return np.array(sorted(set(bc)), dtype=np.int32)


def right_ux_dofs(mesh) -> np.ndarray:
    """Return ux DOFs at right edge (nodes 1,2)."""
    nid_to_idx = mesh.node_id_to_index()
    return np.array([nid_to_idx[1] * 2, nid_to_idx[2] * 2], dtype=np.int32)


# ---------------------------------------------------------------------------
# Mesh / mass sanity
# ---------------------------------------------------------------------------

class TestSolverBasics:
    """Low-level sanity checks before running solve_step."""

    def test_create_solver_hyperelastic(self):
        """Hyperelastic solver can be created."""
        mesh = unit_square_mesh()
        mat = NeoHookean()
        solver = DynamicSolver(mesh, mat, rho=1.0)
        assert solver.n_dofs == 8
        assert solver.state is None

    def test_create_solver_plastic(self):
        """Stateful solver initialises internal variables."""
        mesh = unit_square_mesh()
        mat = J2Plasticity(E=1000.0, nu=0.3, sigma_y0=10.0)
        solver = DynamicSolver(mesh, mat, rho=1.0)
        assert solver.material.n_internal_vars == 5
        assert solver.state is not None
        assert solver.state.shape == (1, 4, 5)

    def test_mass_matrix_positive(self):
        """Lumped mass has positive diagonal; total = area * rho."""
        mesh = unit_square_mesh()
        mat = NeoHookean()
        solver = DynamicSolver(mesh, mat, rho=1.0)
        assert np.all(solver.M > 0.0)
        # Each node's mass = rho*area/n_nodes = 0.25 appears at both DOFs (ux, uy)
        assert abs(np.sum(solver.M) - 2.0) < 1e-12

    def test_ext_force_zero_initial(self):
        mesh = unit_square_mesh()
        mat = NeoHookean()
        solver = DynamicSolver(mesh, mat, rho=1.0)
        assert np.all(solver.f_ext == 0.0)


# ---------------------------------------------------------------------------
# Newton-Raphson convergence
# ---------------------------------------------------------------------------

class TestNRConvergence:
    """NR should converge for small steps with both material types."""

    def test_small_step_hyperelastic(self):
        """NeoHookean, 0.1% stretch — NR converges quickly."""
        mesh = unit_square_mesh()
        mat = NeoHookean()
        solver = DynamicSolver(mesh, mat, rho=1.0, material_params={'mu': 100.0, 'lambda': 150.0},
                               max_iter=40, tol=1e-10)

        bc = make_bc_dofs(mesh)
        right = right_ux_dofs(mesh)
        all_bc = np.concatenate([bc, right])
        all_vals = np.zeros(len(all_bc))
        all_vals[len(bc):] = 1e-4  # right edge ux = 1e-4
        solver.set_prescribed_dofs(all_bc, all_vals)

        n_iter = solver.solve_step(dt=1.0)
        assert n_iter > 0, f"NR should converge, got {n_iter}"

    def test_larger_step_neo_hookean(self):
        """10% stretch — NR converges in reasonable iterations."""
        mesh = unit_square_mesh()
        mat = NeoHookean()
        solver = DynamicSolver(mesh, mat, rho=1.0, material_params={'mu': 100.0, 'lambda': 150.0},
                               max_iter=40, tol=1e-10)

        bc = make_bc_dofs(mesh)
        right = right_ux_dofs(mesh)
        all_bc = np.concatenate([bc, right])
        all_vals = np.zeros(len(all_bc))
        all_vals[len(bc):] = 0.1
        solver.set_prescribed_dofs(all_bc, all_vals)

        n_iter = solver.solve_step(dt=1.0)
        assert n_iter > 0, f"NR should converge, got {n_iter}"
        assert n_iter <= 40, f"Took {n_iter} iters, expected <= 40"

    def test_plastic_step_converges(self):
        """Plastic step (2% stretch) — NR converges, eqps > 0."""
        mesh = unit_square_mesh()
        mat = J2Plasticity(E=1000.0, nu=0.3, sigma_y0=10.0)
        solver = DynamicSolver(mesh, mat, rho=1.0, max_iter=40, tol=1e-10)

        bc = make_bc_dofs(mesh)
        right = right_ux_dofs(mesh)
        all_bc = np.concatenate([bc, right])
        all_vals = np.zeros(len(all_bc))
        all_vals[len(bc):] = 0.02
        solver.set_prescribed_dofs(all_bc, all_vals)

        n_iter = solver.solve_step(dt=1.0)
        assert n_iter > 0, f"NR should converge, got {n_iter}"

        eqps = solver.state[0, :, 4]
        assert np.all(eqps > 0.0), f"eqps should be > 0, got {eqps}"

    def test_plastic_larger_converges(self):
        """Plastic step (15% stretch) — NR converges."""
        mesh = unit_square_mesh()
        mat = J2Plasticity(E=1000.0, nu=0.3, sigma_y0=10.0)
        solver = DynamicSolver(mesh, mat, rho=1.0, max_iter=40, tol=1e-10)

        bc = make_bc_dofs(mesh)
        right = right_ux_dofs(mesh)
        all_bc = np.concatenate([bc, right])
        all_vals = np.zeros(len(all_bc))
        all_vals[len(bc):] = 0.15
        solver.set_prescribed_dofs(all_bc, all_vals)

        n_iter = solver.solve_step(dt=1.0)
        assert n_iter > 0, f"NR should converge, got {n_iter}"
        eqps = solver.state[0, :, 4]
        assert np.all(eqps > 0.0)


# ---------------------------------------------------------------------------
# Equilibrium / reaction
# ---------------------------------------------------------------------------

class TestEquilibrium:
    """Internal force and reactions balance at convergence."""

    def test_tensile_reaction(self):
        """Stretch produces tensile reaction at right edge."""
        mesh = unit_square_mesh()
        mat = NeoHookean()
        solver = DynamicSolver(mesh, mat, rho=1.0, material_params={'mu': 100.0, 'lambda': 150.0},
                               max_iter=40, tol=1e-12)

        bc = make_bc_dofs(mesh)
        right = right_ux_dofs(mesh)
        all_bc = np.concatenate([bc, right])
        all_vals = np.zeros(len(all_bc))
        all_vals[len(bc):] = 1e-3
        solver.set_prescribed_dofs(all_bc, all_vals)
        solver.solve_step(dt=1.0)

        R = solver.reaction_forces()
        assert R[right[0]] > 0, f"Tensile reaction expected, got {R[right[0]]:.4e}"

    def test_reaction_equals_f_int_at_bc(self):
        """At BC DOFs, reaction = f_int (since f_ext=0, a≈0)."""
        mesh = unit_square_mesh()
        mat = NeoHookean()
        solver = DynamicSolver(mesh, mat, rho=1.0, material_params={'mu': 100.0, 'lambda': 150.0},
                               max_iter=40, tol=1e-12)

        bc = make_bc_dofs(mesh)
        right = right_ux_dofs(mesh)
        all_bc = np.concatenate([bc, right])
        all_vals = np.zeros(len(all_bc))
        all_vals[len(bc):] = 1e-4
        solver.set_prescribed_dofs(all_bc, all_vals)
        solver.solve_step(dt=1.0)

        R = solver.reaction_forces()
        f_int, _, _ = solver._assemble(solver.u)
        for dof in all_bc:
            expected_R = f_int[dof] + solver.M[dof] * solver.a[dof]
            assert abs(R[dof] - expected_R) < 1e-12, \
                f"DOF {dof}: R={R[dof]:.4e} != expected={expected_R:.4e}"


# ---------------------------------------------------------------------------
# State consistency
# ---------------------------------------------------------------------------

class TestStateConsistency:
    """Internal state propagates correctly through multiple steps."""

    def test_eqps_monotonic(self):
        """eqps increases monotonically over multiple plastic steps."""
        mesh = unit_square_mesh()
        mat = J2Plasticity(E=1000.0, nu=0.3, sigma_y0=10.0, H=100.0)
        solver = DynamicSolver(mesh, mat, rho=1.0, max_iter=40, tol=1e-10)

        bc = make_bc_dofs(mesh)
        right = right_ux_dofs(mesh)
        all_bc = np.concatenate([bc, right])

        prev_eqps = np.zeros(4)
        for stretch in [0.02, 0.04, 0.06, 0.08]:
            all_vals = np.zeros(len(all_bc))
            all_vals[len(bc):] = stretch
            solver.set_prescribed_dofs(all_bc, all_vals)
            n_iter = solver.solve_step(dt=1.0)
            assert n_iter > 0, f"NR failed at stretch={stretch}"

            eqps = solver.state[0, :, 4]
            assert np.all(eqps >= prev_eqps - 1e-15), \
                f"eqps should be monotonic: {prev_eqps} -> {eqps}"
            prev_eqps = eqps.copy()

    def test_elastic_after_plastic_unload(self):
        """After plastic loading, small unload is mostly elastic."""
        mesh = unit_square_mesh()
        mat = J2Plasticity(E=1000.0, nu=0.3, sigma_y0=10.0, H=100.0)
        solver = DynamicSolver(mesh, mat, rho=1.0, max_iter=40, tol=1e-10)

        bc = make_bc_dofs(mesh)
        right = right_ux_dofs(mesh)
        all_bc = np.concatenate([bc, right])

        # Load
        all_vals = np.zeros(len(all_bc))
        all_vals[len(bc):] = 0.10
        solver.set_prescribed_dofs(all_bc, all_vals)
        n_iter = solver.solve_step(dt=1.0)
        assert n_iter > 0

        eqps_loaded = solver.state[0, :, 4].copy()

        # Slight unload (small reduction in prescribed ux)
        all_vals[len(bc):] = 0.095
        solver.set_prescribed_dofs(all_bc, all_vals)
        n_iter = solver.solve_step(dt=1.0)
        assert n_iter > 0

        eqps_unloaded = solver.state[0, :, 4]
        # eqps should not increase during unloading
        assert np.all(eqps_unloaded <= eqps_loaded + 1e-15), \
            f"eqps increased during unload: {eqps_loaded} -> {eqps_unloaded}"
