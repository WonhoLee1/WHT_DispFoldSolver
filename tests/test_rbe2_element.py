"""
test_rbe2_element.py
====================
TDD RED tests for RBE2HingeElement.

These tests define the interface contract for the RBE2 hinge element. They are
expected to FAIL (RED) until Tasks 3–4 implement the actual element logic.

The element will use static condensation (Q4_EAS pattern) to eliminate internal
DOFs (θ and Lagrange multipliers λ), exposing only the 2·(m+1) translational
DOFs to the global system.
"""

import numpy as np
import pytest


class TestRBE2ElementInterface:
    """Verify the RBE2HingeElement interface contract.

    All tests expect ImportError or NotImplementedError at this stage
    (TDD RED). They will turn GREEN once Tasks 3-4 fill in the implementation.
    """

    def test_compute_contributions_exists(self):
        """Class RBE2HingeElement must have a compute_contributions method."""
        from dispsolver.element.rbe2 import RBE2HingeElement

        assert hasattr(RBE2HingeElement, "compute_contributions"), (
            "RBE2HingeElement must define compute_contributions"
        )

    def test_returns_f_e_K_e_state(self):
        """compute_contributions must return (f_e, K_e, state_new) tuple.

        For 1 slave: 4 external DOF (master x,y + slave x,y).
        f_e: (4,), K_e: (4,4), state_new: RBE2State (or None).
        """
        from dispsolver.element.rbe2 import RBE2HingeElement

        master_id = 0
        slave_ids = [1]
        coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

        elem = RBE2HingeElement(master_id, slave_ids, coords)
        u_elem = np.zeros(4, dtype=np.float64)
        f_e, K_e, state_new = elem.compute_contributions(coords, u_elem, None)

        assert isinstance(f_e, np.ndarray), "f_e must be numpy array"
        assert f_e.shape == (4,), f"f_e shape expected (4,), got {f_e.shape}"
        assert isinstance(K_e, np.ndarray), "K_e must be numpy array"
        assert K_e.shape == (4, 4), f"K_e shape expected (4,4), got {K_e.shape}"
        assert state_new is not None, "state_new must not be None"

    def test_rigid_translation(self):
        """Rigid translation produces zero internal force.

        Master (0,0) and slave (1,0) translated by (0.5, 0.3).
        The constraint g = 0 should give f_e ≈ 0.
        """
        from dispsolver.element.rbe2 import RBE2HingeElement

        master_id = 0
        slave_ids = [1]
        coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

        elem = RBE2HingeElement(master_id, slave_ids, coords)
        u_elem = np.array([0.5, 0.3, 0.5, 0.3], dtype=np.float64)
        f_e, K_e, _ = elem.compute_contributions(coords, u_elem, None)

        assert np.allclose(f_e, 0.0, atol=1e-10), (
            f"Rigid translation should give zero force, got norm={np.linalg.norm(f_e):.2e}"
        )

    def test_rigid_rotation(self):
        """Small rigid rotation produces ~zero internal force.

        Master (0,0) fixed, slave (1,0) rotated by Δθ=0.01 rad.
        Slave displacement: u_s = (cos(Δθ)-1, sin(Δθ)).
        """
        from dispsolver.element.rbe2 import RBE2HingeElement

        master_id = 0
        slave_ids = [1]
        coords = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)

        elem = RBE2HingeElement(master_id, slave_ids, coords)
        dtheta = 0.01
        u_s = np.array([np.cos(dtheta) - 1.0, np.sin(dtheta)])
        u_elem = np.array([0.0, 0.0, u_s[0], u_s[1]], dtype=np.float64)
        f_e, K_e, _ = elem.compute_contributions(coords, u_elem, None)

        assert np.allclose(f_e, 0.0, atol=1e-8), (
            f"Rigid rotation should give ~zero force, got norm={np.linalg.norm(f_e):.2e}"
        )


class TestSolverIntegration:
    """Verify DynamicSolver correctly integrates RBE2HingeElement assembly.

    The solver must:
    - Accept rbe2_elements parameter without error.
    - Call compute_contributions during the Newton loop.
    - Accumulate RBE2 forces into f_int and stiffness into K_T.
    - Persist RBE2 element state across converged steps.
    """

    @pytest.fixture
    def two_quad_mesh(self):
        """Two QUAD4 elements with a hinge master at (0,0).

        Node layout (node_id: coordinates):
        0: (-2, 0)  1: (-2, 2)
        2: ( 0, 0)  3: ( 0, 2)
        4: ( 2, 0)  5: ( 2, 2)

        Elements: [QUAD4 (0,2,3,1)] left, [QUAD4 (2,4,5,3)] right.
        RBE2 hinge: master=2, slaves=[0] (left wing tip).
        """
        from dispsolver.mesh import Mesh
        mesh = Mesh()
        mesh.add_node(0, -2.0, 0.0)
        mesh.add_node(1, -2.0, 2.0)
        mesh.add_node(2,  0.0, 0.0)
        mesh.add_node(3,  0.0, 2.0)
        mesh.add_node(4,  2.0, 0.0)
        mesh.add_node(5,  2.0, 2.0)
        mesh.add_element(0, [0, 2, 3, 1], 'QUAD4')
        mesh.add_element(1, [2, 4, 5, 3], 'QUAD4')
        return mesh

    def test_accepts_rbe2_elements(self, two_quad_mesh):
        """Solver constructor accepts rbe2_elements with zero constraints."""
        from dispsolver.element.rbe2 import RBE2HingeElement
        from dispsolver.solver import DynamicSolver
        from dispsolver.material import J2Plasticity

        coords = two_quad_mesh.nodes_array()
        rbe2 = RBE2HingeElement(2, [0], coords)
        mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)

        solver = DynamicSolver(
            two_quad_mesh, mat, rho=1000.0,
            constraints=[],
            rbe2_elements=[rbe2],
            max_iter=5, tol=1e-3,
            element_type='Q4', mode='quasistatic',
        )
        assert solver.rbe2_elements == [rbe2]
        assert solver.n_extra == 0     # no KKT extra DOFs
        assert solver.n_lambdas == 0   # no KKT multipliers

    def test_zero_load_no_crash(self, two_quad_mesh):
        """Solver step with zero load converges trivially; RBE2 assembly runs."""
        from dispsolver.element.rbe2 import RBE2HingeElement
        from dispsolver.solver import DynamicSolver
        from dispsolver.material import J2Plasticity

        coords = two_quad_mesh.nodes_array()
        nid_to_idx = two_quad_mesh.node_id_to_index()
        rbe2 = RBE2HingeElement(2, [0], coords)
        mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)

        solver = DynamicSolver(
            two_quad_mesh, mat, rho=1000.0,
            constraints=[], rbe2_elements=[rbe2],
            max_iter=5, tol=1e-3,
            element_type='Q4', mode='quasistatic',
        )
        # Fix master (2) and left edge (0,1)
        solver.set_prescribed_dofs(
            [nid_to_idx[2]*2, nid_to_idx[2]*2+1],
            [0.0, 0.0],
        )
        # Zero load step → converges in 1 iteration (no crash)
        n_iter = solver.solve_step(1.0)
        assert n_iter >= 0, f"Solver should converge with zero load, got {n_iter}"
        # RBE2 element state should be persisted
        assert rbe2.state is not None, "RBE2 state must be persisted after convergence"

    def test_state_persists_across_steps(self, two_quad_mesh):
        """RBE2 element state persists across multiple converged time steps."""
        from dispsolver.element.rbe2 import RBE2HingeElement
        from dispsolver.solver import DynamicSolver
        from dispsolver.material import J2Plasticity
        from dispsolver.load import Amplitude

        coords = two_quad_mesh.nodes_array()
        nid_to_idx = two_quad_mesh.node_id_to_index()
        rbe2 = RBE2HingeElement(2, [0], coords)
        mat = J2Plasticity(E=4000.0, nu=0.3, sigma_y0=80.0, H=620.0)

        solver = DynamicSolver(
            two_quad_mesh, mat, rho=1000.0,
            constraints=[],
            rbe2_elements=[rbe2],
            max_iter=10, tol=1e-3,
            element_type='Q4', mode='quasistatic',
        )
        # Fix master (2)
        solver.set_prescribed_dofs(
            [nid_to_idx[2]*2, nid_to_idx[2]*2+1],
            [0.0, 0.0],
        )
        # Step 1
        solver.solve_step(0.5)
        state_1 = rbe2.state
        # Step 2 (zero load, should converge)
        solver.solve_step(0.5)
        state_2 = rbe2.state
        # RBE2 internal state (λ accumulated penalty) should persist
        # and may evolve if any constraint gap opens
        assert state_2 is not None, "RBE2 state must persist after step 2"
        # The state object identity may differ across steps (new state is
        # computed each compute_contributions call), but the state must
        # not be None after convergence.
        assert state_2 is not None
