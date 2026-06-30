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
