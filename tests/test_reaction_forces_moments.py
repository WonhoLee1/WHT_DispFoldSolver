"""
tests/test_reaction_forces_moments.py
=====================================
Unit tests for reaction forces, moments, and strain energy calculations
in DynamicSolver2D and DynamicSolver3D.
"""

import pytest
import numpy as np
from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


def test_2d_cantilever_reaction_moment():
    """Verify that 2D clamped root reaction forces and moments satisfy statics."""
    mesh = Mesh2D()
    # 2x1 elements beam, L = 10.0, H = 2.0 (y from -1.0 to 1.0, mid at y=0)
    # 6 nodes
    mesh.add_node(1, 0.0, -1.0)
    mesh.add_node(2, 5.0, -1.0)
    mesh.add_node(3, 10.0, -1.0)
    mesh.add_node(4, 0.0, 1.0)
    mesh.add_node(5, 5.0, 1.0)
    mesh.add_node(6, 10.0, 1.0)

    mesh.add_element(1, [1, 2, 5, 4], elem_type="CPE4", pid=1)
    mesh.add_element(2, [2, 3, 6, 5], elem_type="CPE4", pid=1)

    materials = {1: {"E": 1000.0, "nu": 0.25}}
    solver = DynamicSolver2D(mesh, materials=materials, nlgeom=False)

    # Clamp root (nodes 1 and 4 at x=0)
    solver.fix_dof(1, 0, 0.0)
    solver.fix_dof(1, 1, 0.0)
    solver.fix_dof(4, 0, 0.0)
    solver.fix_dof(4, 1, 0.0)

    # Prescribe downward displacement of -0.1 at tip (nodes 3, 6)
    delta = -0.1
    solver.fix_dof(3, 1, delta)
    solver.fix_dof(6, 1, delta)

    conv, iters = solver.solve_step(dt=1.0)
    assert conv, "Solve failed to converge"

    # Compute reactions
    R = solver.compute_reaction_forces()
    assert len(R) == solver.num_dofs

    root_sec = solver.compute_section_reactions([1, 4], center_pt=(0.0, 0.0))
    tip_sec = solver.compute_section_reactions([3, 6], center_pt=(10.0, 0.0))

    # Static equilibrium: sum of vertical forces = 0
    # Root upward force + Tip downward force = 0
    assert abs(root_sec["Fy"] + tip_sec["Fy"]) < 1e-4

    # Moment equilibrium about origin:
    # M_root + 10.0 * F_tip_y + M_tip = 0
    M_total = root_sec["Mz"] + tip_sec["Mz"] + 10.0 * tip_sec["Fy"]
    assert abs(M_total) < 1e-3

    # Strain energy must be positive
    U = solver.compute_strain_energy()
    assert U > 0.0


def test_3d_cantilever_reaction_moment():
    """Verify that 3D clamped root reaction forces and moments satisfy statics."""
    mesh = Mesh3D()
    # 1 element box, x in [0, 10], y in [0, 1], z in [-1, 1] (mid at z=0)
    mesh.add_node(1, 0.0, 0.0, -1.0)
    mesh.add_node(2, 10.0, 0.0, -1.0)
    mesh.add_node(3, 10.0, 1.0, -1.0)
    mesh.add_node(4, 0.0, 1.0, -1.0)
    mesh.add_node(5, 0.0, 0.0, 1.0)
    mesh.add_node(6, 10.0, 0.0, 1.0)
    mesh.add_node(7, 10.0, 1.0, 1.0)
    mesh.add_node(8, 0.0, 1.0, 1.0)

    mesh.add_element(1, [1, 2, 3, 4, 5, 6, 7, 8], elem_type="C3D8", pid=1)

    solver = DynamicSolver3D(mesh, material_params={"E": 1000.0, "nu": 0.25}, nlgeom=True)
    solver.materials = {1: {"E": 1000.0, "nu": 0.25}}
    solver._setup_numba_topology()


    # Clamp root (nodes 1, 4, 5, 8 at x=0)
    root_nodes = [1, 4, 5, 8]
    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)
        solver.fix_dof(nid, 2, 0.0)

    # Prescribe downward displacement uz = -0.1 at tip (nodes 2, 3, 6, 7 at x=10)
    tip_nodes = [2, 3, 6, 7]
    for nid in tip_nodes:
        solver.fix_dof(nid, 2, -0.1)

    conv, iters = solver.solve_step(dt=1.0)
    assert conv, "Solve failed to converge"

    root_sec = solver.compute_section_reactions(root_nodes, center_pt=(0.0, 0.5, 0.0))
    tip_sec = solver.compute_section_reactions(tip_nodes, center_pt=(10.0, 0.5, 0.0))

    # Equilibrium of vertical force Fz
    assert abs(root_sec["Fz"] + tip_sec["Fz"]) < 1e-4

    # Pitch moment equilibrium My:
    # M_root_y + 10.0 * (-Fz_tip) + M_tip_y = 0
    # In 3D section reactions: My = dz * Fx - dx * Fz. At x=10, -dx * Fz = -10 * Fz.
    # Therefore root_sec["My"] + tip_sec["My"] - 10.0 * tip_sec["Fz"] should be ~ 0
    # Note tip_sec["My"] was computed about center (10, 0.5, 0), so dx=0 for tip nodes.
    # Global moment about origin (small geometric nonlinear shift due to finite deflection):
    M_global_y = root_sec["My"] + tip_sec["My"] - 10.0 * tip_sec["Fz"]
    assert abs(M_global_y) < 0.02, f"M_global_y error = {M_global_y}"

    # Strain energy must be positive
    U = solver.compute_strain_energy()
    assert U > 0.0


def test_2d_large_rotation_pure_moment_equilibrium():
    """Verify that in large rotation pure moment bending, root and tip section moments match."""
    from benchmark_element.benchmark_pure_moment_rollup import make_2d_mesh, compute_roll_up_displacements, PID_PET, PID_PSA, E_PET, NU_PET, E_PSA, NU_PSA, Z_MID

    mesh, root_nodes, tip_nodes, _ = make_2d_mesh("CPE4I", "CPE4H")
    materials = {PID_PET: {"E": E_PET, "nu": NU_PET}, PID_PSA: {"E": E_PSA, "nu": NU_PSA}}
    solver = DynamicSolver2D(mesh, materials=materials, nlgeom=True)

    for nid in root_nodes:
        solver.fix_dof(nid, 0, 0.0)
        solver.fix_dof(nid, 1, 0.0)

    # Smoothly ramp to 45 deg in 5 steps to maintain pure moment statics
    for s in range(1, 6):
        frac = s / 5.0
        ramp = frac * frac * (3.0 - 2.0 * frac)
        theta_s = ramp * np.radians(45.0)
        for nid in tip_nodes:
            node = mesh.nodes[nid]
            ux_t, uy_t = compute_roll_up_displacements(node.y, theta_s)
            solver.fix_dof(nid, 0, ux_t)
            solver.fix_dof(nid, 1, uy_t)
        conv, _ = solver.solve_step(dt=1.0, max_iters=25)
        assert conv, f"Solve failed at step {s}"

    root_rx = solver.compute_section_reactions(root_nodes, center_pt=(0.0, Z_MID))
    tip_rx = solver.compute_section_reactions(tip_nodes)  # automatic centroid in deformed current coords

    # Magnitudes of root and tip moments should match within 5%
    m_root = abs(root_rx["Mz"])
    m_tip = abs(tip_rx["Mz"])
    diff_pct = abs(m_root - m_tip) / m_root * 100.0
    assert diff_pct < 5.0, f"Root ({m_root:.4f}) and Tip ({m_tip:.4f}) moment discrepancy too high ({diff_pct:.2f}%)"
