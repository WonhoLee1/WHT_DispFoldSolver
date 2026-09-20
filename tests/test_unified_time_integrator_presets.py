"""
test_unified_time_integrator_presets.py
========================================
Unit and integration tests for unified 2D/3D time integrator presets
and manual individual parameter overrides.
"""

import pytest
import numpy as np

from dispsolver.solver.time_integrator_presets import resolve_time_integrator, TimeIntegratorConfig
from dispsolver.mesh2d.mesh2d import Mesh2D
from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver2d.dynamic2d import DynamicSolver2D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


def test_preset_resolution_defaults():
    """Verify standard preset defaults in resolve_time_integrator."""
    # Quasi-static
    cfg_qs = resolve_time_integrator("quasistatic")
    assert cfg_qs.is_quasistatic is True
    assert cfg_qs.is_emcs is False
    assert cfg_qs.alpha_hht == 0.0
    assert cfg_qs.beta == 0.25
    assert cfg_qs.gamma == 0.5

    # EMCS
    cfg_emcs = resolve_time_integrator("emcs")
    assert cfg_emcs.is_quasistatic is False
    assert cfg_emcs.is_emcs is True

    # Moderate-1
    cfg_m1 = resolve_time_integrator("moderate-1")
    assert cfg_m1.alpha_hht == -0.05
    assert pytest.approx(cfg_m1.beta, 1e-6) == (1.0 - (-0.05)) ** 2 / 4.0
    assert pytest.approx(cfg_m1.gamma, 1e-6) == 0.5 - (-0.05)

    # Moderate-3 (Abaqus default)
    cfg_m3 = resolve_time_integrator("moderate-3")
    assert cfg_m3.alpha_hht == -0.15
    assert pytest.approx(cfg_m3.beta, 1e-6) == 0.330625
    assert pytest.approx(cfg_m3.gamma, 1e-6) == 0.65

    # Generalized-0.8 (OptiStruct default)
    cfg_g08 = resolve_time_integrator("generalized-0.8")
    assert pytest.approx(cfg_g08.rho_inf, 1e-6) == 0.8
    assert pytest.approx(cfg_g08.alpha_m, 1e-6) == (2 * 0.8 - 1.0) / (0.8 + 1.0)
    assert pytest.approx(cfg_g08.alpha_f, 1e-6) == 0.8 / 1.8


def test_manual_individual_overrides():
    """Verify manual individual parameter overrides over presets."""
    # Override beta and gamma over moderate-3 preset
    cfg = resolve_time_integrator("moderate-3", beta=0.30, gamma=0.60)
    assert cfg.alpha_hht == -0.15
    assert cfg.beta == 0.30
    assert cfg.gamma == 0.60

    # Override alpha_hht manually
    cfg_alpha = resolve_time_integrator("transient", alpha_hht=-0.08)
    assert cfg_alpha.alpha_hht == -0.08
    assert pytest.approx(cfg_alpha.beta, 1e-6) == (1.0 - (-0.08)) ** 2 / 4.0
    assert pytest.approx(cfg_alpha.gamma, 1e-6) == 0.58

    # Override rho_inf manually
    cfg_rho = resolve_time_integrator("quasistatic", rho_inf=0.7)
    assert cfg_rho.rho_inf == 0.7
    assert pytest.approx(cfg_rho.alpha_m, 1e-6) == (2 * 0.7 - 1.0) / 1.7


def test_dynamic_solver_2d_unified_presets():
    """Verify DynamicSolver2D initialization and step solve with custom presets and manual overrides."""
    mesh = Mesh2D()
    mesh.add_node(0, 0.0, 0.0)
    mesh.add_node(1, 1.0, 0.0)
    mesh.add_node(2, 1.0, 1.0)
    mesh.add_node(3, 0.0, 1.0)
    mesh.add_element(0, [0, 1, 2, 3], "CPE4")

    # Test initialization with moderate-2 preset + manual beta override
    solver2d = DynamicSolver2D(
        mesh,
        time_integrator="moderate-2",
        beta=0.31,
        gamma=0.62,
        density=7.85e-9,
    )
    assert solver2d.time_integrator == "moderate-2"
    assert solver2d.alpha_hht == -0.10
    assert solver2d.beta_hht == 0.31
    assert solver2d.gamma_hht == 0.62

    # Fix base nodes
    solver2d.fix_dof(0, 0, 0.0)
    solver2d.fix_dof(0, 1, 0.0)
    solver2d.fix_dof(1, 1, 0.0)

    # Solve dynamic step
    conv, iters = solver2d.solve_step(dt=0.01)
    assert conv is True
    assert iters >= 1


def test_dynamic_solver_3d_unified_presets():
    """Verify DynamicSolver3D initialization and step solve with custom presets and manual overrides."""
    mesh = Mesh3D()
    mesh.add_node(0, 0.0, 0.0, 0.0)
    mesh.add_node(1, 1.0, 0.0, 0.0)
    mesh.add_node(2, 1.0, 1.0, 0.0)
    mesh.add_node(3, 0.0, 1.0, 0.0)
    mesh.add_node(4, 0.0, 0.0, 1.0)
    mesh.add_node(5, 1.0, 0.0, 1.0)
    mesh.add_node(6, 1.0, 1.0, 1.0)
    mesh.add_node(7, 0.0, 1.0, 1.0)
    mesh.add_element(0, [0, 1, 2, 3, 4, 5, 6, 7], "C3D8")

    # Test initialization with optistruct_galpha preset + manual alpha_hht override
    solver3d = DynamicSolver3D(
        mesh,
        time_integrator="optistruct_galpha",
        alpha_hht=-0.12,
        density=7.85e-9,
    )
    assert solver3d.time_integrator == "optistruct_galpha"
    assert solver3d.alpha_hht == -0.12

    # Fix bottom face
    for n in range(4):
        solver3d.fix_dof(n, 0, 0.0)
        solver3d.fix_dof(n, 1, 0.0)
        solver3d.fix_dof(n, 2, 0.0)

    # Solve dynamic step
    conv, iters = solver3d.solve_step(dt=0.01)
    assert conv is True
    assert iters >= 1


def test_2d_3d_api_parity():
    """Verify 100% public API parity between DynamicSolver2D and DynamicSolver3D."""
    mesh2d = Mesh2D()
    mesh2d.add_node(0, 0.0, 0.0)
    mesh2d.add_node(1, 1.0, 0.0)
    mesh2d.add_node(2, 1.0, 1.0)
    mesh2d.add_node(3, 0.0, 1.0)
    mesh2d.add_element(0, [0, 1, 2, 3], "CPE4")
    s2d = DynamicSolver2D(mesh2d, time_integrator="moderate-3", beta=0.33, gamma=0.65)

    mesh3d = Mesh3D()
    mesh3d.add_node(0, 0.0, 0.0, 0.0)
    mesh3d.add_node(1, 1.0, 0.0, 0.0)
    mesh3d.add_node(2, 1.0, 1.0, 0.0)
    mesh3d.add_node(3, 0.0, 1.0, 0.0)
    mesh3d.add_node(4, 0.0, 0.0, 1.0)
    mesh3d.add_node(5, 1.0, 0.0, 1.0)
    mesh3d.add_node(6, 1.0, 1.0, 1.0)
    mesh3d.add_node(7, 0.0, 1.0, 1.0)
    mesh3d.add_element(0, [0, 1, 2, 3, 4, 5, 6, 7], "C3D8")
    s3d = DynamicSolver3D(mesh3d, time_integrator="moderate-3", beta=0.33, gamma=0.65)

    # Check shared API methods exist on both classes
    methods = [
        "add_constraint",
        "set_dirichlet_bc",
        "fix_dof",
        "add_nodal_force",
        "assemble_system",
        "apply_boundary_conditions",
        "solve_step",
        "compute_reaction_forces",
        "compute_section_reactions",
        "compute_strain_energy",
        "compute_total_energy",
        "get_state",
    ]

    for m in methods:
        assert hasattr(s2d, m), f"DynamicSolver2D missing method: {m}"
        assert hasattr(s3d, m), f"DynamicSolver3D missing method: {m}"

    # Check method call results parity
    st2d = s2d.get_state()
    st3d = s3d.get_state()
    assert st2d["time_integrator"] == st3d["time_integrator"] == "moderate-3"

    e2d_tot, e2d_kin, e2d_pot = s2d.compute_total_energy()
    e3d_tot, e3d_kin, e3d_pot = s3d.compute_total_energy()
    assert isinstance(e2d_tot, float)
    assert isinstance(e3d_tot, float)

