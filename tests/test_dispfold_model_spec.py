"""
test_dispfold_model_spec.py
===========================
TDD Unit Verification Suite for Seam 1: Unified Model Specifications and Analytical Mechanics.

Verifies:
1. Data structures: LayerSpec, DisplayStackConfig, FoldingCaseSpec instantiation and field integrity.
2. Analytical composite beam theory: Free-slip bending stiffness (D_slip), monolithic fully-bonded
   bending stiffness (D_mono), and neutral axis (y_NA) against exact theoretical derivations.
3. Presets: teardrop, compact_4, u_shape, compact_waterdrop, crease_zero_gap.
4. Intelligent Contact Guard logic: requires_self_contact() determination.
"""

from __future__ import annotations
import pytest
import numpy as np

from example_disp_fold.model_spec import (
    LayerSpec,
    MaterialSpec,
    HingeGeometrySpec,
    MeshSpec,
    SimulationCaseSpec,
    UnifiedFoldModelConfig,
    DisplayStackConfig,
    FoldingCaseSpec,
)


def test_layer_spec_creation():
    """Verify LayerSpec creation and field properties."""
    layer = LayerSpec(
        name="PSA_Sample",
        thickness=0.030,
        material_name="PSA",
        is_void_in_hinge=True,
        elem_type_2d="CPE4H_CR",
        elem_type_3d="C3D8H_CR",
        ny_elements=2,
        color="#e67e22",
    )
    assert layer.name == "PSA_Sample"
    assert layer.thickness == 0.030
    assert layer.material_name == "PSA"
    assert layer.is_void_in_hinge is True
    assert layer.elem_type_2d == "CPE4H_CR"
    assert layer.elem_type_3d == "C3D8H_CR"
    assert layer.ny_elements == 2
    assert layer.color == "#e67e22"


def test_folding_case_spec_creation():
    """Verify FoldingCaseSpec (alias of SimulationCaseSpec) instantiation and defaults."""
    case = FoldingCaseSpec(
        dim="2D",
        preset_name="compact_4",
        theta_max_deg=90.0,
        total_time=1.0,
        dt_init=0.005,
        dt_min=1.0e-5,
        dt_max=0.02,
        solution_preset="ADAPTIVE",
        enable_predictor=True,
        tie_method="MPC",
        contact_mode="AUTO",
    )
    assert case.dim == "2D"
    assert case.preset_name == "compact_4"
    assert case.theta_max_deg == 90.0
    assert case.total_time == 1.0
    assert case.enable_predictor is True
    assert case.tie_method == "MPC"
    assert case.contact_mode == "AUTO"


def test_display_stack_config_basics():
    """Verify DisplayStackConfig instantiation, dimension setting, and layer modification."""
    config = DisplayStackConfig(preset="compact_4", dim="2D")
    assert config.simulation_case.dim == "2D"
    assert config.simulation_case.preset_name == "compact_4"
    assert len(config.layers) == 4

    # Dimension switching
    config.set_dimension("3D")
    assert config.simulation_case.dim == "3D"
    config.set_dimension("2D")
    assert config.simulation_case.dim == "2D"

    with pytest.raises(ValueError):
        config.set_dimension("1D")

    # Layer modification
    orig_thickness = config.layers[0].thickness
    config.modify_layer(0, thickness=0.080)
    assert config.layers[0].thickness == 0.080
    config.modify_layer(0, thickness=orig_thickness)

    with pytest.raises(IndexError):
        config.modify_layer(999, thickness=0.05)


def test_compact_4_bending_stiffness_analytical():
    """Verify 4-layer compact stackup bending stiffness and neutral axis against exact theory.

    Stackup (bottom to top):
    - Layer 0: PET (t=0.060 mm, E=4000.0 MPa, nu=0.30)
    - Layer 1: PSA (t=0.030 mm, E=0.05 MPa, nu=0.499)
    - Layer 2: PET (t=0.060 mm, E=4000.0 MPa, nu=0.30)
    - Layer 3: GLASS (t=0.060 mm, E=70000.0 MPa, nu=0.22)

    Exact Parallel Axis Theorem Derivation (width b = 1.0 mm):
    1. Effective plane strain moduli:
       E'_PET = 4000 / (1 - 0.3^2) = 4395.6044 MPa
       E'_PSA = 0.05 / (1 - 0.499^2) = 0.066578 MPa
       E'_GLASS = 70000 / (1 - 0.22^2) = 73560.3195 MPa
    2. Layer areas and centroids:
       L0 (PET):   t=0.060, y_c = 0.030, EA = 263.7363 N
       L1 (PSA):   t=0.030, y_c = 0.075, EA = 0.0020 N
       L2 (PET):   t=0.060, y_c = 0.120, EA = 263.7363 N
       L3 (GLASS): t=0.060, y_c = 0.180, EA = 4413.6192 N
    3. Neutral axis:
       sum(EA) = 4941.0937 N
       sum(EA * y_c) = 834.0120 N*mm
       y_NA = 834.0120 / 4941.0937 = 0.168791 mm
    4. Free-slip stiffness:
       D_slip = sum(E'_i * b * t_i^3 / 12) = 1.482328 N*mm^2
    5. Monolithic bonded stiffness:
       D_mono = D_slip + sum(EA_i * (y_c_i - y_NA)^2) = 7.745056 N*mm^2
    """
    config = DisplayStackConfig(preset="compact_4", dim="2D")
    assert pytest.approx(config.total_thickness, rel=1e-6) == 0.210

    d_slip, d_mono, y_na = config.compute_composite_bending_stiffness(width=1.0)

    # Hand-derived exact analytical values
    expected_y_na = 0.16879097
    expected_d_slip = 1.4823284
    expected_d_mono = 7.7450562

    assert pytest.approx(y_na, rel=1e-5) == expected_y_na
    assert pytest.approx(d_slip, rel=1e-5) == expected_d_slip
    assert pytest.approx(d_mono, rel=1e-5) == expected_d_mono

    # Fundamental mechanical hierarchy
    assert d_slip < d_mono
    assert 0.0 < y_na < config.total_thickness


def test_14_layer_teardrop_bending_stiffness():
    """Verify 14-layer canonical display stackup bending stiffness and neutral axis."""
    config = DisplayStackConfig(preset="teardrop", dim="3D")
    assert len(config.layers) == 14
    # 7 PSA (30 um) + 6 PET (60 um) + 1 GLASS (60 um) = 0.210 + 0.360 + 0.060 = 0.630 mm
    assert pytest.approx(config.total_thickness, rel=1e-6) == 0.630

    d_slip, d_mono, y_na = config.compute_composite_bending_stiffness(width=1.0)

    # 14-layer analytical values
    expected_y_na = 0.516868
    expected_d_slip = 1.798812
    expected_d_mono = 154.761994

    assert pytest.approx(y_na, rel=1e-4) == expected_y_na
    assert pytest.approx(d_slip, rel=1e-4) == expected_d_slip
    assert pytest.approx(d_mono, rel=1e-4) == expected_d_mono
    assert d_slip < d_mono
    assert 0.0 < y_na < config.total_thickness


def test_presets_and_requires_self_contact():
    """Verify all predefined presets and their contact guard requirements.

    Contact Guard Rules:
    - hinge_pivot_x >= 0.8 mm: Teardrop open loop, top surfaces do not collide -> requires_self_contact() == False
    - hinge_pivot_x < 0.8 mm or crease: Semicircle / slim loop / zero-gap crease -> requires_self_contact() == True
    """
    # 1. Teardrop (pivot = 1.4 mm >= 0.8 mm)
    cfg_tear = DisplayStackConfig(preset="teardrop")
    assert cfg_tear.geometry.hinge_pivot_x == 1.4
    assert cfg_tear.requires_self_contact() is False
    assert cfg_tear.resolve_contact_guard() == "OFF"

    # 2. Compact 4-layer (pivot = 1.4 mm >= 0.8 mm)
    cfg_c4 = DisplayStackConfig(preset="compact_4")
    assert cfg_c4.geometry.hinge_pivot_x == 1.4
    assert cfg_c4.requires_self_contact() is False
    assert cfg_c4.resolve_contact_guard() == "OFF"

    # 3. U-Shape (pivot = 0.0 mm < 0.8 mm)
    cfg_u = DisplayStackConfig(preset="u_shape")
    assert cfg_u.geometry.hinge_pivot_x == 0.0
    assert cfg_u.requires_self_contact() is True
    assert cfg_u.resolve_contact_guard() == "DUAL_MORTAR"

    # 4. Compact Waterdrop (pivot = 0.75 mm < 0.8 mm)
    cfg_cwd = DisplayStackConfig(preset="compact_waterdrop")
    assert cfg_cwd.geometry.hinge_pivot_x == 0.75
    assert cfg_cwd.requires_self_contact() is True
    assert cfg_cwd.resolve_contact_guard() == "DUAL_MORTAR"

    # 5. Crease Zero Gap (pivot = 0.0 mm, contact_mode = DUAL_MORTAR)
    cfg_zero = DisplayStackConfig(preset="crease_zero_gap")
    assert cfg_zero.geometry.hinge_pivot_x == 0.0
    assert cfg_zero.simulation_case.contact_mode == "DUAL_MORTAR"
    assert cfg_zero.requires_self_contact() is True
    assert cfg_zero.resolve_contact_guard() == "DUAL_MORTAR"

    # Invalid preset
    with pytest.raises(ValueError):
        DisplayStackConfig(preset="non_existent_preset")
