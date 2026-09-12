"""
test_section_controls.py
========================
Tests for Abaqus-compatible SectionControls, distortion control barrier,
and anti-inversion safeguards in 3D solid elements and DynamicSolver3D.
"""

import numpy as np
import pytest

from dispsolver.model.section import SectionControls, SolidSection
from dispsolver.model.model import Model
from dispsolver.element3d.c3d8_hybrid_numba import compute_c3d8_hybrid_element_umat_numba
from dispsolver.element3d.c3d8_corotational_numba import compute_c3d8_corotational_element_umat_numba
from dispsolver.material3d.numba_materials import MAT_HYPERELASTIC_NEOHOOKEAN
from dispsolver.mesh3d.mesh3d import Mesh3D
from dispsolver.solver3d.dynamic3d import DynamicSolver3D


def test_section_controls_cae_api_and_array_conversion():
    """Verify Abaqus-compatible CAE API instantiation and array serialization."""
    model = Model("TestModel")

    # 1. Abaqus camelCase kwargs
    sc_abaqus = model.SectionControls(
        name="AbaqusStyleControls",
        distortionControl=True,
        lengthRatio=0.15,
        viscousDamping=0.005,
        antiInversionBarrier=True,
        minDetF=0.03
    )

    assert sc_abaqus.name == "AbaqusStyleControls"
    assert sc_abaqus.distortion_control is True
    assert sc_abaqus.length_ratio == 0.15
    assert sc_abaqus.viscous_damping == 0.005
    assert sc_abaqus.anti_inversion_barrier is True
    assert sc_abaqus.min_det_f == 0.03

    arr = sc_abaqus.to_control_array()
    assert arr.shape == (8,)
    assert arr[0] == 1.0
    assert arr[1] == 0.15
    assert arr[2] == 0.005
    assert arr[3] == 1.0
    assert arr[4] == 0.03

    # 2. Section assignment with controls
    section = model.SolidSection(
        name="PSAHybridSection",
        material="PSA",
        controls=sc_abaqus
    )
    assert section.controls is not None
    assert section.controls.length_ratio == 0.15


def _create_unit_hex_coords():
    """Create a unit cube [0, 1]^3 with standard 8-node hex connectivity."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ], dtype=np.float64)
    return coords


def test_c3d8_hybrid_distortion_control_barrier():
    """Verify that distortion control adds a steep repulsive barrier under extreme compression."""
    coords = _create_unit_hex_coords()

    # Extreme triaxial compression: 95% volume reduction (J ≈ 0.05 < j_crit=0.1)
    # Target scale s such that s^3 = 0.05 -> s ≈ 0.3684
    s = 0.368403
    u_comp = np.zeros(24, dtype=np.float64)
    for i in range(8):
        u_comp[3 * i + 0] = (s - 1.0) * coords[i, 0]
        u_comp[3 * i + 1] = (s - 1.0) * coords[i, 1]
        u_comp[3 * i + 2] = (s - 1.0) * coords[i, 2]

    # Material: Neo-Hookean with mu=0.5, K=50.0
    c10 = 0.25
    d1 = 2.0 / 50.0
    props = np.array([1.0, 50.0, c10, d1], dtype=np.float64)
    sdvs = np.zeros((8, 0), dtype=np.float64)

    # 1. Without distortion control (controls[0] = 0.0)
    ctrl_off = np.zeros(8, dtype=np.float64)
    ctrl_off[0] = 0.0
    f_off, K_off, err_off = compute_c3d8_hybrid_element_umat_numba(
        coords, u_comp, MAT_HYPERELASTIC_NEOHOOKEAN, props, sdvs, dt=1.0, controls=ctrl_off
    )
    assert err_off == 0

    # 2. With distortion control (controls[0] = 1.0, j_crit = 0.1)
    ctrl_on = np.zeros(8, dtype=np.float64)
    ctrl_on[0] = 1.0
    ctrl_on[1] = 0.1
    f_on, K_on, err_on = compute_c3d8_hybrid_element_umat_numba(
        coords, u_comp, MAT_HYPERELASTIC_NEOHOOKEAN, props, sdvs, dt=1.0, controls=ctrl_on
    )
    assert err_on == 0

    # Distortion control should produce significantly stronger compressive repulsive forces and stiffness
    norm_f_off = np.linalg.norm(f_off)
    norm_f_on = np.linalg.norm(f_on)
    trace_K_off = np.trace(K_off)
    trace_K_on = np.trace(K_on)

    assert norm_f_on > norm_f_off * 1.5, f"Expected repulsive barrier force, got on={norm_f_on}, off={norm_f_off}"
    assert trace_K_on > trace_K_off * 1.5, f"Expected barrier stiffness enhancement, got on={trace_K_on}, off={trace_K_off}"


def test_anti_inversion_barrier_error_flag():
    """Verify that anti-inversion barrier triggers error_flag on severe inversion (J < min_det_f)."""
    coords = _create_unit_hex_coords()

    # Complete element inversion / flat crushing (J ≈ 0.005 < min_det_f = 0.02)
    s = 0.17
    u_inv = np.zeros(24, dtype=np.float64)
    for i in range(8):
        u_inv[3 * i + 0] = (s - 1.0) * coords[i, 0]
        u_inv[3 * i + 1] = (s - 1.0) * coords[i, 1]
        u_inv[3 * i + 2] = (s - 1.0) * coords[i, 2]

    props = np.array([1.0, 50.0, 0.25, 0.04], dtype=np.float64)
    sdvs = np.zeros((8, 0), dtype=np.float64)

    # With anti-inversion barrier ON (controls[3] = 1.0, min_det_f = 0.02)
    ctrl_barrier = np.zeros(8, dtype=np.float64)
    ctrl_barrier[0] = 1.0
    ctrl_barrier[1] = 0.1
    ctrl_barrier[3] = 1.0
    ctrl_barrier[4] = 0.02

    _, _, err = compute_c3d8_hybrid_element_umat_numba(
        coords, u_inv, MAT_HYPERELASTIC_NEOHOOKEAN, props, sdvs, dt=1.0, controls=ctrl_barrier
    )
    assert err == 1, "Expected anti-inversion safeguard to trigger error_flag=1 for severely compressed/inverted element"


def test_solver3d_integration_with_section_controls():
    """Verify DynamicSolver3D can assign section controls and assemble correctly."""
    mesh = Mesh3D()
    coords = _create_unit_hex_coords()
    for i in range(8):
        mesh.add_node(i + 1, coords[i, 0], coords[i, 1], coords[i, 2])
    mesh.add_element(1, list(range(1, 9)), elem_type="C3D8H", pid=0)

    solver = DynamicSolver3D(
        mesh=mesh,
        materials={0: {"type": "neohookean", "C10": 0.5, "D1": 0.001}}
    )

    # Verify default controls allocation
    assert solver.elem_controls is not None
    assert solver.elem_controls.shape == (1, 8)
    assert solver.elem_controls[0, 0] == 1.0  # distortion_control default ON

    # Customize controls via SectionControls object
    sc = SectionControls(
        name="CustomControl",
        distortion_control=True,
        length_ratio=0.12,
        viscous_damping=0.01,
        anti_inversion_barrier=True,
        min_det_f=0.025
    )
    solver.set_element_controls(0, sc)
    assert solver.elem_controls[0, 1] == 0.12
    assert solver.elem_controls[0, 2] == 0.01
    assert solver.elem_controls[0, 4] == 0.025

    # Assemble system at rest
    u_0 = np.zeros(24, dtype=np.float64)
    K_g, f_int = solver.assemble_system(u_0)
    assert K_g.shape == (24, 24)
    assert np.allclose(f_int, 0.0)
    assert solver.last_assembly_error == 0
