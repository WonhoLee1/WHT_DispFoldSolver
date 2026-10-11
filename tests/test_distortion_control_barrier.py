"""
test_distortion_control_barrier.py
==================================
Unit and verification tests for Universal Distortion Control:
1. 2D CPE4R Distortion Control barrier activation and restoring force energetics.
2. 3D C3D8R Distortion Control barrier activation under severe compressive pinching.
3. Abaqus *SECTION CONTROLS input reader parsing and binding to models.
"""

import numpy as np
import pytest

from dispsolver.element2d.cpe4r_numba import compute_cpe4r_element_numba
from dispsolver.element3d.c3d8r_numba import compute_c3d8r_element_numba
from dispsolver.model.section import SectionControls
from dispsolver.io.abaqus_reader import read_inp


def test_2d_distortion_control_barrier():
    """Verify that 2D CPE4R activates positive restoring pressure when det(F) < r_c."""
    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float64)
    props = np.array([1000.0, 0.3], dtype=np.float64)  # E=1000, nu=0.3
    sdvs = np.zeros((1, 7), dtype=np.float64)

    # 1. Uncompressed / Mild deformation (det(F) ~ 1.0)
    u_elem_mild = np.zeros(8, dtype=np.float64)
    ctrl_on = SectionControls(name="CTRL1", distortion_control=True, length_ratio=0.20)
    ctrl_arr = ctrl_on.to_control_array()

    fe_mild, Ke_mild, err_mild = compute_cpe4r_element_numba(
        coords, u_elem_mild, 0, props, sdvs, 0.01, elem_controls=ctrl_arr
    )
    assert err_mild == 0
    # No distortion force when flat/uncompressed
    np.testing.assert_allclose(fe_mild, 0.0, atol=1e-12)

    # 2. Extreme compressive pinching: compress in X and Y such that J = 0.05 < r_c = 0.20
    # X stretch = 0.25, Y stretch = 0.20 => J = 0.05
    u_elem_pinch = np.array([
        0.0,   0.0,
        -0.75, 0.0,
        -0.75, -0.80,
        0.0,   -0.80
    ], dtype=np.float64)

    # Solve WITH distortion control
    fe_dist, Ke_dist, err_dist = compute_cpe4r_element_numba(
        coords, u_elem_pinch, 0, props, sdvs, 0.01, elem_controls=ctrl_arr
    )

    # Solve WITHOUT distortion control (distortion_control=False)
    ctrl_off = SectionControls(name="CTRL_OFF", distortion_control=False, length_ratio=0.20)
    fe_nodist, Ke_nodist, err_nodist = compute_cpe4r_element_numba(
        coords, u_elem_pinch, 0, props, sdvs, 0.01, elem_controls=ctrl_off.to_control_array()
    )

    # Distortion barrier MUST add repulsive restoring forces that oppose the compression!
    f_barrier = fe_dist - fe_nodist
    assert np.linalg.norm(f_barrier) > 0.0, "Distortion barrier must produce non-zero restoring force"
    # Tangent stiffness must be strictly stiffer with barrier active
    assert np.trace(Ke_dist) > np.trace(Ke_nodist), "Distortion barrier must stiffen tangent to resist collapse"
    # Symmetry check for Ke_dist
    diff_sym = np.max(np.abs(Ke_dist - Ke_dist.T))
    assert diff_sym < 1e-10, f"Ke_dist must be symmetric, max asymmetry {diff_sym}"


def test_3d_distortion_control_barrier():
    """Verify that 3D C3D8R activates positive restoring pressure when det(F) < r_c."""
    coords = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
    ], dtype=np.float64)
    props = np.array([2000.0, 0.3], dtype=np.float64)  # E=2000, nu=0.3
    sdvs = np.zeros((1, 7), dtype=np.float64)

    ctrl_on = SectionControls(name="CTRL3D", distortion_control=True, length_ratio=0.15)
    ctrl_arr = ctrl_on.to_control_array()

    # Extreme volumetric compression: 80% compression in all 3 directions -> J = 0.2^3 = 0.008 < 0.15
    u_elem_pinch = np.zeros(24, dtype=np.float64)
    for a in range(8):
        u_elem_pinch[3 * a + 0] = -0.80 * coords[a, 0]
        u_elem_pinch[3 * a + 1] = -0.80 * coords[a, 1]
        u_elem_pinch[3 * a + 2] = -0.80 * coords[a, 2]

    # Note signature for compute_c3d8r_element_numba: (Ke, fe, err)
    Ke_dist, fe_dist, err_dist = compute_c3d8r_element_numba(
        coords, u_elem_pinch, 0, props, sdvs, 0.01, elem_controls=ctrl_arr
    )

    ctrl_off = SectionControls(name="CTRL3D_OFF", distortion_control=False, length_ratio=0.15)
    Ke_nodist, fe_nodist, err_nodist = compute_c3d8r_element_numba(
        coords, u_elem_pinch, 0, props, sdvs, 0.01, elem_controls=ctrl_off.to_control_array()
    )

    f_barrier = fe_dist - fe_nodist
    assert np.linalg.norm(f_barrier) > 0.0, "3D Distortion barrier must produce non-zero restoring force"
    assert np.trace(Ke_dist) > np.trace(Ke_nodist), "3D Distortion barrier must stiffen tangent against collapse"
    diff_sym = np.max(np.abs(Ke_dist - Ke_dist.T))
    assert diff_sym < 1e-10, f"3D Ke_dist must be symmetric, max asymmetry {diff_sym}"


def test_section_controls_inp_parsing():
    """Verify that *SECTION CONTROLS keyword is correctly parsed from Abaqus deck."""
    deck = """*HEADING
Model with SectionControls
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPE4R, ELSET=EALL
1, 1, 2, 3, 4
*SECTION CONTROLS, NAME=MY_CTRL, DISTORTION CONTROL=YES, LENGTH RATIO=0.15, HOURGLASS=ENHANCED
*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1, CONTROLS=MY_CTRL
1.0
*MATERIAL, NAME=MAT1
*ELASTIC
1000.0, 0.3
*STEP
*STATIC
0.1, 1.0
*END STEP
"""
    model = read_inp(deck, dim=2)

    assert "MY_CTRL" in model.section_controls
    ctrl = model.section_controls["MY_CTRL"]
    assert ctrl.distortion_control is True
    assert abs(ctrl.length_ratio - 0.15) < 1e-8
    assert ctrl.hourglass_control.upper() == "ENHANCED"

    # Flatten assembly and verify elem_controls array propagation
    assembly = model.root_assembly
    sys = assembly.build_solver_system(model=model)
    assert sys.elem_controls is not None
    assert sys.elem_controls.shape[0] == 1
    # Check length_ratio set to 0.15 and distortion_control ON
    assert sys.elem_controls[0, 0] == 1.0
    assert abs(sys.elem_controls[0, 1] - 0.15) < 1e-8
