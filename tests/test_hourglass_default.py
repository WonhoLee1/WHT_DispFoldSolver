"""*SECTION CONTROLS, HOURGLASS= -- the Keywords Guide default when it is not given.

Not given: STIFFNESS, except ENHANCED for hyperelastic materials. a450070 had made it ENHANCED for every
material; for C3D8R that is the constant G*Le/12 stabilization and it locks pure bending (ratio 0.498).
An explicit HOURGLASS= is always honoured.
"""
import numpy as np
import pytest

from dispsolver.material3d.numba_materials import (
    HG_UNSET, MAT_HYPERELASTIC_ARRUDA_BOYCE, MAT_HYPERELASTIC_NEOHOOKEAN, MAT_HYPERELASTIC_YEOH,
    MAT_HYPER_VISCOELASTIC, MAT_J2_PLASTICITY, MAT_LINEAR_ELASTIC, resolve_hourglass_mode)
from dispsolver.model.section import SectionControls


@pytest.mark.parametrize("mat", [MAT_LINEAR_ELASTIC, MAT_J2_PLASTICITY])
def test_not_given_is_stiffness_for_non_hyperelastic(mat):
    assert resolve_hourglass_mode(HG_UNSET, mat) == 1.0


@pytest.mark.parametrize("mat", [MAT_HYPERELASTIC_NEOHOOKEAN, MAT_HYPERELASTIC_ARRUDA_BOYCE,
                                 MAT_HYPERELASTIC_YEOH, MAT_HYPER_VISCOELASTIC])
def test_not_given_is_enhanced_for_hyperelastic(mat):
    assert resolve_hourglass_mode(HG_UNSET, mat) == 0.0


@pytest.mark.parametrize("mat", [MAT_LINEAR_ELASTIC, MAT_HYPERELASTIC_NEOHOOKEAN])
@pytest.mark.parametrize("code", [0.0, 1.0, 2.0])
def test_an_explicit_choice_is_never_overridden(mat, code):
    assert resolve_hourglass_mode(code, mat) == code


def test_section_controls_encode_not_given_distinctly():
    arr = SectionControls(name="C").to_control_array()
    assert arr[5] == HG_UNSET
    assert SectionControls(name="C", hourglass_control="ENHANCED").to_control_array()[5] == 0.0
    assert SectionControls(name="C", hourglass_control="STIFFNESS").to_control_array()[5] == 1.0
