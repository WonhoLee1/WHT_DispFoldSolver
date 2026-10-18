"""Tests for P1 reader keywords: *CONTROLS, *CONTACT CONTROLS, *CONTACT STABILIZATION, *SPRING, *DASHPOT."""

import pytest
from dispsolver.io.abaqus_reader import _parse_source, execute_inp
from dispsolver.io.keyword_registry import GLOBAL_KEYWORD_REGISTRY


def test_keyword_registration():
    """Verify all 5 keywords are registered in GLOBAL_KEYWORD_REGISTRY."""
    expected = {
        "CONTROLS": "FULL",
        "CONTACT CONTROLS": "FULL",
        "CONTACT STABILIZATION": "FULL",
        "SPRING": "PARTIAL",
        "DASHPOT": "PARTIAL",
    }
    for kw, expected_support in expected.items():
        spec = GLOBAL_KEYWORD_REGISTRY.get_handler(kw)
        assert spec is not None, f"Keyword '{kw}' must be registered"
        assert spec.support == expected_support, f"Keyword '{kw}' support must be {expected_support}, got {spec.support}"


def test_controls_time_incrementation():
    """Verify *CONTROLS, PARAMETERS=TIME INCREMENTATION updates SolutionControl."""
    deck = """*HEADING
Test Controls Time Incrementation
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
*STEP, NAME=Step-1
*STATIC
0.1, 1.0
*CONTROLS, PARAMETERS=TIME INCREMENTATION
20, 18, 11, 30, 3, 5, 1.5, 0.3
*END STEP
"""
    reader, model = _parse_source(deck, dim=3)
    step = model.steps["Step-1"]
    assert hasattr(step, "control")
    t = step.control.time_incrementation
    assert t.I_0 == 20
    assert t.I_R == 18
    assert t.I_P == 11
    assert t.I_C == 30
    assert t.I_L == 3
    assert t.I_G == 5
    assert t.D_G == 1.5
    assert t.D_C == 0.3


def test_controls_field_and_linesearch():
    """Verify *CONTROLS, PARAMETERS=FIELD and LINE SEARCH update SolutionControl."""
    deck = """*HEADING
Test Controls Field & LineSearch
*NODE
1, 0.0, 0.0, 0.0
*STEP, NAME=Step-1
*STATIC
*CONTROLS, PARAMETERS=FIELD, CONVERGENCE CHECK=STANDARD
0.01, 0.02, 10.0, 0.05, 0.04, 0.002
*CONTROLS, PARAMETERS=LINE SEARCH
6, 1.2, 0.0005, 0.3
*END STEP
"""
    reader, model = _parse_source(deck, dim=3)
    step = model.steps["Step-1"]
    f = step.control.field
    assert f.R_n == 0.01
    assert f.C_n == 0.02
    assert f.q_0 == 10.0
    assert f.c_0 == 0.05
    assert f.R_p == 0.04
    assert f.C_a == 0.002

    ls = step.control.line_search
    assert ls.N_ls == 6
    assert ls.s_ls == 1.2
    assert ls.f_ls == 0.0005
    assert ls.eta_ls == 0.3


def test_controls_discontinuous_and_reset():
    """Verify *CONTROLS, ANALYSIS=DISCONTINUOUS and RESET switch presets."""
    deck = """*HEADING
Test Controls Discontinuous
*NODE
1, 0.0, 0.0, 0.0
*STEP, NAME=Step-1
*STATIC
*CONTROLS, ANALYSIS=DISCONTINUOUS
*END STEP
*STEP, NAME=Step-2
*STATIC
*CONTROLS, RESET
*END STEP
"""
    reader, model = _parse_source(deck, dim=3)
    step1 = model.steps["Step-1"]
    assert step1.control.preset == "RELAXED"

    step2 = model.steps["Step-2"]
    assert step2.control.preset == "STANDARD"


def test_contact_controls_parsing():
    """Verify *CONTACT CONTROLS registers in Model.contact_controls."""
    deck = """*HEADING
Test Contact Controls
*CONTACT CONTROLS, NAME=MyContCtrl, STABILIZE=2.5, APPROACH
1.5e-3, 0.1, 0.05, 0.2
"""
    reader, model = _parse_source(deck, dim=3)
    assert "MyContCtrl" in model.contact_controls
    cc = model.contact_controls["MyContCtrl"]
    assert cc.name == "MyContCtrl"
    assert cc.factor == 2.5
    assert cc.damping_coefficient == 1.5e-3
    assert cc.tangent_fraction == 0.1
    assert cc.clearance == 0.05
    assert cc.ramp_down_factor == 0.2
    assert cc.approach is True


def test_contact_stabilization_parsing():
    """Verify *CONTACT STABILIZATION registers in Model.contact_stabilizations."""
    deck = """*HEADING
Test Contact Stabilization
*CONTACT STABILIZATION, NAME=MyStab, DEFINITION=DAMPING COEFFICIENT, SCALE FACTOR=1.5, TANGENT FRACTION=0.05
2.0e-4, 0.05, 0.02, 0.1
"""
    reader, model = _parse_source(deck, dim=3)
    assert "MyStab" in model.contact_stabilizations
    cs = model.contact_stabilizations["MyStab"]
    assert cs.name == "MyStab"
    assert cs.definition == "DAMPING COEFFICIENT"
    assert cs.damping_coefficient == 2.0e-4
    assert cs.tangent_fraction == 0.05
    assert cs.clearance == 0.02
    assert cs.ramp_down_factor == 0.1


def test_spring_and_dashpot_parsing():
    """Verify *SPRING and *DASHPOT are parsed and stored on part and model."""
    deck = """*HEADING
Test Spring and Dashpot
*PART, NAME=PART1
*NODE
1, 0.0, 0.0, 0.0
2, 1.0, 0.0, 0.0
*ELEMENT, TYPE=SPRING1, ELSET=ESPRING
1, 1
*ELEMENT, TYPE=DASHPOT1, ELSET=EDASH
2, 2
*SPRING, ELSET=ESPRING
1
1500.0
*DASHPOT, ELSET=EDASH
1
25.0
*END PART
"""
    reader, model = _parse_source(deck, dim=3)
    part = model.parts["PART1"]
    assert hasattr(part, "springs")
    assert "ESPRING" in part.springs
    assert part.springs["ESPRING"]["stiffness"] == 1500.0
    assert part.springs["ESPRING"]["dof"] == 1

    assert hasattr(part, "dashpots")
    assert "EDASH" in part.dashpots
    assert part.dashpots["EDASH"]["coefficient"] == 25.0
    assert part.dashpots["EDASH"]["dof"] == 1


def test_execute_inp_with_controls_e2e():
    """Verify an end-to-end solve executes cleanly with *CONTROLS present."""
    deck = """*HEADING
End-to-End Test with *CONTROLS
*PART, NAME=BAR
*NODE
1, 0.0, 0.0
2, 1.0, 0.0
3, 1.0, 1.0
4, 0.0, 1.0
*ELEMENT, TYPE=CPE4, ELSET=EALL
1, 1, 2, 3, 4
*SOLID SECTION, ELSET=EALL, MATERIAL=MAT1
1.0
*END PART
*ASSEMBLY, NAME=ASSEMBLY
*INSTANCE, NAME=BAR_1, PART=BAR
*END INSTANCE
*NSET, NSET=FIX, INSTANCE=BAR_1
1, 4
*NSET, NSET=PULL, INSTANCE=BAR_1
2, 3
*END ASSEMBLY
*MATERIAL, NAME=MAT1
*ELASTIC
1000.0, 0.3
*STEP, NAME=PULL_STEP
*STATIC
0.5, 1.0
*CONTROLS, PARAMETERS=TIME INCREMENTATION
16, 16, 9, 25, 2, 4, 1.25, 0.25
*CONTROLS, PARAMETERS=FIELD
0.005, 0.01, , , 0.02, 0.001
*BOUNDARY
FIX, 1, 2, 0.0
PULL, 1, 1, 0.01
*END STEP
"""
    result = execute_inp(deck, dim=2, verbose=False)
    assert result["converged"] is True
    assert result["u"] is not None
    assert result["iterations"] > 0
    # Verify step control was properly parsed and bound to step
    step = result["model"].steps["PULL_STEP"]
    assert step.control.field.R_n == 0.005
    assert step.control.field.C_n == 0.01
