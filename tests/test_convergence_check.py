"""*CONTROLS, CONVERGENCE CHECK= -- STRICT | MODERATE | RELAXED | CONCEPT DESIGN.

The four values differ in which tests apply and how they are joined (Abaqus Keywords
Guide), not in tolerance size. Default (unset) is STRICT and must not change behaviour.
"""
import pytest

from dispsolver.model.control import (
    CONVERGENCE_CHECKS, SolutionControl, normalize_convergence_check)


def test_default_is_strict_and_unchanged():
    c = SolutionControl()
    assert c.convergence_check == "STRICT"
    assert (c.field.R_n, c.field.C_n) == (0.005, 0.01)


@pytest.mark.parametrize("raw,want", [
    ("strict", "STRICT"), ("STANDARD", "STRICT"), ("moderate", "MODERATE"),
    ("Relaxed", "RELAXED"), ("concept design", "CONCEPT DESIGN"),
    ("CONCEPT_DESIGN", "CONCEPT DESIGN"),
])
def test_normalize(raw, want):
    assert normalize_convergence_check(raw) == want


def test_unknown_value_rejected():
    with pytest.raises(ValueError):
        normalize_convergence_check("LOOSE")


# (residual vs time-averaged flux, residual vs largest flux, correction) -> accepted?
TRUTH = {
    "STRICT":         lambda rs, rm, c: rs and c,
    "MODERATE":       lambda rs, rm, c: rm and c,
    "RELAXED":        lambda rs, rm, c: c,
    "CONCEPT DESIGN": lambda rs, rm, c: rs or c,
}


@pytest.mark.parametrize("mode", CONVERGENCE_CHECKS)
def test_field_accepts_truth_table(mode):
    c = SolutionControl()
    c.set_convergence_check(mode)
    for rs in (False, True):
        for rm in (False, True):
            for cc in (False, True):
                assert c.field_accepts(rs, rm, cc) == TRUTH[mode](rs, rm, cc), (mode, rs, rm, cc)


def test_setvalues_and_snippet_roundtrip():
    c = SolutionControl()
    c.setValues(convergenceCheck="MODERATE")
    assert c.convergence_check == "MODERATE"
    assert "CONVERGENCE CHECK=MODERATE" in c.to_inp_snippet()


def test_reader_sets_check_without_touching_tolerances():
    from tests.test_p1_reader_keywords import _parse_source
    deck = """*HEADING
t
*NODE
1, 0.0, 0.0, 0.0
*STEP, NAME=Step-1
*STATIC
*CONTROLS, PARAMETERS=FIELD, CONVERGENCE CHECK=CONCEPT DESIGN
*END STEP
"""
    _, model = _parse_source(deck, dim=3)
    ctrl = model.steps["Step-1"].control
    assert ctrl.convergence_check == "CONCEPT DESIGN"
    assert (ctrl.field.R_n, ctrl.field.C_n) == (0.005, 0.01)   # not a preset switch


# ---------------------------------------------------------------------------
# Contact-scoped CONVERGENCE CHECK (DynamicSolver.contact_convergence_check)
# ---------------------------------------------------------------------------
import numpy as np


class _FakeContact:
    def __init__(self, dofs):
        self.last_contact_dofs = np.asarray(dofs, dtype=np.int64)


def _stub(mode, contact_dofs, n=6):
    from dispsolver.solver.dynamic import DynamicSolver
    s = DynamicSolver.__new__(DynamicSolver)
    s.n_dofs = n
    s.penalty_constraints = [_FakeContact(contact_dofs)]
    s.set_contact_convergence_check(mode)
    return s


def _verdict(mode, R, du, f_int=None, contact_dofs=(2, 3)):
    s = _stub(mode, contact_dofs)
    R = np.asarray(R, float)
    du = np.asarray(du, float)
    f = np.full(6, 1.0) if f_int is None else np.asarray(f_int, float)
    return s._contact_scoped_converged(R, du, f, np.ones(6, bool), 0.005, 0.01, 1.0, 1.0)


GOOD = [0, 0, 0, 0, 0, 0]


def test_contact_check_default_is_strict_global():
    from dispsolver.solver.dynamic import DynamicSolver
    assert DynamicSolver.contact_convergence_check == "STRICT"


def test_contact_dof_residual_blowup_is_the_only_thing_relaxed_forgives():
    R = [0, 0, 0.5, 0, 0, 0]          # big residual on a contact DOF, tiny correction
    assert _verdict("STRICT", R, GOOD) is False
    assert _verdict("RELAXED", R, GOOD) is True
    assert _verdict("CONCEPT DESIGN", R, GOOD) is True
    # MODERATE: contact residual vs the largest contact force (f_c = 100 N here)
    assert _verdict("MODERATE", R, GOOD, f_int=[1, 1, 100, 1, 1, 1]) is True
    assert _verdict("MODERATE", R, GOOD, f_int=[1, 1, 10, 1, 1, 1]) is False


def test_large_contact_correction_blocks_relaxed_but_not_concept_design_with_small_residual():
    du = [0, 0, 0.5, 0, 0, 0]
    assert _verdict("RELAXED", GOOD, du) is False
    assert _verdict("CONCEPT DESIGN", GOOD, du) is True      # residual test alone suffices


def test_non_contact_dofs_stay_strict_in_every_mode():
    R = [0.5, 0, 0, 0, 0, 0]          # residual on a NON-contact DOF
    for mode in ("MODERATE", "RELAXED", "CONCEPT DESIGN"):
        assert _verdict(mode, R, GOOD) is False, mode


def test_no_contact_dofs_returns_none_so_global_test_stays():
    s = _stub("RELAXED", [])
    out = s._contact_scoped_converged(np.zeros(6), np.zeros(6), np.ones(6), np.ones(6, bool),
                                      0.005, 0.01, 1.0, 1.0)
    assert out is None


def test_contact_controls_keyword_carries_check():
    from tests.test_p1_reader_keywords import _parse_source
    deck = """*HEADING
t
*NODE
1, 0.0, 0.0, 0.0
*CONTACT CONTROLS, NAME=CC1, CONVERGENCE CHECK=MODERATE
*STEP, NAME=Step-1
*STATIC
*END STEP
"""
    _, model = _parse_source(deck, dim=3)
    assert model.contact_controls["CC1"].convergence_check == "MODERATE"


def test_rigid_segment_contact_exposes_contact_dofs():
    from dispsolver.constraint.rigid_segment_contact2d import RotatingRigidSegmentContact2D
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]])
    nid = {1: 0, 2: 1, 3: 2}
    c = RotatingRigidSegmentContact2D([1, 2, 3], nid, X, pivot=[0, 0], tangent=[1, 0],
                                      normal=[0, 1], s_range=(0.0, 3.0), penalty_stiffness=1e3)
    u = np.zeros(6)
    u[1] = -0.1                      # node 1 pushed below the face y=0
    f = np.zeros(6)
    c.apply_penalty(u, f)
    assert 1 in set(c.last_contact_dofs.tolist())
    assert 5 not in set(c.last_contact_dofs.tolist())       # node 3 is clear of the plate
